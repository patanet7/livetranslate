"""LLM client for translation — calls Ollama/vLLM OpenAI-compatible API.

Ported from translation-service's OpenAICompatibleTranslator (~150 lines
of core logic). Everything else was proxy/wrapper code.

Handles: prompt construction, HTTP POST, response parsing, retry with
exponential backoff. Supports streaming for real-time token delivery.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator

import httpx
from livetranslate_common.logging import get_logger
from livetranslate_common.models import TranslationContext

from translation.config import TranslationConfig

logger = get_logger()


def extract_translation_text(response: str) -> str:
    """Clean up LLM output so callers only keep the translated text."""
    text = response.strip()
    # Strip Qwen3 <think>...</think> reasoning blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip unclosed <think> blocks (max_tokens cutoff)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL).strip()
    # Remove common LLM prefix patterns (case-insensitive)
    text = re.sub(
        r'^(?:translation|translated text|output|result|the translation is|翻译|译文)\s*[:：]\s*',
        '',
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    # Remove surrounding quotes (straight or curly, single or double)
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or \
           (text[0] == '\u201c' and text[-1] == '\u201d') or \
           (text[0] == "'" and text[-1] == "'") or \
           (text[0] == '\u2018' and text[-1] == '\u2019'):
            text = text[1:-1]
    return text.strip()


class LLMClient:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.llm_base_url,
            timeout=httpx.Timeout(config.timeout_s),
            max_redirects=0,
        )
        # Detect Ollama + Qwen3: use native API to avoid reasoning field bug
        self._use_ollama_native = (
            "11434" in config.llm_base_url and
            config.model.lower().startswith("qwen3")
        )
        if self._use_ollama_native:
            # Native Ollama API is at the root, not /v1
            base = config.llm_base_url.replace("/v1", "")
            self._ollama_client = httpx.AsyncClient(
                base_url=base,
                timeout=httpx.Timeout(config.timeout_s),
            )
            logger.info(
                "llm_client_using_ollama_native",
                model=config.model,
                reason="Qwen3 on Ollama loses response in OpenAI-compat layer",
            )

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[TranslationContext] | None = None,
        glossary_terms: dict[str, str] | None = None,
        max_tokens: int | None = None,
        max_retries: int = 1,
    ) -> str:
        """Translate text using the LLM. Returns translated string.

        Args:
            glossary_terms: Optional dict of {source_term: target_term} for
                consistent terminology. Passed to the prompt builder so the
                LLM uses the preferred translations for domain-specific terms.
            max_tokens: Override config.max_tokens for this call.
            max_retries: Number of retries on failure (0 = fail immediately).
        """
        messages = self._build_messages(
            text, source_language, target_language,
            context or [], glossary_terms,
        )

        start = time.monotonic()
        response_text = await self._call_llm(messages, max_retries=max_retries, max_tokens=max_tokens)
        latency_ms = (time.monotonic() - start) * 1000

        logger.debug(
            "llm_raw_response",
            raw_len=len(response_text),
            raw_preview=response_text[:200],
        )
        translation = self._extract_translation(response_text)

        logger.info(
            "translation_complete",
            source_lang=source_language,
            target_lang=target_language,
            model=self.config.model,
            latency_ms=round(latency_ms, 1),
            input_len=len(text),
            output_len=len(translation),
        )

        return translation

    def _build_messages(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[TranslationContext],
        glossary_terms: dict[str, str] | None = None,
        cross_context: list[TranslationContext] | None = None,
    ) -> list[dict]:
        """Build LLM messages for translation.

        System prompt variant is selected based on whether context is present:
        - With context: includes "Never repeat context" guard to prevent
          the LLM from regurgitating prior examples.
        - Without context: shorter prompt for lower latency.

        If glossary_terms is provided, the terms are injected inline before
        the text to translate. See:
        modules/orchestration-service/src/services/translation_prompt_builder.py
        """
        # Validate glossary terms to prevent prompt injection via oversized entries
        if glossary_terms:
            glossary_terms = {
                k[:100]: v[:100]
                for k, v in list(glossary_terms.items())[:50]
            }

        # Language display names for clear prompts (Qwen is Chinese-dominant
        # and code-switches without explicit constraints)
        _LANG_NAMES = {
            "zh": "Chinese", "en": "English", "es": "Spanish",
            "ja": "Japanese", "ko": "Korean", "fr": "French",
            "de": "German", "pt": "Portuguese", "it": "Italian",
            "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
        }
        src_name = _LANG_NAMES.get(source_language, source_language)
        tgt_name = _LANG_NAMES.get(target_language, target_language)

        # Direction-specific overrides (e.g., "Use simplified characters" for zh)
        _EXTRA_INSTRUCTIONS = {
            ("en", "zh"): " Use simplified characters.",
        }
        extra = _EXTRA_INSTRUCTIONS.get((source_language, target_language), "")

        # System prompt: two variants
        # - Final path with context: needs "Never repeat context" guard
        # - Draft path / no context: shorter prompt, lower latency
        has_context = bool(context)
        if has_context:
            system_prompt = (
                f"Translate {src_name} speech to {tgt_name}.{extra} "
                f"Output ONLY the {tgt_name} translation. "
                f"Never repeat context."
            )
        else:
            system_prompt = (
                f"Translate {src_name} speech to {tgt_name}.{extra} "
                f"Output ONLY the {tgt_name} translation."
            )

        user_parts = []

        # Glossary: compact format, sanitize newlines to prevent prompt injection
        if glossary_terms:
            sanitized = {
                k.replace("\n", " "): v.replace("\n", " ")
                for k, v in glossary_terms.items()
            }
            terms = ", ".join(f"{k}={v}" for k, v in sanitized.items())
            user_parts.append(f"Terms: {terms}")
            user_parts.append("")

        # Context: bilingual pairs for consistency
        if context:
            user_parts.append("[Prior:]")
            for ctx in context:
                src = ctx.text.replace("\n", " ")
                tgt = ctx.translation.replace("\n", " ")
                user_parts.append(f"[{src_name}] {src}")
                user_parts.append(f"[{tgt_name}] {tgt}")
            user_parts.append("")

        # Cross-direction context: recent entries from the opposite direction
        # (interpreter mode — helps with referent tracking across speakers).
        # Labels are swapped: cross-context entries come from the opposite
        # direction (tgt→src), so ctx.text is in target_language and
        # ctx.translation is in source_language.
        if cross_context:
            user_parts.append("[Recent context (other speaker):]")
            for ctx in cross_context:
                src = ctx.text.replace("\n", " ")
                tgt = ctx.translation.replace("\n", " ")
                user_parts.append(f"[{tgt_name}] {src}")
                user_parts.append(f"[{src_name}] {tgt}")
            user_parts.append("")

        # Text to translate: compact format for drafts, labeled for finals
        if has_context:
            user_parts.append("[New:]")
            user_parts.append(text)
        else:
            user_parts.append(f"Translate: {text}")

        # NOTE: /nothink suffix removed — it breaks Ollama's Qwen3.5 where the
        # model returns empty content with reasoning in a separate field.
        # vLLM-MLX handles this via chat_template_kwargs instead.
        user_content = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _request_body(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> dict:
        """Build the OpenAI-compatible request body.

        Centralizes model, sampling, and Qwen3 /nothink parameters so
        _call_llm() and translate_stream() stay in sync.
        """
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "stream": stream,
            # Qwen3 /nothink mode — per Alibaba docs (qwenlm/qwen3):
            # https://github.com/qwenlm/qwen3/blob/main/docs/source/deployment/vllm.md
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.05,
            # Disable thinking for vLLM/SGLang (Jinja chat template param)
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def _call_llm(
        self,
        messages: list[dict],
        max_retries: int = 1,
        max_tokens: int | None = None,
    ) -> str:
        """Call the LLM API with retry logic."""
        # Use native Ollama API for Qwen3 models (avoids reasoning field bug)
        if self._use_ollama_native:
            return await self._call_ollama_native(messages, max_retries, max_tokens)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=self._request_body(messages, max_tokens=max_tokens),
                )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                content = message.get("content", "")

                # Qwen3/3.5 on Ollama puts output in a separate "reasoning" field
                # when content is empty. Extract the translation from reasoning.
                if not content and "reasoning" in message:
                    content = self._extract_from_reasoning(message["reasoning"])

                return content
            except (httpx.HTTPError, KeyError, IndexError) as e:
                last_error = e
                if attempt < max_retries:
                    wait = 0.3 * (2 ** attempt)
                    logger.warning(
                        "llm_retry",
                        attempt=attempt + 1,
                        error=str(e),
                        wait_s=wait,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_error}")

    async def _call_ollama_native(
        self,
        messages: list[dict],
        max_retries: int = 1,
        max_tokens: int | None = None,
    ) -> str:
        """Call Ollama's native /api/generate endpoint.

        Ollama's OpenAI-compatible layer has a bug where Qwen3's response
        ends up in a "reasoning" field instead of "content". The native
        API correctly returns the translation in the "response" field.
        """
        # Build prompt from messages (system + user format)
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        body = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
                "top_p": 0.8,
                "top_k": 20,
                "repeat_penalty": 1.05,
            },
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._ollama_client.post("/api/generate", json=body)
                response.raise_for_status()
                data = response.json()

                # Native API returns translation in "response" field
                content = data.get("response", "")

                # Qwen3/3.5 on Ollama uses thinking mode — actual answer is in
                # "thinking" field, "response" is empty. Extract from thinking.
                if not content and "thinking" in data:
                    thinking = data["thinking"]
                    content = self._extract_from_reasoning(thinking)
                    logger.debug(
                        "ollama_native_extracted_from_thinking",
                        thinking_len=len(thinking),
                        extracted_len=len(content),
                        extracted_preview=content[:100] if content else "",
                    )

                if content:
                    logger.debug(
                        "ollama_native_response",
                        response_len=len(content),
                        response_preview=content[:100],
                    )
                return content

            except (httpx.HTTPError, KeyError) as e:
                last_error = e
                if attempt < max_retries:
                    wait = 0.3 * (2 ** attempt)
                    logger.warning(
                        "ollama_native_retry",
                        attempt=attempt + 1,
                        error=str(e),
                        wait_s=wait,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(f"Ollama native call failed after {max_retries + 1} attempts: {last_error}")

    def _extract_from_reasoning(self, reasoning: str) -> str:
        """Extract the actual translation from Qwen3's reasoning field.

        Qwen3/3.5 on Ollama returns reasoning in a separate field. The actual
        answer is typically after phrases like "Final Decision:", in quotes,
        or as "Option N: <translation>".
        """
        if not reasoning:
            return ""

        import re

        # Meta-reasoning phrases that should be filtered out
        meta_phrases = [
            "let me", "i should", "i need", "thinking", "step",
            "first", "second", "third", "finally", "therefore",
            "so ", "thus", "hence", "because", "since", "as ",
            "wait", "actually", "hmm", "ok", "okay", "note",
            "polished", "alternative", "literal", "meaning",
            "analysis", "breakdown", "context", "constraint",
            "translate the", "translation task", "going with",
            "check", "double-check", "verify", "review",
        ]

        def is_meta_text(text: str) -> bool:
            """Check if text looks like reasoning rather than translation."""
            lower = text.lower()
            return any(p in lower for p in meta_phrases)

        # Priority 1: Look for "Final Output Generation:" or "Draft Output:" sections
        # Qwen3.5 often puts the answer here after "6.  **Final Output Generation:**"
        for section_marker in [
            r'\*{0,2}Final Output(?:\s+Generation)?[:\*]{0,2}\s*[:\s]*(.+?)(?:\n|$)',
            r'\*{0,2}Draft Output[:\*]{0,2}\s*[:\s]*(.+?)(?:\n|$)',
        ]:
            section_match = re.search(section_marker, reasoning, re.IGNORECASE)
            if section_match:
                result = section_match.group(1).strip()
                # Clean up markdown and quotes
                result = re.sub(r'^[\"\'\*\s\-]+|[\"\'\*\s\-]+$', '', result)
                # Remove parenthetical notes at the end
                result = re.sub(r'\s*\([^)]+\)\s*$', '', result)
                # Remove "or similar..." trailing text
                result = re.sub(r'\s+(?:or similar|or just|Let\'s go).*$', '', result, flags=re.IGNORECASE)
                if result and len(result) > 2 and not is_meta_text(result):
                    return result

        # Priority 2: Look for quoted translations after "Most standard" or similar
        standard_match = re.search(
            r'(?:Most\s+)?(?:standard|common|natural)[^:]*:\s*["\']?([A-Z][^"\'.\n]+)["\']?',
            reasoning,
            re.IGNORECASE
        )
        if standard_match:
            result = standard_match.group(1).strip()
            result = re.sub(r'^[\"\'\s]+|[\"\'\s]+$', '', result)
            if result and len(result) > 2 and not is_meta_text(result):
                return result

        # Priority 2: Look for "Final Decision:" or similar final markers
        final_markers = [
            r"Final Decision:\s*[\"']?([^\"'\n]+)[\"']?",
            r"Final Response:\s*[\"']?([^\"'\n]+)[\"']?",
            r"Final Output:\s*[\"']?([^\"'\n]+)[\"']?",
            r"Final Answer:\s*[\"']?([^\"'\n]+)[\"']?",
            r"(?:Let's go with|Going with)[:\s]*[\"']?([^\"'\n]+)[\"']?",
        ]
        for pattern in final_markers:
            match = re.search(pattern, reasoning, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                # Remove trailing punctuation from reasoning
                result = re.sub(r'[\.\,]$', '', result).strip()
                if result and len(result) > 2 and not is_meta_text(result):
                    return result

        # Priority 3: Look for "Full sentence:" or similar markers with translation
        full_match = re.search(
            r'Full\s+(?:sentence|translation)[:\s]*["\']?([A-Z][^"]+?)["\']?\s*(?:\.|or\s)',
            reasoning,
            re.IGNORECASE
        )
        if full_match:
            result = full_match.group(1).strip()
            result = re.sub(r'^[\"\'\s]+|[\"\'\s]+$', '', result)
            if result and len(result) > 3 and not is_meta_text(result):
                return result

        # Priority 4: Look for double-quoted translations (allowing apostrophes inside)
        # "Welcome to today's meeting." should be matched
        dquoted = re.findall(r'"([A-Z][^"]{3,})"', reasoning)
        if dquoted:
            valid_quotes = [q for q in dquoted if not is_meta_text(q)]
            if valid_quotes:
                # Prefer quotes that look like complete sentences
                for q in valid_quotes:
                    if q.endswith('.') or ' ' in q:
                        return q.rstrip('.')
                # Fallback to longest
                longest = max(valid_quotes, key=len)
                if len(longest) > 5:
                    return longest.rstrip('.')

        # Priority 4: Look for "Option N: translation" patterns
        option_match = re.search(
            r'Option \d+:\s*[\""]?([A-Z][^\""\n\.]+)[\""]?',
            reasoning
        )
        if option_match:
            result = option_match.group(1).strip()
            if not is_meta_text(result):
                return result

        # Priority 5: Fallback - last line that starts with capital and looks like English
        lines = [l.strip() for l in reasoning.split("\n") if l.strip()]
        for line in reversed(lines):
            # Skip reasoning/meta text
            if is_meta_text(line):
                continue
            # Skip lines with markdown markers
            if line.startswith(("*", "[", "-", "(", "#")):
                continue
            # Skip lines ending with : or ?
            if line.endswith((":", "?")):
                continue
            # Return if it starts with capital letter and looks like a sentence
            if len(line) > 5 and line[0].isupper() and " " in line:
                return line

        return ""

    def _extract_translation(self, response: str) -> str:
        """Clean up LLM response to extract just the translation."""
        return extract_translation_text(response)

    async def translate_stream(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[TranslationContext] | None = None,
        glossary_terms: dict[str, str] | None = None,
        max_tokens: int | None = None,
        cross_context: list[TranslationContext] | None = None,
    ) -> AsyncIterator[str]:
        """Stream translation tokens from the LLM.

        Same prompt construction as translate(), but uses SSE streaming.
        Yields raw delta strings as they arrive. No retry — caller should
        fall back to non-streaming translate() on failure.
        """
        # Use native Ollama streaming for Qwen3 models
        if self._use_ollama_native:
            async for delta in self._stream_ollama_native(
                text, source_language, target_language,
                context, glossary_terms, max_tokens, cross_context,
            ):
                yield delta
            return

        messages = self._build_messages(
            text, source_language, target_language,
            context or [], glossary_terms,
            cross_context=cross_context,
        )

        async with self._client.stream(
            "POST",
            "/chat/completions",
            json=self._request_body(messages, stream=True, max_tokens=max_tokens),
        ) as response:
            response.raise_for_status()
            line_count = 0
            yield_count = 0
            async for line in response.aiter_lines():
                line_count += 1
                if not line.startswith("data: "):
                    continue
                payload = line[6:]  # strip "data: " prefix
                if payload.strip() == "[DONE]":
                    logger.debug(
                        "stream_done",
                        lines=line_count,
                        yields=yield_count,
                    )
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield_count += 1
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            else:
                # aiter_lines exhausted without [DONE] — stream ended unexpectedly
                logger.warning(
                    "stream_ended_without_done",
                    lines=line_count,
                    yields=yield_count,
                )

    async def _stream_ollama_native(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[TranslationContext] | None = None,
        glossary_terms: dict[str, str] | None = None,
        max_tokens: int | None = None,
        cross_context: list[TranslationContext] | None = None,
    ) -> AsyncIterator[str]:
        """Stream translation using Ollama's native /api/generate endpoint.

        Ollama native streaming returns newline-delimited JSON with "response"
        field containing each token delta.
        """
        messages = self._build_messages(
            text, source_language, target_language,
            context or [], glossary_terms,
            cross_context=cross_context,
        )

        # Build prompt from messages (ChatML format for Qwen3)
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        body = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
                "top_p": 0.8,
                "top_k": 20,
                "repeat_penalty": 1.05,
            },
        }

        async with self._ollama_client.stream("POST", "/api/generate", json=body) as response:
            response.raise_for_status()
            yield_count = 0
            # Qwen3/3.5 on Ollama puts content in "thinking" field, not "response".
            # Accumulate thinking chunks and extract translation at the end.
            accumulated_thinking = []
            has_thinking_mode = False

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)

                    # Check if model is using thinking mode
                    if "thinking" in chunk:
                        has_thinking_mode = True
                        thinking_delta = chunk.get("thinking", "")
                        if thinking_delta:
                            accumulated_thinking.append(thinking_delta)

                    # Try normal response field first
                    delta = chunk.get("response", "")
                    if delta:
                        yield_count += 1
                        yield delta

                    # Check for completion
                    if chunk.get("done", False):
                        # If we accumulated thinking but yielded nothing, extract translation
                        if has_thinking_mode and yield_count == 0 and accumulated_thinking:
                            full_thinking = "".join(accumulated_thinking)
                            extracted = self._extract_from_reasoning(full_thinking)
                            if extracted:
                                logger.debug(
                                    "ollama_native_stream_extracted_from_thinking",
                                    thinking_len=len(full_thinking),
                                    extracted_len=len(extracted),
                                )
                                yield extracted
                                yield_count = 1

                        logger.debug(
                            "ollama_native_stream_done",
                            yields=yield_count,
                            had_thinking_mode=has_thinking_mode,
                        )
                        break
                except json.JSONDecodeError:
                    continue

    async def close(self) -> None:
        await self._client.aclose()
        if self._use_ollama_native:
            await self._ollama_client.aclose()
