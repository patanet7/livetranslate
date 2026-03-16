"""LLM client for translation — calls Ollama/vLLM OpenAI-compatible API.

Ported from translation-service's OpenAICompatibleTranslator (~150 lines
of core logic). Everything else was proxy/wrapper code.

Handles: prompt construction, HTTP POST, response parsing, retry with
exponential backoff. Supports streaming for real-time token delivery.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx
from livetranslate_common.logging import get_logger
from livetranslate_common.models import TranslationContext

from translation.config import TranslationConfig

logger = get_logger()


class LLMClient:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.llm_base_url,
            timeout=httpx.Timeout(config.timeout_s),
            max_redirects=0,
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
    ) -> list[dict]:
        """Build LLM messages for translation.

        If glossary_terms is provided AND the existing TranslationPromptBuilder
        is available, delegates to it for richer prompt construction (glossary
        injection, speaker attribution, etc.). Otherwise uses the inline prompt.

        See: modules/orchestration-service/src/services/translation_prompt_builder.py
        """
        # Validate glossary terms to prevent prompt injection via oversized entries
        if glossary_terms:
            glossary_terms = {
                k[:100]: v[:100]
                for k, v in list(glossary_terms.items())[:50]
            }

        # Direction-specific system prompts for better register + output
        _SYSTEM_PROMPTS = {
            ("zh", "en"): "Translate spoken Chinese to natural English. Output only the translation.",
            ("en", "zh"): "Translate spoken English to natural Mandarin Chinese. Use simplified characters. Output only the translation.",
            ("ja", "en"): "Translate spoken Japanese to natural English. Output only the translation.",
            ("en", "ja"): "Translate spoken English to natural Japanese. Output only the translation.",
            ("es", "en"): "Translate spoken Spanish to natural English. Output only the translation.",
            ("en", "es"): "Translate spoken English to natural Spanish. Output only the translation.",
        }
        system_prompt = _SYSTEM_PROMPTS.get(
            (source_language, target_language),
            f"Translate spoken {source_language} to natural {target_language}. Output only the translation.",
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

        # Context: previous translations for consistency (do NOT re-translate)
        if context:
            user_parts.append("[Context — previous conversation, do NOT translate:]")
            for ctx in context:
                user_parts.append(f"- {ctx.translation}")
            user_parts.append("")

        user_parts.append(f"[Translate this:]")
        user_parts.append(text)

        # Append /nothink suffix — Qwen3 chat template recognizes this to
        # disable the <think> reasoning phase. Belt-and-suspenders with
        # chat_template_kwargs for servers that don't support template params.
        user_content = "\n".join(user_parts) + " /nothink"

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
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=self._request_body(messages, max_tokens=max_tokens),
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
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

    def _extract_translation(self, response: str) -> str:
        """Clean up LLM response to extract just the translation."""
        import re
        text = response.strip()
        # Strip Qwen3 <think>...</think> reasoning blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Strip unclosed <think> blocks (max_tokens cutoff)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL).strip()
        # Remove common LLM prefix patterns (case-insensitive)
        text = re.sub(
            r'^(?:translation|translated text|output|result|the translation is|翻译|译文)\s*[:：]\s*',
            '', text, count=1, flags=re.IGNORECASE,
        )
        # Remove surrounding quotes (straight or curly, single or double)
        if len(text) >= 2:
            if (text[0] == '"' and text[-1] == '"') or \
               (text[0] == '\u201c' and text[-1] == '\u201d') or \
               (text[0] == "'" and text[-1] == "'") or \
               (text[0] == '\u2018' and text[-1] == '\u2019'):
                text = text[1:-1]
        return text.strip()

    async def translate_stream(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[TranslationContext] | None = None,
        glossary_terms: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream translation tokens from the LLM.

        Same prompt construction as translate(), but uses SSE streaming.
        Yields raw delta strings as they arrive. No retry — caller should
        fall back to non-streaming translate() on failure.
        """
        messages = self._build_messages(
            text, source_language, target_language,
            context or [], glossary_terms,
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

    async def close(self) -> None:
        await self._client.aclose()
