"""LLM client for translation — calls Ollama/vLLM OpenAI-compatible API.

Ported from translation-service's OpenAICompatibleTranslator (~150 lines
of core logic). Everything else was proxy/wrapper code.

Handles: prompt construction, HTTP POST, response parsing, retry with
exponential backoff.
"""
from __future__ import annotations

import asyncio
import time

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
    ) -> str:
        """Translate text using the LLM. Returns translated string.

        Args:
            glossary_terms: Optional dict of {source_term: target_term} for
                consistent terminology. Passed to the prompt builder so the
                LLM uses the preferred translations for domain-specific terms.
        """
        messages = self._build_messages(
            text, source_language, target_language,
            context or [], glossary_terms,
        )

        start = time.monotonic()
        response_text = await self._call_llm(messages)
        latency_ms = (time.monotonic() - start) * 1000

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

        # Try using TranslationPromptBuilder for glossary-aware prompts
        if glossary_terms:
            try:
                from services.translation_prompt_builder import (
                    TranslationPromptBuilder,
                    PromptContext,
                )
                builder = TranslationPromptBuilder()
                previous_sentences = [ctx.text for ctx in context] if context else None
                result = builder.build(PromptContext(
                    current_sentence=text,
                    target_language=target_language,
                    source_language=source_language,
                    previous_sentences=previous_sentences,
                    glossary_terms=glossary_terms,
                ))
                return [
                    {"role": "system", "content": result.system_prompt if hasattr(result, 'system_prompt') else (
                        f"You are a professional translator from {source_language} "
                        f"to {target_language}. Output ONLY the translation."
                    )},
                    {"role": "user", "content": result.prompt},
                ]
            except Exception:
                logger.debug(
                    "prompt_builder_unavailable",
                    msg="TranslationPromptBuilder not available, using inline prompt with glossary",
                )

        system_prompt = (
            f"You are a professional translator from {source_language} to {target_language}. "
            "Translate the given text naturally and accurately. "
            "Output ONLY the translation, nothing else. "
            "Preserve the original meaning, tone, and style."
        )

        user_parts = []

        # Glossary section (inline fallback when TranslationPromptBuilder unavailable)
        if glossary_terms:
            user_parts.append("Glossary (use these exact translations for the listed terms):")
            for src_term, tgt_term in glossary_terms.items():
                user_parts.append(f"  {src_term} -> {tgt_term}")
            user_parts.append("")

        if context:
            user_parts.append("Context (previous sentences for reference):")
            for ctx in context:
                user_parts.append(f"  Original: {ctx.text}")
                user_parts.append(f"  Translation: {ctx.translation}")
            user_parts.append("")

        user_parts.append(f"Translate the following from {source_language} to {target_language}:")
        user_parts.append(text)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    async def _call_llm(self, messages: list[dict], max_retries: int = 1) -> str:
        """Call the LLM API with retry logic."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": messages,
                        "temperature": self.config.temperature,
                    },
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

    async def close(self) -> None:
        await self._client.aclose()
