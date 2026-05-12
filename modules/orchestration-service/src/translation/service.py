"""TranslationService — combines LLM client, rolling context, and backpressure.

This is the high-level interface that the meeting pipeline calls.
It manages:
- LLM client (merged `livetranslate_common.llm.client.LLMClient`)
- Rolling context window for quality
- Bounded queue with drop-newest backpressure
- Per-call sampling overrides (LLMParameterOverrides)
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from livetranslate_common.llm.client import LLMClient
from livetranslate_common.llm.qwen import extract_translation_text
from livetranslate_common.logging import get_logger
from livetranslate_common.models import (
    TranslationContext,
    TranslationRequest,
    TranslationResponse,
)
from livetranslate_common.models.llm import LLMConnection, LLMParameterOverrides

from translation.config import TranslationConfig
from translation.context_store import DirectionalContextStore
from translation.prompt import build_messages

logger = get_logger()


class TranslationService:
    """Orchestrates LLM translation calls with rolling context + backpressure.

    Args:
        connection: Endpoint identity + default sampling parameters.
            Per-call overrides via `LLMParameterOverrides` layer on top.
        behavioral: Pipeline behavior knobs (context window, draft tokens,
            queue depth). No endpoint/sampling info here.
        context_store: Optional pre-built directional context store. If
            omitted, a fresh one is created from `behavioral` settings.
    """

    def __init__(
        self,
        connection: LLMConnection,
        behavioral: TranslationConfig,
        context_store: DirectionalContextStore | None = None,
    ) -> None:
        self._base_connection = connection
        self.behavioral = behavioral
        # Back-compat alias for callers still accessing `.config`
        self.config = behavioral
        self._client = LLMClient(connection)
        self.context_store = context_store or DirectionalContextStore(
            max_entries=behavioral.context_window_size,
            max_tokens=behavioral.max_context_tokens,
            cross_direction_max_tokens=behavioral.cross_direction_max_tokens,
        )
        self._queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=behavioral.max_queue_depth)
        self._processing = False
        self._process_task: asyncio.Task | None = None
        self._concurrency = asyncio.Semaphore(min(behavioral.max_queue_depth, 3))

    @property
    def base_connection(self) -> LLMConnection:
        """Read-only access to the immutable base connection."""
        return self._base_connection

    # -- selection helpers ---------------------------------------------------

    def _select_store(
        self,
        context_store: DirectionalContextStore | None = None,
    ) -> DirectionalContextStore:
        return context_store or self.context_store

    # -- core translation ---------------------------------------------------

    async def translate_draft(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: list[TranslationContext] | None = None,
        context_store: DirectionalContextStore | None = None,
        overrides: LLMParameterOverrides | None = None,
    ) -> str:
        """Draft translation: fast, fail-fast, no context write-back.

        Uses behavioral.draft_max_tokens for shorter output. Returns the
        cleaned-up translated string.
        """
        if context is None:
            context = self._select_store(context_store).get(source_lang, target_lang)[-3:]
        messages = build_messages(text, source_lang, target_lang, context=context)
        draft_overrides = LLMParameterOverrides(
            max_tokens=self.behavioral.draft_max_tokens,
            timeout_s=float(self.behavioral.draft_timeout_s),
        )
        # Layer caller overrides on top of draft defaults via re-merge dance:
        # caller overrides win if both specify the same field.
        effective = draft_overrides
        if overrides is not None:
            patch = overrides.model_dump(exclude_none=True)
            effective = LLMParameterOverrides(**{
                **draft_overrides.model_dump(exclude_none=True),
                **patch,
            })
        return await self._client.complete(messages=messages, overrides=effective)

    async def translate(
        self,
        request: TranslationRequest,
        skip_context: bool = False,
        context_store: DirectionalContextStore | None = None,
        overrides: LLMParameterOverrides | None = None,
    ) -> TranslationResponse:
        """Translate text synchronously (blocking until complete).

        Accepts a TranslationRequest and optional per-call sampling overrides.
        On success, the source/translation pair is added to the rolling
        context store keyed by direction (unless `skip_context=True`).
        """
        store = self._select_store(context_store)
        context = request.context if request.context else store.get(
            request.source_language, request.target_language,
        )
        messages = build_messages(
            request.text,
            request.source_language,
            request.target_language,
            context=context,
            glossary_terms=request.glossary_terms,
        )
        start = time.monotonic()
        translated = await self._client.complete(messages=messages, overrides=overrides)
        latency_ms = (time.monotonic() - start) * 1000

        if not skip_context:
            store.add(
                request.source_language,
                request.target_language,
                request.text,
                translated,
            )

        # Effective model: override.model wins, else base connection.model
        model_used = (
            overrides.model if (overrides and overrides.model) else self._base_connection.model
        )
        return TranslationResponse(
            translated_text=translated,
            source_language=request.source_language,
            target_language=request.target_language,
            model_used=model_used,
            latency_ms=round(latency_ms, 1),
        )

    async def stream_translate(
        self,
        request: TranslationRequest,
        on_delta: Callable[[str], Awaitable[bool | None]],
        *,
        context_store: DirectionalContextStore | None = None,
        skip_context: bool = False,
        cross_context: list[TranslationContext] | None = None,
        overrides: LLMParameterOverrides | None = None,
    ) -> TranslationResponse | None:
        """Stream a translation through a callback with fallback."""
        store = self._select_store(context_store)
        context = request.context if request.context else store.get(
            request.source_language,
            request.target_language,
        )
        messages = build_messages(
            request.text,
            request.source_language,
            request.target_language,
            context=context,
            glossary_terms=request.glossary_terms,
            cross_context=cross_context,
        )
        start = time.monotonic()
        full_text = ""
        try:
            async for delta in self._client.stream(messages=messages, overrides=overrides):
                full_text += delta
                keep_going = await on_delta(delta)
                if keep_going is False:
                    logger.info(
                        "translation_stream_aborted",
                        source_lang=request.source_language,
                        target_lang=request.target_language,
                    )
                    return None
            translated = extract_translation_text(full_text)
        except Exception as stream_exc:
            logger.warning(
                "translation_stream_fallback",
                source_lang=request.source_language,
                target_lang=request.target_language,
                error=str(stream_exc),
            )
            fallback_response = await self.translate(
                request,
                skip_context=skip_context,
                context_store=store,
                overrides=overrides,
            )
            return TranslationResponse(
                translated_text=fallback_response.translated_text,
                source_language=fallback_response.source_language,
                target_language=fallback_response.target_language,
                model_used=fallback_response.model_used,
                latency_ms=round((time.monotonic() - start) * 1000, 1),
            )

        latency_ms = (time.monotonic() - start) * 1000
        if not skip_context:
            store.add(
                request.source_language,
                request.target_language,
                request.text,
                translated,
            )
        model_used = (
            overrides.model if (overrides and overrides.model) else self._base_connection.model
        )
        return TranslationResponse(
            translated_text=translated,
            source_language=request.source_language,
            target_language=request.target_language,
            model_used=model_used,
            latency_ms=round(latency_ms, 1),
        )

    # -- queue / backpressure ----------------------------------------------

    async def enqueue_translation(self, request: TranslationRequest) -> TranslationResponse:
        """Queue a translation request with drop-newest backpressure."""
        future: asyncio.Future[TranslationResponse] = asyncio.get_running_loop().create_future()
        if self._queue.full():
            logger.warning(
                "translation_rejected",
                text_len=len(request.text),
                reason="queue_full",
            )
            raise RuntimeError("Translation request rejected (backpressure — queue full)")
        await self._queue.put((request, future))
        if not self._processing:
            self._processing = True
            self._process_task = asyncio.create_task(self._process_queue())
        return await future

    async def _process_queue(self) -> None:
        try:
            while True:
                tasks = []
                while not self._queue.empty():
                    request, future = await self._queue.get()
                    task = asyncio.create_task(self._process_one(request, future))
                    tasks.append(task)
                if not tasks:
                    break
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._processing = False

    async def _process_one(self, request: TranslationRequest, future: asyncio.Future) -> None:
        async with self._concurrency:
            try:
                result = await self.translate(request)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

    # -- context helpers ----------------------------------------------------

    def get_context(
        self,
        source_lang: str = "",
        target_lang: str = "",
    ) -> list[TranslationContext]:
        return self.context_store.get(source_lang, target_lang)

    def clear_context(
        self,
        source_lang: str = "",
        target_lang: str = "",
    ) -> None:
        if not source_lang and not target_lang:
            self.context_store.clear_all()
        else:
            self.context_store.clear_direction(source_lang, target_lang)

    async def close(self) -> None:
        if self._process_task and not self._process_task.done():
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        await self._client.aclose()
