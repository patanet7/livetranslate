"""TranslationService — combines LLM client, rolling context, and backpressure.

This is the high-level interface that the meeting pipeline calls.
It manages:
- LLM client for actual translation
- Rolling context window for quality
- Bounded queue with drop-oldest backpressure
"""
from __future__ import annotations

import asyncio
import time

from livetranslate_common.logging import get_logger
from livetranslate_common.models import TranslationContext, TranslationRequest, TranslationResponse

from translation.config import TranslationConfig
from translation.context import RollingContextWindow
from translation.llm_client import LLMClient

logger = get_logger()


class TranslationService:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self._client = LLMClient(config)
        self._context = RollingContextWindow(
            max_entries=config.context_window_size,
            max_tokens=config.max_context_tokens,
        )
        self._queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=config.max_queue_depth)
        self._processing = False
        self._process_task: asyncio.Task | None = None

    async def translate(
        self,
        request: TranslationRequest,
    ) -> TranslationResponse:
        """Translate text synchronously (blocking until complete).

        Accepts a TranslationRequest (from Plan 0 shared contracts) instead
        of bare parameters. Context from the request is merged with the
        rolling context window.

        Used for direct translation. For queued translation with
        backpressure, use enqueue_translation().
        """
        # Merge: use request context if provided, otherwise use rolling window
        context = request.context if request.context else self._context.get_context()
        start = time.monotonic()

        translated = await self._client.translate(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            context=context,
            glossary_terms=request.glossary_terms or None,
        )

        latency_ms = (time.monotonic() - start) * 1000

        # Only add to context on success
        self._context.add(request.text, translated)

        return TranslationResponse(
            translated_text=translated,
            source_language=request.source_language,
            target_language=request.target_language,
            model_used=self.config.model,
            latency_ms=round(latency_ms, 1),
        )

    async def enqueue_translation(
        self,
        text: str,
        source_language: str,
        target_language: str,
        glossary_terms: dict[str, str] | None = None,
    ) -> TranslationResponse:
        """Queue a translation request with backpressure.

        If queue is full, drops the oldest pending request.
        Each queued item stores its own language pair so _process_queue
        does not need to assume a single pair for all items.
        """
        future: asyncio.Future[TranslationResponse] = asyncio.get_running_loop().create_future()

        if self._queue.full():
            # Drop oldest
            try:
                old_text, _old_src, _old_tgt, _old_glossary, old_future = self._queue.get_nowait()
                old_future.set_exception(
                    RuntimeError("Translation request dropped (backpressure)")
                )
                logger.warning("translation_dropped", text_len=len(old_text))
            except asyncio.QueueEmpty:
                pass

        await self._queue.put((text, source_language, target_language, glossary_terms, future))

        if not self._processing:
            self._processing = True
            self._process_task = asyncio.create_task(self._process_queue())

        return await future

    async def _process_queue(self) -> None:
        """Process queued translation requests. Each item carries its own language pair."""
        try:
            while not self._queue.empty():
                text, source_language, target_language, glossary_terms, future = await self._queue.get()
                try:
                    request = TranslationRequest(
                        text=text,
                        source_language=source_language,
                        target_language=target_language,
                        context=self._context.get_context(),
                        glossary_terms=glossary_terms or {},
                    )
                    result = await self.translate(request)
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
        finally:
            self._processing = False

    def get_context(self) -> list[TranslationContext]:
        return self._context.get_context()

    def clear_context(self) -> None:
        self._context.clear()

    async def close(self) -> None:
        if self._process_task and not self._process_task.done():
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        await self._client.close()
