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
        self._contexts: dict[str | None, RollingContextWindow] = {}
        self._queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=config.max_queue_depth)
        self._processing = False
        self._process_task: asyncio.Task | None = None

    def _get_context_window(self, speaker: str | None = None) -> RollingContextWindow:
        """Get or create a context window for a speaker."""
        if speaker not in self._contexts:
            self._contexts[speaker] = RollingContextWindow(
                max_entries=self.config.context_window_size,
                max_tokens=self.config.max_context_tokens,
            )
        return self._contexts[speaker]

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
        # Merge: use request context if provided, otherwise use per-speaker rolling window
        context = request.context if request.context else self._get_context_window(request.speaker_name).get_context()
        start = time.monotonic()

        translated = await self._client.translate(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            context=context,
            glossary_terms=request.glossary_terms,
        )

        latency_ms = (time.monotonic() - start) * 1000

        # Only add to context on success, keyed by speaker
        self._get_context_window(request.speaker_name).add(request.text, translated)

        return TranslationResponse(
            translated_text=translated,
            source_language=request.source_language,
            target_language=request.target_language,
            model_used=self.config.model,
            latency_ms=round(latency_ms, 1),
        )

    async def enqueue_translation(self, request: TranslationRequest) -> TranslationResponse:
        """Queue a translation request with backpressure.

        If queue is full, drops the oldest pending request.
        Accepts a TranslationRequest so speaker_name and all fields are preserved.
        """
        future: asyncio.Future[TranslationResponse] = asyncio.get_running_loop().create_future()

        if self._queue.full():
            # Drop oldest
            try:
                old_request, old_future = self._queue.get_nowait()
                old_future.set_exception(
                    RuntimeError("Translation request dropped (backpressure)")
                )
                logger.warning("translation_dropped", text_len=len(old_request.text))
            except asyncio.QueueEmpty:
                pass

        await self._queue.put((request, future))

        if not self._processing:
            self._processing = True
            self._process_task = asyncio.create_task(self._process_queue())

        return await future

    async def _process_queue(self) -> None:
        """Process queued translation requests."""
        try:
            while not self._queue.empty():
                request, future = await self._queue.get()
                try:
                    result = await self.translate(request)
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
        finally:
            self._processing = False

    def get_context(self, speaker: str | None = None) -> list[TranslationContext]:
        return self._get_context_window(speaker).get_context()

    def clear_context(self, speaker: str | None = None) -> None:
        if speaker is None:
            self._contexts.clear()
        elif speaker in self._contexts:
            self._contexts[speaker].clear()

    async def close(self) -> None:
        if self._process_task and not self._process_task.done():
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        await self._client.close()
