"""TranslationService — combines LLM client, rolling context, and backpressure.

This is the high-level interface that the meeting pipeline calls.
It manages:
- LLM client for actual translation
- Rolling context window for quality
- Bounded queue with drop-newest backpressure
"""
from __future__ import annotations

import asyncio
import time

from livetranslate_common.logging import get_logger
from livetranslate_common.models import TranslationContext, TranslationRequest, TranslationResponse

from translation.config import TranslationConfig
from translation.context_store import DirectionalContextStore
from translation.llm_client import LLMClient

logger = get_logger()


class TranslationService:
    def __init__(
        self,
        config: TranslationConfig,
        context_store: DirectionalContextStore | None = None,
    ):
        self.config = config
        self._client = LLMClient(config)
        self.context_store = context_store or DirectionalContextStore(
            max_entries=config.context_window_size,
            max_tokens=config.max_context_tokens,
            cross_direction_max_tokens=config.cross_direction_max_tokens,
        )
        self._queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=config.max_queue_depth)
        self._processing = False
        self._process_task: asyncio.Task | None = None
        self._concurrency = asyncio.Semaphore(min(config.max_queue_depth, 3))

    async def translate_draft(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: list[TranslationContext] | None = None,
    ) -> str:
        """Draft translation: fast, fail-fast, no context write-back.

        Uses draft_max_tokens for shorter output and max_retries=0 for
        immediate failure. The caller passes context from context_store.get()
        — this method never reads or writes the internal context store.
        """
        return await self._client.translate(
            text=text,
            source_language=source_lang,
            target_language=target_lang,
            context=context,
            max_tokens=self.config.draft_max_tokens,
            max_retries=0,
        )

    async def translate(
        self,
        request: TranslationRequest,
        skip_context: bool = False,
    ) -> TranslationResponse:
        """Translate text synchronously (blocking until complete).

        Accepts a TranslationRequest (from Plan 0 shared contracts) instead
        of bare parameters. Context from the request is merged with the
        rolling context window.

        Args:
            skip_context: If True, skip writing to the rolling context window.
                Safety net for draft translations that should never pollute context.

        Used for direct translation. For queued translation with
        backpressure, use enqueue_translation().
        """
        # Merge: use request context if provided, otherwise use directional rolling window
        context = request.context if request.context else self.context_store.get(
            request.source_language, request.target_language,
        )
        start = time.monotonic()

        translated = await self._client.translate(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            context=context,
            glossary_terms=request.glossary_terms,
        )

        latency_ms = (time.monotonic() - start) * 1000

        # Only add to context on success, keyed by direction
        if not skip_context:
            self.context_store.add(
                request.source_language, request.target_language,
                request.text, translated,
            )

        return TranslationResponse(
            translated_text=translated,
            source_language=request.source_language,
            target_language=request.target_language,
            model_used=self.config.model,
            latency_ms=round(latency_ms, 1),
        )

    async def enqueue_translation(self, request: TranslationRequest) -> TranslationResponse:
        """Queue a translation request with backpressure.

        If queue is full, rejects the NEWEST (incoming) request.
        For live captions, the oldest queued item is closest to what the
        viewer expects next — dropping it creates visible caption gaps.
        """
        future: asyncio.Future[TranslationResponse] = asyncio.get_running_loop().create_future()

        if self._queue.full():
            logger.warning("translation_rejected", text_len=len(request.text), reason="queue_full")
            raise RuntimeError("Translation request rejected (backpressure — queue full)")

        await self._queue.put((request, future))

        if not self._processing:
            self._processing = True
            self._process_task = asyncio.create_task(self._process_queue())

        return await future

    async def _process_queue(self) -> None:
        """Process queued translation requests with bounded concurrency."""
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
        """Process a single translation with concurrency limit."""
        async with self._concurrency:
            try:
                result = await self.translate(request)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

    def get_context(
        self,
        source_lang: str = "",
        target_lang: str = "",
    ) -> list[TranslationContext]:
        """Return context entries for the given direction."""
        return self.context_store.get(source_lang, target_lang)

    def clear_context(
        self,
        source_lang: str = "",
        target_lang: str = "",
    ) -> None:
        """Clear context. If both langs are empty, clears all directions."""
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
        await self._client.close()
