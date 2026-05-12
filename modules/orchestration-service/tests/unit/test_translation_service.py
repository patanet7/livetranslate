"""TranslationService — backpressure and per-direction context.

After the consolidation, endpoint identity lives on `LLMConnection`. These
tests construct a connection pointed at an unreachable URL so the queue /
context behavior can be exercised without depending on any real LLM. The
"hits a real backend" tests moved to `tests/integration/test_translation_*`.
"""

from __future__ import annotations

import asyncio

import pytest

from livetranslate_common.models import TranslationRequest
from livetranslate_common.models.llm import LLMConnection
from translation.config import TranslationConfig
from translation.service import TranslationService


def _unreachable_connection() -> LLMConnection:
    return LLMConnection(
        engine="openai_compatible",
        base_url="http://127.0.0.1:1/v1",
        model="nonexistent",
        max_retries=0,
        timeout_s=1.0,
    )


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_queue_rejects_newest_when_full(self) -> None:
        """When the queue is full, newest enqueue rejects immediately.

        Schedule three tasks atomically, yield once so coroutines reach their
        first suspension point. The first claims the single slot, the second
        and third see queue.full() and raise RuntimeError before yielding.
        """
        behavioral = TranslationConfig(max_queue_depth=1)
        service = TranslationService(_unreachable_connection(), behavioral)
        try:
            tasks = [
                asyncio.create_task(
                    service.enqueue_translation(
                        TranslationRequest(
                            text=f"句子{i}",
                            source_language="zh",
                            target_language="en",
                        )
                    )
                )
                for i in range(3)
            ]
            await asyncio.sleep(0)
            for t in tasks:
                if not t.done():
                    t.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            rejected = [
                r for r in results
                if isinstance(r, RuntimeError) and "rejected" in str(r)
            ]
            assert len(rejected) >= 1
        finally:
            await service.close()


class TestPerDirectionContext:
    @pytest.mark.asyncio
    async def test_per_direction_context_isolation(self) -> None:
        """Different language directions keep isolated rolling context windows."""
        service = TranslationService(_unreachable_connection(), TranslationConfig())
        try:
            service.context_store.add("en", "zh", "Hello", "你好")
            service.context_store.add("zh", "en", "再见", "Goodbye")

            en_zh = service.get_context("en", "zh")
            zh_en = service.get_context("zh", "en")
            ja_en = service.get_context("ja", "en")

            assert [c.text for c in en_zh] == ["Hello"]
            assert [c.text for c in zh_en] == ["再见"]
            assert ja_en == []

            service.clear_context("en", "zh")
            assert service.get_context("en", "zh") == []
            # other direction untouched
            assert len(service.get_context("zh", "en")) == 1

            service.clear_context()
            assert service.get_context("zh", "en") == []
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_base_connection_exposed(self) -> None:
        """TranslationService exposes its base connection as a read-only property."""
        conn = _unreachable_connection()
        service = TranslationService(conn, TranslationConfig())
        try:
            assert service.base_connection is conn
            assert service.base_connection.model == "nonexistent"
        finally:
            await service.close()
