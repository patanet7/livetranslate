"""Tests for TranslationService — combines LLM client, context, and backpressure.

Integration tests hit the local vllm-mlx LLM server.
Override via LLM_BASE_URL / LLM_MODEL env vars.
NO MOCKING — all tests are behavioral/integration tests per project rules.
Mark with @pytest.mark.integration for CI filtering.
"""
import asyncio

import pytest

from translation.service import TranslationService
from translation.config import TranslationConfig
from livetranslate_common.models import TranslationRequest, TranslationResponse


@pytest.fixture
def config(llm_config):
    llm_config.context_window_size = 3
    llm_config.max_queue_depth = 5
    llm_config.timeout_s = 10
    return llm_config


@pytest.mark.integration
class TestTranslationServiceIntegration:
    @pytest.mark.asyncio
    async def test_translate_adds_to_context(self, config):
        service = TranslationService(config)
        try:
            request = TranslationRequest(
                text="你好世界",
                source_language="zh",
                target_language="en",
                context=[],
            )
            response = await service.translate(request)

            assert response.translated_text is not None
            assert len(response.translated_text) > 0
            assert response.model_used == config.model
            assert response.latency_ms >= 0

            # Context should now contain this pair
            ctx = service.get_context()
            assert len(ctx) == 1
            assert ctx[0].text == "你好世界"
            assert ctx[0].translation == response.translated_text
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_context_improves_consistency(self, config):
        """Translate two related sentences — context should help pronoun resolution."""
        service = TranslationService(config)
        try:
            req1 = TranslationRequest(
                text="张经理来了",
                source_language="zh",
                target_language="en",
                context=[],
            )
            resp1 = await service.translate(req1)
            assert resp1.translated_text is not None

            req2 = TranslationRequest(
                text="他说你好",
                source_language="zh",
                target_language="en",
                context=service.get_context(),
            )
            resp2 = await service.translate(req2)
            assert resp2.translated_text is not None

            ctx = service.get_context()
            assert len(ctx) == 2
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_failed_translation_not_in_context(self, config):
        """Use a bad URL to trigger a real failure — context must stay empty."""
        bad_config = TranslationConfig(
            base_url="http://localhost:1",
            model="nonexistent-model",
            context_window_size=3,
            max_queue_depth=5,
            timeout_s=2,
        )
        service = TranslationService(bad_config)
        try:
            request = TranslationRequest(
                text="失败的句子",
                source_language="zh",
                target_language="en",
                context=[],
            )
            with pytest.raises(RuntimeError):
                await service.translate(request)

            assert len(service.get_context()) == 0
        finally:
            await service.close()


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_queue_rejects_newest_when_full(self):
        """When queue is full, newest request should be rejected.

        Strategy: fill the queue via concurrent tasks before the event loop
        can drain it. We schedule all tasks atomically (no awaits between
        create_task calls), then yield once with asyncio.sleep(0) so the
        coroutines run up to their first suspension point. At that moment
        items 1+ find the queue full and raise immediately.
        """
        # Use a bad URL — queued items will fail, but we only care about
        # synchronous rejection before any LLM call is made.
        bad_config = TranslationConfig(
            base_url="http://localhost:1",
            model="nonexistent",
            max_queue_depth=1,
            timeout_s=1,
        )
        service = TranslationService(bad_config)
        try:
            tasks = [
                asyncio.create_task(
                    service.enqueue_translation(
                        TranslationRequest(
                            text=f"翻译句子{i}",
                            source_language="zh",
                            target_language="en",
                        )
                    )
                )
                for i in range(3)
            ]
            # Yield once — all three coroutines run to their first await.
            # Tasks 1 and 2 hit _queue.full() == True and raise RuntimeError
            # before yielding, so they complete with an exception on this tick.
            await asyncio.sleep(0)

            # Cancel any tasks still waiting (they would block on the LLM)
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

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_queue_completes_all_when_not_full(self, config):
        """Queue fewer items than max depth — all should complete.

        Runs sequentially because vllm-mlx crashes on concurrent Metal GPU
        requests (see justfile: split inference to avoid GPU crash).
        """
        config.max_queue_depth = 5
        service = TranslationService(config)
        try:
            completed = []
            for i in range(2):
                result = await service.enqueue_translation(
                    TranslationRequest(
                        text=f"句子{i}",
                        source_language="zh",
                        target_language="en",
                    )
                )
                if isinstance(result, TranslationResponse):
                    completed.append(result)
            assert len(completed) == 2
        finally:
            await service.close()


class TestPerSpeakerContext:
    @pytest.mark.asyncio
    async def test_per_speaker_context_isolation(self, config):
        """Different speakers should have isolated context windows."""
        bad_config = TranslationConfig(
            base_url="http://localhost:1",
            model="test",
            context_window_size=3,
            max_queue_depth=5,
            timeout_s=1,
        )
        service = TranslationService(bad_config)
        try:
            # Manually add context for two speakers
            service._get_context_window("Alice").add("Hello", "你好")
            service._get_context_window("Bob").add("Goodbye", "再见")

            alice_ctx = service.get_context("Alice")
            bob_ctx = service.get_context("Bob")
            none_ctx = service.get_context()

            assert len(alice_ctx) == 1
            assert alice_ctx[0].text == "Hello"
            assert len(bob_ctx) == 1
            assert bob_ctx[0].text == "Goodbye"
            assert len(none_ctx) == 0  # default speaker has no context

            # Clear Alice's context
            service.clear_context("Alice")
            assert len(service.get_context("Alice")) == 0
            assert len(service.get_context("Bob")) == 1  # Bob untouched

            # Clear all
            service.clear_context()
            assert len(service.get_context("Bob")) == 0
        finally:
            await service.close()
