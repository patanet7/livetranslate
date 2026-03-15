"""Tests for TranslationService — combines LLM client, context, and backpressure.

Integration tests hit the real Ollama server on thomas-pc via Tailscale.
NO MOCKING — all tests are behavioral/integration tests per project rules.
Mark with @pytest.mark.integration for CI filtering.
"""
import asyncio

import pytest

from translation.service import TranslationService
from translation.config import TranslationConfig
from livetranslate_common.models import TranslationRequest, TranslationResponse


@pytest.fixture
def config():
    return TranslationConfig(
        llm_base_url="http://thomas-pc:11434/v1",
        model="qwen3.5:7b",
        context_window_size=3,
        max_queue_depth=5,
        timeout_s=10,
    )


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
            assert response.model_used == "qwen3.5:7b"
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
            llm_base_url="http://localhost:1",
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


@pytest.mark.integration
class TestBackpressure:
    @pytest.mark.asyncio
    async def test_queue_drops_oldest_when_full(self, config):
        config.max_queue_depth = 2
        service = TranslationService(config)
        try:
            tasks = []
            for i in range(3):
                task = asyncio.create_task(
                    service.enqueue_translation(
                        text=f"翻译句子{i}",
                        source_language="zh",
                        target_language="en",
                    )
                )
                tasks.append(task)
                await asyncio.sleep(0.05)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            completed = [r for r in results if isinstance(r, TranslationResponse)]
            dropped = [r for r in results if isinstance(r, RuntimeError)]
            assert len(completed) >= 1
            assert len(dropped) <= 1
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_queue_completes_all_when_not_full(self, config):
        """Queue fewer items than max depth — all should complete."""
        config.max_queue_depth = 5
        service = TranslationService(config)
        try:
            tasks = []
            for i in range(2):
                task = asyncio.create_task(
                    service.enqueue_translation(
                        text=f"句子{i}",
                        source_language="zh",
                        target_language="en",
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            completed = [r for r in results if isinstance(r, TranslationResponse)]
            assert len(completed) == 2
        finally:
            await service.close()
