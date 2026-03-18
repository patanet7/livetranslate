"""Tests for translation LLM client — calls Ollama/vLLM directly.

Unit tests use no server. Integration tests hit the local vllm-mlx LLM
instance (:8006 from `just dev`). Override via LLM_BASE_URL / LLM_MODEL env vars.
"""
import httpx
import pytest
from translation.llm_client import LLMClient


@pytest.fixture
def config(llm_config):
    return llm_config


class TestLLMClientUnit:
    def test_build_prompt_no_context(self, config):
        client = LLMClient(config)
        messages = client._build_messages(
            text="你好世界",
            source_language="zh",
            target_language="en",
            context=[],
        )
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert "translate" in messages[0]["content"].lower()
        assert "你好世界" in messages[1]["content"]

    def test_extract_translation_strips_prefixes(self, config):
        client = LLMClient(config)
        assert client._extract_translation("Translation: Hello world") == "Hello world"
        assert client._extract_translation("翻译：你好世界") == "你好世界"
        assert client._extract_translation("译文: 测试") == "测试"
        assert client._extract_translation('"Hello world"') == "Hello world"
        assert client._extract_translation('\u201cHello world\u201d') == "Hello world"
        assert client._extract_translation("Just the translation") == "Just the translation"

    def test_build_prompt_with_context(self, config):
        from livetranslate_common.models import TranslationContext

        client = LLMClient(config)
        context = [
            TranslationContext(text="之前的话", translation="Previous words"),
        ]
        messages = client._build_messages(
            text="这是新的",
            source_language="zh",
            target_language="en",
            context=context,
        )
        user_msg = messages[1]["content"]
        # Context now sends translation only (not source text)
        assert "Previous words" in user_msg
        assert "这是新的" in user_msg


async def _require_llm_server(config) -> None:
    """Skip integration tests cleanly when no local LLM endpoint is running."""
    models_url = f"{config.base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(models_url)
        if response.status_code >= 500:
            pytest.skip(f"LLM endpoint unhealthy at {models_url}")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint not running at {models_url}")


@pytest.mark.integration
class TestLLMClientIntegration:
    @pytest.mark.asyncio
    async def test_translate_simple(self, config):
        await _require_llm_server(config)
        client = LLMClient(config)
        result = await client.translate(
            text="你好世界",
            source_language="zh",
            target_language="en",
        )
        assert result is not None
        assert len(result) > 0
        # Should contain something related to "hello" or "world"

    @pytest.mark.asyncio
    async def test_translate_with_context(self, config):
        await _require_llm_server(config)
        from livetranslate_common.models import TranslationContext

        client = LLMClient(config)
        context = [
            TranslationContext(text="张经理来了", translation="Manager Zhang has arrived"),
        ]
        result = await client.translate(
            text="他说你好",
            source_language="zh",
            target_language="en",
            context=context,
        )
        assert result is not None
