"""Tests for translation LLM client — calls Ollama/vLLM directly.

These tests hit the real Ollama server on thomas-pc via Tailscale.
Mark with @pytest.mark.integration for CI filtering.
"""
import pytest
from translation.config import TranslationConfig
from translation.llm_client import LLMClient


@pytest.fixture
def config():
    return TranslationConfig(
        llm_base_url="http://thomas-pc:11434/v1",
        model="qwen3.5:7b",
        temperature=0.3,
        timeout_s=10,
    )


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
        assert "translator" in messages[0]["content"].lower()
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
        assert "之前的话" in user_msg
        assert "Previous words" in user_msg
        assert "这是新的" in user_msg


@pytest.mark.integration
class TestLLMClientIntegration:
    @pytest.mark.asyncio
    async def test_translate_simple(self, config):
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
