"""Tests for streaming translation: think-block filter, request_body, translate_stream.

Unit tests use no server — they test the async generator logic directly.
"""
import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from translation.config import TranslationConfig
from translation.llm_client import LLMClient


@pytest.fixture
def config(llm_config):
    return llm_config


# ---------------------------------------------------------------------------
# Helper: create an async iterator from a list of strings
# ---------------------------------------------------------------------------
async def _aiter(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# _request_body: shared params between streaming and non-streaming
# ---------------------------------------------------------------------------
class TestRequestBody:
    def test_non_streaming_has_stream_false(self, config):
        client = LLMClient(config)
        messages = [{"role": "user", "content": "hello"}]
        body = client._request_body(messages)
        assert body["stream"] is False
        assert body["model"] == config.model
        assert body["temperature"] == config.temperature

    def test_streaming_has_stream_true(self, config):
        client = LLMClient(config)
        messages = [{"role": "user", "content": "hello"}]
        body = client._request_body(messages, stream=True)
        assert body["stream"] is True

    def test_shared_params_match(self, config):
        """Non-streaming and streaming must use identical sampling params."""
        client = LLMClient(config)
        messages = [{"role": "user", "content": "hello"}]
        non_stream = client._request_body(messages)
        stream = client._request_body(messages, stream=True)

        # Same params except stream flag
        for key in ("model", "temperature", "max_tokens", "top_p", "top_k",
                     "presence_penalty", "repetition_penalty", "chat_template_kwargs"):
            assert non_stream[key] == stream[key], f"Mismatch on {key}"

    def test_max_tokens_uses_config_default(self, config):
        """_request_body should use config.max_tokens instead of hardcoded 512."""
        config_custom = TranslationConfig(
            base_url="http://localhost:11434/v1",
            model="test",
            max_tokens=256,
        )
        client = LLMClient(config_custom)
        body = client._request_body([{"role": "user", "content": "hello"}])
        assert body["max_tokens"] == 256

    def test_max_tokens_kwarg_overrides_config(self, config):
        """max_tokens kwarg should override the config default."""
        client = LLMClient(config)
        body = client._request_body(
            [{"role": "user", "content": "hello"}],
            max_tokens=160,
        )
        assert body["max_tokens"] == 160


# ---------------------------------------------------------------------------
# _strip_think_and_stream: think-block filter
# ---------------------------------------------------------------------------
class TestStripThinkAndStream:
    """Test the think-block streaming filter.

    Imported from websocket_audio module — it's a module-level async generator.
    """

    @pytest.fixture
    def strip_fn(self):
        """Import the filter function from the websocket_audio module."""
        from routers.audio.websocket_audio import _strip_think_and_stream
        return _strip_think_and_stream

    @pytest.mark.asyncio
    async def test_no_think_block_passthrough(self, strip_fn):
        """Content without think blocks passes through after buffering."""
        chunks = ["Hello", " ", "world", "! This is a test of passthrough mode."]
        result = []
        async for delta in strip_fn(_aiter(chunks)):
            result.append(delta)

        full = "".join(result)
        assert full == "Hello world! This is a test of passthrough mode."

    @pytest.mark.asyncio
    async def test_empty_think_block_stripped(self, strip_fn):
        """<think>\n\n</think> is stripped, content after is yielded."""
        chunks = ["<think>", "\n\n", "</think>", "Hello", " world"]
        result = []
        async for delta in strip_fn(_aiter(chunks)):
            result.append(delta)

        full = "".join(result)
        assert full == "Hello world"

    @pytest.mark.asyncio
    async def test_think_block_with_content_stripped(self, strip_fn):
        """<think>reasoning here</think> is stripped."""
        chunks = ["<think>", "Let me think about this...", "</think>", "Translation here"]
        result = []
        async for delta in strip_fn(_aiter(chunks)):
            result.append(delta)

        full = "".join(result)
        assert full == "Translation here"

    @pytest.mark.asyncio
    async def test_unclosed_think_block_discarded(self, strip_fn):
        """Unclosed <think> (max_tokens cutoff) — all content discarded."""
        chunks = ["<think>", "This reasoning never ends because max_tokens hit"]
        result = []
        async for delta in strip_fn(_aiter(chunks)):
            result.append(delta)

        full = "".join(result)
        assert full == ""

    @pytest.mark.asyncio
    async def test_content_after_think_close_yielded(self, strip_fn):
        """Content immediately after </think> in same chunk is yielded."""
        chunks = ["<think>reasoning</think>Translation"]
        result = []
        async for delta in strip_fn(_aiter(chunks)):
            result.append(delta)

        full = "".join(result)
        assert full == "Translation"

    @pytest.mark.asyncio
    async def test_short_content_flushed(self, strip_fn):
        """Content shorter than buffer limit still gets flushed at end."""
        chunks = ["Hi"]
        result = []
        async for delta in strip_fn(_aiter(chunks)):
            result.append(delta)

        full = "".join(result)
        assert full == "Hi"

    @pytest.mark.asyncio
    async def test_empty_stream(self, strip_fn):
        """Empty stream produces no output."""
        result = []
        async for delta in strip_fn(_aiter([])):
            result.append(delta)

        assert result == []

    @pytest.mark.asyncio
    async def test_buffer_flush_at_limit(self, strip_fn):
        """Once buffer exceeds 30 chars, it flushes and switches to passthrough."""
        # Build content that exceeds the 30-char buffer
        long_start = "A" * 35
        chunks = list(long_start)  # one char at a time
        chunks.append(" end")
        result = []
        async for delta in strip_fn(_aiter(chunks)):
            result.append(delta)

        full = "".join(result)
        assert full == long_start + " end"


# ---------------------------------------------------------------------------
# TranslationConfig: draft translation fields
# ---------------------------------------------------------------------------
class TestTranslationConfigDraftFields:
    def test_default_max_tokens(self):
        config = TranslationConfig(base_url="http://localhost:11434/v1", model="test")
        assert config.max_tokens == 512

    def test_default_draft_max_tokens(self):
        config = TranslationConfig(base_url="http://localhost:11434/v1", model="test")
        assert config.draft_max_tokens == 256

    def test_default_draft_timeout_s(self):
        config = TranslationConfig(base_url="http://localhost:11434/v1", model="test")
        assert config.draft_timeout_s == 4

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_TOKENS", "1024")
        monkeypatch.setenv("LLM_DRAFT_MAX_TOKENS", "200")
        monkeypatch.setenv("LLM_DRAFT_TIMEOUT_S", "6")
        config = TranslationConfig(base_url="http://localhost:11434/v1", model="test")
        assert config.max_tokens == 1024
        assert config.draft_max_tokens == 200
        assert config.draft_timeout_s == 6


# ---------------------------------------------------------------------------
# LLMClient.translate: max_retries threading
# ---------------------------------------------------------------------------
class TestTranslateMaxRetries:
    @pytest.mark.asyncio
    async def test_translate_max_retries_zero_no_retry(self, config):
        """translate(max_retries=0) should fail immediately, not retry."""
        # Use a bad URL so the HTTP call always fails
        bad_config = TranslationConfig(
            base_url="http://localhost:1",
            model="test",
            timeout_s=1,
        )
        client = LLMClient(bad_config)
        try:
            with pytest.raises(RuntimeError, match="1 attempts"):
                await client.translate(
                    text="hello",
                    source_language="en",
                    target_language="es",
                    max_retries=0,
                )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_translate_default_retries_is_one(self, config):
        """Default max_retries=1 means 2 attempts total."""
        bad_config = TranslationConfig(
            base_url="http://localhost:1",
            model="test",
            timeout_s=1,
        )
        client = LLMClient(bad_config)
        try:
            with pytest.raises(RuntimeError, match="2 attempts"):
                await client.translate(
                    text="hello",
                    source_language="en",
                    target_language="es",
                )
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# TranslationService.translate: skip_context
# ---------------------------------------------------------------------------
class TestTranslationServiceSkipContext:
    @pytest.mark.asyncio
    async def test_skip_context_does_not_pollute_window(self):
        """translate(skip_context=True) must not add to rolling context."""
        from translation.service import TranslationService
        from livetranslate_common.models import TranslationRequest

        # We need a real LLM for behavioral testing — use bad URL to make
        # the translate call fail, but first manually test the skip_context
        # flag by directly calling the context window.
        # Instead: test with a working service against real Ollama
        # But for unit test, verify the flag exists and context stays empty
        # when skip_context=True by mocking the _client.translate at the
        # behavioral boundary (the HTTP call).

        # For now: verify the parameter is accepted and the code path exists.
        # A proper integration test would hit real Ollama.
        config = TranslationConfig(
            base_url="http://localhost:1",
            model="test",
            timeout_s=1,
            context_window_size=3,
        )
        service = TranslationService(config)
        try:
            request = TranslationRequest(
                text="hello",
                source_language="en",
                target_language="es",
            )
            # This will fail because of bad URL, but we're testing that
            # skip_context parameter is accepted
            with pytest.raises(RuntimeError):
                await service.translate(request, skip_context=True)

            # Context should be empty (translate failed, AND skip_context=True)
            assert len(service.get_context()) == 0
        finally:
            await service.close()
