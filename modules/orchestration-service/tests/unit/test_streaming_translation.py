"""Tests for streaming translation: think-block filter, request_body, translate_stream.

Unit tests use no server — they test the async generator logic directly.
"""
import asyncio
import json
import os
from collections.abc import AsyncIterator

import pytest
from translation.config import TranslationConfig
from translation.llm_client import LLMClient


@pytest.fixture
def config():
    return TranslationConfig(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8006/v1"),
        model=os.getenv("LLM_MODEL", "mlx-community/Qwen3-4B-4bit"),
        temperature=0.7,
        timeout_s=15,
    )


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
