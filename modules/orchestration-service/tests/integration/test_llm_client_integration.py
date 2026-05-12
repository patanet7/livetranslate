"""Integration tests for the merged LLMClient.

Real httpx calls against in-process aiohttp fakes (see tests/integration/fakes/).
Each test exercises one feature of the merged client end-to-end: auth, retry,
circuit breaker, Qwen3-native fallback, Anthropic engine path, SSE streaming,
NDJSON streaming, proxy mode.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Make tests/ importable so `from integration.fakes import ...` works.
_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from livetranslate_common.llm.client import (  # noqa: E402
    CircuitBreakerOpenError,
    LLMClient,
)
from livetranslate_common.models.llm import (  # noqa: E402
    LLMConnection,
    LLMParameterOverrides,
)


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Engine dispatch + auth
# ---------------------------------------------------------------------------


async def test_openai_auth_header_sent(fake_openai_server, llm_connection_factory) -> None:
    fake_openai_server.require_api_key("sk-test")
    fake_openai_server.set_response_text("Hello")
    conn = llm_connection_factory(
        engine="openai",
        base_url=fake_openai_server.base_url + "/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
    )
    async with LLMClient(conn) as client:
        out = await client.complete(messages=[{"role": "user", "content": "hi"}])
    assert out == "Hello"
    req = fake_openai_server.recorded_requests[-1]
    assert req["headers"]["authorization"] == "Bearer sk-test"


async def test_openai_compatible_no_auth_when_empty_key(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.set_response_text("ok")
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        api_key="",
        model="local",
    )
    async with LLMClient(conn) as client:
        await client.complete(messages=[{"role": "user", "content": "x"}])
    req = fake_openai_server.recorded_requests[-1]
    # No Authorization header when api_key is empty
    assert "authorization" not in req["headers"]


async def test_anthropic_headers_and_path(
    fake_anthropic_server, llm_connection_factory
) -> None:
    fake_anthropic_server.set_response_text("hi back")
    conn = llm_connection_factory(
        engine="anthropic",
        base_url=fake_anthropic_server.base_url,
        api_key="ant-key",
        model="claude-3-5-sonnet",
    )
    async with LLMClient(conn) as client:
        out = await client.complete(messages=[{"role": "user", "content": "yo"}])
    assert out == "hi back"
    req = fake_anthropic_server.recorded_requests[-1]
    assert req["path"] == "/v1/messages"
    assert req["headers"]["x-api-key"] == "ant-key"
    assert "anthropic-version" in req["headers"]


# ---------------------------------------------------------------------------
# Ollama-native Qwen3 fallback
# ---------------------------------------------------------------------------


async def test_ollama_native_path_used_for_qwen3(
    fake_ollama_server, llm_connection_factory
) -> None:
    """Connection with engine=ollama + model=qwen3:* hits /api/generate, not /v1/chat/completions."""
    fake_ollama_server.set_response_text("Hola")
    conn = llm_connection_factory(
        engine="ollama",
        base_url=fake_ollama_server.base_url,
        model="qwen3:14b",
    )
    async with LLMClient(conn) as client:
        out = await client.complete(messages=[{"role": "user", "content": "Hello"}])
    assert out == "Hola"
    paths = [r["path"] for r in fake_ollama_server.recorded_requests]
    assert "/api/generate" in paths
    assert "/v1/chat/completions" not in paths


async def test_ollama_qwen3_thinking_field_extraction(
    fake_ollama_server, llm_connection_factory
) -> None:
    """Qwen3 with empty `response` + non-empty `thinking` field — client salvages translation."""
    fake_ollama_server.set_thinking_response(
        reasoning="let me reason about this...\nFinal Decision: Hello world",
        content="",
    )
    conn = llm_connection_factory(
        engine="ollama",
        base_url=fake_ollama_server.base_url,
        model="qwen3:14b",
    )
    async with LLMClient(conn) as client:
        out = await client.complete(messages=[{"role": "user", "content": "你好"}])
    assert out == "Hello world"


async def test_ollama_non_qwen3_uses_openai_compat(
    fake_ollama_server, llm_connection_factory
) -> None:
    """Ollama + llama3 (no qwen3 substring) → OpenAI-compat path, not /api/generate."""
    fake_ollama_server.set_response_text("ok")
    conn = llm_connection_factory(
        engine="ollama",
        base_url=fake_ollama_server.base_url,
        model="llama3.1:8b",
    )
    async with LLMClient(conn) as client:
        await client.complete(messages=[{"role": "user", "content": "x"}])
    paths = [r["path"] for r in fake_ollama_server.recorded_requests]
    assert "/v1/chat/completions" in paths
    assert "/api/generate" not in paths


# ---------------------------------------------------------------------------
# Retry + circuit breaker
# ---------------------------------------------------------------------------


async def test_retry_one_failure_then_success(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.set_response_text("OK")
    fake_openai_server.fail_n_times(1, status=503)
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        model="m",
        max_retries=2,
    )
    async with LLMClient(conn) as client:
        out = await client.complete(messages=[{"role": "user", "content": "x"}])
    assert out == "OK"
    # 1 fail + 1 success = 2 recorded
    assert len(fake_openai_server.recorded_requests) == 2


async def test_retry_exhausts_then_raises(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.fail_with_status(500)
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        model="m",
        max_retries=2,
    )
    async with LLMClient(conn) as client:
        with pytest.raises(Exception):
            await client.complete(messages=[{"role": "user", "content": "x"}])
    # max_retries=2 means up to 3 attempts
    assert len(fake_openai_server.recorded_requests) == 3


async def test_circuit_breaker_opens_after_consecutive_failures(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.fail_with_status(500)
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        model="m",
        max_retries=0,
    )
    # breaker_recovery_s=10 keeps it OPEN long enough for the assertion
    async with LLMClient(conn, breaker_failure_threshold=3, breaker_recovery_s=10.0) as client:
        for _ in range(3):
            with pytest.raises(Exception):
                await client.complete(messages=[{"role": "user", "content": "x"}])
        before = len(fake_openai_server.recorded_requests)
        with pytest.raises(CircuitBreakerOpenError):
            await client.complete(messages=[{"role": "user", "content": "x"}])
        # Breaker should fast-fail — no extra request
        assert len(fake_openai_server.recorded_requests) == before


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_streaming_sse_yields_chunks(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.set_stream_chunks(["Hello", " ", "world"])
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        model="m",
    )
    async with LLMClient(conn) as client:
        out_chunks = []
        async for delta in client.stream(messages=[{"role": "user", "content": "hi"}]):
            out_chunks.append(delta)
    assert "".join(out_chunks) == "Hello world"


async def test_streaming_ollama_native_parses_ndjson(
    fake_ollama_server, llm_connection_factory
) -> None:
    fake_ollama_server.set_stream_chunks(["Bonj", "our"])
    conn = llm_connection_factory(
        engine="ollama",
        base_url=fake_ollama_server.base_url,
        model="qwen3:14b",
    )
    async with LLMClient(conn) as client:
        out_chunks = []
        async for delta in client.stream(messages=[{"role": "user", "content": "Hello"}]):
            out_chunks.append(delta)
    assert "".join(out_chunks) == "Bonjour"


# ---------------------------------------------------------------------------
# Proxy mode
# ---------------------------------------------------------------------------


async def test_proxy_mode_routes_to_translate_endpoint(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.set_response_text("Hola")
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url,  # no /v1 — proxy mode targets the translation service
        model="proxy-model",
    )
    async with LLMClient(conn, proxy_mode=True) as client:
        await client.complete(messages=[{"role": "user", "content": "Hello"}])
    paths = [r["path"] for r in fake_openai_server.recorded_requests]
    assert any(p == "/api/v3/translate" for p in paths)


# ---------------------------------------------------------------------------
# Per-call overrides land in the request body
# ---------------------------------------------------------------------------


async def test_request_body_carries_temperature_override(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.set_response_text("ok")
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        model="m",
        temperature=0.7,
    )
    async with LLMClient(conn) as client:
        await client.complete(
            messages=[{"role": "user", "content": "x"}],
            overrides=LLMParameterOverrides(temperature=0.2, max_tokens=99, top_p=0.5),
        )
    body = fake_openai_server.recorded_requests[-1]["json"]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 99
    assert body["top_p"] == 0.5
