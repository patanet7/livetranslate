"""Smoke tests for the in-process LLM fake servers.

These are NOT mocks — they are real aiohttp servers bound to localhost on
an OS-assigned port. The merged `LLMClient` makes real HTTP calls against
them, which is consistent with the project's "no mocks" rule.

Phase 4 verifies only that the servers start/stop cleanly and record
requests. Phase 5 uses them to drive `LLMClient` integration tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

# Make `tests/` importable so we can use `from integration.fakes import ...`.
# Matches the pattern in tests/fireflies/integration/test_mock_server_contract.py.
_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from integration.fakes import (  # noqa: E402
    FakeAnthropicServer,
    FakeOllamaServer,
    FakeOpenAIServer,
)


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestFakeOllamaServer:
    async def test_starts_and_stops(self) -> None:
        server = FakeOllamaServer()
        await server.start()
        try:
            assert server.base_url.startswith("http://")
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{server.base_url}/v1/models")
            assert resp.status_code == 200
        finally:
            await server.stop()

    async def test_records_chat_completions_request(self) -> None:
        server = FakeOllamaServer()
        await server.start()
        try:
            server.set_response_text("Hello")
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{server.base_url}/v1/chat/completions",
                    json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test"},
                )
            assert resp.status_code == 200
            assert len(server.recorded_requests) == 1
            recorded = server.recorded_requests[0]
            assert recorded["path"] == "/v1/chat/completions"
            assert recorded["headers"].get("authorization") == "Bearer test"
            assert recorded["json"]["model"] == "x"
        finally:
            await server.stop()

    async def test_records_native_generate_request(self) -> None:
        server = FakeOllamaServer()
        await server.start()
        try:
            server.set_response_text("Hola")
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{server.base_url}/api/generate",
                    json={"model": "qwen3", "prompt": "translate"},
                )
            assert resp.status_code == 200
            data = resp.json()
            assert data["response"] == "Hola"
            assert any(r["path"] == "/api/generate" for r in server.recorded_requests)
        finally:
            await server.stop()


class TestFakeOpenAIServer:
    async def test_starts_and_stops(self) -> None:
        server = FakeOpenAIServer()
        await server.start()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{server.base_url}/v1/models")
            assert resp.status_code == 200
        finally:
            await server.stop()

    async def test_require_api_key_enforces_auth(self) -> None:
        server = FakeOpenAIServer()
        await server.start()
        try:
            server.set_response_text("ok")
            server.require_api_key("sk-correct")
            async with httpx.AsyncClient() as client:
                # Missing header
                r1 = await client.post(
                    f"{server.base_url}/v1/chat/completions",
                    json={"model": "x", "messages": []},
                )
                assert r1.status_code == 401
                # Wrong key
                r2 = await client.post(
                    f"{server.base_url}/v1/chat/completions",
                    json={"model": "x", "messages": []},
                    headers={"Authorization": "Bearer sk-wrong"},
                )
                assert r2.status_code == 401
                # Correct key
                r3 = await client.post(
                    f"{server.base_url}/v1/chat/completions",
                    json={"model": "x", "messages": []},
                    headers={"Authorization": "Bearer sk-correct"},
                )
                assert r3.status_code == 200
        finally:
            await server.stop()

    async def test_fail_n_times_recovers(self) -> None:
        server = FakeOpenAIServer()
        await server.start()
        try:
            server.set_response_text("ok")
            server.fail_n_times(2, status=503)
            async with httpx.AsyncClient() as client:
                r1 = await client.post(
                    f"{server.base_url}/v1/chat/completions",
                    json={"model": "x", "messages": []},
                )
                assert r1.status_code == 503
                r2 = await client.post(
                    f"{server.base_url}/v1/chat/completions",
                    json={"model": "x", "messages": []},
                )
                assert r2.status_code == 503
                r3 = await client.post(
                    f"{server.base_url}/v1/chat/completions",
                    json={"model": "x", "messages": []},
                )
                assert r3.status_code == 200
        finally:
            await server.stop()


class TestFixtureWiring:
    async def test_fake_ollama_fixture(self, fake_ollama_server: FakeOllamaServer) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{fake_ollama_server.base_url}/v1/models")
        assert resp.status_code == 200

    async def test_fake_openai_fixture(self, fake_openai_server: FakeOpenAIServer) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{fake_openai_server.base_url}/v1/models")
        assert resp.status_code == 200

    async def test_llm_connection_factory_defaults(self, llm_connection_factory) -> None:
        conn = llm_connection_factory()
        assert conn.engine == "openai_compatible"
        assert conn.model == "test-model"
        assert conn.temperature == 0.3

    async def test_llm_connection_factory_overrides(self, llm_connection_factory) -> None:
        conn = llm_connection_factory(engine="anthropic", model="claude-3-5", temperature=0.0)
        assert conn.engine == "anthropic"
        assert conn.model == "claude-3-5"
        assert conn.temperature == 0.0


class TestFakeAnthropicServer:
    async def test_starts_and_records_messages_request(self) -> None:
        server = FakeAnthropicServer()
        await server.start()
        try:
            server.set_response_text("hi back")
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{server.base_url}/v1/messages",
                    json={
                        "model": "claude-3-5-sonnet",
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    headers={"x-api-key": "ant-key", "anthropic-version": "2023-06-01"},
                )
            assert resp.status_code == 200
            assert any(r["path"] == "/v1/messages" for r in server.recorded_requests)
            assert server.recorded_requests[0]["headers"].get("x-api-key") == "ant-key"
        finally:
            await server.stop()
