"""Shared base for in-process LLM fake servers.

Why aiohttp and not starlette/uvicorn? The repo already uses aiohttp for
`FirefliesMockServer` (tests/fireflies/mocks/fireflies_mock_server.py:555) and
aiohttp's `web.AppRunner` + OS-assigned port idiom is the lightest way to
spin a server up inside a pytest-asyncio fixture without leaking processes.

Each fake exposes:
- `.base_url` — http://127.0.0.1:<port> once started
- `.recorded_requests` — list of {method, path, headers, json, query, body}
- knobs: set_response_text, set_stream_chunks, fail_with_status,
  fail_n_times, require_api_key, delay_seconds
"""

from __future__ import annotations

import asyncio
import json as _json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    json: Any | None
    query: dict[str, str]
    body: bytes


class FakeLLMServerBase:
    """Common lifecycle + knobs for fake LLM HTTP servers.

    Subclasses add the route handlers in `_install_routes()`.
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port: int | None = None  # assigned by OS at .start()

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

        # Response shape
        self._response_text: str = "OK"
        self._thinking_response: tuple[str, str] | None = None  # (reasoning, content)
        self._stream_chunks: list[str] = []

        # Failure injection
        self._fail_with_status: int | None = None
        self._fail_n_times: int = 0
        self._fail_status: int = 500

        # Auth
        self._required_api_key: str | None = None
        self._api_key_header: str = "authorization"  # subclass override (Anthropic uses x-api-key)
        self._api_key_prefix: str = "Bearer "  # subclass override (Anthropic uses raw key)

        # Timing
        self._delay_seconds: float = 0.0

        # Capture
        self.recorded_requests: list[dict[str, Any]] = []

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._app = web.Application()
        self._install_routes(self._app)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, 0)
        await self._site.start()
        # Discover the OS-assigned port via the server's first listening socket.
        srv = self._site._server  # type: ignore[attr-defined]
        sockets = srv.sockets if srv else None
        if not sockets:
            raise RuntimeError("fake server failed to bind a socket")
        self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._site is not None:
            await self._site.stop()
        if self._runner is not None:
            await self._runner.cleanup()
        self._site = None
        self._runner = None
        self._app = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise RuntimeError("server not started")
        return f"http://{self.host}:{self.port}"

    # -- knobs ---------------------------------------------------------------

    def set_response_text(self, text: str) -> None:
        self._response_text = text
        self._thinking_response = None

    def set_thinking_response(self, reasoning: str, content: str = "") -> None:
        """Make the server return `reasoning` in a separate field
        (mimics the Qwen3-on-Ollama OpenAI-compat-layer bug)."""
        self._thinking_response = (reasoning, content)

    def set_stream_chunks(self, chunks: list[str]) -> None:
        self._stream_chunks = list(chunks)

    def fail_with_status(self, status: int) -> None:
        """Every request returns `status`."""
        self._fail_with_status = status

    def fail_n_times(self, n: int, status: int = 500) -> None:
        """The next `n` requests return `status`; subsequent succeed."""
        self._fail_n_times = n
        self._fail_status = status

    def require_api_key(self, key: str) -> None:
        self._required_api_key = key

    def delay_seconds(self, seconds: float) -> None:
        self._delay_seconds = seconds

    # -- subclass hook -------------------------------------------------------

    def _install_routes(self, app: web.Application) -> None:
        raise NotImplementedError

    # -- helpers for handlers ------------------------------------------------

    async def _record(self, request: web.Request) -> dict[str, Any]:
        body = await request.read()
        try:
            parsed_json = _json.loads(body) if body else None
        except _json.JSONDecodeError:
            parsed_json = None
        record = {
            "method": request.method,
            "path": request.path,
            "headers": {k.lower(): v for k, v in request.headers.items()},
            "json": parsed_json,
            "query": dict(request.query),
            "body": body,
        }
        self.recorded_requests.append(record)
        return record

    def _maybe_fail(self) -> web.Response | None:
        """Return an error response if failure injection is configured."""
        if self._fail_with_status is not None:
            return web.json_response(
                {"error": "injected failure"}, status=self._fail_with_status
            )
        if self._fail_n_times > 0:
            self._fail_n_times -= 1
            return web.json_response(
                {"error": "transient failure"}, status=self._fail_status
            )
        return None

    def _maybe_unauthorized(self, record: dict[str, Any]) -> web.Response | None:
        if self._required_api_key is None:
            return None
        header_val = record["headers"].get(self._api_key_header.lower(), "")
        expected = (
            f"{self._api_key_prefix}{self._required_api_key}"
            if self._api_key_prefix
            else self._required_api_key
        )
        if header_val != expected:
            return web.json_response(
                {"error": "unauthorized"}, status=401
            )
        return None

    async def _maybe_delay(self) -> None:
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)

    # -- common precheck pipeline -------------------------------------------

    async def _precheck(self, request: web.Request) -> tuple[dict[str, Any], web.Response | None]:
        record = await self._record(request)
        await self._maybe_delay()
        for check in (self._maybe_unauthorized(record), self._maybe_fail()):
            if check is not None:
                return record, check
        return record, None

    # -- SSE helpers (OpenAI-compat streaming) ------------------------------

    async def _sse_response(
        self, request: web.Request, chunks: list[str], model: str
    ) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
        )
        await resp.prepare(request)
        for chunk in chunks:
            payload = {
                "choices": [{"delta": {"content": chunk}, "index": 0, "finish_reason": None}],
                "model": model,
            }
            await resp.write(f"data: {_json.dumps(payload)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _ollama_native_stream(
        self, request: web.Request, chunks: list[str], model: str
    ) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "application/x-ndjson"},
        )
        await resp.prepare(request)
        for chunk in chunks:
            payload = {"model": model, "response": chunk, "done": False}
            await resp.write((_json.dumps(payload) + "\n").encode())
        await resp.write((_json.dumps({"model": model, "response": "", "done": True}) + "\n").encode())
        await resp.write_eof()
        return resp
