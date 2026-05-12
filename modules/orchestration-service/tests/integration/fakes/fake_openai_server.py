"""In-process OpenAI-compatible HTTP fake.

Routes:
    GET  /v1/models                 — model list
    POST /v1/chat/completions       — chat completions (SSE streaming optional)
    POST /api/v3/translate          — Translation-Service V3 proxy mode
    POST /api/v3/translate/stream   — streaming proxy mode
"""

from __future__ import annotations

import json as _json

from aiohttp import web

from ._base import FakeLLMServerBase


class FakeOpenAIServer(FakeLLMServerBase):
    def _install_routes(self, app: web.Application) -> None:
        app.router.add_get("/v1/models", self._models)
        app.router.add_post("/v1/chat/completions", self._chat_completions)
        app.router.add_post("/api/v3/translate", self._translate_proxy)
        app.router.add_post("/api/v3/translate/stream", self._translate_proxy_stream)

    async def _models(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response({"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]})

    async def _chat_completions(self, request: web.Request) -> web.StreamResponse:
        record, early = await self._precheck(request)
        if early is not None:
            return early
        body = record["json"] or {}
        stream = bool(body.get("stream"))
        model = body.get("model", "gpt-4o-mini")
        if stream:
            return await self._sse_response(
                request, self._stream_chunks or [self._response_text], model
            )
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": self._response_text},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "model": model,
            }
        )

    async def _translate_proxy(self, request: web.Request) -> web.Response:
        _record, early = await self._precheck(request)
        if early is not None:
            return early
        return web.json_response({"translated_text": self._response_text})

    async def _translate_proxy_stream(self, request: web.Request) -> web.StreamResponse:
        record, early = await self._precheck(request)
        if early is not None:
            return early
        body = record["json"] or {}
        model = body.get("model", "proxy")
        return await self._sse_response(
            request, self._stream_chunks or [self._response_text], model
        )
