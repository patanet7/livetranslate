"""In-process Ollama-shaped HTTP fake.

Routes:
    GET  /v1/models                — OpenAI-compat model list
    POST /v1/chat/completions      — OpenAI-compat chat completions
    POST /api/chat                  — Native Ollama chat
    POST /api/generate              — Native Ollama generate (used by Qwen3 fallback)

Streaming on /v1/chat/completions uses SSE (data: …\\n\\n + [DONE]).
Streaming on /api/generate uses NDJSON (one JSON per line, terminator {"done": true}).
"""

from __future__ import annotations

import json as _json

from aiohttp import web

from ._base import FakeLLMServerBase


class FakeOllamaServer(FakeLLMServerBase):
    def _install_routes(self, app: web.Application) -> None:
        app.router.add_get("/v1/models", self._models)
        app.router.add_post("/v1/chat/completions", self._chat_completions)
        app.router.add_post("/api/chat", self._native_chat)
        app.router.add_post("/api/generate", self._native_generate)

    async def _models(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response({"data": [{"id": "qwen3:14b"}, {"id": "llama3.1:8b"}]})

    async def _chat_completions(self, request: web.Request) -> web.StreamResponse:
        record, early = await self._precheck(request)
        if early is not None:
            return early
        body = record["json"] or {}
        stream = bool(body.get("stream"))
        model = body.get("model", "qwen3:14b")
        if stream:
            return await self._sse_response(request, self._stream_chunks or [self._response_text], model)
        message: dict[str, str] = {"role": "assistant", "content": self._response_text}
        if self._thinking_response is not None:
            reasoning, content = self._thinking_response
            message = {"role": "assistant", "content": content, "reasoning": reasoning}
        return web.json_response(
            {"choices": [{"message": message, "index": 0, "finish_reason": "stop"}], "model": model}
        )

    async def _native_chat(self, request: web.Request) -> web.Response:
        record, early = await self._precheck(request)
        if early is not None:
            return early
        body = record["json"] or {}
        return web.json_response(
            {"message": {"role": "assistant", "content": self._response_text}, "model": body.get("model", "qwen3:14b"), "done": True}
        )

    async def _native_generate(self, request: web.Request) -> web.StreamResponse:
        record, early = await self._precheck(request)
        if early is not None:
            return early
        body = record["json"] or {}
        stream = bool(body.get("stream"))
        model = body.get("model", "qwen3:14b")
        if stream:
            return await self._ollama_native_stream(request, self._stream_chunks or [self._response_text], model)
        # Non-streaming: native API returns translation in "response" field.
        if self._thinking_response is not None:
            reasoning, content = self._thinking_response
            return web.json_response(
                {"model": model, "response": content, "thinking": reasoning, "done": True}
            )
        return web.json_response({"model": model, "response": self._response_text, "done": True})
