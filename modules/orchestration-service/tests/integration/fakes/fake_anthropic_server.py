"""In-process Anthropic-shaped HTTP fake.

Routes:
    POST /v1/messages       — Messages API. Validates x-api-key + anthropic-version.

SSE streaming follows Anthropic's `event: …\\ndata: …\\n\\n` format with
`message_start`, `content_block_delta`, and `message_stop` events.
"""

from __future__ import annotations

import json as _json

from aiohttp import web

from ._base import FakeLLMServerBase


class FakeAnthropicServer(FakeLLMServerBase):
    def __init__(self, host: str = "127.0.0.1") -> None:
        super().__init__(host=host)
        # Anthropic uses the bare API key in the x-api-key header (no "Bearer" prefix)
        self._api_key_header = "x-api-key"
        self._api_key_prefix = ""

    def _install_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/messages", self._messages)

    async def _messages(self, request: web.Request) -> web.StreamResponse:
        record, early = await self._precheck(request)
        if early is not None:
            return early
        # Validate anthropic-version header for realism
        if not record["headers"].get("anthropic-version"):
            return web.json_response({"error": "missing anthropic-version"}, status=400)
        body = record["json"] or {}
        stream = bool(body.get("stream"))
        model = body.get("model", "claude-3-5-sonnet")
        if stream:
            return await self._anthropic_sse_stream(
                request, self._stream_chunks or [self._response_text], model
            )
        return web.json_response(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": self._response_text}],
                "stop_reason": "end_turn",
            }
        )

    async def _anthropic_sse_stream(
        self, request: web.Request, chunks: list[str], model: str
    ) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        await resp.write(
            b"event: message_start\ndata: "
            + _json.dumps({"type": "message_start", "message": {"id": "msg_test", "model": model}}).encode()
            + b"\n\n"
        )
        for chunk in chunks:
            payload = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk},
            }
            await resp.write(b"event: content_block_delta\ndata: " + _json.dumps(payload).encode() + b"\n\n")
        await resp.write(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
        await resp.write_eof()
        return resp
