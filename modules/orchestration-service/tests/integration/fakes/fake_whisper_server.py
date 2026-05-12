"""In-process OpenAI-compatible Whisper transcription fake.

Routes:
    GET  /health                           — health probe
    GET  /v1/models                        — model list
    POST /v1/audio/transcriptions          — multipart audio upload, verbose_json

The OpenAI Whisper API expects a multipart form with a `file` part containing
audio (WAV/MP3/etc.) and optional fields: `model`, `language`, `prompt`,
`response_format`, `temperature`. Returns text + segments + metadata.

Mirrors the wire shape of vllm-mlx, faster-whisper-server, and openai.com.
"""

from __future__ import annotations

from aiohttp import web

from ._base import FakeLLMServerBase


class FakeWhisperServer(FakeLLMServerBase):
    """Multipart Whisper fake — accepts /v1/audio/transcriptions uploads.

    Knobs (from base):
      - set_response_text(text)       — controls the returned transcript
      - set_segments(list[dict])      — verbose_json segments (overrides default)
      - fail_with_status(status)      — every request returns this status
      - fail_n_times(n, status=500)   — next n requests fail, rest succeed
      - require_api_key("dummy")      — enforce Authorization: Bearer dummy
      - delay_seconds(s)              — pre-response sleep (for timeout tests)
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        super().__init__(host=host)
        self._detected_language: str = "en"
        self._segments: list[dict] | None = None

    def set_detected_language(self, lang: str) -> None:
        self._detected_language = lang

    def set_segments(self, segments: list[dict]) -> None:
        """Override the default verbose_json segments. Each segment is a dict
        with keys matching the OpenAI Whisper API: id, start, end, text,
        avg_logprob, no_speech_prob, compression_ratio."""
        self._segments = list(segments)

    def _install_routes(self, app: web.Application) -> None:
        app.router.add_get("/health", self._health)
        app.router.add_get("/v1/models", self._models)
        app.router.add_post("/v1/audio/transcriptions", self._transcriptions)

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _models(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(
            {
                "data": [
                    {"id": "mlx-community/whisper-large-v3-turbo"},
                    {"id": "mlx-community/whisper-medium"},
                ]
            }
        )

    async def _transcriptions(self, request: web.Request) -> web.Response:
        # Custom record path — multipart bodies aren't JSON-decodable.
        # Capture both raw body and parsed form fields.
        body = await request.read()
        record = {
            "method": request.method,
            "path": request.path,
            "headers": {k.lower(): v for k, v in request.headers.items()},
            "json": None,
            "query": dict(request.query),
            "body": body,
            "form_fields": {},
            "audio_size": 0,
        }

        # Re-parse multipart from the raw body to extract form fields.
        # Doing this inside the handler avoids consuming the stream twice.
        try:
            content_type = request.headers.get("Content-Type", "")
            if "multipart" in content_type:
                # Re-create a request-like reader from the captured body.
                form_fields, audio_size = _parse_multipart(body, content_type)
                record["form_fields"] = form_fields
                record["audio_size"] = audio_size
        except Exception:  # pragma: no cover — best-effort capture
            pass

        self.recorded_requests.append(record)

        await self._maybe_delay()
        unauthorized = self._maybe_unauthorized(record)
        if unauthorized is not None:
            return unauthorized
        failure = self._maybe_fail()
        if failure is not None:
            return failure

        response_format = record["form_fields"].get("response_format", "json")
        language = record["form_fields"].get("language") or self._detected_language

        text = self._response_text
        if self._segments is not None:
            segments = list(self._segments)
        else:
            segments = [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.5,
                    "text": text,
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.01,
                    "compression_ratio": 1.4,
                }
            ]

        if response_format in ("verbose_json", "json"):
            payload: dict = {"text": text, "language": language}
            if response_format == "verbose_json":
                payload["segments"] = segments
                payload["duration"] = (
                    max((s.get("end", 0.0) for s in segments), default=0.0)
                )
            return web.json_response(payload)

        # text/srt/vtt — caller asked for a plain string body
        return web.Response(text=text, content_type="text/plain")


def _parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], int]:
    """Lightweight multipart parser — extracts text form fields and audio size.

    We don't decode the audio (verifying the bytes hit the wire is enough for
    tests). Returns (form_fields, audio_byte_count).
    """
    import email
    import re
    from email.policy import default

    # Build a faux MIME message so we can use stdlib email parser
    boundary_match = re.search(r"boundary=([^;]+)", content_type)
    if not boundary_match:
        return {}, 0
    boundary = boundary_match.group(1).strip().strip('"')

    parts = body.split(("--" + boundary).encode())
    form_fields: dict[str, str] = {}
    audio_size = 0
    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        try:
            header_block, _, value = part.partition(b"\r\n\r\n")
        except ValueError:
            continue
        if not header_block:
            continue
        headers = {}
        for hline in header_block.split(b"\r\n"):
            if b":" in hline:
                k, _, v = hline.partition(b":")
                headers[k.decode().lower().strip()] = v.decode().strip()
        disposition = headers.get("content-disposition", "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue
        name = name_match.group(1)
        if 'filename="' in disposition or "file" == name:
            audio_size = len(value)
        else:
            form_fields[name] = value.decode("utf-8", errors="replace").rstrip("\r\n")
    return form_fields, audio_size
