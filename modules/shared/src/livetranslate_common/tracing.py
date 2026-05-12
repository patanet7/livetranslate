"""Standardized request-lifecycle tracing across every LiveTranslate service.

Every observable operation in the system (LLM call, Whisper call, translation
pipeline stage, VAD frame, session restart) emits structured events through
``trace_request``. The event vocabulary is fixed:

    {domain}.{action}.start    — emitted on entry
    {domain}.{action}.complete — emitted on successful exit (with duration_ms)
    {domain}.{action}.failed   — emitted on exception (with duration_ms + error)

Standard field names (use these or extend — never invent new spellings for
the same concept):

    Required on every event:
        request_id     — short uuid hex (8 chars). Auto-injected if not provided.
        duration_ms    — float, 2-decimal precision. Auto-set on complete/failed.

    Whisper domain ("whisper.*"):
        connection_id      — str | None — DB row id of the WhisperConnection
        engine             — str — "openai_compatible" / "mlx_local" / etc.
        model              — str — full HF repo path
        audio_duration_s   — float — input audio length
        language_hint      — str | None — BCP-47 code passed to decoder
        language_detected  — str — what Whisper returned (added on complete)
        text_chars         — int — output text length (added on complete)
        segment_count      — int — number of segments returned
        no_speech_prob     — float | None — max across segments (worst case)
        confidence         — float — overall confidence

    LLM domain ("llm.*"):
        connection_id    — str | None
        engine           — str — "ollama" / "openai" / "openai_compatible" / etc.
        model            — str
        prompt_chars     — int — pre-call prompt length
        response_chars   — int — added on complete
        stream           — bool — whether this was a streaming call

    Resolver domain ("{domain}.connection.resolved"):
        purpose          — str
        source           — Literal["step1","step2","step3","step4","step5"]
        connection_id    — str | None
        engine           — str
        base_url_host    — str — base URL host:port (NO api_key)

Why a context manager and not a decorator? Two reasons:
  1. Late enrichment — after a Whisper response parses we need to add
     language_detected / text_chars / segment_count. A decorator can't see
     into the function's return value generically.
  2. Failure capture — exception class and message belong on the .failed
     event; the context manager captures both without polluting the caller.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

from livetranslate_common.logging.setup import get_logger


def make_request_id() -> str:
    """8-char uuid hex prefix — short enough for log readability, unique enough."""
    return uuid.uuid4().hex[:8]


class TraceContext:
    """Mutable trace fields. The context manager passes one in as `t`; callers
    use ``t.add(language_detected=...)`` to enrich the .complete event after
    the response parses."""

    __slots__ = ("_fields",)

    def __init__(self, fields: dict[str, Any]) -> None:
        self._fields = fields

    def add(self, **fields: Any) -> None:
        self._fields.update(fields)

    @property
    def request_id(self) -> str:
        return str(self._fields.get("request_id", ""))

    def snapshot(self) -> dict[str, Any]:
        return dict(self._fields)


@contextmanager
def trace_request(
    domain: str,
    action: str,
    *,
    request_id: str | None = None,
    logger: structlog.stdlib.BoundLogger | None = None,
    **fields: Any,
) -> Iterator[TraceContext]:
    """Emit {domain}.{action}.start / .complete / .failed with duration_ms.

    Usage:
        with trace_request("whisper", "request",
                           connection_id=conn.connection_id,
                           engine=conn.engine,
                           model=conn.model,
                           audio_duration_s=audio.size / 16000,
                           language_hint=conn.language_hint) as t:
            response = await self._client.post(...)
            result = parse(response)
            t.add(language_detected=result.language,
                  text_chars=len(result.text),
                  segment_count=len(result.segments))

    On exception, emits {domain}.{action}.failed with error_class + error;
    the exception propagates after logging.
    """
    log = logger or get_logger()
    rid = request_id or make_request_id()
    payload: dict[str, Any] = {"request_id": rid, **fields}

    log.info(f"{domain}.{action}.start", **payload)

    start = time.perf_counter()
    trace_fields = dict(payload)
    ctx = TraceContext(trace_fields)
    try:
        yield ctx
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        log.warning(
            f"{domain}.{action}.failed",
            duration_ms=elapsed_ms,
            error_class=exc.__class__.__name__,
            error=str(exc),
            **ctx.snapshot(),
        )
        raise
    else:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        log.info(
            f"{domain}.{action}.complete",
            duration_ms=elapsed_ms,
            **ctx.snapshot(),
        )


def base_url_host(url: str) -> str:
    """Extract host:port from a URL for safe logging (drops path + query).

    Logging the full base_url is fine — it contains no secrets (api_key is
    a separate field) — but stripping the path/query keeps the event compact
    and stable across endpoints (e.g., http://x:8089/v1 and http://x:8089/v1/
    log identically).
    """
    if not url:
        return ""
    # parse just enough: scheme://host[:port][/path]
    after_scheme = url.split("://", 1)[-1]
    return after_scheme.split("/", 1)[0]
