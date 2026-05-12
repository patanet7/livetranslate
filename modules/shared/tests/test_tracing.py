"""Tests for the standardized tracing context manager.

Uses capsys + JSON parsing to inspect structlog-rendered events, matching the
existing pattern in test_performance_logging.py. structlog renders to stderr
in JSON format, not via stdlib logging.

The contract:
  - .start event always fires on entry with the passed fields
  - .complete fires on successful exit with duration_ms + any late-added fields
  - .failed fires on exception with duration_ms + error_class + error
  - request_id is auto-generated and stable across .start/.complete/.failed
  - TraceContext.add() can enrich the .complete event after work runs
"""

from __future__ import annotations

import json
import re

import pytest

from livetranslate_common.logging import setup_logging
from livetranslate_common.tracing import (
    TraceContext,
    base_url_host,
    make_request_id,
    trace_request,
)


def _captured_events(capsys: pytest.CaptureFixture[str]) -> list[dict]:
    """Parse JSON lines from stderr; non-JSON lines silently dropped."""
    captured = capsys.readouterr()
    out: list[dict] = []
    for line in captured.err.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _setup() -> None:
    """Re-bind structlog inside the test (after pytest installs capsys)."""
    setup_logging(service_name="test-tracing", log_format="json")


class TestRequestId:
    def test_make_request_id_is_8_chars(self) -> None:
        rid = make_request_id()
        assert len(rid) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", rid)

    def test_make_request_id_uniqueness(self) -> None:
        ids = {make_request_id() for _ in range(1000)}
        assert len(ids) == 1000


class TestBaseUrlHost:
    def test_strips_path(self) -> None:
        assert base_url_host("http://x:8089/v1") == "x:8089"

    def test_strips_trailing_slash(self) -> None:
        assert base_url_host("http://x:8089/v1/") == "x:8089"

    def test_no_scheme(self) -> None:
        assert base_url_host("x:8089/v1") == "x:8089"

    def test_empty(self) -> None:
        assert base_url_host("") == ""

    def test_https(self) -> None:
        assert base_url_host("https://api.openai.com/v1") == "api.openai.com"


class TestTraceRequestSuccess:
    def test_emits_start_then_complete(self, capsys) -> None:
        _setup()
        with trace_request("whisper", "request", model="m"):
            pass
        events = _captured_events(capsys)
        names = [e.get("event") for e in events]
        assert "whisper.request.start" in names
        assert "whisper.request.complete" in names

    def test_complete_has_duration_ms(self, capsys) -> None:
        _setup()
        with trace_request("llm", "request"):
            pass
        events = _captured_events(capsys)
        completes = [e for e in events if e.get("event") == "llm.request.complete"]
        assert len(completes) == 1
        assert isinstance(completes[0]["duration_ms"], float)
        assert completes[0]["duration_ms"] >= 0.0

    def test_request_id_auto_generated_and_stable(self, capsys) -> None:
        _setup()
        with trace_request("whisper", "request") as t:
            assert len(t.request_id) == 8
            captured_rid = t.request_id
        events = _captured_events(capsys)
        starts = [e for e in events if e.get("event") == "whisper.request.start"]
        completes = [e for e in events if e.get("event") == "whisper.request.complete"]
        assert starts[0]["request_id"] == captured_rid
        assert completes[0]["request_id"] == captured_rid  # stable across phases

    def test_request_id_explicit(self, capsys) -> None:
        _setup()
        with trace_request("whisper", "request", request_id="cafebabe") as t:
            assert t.request_id == "cafebabe"
        events = _captured_events(capsys)
        for e in events:
            if "request" in e.get("event", ""):
                assert e["request_id"] == "cafebabe"

    def test_late_enrichment_via_add(self, capsys) -> None:
        _setup()
        with trace_request("whisper", "request", model="m") as t:
            t.add(language_detected="zh", text_chars=42)
        events = _captured_events(capsys)
        completes = [e for e in events if e.get("event") == "whisper.request.complete"]
        assert completes[0]["language_detected"] == "zh"
        assert completes[0]["text_chars"] == 42
        # start-time fields preserved
        assert completes[0]["model"] == "m"

    def test_start_fields_appear_on_start_event(self, capsys) -> None:
        _setup()
        with trace_request(
            "whisper", "request", model="m", engine="openai_compatible"
        ):
            pass
        events = _captured_events(capsys)
        starts = [e for e in events if e.get("event") == "whisper.request.start"]
        assert starts[0]["model"] == "m"
        assert starts[0]["engine"] == "openai_compatible"


class TestTraceRequestFailure:
    def test_failed_event_emitted(self, capsys) -> None:
        _setup()
        with pytest.raises(ValueError, match="boom"):
            with trace_request("whisper", "request", model="m"):
                raise ValueError("boom")
        events = _captured_events(capsys)
        names = [e.get("event") for e in events]
        assert "whisper.request.failed" in names
        assert "whisper.request.complete" not in names

    def test_failed_captures_error_class_and_msg(self, capsys) -> None:
        _setup()
        with pytest.raises(RuntimeError):
            with trace_request("llm", "request"):
                raise RuntimeError("kaboom")
        events = _captured_events(capsys)
        faileds = [e for e in events if e.get("event") == "llm.request.failed"]
        assert faileds[0]["error_class"] == "RuntimeError"
        assert faileds[0]["error"] == "kaboom"

    def test_failed_has_duration_ms(self, capsys) -> None:
        _setup()
        with pytest.raises(ValueError):
            with trace_request("whisper", "request"):
                raise ValueError("x")
        events = _captured_events(capsys)
        faileds = [e for e in events if e.get("event") == "whisper.request.failed"]
        assert "duration_ms" in faileds[0]

    def test_failure_preserves_start_fields(self, capsys) -> None:
        _setup()
        """Fields passed to trace_request must survive onto .failed."""
        with pytest.raises(ValueError):
            with trace_request(
                "whisper", "request", connection_id="c1", model="m"
            ) as t:
                t.add(audio_duration_s=6.0)
                raise ValueError("boom")
        events = _captured_events(capsys)
        faileds = [e for e in events if e.get("event") == "whisper.request.failed"]
        assert faileds[0]["connection_id"] == "c1"
        assert faileds[0]["model"] == "m"
        assert faileds[0]["audio_duration_s"] == 6.0


class TestTraceContext:
    def test_snapshot_returns_copy(self) -> None:
        ctx = TraceContext({"a": 1, "b": 2})
        snap = ctx.snapshot()
        snap["a"] = 999
        assert ctx.snapshot()["a"] == 1

    def test_add_merges(self) -> None:
        ctx = TraceContext({"a": 1})
        ctx.add(b=2, c=3)
        assert ctx.snapshot() == {"a": 1, "b": 2, "c": 3}

    def test_add_overwrites(self) -> None:
        ctx = TraceContext({"x": "first"})
        ctx.add(x="second")
        assert ctx.snapshot()["x"] == "second"
