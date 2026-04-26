"""Tests for WSRecorder — every kind/payload captured to JSONL (B7)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from livedemo.recorder import WSRecorder


def test_recorder_writes_each_call_as_a_jsonl_line(tmp_path):
    rec = WSRecorder(run_dir=tmp_path)
    rec.record("caption", {"caption_id": "x"})
    rec.record("ws_send", {"bytes": 320})
    rec.record("frame", {"size": 12345})
    rec.close()

    lines = (tmp_path / "messages.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
    parsed = [json.loads(l) for l in lines]
    assert [p["kind"] for p in parsed] == ["caption", "ws_send", "frame"]
    assert parsed[0]["payload"]["caption_id"] == "x"
    # Every line has a numeric ts and ISO-8601 wall.
    for p in parsed:
        assert isinstance(p["ts"], (int, float))
        assert "T" in p["wall"]


def test_recorder_disabled_writes_nothing(tmp_path):
    rec = WSRecorder(run_dir=tmp_path, enabled=False)
    rec.record("caption", {"caption_id": "x"})
    rec.close()
    p = tmp_path / "messages.jsonl"
    # File may or may not exist — either way must contain no caption entries.
    if p.exists():
        assert p.read_text() == ""


def test_recorder_can_be_used_as_context_manager(tmp_path):
    with WSRecorder(run_dir=tmp_path) as rec:
        rec.record("caption", {"caption_id": "y"})
    p = tmp_path / "messages.jsonl"
    assert p.exists() and p.stat().st_size > 0
