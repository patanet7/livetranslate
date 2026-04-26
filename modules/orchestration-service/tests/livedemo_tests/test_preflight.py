"""Tests for preflight — registry + per-source dependency matrix.

These tests validate the framework, not real-service health. Real-service health
is exercised by integration tests (mic source, fireflies source) which call
through the actual checks against running stacks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from livedemo.config import LiveDemoConfig
from livedemo.preflight import CheckResult, check_all, register_check


def _make_cfg(**overrides) -> LiveDemoConfig:
    base = dict(
        meeting_url="https://meet.google.com/aaa-bbbb-ccc",
        source="file",
        replay_jsonl=Path("/tmp/x.jsonl"),
    )
    base.update(overrides)
    return LiveDemoConfig(**base)


def test_check_result_is_dataclass_with_required_fields():
    r = CheckResult(name="x", ok=True, hint=None, detail="ok")
    assert r.name == "x"
    assert r.ok is True
    assert r.hint is None
    assert r.detail == "ok"


def test_check_all_runs_only_relevant_checks_for_source(monkeypatch):
    """B1 dependency matrix: file source skips orchestration_ws + mic_device."""
    cfg = _make_cfg()
    cfg.replay_jsonl.touch(exist_ok=True)
    results = check_all(cfg)
    names = {r.name for r in results}
    # Always-on checks
    assert "playwright_chromium" in names or "chrome_profile" in names
    # File-source-specific
    assert "replay_jsonl" in names
    # Mic-specific check should NOT run for file source
    assert "orchestration_ws" not in names
    assert "mic_device" not in names


def test_check_all_runs_orchestration_for_mic_source():
    cfg = _make_cfg(source="mic", replay_jsonl=None)
    results = check_all(cfg)
    names = {r.name for r in results}
    assert "orchestration_ws" in names
    assert "transcription_service" in names


def test_check_all_runs_fireflies_check_for_fireflies_source():
    cfg = _make_cfg(source="fireflies", fireflies_meeting_id="FF1", replay_jsonl=None)
    results = check_all(cfg)
    names = {r.name for r in results}
    assert "fireflies_api" in names


def test_register_custom_check_runs(monkeypatch):
    """Custom checks can be added to the registry."""
    called = []

    def _custom(cfg):
        called.append(cfg.source)
        return CheckResult(name="custom_demo", ok=True)

    register_check("custom_demo", _custom, sources={"file"})
    cfg = _make_cfg()
    cfg.replay_jsonl.touch(exist_ok=True)
    results = check_all(cfg)
    names = {r.name for r in results}
    assert "custom_demo" in names
    assert called == ["file"]


def test_replay_jsonl_check_fails_when_missing(tmp_path):
    cfg = _make_cfg(replay_jsonl=tmp_path / "does-not-exist.jsonl")
    results = check_all(cfg)
    by_name = {r.name: r for r in results}
    assert "replay_jsonl" in by_name
    assert by_name["replay_jsonl"].ok is False
    assert by_name["replay_jsonl"].hint  # actionable hint set


def test_replay_jsonl_check_passes_when_present(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"kind":"caption","payload":{}}\n')
    cfg = _make_cfg(replay_jsonl=p)
    results = check_all(cfg)
    by_name = {r.name: r for r in results}
    assert by_name["replay_jsonl"].ok is True
