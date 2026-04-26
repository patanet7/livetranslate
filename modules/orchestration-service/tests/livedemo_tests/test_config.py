"""Tests for LiveDemoConfig — env > YAML > defaults precedence (B10)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from livedemo.config import LiveDemoConfig


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip any LIVEDEMO_* env vars and unrelated dotenv loaders."""
    for key in list(monkeypatch.delenv.__self__._setitem.keys() if False else []):
        pass
    # Defensive: clear any LIVEDEMO_ overrides set in the shell that runs tests.
    import os

    for key in [k for k in os.environ if k.startswith("LIVEDEMO_")]:
        monkeypatch.delenv(key, raising=False)


def _yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_defaults_apply_when_no_overrides():
    cfg = LiveDemoConfig(
        meeting_url="https://meet.google.com/aaa-bbbb-ccc",
        source="file",
        replay_jsonl=Path("/tmp/whatever.jsonl"),
    )
    assert cfg.canvas_ws_port == 7081
    assert cfg.frame_fps == 10
    assert cfg.sink == "canvas"
    assert cfg.source_language == "auto"
    assert cfg.target_language == "en"
    assert cfg.bot_show_diarization_ids is False
    assert cfg.fireflies_replay_speed == 1.0
    assert cfg.record_messages is True


def test_yaml_overrides_defaults(tmp_path):
    yaml_path = _yaml(
        tmp_path / "demo.yaml",
        {
            "meeting_url": "https://meet.google.com/aaa-bbbb-ccc",
            "source": "fireflies",
            "fireflies_meeting_id": "FF1",
            "fireflies_replay_speed": 2.5,
            "target_language": "zh",
            "frame_fps": 20,
        },
    )
    cfg = LiveDemoConfig.from_yaml(yaml_path)
    assert str(cfg.meeting_url).startswith("https://meet.google.com/")
    assert cfg.source == "fireflies"
    assert cfg.fireflies_meeting_id == "FF1"
    assert cfg.fireflies_replay_speed == 2.5
    assert cfg.target_language == "zh"
    assert cfg.frame_fps == 20


def test_env_overrides_yaml(tmp_path, monkeypatch):
    yaml_path = _yaml(
        tmp_path / "demo.yaml",
        {
            "meeting_url": "https://meet.google.com/aaa-bbbb-ccc",
            "source": "fireflies",
            "fireflies_meeting_id": "FF1",
            "target_language": "zh",
        },
    )
    monkeypatch.setenv("LIVEDEMO_TARGET_LANGUAGE", "ja")
    monkeypatch.setenv("LIVEDEMO_FIREFLIES_MEETING_ID", "FF_FROM_ENV")
    cfg = LiveDemoConfig.from_yaml(yaml_path)
    assert cfg.target_language == "ja"
    assert cfg.fireflies_meeting_id == "FF_FROM_ENV"


def test_snapshot_yaml_is_resolved(tmp_path, monkeypatch):
    """B10 — snapshot must reflect fully resolved config, post-merge."""
    yaml_path = _yaml(
        tmp_path / "demo.yaml",
        {
            "meeting_url": "https://meet.google.com/aaa-bbbb-ccc",
            "source": "file",
            "replay_jsonl": "/tmp/x.jsonl",
            "target_language": "zh",
        },
    )
    monkeypatch.setenv("LIVEDEMO_TARGET_LANGUAGE", "ja")
    cfg = LiveDemoConfig.from_yaml(yaml_path)

    snap_path = tmp_path / "snap.yaml"
    cfg.write_snapshot(snap_path)

    snap = yaml.safe_load(snap_path.read_text())
    assert snap["target_language"] == "ja"  # env wins over yaml
    assert snap["frame_fps"] == 10  # default
    assert snap["source"] == "file"


def test_validates_source_specific_required_fields():
    """source=file requires replay_jsonl."""
    with pytest.raises(ValueError):
        LiveDemoConfig(
            meeting_url="https://meet.google.com/aaa-bbbb-ccc",
            source="file",  # missing replay_jsonl
        )


def test_validates_fireflies_requires_meeting_id():
    with pytest.raises(ValueError):
        LiveDemoConfig(
            meeting_url="https://meet.google.com/aaa-bbbb-ccc",
            source="fireflies",  # missing fireflies_meeting_id
        )
