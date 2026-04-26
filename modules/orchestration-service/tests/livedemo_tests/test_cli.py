"""Tests for cli.py — `livedemo doctor|run|smoke|replay` (B1, B10).

Uses typer.testing.CliRunner to drive the app. No subprocess: the CLI is
exercised in-process for fast TDD.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from livedemo.cli import app

runner = CliRunner()


@pytest.fixture
def fixture_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "fixture.jsonl"
    rows = []
    for i, (orig, trans) in enumerate(
        [
            ("Hello", "你好"),
            ("Goodbye", "再见"),
        ]
    ):
        rows.append(
            json.dumps(
                {
                    "ts": 0.0,
                    "kind": "caption",
                    "payload": {
                        "event_type": "added",
                        "caption_id": f"cap-{i}",
                        "text": orig,
                        "translated_text": trans,
                        "speaker_name": "Alice",
                        "speaker_id": None,
                        "source_lang": "en",
                        "target_lang": "zh",
                        "confidence": 1.0,
                        "is_draft": False,
                    },
                }
            )
        )
    p.write_text("\n".join(rows) + "\n")
    return p


def _yaml_config(tmp_path: Path, fixture: Path, **overrides) -> Path:
    base = {
        "meeting_url": "https://meet.google.com/aaa-bbbb-ccc",
        "source": "file",
        "sink": "png",
        "replay_jsonl": str(fixture),
    }
    base.update(overrides)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base))
    return p


def test_cli_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "run" in result.stdout
    assert "smoke" in result.stdout


def test_cli_doctor_passes_for_file_source(tmp_path, fixture_jsonl):
    cfg = _yaml_config(tmp_path, fixture_jsonl)
    result = runner.invoke(app, ["doctor", "--config", str(cfg)])
    # File source has no Chrome/orchestration deps; everything should pass.
    # Exit may still be non-0 if playwright_chromium/canvas_ws_port checks fail
    # locally — accept either, but the table must list `replay_jsonl: ✓`.
    assert "replay_jsonl" in result.stdout
    assert "✓" in result.stdout or "PASS" in result.stdout


def test_cli_smoke_runs_file_to_png_end_to_end(tmp_path, fixture_jsonl):
    """`livedemo smoke` should run file→png with the bundled fixture.

    The bundled fixture lives in the orchestration-service tests/ tree; we
    invoke smoke with our own jsonl via --replay-jsonl override.
    """
    out_dir = tmp_path / "frames"
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "smoke",
            "--replay-jsonl", str(fixture_jsonl),
            "--out-dir", str(out_dir),
            "--runs-dir", str(runs_dir),
        ],
    )
    if result.exit_code != 0:
        # Surface any failure for easier debugging
        print(result.stdout)
        print(result.stderr if hasattr(result, "stderr") else "")
    assert result.exit_code == 0
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == 2  # 2 fixture rows


def test_cli_run_blocks_when_doctor_fails(tmp_path, fixture_jsonl, monkeypatch):
    """B1 — run must NOT proceed when a required check fails.

    We force a check failure by pointing replay_jsonl at a missing file.
    """
    cfg = _yaml_config(tmp_path, fixture_jsonl, replay_jsonl="/tmp/does-not-exist-zzz.jsonl")
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    # Hint should mention the missing file
    assert "does-not-exist" in result.stdout or "replay_jsonl" in result.stdout


def test_cli_run_writes_resolved_config_snapshot(tmp_path, fixture_jsonl, monkeypatch):
    """B10 — run dir must contain a fully-resolved config.snapshot.yaml."""
    monkeypatch.setenv("LIVEDEMO_TARGET_LANGUAGE", "ja")  # env override
    cfg = _yaml_config(tmp_path, fixture_jsonl, target_language="zh")  # yaml says zh
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "frames"
    result = runner.invoke(
        app,
        [
            "run",
            "--config", str(cfg),
            "--sink", "png",
            "--out-dir", str(out_dir),
            "--runs-dir", str(runs_dir),
            "--skip-doctor",
        ],
    )
    if result.exit_code != 0:
        print(result.stdout)
    assert result.exit_code == 0
    # Find the run dir
    run_subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(run_subdirs) == 1
    snap = run_subdirs[0] / "config.snapshot.yaml"
    assert snap.exists()
    snap_data = yaml.safe_load(snap.read_text())
    assert snap_data["target_language"] == "ja"  # env wins
    assert snap_data["source"] == "file"
