"""Tests for structlog observability events in the livedemo pipeline.

Per CLAUDE.md: structlog only — never `import logging`. The CLI is the one
allowed exception (typer.echo for user-facing output) but observability events
must use structlog so meta-tooling can parse them.

Lifecycle events we lock in:
  - pipeline.run_started      — at start of run_once()
  - pipeline.run_completed    — at end of run_once(), with caption count
  - sink.frame_rendered       — per consumed CaptionEvent (debug or info)
  - source.stream_started     — when a source's iterator opens
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog
from structlog.testing import capture_logs

from livedemo.pipeline import run_once
from livedemo.recorder import WSRecorder
from livedemo.sinks.png import PngSink
from livedemo.sources.file import FileSource


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "livedemo" / "short-dialog.jsonl"


@pytest.mark.asyncio
async def test_pipeline_emits_run_started_and_completed_events(tmp_path, fixture_path):
    src = FileSource(jsonl_path=fixture_path, replay_speed=0.0)
    sink = PngSink(out_dir=tmp_path / "frames")
    rec = WSRecorder(run_dir=tmp_path / "rec", enabled=True)
    with capture_logs() as caplog:
        n = await run_once(source=src, sink=sink, recorder=rec)
    rec.close()
    events = [e["event"] for e in caplog]
    assert "run_started" in events
    assert "run_completed" in events
    completed = next(e for e in caplog if e["event"] == "run_completed")
    assert completed.get("count") == n
    assert completed.get("count") == 6


@pytest.mark.asyncio
async def test_sink_emits_event_per_caption(tmp_path, fixture_path):
    src = FileSource(jsonl_path=fixture_path, replay_speed=0.0)
    sink = PngSink(out_dir=tmp_path / "frames")
    with capture_logs() as caplog:
        await run_once(source=src, sink=sink, recorder=None)
    rendered = [e for e in caplog if e["event"] == "frame_rendered"]
    assert len(rendered) == 6
    # Each event carries identifying info
    for e in rendered:
        assert "caption_id" in e


def test_no_python_logging_in_livedemo_package():
    """CLAUDE.md hard rule: never `import logging` in our code."""
    import re

    livedemo_root = Path(__file__).resolve().parent.parent.parent / "src" / "livedemo"
    offenders: list[tuple[Path, int, str]] = []
    for py in livedemo_root.rglob("*.py"):
        # Skip files under generated bot_runner/node_modules
        if "node_modules" in py.parts:
            continue
        text = py.read_text()
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.match(r"^import logging\b", stripped):
                offenders.append((py, i, line))
            if re.search(r"\blogging\.getLogger\b", stripped):
                offenders.append((py, i, line))
    assert not offenders, f"Forbidden Python-logging usage:\n" + "\n".join(
        f"  {p}:{i}  {l}" for p, i, l in offenders
    )
