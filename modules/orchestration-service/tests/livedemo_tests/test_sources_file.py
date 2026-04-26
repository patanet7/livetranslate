"""Tests for sources/file.py — recorder JSONL replay (B8)."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from livedemo.sources.file import FileSource
from services.pipeline.adapters.source_adapter import CaptionEvent


def _write_jsonl(path: Path, events: list[CaptionEvent], deltas: list[float]) -> None:
    """Write events as recorder JSONL with per-line ts deltas (seconds)."""
    assert len(deltas) == len(events)
    t = 0.0
    lines = []
    for evt, dt in zip(events, deltas, strict=True):
        t += dt
        lines.append(
            json.dumps(
                {
                    "ts": t,
                    "kind": "caption",
                    "payload": {
                        "event_type": evt.event_type,
                        "caption_id": evt.caption_id,
                        "text": evt.text,
                        "translated_text": evt.translated_text,
                        "speaker_name": evt.speaker_name,
                        "speaker_id": evt.speaker_id,
                        "source_lang": evt.source_lang,
                        "target_lang": evt.target_lang,
                        "confidence": evt.confidence,
                        "is_draft": evt.is_draft,
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _evt(i: int) -> CaptionEvent:
    return CaptionEvent(
        event_type="added",
        caption_id=f"cap-{i:03d}",
        text=f"original-{i}",
        translated_text=f"翻译-{i}",
        speaker_name=("Alice" if i % 2 == 0 else "Bob"),
        speaker_id=("SPEAKER_00" if i % 2 == 0 else "SPEAKER_01"),
        source_lang="en",
        target_lang="zh",
    )


@pytest.mark.asyncio
async def test_file_source_yields_all_events_in_order(tmp_path):
    events = [_evt(i) for i in range(6)]
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, events, deltas=[0.0] * 6)  # instant replay
    src = FileSource(jsonl_path=p, replay_speed=0.0)  # 0 = no sleep
    received: list[CaptionEvent] = []
    async for evt in src.stream():
        received.append(evt)
    assert [r.caption_id for r in received] == [e.caption_id for e in events]
    assert [r.text for r in received] == [e.text for e in events]
    assert [r.translated_text for r in received] == [e.translated_text for e in events]
    assert [r.speaker_id for r in received] == [e.speaker_id for e in events]


@pytest.mark.asyncio
async def test_file_source_respects_relative_timing(tmp_path):
    """Two events 200ms apart should produce ≥180ms delay between yields."""
    events = [_evt(0), _evt(1)]
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, events, deltas=[0.0, 0.2])
    src = FileSource(jsonl_path=p, replay_speed=1.0)
    timestamps: list[float] = []
    async for _evt_yielded in src.stream():
        timestamps.append(time.monotonic())
    assert len(timestamps) == 2
    delta = timestamps[1] - timestamps[0]
    assert 0.18 <= delta <= 0.30, f"delta={delta:.3f}s outside expected ~0.2s"


@pytest.mark.asyncio
async def test_file_source_replay_speed_compresses_time(tmp_path):
    """replay_speed=10.0 → 200ms wait becomes ~20ms."""
    events = [_evt(0), _evt(1)]
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, events, deltas=[0.0, 0.2])
    src = FileSource(jsonl_path=p, replay_speed=10.0)
    t0 = time.monotonic()
    async for _ in src.stream():
        pass
    elapsed = time.monotonic() - t0
    assert elapsed < 0.10, f"replay_speed=10x should finish well under 100ms, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_file_source_skips_non_caption_kinds(tmp_path):
    """Recorder also writes ws_send/ws_recv/frame entries; file source ignores those."""
    p = tmp_path / "x.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"ts": 0.0, "kind": "ws_send", "payload": {"x": 1}}),
                json.dumps(
                    {
                        "ts": 0.0,
                        "kind": "caption",
                        "payload": {
                            "event_type": "added",
                            "caption_id": "cap-1",
                            "text": "hi",
                            "translated_text": "hi",
                            "speaker_name": "Alice",
                            "speaker_id": None,
                            "source_lang": "en",
                            "target_lang": "en",
                            "confidence": 1.0,
                            "is_draft": False,
                        },
                    }
                ),
                json.dumps({"ts": 0.0, "kind": "frame", "payload": {"size": 1234}}),
            ]
        )
        + "\n"
    )
    src = FileSource(jsonl_path=p, replay_speed=0.0)
    out = [e async for e in src.stream()]
    assert len(out) == 1
    assert out[0].caption_id == "cap-1"
