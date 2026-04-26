"""Behavioral smoke E2E — file → recorder → png sink, fully offline.

This is the load-bearing CI smoke test. It runs the real pipeline glue against
real components (VirtualWebcamManager, real PIL, real PNG encoder, real disk
I/O). No mocks, per CLAUDE.md.

Validates B7 (every event recorded), B8 (round-trip is byte-exact), and the
overall happy path for a Phase-6 `livedemo smoke` invocation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from livedemo.pipeline import run_once
from livedemo.recorder import WSRecorder
from livedemo.sinks.png import PngSink
from livedemo.sources.file import FileSource
from services.pipeline.adapters.source_adapter import CaptionEvent


@pytest.fixture
def fixture_jsonl(tmp_path: Path) -> Path:
    """Six-caption deterministic dialog, instant timing for fast tests."""
    p = tmp_path / "fixture.jsonl"
    captions = [
        ("Alice", "SPEAKER_00", "en", "zh", "Hello everyone, welcome to the meeting.", "大家好，欢迎参加会议。"),
        ("Bob",   "SPEAKER_01", "zh", "en", "谢谢，很高兴见到你们。", "Thank you, nice to meet you all."),
        ("Alice", "SPEAKER_00", "en", "zh", "Let's go through the agenda for today.", "让我们看一下今天的议程。"),
        ("Bob",   "SPEAKER_01", "zh", "en", "好的，我们从产品更新开始。", "Sure, let's start with the product update."),
        ("Alice", "SPEAKER_00", "en", "zh", "The new build is ready for testing.", "新版本已经准备好进行测试了。"),
        ("Bob",   "SPEAKER_01", "zh", "en", "太好了，我会发送测试链接。", "Great, I'll send the test link."),
    ]
    lines = []
    for i, (name, sid, src, tgt, orig, trans) in enumerate(captions):
        lines.append(
            json.dumps(
                {
                    "ts": float(i) * 0.0,  # instant
                    "kind": "caption",
                    "payload": {
                        "event_type": "added",
                        "caption_id": f"cap-{i:03d}",
                        "text": orig,
                        "translated_text": trans,
                        "speaker_name": name,
                        "speaker_id": sid,
                        "source_lang": src,
                        "target_lang": tgt,
                        "confidence": 0.92,
                        "is_draft": False,
                    },
                }
            )
        )
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.mark.asyncio
async def test_smoke_file_to_png_writes_n_frames(tmp_path: Path, fixture_jsonl: Path):
    """B7 surrogate: every yielded event produces exactly one frame."""
    out_dir = tmp_path / "frames"
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    src = FileSource(jsonl_path=fixture_jsonl, replay_speed=0.0)
    sink = PngSink(out_dir=out_dir)
    recorder = WSRecorder(run_dir=rec_dir)

    n = await run_once(source=src, sink=sink, recorder=recorder)
    recorder.close()

    pngs = sorted(out_dir.glob("*.png"))
    assert n == 6
    assert len(pngs) == 6
    # Each PNG is a real, non-empty file
    assert all(p.stat().st_size > 1024 for p in pngs)


@pytest.mark.asyncio
async def test_smoke_recorder_captures_every_caption(tmp_path: Path, fixture_jsonl: Path):
    """B7 — recorder JSONL line count >= caption count."""
    out_dir = tmp_path / "frames"
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    src = FileSource(jsonl_path=fixture_jsonl, replay_speed=0.0)
    sink = PngSink(out_dir=out_dir)
    recorder = WSRecorder(run_dir=rec_dir)

    await run_once(source=src, sink=sink, recorder=recorder)
    recorder.close()

    lines = (rec_dir / "messages.jsonl").read_text().strip().split("\n")
    parsed = [json.loads(l) for l in lines]
    captions = [p for p in parsed if p["kind"] == "caption"]
    assert len(captions) == 6


@pytest.mark.asyncio
async def test_smoke_record_replay_round_trip_is_lossless(tmp_path: Path, fixture_jsonl: Path):
    """B8 — record run #1, replay through file source, record run #2; payloads identical.

    Wall-clock timestamps differ between runs (expected — they're real time). The
    invariant is the *payload* sequence: same caption_id, same text, same order.
    """
    rec1 = tmp_path / "rec1"
    rec2 = tmp_path / "rec2"
    rec1.mkdir(); rec2.mkdir()

    # Pass 1: replay original fixture, record to rec1
    src1 = FileSource(jsonl_path=fixture_jsonl, replay_speed=0.0)
    sink1 = PngSink(out_dir=tmp_path / "f1")
    r1 = WSRecorder(run_dir=rec1)
    await run_once(source=src1, sink=sink1, recorder=r1)
    r1.close()

    # Pass 2: replay rec1's recording, record to rec2
    src2 = FileSource(jsonl_path=rec1 / "messages.jsonl", replay_speed=0.0)
    sink2 = PngSink(out_dir=tmp_path / "f2")
    r2 = WSRecorder(run_dir=rec2)
    await run_once(source=src2, sink=sink2, recorder=r2)
    r2.close()

    payloads1 = [
        json.loads(l)["payload"]
        for l in (rec1 / "messages.jsonl").read_text().splitlines()
        if json.loads(l)["kind"] == "caption"
    ]
    payloads2 = [
        json.loads(l)["payload"]
        for l in (rec2 / "messages.jsonl").read_text().splitlines()
        if json.loads(l)["kind"] == "caption"
    ]
    assert payloads1 == payloads2, "round-trip payloads diverged"
