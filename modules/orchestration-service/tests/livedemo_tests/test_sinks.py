"""Tests for sinks/png + sinks/canvas_ws (B3, B4).

We use the canonical CaptionEvent already defined in
`services/pipeline/adapters/source_adapter.py` — DRY, no duplicate schema.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from livedemo.sinks.png import PngSink
from livedemo.sinks.canvas_ws import CanvasWsSink, _StubBotHarness
from services.pipeline.adapters.source_adapter import CaptionEvent


def _evt(
    *,
    original: str = "Hello everyone",
    translation: str | None = "你好大家",
    speaker_name: str | None = "Alice",
    speaker_id: str | None = "SPEAKER_00",
    src: str = "en",
    tgt: str = "zh",
) -> CaptionEvent:
    return CaptionEvent(
        event_type="added",
        caption_id="cap-1",
        text=original,
        translated_text=translation,
        speaker_name=speaker_name,
        speaker_id=speaker_id,
        source_lang=src,
        target_lang=tgt,
    )


# ─── PNG sink ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_png_sink_writes_one_file_per_event(tmp_path):
    sink = PngSink(out_dir=tmp_path)
    async with sink:
        await sink.consume(_evt(original="A", translation="B"))
        await sink.consume(_evt(original="C", translation="D"))
    files = sorted(tmp_path.glob("*.png"))
    assert len(files) == 2
    assert files[0].stat().st_size > 0


@pytest.mark.asyncio
async def test_png_sink_files_are_sequential(tmp_path):
    sink = PngSink(out_dir=tmp_path)
    async with sink:
        for _ in range(3):
            await sink.consume(_evt())
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == ["frame-0001.png", "frame-0002.png", "frame-0003.png"]


@pytest.mark.asyncio
async def test_png_sink_renders_paired_block_layout(tmp_path):
    """B3 — frame must be a valid 1280x720 RGB after consuming a pair."""
    sink = PngSink(out_dir=tmp_path)
    async with sink:
        await sink.consume(
            _evt(
                original="Hello everyone, welcome to the meeting.",
                translation="大家好，欢迎参加会议。",
            )
        )
    frame = sink.last_frame  # numpy RGB exposed for tests
    assert frame is not None
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8


@pytest.mark.asyncio
async def test_png_sink_hides_diarization_ids_by_default(tmp_path):
    """B4 — speaker_id like 'SPEAKER_00' must NOT appear in the rendered label."""
    sink = PngSink(out_dir=tmp_path)
    async with sink:
        await sink.consume(_evt(speaker_name="Alice", speaker_id="SPEAKER_00"))
    # Inspect the manager's last caption — display label should be 'Alice', not 'Alice (SPEAKER_00)'
    last = sink.last_caption_label
    assert last is not None
    assert "SPEAKER_" not in last, f"diarization tag leaked into label: {last!r}"
    assert "Alice" in last


@pytest.mark.asyncio
async def test_png_sink_can_show_diarization_when_enabled(tmp_path):
    """Inverse of B4 — opt-in flag re-enables diarization tag."""
    sink = PngSink(out_dir=tmp_path, show_diarization_ids=True)
    async with sink:
        await sink.consume(_evt(speaker_name="Alice", speaker_id="SPEAKER_00"))
    last = sink.last_caption_label
    assert last is not None
    assert "SPEAKER_00" in last


# ─── canvas_ws sink (Phase 1 stub harness) ───────────────


@pytest.mark.asyncio
async def test_canvas_ws_sink_pushes_frames_to_harness():
    harness = _StubBotHarness()
    sink = CanvasWsSink(harness=harness)
    async with sink:
        await sink.consume(_evt())
        await sink.consume(_evt(original="x", translation="y"))
    # Surrogate: harness recorded N frames matching N events.
    assert harness.frame_count == 2
    assert harness.last_frame_shape == (720, 1280, 3)


@pytest.mark.asyncio
async def test_canvas_ws_sink_round_trips_through_real_harness(tmp_path):
    """End-to-end: PNG → WS → fake bot client receives frame messages.

    Uses the real BotHarness in spawn_bot=False mode so we exercise the actual
    WS protocol the production bot will speak.
    """
    import asyncio
    import json
    import socket

    import websockets

    from livedemo.bot_harness import BotHarness
    from livedemo.config import LiveDemoConfig

    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    cfg = LiveDemoConfig(
        meeting_url="https://meet.google.com/aaa-bbbb-ccc",
        source="file",
        replay_jsonl=tmp_path / "_unused.jsonl",
        canvas_ws_port=port,
    )
    harness = BotHarness(cfg, spawn_bot=False)
    received_frames: list[dict] = []

    async def fake_bot():
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            # consume hello
            await asyncio.wait_for(ws.recv(), timeout=2.0)
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    received_frames.append(json.loads(raw))
            except asyncio.TimeoutError:
                pass

    async with harness:
        bot_task = asyncio.create_task(fake_bot())
        await harness._await_client_ready()
        sink = CanvasWsSink(harness=harness)
        async with sink:
            await sink.consume(_evt(original="hello", translation="你好"))
            await sink.consume(_evt(original="bye", translation="再见"))
        await asyncio.sleep(0.2)  # let frames drain
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass

    assert len(received_frames) == 2
    assert all(f["type"] == "frame" for f in received_frames)
    assert all(len(f["data"]) > 100 for f in received_frames)
