"""Phase 6.5 — full Python pipeline E2E: file source → canvas_ws sink → BotHarness → fake bot.

This is the integration capstone. Exercises:
  - real FileSource (reads JSONL, parses CaptionEvent)
  - real CanvasWsSink (renders via VirtualWebcamManager, calls harness.push_frame)
  - real BotHarness (WS server, frame protocol, recorder integration)
  - real WSRecorder (writes JSONL with caption + frame entries)
  - fake bot client (validates the protocol contract)

Combined with the bot_runner vitest tests (which prove the runner.ts side of
the protocol works against a real <video> element), this test gives us full
behavioral coverage of every component except the live Meet UI itself.
"""
from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import pytest
import websockets

from livedemo.bot_harness import BotHarness
from livedemo.config import LiveDemoConfig
from livedemo.pipeline import run_once
from livedemo.recorder import WSRecorder
from livedemo.sinks.canvas_ws import CanvasWsSink
from livedemo.sources.file import FileSource


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close()
    return p


@pytest.fixture
def fixture_path() -> Path:
    """Use the checked-in deterministic 6-caption fixture."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "livedemo" / "short-dialog.jsonl"


@pytest.mark.asyncio
async def test_full_pipeline_file_source_to_fake_bot(tmp_path, fixture_path):
    """End-to-end: 6 captions in fixture → 6 frames received by fake bot."""
    assert fixture_path.exists(), f"fixture missing: {fixture_path}"

    port = _free_port()
    cfg = LiveDemoConfig(
        meeting_url="https://meet.google.com/aaa-bbbb-ccc",
        source="file",
        sink="canvas",
        replay_jsonl=fixture_path,
        canvas_ws_port=port,
    )
    received_frames: list[dict] = []

    async def fake_bot():
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await asyncio.wait_for(ws.recv(), timeout=2.0)  # consume hello
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    received_frames.append(json.loads(raw))
            except asyncio.TimeoutError:
                pass

    rec_dir = tmp_path / "rec"
    recorder = WSRecorder(run_dir=rec_dir, enabled=True)
    harness = BotHarness(cfg, spawn_bot=False, recorder=recorder)

    async with harness:
        bot_task = asyncio.create_task(fake_bot())
        await harness._await_client_ready()

        src = FileSource(jsonl_path=fixture_path, replay_speed=0.0)
        sink = CanvasWsSink(harness=harness)
        n = await run_once(source=src, sink=sink, recorder=recorder)

        # Allow buffered frames to flush over WS
        await asyncio.sleep(0.3)
        bot_task.cancel()
        try: await bot_task
        except asyncio.CancelledError: pass

    recorder.close()

    # Pipeline counts
    assert n == 6, f"expected 6 captions, processed {n}"

    # Bot received 6 frames
    assert len(received_frames) == 6
    assert all(f["type"] == "frame" for f in received_frames)
    # Each frame has real PNG bytes (≥1KB after base64)
    assert all(len(f["data"]) > 1000 for f in received_frames)

    # Recorder captured both captions AND frame pushes (B7)
    rec_lines = (rec_dir / "messages.jsonl").read_text().strip().split("\n")
    parsed = [json.loads(l) for l in rec_lines]
    captions = [p for p in parsed if p["kind"] == "caption"]
    frames = [p for p in parsed if p["kind"] == "frame"]
    assert len(captions) == 6
    assert len(frames) == 6


@pytest.mark.asyncio
async def test_full_pipeline_record_replay_round_trip(tmp_path, fixture_path):
    """B8 — round-trip the FULL pipeline: original fixture → record → replay → record-2; payloads identical."""
    port1 = _free_port()
    port2 = _free_port()

    async def run_with_recording(jsonl_in: Path, port: int, rec_dir: Path) -> None:
        cfg = LiveDemoConfig(
            meeting_url="https://meet.google.com/aaa-bbbb-ccc",
            source="file", sink="canvas",
            replay_jsonl=jsonl_in,
            canvas_ws_port=port,
        )
        recorder = WSRecorder(run_dir=rec_dir, enabled=True)
        harness = BotHarness(cfg, spawn_bot=False, recorder=recorder)

        async def fake_bot():
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await asyncio.wait_for(ws.recv(), timeout=2.0)
                try:
                    while True:
                        await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass

        async with harness:
            bot_task = asyncio.create_task(fake_bot())
            await harness._await_client_ready()
            src = FileSource(jsonl_path=jsonl_in, replay_speed=0.0)
            sink = CanvasWsSink(harness=harness)
            await run_once(source=src, sink=sink, recorder=recorder)
            await asyncio.sleep(0.2)
            bot_task.cancel()
            try: await bot_task
            except asyncio.CancelledError: pass
        recorder.close()

    rec1 = tmp_path / "rec1"
    rec2 = tmp_path / "rec2"
    await run_with_recording(fixture_path, port1, rec1)
    # Now replay the recording itself through the same pipeline
    await run_with_recording(rec1 / "messages.jsonl", port2, rec2)

    def _captions(d: Path) -> list[dict]:
        return [
            json.loads(l)["payload"]
            for l in (d / "messages.jsonl").read_text().splitlines()
            if json.loads(l)["kind"] == "caption"
        ]

    assert _captions(rec1) == _captions(rec2), "round-trip diverged"
