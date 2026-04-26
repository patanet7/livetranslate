"""Python-side tests for BotHarness — protocol over WS, no real bot.

We connect a fake WS client (the "bot") and exercise the harness's frame-push
contract. Real Playwright integration is in test_bot_runner_e2e.py (slow).

Validates the harness's surface that sinks/canvas_ws.py depends on:
  async with BotHarness(config) as bot:
      await bot.push_frame(rgb_array)
"""
from __future__ import annotations

import asyncio
import json
import socket

import numpy as np
import pytest
import websockets

from livedemo.bot_harness import BotHarness
from livedemo.config import LiveDemoConfig


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _cfg(*, port: int) -> LiveDemoConfig:
    return LiveDemoConfig(
        meeting_url="https://meet.google.com/aaa-bbbb-ccc",
        source="file",
        replay_jsonl="/tmp/_unused.jsonl",
        canvas_ws_port=port,
    )


@pytest.mark.asyncio
async def test_harness_is_async_context_manager_starts_ws_server(tmp_path):
    """Harness opens a WS server on canvas_ws_port without spawning a bot."""
    port = _free_port()
    cfg = _cfg(port=port)
    harness = BotHarness(cfg, spawn_bot=False)  # test mode: no Chromium
    async with harness:
        # Fake bot connects
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            handshake = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(handshake)
            assert msg["type"] == "hello"
            assert "version" in msg


@pytest.mark.asyncio
async def test_harness_push_frame_round_trip(tmp_path):
    """push_frame(rgb) must arrive at the connected client as a frame message."""
    port = _free_port()
    cfg = _cfg(port=port)
    harness = BotHarness(cfg, spawn_bot=False)
    async with harness:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await asyncio.wait_for(ws.recv(), timeout=2.0)  # consume hello
            await harness._await_client_ready()
            # Push a real frame
            rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
            rgb[100:200, 100:200] = (255, 0, 0)  # red square
            await harness.push_frame(rgb)
            received = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(received)
            assert msg["type"] == "frame"
            assert "data" in msg  # base64-encoded PNG
            assert isinstance(msg["data"], str)
            assert len(msg["data"]) > 100  # actual PNG bytes


@pytest.mark.asyncio
async def test_harness_push_frame_before_client_connects_buffers(tmp_path):
    """Frames pushed before client connects are dropped (non-fatal)."""
    port = _free_port()
    cfg = _cfg(port=port)
    harness = BotHarness(cfg, spawn_bot=False)
    async with harness:
        # Push without any client connected — must not raise
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        await harness.push_frame(rgb)
        # No assertion: just confirm no exception.


@pytest.mark.asyncio
async def test_harness_records_frame_pushes_when_recorder_present(tmp_path):
    """B7 surrogate — recorder receives 'frame' kind for each push_frame."""
    from livedemo.recorder import WSRecorder

    port = _free_port()
    cfg = _cfg(port=port)
    rec = WSRecorder(run_dir=tmp_path)
    harness = BotHarness(cfg, spawn_bot=False, recorder=rec)
    async with harness:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await asyncio.wait_for(ws.recv(), timeout=2.0)
            await harness._await_client_ready()
            for _ in range(3):
                await harness.push_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
                await ws.recv()  # drain
    rec.close()
    import json as _json
    lines = (tmp_path / "messages.jsonl").read_text().strip().split("\n")
    parsed = [_json.loads(l) for l in lines]
    frame_lines = [p for p in parsed if p["kind"] == "frame"]
    assert len(frame_lines) == 3
