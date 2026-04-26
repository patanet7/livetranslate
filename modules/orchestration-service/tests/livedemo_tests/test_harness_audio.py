"""Tests for BotHarness audio ingestion (Phase 9.2.a).

The harness accepts binary WS frames from the bot runner and exposes them as
an async iterator `audio_chunks()`. This is the upstream side of the
MeetAudioSource pipeline: bot captures audio in-page → forwards as binary
WS frames → harness queues them → MeetAudioSource pumps to orchestration.
"""
from __future__ import annotations

import asyncio
import socket

import pytest
import websockets

from livedemo.bot_harness import BotHarness
from livedemo.config import LiveDemoConfig


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close()
    return p


def _cfg(*, port: int) -> LiveDemoConfig:
    return LiveDemoConfig(
        meeting_url="https://meet.google.com/aaa-bbbb-ccc",
        source="file",
        replay_jsonl="/tmp/_unused.jsonl",
        canvas_ws_port=port,
    )


@pytest.mark.asyncio
async def test_harness_buffers_binary_audio_frames():
    port = _free_port()
    harness = BotHarness(_cfg(port=port), spawn_bot=False)
    async with harness:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await asyncio.wait_for(ws.recv(), timeout=2.0)  # consume hello
            await harness._await_client_ready()
            # Send three pretend audio chunks (320 samples * 2 bytes = 640 bytes)
            for i in range(3):
                await ws.send(b"\x00\x01" * 320)
            # Drain three from harness
            received: list[bytes] = []
            for _ in range(3):
                chunk = await asyncio.wait_for(harness.next_audio_chunk(), timeout=1.0)
                received.append(chunk)
    assert len(received) == 3
    assert all(len(c) == 640 for c in received)


@pytest.mark.asyncio
async def test_harness_audio_chunks_async_iterator_yields_in_order():
    port = _free_port()
    harness = BotHarness(_cfg(port=port), spawn_bot=False)
    async with harness:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await asyncio.wait_for(ws.recv(), timeout=2.0)
            await harness._await_client_ready()
            # Send chunks with distinguishable content
            payloads = [b"A" * 320 + b"\x00", b"B" * 320 + b"\x00", b"C" * 320 + b"\x00"]
            for p in payloads:
                await ws.send(p)
            collected: list[bytes] = []

            async def collect():
                async for chunk in harness.audio_chunks():
                    collected.append(chunk)
                    if len(collected) >= 3:
                        return

            await asyncio.wait_for(collect(), timeout=2.0)
    assert collected == payloads


@pytest.mark.asyncio
async def test_harness_audio_chunks_records_bytes_count_to_recorder(tmp_path):
    """Frame-level observability — recorder captures every audio chunk size."""
    from livedemo.recorder import WSRecorder

    port = _free_port()
    rec = WSRecorder(run_dir=tmp_path)
    harness = BotHarness(_cfg(port=port), spawn_bot=False, recorder=rec)
    async with harness:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await asyncio.wait_for(ws.recv(), timeout=2.0)
            await harness._await_client_ready()
            await ws.send(b"\x00" * 640)
            await ws.send(b"\x00" * 640)
            for _ in range(2):
                await asyncio.wait_for(harness.next_audio_chunk(), timeout=1.0)
    rec.close()
    import json
    lines = (tmp_path / "messages.jsonl").read_text().splitlines()
    parsed = [json.loads(l) for l in lines]
    audio_lines = [p for p in parsed if p["kind"] == "bot_audio"]
    assert len(audio_lines) == 2
    assert audio_lines[0]["payload"]["bytes"] == 640
