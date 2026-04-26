"""Tests for MeetAudioSource — composes BotHarness audio queue with MicSource WS protocol.

The bot pumps audio captured in-page (via init_script's getUserMedia tap) over
the canvas WS as binary frames. The harness queues them. MeetAudioSource feeds
those into the orchestration `/api/audio/stream` and yields CaptionEvents from
returned `translation` messages.

Same protocol as MicSource — only the audio source differs (laptop mic vs bot
in-page audio).
"""
from __future__ import annotations

import asyncio
import json
import socket

import pytest
import websockets

from livedemo.bot_harness import BotHarness
from livedemo.config import LiveDemoConfig
from livedemo.sources.meet_audio import MeetAudioSource
from services.pipeline.adapters.source_adapter import CaptionEvent


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close()
    return p


def _cfg(*, canvas_port: int, ws_url: str) -> LiveDemoConfig:
    return LiveDemoConfig(
        meeting_url="https://meet.google.com/aaa-bbbb-ccc",
        source="file",  # not used; we drive MeetAudioSource directly
        replay_jsonl="/tmp/_unused.jsonl",
        canvas_ws_port=canvas_port,
        orchestration_ws_url=ws_url,
        target_language="zh",
        source_language="en",
    )


@pytest.mark.asyncio
async def test_meet_audio_source_forwards_bot_audio_to_orchestration(tmp_path):
    """End-to-end protocol: bot binary frames → harness queue → MeetAudioSource → orchestration WS."""
    canvas_port = _free_port()
    orch_port = _free_port()

    received_audio_bytes = [0]
    sent_translations: list[dict] = []

    async def fake_orchestration(ws):
        await ws.send(json.dumps({
            "type": "connected", "protocol_version": 1, "session_id": "S1"
        }))
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    received_audio_bytes[0] += len(raw)
                    if received_audio_bytes[0] >= 1280:  # ≥2 chunks of 640
                        # Send a translation back to trigger CaptionEvent
                        await ws.send(json.dumps({
                            "type": "translation",
                            "text": "你好",
                            "source_lang": "en", "target_lang": "zh",
                            "transcript_id": 1, "is_draft": False,
                        }))
                        sent_translations.append({"transcript_id": 1, "text": "你好"})
                        return
        except websockets.exceptions.ConnectionClosed:
            pass

    orch_server = await websockets.serve(fake_orchestration, "127.0.0.1", orch_port)
    cfg = _cfg(canvas_port=canvas_port, ws_url=f"ws://127.0.0.1:{orch_port}")
    harness = BotHarness(cfg, spawn_bot=False)

    try:
        async with harness:
            async with websockets.connect(f"ws://127.0.0.1:{canvas_port}") as bot_ws:
                await asyncio.wait_for(bot_ws.recv(), timeout=2.0)  # consume hello
                await harness._await_client_ready()

                # Spawn MeetAudioSource — it runs concurrently with our audio injector
                src = MeetAudioSource(harness=harness, config=cfg)
                received_events: list[CaptionEvent] = []

                async def collect():
                    async for evt in src.stream():
                        received_events.append(evt)
                        if len(received_events) >= 1:
                            return

                collector = asyncio.create_task(collect())

                # Pretend bot is capturing audio: push 3 chunks via canvas WS
                await asyncio.sleep(0.1)  # let MeetAudioSource connect to orch
                for _ in range(3):
                    await bot_ws.send(b"\x00\x01" * 320)

                await asyncio.wait_for(collector, timeout=5.0)

                assert len(received_events) == 1
                assert received_events[0].translated_text == "你好"
                assert received_events[0].target_lang == "zh"
                assert received_audio_bytes[0] >= 640
    finally:
        orch_server.close()
        await orch_server.wait_closed()


@pytest.mark.asyncio
async def test_meet_audio_source_sends_start_session_with_correct_params(tmp_path):
    """Verifies start_session(sample_rate, channels) matches MicSource's protocol."""
    canvas_port = _free_port()
    orch_port = _free_port()
    received_msgs: list[dict] = []

    async def fake_orchestration(ws):
        await ws.send(json.dumps({"type": "connected", "protocol_version": 1, "session_id": "S1"}))
        try:
            async for raw in ws:
                if isinstance(raw, str):
                    received_msgs.append(json.loads(raw))
        except websockets.exceptions.ConnectionClosed:
            pass

    orch_server = await websockets.serve(fake_orchestration, "127.0.0.1", orch_port)
    cfg = _cfg(canvas_port=canvas_port, ws_url=f"ws://127.0.0.1:{orch_port}")
    harness = BotHarness(cfg, spawn_bot=False)

    try:
        async with harness:
            async with websockets.connect(f"ws://127.0.0.1:{canvas_port}") as bot_ws:
                await asyncio.wait_for(bot_ws.recv(), timeout=2.0)
                await harness._await_client_ready()

                src = MeetAudioSource(harness=harness, config=cfg)

                async def drain():
                    async for _ in src.stream():
                        pass

                drain_task = asyncio.create_task(drain())
                await asyncio.sleep(0.5)
                drain_task.cancel()
                try: await drain_task
                except asyncio.CancelledError: pass
    finally:
        orch_server.close()
        await orch_server.wait_closed()

    types = [m["type"] for m in received_msgs]
    assert "start_session" in types
    start = next(m for m in received_msgs if m["type"] == "start_session")
    # Default 16kHz mono — matches MicSource defaults.
    assert start["sample_rate"] == 16000
    assert start["channels"] == 1
    assert "config" in types
    cfg_msg = next(m for m in received_msgs if m["type"] == "config")
    assert cfg_msg["target_language"] == "zh"
