"""Tests for sources/mic.py — protocol with the production /api/audio/stream WS.

We use a fake WS server that mirrors orchestration's protocol. The test
asserts:
  - source sends `start_session` then `config(target_language)`
  - source pumps binary PCM frames continuously
  - source yields a CaptionEvent for every `translation` message with is_draft=False
  - source ignores draft translations and `translation_chunk` deltas (those are
    streaming-incremental UI updates, not new captions)

The real-network E2E test (test_mic_e2e.py) hits an actual orchestration server.
"""
from __future__ import annotations

import asyncio
import json
import socket
from typing import Any

import pytest
import websockets

from livedemo.sources.mic import MicSource
from services.pipeline.adapters.source_adapter import CaptionEvent


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close()
    return p


async def _silent_audio_frames(n: int = 2):
    """Yield N empty audio frames so the source has *something* to send."""
    for _ in range(n):
        yield b"\x00" * 320  # ~10ms 16kHz mono int16
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_mic_source_sends_start_session_then_config(tmp_path):
    """Source must open with start_session(sample_rate, channels), then config(target_language)."""
    port = _free_port()
    received_messages: list[dict] = []
    audio_byte_count = [0]

    async def server(ws):
        # Simulate orchestration: send `connected`, collect client traffic
        await ws.send(json.dumps({"type": "connected", "protocol_version": 1, "session_id": "S1"}))
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    audio_byte_count[0] += len(raw)
                else:
                    received_messages.append(json.loads(raw))
        except websockets.exceptions.ConnectionClosed:
            pass

    server_task = await websockets.serve(server, "127.0.0.1", port)
    try:
        src = MicSource(
            ws_url=f"ws://127.0.0.1:{port}",
            target_language="zh",
            source_language="en",
            audio_provider=_silent_audio_frames,
            sample_rate=16000,
            channels=1,
        )
        # Drain the source — server never sends translations, so source ends when audio ends.
        timeout_task = asyncio.create_task(asyncio.sleep(0.5))
        async def _drain():
            async for _ in src.stream():
                pass
        drain = asyncio.create_task(_drain())
        await asyncio.wait({timeout_task, drain}, return_when=asyncio.FIRST_COMPLETED)
        drain.cancel()
        try: await drain
        except asyncio.CancelledError: pass
    finally:
        server_task.close()
        await server_task.wait_closed()

    # Must have sent start_session
    types = [m["type"] for m in received_messages]
    assert "start_session" in types
    start_msg = next(m for m in received_messages if m["type"] == "start_session")
    assert start_msg["sample_rate"] == 16000
    assert start_msg["channels"] == 1
    # Must have sent config with target_language
    assert "config" in types
    cfg_msg = next(m for m in received_messages if m["type"] == "config")
    assert cfg_msg["target_language"] == "zh"
    # Must have sent at least one binary audio frame
    assert audio_byte_count[0] > 0


@pytest.mark.asyncio
async def test_mic_source_yields_caption_per_translation(tmp_path):
    """For every server `translation` (is_draft=False), source yields one CaptionEvent."""
    port = _free_port()

    async def server(ws):
        await ws.send(json.dumps({"type": "connected", "protocol_version": 1, "session_id": "S1"}))
        # Receive at least the start_session before sending translations.
        await ws.recv()  # start_session
        # Drain audio frames in background — keep WS alive
        async def drain():
            try:
                async for _ in ws: pass
            except websockets.exceptions.ConnectionClosed: pass
        asyncio.create_task(drain())
        await asyncio.sleep(0.05)
        # Send a draft translation (must be ignored), then a final.
        await ws.send(json.dumps({
            "type": "translation",
            "text": "draft you",
            "source_lang": "en", "target_lang": "zh",
            "transcript_id": 1, "is_draft": True,
        }))
        await ws.send(json.dumps({
            "type": "translation",
            "text": "你好",
            "source_lang": "en", "target_lang": "zh",
            "transcript_id": 1, "is_draft": False,
        }))
        await ws.send(json.dumps({
            "type": "translation",
            "text": "再见",
            "source_lang": "en", "target_lang": "zh",
            "transcript_id": 2, "is_draft": False,
        }))
        await asyncio.sleep(0.2)

    server_task = await websockets.serve(server, "127.0.0.1", port)
    try:
        src = MicSource(
            ws_url=f"ws://127.0.0.1:{port}",
            target_language="zh",
            source_language="en",
            audio_provider=lambda: _infinite_silence(),
        )
        out: list[CaptionEvent] = []
        async def collect():
            async for evt in src.stream():
                out.append(evt)
                if len(out) >= 2:
                    return
        await asyncio.wait_for(collect(), timeout=3.0)
    finally:
        server_task.close()
        await server_task.wait_closed()

    assert len(out) == 2
    assert out[0].translated_text == "你好"
    assert out[0].is_draft is False
    assert out[0].source_lang == "en"
    assert out[0].target_lang == "zh"
    assert out[1].translated_text == "再见"


async def _infinite_silence():
    while True:
        yield b"\x00" * 320
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_mic_source_caption_text_uses_segment_text_when_available(tmp_path):
    """If a `segment` message preceded `translation`, CaptionEvent.text = segment text."""
    port = _free_port()

    async def server(ws):
        await ws.send(json.dumps({"type": "connected", "protocol_version": 1, "session_id": "S1"}))
        await ws.recv()  # start_session
        async def drain():
            try:
                async for _ in ws: pass
            except websockets.exceptions.ConnectionClosed: pass
        asyncio.create_task(drain())
        await asyncio.sleep(0.05)
        await ws.send(json.dumps({
            "type": "segment",
            "segment_id": 7, "text": "Hello world",
            "language": "en", "confidence": 0.9,
            "stable_text": "Hello world", "unstable_text": "",
            "is_final": True, "is_draft": False, "speaker_id": "SPEAKER_00",
        }))
        await ws.send(json.dumps({
            "type": "translation",
            "text": "你好世界",
            "source_lang": "en", "target_lang": "zh",
            "transcript_id": 7, "is_draft": False,
        }))
        await asyncio.sleep(0.2)

    server_task = await websockets.serve(server, "127.0.0.1", port)
    try:
        src = MicSource(
            ws_url=f"ws://127.0.0.1:{port}",
            target_language="zh",
            source_language="en",
            audio_provider=lambda: _infinite_silence(),
        )
        evt: CaptionEvent | None = None
        async def collect():
            nonlocal evt
            async for e in src.stream():
                evt = e
                return
        await asyncio.wait_for(collect(), timeout=3.0)
    finally:
        server_task.close()
        await server_task.wait_closed()

    assert evt is not None
    assert evt.text == "Hello world"  # original
    assert evt.translated_text == "你好世界"
    assert evt.speaker_id == "SPEAKER_00"
