"""Tests for BotHarness.send_leave_request + send_chat — protocol round-trip."""
from __future__ import annotations

import asyncio
import json
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
async def test_harness_send_leave_request_reaches_client():
    port = _free_port()
    harness = BotHarness(_cfg(port=port), spawn_bot=False)
    received: list[dict] = []
    async with harness:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await asyncio.wait_for(ws.recv(), timeout=2.0)  # consume hello
            await harness._await_client_ready()
            await harness.send_leave_request()
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            received.append(json.loads(raw))
    assert received[0] == {"type": "leave_request"}


@pytest.mark.asyncio
async def test_harness_send_chat_reaches_client():
    port = _free_port()
    harness = BotHarness(_cfg(port=port), spawn_bot=False)
    received: list[dict] = []
    async with harness:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await asyncio.wait_for(ws.recv(), timeout=2.0)
            await harness._await_client_ready()
            await harness.send_chat("Hello world")
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            received.append(json.loads(raw))
    assert received[0]["type"] == "chat_send"
    assert received[0]["text"] == "Hello world"
