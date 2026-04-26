"""BotHarness — async ctx mgr that owns the bot subprocess + frame WS server.

Two-mode design:
- `spawn_bot=True`  → spawns the Node bot_runner, awaits its WS handshake, then
                      meets the meeting and starts pushing frames on `push_frame`.
- `spawn_bot=False` → testing only. Opens the WS server but does NOT spawn the
                      bot. Tests connect a fake client themselves.

Protocol (WS text-framing, JSON):
  Server → Client (immediately on accept):
      {"type": "hello", "version": 1}
  Server → Client (per push_frame):
      {"type": "frame", "data": "<base64 PNG>", "ts": <float>}
  Client → Server (B5 in-call signal):
      {"type": "in_call"}
  Client → Server (frame ack — optional, future):
      {"type": "frame_ack", "id": ...}

We use base64 over the text channel (rather than binary frames) so the JS side
can do a simple `Image.onload(data:image/png;base64,...)` without binary
plumbing. Frame size at 1280x720 RGB → ~200KB PNG → ~270KB base64 → fine for
localhost.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from typing import Any

import numpy as np
import websockets
from PIL import Image
from websockets.asyncio.server import ServerConnection, serve

from .config import LiveDemoConfig
from .recorder import WSRecorder

PROTOCOL_VERSION = 1


class BotHarness:
    def __init__(
        self,
        config: LiveDemoConfig,
        *,
        spawn_bot: bool = True,
        recorder: WSRecorder | None = None,
    ) -> None:
        self.config = config
        self.spawn_bot = spawn_bot
        self.recorder = recorder
        self._server: Any | None = None
        self._client: ServerConnection | None = None
        self._client_ready = asyncio.Event()
        self._proc: asyncio.subprocess.Process | None = None
        self._in_call = asyncio.Event()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=512)

    # ── lifecycle ─────────────────────────────────────────

    async def __aenter__(self) -> "BotHarness":
        self._server = await serve(
            self._on_connect,
            host="127.0.0.1",
            port=self.config.canvas_ws_port,
        )
        if self.spawn_bot:
            await self._spawn_runner()
            try:
                await asyncio.wait_for(self._client_ready.wait(), timeout=30.0)
                # Optional: wait for in-call signal (B5). Not fatal if absent —
                # caller may want to push frames pre-admission for the lobby.
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "Bot runner failed to handshake within 30s — check bot.stderr.log"
                ) from exc
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._proc.kill()
            self._proc = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ── public API ────────────────────────────────────────

    async def push_frame(self, rgb: np.ndarray) -> None:
        """Encode an RGB numpy frame as PNG and send to the connected bot.

        Frames pushed before any client connects are silently dropped. This is
        intentional: the canvas captureStream is live regardless of whether
        Python is feeding frames; missed frames just leave the previous canvas
        content visible.
        """
        png_bytes = _encode_png(rgb)
        if self.recorder is not None:
            self.recorder.record("frame", {"size": len(png_bytes), "ts": time.time()})
        if self._client is None:
            return
        msg = json.dumps(
            {
                "type": "frame",
                "data": base64.b64encode(png_bytes).decode("ascii"),
                "ts": time.time(),
            }
        )
        try:
            await self._client.send(msg)
        except websockets.exceptions.ConnectionClosed:
            self._client = None
            self._client_ready.clear()

    async def wait_for_in_call(self, timeout: float = 60.0) -> None:
        """Block until bot reports it's past the lobby (B5)."""
        await asyncio.wait_for(self._in_call.wait(), timeout=timeout)

    async def send_leave_request(self) -> None:
        """Ask the bot to leave the meeting gracefully.

        Triggered by orchestration `/stop` command per the meeting-subtitle
        system design. Bot will click Leave then close the WS.
        """
        if self._client is None:
            return
        try:
            await self._client.send(json.dumps({"type": "leave_request"}))
        except websockets.exceptions.ConnectionClosed:
            self._client = None
            self._client_ready.clear()

    async def next_audio_chunk(self) -> bytes:
        """Block until the next audio chunk arrives from the bot, return it."""
        return await self._audio_queue.get()

    async def audio_chunks(self):
        """Async iterator over audio chunks pushed by the bot.

        Loops forever until the harness is torn down. Used as an audio source
        for MeetAudioSource (composes this with the orchestration WS protocol).
        """
        while True:
            yield await self._audio_queue.get()

    async def send_chat(self, text: str) -> None:
        """Ask the bot to post a chat message into Meet."""
        if self._client is None:
            return
        try:
            await self._client.send(json.dumps({"type": "chat_send", "text": text}))
        except websockets.exceptions.ConnectionClosed:
            self._client = None
            self._client_ready.clear()

    # ── test hooks ────────────────────────────────────────

    async def _await_client_ready(self) -> None:
        """Tests use this to synchronise; production uses wait_for_in_call."""
        await self._client_ready.wait()

    # ── internals ─────────────────────────────────────────

    async def _on_connect(self, ws: ServerConnection) -> None:
        self._client = ws
        await ws.send(json.dumps({"type": "hello", "version": PROTOCOL_VERSION}))
        self._client_ready.set()
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    # Binary frames = audio PCM from the bot's getUserMedia tap.
                    try:
                        self._audio_queue.put_nowait(raw)
                    except asyncio.QueueFull:
                        # Drop chunk if downstream is slow — better than blocking
                        # the WS reader (which would back-pressure the bot).
                        pass
                    if self.recorder is not None:
                        self.recorder.record("bot_audio", {"bytes": len(raw)})
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                if kind == "in_call":
                    self._in_call.set()
                    if self.recorder is not None:
                        self.recorder.record("bot_in_call", {})
                elif kind == "log":
                    if self.recorder is not None:
                        self.recorder.record(
                            "bot_log",
                            {"level": msg.get("level"), "msg": msg.get("msg")},
                        )
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if self._client is ws:
                self._client = None
                self._client_ready.clear()

    async def _spawn_runner(self) -> None:
        """Spawn the bot_runner Node subprocess via `npm exec tsx`.

        Env passed to the child:
            BOT_WS_PORT, BOT_MEETING_URL, BOT_PROFILE_DIR, BOT_HEADLESS
        """
        import os
        from pathlib import Path

        runner_dir = Path(__file__).resolve().parent / "bot_runner"
        runner_entry = runner_dir / "src" / "runner.ts"
        if not runner_entry.exists():
            raise FileNotFoundError(f"bot_runner missing: {runner_entry}")

        env = {
            **os.environ,
            "BOT_WS_PORT": str(self.config.canvas_ws_port),
            "BOT_MEETING_URL": str(self.config.meeting_url),
            "BOT_PROFILE_DIR": str(self.config.chrome_profile_dir),
            "BOT_HEADLESS": "0",
        }
        self._proc = await asyncio.create_subprocess_exec(
            "npx",
            "tsx",
            str(runner_entry),
            cwd=str(runner_dir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )


def _encode_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    return buf.getvalue()
