"""MicSource — live mic → orchestration /api/audio/stream → translation events.

This is the canonical end-to-end path: if mic mode renders subtitles correctly,
the production WebSocket framing, draft/final routing, context store, and LLM
client all work.

Architecture:
  audio_provider yields PCM bytes
       │
       ▼
  ws.send_bytes() — pushed continuously in a background task
       │
       ▼
  orchestration /api/audio/stream
       │
       ▼ (transcription, translation)
  ws.recv()  — `segment` + `translation` JSON
       │
       ▼
  yield CaptionEvent on each `translation` (is_draft=False)
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Awaitable, Callable

import websockets

from .base import SubtitleSource
from services.pipeline.adapters.source_adapter import CaptionEvent

AudioProvider = Callable[[], AsyncIterator[bytes]]


class MicSource(SubtitleSource):
    def __init__(
        self,
        *,
        ws_url: str,
        target_language: str,
        source_language: str = "auto",
        sample_rate: int = 16000,
        channels: int = 1,
        encoding: str = "int16",
        audio_provider: AudioProvider | None = None,
    ) -> None:
        self.ws_url = ws_url
        self.target_language = target_language
        self.source_language = source_language
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoding = encoding
        self._audio_provider = audio_provider or _default_mic_audio

    async def stream(self) -> AsyncIterator[CaptionEvent]:
        async with websockets.connect(self.ws_url) as ws:
            # Wait for the server's `connected` message before sending anything.
            first = await asyncio.wait_for(ws.recv(), timeout=5.0)
            try:
                first_msg = json.loads(first) if isinstance(first, str) else {}
            except json.JSONDecodeError:
                first_msg = {}
            if first_msg.get("type") != "connected":
                # Not fatal — some servers may skip the connected hello.
                pass

            # Lifecycle messages.
            await ws.send(json.dumps({
                "type": "start_session",
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "encoding": self.encoding,
                "source": "mic",
            }))
            await ws.send(json.dumps({
                "type": "config",
                "language": None if self.source_language == "auto" else self.source_language,
                "target_language": self.target_language,
            }))

            # Background task: pump audio bytes into ws.
            stop_audio = asyncio.Event()
            audio_task = asyncio.create_task(_pump_audio(ws, self._audio_provider, stop_audio))

            # Track latest segment text per transcript_id so translations can
            # be paired with their original.
            seg_text: dict[int, dict] = {}

            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    kind = msg.get("type")
                    if kind == "segment":
                        sid = msg.get("segment_id")
                        if sid is not None:
                            seg_text[sid] = {
                                "text": msg.get("text", ""),
                                "speaker_id": msg.get("speaker_id"),
                                "language": msg.get("language", self.source_language),
                            }
                    elif kind == "translation" and not msg.get("is_draft", False):
                        tid = msg.get("transcript_id")
                        prev = seg_text.get(tid, {}) if tid is not None else {}
                        yield CaptionEvent(
                            event_type="added",
                            caption_id=f"mic-{tid}" if tid is not None else f"mic-{id(msg)}",
                            text=prev.get("text", ""),
                            translated_text=msg.get("text", ""),
                            speaker_name=None,
                            speaker_id=prev.get("speaker_id"),
                            source_lang=msg.get("source_lang", self.source_language),
                            target_lang=msg.get("target_lang", self.target_language),
                            confidence=1.0,
                            is_draft=False,
                        )
            finally:
                stop_audio.set()
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass


async def _pump_audio(ws, audio_provider: AudioProvider, stop: asyncio.Event) -> None:
    try:
        async for chunk in audio_provider():
            if stop.is_set():
                break
            try:
                await ws.send(chunk)
            except websockets.exceptions.ConnectionClosed:
                break
    except asyncio.CancelledError:
        pass


async def _default_mic_audio() -> AsyncIterator[bytes]:
    """Default: capture from default input via sounddevice → 16kHz mono int16.

    Each yielded chunk is 20ms of PCM (320 samples × 2 bytes).
    """
    import sounddevice as sd

    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
    loop = asyncio.get_running_loop()

    def _callback(indata, frames, time_info, status):
        # sounddevice runs on its own thread; thread-safe handoff to async queue.
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
        blocksize=320,  # 20ms
        callback=_callback,
    )
    with stream:
        while True:
            chunk = await queue.get()
            yield chunk
