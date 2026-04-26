"""MeetAudioSource — bot in-page audio → orchestration → translations.

Composes :class:`BotHarness` (which queues binary audio frames pushed by the
bot's getUserMedia tap) with :class:`MicSource` (which speaks the canonical
orchestration `/api/audio/stream` protocol).

Architecture::

    Meet `<audio>` elements
            │  (init_script tap, MediaStream)
            ▼
    bot_runner page
            │  (binary WS frames over canvas WS)
            ▼
    BotHarness._audio_queue
            │  (audio_chunks() async iter)
            ▼
    MicSource(audio_provider=harness.audio_chunks)
            │  (start_session + binary frames over orchestration WS)
            ▼
    orchestration → transcription → translation
            │
            ▼
    yield CaptionEvent

Same protocol as MicSource — only the audio source changes.
"""
from __future__ import annotations

from typing import AsyncIterator

from livedemo.bot_harness import BotHarness
from livedemo.config import LiveDemoConfig
from livedemo.sources.base import SubtitleSource
from livedemo.sources.mic import MicSource
from services.pipeline.adapters.source_adapter import CaptionEvent


class MeetAudioSource(SubtitleSource):
    """Replays audio captured inside the bot's Chromium page through orchestration.

    The harness must already be running (audio frames arrive on its canvas WS).
    Caller is responsible for harness lifecycle; MeetAudioSource just connects
    a :class:`MicSource` to ``harness.audio_chunks()``.
    """

    def __init__(
        self,
        *,
        harness: BotHarness,
        config: LiveDemoConfig,
        sample_rate: int = 16000,
        channels: int = 1,
        encoding: str = "int16",
    ) -> None:
        self._harness = harness
        self._inner = MicSource(
            ws_url=config.orchestration_ws_url,
            target_language=config.target_language,
            source_language=config.source_language,
            sample_rate=sample_rate,
            channels=channels,
            encoding=encoding,
            audio_provider=lambda: self._harness.audio_chunks(),
        )

    async def stream(self) -> AsyncIterator[CaptionEvent]:
        async for evt in self._inner.stream():
            yield evt
