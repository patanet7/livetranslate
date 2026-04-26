"""Caption Source Adapters — lifecycle-managing source connectors.

Two-layer abstraction:
1. CaptionSourceAdapter (this file) — stateful, lifecycle, event emission
2. ChunkAdapter (existing adapters/) — stateless data transformation

The source adapter USES a chunk adapter internally for format conversion.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from livetranslate_common.logging import get_logger
from livetranslate_common.theme import SPEAKER_COLORS

logger = get_logger()


@dataclass
class CaptionEvent:
    """Canonical event emitted by source adapters, consumed by renderers."""

    event_type: str  # "added", "updated", "expired", "cleared"
    caption_id: str
    text: str
    speaker_name: str | None = None
    speaker_id: str | None = None  # diarization id (e.g. "SPEAKER_00"); rendered iff WebcamConfig.show_diarization_ids
    speaker_color: str = "#4CAF50"
    source_lang: str = "auto"
    target_lang: str | None = None
    translated_text: str | None = None
    confidence: float = 1.0
    is_draft: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None


@runtime_checkable
class CaptionSourceAdapter(Protocol):
    """Protocol for caption source connectors."""

    is_running: bool
    on_caption: Callable[[CaptionEvent], Any] | None

    async def start(self, config: Any) -> None: ...
    async def stop(self) -> None: ...


class BotAudioCaptionSource:
    """Wraps the existing audio WebSocket → transcription pipeline."""

    def __init__(self):
        self.is_running: bool = False
        self.on_caption: Callable[[CaptionEvent], Any] | None = None
        self._speaker_color_idx: int = 0
        self._speaker_colors: dict[str, str] = {}

    async def start(self, config: Any) -> None:
        self.is_running = True
        logger.info("bot_audio_source_started", session_id=getattr(config, "session_id", "?"))

    async def stop(self) -> None:
        self.is_running = False
        logger.info("bot_audio_source_stopped")

    def _get_speaker_color(self, speaker_name: str) -> str:
        if speaker_name not in self._speaker_colors:
            self._speaker_colors[speaker_name] = SPEAKER_COLORS[
                self._speaker_color_idx % len(SPEAKER_COLORS)
            ]
            self._speaker_color_idx += 1
        return self._speaker_colors[speaker_name]

    async def handle_transcription(
        self,
        text: str,
        speaker_name: str,
        source_lang: str = "auto",
        confidence: float = 1.0,
        is_final: bool = False,
    ) -> None:
        """Called when a transcription segment arrives from the audio pipeline."""
        if not self.is_running or not self.on_caption:
            return

        event = CaptionEvent(
            event_type="added",
            caption_id=str(uuid.uuid4()),
            text=text,
            speaker_name=speaker_name,
            speaker_color=self._get_speaker_color(speaker_name),
            source_lang=source_lang,
            confidence=confidence,
            is_draft=not is_final,
        )

        result = self.on_caption(event)
        if hasattr(result, "__await__"):
            await result
