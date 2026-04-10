"""FirefliesCaptionSource — wraps Fireflies as a CaptionSourceAdapter.

Receives Fireflies transcript chunks and emits CaptionEvents
for the CaptionBuffer/renderers.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Callable

from livetranslate_common.logging import get_logger
from livetranslate_common.theme import SPEAKER_COLORS

from .source_adapter import CaptionEvent

logger = get_logger()


class FirefliesCaptionSource:
    """Lifecycle-managing Fireflies caption source."""

    def __init__(self) -> None:
        self.is_running: bool = False
        self.on_caption: Callable[[CaptionEvent], Any] | None = None
        self._speaker_color_idx: int = 0
        self._speaker_colors: dict[str, str] = {}

    async def start(self, config: Any) -> None:
        self.is_running = True
        logger.info("fireflies_caption_source_started")

    async def stop(self) -> None:
        self.is_running = False
        logger.info("fireflies_caption_source_stopped")

    def _get_speaker_color(self, speaker_name: str) -> str:
        if speaker_name not in self._speaker_colors:
            self._speaker_colors[speaker_name] = SPEAKER_COLORS[
                self._speaker_color_idx % len(SPEAKER_COLORS)
            ]
            self._speaker_color_idx += 1
        return self._speaker_colors[speaker_name]

    async def handle_chunk(self, raw_chunk: dict[str, Any]) -> None:
        if not self.is_running or not self.on_caption:
            return

        text = raw_chunk.get("text", "")
        speaker_name = raw_chunk.get("speaker_name", "Unknown")
        chunk_id = raw_chunk.get("chunk_id", str(uuid.uuid4()))

        event = CaptionEvent(
            event_type="added",
            caption_id=chunk_id,
            text=text,
            speaker_name=speaker_name,
            speaker_color=self._get_speaker_color(speaker_name),
            source_lang="auto",
            confidence=1.0,
            is_draft=False,
        )

        result = self.on_caption(event)
        if hasattr(result, "__await__"):
            await result
