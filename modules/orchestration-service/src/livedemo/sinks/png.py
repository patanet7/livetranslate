"""PNG sink — render each CaptionEvent to a numbered PNG on disk.

Used for:
- CI smoke tests (no bot, no network, deterministic)
- Layout iteration without spinning up Chromium

Internally uses :class:`VirtualWebcamManager` so the rendered frame is identical
to what the canvas_ws sink (Phase 1) pushes to the bot.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from PIL import Image

from bot.virtual_webcam import VirtualWebcamManager, WebcamConfig
from livetranslate_common.logging import get_logger
from .base import CaptionSink, apply_meeting_config_snapshot
from livetranslate_common.theme import DisplayMode
from services.pipeline.adapters.source_adapter import CaptionEvent

logger = get_logger()


class PngSink(CaptionSink):
    """Render-to-PNG sink. Filenames are zero-padded sequential."""

    def __init__(
        self,
        out_dir: Path | str,
        *,
        display_mode: DisplayMode = DisplayMode.SUBTITLE,
        width: int = 1280,
        height: int = 720,
        show_diarization_ids: bool = False,
        meeting_config: object | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._cfg = WebcamConfig(
            width=width,
            height=height,
            fps=30,
            display_mode=display_mode,
            show_diarization_ids=show_diarization_ids,
        )
        self._manager = VirtualWebcamManager(self._cfg)
        self._counter = 0
        # Test hooks
        self.last_frame: np.ndarray | None = None
        self.last_caption_label: str | None = None
        # Live-config wiring (Phase 9.7)
        self._meeting_config = meeting_config
        self._dirty_meeting_config = meeting_config is not None
        if meeting_config is not None and hasattr(meeting_config, "subscribe"):
            meeting_config.subscribe(self._on_meeting_config_changed)

    def _on_meeting_config_changed(self, _changed: set) -> None:
        self._dirty_meeting_config = True

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._meeting_config is not None and hasattr(self._meeting_config, "unsubscribe"):
            self._meeting_config.unsubscribe(self._on_meeting_config_changed)
        return await super().__aexit__(exc_type, exc, tb)

    async def consume(self, caption: CaptionEvent) -> None:
        if self._dirty_meeting_config and self._meeting_config is not None:
            apply_meeting_config_snapshot(self._cfg, self._meeting_config.snapshot())
            self._dirty_meeting_config = False
        original = caption.text or None
        translated = caption.translated_text or None
        # Hand off to existing paired-block API (reused, not duplicated).
        self._manager.add_caption(
            original_text=original,
            translated_text=translated,
            speaker_name=caption.speaker_name,
            speaker_id=getattr(caption, "speaker_id", None),
            source_language=caption.source_lang,
            target_language=caption.target_lang or "en",
            confidence=caption.confidence,
            pair_id=caption.caption_id,
        )
        # Render synchronously (cheap PIL work) then offload encode to thread.
        self._manager._generate_frame()  # writes self._manager.current_frame
        frame = self._manager.current_frame
        if frame is None:
            return
        self.last_frame = frame
        if self._manager.current_translations:
            self.last_caption_label = self._manager.current_translations[-1].speaker_name

        self._counter += 1
        path = self.out_dir / f"frame-{self._counter:04d}.png"
        await asyncio.to_thread(_save_png, path, frame)
        logger.info(
            "frame_rendered",
            sink="png",
            caption_id=caption.caption_id,
            path=str(path),
            frame_index=self._counter,
        )


def _save_png(path: Path, frame: np.ndarray) -> None:
    Image.fromarray(frame, "RGB").save(path)
