"""Canvas-WS sink — push rendered frames to the bot harness over a localhost WS.

Sink renders each :class:`CaptionEvent` via :class:`VirtualWebcamManager` (DRY
with the PNG sink — same renderer, same paired-block layout) and forwards the
resulting RGB array to anything implementing `push_frame(rgb)`. The real
:class:`BotHarness` (Phase 1) and the test :class:`_StubBotHarness` both
satisfy this Protocol, so the sink works against either without code change.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from bot.virtual_webcam import VirtualWebcamManager, WebcamConfig
from livetranslate_common.logging import get_logger
from .base import CaptionSink, apply_meeting_config_snapshot
from livetranslate_common.theme import DisplayMode
from services.pipeline.adapters.source_adapter import CaptionEvent

logger = get_logger()


class _PushFrameTarget(Protocol):
    async def push_frame(self, rgb: np.ndarray) -> None: ...


class _StubBotHarness:
    """In-memory recorder used by tests until the real BotHarness lands.

    Mirrors the public surface of `BotHarness.push_frame(rgb)` only. No WS, no
    subprocess, no Playwright.
    """

    def __init__(self) -> None:
        self.frame_count = 0
        self.last_frame_shape: tuple[int, int, int] | None = None
        self.frames: list[np.ndarray] = []

    async def push_frame(self, rgb: np.ndarray) -> None:
        self.frame_count += 1
        self.last_frame_shape = rgb.shape  # type: ignore[assignment]
        self.frames.append(rgb)


class CanvasWsSink(CaptionSink):
    """Renders each CaptionEvent and forwards the RGB frame to the bot harness."""

    def __init__(
        self,
        *,
        harness: _PushFrameTarget,
        display_mode: DisplayMode = DisplayMode.SUBTITLE,
        width: int = 1280,
        height: int = 720,
        show_diarization_ids: bool = False,
        meeting_config: object | None = None,
    ) -> None:
        self._harness = harness
        self._cfg = WebcamConfig(
            width=width,
            height=height,
            fps=30,
            display_mode=display_mode,
            show_diarization_ids=show_diarization_ids,
        )
        self._manager = VirtualWebcamManager(self._cfg)
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
        self._manager.add_caption(
            original_text=caption.text or None,
            translated_text=caption.translated_text or None,
            speaker_name=caption.speaker_name,
            speaker_id=getattr(caption, "speaker_id", None),
            source_language=caption.source_lang,
            target_language=caption.target_lang or "en",
            confidence=caption.confidence,
            pair_id=caption.caption_id,
        )
        self._manager._generate_frame()
        frame = self._manager.current_frame
        if frame is None:
            return
        await self._harness.push_frame(frame)
        logger.info(
            "frame_rendered",
            sink="canvas_ws",
            caption_id=caption.caption_id,
        )
