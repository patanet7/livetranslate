"""PyVirtualCamSink — production-spec virtual-camera output.

Wraps `pyvirtualcam.Camera` as a CaptionSink. Same renderer as PngSink and
CanvasWsSink (VirtualWebcamManager → paired-block PIL frames) so all three
sinks produce identical pixels.

Use this sink when:
- macOS with OBS Virtual Camera installed (one-time manual activation)
- Linux with v4l2loopback module loaded

Use CanvasWsSink when:
- macOS without OBS, or for demos that should run on any laptop without setup
- Tests / CI where Chromium is acceptable but a system virtual cam isn't

The two are interchangeable — switch via `cfg.sink ∈ {pyvirtualcam, canvas, png}`.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from bot.virtual_webcam import VirtualWebcamManager, WebcamConfig
from livetranslate_common.logging import get_logger
from livetranslate_common.theme import DisplayMode
from .base import CaptionSink, apply_meeting_config_snapshot
from services.pipeline.adapters.source_adapter import CaptionEvent

logger = get_logger()


class _CameraProtocol(Protocol):
    """Subset of pyvirtualcam.Camera the sink relies on (DI surface)."""

    def send(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


class PyVirtualCamSink(CaptionSink):
    def __init__(
        self,
        *,
        camera: _CameraProtocol | None = None,
        display_mode: DisplayMode = DisplayMode.SUBTITLE,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        show_diarization_ids: bool = False,
        meeting_config: object | None = None,
    ) -> None:
        self._camera = camera  # If None, real camera built lazily on __aenter__
        self._explicit_camera = camera is not None
        self._cfg = WebcamConfig(
            width=width,
            height=height,
            fps=fps,
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

    async def __aenter__(self) -> "PyVirtualCamSink":
        if self._camera is None:
            try:
                import pyvirtualcam
                from pyvirtualcam import PixelFormat
            except ImportError as exc:
                raise RuntimeError(
                    "pyvirtualcam not installed. Install OBS Virtual Camera (macOS) "
                    "or v4l2loopback (Linux), then `uv add pyvirtualcam`."
                ) from exc
            self._camera = pyvirtualcam.Camera(
                width=self._cfg.width,
                height=self._cfg.height,
                fps=self._cfg.fps,
                fmt=PixelFormat.RGB,
            )
            logger.info(
                "pyvirtualcam_opened",
                device=getattr(self._camera, "device", "unknown"),
                width=self._cfg.width,
                height=self._cfg.height,
                fps=self._cfg.fps,
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._meeting_config is not None and hasattr(self._meeting_config, "unsubscribe"):
            self._meeting_config.unsubscribe(self._on_meeting_config_changed)
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:  # pragma: no cover - close should not raise
                pass
            if not self._explicit_camera:
                self._camera = None

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
        if frame is None or self._camera is None:
            return
        self._camera.send(frame)
        logger.info(
            "frame_rendered",
            sink="pyvirtualcam",
            caption_id=caption.caption_id,
        )
