"""PIL Virtual Camera Renderer.

Wires VirtualWebcamManager (PIL frame generation) to pyvirtualcam (device output).
Features:
- Frame-paced timer (not busy-wait)
- Config snapshot per frame (prevents mid-frame tearing)
- Subscribes to CaptionBuffer events (dirty-flag rendering)
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from bot.virtual_webcam import VirtualWebcamManager, WebcamConfig
from livetranslate_common.logging import get_logger

logger = get_logger()

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat

    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False


class PILVirtualCamRenderer:
    """Renders subtitle frames via PIL and outputs to a virtual camera device.

    Subscribes to both MeetingSessionConfig changes and CaptionBuffer events.
    Renders frames on a background thread at the configured FPS.
    """

    def __init__(
        self,
        config: Any,  # MeetingSessionConfig
        caption_buffer: Any,  # CaptionBuffer
        use_virtual_cam: bool = True,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self._config = config
        self._caption_buffer = caption_buffer
        self._use_virtual_cam = use_virtual_cam and PYVIRTUALCAM_AVAILABLE
        self._width = width
        self._height = height
        self._fps = fps

        # Create the PIL renderer
        self._webcam_manager = VirtualWebcamManager(
            config=WebcamConfig(width=width, height=height, fps=fps)
        )

        # Rendering state
        self._running = False
        self._thread: threading.Thread | None = None
        self._cam: Any = None
        self._dirty = True

        # Public stats
        self.frames_rendered: int = 0
        self.last_frame: np.ndarray | None = None
        self.last_config_snapshot: dict[str, Any] = {}

        # Subscribe to config changes
        config.subscribe(self._on_config_changed)

        # Subscribe to caption buffer events
        caption_buffer.subscribe(self._on_caption_event)

    @property
    def is_running(self) -> bool:
        return self._running

    def _on_config_changed(self, changed_fields: set[str]) -> None:
        """Mark dirty when config changes."""
        self._dirty = True

    def _on_caption_event(self, event_type: str, caption: Any) -> None:
        """Mark dirty when captions change. Also sync captions to VirtualWebcamManager."""
        self._dirty = True

        translation_data = {
            "translation_id": caption.id,
            "translated_text": caption.translated_text or caption.original_text or "",
            "source_language": "",
            "target_language": caption.target_language,
            "speaker_name": caption.speaker_name,
            "speaker_id": None,
            "translation_confidence": caption.confidence,
        }

        if event_type == "added":
            self._webcam_manager.add_translation(translation_data)
        elif event_type == "updated":
            # Find and update existing translation with same ID
            for t in self._webcam_manager.current_translations:
                if t.translation_id == caption.id:
                    t.text = translation_data["translated_text"]
                    break
        elif event_type == "expired":
            # Remove expired translation from deque
            translations = self._webcam_manager.current_translations
            for i, t in enumerate(list(translations)):
                if t.translation_id == caption.id:
                    del translations[i]
                    break

    def start_rendering(self) -> None:
        """Start the render loop in a background thread."""
        if self._running:
            return

        if self._use_virtual_cam:
            self._cam = pyvirtualcam.Camera(
                self._width, self._height, self._fps, fmt=PixelFormat.RGB
            )
            logger.info("virtual_camera_started", device=self._cam.device)

        self._running = True
        self._thread = threading.Thread(
            target=self._render_loop, daemon=True, name="pil-vcam-renderer"
        )
        self._thread.start()

    def stop_rendering(self) -> None:
        """Stop the render loop and close virtual camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cam:
            self._cam.close()
            self._cam = None

    def _render_loop(self) -> None:
        """Frame-paced render loop."""
        frame_interval = 1.0 / self._fps
        next_frame_time = time.monotonic()

        while self._running:
            next_frame_time += frame_interval

            # Snapshot config for this frame (atomic read, prevents tearing)
            self.last_config_snapshot = self._config.snapshot()

            # Generate frame via PIL
            self._webcam_manager._generate_frame()

            if self._webcam_manager.current_frame is not None:
                frame = self._webcam_manager.current_frame

                # Ensure RGB (composite RGBA onto black if needed)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
                    frame = (frame[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

                self.last_frame = frame
                self.frames_rendered += 1
                self._dirty = False

                # Send to virtual camera device
                if self._cam:
                    self._cam.send(frame)

            # Sleep until next frame is due
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.monotonic()
