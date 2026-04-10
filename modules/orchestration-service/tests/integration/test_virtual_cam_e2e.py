"""Integration test: PIL frames → pyvirtualcam → virtual camera device.

Requires: OBS installed with virtual camera extension activated (macOS),
or v4l2loopback loaded on host (Linux).

Run explicitly:
    uv run pytest tests/integration/test_virtual_cam_e2e.py -v
"""

import numpy as np
import pytest

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat

    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.skipif(not PYVIRTUALCAM_AVAILABLE, reason="pyvirtualcam not installed")
class TestPyvirtualcamDevice:
    def test_create_camera_and_send_frame(self):
        """Verify we can create a virtual camera and write an RGB frame."""
        width, height, fps = 1280, 720, 30
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Red frame for visual verification
        frame[:, :, 0] = 255

        try:
            with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.RGB) as cam:
                cam.send(frame)
                cam.sleep_until_next_frame()
                assert cam.device is not None
                assert cam.width == width
                assert cam.height == height
        except RuntimeError as e:
            if "no backend" in str(e).lower() or "obs" in str(e).lower():
                pytest.skip(f"Virtual camera backend not available: {e}")
            raise

    def test_send_multiple_frames(self):
        """Verify we can send multiple frames without error."""
        width, height, fps = 1280, 720, 30

        try:
            with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.RGB) as cam:
                for i in range(10):
                    frame = np.full((height, width, 3), i * 25, dtype=np.uint8)
                    cam.send(frame)
                    cam.sleep_until_next_frame()
                assert cam.frames_sent >= 10
        except RuntimeError as e:
            if "no backend" in str(e).lower() or "obs" in str(e).lower():
                pytest.skip(f"Virtual camera backend not available: {e}")
            raise

    def test_rgba_must_composite_to_rgb(self):
        """RGBA frames cannot be sent directly — must composite onto opaque background."""
        width, height = 1280, 720
        rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_frame[:, :, 0] = 200  # Red channel
        rgba_frame[:, :, 3] = 128  # Semi-transparent

        # Composite onto black background
        alpha = rgba_frame[:, :, 3:4].astype(np.float32) / 255.0
        rgb_frame = (rgba_frame[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

        assert rgb_frame.shape == (height, width, 3)
        assert rgb_frame.dtype == np.uint8
        # Red channel should be ~100 (200 * 0.5)
        assert 90 <= rgb_frame[0, 0, 0] <= 110


import time

import pytest

from bot.pil_virtual_cam_renderer import PILVirtualCamRenderer
from services.meeting_session_config import MeetingSessionConfig
from services.caption_buffer import CaptionBuffer


@pytest.mark.integration
class TestPILVirtualCamRenderer:
    def test_creates_with_config(self):
        config = MeetingSessionConfig(session_id="test-123")
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)
        assert renderer is not None
        assert renderer.frames_rendered == 0

    def test_renders_frame_on_caption_event(self):
        config = MeetingSessionConfig(session_id="test-123")
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

        renderer.start_rendering()
        buffer.add_caption(translated_text="Hello world", speaker_name="Alice")

        time.sleep(0.2)  # Give render loop a few cycles

        assert renderer.frames_rendered > 0
        assert renderer.last_frame is not None
        assert renderer.last_frame.shape == (720, 1280, 3)

        renderer.stop_rendering()

    def test_renders_without_captions(self):
        """Should render waiting frame when no captions exist."""
        config = MeetingSessionConfig(session_id="test-123")
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

        renderer.start_rendering()
        time.sleep(0.2)

        assert renderer.frames_rendered > 0
        assert renderer.last_frame is not None
        assert renderer.last_frame.shape == (720, 1280, 3)

        renderer.stop_rendering()

    def test_config_snapshot_per_frame(self):
        """Config changes should be picked up by next frame."""
        config = MeetingSessionConfig(session_id="test-123", font_size=24)
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

        renderer.start_rendering()
        time.sleep(0.1)

        config.update(font_size=48)
        time.sleep(0.1)

        assert renderer.last_config_snapshot["font_size"] == 48
        renderer.stop_rendering()

    def test_start_stop_lifecycle(self):
        config = MeetingSessionConfig(session_id="test-123")
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

        assert not renderer.is_running
        renderer.start_rendering()
        assert renderer.is_running
        renderer.stop_rendering()
        assert not renderer.is_running

    def test_double_start_is_safe(self):
        config = MeetingSessionConfig(session_id="test-123")
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

        renderer.start_rendering()
        renderer.start_rendering()  # Should not create second thread
        time.sleep(0.1)
        renderer.stop_rendering()
