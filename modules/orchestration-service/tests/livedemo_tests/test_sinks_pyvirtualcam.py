"""Tests for sinks/pyvirtualcam.py — production-spec virtual-camera sink.

DI-friendly: takes a camera object via constructor so we can run the protocol
test without OBS Virtual Camera or v4l2loopback installed. Default factory
builds the real `pyvirtualcam.Camera`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from livedemo.sinks.pyvirtualcam import PyVirtualCamSink
from services.pipeline.adapters.source_adapter import CaptionEvent


class _FakeCamera:
    """Stand-in for pyvirtualcam.Camera. Records send() and close() calls."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_sent: list[np.ndarray] = []
        self.closed = False

    def send(self, frame: np.ndarray) -> None:
        self.frames_sent.append(frame.copy())

    def sleep_until_next_frame(self) -> None:  # pragma: no cover - timing only
        pass

    def close(self) -> None:
        self.closed = True


def _evt(i: int = 0) -> CaptionEvent:
    return CaptionEvent(
        event_type="added",
        caption_id=f"cap-{i:03d}",
        text=f"original-{i}",
        translated_text=f"translation-{i}",
        speaker_name="Alice",
        speaker_id="SPEAKER_00",
        source_lang="en",
        target_lang="zh",
    )


@pytest.mark.asyncio
async def test_pyvirtualcam_sink_pushes_one_frame_per_event():
    fake = _FakeCamera(1280, 720, 30)
    sink = PyVirtualCamSink(camera=fake)
    async with sink:
        await sink.consume(_evt(1))
        await sink.consume(_evt(2))
    assert len(fake.frames_sent) == 2
    assert fake.frames_sent[0].shape == (720, 1280, 3)
    assert fake.frames_sent[0].dtype == np.uint8


@pytest.mark.asyncio
async def test_pyvirtualcam_sink_closes_camera_on_exit():
    fake = _FakeCamera(1280, 720, 30)
    sink = PyVirtualCamSink(camera=fake)
    async with sink:
        await sink.consume(_evt())
    assert fake.closed is True


@pytest.mark.asyncio
async def test_pyvirtualcam_sink_emits_structlog_event_per_frame():
    """B7 conformance — frame_rendered events flow through structlog."""
    from structlog.testing import capture_logs

    fake = _FakeCamera(1280, 720, 30)
    sink = PyVirtualCamSink(camera=fake)
    with capture_logs() as caplog:
        async with sink:
            await sink.consume(_evt(1))
            await sink.consume(_evt(2))
    rendered = [e for e in caplog if e["event"] == "frame_rendered" and e.get("sink") == "pyvirtualcam"]
    assert len(rendered) == 2
