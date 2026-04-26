"""Tests that MeetingSessionConfig updates propagate to active sinks per-frame.

Phase 9.7 spec conformance: when an operator runs `/mode split` or `/theme light`
during a meeting, the next frame the bot renders must reflect the new state
without a restart.

Validates B5 of the canonical subtitle system design (config snapshot per frame
prevents tearing).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from livedemo.sinks.png import PngSink
from livedemo.sinks.pyvirtualcam import PyVirtualCamSink
from livetranslate_common.theme import DisplayMode
from services.meeting_session_config import MeetingSessionConfig
from services.pipeline.adapters.source_adapter import CaptionEvent


def _evt(i: int) -> CaptionEvent:
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
async def test_png_sink_reads_display_mode_from_meeting_config_per_frame(tmp_path):
    cfg = MeetingSessionConfig(session_id="S1", display_mode="subtitle")
    sink = PngSink(out_dir=tmp_path, meeting_config=cfg)
    async with sink:
        await sink.consume(_evt(1))
        first_mode = sink._manager.config.display_mode
        assert first_mode == DisplayMode.SUBTITLE

        # Live update — operator changes mode mid-run
        cfg.update(display_mode="split")

        await sink.consume(_evt(2))
        second_mode = sink._manager.config.display_mode
        assert second_mode == DisplayMode.SPLIT


@pytest.mark.asyncio
async def test_png_sink_reads_show_diarization_from_meeting_config(tmp_path):
    """show_diarization_ids must follow MeetingSessionConfig.show_speakers
    (or a dedicated flag). Test asserts the show flag flows through."""
    cfg = MeetingSessionConfig(
        session_id="S1",
        display_mode="subtitle",
        show_speakers=True,
    )
    sink = PngSink(out_dir=tmp_path, meeting_config=cfg)
    async with sink:
        await sink.consume(_evt(1))
        # Cycle through show_speakers off; subsequent frame should rebuild WebcamConfig.
        cfg.update(show_speakers=False)
        await sink.consume(_evt(2))
        assert sink._manager.config.show_speaker_names is False


class _FakeCamera:
    def __init__(self, *args, **kwargs) -> None:
        self.frames_sent = []
        self.closed = False
    def send(self, frame): self.frames_sent.append(frame.copy())
    def close(self): self.closed = True


@pytest.mark.asyncio
async def test_pyvirtualcam_sink_picks_up_mode_changes(tmp_path):
    cfg = MeetingSessionConfig(session_id="S1", display_mode="subtitle")
    fake = _FakeCamera()
    sink = PyVirtualCamSink(camera=fake, meeting_config=cfg)
    async with sink:
        await sink.consume(_evt(1))
        assert sink._manager.config.display_mode == DisplayMode.SUBTITLE

        cfg.update(display_mode="interpreter")
        await sink.consume(_evt(2))
        assert sink._manager.config.display_mode == DisplayMode.INTERPRETER

    assert len(fake.frames_sent) == 2


@pytest.mark.asyncio
async def test_meeting_config_unsubscribed_on_sink_exit(tmp_path):
    """No subscriber leak — sink __aexit__ must unsubscribe from the config."""
    cfg = MeetingSessionConfig(session_id="S1")
    before = len(cfg._subscribers)
    sink = PngSink(out_dir=tmp_path, meeting_config=cfg)
    async with sink:
        assert len(cfg._subscribers) == before + 1
    assert len(cfg._subscribers) == before
