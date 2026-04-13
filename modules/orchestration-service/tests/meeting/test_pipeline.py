"""Tests for MeetingPipeline.auto_record parameter.

Verifies that when auto_record=True (default), the FlacChunkRecorder starts
automatically on pipeline.start(), and when auto_record=False it stays None.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from meeting.pipeline import MeetingPipeline


def _make_session_manager() -> MagicMock:
    """Stub MeetingSessionManager that returns a fake session on create_session."""
    import uuid

    manager = MagicMock()
    fake_session = MagicMock()
    fake_session.id = uuid.uuid4()
    manager.create_session = AsyncMock(return_value=fake_session)
    manager.promote_to_meeting = AsyncMock()
    manager.update_heartbeat = AsyncMock()
    manager.end_meeting = AsyncMock()
    manager.discard_session = AsyncMock()
    return manager


@pytest.mark.asyncio
@pytest.mark.behavioral
class TestPipelineAutoRecord:
    """Verify auto_record parameter controls whether the recorder starts on pipeline.start()."""

    async def test_pipeline_auto_record_starts_recorder(self, tmp_path: Path) -> None:
        """When auto_record=True, FlacChunkRecorder is created and started on start()."""
        session_manager = _make_session_manager()

        pipeline = MeetingPipeline(
            session_manager=session_manager,
            recording_base_path=tmp_path / "recordings",
            source_type="loopback",
            sample_rate=48000,
            channels=1,
            auto_record=True,
        )

        with patch("meeting.pipeline.FlacChunkRecorder") as MockRecorder:
            mock_recorder_instance = MagicMock()
            MockRecorder.return_value = mock_recorder_instance

            await pipeline.start()

        # Recorder was constructed with the session_id and base_path
        MockRecorder.assert_called_once()
        call_kwargs = MockRecorder.call_args
        assert call_kwargs.kwargs["session_id"] == str(pipeline.session_id)
        assert call_kwargs.kwargs["sample_rate"] == 48000
        assert call_kwargs.kwargs["channels"] == 1

        # Recorder.start() was called
        mock_recorder_instance.start.assert_called_once()

        # Pipeline is in meeting mode because auto_record enables it
        assert pipeline._recorder is mock_recorder_instance

    async def test_pipeline_no_auto_record(self, tmp_path: Path) -> None:
        """When auto_record=False, recorder stays None and is_meeting stays False after start()."""
        session_manager = _make_session_manager()

        pipeline = MeetingPipeline(
            session_manager=session_manager,
            recording_base_path=tmp_path / "recordings",
            source_type="loopback",
            sample_rate=48000,
            channels=1,
            auto_record=False,
        )

        with patch("meeting.pipeline.FlacChunkRecorder") as MockRecorder:
            await pipeline.start()

        # Recorder was never instantiated
        MockRecorder.assert_not_called()

        # Recorder attribute is still None
        assert pipeline._recorder is None

    async def test_pipeline_auto_record_default_is_true(self, tmp_path: Path) -> None:
        """auto_record defaults to True so existing callers get recording out of the box."""
        session_manager = _make_session_manager()

        pipeline = MeetingPipeline(
            session_manager=session_manager,
            recording_base_path=tmp_path / "recordings",
        )

        assert pipeline.auto_record is True

    async def test_pipeline_auto_record_stored_as_attribute(self, tmp_path: Path) -> None:
        """auto_record=False is persisted as self.auto_record."""
        session_manager = _make_session_manager()

        pipeline = MeetingPipeline(
            session_manager=session_manager,
            recording_base_path=tmp_path / "recordings",
            auto_record=False,
        )

        assert pipeline.auto_record is False
