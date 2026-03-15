"""Tests for MeetingPipeline — behavioral tests against a real DB.

Uses the same db_session fixture from conftest.py as test_session_manager.py.
No mocks — real MeetingSessionManager with a real PostgreSQL testcontainer session.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import Meeting
from meeting.pipeline import MeetingPipeline
from meeting.session_manager import MeetingSessionManager


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingPipeline:
    @pytest.fixture
    def manager(self, db_session: AsyncSession, tmp_path):
        return MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
            heartbeat_timeout_s=120,
        )

    async def test_start_creates_ephemeral(self, manager, tmp_path):
        """start() must persist an ephemeral session and return its UUID."""
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
        )
        session_id = await pipeline.start()
        assert session_id is not None
        assert not pipeline.is_meeting

        session = await manager.db.get(Meeting, session_id)
        assert session is not None
        assert session.status == "ephemeral"

    async def test_process_audio_returns_downsampled(self, manager, tmp_path):
        """process_audio must return a 16kHz mono array regardless of input rate."""
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
        )
        await pipeline.start()

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = await pipeline.process_audio(audio)
        assert len(result) == 16000

    async def test_promote_starts_recording(self, manager, db_session: AsyncSession, tmp_path):
        """promote_to_meeting() must flip is_meeting and update DB status to active."""
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        assert pipeline.is_meeting is True

        session = await db_session.get(Meeting, pipeline.session_id)
        assert session.status == "active"
        assert session.recording_path is not None

    async def test_promote_records_audio_to_flac(self, manager, tmp_path):
        """After promotion, process_audio must write FLAC chunks to disk."""
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        # Write 1 second of audio — the recorder flushes on stop
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        await pipeline.process_audio(audio)
        await pipeline.end()

        session_dir = tmp_path / "recordings" / str(pipeline.session_id)
        manifest = json.loads((session_dir / "manifest.json").read_text())
        assert manifest["total_samples"] > 0

    async def test_end_stops_everything(self, manager, db_session: AsyncSession, tmp_path):
        """end() must set is_meeting=False and mark the DB session as completed."""
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()
        session_id = pipeline.session_id
        await pipeline.end()

        assert pipeline.is_meeting is False

        session = await db_session.get(Meeting, session_id)
        assert session.status == "completed"
        assert session.ended_at is not None

    async def test_ephemeral_audio_not_recorded(self, manager, tmp_path):
        """In ephemeral mode, no FLAC files should be written to disk."""
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        # Do NOT promote — stay ephemeral

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = await pipeline.process_audio(audio)

        # Downsampled audio is still returned
        assert len(result) == 16000

        # No session directory should exist (no recording)
        session_dir = tmp_path / "recordings" / str(pipeline.session_id)
        assert not session_dir.exists()

    async def test_process_audio_before_start_returns_empty(self, manager, tmp_path):
        """process_audio called before start() must return an empty array."""
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
        )
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = await pipeline.process_audio(audio)
        assert len(result) == 0
