"""Tests for meeting SQLAlchemy models -- behavioral tests against real DB."""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.meeting_models import MeetingSession, MeetingTranscript, SessionTranslation


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingSession:
    async def test_create_session(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        session = MeetingSession(
            id=session_id,
            source_type="loopback",
            status="ephemeral",
            started_at=datetime.now(timezone.utc),
        )
        db_session.add(session)
        await db_session.commit()

        result = await db_session.get(MeetingSession, session_id)
        assert result is not None
        assert result.source_type == "loopback"
        assert result.status == "ephemeral"

    async def test_promote_to_active(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        session = MeetingSession(
            id=session_id,
            source_type="loopback",
            status="ephemeral",
            started_at=datetime.now(timezone.utc),
        )
        db_session.add(session)
        await db_session.commit()

        session.status = "active"
        session.recording_path = f"recordings/{session_id}"
        await db_session.commit()

        result = await db_session.get(MeetingSession, session_id)
        assert result.status == "active"
        assert result.recording_path is not None


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingTranscript:
    async def test_add_transcript(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        session = MeetingSession(
            id=session_id,
            source_type="loopback",
            status="active",
            started_at=datetime.now(timezone.utc),
        )
        db_session.add(session)
        await db_session.commit()

        transcript = MeetingTranscript(
            session_id=session_id,
            timestamp_ms=1000,
            text="Hello world",
            source_language="en",
            confidence=0.95,
            is_final=True,
        )
        db_session.add(transcript)
        await db_session.commit()

        result = await db_session.execute(
            select(MeetingTranscript).where(
                MeetingTranscript.session_id == session_id
            )
        )
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].text == "Hello world"


@pytest.mark.asyncio
@pytest.mark.integration
class TestSessionTranslation:
    async def test_add_translation(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        session = MeetingSession(
            id=session_id,
            source_type="gmeet",
            status="active",
            started_at=datetime.now(timezone.utc),
        )
        db_session.add(session)
        await db_session.commit()

        transcript = MeetingTranscript(
            session_id=session_id,
            timestamp_ms=2000,
            text="Good morning",
            source_language="en",
            is_final=True,
        )
        db_session.add(transcript)
        await db_session.commit()

        translation = SessionTranslation(
            transcript_id=transcript.id,
            target_language="es",
            translated_text="Buenos dias",
            model_used="qwen3.5:7b",
        )
        db_session.add(translation)
        await db_session.commit()

        result = await db_session.execute(
            select(SessionTranslation).where(
                SessionTranslation.transcript_id == transcript.id
            )
        )
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].translated_text == "Buenos dias"
        assert rows[0].target_language == "es"
