"""Tests for meeting SQLAlchemy models -- behavioral tests against real DB.

Uses the canonical Meeting / MeetingChunk / MeetingTranslation models from
database.models.  The names MeetingSession / MeetingTranscript /
SessionTranslation are imported from the compatibility shim in
database.meeting_models and resolve to the same classes.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import Meeting, MeetingChunk, MeetingTranslation

# Compatibility aliases — same classes, pipeline-friendly names
MeetingSession = Meeting
MeetingTranscript = MeetingChunk
SessionTranslation = MeetingTranslation


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingSession:
    async def test_create_session(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        session = Meeting(
            id=session_id,
            source="loopback",
            status="ephemeral",
            started_at=now,
            last_activity_at=now,
        )
        db_session.add(session)
        await db_session.commit()

        result = await db_session.get(Meeting, session_id)
        assert result is not None
        assert result.source == "loopback"
        assert result.source_type == "loopback"  # property alias
        assert result.status == "ephemeral"

    async def test_promote_to_active(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        session = Meeting(
            id=session_id,
            source="loopback",
            status="ephemeral",
            started_at=now,
            last_activity_at=now,
        )
        db_session.add(session)
        await db_session.commit()

        session.status = "active"
        session.recording_path = f"recordings/{session_id}"
        await db_session.commit()

        result = await db_session.get(Meeting, session_id)
        assert result.status == "active"
        assert result.recording_path is not None


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingTranscript:
    async def test_add_transcript(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        session = Meeting(
            id=session_id,
            source="loopback",
            status="active",
            started_at=now,
            last_activity_at=now,
        )
        db_session.add(session)
        await db_session.commit()

        chunk = MeetingChunk(
            meeting_id=session_id,
            chunk_id=str(uuid.uuid4()),
            text="Hello world",
            timestamp_ms=1000,
            source_language="en",
            confidence=0.95,
            is_final=True,
        )
        db_session.add(chunk)
        await db_session.commit()

        result = await db_session.execute(
            select(MeetingChunk).where(
                MeetingChunk.meeting_id == session_id
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
        now = datetime.now(timezone.utc)
        session = Meeting(
            id=session_id,
            source="gmeet",
            status="active",
            started_at=now,
            last_activity_at=now,
        )
        db_session.add(session)
        await db_session.commit()

        chunk = MeetingChunk(
            meeting_id=session_id,
            chunk_id=str(uuid.uuid4()),
            text="Good morning",
            timestamp_ms=2000,
            source_language="en",
            is_final=True,
        )
        db_session.add(chunk)
        await db_session.commit()

        translation = MeetingTranslation(
            chunk_id=chunk.id,
            target_language="es",
            translated_text="Buenos dias",
            model_used="qwen3.5:7b",
        )
        db_session.add(translation)
        await db_session.commit()

        result = await db_session.execute(
            select(MeetingTranslation).where(
                MeetingTranslation.chunk_id == chunk.id
            )
        )
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].translated_text == "Buenos dias"
        assert rows[0].target_language == "es"


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingSessionServerDefaults:
    async def test_last_activity_at_auto_populated(self, db_session: AsyncSession):
        """last_activity_at should be set by the server default on insert."""
        session_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        session = Meeting(
            id=session_id,
            source="loopback",
            status="ephemeral",
            started_at=now,
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        assert session.last_activity_at is not None

    async def test_source_languages_array_roundtrip(self, db_session: AsyncSession):
        """source_languages and target_languages ARRAY columns must round-trip correctly."""
        session_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        session = Meeting(
            id=session_id,
            source="loopback",
            status="ephemeral",
            started_at=now,
            source_languages=["en", "fr"],
            target_languages=["es", "de", "ja"],
        )
        db_session.add(session)
        await db_session.commit()

        result = await db_session.get(Meeting, session_id)
        assert result.source_languages == ["en", "fr"]
        assert result.target_languages == ["es", "de", "ja"]

    async def test_transcript_relationship_traversal(self, db_session: AsyncSession):
        """Loading session.chunks via ORM relationship should return added chunks."""
        session_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        session = Meeting(
            id=session_id,
            source="loopback",
            status="active",
            started_at=now,
            last_activity_at=now,
        )
        db_session.add(session)
        await db_session.commit()

        for i in range(3):
            chunk = MeetingChunk(
                meeting_id=session_id,
                chunk_id=str(uuid.uuid4()),
                text=f"Transcript {i}",
                timestamp_ms=i * 1000,
                source_language="en",
                is_final=True,
            )
            db_session.add(chunk)
        await db_session.commit()

        # Expire the cached session object (sync) and reload with eager-loaded chunks
        db_session.expire(session)
        result = await db_session.execute(
            select(Meeting)
            .where(Meeting.id == session_id)
            .options(selectinload(Meeting.chunks))
        )
        loaded = result.scalar_one()
        assert len(loaded.chunks) == 3
        texts = {c.text for c in loaded.chunks}
        assert texts == {"Transcript 0", "Transcript 1", "Transcript 2"}
