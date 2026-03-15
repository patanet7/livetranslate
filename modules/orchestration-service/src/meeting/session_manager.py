"""MeetingSessionManager — session lifecycle, promotion, heartbeat orphan detection.

Manages the full lifecycle: ephemeral → active → completed/interrupted.
Heartbeat monitoring detects sessions abandoned without an end event.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.meeting_models import MeetingSession, MeetingTranscript, SessionTranslation
from livetranslate_common.logging import get_logger

logger = get_logger()


class MeetingSessionManager:
    def __init__(
        self,
        db: AsyncSession,
        recording_base_path: Path,
        heartbeat_timeout_s: int = 120,
    ):
        self.db = db
        self.recording_base_path = recording_base_path
        self.heartbeat_timeout_s = heartbeat_timeout_s

    async def create_session(
        self,
        source_type: str,
        sample_rate: int = 48000,
        channels: int = 2,
    ) -> MeetingSession:
        session = MeetingSession(
            id=uuid.uuid4(),
            source_type=source_type,
            status="ephemeral",
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("session_created", session_id=str(session.id), source=source_type)
        return session

    async def promote_to_meeting(self, session_id: uuid.UUID) -> MeetingSession:
        session = await self.db.get(MeetingSession, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.status = "active"
        session.recording_path = str(self.recording_base_path / str(session_id))
        session.last_activity_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("session_promoted", session_id=str(session_id))
        return session

    async def end_meeting(self, session_id: uuid.UUID) -> MeetingSession:
        session = await self.db.get(MeetingSession, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.status = "completed"
        session.ended_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("session_ended", session_id=str(session_id))
        return session

    async def update_heartbeat(self, session_id: uuid.UUID) -> None:
        """Update the last_activity_at timestamp for an active session."""
        await self.db.execute(
            update(MeetingSession)
            .where(MeetingSession.id == session_id)
            .values(last_activity_at=datetime.now(timezone.utc))
        )
        await self.db.commit()

    async def add_transcript(
        self,
        session_id: uuid.UUID,
        text: str,
        timestamp_ms: int,
        language: str,
        confidence: float,
        is_final: bool,
        speaker_id: str | None = None,
        source_id: str | None = None,
    ) -> MeetingTranscript:
        transcript = MeetingTranscript(
            session_id=session_id,
            timestamp_ms=timestamp_ms,
            text=text,
            source_language=language,
            confidence=confidence,
            is_final=is_final,
            speaker_id=speaker_id,
            source_id=source_id,
        )
        self.db.add(transcript)
        await self.db.commit()
        await self.db.refresh(transcript)
        return transcript

    async def detect_orphans(self) -> list[MeetingSession]:
        """Return active sessions whose last heartbeat exceeded the timeout threshold."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.heartbeat_timeout_s)
        result = await self.db.execute(
            select(MeetingSession).where(
                MeetingSession.status == "active",
                MeetingSession.last_activity_at < cutoff,
            )
        )
        return list(result.scalars().all())

    async def mark_interrupted(self, session_id: uuid.UUID) -> None:
        """Mark a session as interrupted (orphaned or crash-recovered)."""
        await self.db.execute(
            update(MeetingSession)
            .where(MeetingSession.id == session_id)
            .values(status="interrupted", ended_at=datetime.now(timezone.utc))
        )
        await self.db.commit()
        logger.warning("session_interrupted", session_id=str(session_id))

    async def recover_on_startup(self) -> list[MeetingSession]:
        """Find sessions still marked 'active' at startup and mark them interrupted.

        Called once at service start to clean up sessions that never received
        an end event (e.g., due to a crash).
        """
        result = await self.db.execute(
            select(MeetingSession).where(MeetingSession.status == "active")
        )
        orphans = list(result.scalars().all())
        for orphan in orphans:
            await self.mark_interrupted(orphan.id)
        if orphans:
            logger.warning("startup_orphans_recovered", count=len(orphans))
        return orphans

    async def recover_untranslated(self) -> list[MeetingTranscript]:
        """Return finalized transcripts that have no associated translation.

        Used during crash recovery to re-submit transcripts whose translation
        was lost (e.g., translation service outage or process crash).
        """
        result = await self.db.execute(
            select(MeetingTranscript)
            .outerjoin(
                SessionTranslation,
                MeetingTranscript.id == SessionTranslation.transcript_id,
            )
            .where(
                MeetingTranscript.is_final == True,  # noqa: E712
                SessionTranslation.id == None,  # noqa: E711
            )
        )
        untranslated = list(result.scalars().all())
        if untranslated:
            logger.info("untranslated_transcripts_found", count=len(untranslated))
        return untranslated
