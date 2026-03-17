"""MeetingSessionManager — session lifecycle, promotion, heartbeat orphan detection.

Manages the full lifecycle: ephemeral → active → completed/interrupted.
Heartbeat monitoring detects sessions abandoned without an end event.

Uses the canonical ``Meeting`` and ``MeetingChunk`` models from
``database.models``; the ``MeetingSession`` / ``MeetingTranscript`` /
``SessionTranslation`` names are compatibility aliases that resolve to
the same classes.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from database.models import Meeting, MeetingChunk, MeetingTranslation
from livetranslate_common.logging import get_logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger()

# ---------------------------------------------------------------------------
# Aliases used throughout this module — keep names that match the pipeline
# spec for readability.
# ---------------------------------------------------------------------------
MeetingSession = Meeting
MeetingTranscript = MeetingChunk
SessionTranslation = MeetingTranslation


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
        self._db_lock = asyncio.Lock()

    async def create_session(
        self,
        source_type: str,
        sample_rate: int = 48000,
        channels: int = 2,
    ) -> Meeting:
        """Create a new ephemeral meeting session and persist it."""
        now = datetime.now(UTC)
        session = Meeting(
            id=uuid.uuid4(),
            source=source_type,          # canonical column name
            status="ephemeral",
            started_at=now,
            last_activity_at=now,
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("session_created", session_id=str(session.id), source=source_type)
        return session

    async def promote_to_meeting(self, session_id: uuid.UUID) -> Meeting:
        """Promote an ephemeral session to an active (recording) meeting."""
        session = await self.db.get(Meeting, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.status = "active"
        session.recording_path = str(self.recording_base_path / str(session_id))
        session.last_activity_at = datetime.now(UTC)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("session_promoted", session_id=str(session_id))
        return session

    async def end_meeting(self, session_id: uuid.UUID) -> Meeting:
        """Mark a session as completed and record the end time."""
        session = await self.db.get(Meeting, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.status = "completed"
        session.ended_at = datetime.now(UTC)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("session_ended", session_id=str(session_id))
        return session

    async def update_heartbeat(self, session_id: uuid.UUID) -> None:
        """Update the last_activity_at timestamp for an active session."""
        await self.db.execute(
            update(Meeting)
            .where(Meeting.id == session_id)
            .values(last_activity_at=datetime.now(UTC))
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
    ) -> MeetingChunk:
        """Persist a real-time transcript chunk to meeting_chunks."""
        async with self._db_lock:
            chunk = MeetingChunk(
                meeting_id=session_id,          # canonical FK column
                chunk_id=str(uuid.uuid4()),     # generate dedup key for pipeline chunks
                text=text,
                timestamp_ms=timestamp_ms,
                source_language=language,
                confidence=confidence,
                is_final=is_final,
                speaker_id=speaker_id,
                source_id=source_id,
            )
            self.db.add(chunk)
            await self.db.commit()
            await self.db.refresh(chunk)
            return chunk

    async def detect_orphans(self) -> list[Meeting]:
        """Return active sessions whose last heartbeat exceeded the timeout threshold."""
        cutoff = datetime.now(UTC) - timedelta(seconds=self.heartbeat_timeout_s)
        result = await self.db.execute(
            select(Meeting).where(
                Meeting.status == "active",
                Meeting.last_activity_at < cutoff,
            )
        )
        return list(result.scalars().all())

    async def mark_interrupted(self, session_id: uuid.UUID) -> None:
        """Mark a session as interrupted (orphaned or crash-recovered)."""
        await self.db.execute(
            update(Meeting)
            .where(Meeting.id == session_id)
            .values(status="interrupted", ended_at=datetime.now(UTC))
        )
        await self.db.commit()
        logger.warning("session_interrupted", session_id=str(session_id))

    async def recover_on_startup(self) -> list[Meeting]:
        """Find sessions still marked 'active' at startup and mark them interrupted.

        Called once at service start to clean up sessions that never received
        an end event (e.g., due to a crash).
        """
        result = await self.db.execute(
            select(Meeting).where(Meeting.status == "active")
        )
        orphans = list(result.scalars().all())
        for orphan in orphans:
            await self.mark_interrupted(orphan.id)
        if orphans:
            logger.warning("startup_orphans_recovered", count=len(orphans))
        return orphans

    async def save_translation(
        self,
        chunk_id: uuid.UUID | None,
        translated_text: str,
        source_language: str,
        target_language: str,
        model_used: str,
        translation_time_ms: float | None = None,
    ) -> MeetingTranslation:
        """Persist a translation, optionally linked to a MeetingChunk.

        Called after streaming translation completes. chunk_id may be None
        if the chunk wasn't persisted (ephemeral mode or timing race).
        """
        async with self._db_lock:
            translation = MeetingTranslation(
                chunk_id=chunk_id,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                model_used=model_used,
                translation_time_ms=translation_time_ms,
            )
            self.db.add(translation)
            await self.db.commit()
            await self.db.refresh(translation)
            logger.info(
                "translation_persisted",
                translation_id=str(translation.id),
                chunk_id=str(chunk_id) if chunk_id else None,
                target_language=target_language,
            )
            return translation

    async def recover_untranslated(self) -> list[MeetingChunk]:
        """Return finalized chunks that have no associated translation.

        Used during crash recovery to re-submit chunks whose translation
        was lost (e.g., translation service outage or process crash).
        """
        result = await self.db.execute(
            select(MeetingChunk)
            .outerjoin(
                MeetingTranslation,
                MeetingChunk.id == MeetingTranslation.chunk_id,
            )
            .where(
                MeetingChunk.is_final == True,  # noqa: E712
                MeetingTranslation.id == None,  # noqa: E711
            )
        )
        untranslated = list(result.scalars().all())
        if untranslated:
            logger.info("untranslated_transcripts_found", count=len(untranslated))
        return untranslated
