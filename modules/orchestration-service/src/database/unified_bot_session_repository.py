"""
Unified Bot Session Repository - Orchestration Service

Consolidates all bot session database operations into a single, cohesive repository
that replaces the 5 separate manager classes:

- AudioFileManager -> audio_* methods
- TranscriptManager -> transcript_* methods
- TranslationManager -> translation_* methods
- CorrelationManager -> correlation_* methods
- BotSessionDatabaseManager -> session_* methods

This unified approach reduces complexity while maintaining all functionality
through a clean repository pattern with specialized method groups.

All methods use SQLAlchemy ORM for PostgreSQL compatibility (no raw SQL
with SQLite-style ? placeholders).
"""

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from livetranslate_common.logging import get_logger
from sqlalchemy import select, text

from .database import DatabaseManager
from .models import (
    AudioFile,
    BotSession,
    Correlation,
    Transcript,
    Translation,
)

logger = get_logger()


class UnifiedBotSessionRepository:
    """
    Unified repository for all bot session database operations.

    Consolidates functionality from 5 separate managers into organized method groups:
    - session_* methods: Bot session lifecycle management
    - audio_* methods: Audio file storage and retrieval
    - transcript_* methods: Transcript management
    - translation_* methods: Translation handling
    - correlation_* methods: Speaker correlation tracking
    """

    def __init__(self, database_manager: DatabaseManager):
        """Initialize with database manager."""
        self.db = database_manager

        logger.info("UnifiedBotSessionRepository initialized")

    async def initialize(self) -> bool:
        """Initialize the repository and database connections."""
        try:
            # Ensure database manager is initialized (sync method)
            if not self.db._initialized:
                self.db.initialize()

            logger.info("Unified bot session repository initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize unified bot session repository: {e}")
            return False

    async def shutdown(self):
        """Shutdown repository and close connections."""
        try:
            if self.db:
                await self.db.close()
            logger.info("Unified bot session repository shutdown completed")
        except Exception as e:
            logger.error(f"Error during repository shutdown: {e}")

    # =============================================================================
    # SESSION MANAGEMENT METHODS (replaces BotSessionDatabaseManager)
    # =============================================================================

    async def session_create(
        self,
        session_id: str | None = None,
        meeting_url: str = "",
        bot_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        bot_id: str | None = None,
        meeting_id: str | None = None,
    ) -> BotSession | None:
        """Create a new bot session.

        Maps convenience parameters to the actual ORM model columns:
        - meeting_url -> meeting_uri
        - bot_config + metadata -> session_metadata
        - Auto-generates bot_id and meeting_id if not provided
        """
        try:
            now = datetime.now(UTC).replace(tzinfo=None)

            session = BotSession(
                bot_id=bot_id or f"bot-{uuid.uuid4().hex[:8]}",
                meeting_id=meeting_id or f"meet-{uuid.uuid4().hex[:8]}",
                meeting_uri=meeting_url,
                bot_type="google_meet",
                status="initializing",
                created_at=now,
                started_at=now,
                session_metadata={
                    **(metadata or {}),
                    "bot_config": bot_config or {},
                },
            )

            async with self.db.get_session() as db_session:
                db_session.add(session)
                await db_session.commit()
                await db_session.refresh(session)

            logger.info(f"Created bot session: {session.session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to create bot session: {e}")
            return None

    async def session_get(self, session_id: str | uuid.UUID) -> BotSession | None:
        """Get bot session by ID using SQLAlchemy ORM."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            async with self.db.get_session() as db_session:
                stmt = select(BotSession).where(BotSession.session_id == session_id)
                result = await db_session.execute(stmt)
                return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Failed to get bot session {session_id}: {e}")
            return None

    async def session_update(
        self,
        session_id: str | uuid.UUID,
        status: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Update bot session fields using ORM.

        Supports updating status and any other model fields passed as kwargs.
        Automatically sets ended_at when status is 'completed'.
        """
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            async with self.db.get_session() as db_session:
                stmt = select(BotSession).where(BotSession.session_id == session_id)
                result = await db_session.execute(stmt)
                session = result.scalar_one_or_none()

                if not session:
                    logger.warning(f"Session {session_id} not found for update")
                    return False

                if status is not None:
                    session.status = status
                    if status == "completed":
                        session.ended_at = datetime.now(UTC).replace(tzinfo=None)

                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)

                await db_session.commit()
                await db_session.refresh(session)

            logger.debug(f"Updated session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False

    async def session_update_status(self, session_id: str | uuid.UUID, status: str) -> bool:
        """Update bot session status (convenience wrapper around session_update)."""
        return await self.session_update(session_id=session_id, status=status)

    async def session_list_active(self) -> list[BotSession]:
        """List all active bot sessions using ORM."""
        try:
            async with self.db.get_session() as db_session:
                stmt = select(BotSession).where(
                    BotSession.status.in_(["active", "recording", "processing"])
                )
                result = await db_session.execute(stmt)
                return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to list active sessions: {e}")
            return []

    async def session_cleanup_old(self, older_than_hours: int = 24) -> int:
        """Clean up old completed sessions using ORM."""
        try:
            cutoff_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=older_than_hours)

            async with self.db.get_session() as db_session:
                stmt = select(BotSession).where(
                    BotSession.status == "completed",
                    BotSession.created_at < cutoff_time,
                )
                result = await db_session.execute(stmt)
                sessions = result.scalars().all()

                deleted_count = 0
                for session in sessions:
                    await db_session.delete(session)
                    deleted_count += 1

                await db_session.commit()

                logger.info(f"Cleaned up {deleted_count} old bot sessions")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0

    # =============================================================================
    # AUDIO FILE METHODS (replaces AudioFileManager)
    # =============================================================================

    async def audio_create(
        self,
        session_id: str | uuid.UUID,
        audio_data: bytes,
        metadata: dict[str, Any] | None = None,
        filename: str | None = None,
        file_path: str | None = None,
        mime_type: str = "audio/wav",
    ) -> AudioFile | None:
        """Create an audio file record from raw audio data.

        This method creates the AudioFile ORM record with computed fields
        (file_size, file_hash) and stores metadata including audio properties
        like sample_rate and duration extracted from the metadata dict.
        """
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            now = datetime.now(UTC).replace(tzinfo=None)
            file_hash = hashlib.sha256(audio_data).hexdigest()
            meta = metadata or {}

            audio_file = AudioFile(
                session_id=session_id,
                filename=filename or f"audio_{uuid.uuid4().hex[:8]}.wav",
                file_path=file_path or f"/tmp/audio_{uuid.uuid4().hex[:8]}.wav",
                file_size=len(audio_data),
                file_hash=file_hash,
                mime_type=mime_type,
                duration=meta.get("duration"),
                sample_rate=meta.get("sample_rate"),
                channels=meta.get("channels", 1),
                bit_depth=meta.get("bit_depth", 16),
                created_at=now,
                session_metadata=meta,
            )

            async with self.db.get_session() as db_session:
                db_session.add(audio_file)
                await db_session.commit()
                await db_session.refresh(audio_file)

            logger.debug(f"Created audio file {audio_file.file_id} for session {session_id}")
            return audio_file

        except Exception as e:
            logger.error(f"Failed to create audio file: {e}")
            return None

    async def audio_get(self, file_id: str | uuid.UUID) -> AudioFile | None:
        """Get audio file by ID using ORM."""
        try:
            if isinstance(file_id, str):
                file_id = uuid.UUID(file_id)

            async with self.db.get_session() as db_session:
                stmt = select(AudioFile).where(AudioFile.file_id == file_id)
                result = await db_session.execute(stmt)
                return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Failed to get audio file {file_id}: {e}")
            return None

    async def audio_store_file(
        self,
        session_id: str,
        file_path: str,
        file_type: str = "wav",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store audio file information in database (legacy interface)."""
        try:
            from pathlib import Path as FilePath

            file_size = FilePath(file_path).stat().st_size if FilePath(file_path).exists() else 0
            file_hash = hashlib.sha256(
                FilePath(file_path).read_bytes() if FilePath(file_path).exists() else b""
            ).hexdigest()

            if isinstance(session_id, str):
                session_id_uuid = uuid.UUID(session_id)
            else:
                session_id_uuid = session_id

            audio_file = AudioFile(
                session_id=session_id_uuid,
                filename=FilePath(file_path).name,
                file_path=file_path,
                file_size=file_size,
                file_hash=file_hash,
                mime_type=f"audio/{file_type}",
                session_metadata=metadata or {},
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )

            async with self.db.get_session() as db_session:
                db_session.add(audio_file)
                await db_session.commit()
                await db_session.refresh(audio_file)

            logger.debug(f"Stored audio file {audio_file.file_id} for session {session_id}")
            return str(audio_file.file_id)

        except Exception as e:
            logger.error(f"Failed to store audio file: {e}")
            return None

    async def audio_get_files(self, session_id: str | uuid.UUID) -> list[AudioFile]:
        """Get all audio files for a session using ORM."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            async with self.db.get_session() as db_session:
                stmt = (
                    select(AudioFile)
                    .where(AudioFile.session_id == session_id)
                    .order_by(AudioFile.created_at)
                )
                result = await db_session.execute(stmt)
                return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get audio files for session {session_id}: {e}")
            return []

    async def audio_delete_file(self, file_id: str | uuid.UUID) -> bool:
        """Delete audio file record and optionally the file itself."""
        try:
            from pathlib import Path as FilePath

            if isinstance(file_id, str):
                file_id = uuid.UUID(file_id)

            async with self.db.get_session() as db_session:
                stmt = select(AudioFile).where(AudioFile.file_id == file_id)
                result = await db_session.execute(stmt)
                audio_file = result.scalar_one_or_none()

                if audio_file:
                    file_path = audio_file.file_path

                    await db_session.delete(audio_file)
                    await db_session.commit()

                    # Optionally delete physical file
                    try:
                        if file_path and FilePath(file_path).exists():
                            FilePath(file_path).unlink()
                    except Exception as file_e:
                        logger.warning(f"Failed to delete physical file {file_path}: {file_e}")

                    logger.debug(f"Deleted audio file {file_id}")
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to delete audio file {file_id}: {e}")
            return False

    # =============================================================================
    # TRANSCRIPT METHODS (replaces TranscriptManager)
    # =============================================================================

    async def transcript_create(
        self,
        audio_id: str | uuid.UUID,
        transcription_text: str,
        language: str = "en",
        confidence: float = 1.0,
        segments: list[dict[str, Any]] | None = None,
        speakers: list[dict[str, Any]] | None = None,
        speaker_id: str | None = None,
        speaker_name: str | None = None,
    ) -> Transcript | None:
        """Create a transcript record linked to an audio file.

        Maps convenience parameters to ORM model columns:
        - transcription_text -> text
        - audio_id -> audio_file_id
        - segments, speakers -> stored in session_metadata
        - session_id is looked up from the audio file
        """
        try:
            if isinstance(audio_id, str):
                audio_id = uuid.UUID(audio_id)

            now = datetime.now(UTC).replace(tzinfo=None)

            # Look up the audio file to get session_id
            async with self.db.get_session() as db_session:
                audio_stmt = select(AudioFile).where(AudioFile.file_id == audio_id)
                audio_result = await db_session.execute(audio_stmt)
                audio_file = audio_result.scalar_one_or_none()

                if not audio_file:
                    logger.error(f"Audio file {audio_id} not found for transcript creation")
                    return None

                transcript = Transcript(
                    session_id=audio_file.session_id,
                    text=transcription_text,
                    language=language,
                    confidence=confidence,
                    source="whisper",
                    audio_file_id=audio_id,
                    speaker_id=speaker_id or (speakers[0].get("speaker_id") if speakers else None),
                    speaker_name=speaker_name,
                    start_time=now,
                    end_time=now,
                    session_metadata={
                        "segments": segments or [],
                        "speakers": speakers or [],
                    },
                )

                db_session.add(transcript)
                await db_session.commit()
                await db_session.refresh(transcript)

            logger.debug(f"Created transcript {transcript.transcript_id} " f"for audio {audio_id}")
            return transcript

        except Exception as e:
            logger.error(f"Failed to create transcript: {e}")
            return None

    async def transcript_store(
        self,
        session_id: str | uuid.UUID,
        text_content: str,
        speaker_id: str | None = None,
        timestamp: datetime | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store transcript segment (legacy interface)."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            now = datetime.now(UTC).replace(tzinfo=None)

            transcript = Transcript(
                session_id=session_id,
                text=text_content,
                language="en",
                confidence=confidence,
                source="whisper",
                speaker_id=speaker_id,
                start_time=timestamp or now,
                end_time=timestamp or now,
                session_metadata=metadata or {},
            )

            async with self.db.get_session() as db_session:
                db_session.add(transcript)
                await db_session.commit()
                await db_session.refresh(transcript)

            logger.debug(f"Stored transcript {transcript.transcript_id} for session {session_id}")
            return str(transcript.transcript_id)

        except Exception as e:
            logger.error(f"Failed to store transcript: {e}")
            return None

    async def transcript_get_by_session(
        self, session_id: str | uuid.UUID, speaker_id: str | None = None
    ) -> list[Transcript]:
        """Get transcripts for a session, optionally filtered by speaker."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            async with self.db.get_session() as db_session:
                stmt = select(Transcript).where(Transcript.session_id == session_id)

                if speaker_id:
                    stmt = stmt.where(Transcript.speaker_id == speaker_id)

                stmt = stmt.order_by(Transcript.start_time)

                result = await db_session.execute(stmt)
                return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get transcripts for session {session_id}: {e}")
            return []

    async def transcript_get_recent(
        self, session_id: str | uuid.UUID, minutes: int = 5
    ) -> list[Transcript]:
        """Get recent transcripts from the last N minutes."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            cutoff_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=minutes)

            async with self.db.get_session() as db_session:
                stmt = (
                    select(Transcript)
                    .where(
                        Transcript.session_id == session_id,
                        Transcript.start_time > cutoff_time,
                    )
                    .order_by(Transcript.start_time)
                )
                result = await db_session.execute(stmt)
                return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get recent transcripts: {e}")
            return []

    # =============================================================================
    # TRANSLATION METHODS (replaces TranslationManager)
    # =============================================================================

    async def translation_create(
        self,
        transcript_id: str | uuid.UUID,
        translated_text: str,
        target_language: str,
        confidence: float = 1.0,
        source_language: str = "en",
    ) -> Translation | None:
        """Create a translation record linked to a transcript.

        Looks up the transcript to get session_id and original text.
        """
        try:
            if isinstance(transcript_id, str):
                transcript_id = uuid.UUID(transcript_id)

            now = datetime.now(UTC).replace(tzinfo=None)

            async with self.db.get_session() as db_session:
                # Look up transcript for session_id and original text
                t_stmt = select(Transcript).where(Transcript.transcript_id == transcript_id)
                t_result = await db_session.execute(t_stmt)
                transcript = t_result.scalar_one_or_none()

                if not transcript:
                    logger.error(f"Transcript {transcript_id} not found for translation")
                    return None

                translation = Translation(
                    session_id=transcript.session_id,
                    transcript_id=transcript_id,
                    original_text=transcript.text,
                    translated_text=translated_text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=confidence,
                    start_time=now,
                    end_time=now,
                    word_count=len(translated_text.split()),
                    character_count=len(translated_text),
                )

                db_session.add(translation)
                await db_session.commit()
                await db_session.refresh(translation)

            logger.debug(
                f"Created translation {translation.translation_id} "
                f"for transcript {transcript_id}"
            )
            return translation

        except Exception as e:
            logger.error(f"Failed to create translation: {e}")
            return None

    async def translation_store(
        self,
        transcript_id: str | uuid.UUID,
        target_language: str,
        translated_text: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store translation for a transcript (legacy interface)."""
        translation = await self.translation_create(
            transcript_id=transcript_id,
            translated_text=translated_text,
            target_language=target_language,
            confidence=confidence,
        )
        return str(translation.translation_id) if translation else None

    async def translation_get_by_transcript(
        self, transcript_id: str | uuid.UUID
    ) -> list[Translation]:
        """Get all translations for a transcript using ORM."""
        try:
            if isinstance(transcript_id, str):
                transcript_id = uuid.UUID(transcript_id)

            async with self.db.get_session() as db_session:
                stmt = (
                    select(Translation)
                    .where(Translation.transcript_id == transcript_id)
                    .order_by(Translation.start_time)
                )
                result = await db_session.execute(stmt)
                return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get translations for transcript {transcript_id}: {e}")
            return []

    async def translation_get_by_language(
        self, session_id: str | uuid.UUID, target_language: str
    ) -> list[dict[str, Any]]:
        """Get all translations for a session in a specific language using ORM."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            async with self.db.get_session() as db_session:
                stmt = (
                    select(Translation, Transcript.text.label("original_text"))
                    .join(Transcript, Translation.transcript_id == Transcript.transcript_id)
                    .where(
                        Transcript.session_id == session_id,
                        Translation.target_language == target_language,
                    )
                    .order_by(Transcript.start_time)
                )
                result = await db_session.execute(stmt)
                rows = result.all()

                return [
                    {
                        "translation_id": str(row.Translation.translation_id),
                        "transcript_id": str(row.Translation.transcript_id),
                        "translated_text": row.Translation.translated_text,
                        "target_language": row.Translation.target_language,
                        "confidence": row.Translation.confidence,
                        "original_text": row.original_text,
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get translations by language: {e}")
            return []

    # =============================================================================
    # CORRELATION METHODS (replaces CorrelationManager)
    # =============================================================================

    async def correlation_store(
        self,
        session_id: str | uuid.UUID,
        whisper_speaker_id: str,
        meet_speaker_name: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store speaker correlation between Whisper and Google Meet using ORM."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            now = datetime.now(UTC).replace(tzinfo=None)

            correlation = Correlation(
                session_id=session_id,
                external_source="google_meet",
                external_text=meet_speaker_name,
                external_timestamp=now,
                internal_source="whisper",
                internal_text=whisper_speaker_id,
                internal_timestamp=now,
                confidence=confidence,
                session_metadata=metadata or {},
            )

            async with self.db.get_session() as db_session:
                db_session.add(correlation)
                await db_session.commit()
                await db_session.refresh(correlation)

            logger.debug(f"Stored speaker correlation {correlation.correlation_id}")
            return str(correlation.correlation_id)

        except Exception as e:
            logger.error(f"Failed to store speaker correlation: {e}")
            return None

    async def correlation_get_by_session(self, session_id: str | uuid.UUID) -> list[Correlation]:
        """Get all speaker correlations for a session using ORM."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            async with self.db.get_session() as db_session:
                stmt = (
                    select(Correlation)
                    .where(Correlation.session_id == session_id)
                    .order_by(Correlation.external_timestamp)
                )
                result = await db_session.execute(stmt)
                return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get correlations for session {session_id}: {e}")
            return []

    async def correlation_resolve_speaker(
        self, session_id: str | uuid.UUID, whisper_speaker_id: str
    ) -> str | None:
        """Resolve Whisper speaker ID to Google Meet speaker name using ORM."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            async with self.db.get_session() as db_session:
                stmt = (
                    select(Correlation.external_text)
                    .where(
                        Correlation.session_id == session_id,
                        Correlation.internal_text == whisper_speaker_id,
                    )
                    .order_by(Correlation.confidence.desc())
                    .limit(1)
                )
                result = await db_session.execute(stmt)
                row = result.scalar_one_or_none()
                return row if row else None

        except Exception as e:
            logger.error(f"Failed to resolve speaker: {e}")
            return None

    # =============================================================================
    # UTILITY AND ANALYTICS METHODS
    # =============================================================================

    async def get_session_statistics(self, session_id: str | uuid.UUID) -> dict[str, Any]:
        """Get comprehensive statistics for a session using ORM."""
        try:
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)

            from sqlalchemy import func

            stats: dict[str, Any] = {
                "session_id": str(session_id),
                "audio_files_count": 0,
                "transcripts_count": 0,
                "translations_count": 0,
                "correlations_count": 0,
                "total_duration_seconds": 0,
                "languages": [],
                "speakers": [],
            }

            async with self.db.get_session() as db_session:
                # Audio files count
                result = await db_session.execute(
                    select(func.count(AudioFile.file_id)).where(AudioFile.session_id == session_id)
                )
                stats["audio_files_count"] = result.scalar() or 0

                # Transcripts count
                result = await db_session.execute(
                    select(func.count(Transcript.transcript_id)).where(
                        Transcript.session_id == session_id
                    )
                )
                stats["transcripts_count"] = result.scalar() or 0

                # Translations count
                result = await db_session.execute(
                    select(func.count(Translation.translation_id)).where(
                        Translation.session_id == session_id
                    )
                )
                stats["translations_count"] = result.scalar() or 0

                # Correlations count
                result = await db_session.execute(
                    select(func.count(Correlation.correlation_id)).where(
                        Correlation.session_id == session_id
                    )
                )
                stats["correlations_count"] = result.scalar() or 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {"error": str(e)}

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on the repository."""
        try:
            health_status: dict[str, Any] = {
                "status": "healthy",
                "database_connected": False,
                "tables_accessible": False,
                "last_check": datetime.now(UTC).isoformat(),
            }

            # Check database connection
            async with self.db.get_session() as db_session:
                await db_session.execute(text("SELECT 1"))
                health_status["database_connected"] = True

                # Check table accessibility via ORM
                from sqlalchemy import func

                for model in [BotSession, AudioFile, Transcript, Translation, Correlation]:
                    pk_col = next(iter(model.__table__.primary_key.columns))
                    await db_session.execute(select(func.count(pk_col)))

                health_status["tables_accessible"] = True

            return health_status

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now(UTC).isoformat(),
            }


# Global repository instance
_unified_repository: UnifiedBotSessionRepository | None = None


async def get_unified_bot_session_repository() -> UnifiedBotSessionRepository:
    """Get the global unified bot session repository instance."""
    global _unified_repository

    if _unified_repository is None:
        from .database import get_database_manager

        db_manager = get_database_manager()
        _unified_repository = UnifiedBotSessionRepository(db_manager)
        await _unified_repository.initialize()

    return _unified_repository
