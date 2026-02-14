#!/usr/bin/env python3
"""
Bot Session Database Manager

Comprehensive database integration for Google Meet bot sessions, managing
all aspects of meeting data including audio files, transcripts, translations,
and time correlation with full session lifecycle tracking.

Features:
- Bot session lifecycle management with PostgreSQL
- Audio file storage and metadata tracking
- Google Meet transcript integration and storage
- In-house transcription and translation management
- Time-coded correlation data storage
- Meeting analytics and insights
- File cleanup and archival management
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles
import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BotSessionRecord:
    """Complete bot session record."""

    session_id: str
    bot_id: str
    meeting_id: str
    meeting_title: str | None
    meeting_uri: str | None
    google_meet_space_id: str | None
    conference_record_id: str | None
    status: str  # 'spawning', 'active', 'paused', 'ended', 'error'
    start_time: datetime
    end_time: datetime | None = None
    participant_count: int = 0
    target_languages: list[str] = None
    session_metadata: dict[str, Any] = None
    performance_stats: dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class AudioFileRecord:
    """Audio file storage record."""

    file_id: str
    session_id: str
    file_path: str
    file_name: str
    file_size: int
    file_format: str
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    chunk_start_time: float | None
    chunk_end_time: float | None
    audio_quality_score: float | None
    processing_status: str  # 'pending', 'processing', 'completed', 'failed'
    file_hash: str | None
    metadata: dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class TranscriptRecord:
    """Transcript record (both Google Meet and in-house)."""

    transcript_id: str
    session_id: str
    source_type: str  # 'google_meet', 'whisper_service', 'manual'
    transcript_text: str
    language_code: str
    start_timestamp: float
    end_timestamp: float
    speaker_id: str | None
    speaker_name: str | None
    confidence_score: float | None
    segment_index: int
    audio_file_id: str | None
    google_transcript_entry_id: str | None
    processing_metadata: dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class TranslationRecord:
    """Translation record."""

    translation_id: str
    session_id: str
    source_transcript_id: str
    translated_text: str
    source_language: str
    target_language: str
    translation_confidence: float | None
    translation_service: str  # 'translation_service', 'google_translate', 'manual'
    speaker_id: str | None
    speaker_name: str | None
    start_timestamp: float
    end_timestamp: float
    processing_metadata: dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class CorrelationRecord:
    """Time correlation record."""

    correlation_id: str
    session_id: str
    google_transcript_id: str | None
    inhouse_transcript_id: str | None
    correlation_confidence: float
    timing_offset: float
    correlation_type: str  # 'exact', 'interpolated', 'inferred'
    correlation_method: str
    speaker_id: str | None
    start_timestamp: float
    end_timestamp: float
    correlation_metadata: dict[str, Any] = None
    created_at: datetime = None


class DatabaseConfig:
    """Database configuration with production-ready pool settings."""

    def __init__(self, **kwargs):
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 5432)
        self.database = kwargs.get("database", "livetranslate")
        self.username = kwargs.get("username", "postgres")
        self.password = kwargs.get("password", "livetranslate")
        # Connection pool configuration
        self.min_connections = kwargs.get("min_connections", 5)
        self.max_connections = kwargs.get("max_connections", 20)
        # Timeout configuration (in seconds)
        self.connection_timeout = kwargs.get("connection_timeout", 30.0)
        self.command_timeout = kwargs.get("command_timeout", 60.0)
        # Pool behavior
        self.max_queries = kwargs.get("max_queries", 50000)  # Recycle connection after N queries
        self.max_inactive_connection_lifetime = kwargs.get(
            "max_inactive_connection_lifetime",
            300.0,  # 5 minutes
        )


class AudioFileManager:
    """Manages audio file storage and metadata."""

    def __init__(self, storage_path: str, db_pool):
        self.storage_path = Path(storage_path)
        self.db_pool = db_pool
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def store_audio_file(
        self,
        session_id: str,
        audio_data: bytes,
        file_format: str = "wav",
        metadata: dict | None = None,
    ) -> str:
        """Store audio file and create database record."""
        try:
            # Generate file ID and path
            file_id = f"audio_{uuid.uuid4().hex}"
            timestamp = int(time.time())
            file_name = f"{session_id}_{timestamp}_{file_id}.{file_format}"

            # Create session directory
            session_dir = self.storage_path / session_id
            session_dir.mkdir(exist_ok=True)

            file_path = session_dir / file_name

            # Write audio file
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(audio_data)

            # Calculate file hash
            file_hash = hashlib.sha256(audio_data).hexdigest()

            # Extract audio metadata if available
            duration = metadata.get("duration_seconds") if metadata else None
            sample_rate = metadata.get("sample_rate") if metadata else None
            channels = metadata.get("channels") if metadata else None
            chunk_start = metadata.get("chunk_start_time") if metadata else None
            chunk_end = metadata.get("chunk_end_time") if metadata else None
            quality_score = metadata.get("audio_quality_score") if metadata else None

            # Create database record
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bot_sessions.audio_files (
                        file_id, session_id, file_path, file_name, file_size,
                        file_format, duration_seconds, sample_rate, channels,
                        chunk_start_time, chunk_end_time, audio_quality_score,
                        processing_status, file_hash, metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                    file_id,
                    session_id,
                    str(file_path),
                    file_name,
                    len(audio_data),
                    file_format,
                    duration,
                    sample_rate,
                    channels,
                    chunk_start,
                    chunk_end,
                    quality_score,
                    "completed",
                    file_hash,
                    json.dumps(metadata or {}),
                    datetime.now(UTC).replace(tzinfo=None),
                    datetime.now(UTC).replace(tzinfo=None),
                )

            logger.info(f"Stored audio file: {file_id} ({len(audio_data)} bytes)")
            return file_id

        except Exception as e:
            logger.error(f"Error storing audio file: {e}")
            return None

    async def get_audio_file_info(self, file_id: str) -> AudioFileRecord | None:
        """Get audio file information."""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM bot_sessions.audio_files WHERE file_id = $1
                """,
                    file_id,
                )

                if row:
                    return AudioFileRecord(
                        file_id=row["file_id"],
                        session_id=row["session_id"],
                        file_path=row["file_path"],
                        file_name=row["file_name"],
                        file_size=row["file_size"],
                        file_format=row["file_format"],
                        duration_seconds=row["duration_seconds"],
                        sample_rate=row["sample_rate"],
                        channels=row["channels"],
                        chunk_start_time=row["chunk_start_time"],
                        chunk_end_time=row["chunk_end_time"],
                        audio_quality_score=row["audio_quality_score"],
                        processing_status=row["processing_status"],
                        file_hash=row["file_hash"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                return None

        except Exception as e:
            logger.error(f"Error getting audio file info: {e}")
            return None

    async def list_session_audio_files(self, session_id: str) -> list[AudioFileRecord]:
        """List all audio files for a session."""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM bot_sessions.audio_files
                    WHERE session_id = $1
                    ORDER BY created_at ASC
                """,
                    session_id,
                )

                return [
                    AudioFileRecord(
                        file_id=row["file_id"],
                        session_id=row["session_id"],
                        file_path=row["file_path"],
                        file_name=row["file_name"],
                        file_size=row["file_size"],
                        file_format=row["file_format"],
                        duration_seconds=row["duration_seconds"],
                        sample_rate=row["sample_rate"],
                        channels=row["channels"],
                        chunk_start_time=row["chunk_start_time"],
                        chunk_end_time=row["chunk_end_time"],
                        audio_quality_score=row["audio_quality_score"],
                        processing_status=row["processing_status"],
                        file_hash=row["file_hash"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Error listing session audio files: {e}")
            return []


class TranscriptManager:
    """Manages transcript records from all sources."""

    def __init__(self, db_pool):
        self.db_pool = db_pool

    async def store_transcript(
        self,
        session_id: str,
        source_type: str,
        transcript_text: str,
        language_code: str,
        start_timestamp: float,
        end_timestamp: float,
        speaker_info: dict | None = None,
        audio_file_id: str | None = None,
        processing_metadata: dict | None = None,
    ) -> str:
        """Store transcript record."""
        try:
            transcript_id = f"transcript_{uuid.uuid4().hex}"

            speaker_id = speaker_info.get("speaker_id") if speaker_info else None
            speaker_name = speaker_info.get("speaker_name") if speaker_info else None
            confidence_score = (
                processing_metadata.get("confidence_score") if processing_metadata else None
            )
            segment_index = (
                processing_metadata.get("segment_index", 0) if processing_metadata else 0
            )
            google_entry_id = (
                processing_metadata.get("google_transcript_entry_id")
                if processing_metadata
                else None
            )

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bot_sessions.transcripts (
                        transcript_id, session_id, source_type, transcript_text,
                        language_code, start_timestamp, end_timestamp, speaker_id,
                        speaker_name, confidence_score, segment_index, audio_file_id,
                        google_transcript_entry_id, processing_metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                    transcript_id,
                    session_id,
                    source_type,
                    transcript_text,
                    language_code,
                    start_timestamp,
                    end_timestamp,
                    speaker_id,
                    speaker_name,
                    confidence_score,
                    segment_index,
                    audio_file_id,
                    google_entry_id,
                    json.dumps(processing_metadata or {}),
                    datetime.now(UTC).replace(tzinfo=None),
                    datetime.now(UTC).replace(tzinfo=None),
                )

            logger.debug(f"Stored transcript: {transcript_id} ({source_type})")
            return transcript_id

        except Exception as e:
            logger.error(f"Error storing transcript: {e}")
            return None

    async def get_session_transcripts(
        self, session_id: str, source_type: str | None = None
    ) -> list[TranscriptRecord]:
        """Get transcripts for a session."""
        try:
            query = """
                SELECT * FROM bot_sessions.transcripts
                WHERE session_id = $1
            """
            params = [session_id]

            if source_type:
                query += " AND source_type = $2"
                params.append(source_type)

            query += " ORDER BY start_timestamp ASC"

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

                return [
                    TranscriptRecord(
                        transcript_id=row["transcript_id"],
                        session_id=row["session_id"],
                        source_type=row["source_type"],
                        transcript_text=row["transcript_text"],
                        language_code=row["language_code"],
                        start_timestamp=row["start_timestamp"],
                        end_timestamp=row["end_timestamp"],
                        speaker_id=row["speaker_id"],
                        speaker_name=row["speaker_name"],
                        confidence_score=row["confidence_score"],
                        segment_index=row["segment_index"],
                        audio_file_id=row["audio_file_id"],
                        google_transcript_entry_id=row["google_transcript_entry_id"],
                        processing_metadata=json.loads(row["processing_metadata"])
                        if row["processing_metadata"]
                        else {},
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Error getting session transcripts: {e}")
            return []

    async def get_transcript_by_timerange(
        self, session_id: str, start_time: float, end_time: float
    ) -> list[TranscriptRecord]:
        """Get transcripts within a time range."""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM bot_sessions.transcripts
                    WHERE session_id = $1
                    AND start_timestamp >= $2
                    AND end_timestamp <= $3
                    ORDER BY start_timestamp ASC
                """,
                    session_id,
                    start_time,
                    end_time,
                )

                return [
                    TranscriptRecord(
                        transcript_id=row["transcript_id"],
                        session_id=row["session_id"],
                        source_type=row["source_type"],
                        transcript_text=row["transcript_text"],
                        language_code=row["language_code"],
                        start_timestamp=row["start_timestamp"],
                        end_timestamp=row["end_timestamp"],
                        speaker_id=row["speaker_id"],
                        speaker_name=row["speaker_name"],
                        confidence_score=row["confidence_score"],
                        segment_index=row["segment_index"],
                        audio_file_id=row["audio_file_id"],
                        google_transcript_entry_id=row["google_transcript_entry_id"],
                        processing_metadata=json.loads(row["processing_metadata"])
                        if row["processing_metadata"]
                        else {},
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Error getting transcripts by time range: {e}")
            return []


class TranslationManager:
    """Manages translation records."""

    def __init__(self, db_pool):
        self.db_pool = db_pool

    async def store_translation(
        self,
        session_id: str,
        source_transcript_id: str,
        translated_text: str,
        source_language: str,
        target_language: str,
        translation_service: str,
        speaker_info: dict | None = None,
        timing_info: dict | None = None,
        processing_metadata: dict | None = None,
    ) -> str:
        """Store translation record."""
        try:
            translation_id = f"translation_{uuid.uuid4().hex}"

            speaker_id = speaker_info.get("speaker_id") if speaker_info else None
            speaker_name = speaker_info.get("speaker_name") if speaker_info else None
            confidence = (
                processing_metadata.get("translation_confidence") if processing_metadata else None
            )
            start_timestamp = timing_info.get("start_timestamp") if timing_info else 0.0
            end_timestamp = timing_info.get("end_timestamp") if timing_info else 0.0

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bot_sessions.translations (
                        translation_id, session_id, source_transcript_id, translated_text,
                        source_language, target_language, translation_confidence,
                        translation_service, speaker_id, speaker_name, start_timestamp,
                        end_timestamp, processing_metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                    translation_id,
                    session_id,
                    source_transcript_id,
                    translated_text,
                    source_language,
                    target_language,
                    confidence,
                    translation_service,
                    speaker_id,
                    speaker_name,
                    start_timestamp,
                    end_timestamp,
                    json.dumps(processing_metadata or {}),
                    datetime.now(UTC).replace(tzinfo=None),
                    datetime.now(UTC).replace(tzinfo=None),
                )

            logger.debug(
                f"Stored translation: {translation_id} ({source_language} -> {target_language})"
            )
            return translation_id

        except Exception as e:
            logger.error(f"Error storing translation: {e}")
            return None

    async def get_session_translations(
        self, session_id: str, target_language: str | None = None
    ) -> list[TranslationRecord]:
        """Get translations for a session."""
        try:
            query = """
                SELECT * FROM bot_sessions.translations
                WHERE session_id = $1
            """
            params = [session_id]

            if target_language:
                query += " AND target_language = $2"
                params.append(target_language)

            query += " ORDER BY start_timestamp ASC"

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

                return [
                    TranslationRecord(
                        translation_id=row["translation_id"],
                        session_id=row["session_id"],
                        source_transcript_id=row["source_transcript_id"],
                        translated_text=row["translated_text"],
                        source_language=row["source_language"],
                        target_language=row["target_language"],
                        translation_confidence=row["translation_confidence"],
                        translation_service=row["translation_service"],
                        speaker_id=row["speaker_id"],
                        speaker_name=row["speaker_name"],
                        start_timestamp=row["start_timestamp"],
                        end_timestamp=row["end_timestamp"],
                        processing_metadata=json.loads(row["processing_metadata"])
                        if row["processing_metadata"]
                        else {},
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Error getting session translations: {e}")
            return []


class CorrelationManager:
    """Manages time correlation records."""

    def __init__(self, db_pool):
        self.db_pool = db_pool

    async def store_correlation(
        self,
        session_id: str,
        google_transcript_id: str | None = None,
        inhouse_transcript_id: str | None = None,
        correlation_confidence: float = 0.0,
        timing_offset: float = 0.0,
        correlation_type: str = "unknown",
        correlation_method: str = "unknown",
        speaker_id: str | None = None,
        start_timestamp: float = 0.0,
        end_timestamp: float = 0.0,
        correlation_metadata: dict | None = None,
    ) -> str:
        """Store correlation record."""
        try:
            correlation_id = f"correlation_{uuid.uuid4().hex}"

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bot_sessions.correlations (
                        correlation_id, session_id, google_transcript_id, inhouse_transcript_id,
                        correlation_confidence, timing_offset, correlation_type, correlation_method,
                        speaker_id, start_timestamp, end_timestamp, correlation_metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    correlation_id,
                    session_id,
                    google_transcript_id,
                    inhouse_transcript_id,
                    correlation_confidence,
                    timing_offset,
                    correlation_type,
                    correlation_method,
                    speaker_id,
                    start_timestamp,
                    end_timestamp,
                    json.dumps(correlation_metadata or {}),
                    datetime.now(UTC).replace(tzinfo=None),
                )

            logger.debug(f"Stored correlation: {correlation_id} ({correlation_type})")
            return correlation_id

        except Exception as e:
            logger.error(f"Error storing correlation: {e}")
            return None

    async def get_session_correlations(self, session_id: str) -> list[CorrelationRecord]:
        """Get correlations for a session."""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM bot_sessions.correlations
                    WHERE session_id = $1
                    ORDER BY start_timestamp ASC
                """,
                    session_id,
                )

                return [
                    CorrelationRecord(
                        correlation_id=row["correlation_id"],
                        session_id=row["session_id"],
                        google_transcript_id=row["google_transcript_id"],
                        inhouse_transcript_id=row["inhouse_transcript_id"],
                        correlation_confidence=row["correlation_confidence"],
                        timing_offset=row["timing_offset"],
                        correlation_type=row["correlation_type"],
                        correlation_method=row["correlation_method"],
                        speaker_id=row["speaker_id"],
                        start_timestamp=row["start_timestamp"],
                        end_timestamp=row["end_timestamp"],
                        correlation_metadata=json.loads(row["correlation_metadata"])
                        if row["correlation_metadata"]
                        else {},
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Error getting session correlations: {e}")
            return []


class BotSessionDatabaseManager:
    """
    Comprehensive database manager for bot sessions.

    Central manager that coordinates all aspects of bot session data including
    audio files, transcripts, translations, and correlations.
    """

    def __init__(self, config: DatabaseConfig, audio_storage_path: str):
        self.config = config
        self.audio_storage_path = audio_storage_path
        self.db_pool = None

        # Component managers
        self.audio_manager = None
        self.transcript_manager = None
        self.translation_manager = None
        self.correlation_manager = None

        # Statistics
        self.total_sessions = 0
        self.total_audio_files = 0
        self.total_transcripts = 0
        self.total_translations = 0
        self.total_correlations = 0

    async def initialize(self) -> bool:
        """Initialize database connection pool with production-ready settings."""
        try:
            # Create database connection pool with comprehensive configuration
            self.db_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                # Pool size limits
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                # Timeout configuration (prevents connection exhaustion)
                timeout=self.config.connection_timeout,  # Connection acquisition timeout
                command_timeout=self.config.command_timeout,  # Individual command timeout
                # Connection lifecycle management
                max_queries=self.config.max_queries,  # Recycle connection after N queries
                max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
            )

            logger.info(
                f"Database connection pool initialized: "
                f"min={self.config.min_connections}, max={self.config.max_connections}, "
                f"conn_timeout={self.config.connection_timeout}s, "
                f"cmd_timeout={self.config.command_timeout}s"
            )

            # Initialize component managers
            self.audio_manager = AudioFileManager(self.audio_storage_path, self.db_pool)
            self.transcript_manager = TranscriptManager(self.db_pool)
            self.translation_manager = TranslationManager(self.db_pool)
            self.correlation_manager = CorrelationManager(self.db_pool)

            # Verify database tables exist
            await self._verify_database_schema()

            # Update statistics
            await self._update_statistics()

            logger.info("Bot session database manager initialized successfully")
            logger.info(f"  Sessions: {self.total_sessions}")
            logger.info(f"  Audio files: {self.total_audio_files}")
            logger.info(f"  Transcripts: {self.total_transcripts}")
            logger.info(f"  Translations: {self.total_translations}")
            logger.info(f"  Correlations: {self.total_correlations}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize bot session database manager: {e}")
            return False

    async def close(self):
        """Close database connections."""
        if self.db_pool:
            await self.db_pool.close()

    async def create_bot_session(self, session_data: dict[str, Any]) -> str:
        """Create a new bot session record."""
        try:
            session_id = session_data.get("session_id") or f"session_{uuid.uuid4().hex}"

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bot_sessions.sessions (
                        session_id, bot_id, meeting_id, meeting_title, meeting_uri,
                        google_meet_space_id, conference_record_id, status, start_time,
                        participant_count, target_languages, session_metadata,
                        performance_stats, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                    session_id,
                    session_data.get("bot_id"),
                    session_data.get("meeting_id"),
                    session_data.get("meeting_title"),
                    session_data.get("meeting_uri"),
                    session_data.get("google_meet_space_id"),
                    session_data.get("conference_record_id"),
                    session_data.get("status", "spawning"),
                    session_data.get("start_time", datetime.now(UTC).replace(tzinfo=None)),
                    session_data.get("participant_count", 0),
                    json.dumps(session_data.get("target_languages", [])),
                    json.dumps(session_data.get("session_metadata", {})),
                    json.dumps(session_data.get("performance_stats", {})),
                    datetime.now(UTC).replace(tzinfo=None),
                    datetime.now(UTC).replace(tzinfo=None),
                )

            self.total_sessions += 1
            logger.info(f"Created bot session: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Error creating bot session: {e}")
            return None

    async def update_bot_session(self, session_id: str, updates: dict[str, Any]) -> bool:
        """Update bot session record."""
        try:
            # Build dynamic UPDATE query
            set_clauses = []
            params = []
            param_count = 1

            for key, value in updates.items():
                if key in ["target_languages", "session_metadata", "performance_stats"]:
                    value = json.dumps(value)
                set_clauses.append(f"{key} = ${param_count}")
                params.append(value)
                param_count += 1

            set_clauses.append(f"updated_at = ${param_count}")
            params.append(datetime.now(UTC).replace(tzinfo=None))
            params.append(session_id)  # For WHERE clause

            query = f"""
                UPDATE bot_sessions.sessions
                SET {", ".join(set_clauses)}
                WHERE session_id = ${param_count + 1}
            """

            async with self.db_pool.acquire() as conn:
                await conn.execute(query, *params)

            logger.debug(f"Updated bot session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating bot session: {e}")
            return False

    async def get_bot_session(self, session_id: str) -> BotSessionRecord | None:
        """Get bot session record."""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM bot_sessions.sessions WHERE session_id = $1
                """,
                    session_id,
                )

                if row:
                    return BotSessionRecord(
                        session_id=row["session_id"],
                        bot_id=row["bot_id"],
                        meeting_id=row["meeting_id"],
                        meeting_title=row["meeting_title"],
                        meeting_uri=row["meeting_uri"],
                        google_meet_space_id=row["google_meet_space_id"],
                        conference_record_id=row["conference_record_id"],
                        status=row["status"],
                        start_time=row["start_time"],
                        end_time=row["end_time"],
                        participant_count=row["participant_count"],
                        target_languages=json.loads(row["target_languages"])
                        if row["target_languages"]
                        else [],
                        session_metadata=json.loads(row["session_metadata"])
                        if row["session_metadata"]
                        else {},
                        performance_stats=json.loads(row["performance_stats"])
                        if row["performance_stats"]
                        else {},
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                return None

        except Exception as e:
            logger.error(f"Error getting bot session: {e}")
            return None

    async def get_comprehensive_session_data(self, session_id: str) -> dict[str, Any]:
        """Get comprehensive session data including all related records."""
        try:
            # Get session record
            session = await self.get_bot_session(session_id)
            if not session:
                return None

            # Get all related data
            audio_files = await self.audio_manager.list_session_audio_files(session_id)
            transcripts = await self.transcript_manager.get_session_transcripts(session_id)
            translations = await self.translation_manager.get_session_translations(session_id)
            correlations = await self.correlation_manager.get_session_correlations(session_id)

            # Organize transcripts by source
            google_transcripts = [t for t in transcripts if t.source_type == "google_meet"]
            inhouse_transcripts = [t for t in transcripts if t.source_type == "whisper_service"]

            # Organize translations by language
            translations_by_language = {}
            for translation in translations:
                lang = translation.target_language
                if lang not in translations_by_language:
                    translations_by_language[lang] = []
                translations_by_language[lang].append(translation)

            return {
                "session": asdict(session),
                "audio_files": [asdict(af) for af in audio_files],
                "transcripts": {
                    "google_meet": [asdict(t) for t in google_transcripts],
                    "inhouse": [asdict(t) for t in inhouse_transcripts],
                    "total_count": len(transcripts),
                },
                "translations": {
                    "by_language": {
                        k: [asdict(t) for t in v] for k, v in translations_by_language.items()
                    },
                    "total_count": len(translations),
                },
                "correlations": [asdict(c) for c in correlations],
                "statistics": {
                    "audio_files_count": len(audio_files),
                    "total_audio_duration": sum(af.duration_seconds or 0 for af in audio_files),
                    "transcripts_count": len(transcripts),
                    "translations_count": len(translations),
                    "correlations_count": len(correlations),
                    "languages_detected": list({t.language_code for t in transcripts}),
                    "target_languages": list(translations_by_language.keys()),
                },
            }

        except Exception as e:
            logger.error(f"Error getting comprehensive session data: {e}")
            return None

    async def cleanup_session(self, session_id: str, remove_files: bool = False) -> bool:
        """Clean up session data and optionally remove files."""
        try:
            if remove_files:
                # Get audio files to remove
                audio_files = await self.audio_manager.list_session_audio_files(session_id)

                # Remove physical files
                for audio_file in audio_files:
                    try:
                        file_path = Path(audio_file.file_path)
                        if file_path.exists():
                            file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove audio file {audio_file.file_path}: {e}")

                # Remove session directory if empty
                session_dir = Path(self.audio_storage_path) / session_id
                if session_dir.exists() and not any(session_dir.iterdir()):
                    session_dir.rmdir()

            # Remove database records (cascading deletes should handle related records)
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM bot_sessions.sessions WHERE session_id = $1",
                    session_id,
                )

            logger.info(f"Cleaned up session: {session_id} (files_removed: {remove_files})")
            return True

        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
            return False

    async def _verify_database_schema(self):
        """Verify that required database tables exist."""
        try:
            async with self.db_pool.acquire() as conn:
                # Check if bot_sessions schema exists
                schema_exists = await conn.fetchval(
                    """
                    SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'bot_sessions')
                """
                )

                if not schema_exists:
                    logger.warning(
                        "bot_sessions schema does not exist - some features may not work"
                    )
                else:
                    logger.debug("Database schema verification passed")

        except Exception as e:
            logger.warning(f"Database schema verification failed: {e}")

    async def _update_statistics(self):
        """Update internal statistics."""
        try:
            async with self.db_pool.acquire() as conn:
                self.total_sessions = (
                    await conn.fetchval("SELECT COUNT(*) FROM bot_sessions.sessions") or 0
                )
                self.total_audio_files = (
                    await conn.fetchval("SELECT COUNT(*) FROM bot_sessions.audio_files") or 0
                )
                self.total_transcripts = (
                    await conn.fetchval("SELECT COUNT(*) FROM bot_sessions.transcripts") or 0
                )
                self.total_translations = (
                    await conn.fetchval("SELECT COUNT(*) FROM bot_sessions.translations") or 0
                )
                self.total_correlations = (
                    await conn.fetchval("SELECT COUNT(*) FROM bot_sessions.correlations") or 0
                )

        except Exception as e:
            logger.warning(f"Failed to update statistics: {e}")

    async def get_database_statistics(self) -> dict[str, Any]:
        """Get comprehensive database statistics."""
        await self._update_statistics()

        try:
            async with self.db_pool.acquire() as conn:
                # Recent activity
                recent_sessions = (
                    await conn.fetchval(
                        """
                    SELECT COUNT(*) FROM bot_sessions.sessions
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """
                    )
                    or 0
                )

                # Active sessions
                active_sessions = (
                    await conn.fetchval(
                        """
                    SELECT COUNT(*) FROM bot_sessions.sessions
                    WHERE status IN ('spawning', 'active', 'paused')
                """
                    )
                    or 0
                )

                # Storage usage
                total_audio_size = (
                    await conn.fetchval(
                        """
                    SELECT COALESCE(SUM(file_size), 0) FROM bot_sessions.audio_files
                """
                    )
                    or 0
                )

                return {
                    "total_sessions": self.total_sessions,
                    "active_sessions": active_sessions,
                    "recent_sessions_24h": recent_sessions,
                    "total_audio_files": self.total_audio_files,
                    "total_transcripts": self.total_transcripts,
                    "total_translations": self.total_translations,
                    "total_correlations": self.total_correlations,
                    "storage_usage_bytes": total_audio_size,
                    "storage_usage_mb": round(total_audio_size / (1024 * 1024), 2),
                }

        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {
                "total_sessions": self.total_sessions,
                "total_audio_files": self.total_audio_files,
                "total_transcripts": self.total_transcripts,
                "total_translations": self.total_translations,
                "total_correlations": self.total_correlations,
                "error": str(e),
            }


# Factory function
def create_bot_session_manager(
    database_config: dict[str, Any], audio_storage_path: str
) -> BotSessionDatabaseManager:
    """Create a bot session database manager."""
    config = DatabaseConfig(**database_config)
    return BotSessionDatabaseManager(config, audio_storage_path)


# Example usage
async def main():
    """Example usage of the bot session database manager."""
    # Configuration
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "livetranslate",
        "username": "postgres",
        "password": "livetranslate",
    }

    # Create manager
    manager = create_bot_session_manager(db_config, "/tmp/livetranslate/audio")

    # Initialize
    success = await manager.initialize()
    if not success:
        print("Failed to initialize database manager")
        return

    # Create session
    session_data = {
        "bot_id": "test-bot-001",
        "meeting_id": "test-meeting-123",
        "meeting_title": "Test Database Integration",
        "status": "active",
        "target_languages": ["en", "es", "fr"],
    }

    session_id = await manager.create_bot_session(session_data)
    print(f"Created session: {session_id}")

    # Store audio file
    audio_data = b"fake audio data for testing"
    file_id = await manager.audio_manager.store_audio_file(
        session_id, audio_data, "wav", {"duration_seconds": 10.5}
    )
    print(f"Stored audio file: {file_id}")

    # Store transcript
    transcript_id = await manager.transcript_manager.store_transcript(
        session_id,
        "whisper_service",
        "Hello everyone welcome to the meeting",
        "en",
        0.0,
        3.5,
        {"speaker_id": "speaker_1", "speaker_name": "John Doe"},
        file_id,
        {"confidence_score": 0.95},
    )
    print(f"Stored transcript: {transcript_id}")

    # Store translation
    translation_id = await manager.translation_manager.store_translation(
        session_id,
        transcript_id,
        "Hola a todos, bienvenidos a la reunión",
        "en",
        "es",
        "translation_service",
        {"speaker_id": "speaker_1", "speaker_name": "John Doe"},
        {"start_timestamp": 0.0, "end_timestamp": 3.5},
        {"translation_confidence": 0.92},
    )
    print(f"Stored translation: {translation_id}")

    # Get comprehensive data
    comprehensive_data = await manager.get_comprehensive_session_data(session_id)
    print(f"Comprehensive data: {json.dumps(comprehensive_data, indent=2, default=str)}")

    # Get statistics
    stats = await manager.get_database_statistics()
    print(f"Database statistics: {json.dumps(stats, indent=2)}")

    # Cleanup
    await manager.cleanup_session(session_id, remove_files=True)
    await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
