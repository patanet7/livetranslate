#!/usr/bin/env python3
"""
Database Adapter for Audio Processing Pipeline - Orchestration Service

Specialized database operations for the centralized audio chunking and processing system.
Provides optimized CRUD operations for all audio-related bot_sessions tables with
proper error handling, connection pooling, and performance optimization.

Tables:
- bot_sessions.audio_files: Audio chunk storage and metadata
- bot_sessions.transcripts: Transcription results with speaker information
- bot_sessions.translations: Translation results with lineage tracking
- bot_sessions.correlations: Speaker correlations and timing alignment
- bot_sessions.events: Processing events for debugging and analytics
- bot_sessions.session_statistics: Real-time session analytics
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncpg

from .models import (
    AudioChunkMetadata,
    SpeakerCorrelation,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)


class DatabaseConnectionPool:
    """Async connection pool manager for database operations."""

    def __init__(self, database_url: str, min_size: int = 5, max_size: int = 20):
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> bool:
        """Initialize the connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=30,
                server_settings={
                    "jit": "off",  # Disable JIT for better performance with many small queries
                    "application_name": "livetranslate_audio_coordinator",
                },
            )
            logger.info(
                f"Database pool initialized: {self.min_size}-{self.max_size} connections"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            return False

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        async with self.pool.acquire() as connection:
            yield connection


class AudioDatabaseAdapter:
    """
    Specialized database adapter for audio processing pipeline operations.
    Optimized for high-throughput audio chunk processing with proper error handling.
    """

    def __init__(self, database_url: str, pool_config: Optional[Dict] = None):
        self.database_url = database_url
        pool_config = pool_config or {}

        self.connection_pool = DatabaseConnectionPool(
            database_url,
            min_size=pool_config.get("min_size", 5),
            max_size=pool_config.get("max_size", 20),
        )

        # Performance tracking
        self.operation_stats = {
            "audio_files_stored": 0,
            "transcripts_stored": 0,
            "translations_stored": 0,
            "correlations_stored": 0,
            "total_operations": 0,
            "failed_operations": 0,
            "average_operation_time": 0.0,
        }

        # Prepared statement cache
        self._prepared_statements = {}

    async def initialize(self) -> bool:
        """Initialize the database adapter and connection pool."""
        try:
            success = await self.connection_pool.initialize()
            if success:
                await self._prepare_statements()
                logger.info("Audio database adapter initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize audio database adapter: {e}")
            return False

    async def close(self):
        """Close the database adapter and connection pool."""
        await self.connection_pool.close()
        logger.info("Audio database adapter closed")

    async def _prepare_statements(self):
        """Prepare frequently used SQL statements for better performance."""
        async with self.connection_pool.get_connection() as conn:
            # Audio files operations
            await conn.prepare("""
                INSERT INTO bot_sessions.audio_files (
                    file_id, session_id, file_path, file_name, file_size, file_format,
                    duration_seconds, sample_rate, channels, chunk_start_time, chunk_end_time,
                    audio_quality_score, processing_status, file_hash, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """)

            # Transcripts operations
            await conn.prepare("""
                INSERT INTO bot_sessions.transcripts (
                    transcript_id, session_id, source_type, transcript_text, language_code,
                    start_timestamp, end_timestamp, speaker_id, speaker_name, confidence_score,
                    segment_index, audio_file_id, processing_metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """)

            # Correlations operations
            await conn.prepare("""
                INSERT INTO bot_sessions.correlations (
                    correlation_id, session_id, google_transcript_id, inhouse_transcript_id,
                    correlation_confidence, timing_offset, correlation_type, correlation_method,
                    speaker_id, start_timestamp, end_timestamp, correlation_metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """)

    async def store_audio_chunk(self, chunk_metadata: AudioChunkMetadata) -> str:
        """
        Store audio chunk metadata in bot_sessions.audio_files.

        Args:
            chunk_metadata: AudioChunkMetadata object with complete chunk information

        Returns:
            str: The chunk_id if successful, None if failed
        """
        # Skip if database is not initialized (e.g., in tests or minimal deployments)
        if not self.connection_pool or not self.connection_pool.pool:
            logger.debug("Database pool not initialized - skipping audio chunk storage")
            return None

        start_time = time.time()

        try:
            async with self.connection_pool.get_connection() as conn:
                # Generate hash if not provided
                file_hash = chunk_metadata.file_hash
                if not file_hash and chunk_metadata.file_path:
                    file_hash = self._generate_file_hash(chunk_metadata.file_path)

                # Prepare metadata JSON
                metadata_json = {
                    **chunk_metadata.chunk_metadata,
                    "chunk_sequence": chunk_metadata.chunk_sequence,
                    "overlap_duration": chunk_metadata.overlap_duration,
                    "overlap_metadata": chunk_metadata.overlap_metadata,
                    "processing_pipeline_version": chunk_metadata.processing_pipeline_version,
                    "source_type": chunk_metadata.source_type.value,
                    "created_at": chunk_metadata.created_at.isoformat(),
                }

                await conn.execute(
                    """
                    INSERT INTO bot_sessions.audio_files (
                        file_id, session_id, file_path, file_name, file_size, file_format,
                        duration_seconds, sample_rate, channels, chunk_start_time, chunk_end_time,
                        audio_quality_score, processing_status, file_hash, metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                    chunk_metadata.chunk_id,
                    chunk_metadata.session_id,
                    chunk_metadata.file_path,
                    chunk_metadata.file_name,
                    chunk_metadata.file_size,
                    chunk_metadata.file_format.value,
                    chunk_metadata.duration_seconds,
                    chunk_metadata.sample_rate,
                    chunk_metadata.channels,
                    chunk_metadata.chunk_start_time,
                    chunk_metadata.chunk_end_time,
                    chunk_metadata.audio_quality_score,
                    chunk_metadata.processing_status.value,
                    file_hash,
                    json.dumps(metadata_json),
                    chunk_metadata.created_at,
                    chunk_metadata.updated_at,
                )

                # Log event
                await self._log_event(
                    conn,
                    chunk_metadata.session_id,
                    "audio_chunk_stored",
                    "storage",
                    {
                        "chunk_id": chunk_metadata.chunk_id,
                        "file_size": chunk_metadata.file_size,
                        "duration": chunk_metadata.duration_seconds,
                        "quality_score": chunk_metadata.audio_quality_score,
                    },
                )

                # Update statistics
                self.operation_stats["audio_files_stored"] += 1
                self._update_operation_stats(start_time)

                logger.debug(f"Stored audio chunk: {chunk_metadata.chunk_id}")
                return chunk_metadata.chunk_id

        except Exception as e:
            logger.error(f"Failed to store audio chunk {chunk_metadata.chunk_id}: {e}")
            self.operation_stats["failed_operations"] += 1
            return None

    async def store_transcript(
        self,
        session_id: str,
        transcript_data: Dict[str, Any],
        audio_file_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store transcript in bot_sessions.transcripts with speaker correlation support.

        Args:
            session_id: Bot session identifier
            transcript_data: Transcript information from whisper service
            audio_file_id: Associated audio file ID for correlation

        Returns:
            str: The transcript_id if successful, None if failed
        """
        # Skip if database is not initialized (e.g., in tests or minimal deployments)
        if not self.connection_pool or not self.connection_pool.pool:
            logger.debug("Database pool not initialized - skipping transcript storage")
            return None

        start_time = time.time()
        transcript_id = str(uuid.uuid4())

        try:
            async with self.connection_pool.get_connection() as conn:
                # Extract speaker information
                speaker_info = transcript_data.get("speaker_info", {})
                speaker_id = speaker_info.get("speaker_id") if speaker_info else None
                speaker_name = (
                    speaker_info.get("speaker_name") if speaker_info else None
                )

                # Prepare processing metadata
                processing_metadata = {
                    "confidence_score": transcript_data.get("confidence", 0.0),
                    "processing_time": transcript_data.get("processing_time", 0.0),
                    "chunk_id": transcript_data.get("chunk_id"),
                    "whisper_speaker_id": speaker_id,
                    "speaker_correlation_confidence": transcript_data.get(
                        "speaker_correlation_confidence"
                    ),
                    **transcript_data.get("metadata", {}),
                }

                await conn.execute(
                    """
                    INSERT INTO bot_sessions.transcripts (
                        transcript_id, session_id, source_type, transcript_text, language_code,
                        start_timestamp, end_timestamp, speaker_id, speaker_name, confidence_score,
                        segment_index, audio_file_id, processing_metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                    transcript_id,
                    session_id,
                    transcript_data.get("source_type", "whisper_service"),
                    transcript_data["text"],
                    transcript_data.get("language", "en"),
                    transcript_data["start_timestamp"],
                    transcript_data["end_timestamp"],
                    speaker_id,
                    speaker_name,
                    transcript_data.get("confidence", 0.0),
                    transcript_data.get("segment_index", 0),
                    audio_file_id,
                    json.dumps(processing_metadata),
                    datetime.utcnow(),
                    datetime.utcnow(),
                )

                # Log event
                await self._log_event(
                    conn,
                    session_id,
                    "transcript_stored",
                    "transcription",
                    {
                        "transcript_id": transcript_id,
                        "text_length": len(transcript_data["text"]),
                        "language": transcript_data.get("language", "en"),
                        "speaker_id": speaker_id,
                        "confidence": transcript_data.get("confidence", 0.0),
                    },
                )

                # Update statistics
                self.operation_stats["transcripts_stored"] += 1
                self._update_operation_stats(start_time)

                logger.debug(f"Stored transcript: {transcript_id}")
                return transcript_id

        except Exception as e:
            logger.error(f"Failed to store transcript for session {session_id}: {e}")
            self.operation_stats["failed_operations"] += 1
            return None

    async def store_translation(
        self,
        session_id: str,
        source_transcript_id: str,
        translation_data: Dict[str, Any],
    ) -> Optional[str]:
        """
        Store translation in bot_sessions.translations with full lineage tracking.

        Args:
            session_id: Bot session identifier
            source_transcript_id: Source transcript that was translated
            translation_data: Translation information from translation service

        Returns:
            str: The translation_id if successful, None if failed
        """
        # Skip if database is not initialized (e.g., in tests or minimal deployments)
        if not self.connection_pool or not self.connection_pool.pool:
            logger.debug("Database pool not initialized - skipping translation storage")
            return None

        start_time = time.time()
        translation_id = str(uuid.uuid4())

        try:
            async with self.connection_pool.get_connection() as conn:
                # Prepare processing metadata with lineage
                processing_metadata = {
                    "translation_service": translation_data.get(
                        "translation_service", "unknown"
                    ),
                    "translation_model": translation_data.get("model", "unknown"),
                    "processing_time": translation_data.get("processing_time", 0.0),
                    "chunk_lineage": translation_data.get("chunk_lineage", []),
                    "processing_pipeline_version": translation_data.get(
                        "processing_pipeline_version", "1.0"
                    ),
                    **translation_data.get("metadata", {}),
                }

                await conn.execute(
                    """
                    INSERT INTO bot_sessions.translations (
                        translation_id, session_id, source_transcript_id, translated_text,
                        source_language, target_language, translation_confidence, translation_service,
                        speaker_id, speaker_name, start_timestamp, end_timestamp,
                        processing_metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                    translation_id,
                    session_id,
                    source_transcript_id,
                    translation_data["translated_text"],
                    translation_data["source_language"],
                    translation_data["target_language"],
                    translation_data.get("confidence", 0.0),
                    translation_data.get("translation_service", "unknown"),
                    translation_data.get("speaker_id"),
                    translation_data.get("speaker_name"),
                    translation_data["start_timestamp"],
                    translation_data["end_timestamp"],
                    json.dumps(processing_metadata),
                    datetime.utcnow(),
                    datetime.utcnow(),
                )

                # Log event
                await self._log_event(
                    conn,
                    session_id,
                    "translation_stored",
                    "translation",
                    {
                        "translation_id": translation_id,
                        "source_transcript_id": source_transcript_id,
                        "source_language": translation_data["source_language"],
                        "target_language": translation_data["target_language"],
                        "confidence": translation_data.get("confidence", 0.0),
                        "text_length": len(translation_data["translated_text"]),
                    },
                )

                # Update statistics
                self.operation_stats["translations_stored"] += 1
                self._update_operation_stats(start_time)

                logger.debug(f"Stored translation: {translation_id}")
                return translation_id

        except Exception as e:
            logger.error(f"Failed to store translation for session {session_id}: {e}")
            self.operation_stats["failed_operations"] += 1
            return None

    async def store_speaker_correlation(
        self, correlation: SpeakerCorrelation
    ) -> Optional[str]:
        """
        Store speaker correlation in bot_sessions.correlations.

        Args:
            correlation: SpeakerCorrelation object with complete correlation data

        Returns:
            str: The correlation_id if successful, None if failed
        """
        start_time = time.time()

        try:
            async with self.connection_pool.get_connection() as conn:
                # Prepare correlation metadata
                correlation_metadata = {
                    "text_similarity_score": correlation.text_similarity_score,
                    "temporal_alignment_score": correlation.temporal_alignment_score,
                    "historical_pattern_score": correlation.historical_pattern_score,
                    **correlation.correlation_metadata,
                }

                await conn.execute(
                    """
                    INSERT INTO bot_sessions.correlations (
                        correlation_id, session_id, google_transcript_id, inhouse_transcript_id,
                        correlation_confidence, timing_offset, correlation_type, correlation_method,
                        speaker_id, start_timestamp, end_timestamp, correlation_metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    correlation.correlation_id,
                    correlation.session_id,
                    correlation.google_meet_speaker_id,  # Note: mapping to google_transcript_id
                    correlation.whisper_speaker_id,  # Note: mapping to inhouse_transcript_id
                    correlation.correlation_confidence,
                    correlation.timing_offset,
                    correlation.correlation_type.value,
                    correlation.correlation_method,
                    correlation.whisper_speaker_id,
                    correlation.start_timestamp,
                    correlation.end_timestamp,
                    json.dumps(correlation_metadata),
                    correlation.created_at,
                )

                # Log event
                await self._log_event(
                    conn,
                    correlation.session_id,
                    "speaker_correlation_stored",
                    "correlation",
                    {
                        "correlation_id": correlation.correlation_id,
                        "whisper_speaker_id": correlation.whisper_speaker_id,
                        "google_meet_speaker_id": correlation.google_meet_speaker_id,
                        "google_meet_speaker_name": correlation.google_meet_speaker_name,
                        "confidence": correlation.correlation_confidence,
                        "correlation_type": correlation.correlation_type.value,
                    },
                )

                # Update statistics
                self.operation_stats["correlations_stored"] += 1
                self._update_operation_stats(start_time)

                logger.debug(
                    f"Stored speaker correlation: {correlation.correlation_id}"
                )
                return correlation.correlation_id

        except Exception as e:
            logger.error(
                f"Failed to store speaker correlation {correlation.correlation_id}: {e}"
            )
            self.operation_stats["failed_operations"] += 1
            return None

    async def get_session_audio_chunks(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: str = "chunk_start_time",
    ) -> List[Dict[str, Any]]:
        """Get audio chunks for a session with optional pagination."""
        try:
            async with self.connection_pool.get_connection() as conn:
                query = f"""
                    SELECT * FROM bot_sessions.audio_files 
                    WHERE session_id = $1 
                    ORDER BY {order_by}
                """

                params = [session_id]
                if limit:
                    query += f" LIMIT ${len(params) + 1}"
                    params.append(limit)
                if offset:
                    query += f" OFFSET ${len(params) + 1}"
                    params.append(offset)

                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get audio chunks for session {session_id}: {e}")
            return []

    async def get_session_transcripts(
        self,
        session_id: str,
        source_type: Optional[str] = None,
        speaker_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get transcripts for a session with optional filtering."""
        try:
            async with self.connection_pool.get_connection() as conn:
                query = "SELECT * FROM bot_sessions.transcripts WHERE session_id = $1"
                params = [session_id]

                if source_type:
                    query += f" AND source_type = ${len(params) + 1}"
                    params.append(source_type)

                if speaker_id:
                    query += f" AND speaker_id = ${len(params) + 1}"
                    params.append(speaker_id)

                query += " ORDER BY start_timestamp"

                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get transcripts for session {session_id}: {e}")
            return []

    async def get_speaker_correlations(
        self, session_id: str, min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get speaker correlations for a session."""
        try:
            async with self.connection_pool.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM bot_sessions.correlations 
                    WHERE session_id = $1 AND correlation_confidence >= $2
                    ORDER BY start_timestamp
                """,
                    session_id,
                    min_confidence,
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get correlations for session {session_id}: {e}")
            return []

    async def update_chunk_processing_status(
        self,
        chunk_id: str,
        status: ProcessingStatus,
        processing_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update processing status for an audio chunk."""
        try:
            async with self.connection_pool.get_connection() as conn:
                if processing_metadata:
                    await conn.execute(
                        """
                        UPDATE bot_sessions.audio_files 
                        SET processing_status = $1, metadata = metadata || $2, updated_at = $3
                        WHERE file_id = $4
                    """,
                        status.value,
                        json.dumps(processing_metadata),
                        datetime.utcnow(),
                        chunk_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE bot_sessions.audio_files 
                        SET processing_status = $1, updated_at = $2
                        WHERE file_id = $3
                    """,
                        status.value,
                        datetime.utcnow(),
                        chunk_id,
                    )

                return True

        except Exception as e:
            logger.error(f"Failed to update chunk status {chunk_id}: {e}")
            return False

    async def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a session."""
        try:
            async with self.connection_pool.get_connection() as conn:
                # Get session overview
                session_overview = await conn.fetchrow(
                    """
                    SELECT * FROM bot_sessions.session_overview 
                    WHERE session_id = $1
                """,
                    session_id,
                )

                if not session_overview:
                    return {}

                # Get detailed statistics
                analytics = dict(session_overview)

                # Add chunk processing statistics
                chunk_stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) as total_chunks,
                        AVG(audio_quality_score) as avg_quality,
                        SUM(duration_seconds) as total_audio_duration,
                        COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_chunks,
                        COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_chunks
                    FROM bot_sessions.audio_files 
                    WHERE session_id = $1
                """,
                    session_id,
                )

                analytics["chunk_statistics"] = dict(chunk_stats) if chunk_stats else {}

                # Add correlation statistics
                correlation_stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) as total_correlations,
                        AVG(correlation_confidence) as avg_confidence,
                        COUNT(CASE WHEN correlation_confidence >= 0.8 THEN 1 END) as high_confidence_correlations
                    FROM bot_sessions.correlations 
                    WHERE session_id = $1
                """,
                    session_id,
                )

                analytics["correlation_statistics"] = (
                    dict(correlation_stats) if correlation_stats else {}
                )

                return analytics

        except Exception as e:
            logger.error(f"Failed to get analytics for session {session_id}: {e}")
            return {}

    async def cleanup_old_data(self, retention_days: int = 30) -> Dict[str, int]:
        """Clean up old audio data based on retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        cleanup_stats = {
            "audio_files": 0,
            "transcripts": 0,
            "translations": 0,
            "correlations": 0,
        }

        try:
            async with self.connection_pool.get_connection() as conn:
                # Cleanup old sessions and cascade
                result = await conn.execute(
                    """
                    DELETE FROM bot_sessions.sessions 
                    WHERE created_at < $1 AND status IN ('ended', 'error')
                """,
                    cutoff_date,
                )

                cleanup_stats["sessions"] = int(result.split()[-1])

                logger.info(f"Cleaned up {cleanup_stats['sessions']} old sessions")
                return cleanup_stats

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return cleanup_stats

    async def _log_event(
        self,
        conn: asyncpg.Connection,
        session_id: str,
        event_type: str,
        event_subtype: str,
        event_data: Dict[str, Any],
        severity: str = "info",
    ):
        """Log an event to bot_sessions.events."""
        try:
            await conn.execute(
                """
                INSERT INTO bot_sessions.events (
                    session_id, event_type, event_subtype, event_data, 
                    source_component, severity, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                session_id,
                event_type,
                event_subtype,
                json.dumps(event_data),
                "audio_coordinator",
                severity,
                datetime.utcnow(),
            )
        except Exception as e:
            logger.warning(f"Failed to log event: {e}")

    def _update_operation_stats(self, start_time: float):
        """Update internal operation statistics."""
        operation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.operation_stats["total_operations"] += 1

        # Update average operation time
        total_ops = self.operation_stats["total_operations"]
        current_avg = self.operation_stats["average_operation_time"]
        self.operation_stats["average_operation_time"] = (
            current_avg * (total_ops - 1) + operation_time
        ) / total_ops

    def _generate_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash for a file."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate hash for {file_path}: {e}")
            return ""

    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get database operation statistics."""
        return {
            **self.operation_stats,
            "success_rate": (
                1.0
                - (
                    self.operation_stats["failed_operations"]
                    / max(1, self.operation_stats["total_operations"])
                )
            )
            if self.operation_stats["total_operations"] > 0
            else 0.0,
        }


# Factory function for creating database adapter
def create_audio_database_adapter(
    database_url: str, pool_config: Optional[Dict] = None
) -> AudioDatabaseAdapter:
    """Create and return an AudioDatabaseAdapter instance."""
    return AudioDatabaseAdapter(database_url, pool_config)


# Example usage and testing
async def main():
    """Example usage of the audio database adapter."""
    import os

    # Database URL from environment
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://postgres:password@localhost:5432/livetranslate"
    )

    # Create adapter
    adapter = create_audio_database_adapter(database_url)

    try:
        # Initialize
        success = await adapter.initialize()
        if not success:
            print("Failed to initialize database adapter")
            return

        # Example audio chunk
        from .models import create_audio_chunk_metadata, SourceType

        chunk = create_audio_chunk_metadata(
            session_id="test-session-123",
            file_path="/data/audio/test_chunk_001.wav",
            file_size=64000,
            duration_seconds=3.0,
            chunk_sequence=1,
            chunk_start_time=0.0,
            source_type=SourceType.BOT_AUDIO,
            audio_quality_score=0.85,
        )

        # Store chunk
        chunk_id = await adapter.store_audio_chunk(chunk)
        print(f"Stored chunk: {chunk_id}")

        # Get session analytics
        analytics = await adapter.get_session_analytics("test-session-123")
        print(f"Session analytics: {analytics}")

        # Get operation statistics
        stats = adapter.get_operation_statistics()
        print(f"Operation statistics: {stats}")

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
