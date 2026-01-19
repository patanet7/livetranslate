#!/usr/bin/env python3
"""
Transcription Data Pipeline

Complete data pipeline for processing audio → transcription → translation flow
with integrated database persistence, real-time streaming support, and speaker
diarization tracking.

Features:
- Audio chunk storage with metadata
- Transcription result processing with speaker labels
- Translation result storage and linking
- Timeline reconstruction and querying
- Speaker-specific data retrieval
- Full-text search capabilities
- Segment continuity tracking
- Real-time streaming support

Integration:
- Uses BotSessionDatabaseManager for persistence
- Handles Whisper service transcriptions with speaker diarization
- Links translations to source transcripts
- Maintains temporal ordering and relationships

Author: LiveTranslate Team
Version: 1.0
"""

import asyncio
import logging
import sys
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.bot_session_manager import (
    BotSessionDatabaseManager,
    TranscriptRecord,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioChunkMetadata:
    """Metadata for audio chunks."""

    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    chunk_start_time: float | None = None
    chunk_end_time: float | None = None
    audio_quality_score: float | None = None
    codec: str | None = None
    bitrate: int | None = None


@dataclass
class TranscriptionResult:
    """Transcription result from Whisper service."""

    text: str
    language: str
    start_time: float
    end_time: float
    speaker: str | None = None  # 'SPEAKER_00', 'SPEAKER_01', etc.
    speaker_name: str | None = None
    confidence: float | None = None
    segment_index: int = 0
    is_final: bool = True
    words: list[dict] | None = None  # Word-level timestamps


@dataclass
class TranslationResult:
    """Translation result from translation service."""

    text: str
    source_language: str
    target_language: str
    speaker: str | None = None
    speaker_name: str | None = None
    confidence: float | None = None
    translation_service: str = "translation_service"
    model_name: str | None = None


@dataclass
class TimelineEntry:
    """Single entry in session timeline."""

    timestamp: float
    duration: float
    entry_type: str  # 'transcript', 'translation', 'audio'
    content: str
    language: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    confidence: float | None = None
    metadata: dict | None = None


@dataclass
class SpeakerStatistics:
    """Statistics for a single speaker."""

    session_id: str
    speaker_id: str
    speaker_name: str | None
    identification_method: str | None
    identification_confidence: float | None
    total_segments: int
    total_speaking_time: float
    average_confidence: float
    languages_translated_to: int
    total_translations: int


class TranscriptionDataPipeline:
    """
    Complete data pipeline for transcription processing.

    Manages the full flow from audio capture through transcription and
    translation, with comprehensive database persistence and querying.
    """

    def __init__(
        self,
        db_manager: BotSessionDatabaseManager,
        enable_speaker_tracking: bool = True,
        enable_segment_continuity: bool = True,
        max_cache_size: int = 1000,
    ):
        """
        Initialize the data pipeline.

        Args:
            db_manager: Database manager instance
            enable_speaker_tracking: Enable speaker identity tracking
            enable_segment_continuity: Enable segment continuity tracking
            max_cache_size: Maximum number of entries in segment cache (LRU eviction)
        """
        self.db_manager = db_manager
        self.enable_speaker_tracking = enable_speaker_tracking
        self.enable_segment_continuity = enable_segment_continuity
        self.max_cache_size = max_cache_size

        # Cache for recent segments (for continuity tracking) with LRU eviction
        self._segment_cache: OrderedDict[str, str] = (
            OrderedDict()
        )  # session_id -> last_transcript_id

        # Cache statistics for monitoring
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_evictions: int = 0

        logger.info("Transcription data pipeline initialized")
        logger.info(f"  Speaker tracking: {enable_speaker_tracking}")
        logger.info(f"  Segment continuity: {enable_segment_continuity}")
        logger.info(f"  Max cache size: {max_cache_size}")

    async def process_audio_chunk(
        self,
        session_id: str,
        audio_bytes: bytes,
        file_format: str = "wav",
        metadata: AudioChunkMetadata | None = None,
    ) -> str | None:
        """
        Process and store an audio chunk.

        Args:
            session_id: Session identifier
            audio_bytes: Raw audio data
            file_format: Audio format (wav, mp3, etc.)
            metadata: Audio chunk metadata

        Returns:
            File ID if successful, None otherwise
        """
        try:
            # Convert metadata to dict
            metadata_dict = {}
            if metadata:
                metadata_dict = {
                    "duration_seconds": metadata.duration_seconds,
                    "sample_rate": metadata.sample_rate,
                    "channels": metadata.channels,
                    "chunk_start_time": metadata.chunk_start_time,
                    "chunk_end_time": metadata.chunk_end_time,
                    "audio_quality_score": metadata.audio_quality_score,
                    "codec": metadata.codec,
                    "bitrate": metadata.bitrate,
                }

            # Store audio file
            file_id = await self.db_manager.audio_manager.store_audio_file(
                session_id=session_id,
                audio_data=audio_bytes,
                file_format=file_format,
                metadata=metadata_dict,
            )

            if file_id:
                logger.debug(
                    f"Stored audio chunk: {file_id} ({len(audio_bytes)} bytes) for session {session_id}"
                )
            else:
                logger.error(f"Failed to store audio chunk for session {session_id}")

            return file_id

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            return None

    async def process_transcription_result(
        self,
        session_id: str,
        file_id: str | None,
        transcription: TranscriptionResult,
        source_type: str = "whisper_service",
    ) -> str | None:
        """
        Process and store a transcription result.

        Args:
            session_id: Session identifier
            file_id: Associated audio file ID (optional)
            transcription: Transcription result from Whisper
            source_type: Source of transcription ('whisper_service', 'google_meet', etc.)

        Returns:
            Transcript ID if successful, None otherwise
        """
        try:
            # Prepare speaker info
            speaker_info = None
            if transcription.speaker or transcription.speaker_name:
                speaker_info = {
                    "speaker_id": transcription.speaker,
                    "speaker_name": transcription.speaker_name,
                }

            # Prepare processing metadata
            processing_metadata = {
                "confidence_score": transcription.confidence,
                "segment_index": transcription.segment_index,
                "is_final": transcription.is_final,
                "source_type": source_type,
            }

            if transcription.words:
                processing_metadata["words"] = transcription.words

            # Store transcript
            transcript_id = await self.db_manager.transcript_manager.store_transcript(
                session_id=session_id,
                source_type=source_type,
                transcript_text=transcription.text,
                language_code=transcription.language,
                start_timestamp=transcription.start_time,
                end_timestamp=transcription.end_time,
                speaker_info=speaker_info,
                audio_file_id=file_id,
                processing_metadata=processing_metadata,
            )

            if transcript_id:
                logger.debug(
                    f"Stored transcript: {transcript_id} (speaker: {transcription.speaker}) for session {session_id}"
                )

                # Handle segment continuity
                if self.enable_segment_continuity:
                    await self._update_segment_continuity(session_id, transcript_id)

                # Handle speaker tracking
                if self.enable_speaker_tracking and transcription.speaker:
                    await self._track_speaker(
                        session_id,
                        transcription.speaker,
                        transcription.speaker_name,
                        source_type,
                    )

            else:
                logger.error(f"Failed to store transcript for session {session_id}")

            return transcript_id

        except Exception as e:
            logger.error(f"Error processing transcription result: {e}", exc_info=True)
            return None

    async def process_translation_result(
        self,
        session_id: str,
        transcript_id: str,
        translation: TranslationResult,
        start_time: float,
        end_time: float,
    ) -> str | None:
        """
        Process and store a translation result.

        Args:
            session_id: Session identifier
            transcript_id: Source transcript ID
            translation: Translation result
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Translation ID if successful, None otherwise
        """
        try:
            # Prepare speaker info
            speaker_info = None
            if translation.speaker or translation.speaker_name:
                speaker_info = {
                    "speaker_id": translation.speaker,
                    "speaker_name": translation.speaker_name,
                }

            # Prepare timing info
            timing_info = {"start_timestamp": start_time, "end_timestamp": end_time}

            # Prepare processing metadata
            processing_metadata = {
                "translation_confidence": translation.confidence,
                "translation_service": translation.translation_service,
                "model_name": translation.model_name,
            }

            # Store translation
            translation_id = await self.db_manager.translation_manager.store_translation(
                session_id=session_id,
                source_transcript_id=transcript_id,
                translated_text=translation.text,
                source_language=translation.source_language,
                target_language=translation.target_language,
                translation_service=translation.translation_service,
                speaker_info=speaker_info,
                timing_info=timing_info,
                processing_metadata=processing_metadata,
            )

            if translation_id:
                logger.debug(
                    f"Stored translation: {translation_id} "
                    f"({translation.source_language} → {translation.target_language}) "
                    f"for session {session_id}"
                )
            else:
                logger.error(f"Failed to store translation for session {session_id}")

            return translation_id

        except Exception as e:
            logger.error(f"Error processing translation result: {e}", exc_info=True)
            return None

    async def get_session_timeline(
        self,
        session_id: str,
        start_time: float | None = None,
        end_time: float | None = None,
        include_translations: bool = True,
        language_filter: str | None = None,
        speaker_filter: str | None = None,
    ) -> list[TimelineEntry]:
        """
        Get complete session timeline.

        Args:
            session_id: Session identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            include_translations: Include translations in timeline
            language_filter: Filter by language code
            speaker_filter: Filter by speaker ID

        Returns:
            List of timeline entries in chronological order
        """
        try:
            timeline = []

            # Get transcripts
            if start_time is not None and end_time is not None:
                transcripts = await self.db_manager.transcript_manager.get_transcript_by_timerange(
                    session_id, start_time, end_time
                )
            else:
                transcripts = await self.db_manager.transcript_manager.get_session_transcripts(
                    session_id
                )

            # Add transcripts to timeline
            for transcript in transcripts:
                # Apply filters
                if language_filter and transcript.language_code != language_filter:
                    continue
                if speaker_filter and transcript.speaker_id != speaker_filter:
                    continue

                timeline.append(
                    TimelineEntry(
                        timestamp=transcript.start_timestamp,
                        duration=transcript.end_timestamp - transcript.start_timestamp,
                        entry_type="transcript",
                        content=transcript.transcript_text,
                        language=transcript.language_code,
                        speaker_id=transcript.speaker_id,
                        speaker_name=transcript.speaker_name,
                        confidence=transcript.confidence_score,
                        metadata={
                            "transcript_id": transcript.transcript_id,
                            "source_type": transcript.source_type,
                            "segment_index": transcript.segment_index,
                        },
                    )
                )

            # Add translations if requested
            if include_translations:
                translations = await self.db_manager.translation_manager.get_session_translations(
                    session_id
                )

                for translation in translations:
                    # Apply filters
                    if language_filter and translation.target_language != language_filter:
                        continue
                    if speaker_filter and translation.speaker_id != speaker_filter:
                        continue
                    # NULL-safe timestamp comparisons - prevent TypeError on NULL timestamps
                    if start_time is not None and translation.start_timestamp is not None:
                        if translation.start_timestamp < start_time:
                            continue
                    if end_time is not None and translation.end_timestamp is not None:
                        if translation.end_timestamp > end_time:
                            continue

                    # NULL-safe duration calculation
                    if (
                        translation.start_timestamp is not None
                        and translation.end_timestamp is not None
                    ):
                        duration = translation.end_timestamp - translation.start_timestamp
                        timestamp = translation.start_timestamp
                    else:
                        duration = 0.0
                        timestamp = translation.start_timestamp or 0.0

                    timeline.append(
                        TimelineEntry(
                            timestamp=timestamp,
                            duration=duration,
                            entry_type="translation",
                            content=translation.translated_text,
                            language=translation.target_language,
                            speaker_id=translation.speaker_id,
                            speaker_name=translation.speaker_name,
                            confidence=translation.translation_confidence,
                            metadata={
                                "translation_id": translation.translation_id,
                                "source_language": translation.source_language,
                                "service": translation.translation_service,
                            },
                        )
                    )

            # Sort by timestamp
            timeline.sort(key=lambda x: x.timestamp)

            logger.debug(f"Retrieved timeline for session {session_id}: {len(timeline)} entries")
            return timeline

        except Exception as e:
            logger.error(f"Error getting session timeline: {e}", exc_info=True)
            return []

    async def get_speaker_timeline(
        self,
        session_id: str,
        speaker_id: str,
        include_translations: bool = True,
    ) -> list[TimelineEntry]:
        """
        Get timeline for a specific speaker.

        Args:
            session_id: Session identifier
            speaker_id: Speaker identifier
            include_translations: Include translations

        Returns:
            List of timeline entries for the speaker
        """
        return await self.get_session_timeline(
            session_id=session_id,
            speaker_filter=speaker_id,
            include_translations=include_translations,
        )

    async def get_speaker_statistics(self, session_id: str) -> list[SpeakerStatistics]:
        """
        Get statistics for all speakers in a session.

        Args:
            session_id: Session identifier

        Returns:
            List of speaker statistics
        """
        try:
            # Query speaker statistics view
            async with self.db_manager.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM bot_sessions.speaker_statistics
                    WHERE session_id = $1
                    ORDER BY total_speaking_time DESC
                    """,
                    session_id,
                )

                statistics = []
                for row in rows:
                    statistics.append(
                        SpeakerStatistics(
                            session_id=row["session_id"],
                            speaker_id=row["speaker_id"],
                            speaker_name=row["speaker_name"],
                            identification_method=row["identification_method"],
                            identification_confidence=row["identification_confidence"],
                            total_segments=row["transcript_segments"],
                            total_speaking_time=row["total_speaking_time"],
                            average_confidence=row["avg_confidence"],
                            languages_translated_to=row["languages_translated_to"],
                            total_translations=row["total_translations"],
                        )
                    )

                logger.debug(
                    f"Retrieved statistics for {len(statistics)} speakers in session {session_id}"
                )
                return statistics

        except Exception as e:
            logger.error(f"Error getting speaker statistics: {e}", exc_info=True)
            return []

    async def search_transcripts(
        self,
        session_id: str,
        query: str,
        language: str | None = None,
        use_fuzzy: bool = True,
    ) -> list[TranscriptRecord]:
        """
        Full-text search across transcripts.

        Args:
            session_id: Session identifier
            query: Search query
            language: Optional language filter
            use_fuzzy: Use fuzzy matching (similarity search)

        Returns:
            List of matching transcript records
        """
        try:
            async with self.db_manager.db_pool.acquire() as conn:
                if use_fuzzy:
                    # Use trigram similarity search
                    sql = """
                        SELECT * FROM bot_sessions.transcripts
                        WHERE session_id = $1
                        AND transcript_text % $2
                    """
                    params = [session_id, query]

                    if language:
                        sql += " AND language_code = $3"
                        params.append(language)

                    sql += " ORDER BY similarity(transcript_text, $2) DESC LIMIT 50"

                else:
                    # Use full-text search
                    sql = """
                        SELECT * FROM bot_sessions.transcripts
                        WHERE session_id = $1
                        AND search_vector @@ plainto_tsquery('english', $2)
                    """
                    params = [session_id, query]

                    if language:
                        sql += " AND language_code = $3"
                        params.append(language)

                    sql += """
                        ORDER BY ts_rank(search_vector, plainto_tsquery('english', $2)) DESC
                        LIMIT 50
                    """

                rows = await conn.fetch(sql, *params)

                results = []
                for row in rows:
                    results.append(
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
                            processing_metadata=row["processing_metadata"],
                            created_at=row["created_at"],
                            updated_at=row["updated_at"],
                        )
                    )

                logger.debug(
                    f"Search for '{query}' in session {session_id}: {len(results)} results"
                )
                return results

        except Exception as e:
            logger.error(f"Error searching transcripts: {e}", exc_info=True)
            return []

    async def _update_segment_continuity(self, session_id: str, transcript_id: str) -> None:
        """
        Update segment continuity links with LRU cache.

        Args:
            session_id: Session identifier
            transcript_id: New transcript ID
        """
        try:
            # Get previous segment from cache
            previous_id = self._segment_cache.get(session_id)

            if previous_id:
                self._cache_hits += 1
                # Update previous segment's next_segment_id
                async with self.db_manager.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE bot_sessions.transcripts
                        SET next_segment_id = $1
                        WHERE transcript_id = $2
                        """,
                        transcript_id,
                        previous_id,
                    )

                    # Update current segment's previous_segment_id
                    await conn.execute(
                        """
                        UPDATE bot_sessions.transcripts
                        SET previous_segment_id = $1
                        WHERE transcript_id = $2
                        """,
                        previous_id,
                        transcript_id,
                    )
            else:
                self._cache_misses += 1

            # Update cache with LRU eviction
            # Move to end if exists (mark as recently used)
            if session_id in self._segment_cache:
                self._segment_cache.move_to_end(session_id)

            self._segment_cache[session_id] = transcript_id

            # Evict oldest entry if cache is full
            if len(self._segment_cache) > self.max_cache_size:
                evicted_key = next(iter(self._segment_cache))
                self._segment_cache.pop(evicted_key)
                self._cache_evictions += 1
                logger.debug(
                    f"Evicted session {evicted_key} from segment cache "
                    f"(size: {len(self._segment_cache)}/{self.max_cache_size})"
                )

        except Exception as e:
            logger.warning(f"Error updating segment continuity: {e}")

    async def clear_session_cache(self, session_id: str) -> None:
        """
        Clear cache entries for a specific session.

        Args:
            session_id: Session identifier to clear from cache
        """
        try:
            if session_id in self._segment_cache:
                self._segment_cache.pop(session_id)
                logger.debug(f"Cleared cache for session {session_id}")
        except Exception as e:
            logger.warning(f"Error clearing session cache: {e}")

    def get_cache_statistics(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache metrics (hits, misses, evictions, size, hit rate)
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self._segment_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_evictions": self._cache_evictions,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
        }

    @asynccontextmanager
    async def transaction(self):
        """
        Async context manager for database transactions.

        Provides atomic transaction support for multi-step operations.
        Automatically commits on success, rolls back on failure.

        Usage:
            async with pipeline.transaction() as conn:
                await conn.execute("INSERT ...")
                await conn.execute("UPDATE ...")
                # Automatic commit on exit

        Yields:
            Database connection with active transaction

        Raises:
            Exception: Re-raises any exception after rolling back transaction
        """
        async with self.db_manager.db_pool.acquire() as conn, conn.transaction():
            yield conn

    async def process_complete_segment(
        self,
        session_id: str,
        audio_bytes: bytes,
        transcription: TranscriptionResult,
        translations: list[TranslationResult],
        audio_metadata: AudioChunkMetadata | None = None,
    ) -> dict[str, Any] | None:
        """
        Process complete segment atomically (audio + transcription + translations).

        All operations succeed or fail together. On failure, all changes are rolled back.

        Args:
            session_id: Session identifier
            audio_bytes: Raw audio data
            transcription: Transcription result
            translations: List of translation results
            audio_metadata: Optional audio metadata

        Returns:
            Dictionary with file_id, transcript_id, and translation_ids if successful,
            None if any step fails
        """
        try:
            async with self.transaction():
                # Step 1: Store audio chunk
                file_id = await self.process_audio_chunk(
                    session_id=session_id,
                    audio_bytes=audio_bytes,
                    file_format="wav",
                    metadata=audio_metadata,
                )

                if not file_id:
                    raise Exception("Failed to store audio chunk")

                # Step 2: Store transcription
                transcript_id = await self.process_transcription_result(
                    session_id=session_id,
                    file_id=file_id,
                    transcription=transcription,
                )

                if not transcript_id:
                    raise Exception("Failed to store transcription")

                # Step 3: Store all translations
                translation_ids = []
                for translation in translations:
                    translation_id = await self.process_translation_result(
                        session_id=session_id,
                        transcript_id=transcript_id,
                        translation=translation,
                        start_time=transcription.start_time,
                        end_time=transcription.end_time,
                    )

                    if not translation_id:
                        raise Exception(
                            f"Failed to store translation to {translation.target_language}"
                        )

                    translation_ids.append(translation_id)

                # All operations succeeded
                logger.info(
                    f"Processed complete segment atomically: "
                    f"audio={file_id}, transcript={transcript_id}, "
                    f"translations={len(translation_ids)}"
                )

                return {
                    "file_id": file_id,
                    "transcript_id": transcript_id,
                    "translation_ids": translation_ids,
                    "session_id": session_id,
                }

        except Exception as e:
            logger.error(
                f"Failed to process complete segment (transaction rolled back): {e}",
                exc_info=True,
            )
            return None

    async def _track_speaker(
        self,
        session_id: str,
        speaker_label: str,
        speaker_name: str | None,
        source: str,
    ) -> None:
        """
        Track speaker identity.

        Args:
            session_id: Session identifier
            speaker_label: Speaker label (SPEAKER_00, etc.)
            speaker_name: Optional speaker name
            source: Source of identification
        """
        try:
            # Check if speaker identity already exists
            async with self.db_manager.db_pool.acquire() as conn:
                existing = await conn.fetchrow(
                    """
                    SELECT identity_id FROM bot_sessions.speaker_identities
                    WHERE session_id = $1 AND speaker_label = $2
                    """,
                    session_id,
                    speaker_label,
                )

                if not existing:
                    # Create new speaker identity
                    identity_id = f"identity_{uuid.uuid4().hex}"
                    identification_method = "whisper_diarization"
                    if source == "google_meet":
                        identification_method = "google_meet"

                    await conn.execute(
                        """
                        INSERT INTO bot_sessions.speaker_identities (
                            identity_id, session_id, speaker_label, identified_name,
                            identification_method, identification_confidence, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        identity_id,
                        session_id,
                        speaker_label,
                        speaker_name,
                        identification_method,
                        0.5,  # Default confidence
                        "{}",
                    )

                    logger.debug(
                        f"Created speaker identity: {identity_id} ({speaker_label}) for session {session_id}"
                    )

        except Exception as e:
            logger.warning(f"Error tracking speaker: {e}")


# Factory function
def create_data_pipeline(
    database_manager: "BotSessionDatabaseManager",
    audio_storage_path: str | None = None,
    enable_speaker_tracking: bool = True,
    enable_segment_continuity: bool = True,
) -> TranscriptionDataPipeline:
    """
    Create a transcription data pipeline with existing database manager.

    Args:
        database_manager: Existing BotSessionDatabaseManager instance (already initialized)
        audio_storage_path: Path for audio file storage (optional, for backwards compatibility)
        enable_speaker_tracking: Enable speaker identity tracking
        enable_segment_continuity: Enable segment continuity tracking

    Returns:
        Configured TranscriptionDataPipeline instance (ready to use)
    """
    # Create pipeline with existing database manager
    pipeline = TranscriptionDataPipeline(
        db_manager=database_manager,
        enable_speaker_tracking=enable_speaker_tracking,
        enable_segment_continuity=enable_segment_continuity,
    )

    return pipeline


# Example usage
async def main():
    """Example usage of the transcription data pipeline."""
    # Configuration
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "livetranslate",
        "username": "postgres",
        "password": "livetranslate",
    }

    # Create pipeline
    pipeline = create_data_pipeline(
        db_config=db_config,
        audio_storage_path="/tmp/livetranslate/audio",
        enable_speaker_tracking=True,
        enable_segment_continuity=True,
    )

    # Initialize database
    await pipeline.db_manager.initialize()

    # Create test session
    session_data = {
        "bot_id": "test-bot-001",
        "meeting_id": "test-meeting-pipeline",
        "meeting_title": "Pipeline Integration Test",
        "status": "active",
        "target_languages": ["en", "es"],
    }
    session_id = await pipeline.db_manager.create_bot_session(session_data)
    print(f"✅ Created session: {session_id}")

    # Process audio chunk
    audio_bytes = b"fake audio data for testing pipeline"
    metadata = AudioChunkMetadata(
        duration_seconds=10.0,
        sample_rate=16000,
        channels=1,
        chunk_start_time=0.0,
        chunk_end_time=10.0,
    )
    file_id = await pipeline.process_audio_chunk(session_id, audio_bytes, "wav", metadata)
    print(f"✅ Stored audio: {file_id}")

    # Process transcription
    transcription = TranscriptionResult(
        text="Hello everyone, welcome to the pipeline test.",
        language="en",
        start_time=0.0,
        end_time=3.5,
        speaker="SPEAKER_00",
        speaker_name="John Doe",
        confidence=0.95,
        segment_index=0,
    )
    transcript_id = await pipeline.process_transcription_result(session_id, file_id, transcription)
    print(f"✅ Stored transcript: {transcript_id}")

    # Process translation
    translation = TranslationResult(
        text="Hola a todos, bienvenidos a la prueba del pipeline.",
        source_language="en",
        target_language="es",
        speaker="SPEAKER_00",
        speaker_name="John Doe",
        confidence=0.92,
    )
    translation_id = await pipeline.process_translation_result(
        session_id, transcript_id, translation, 0.0, 3.5
    )
    print(f"✅ Stored translation: {translation_id}")

    # Get timeline
    timeline = await pipeline.get_session_timeline(session_id)
    print(f"\n📊 Timeline ({len(timeline)} entries):")
    for entry in timeline:
        print(f"  [{entry.timestamp:.1f}s] {entry.entry_type}: {entry.content[:50]}...")

    # Get speaker statistics
    stats = await pipeline.get_speaker_statistics(session_id)
    print("\n👥 Speaker Statistics:")
    for stat in stats:
        print(f"  {stat.speaker_name} ({stat.speaker_id}):")
        print(f"    - Speaking time: {stat.total_speaking_time:.1f}s")
        print(f"    - Segments: {stat.total_segments}")
        print(f"    - Translations: {stat.total_translations}")

    # Search
    results = await pipeline.search_transcripts(session_id, "welcome")
    print(f"\n🔍 Search results for 'welcome': {len(results)} matches")
    for result in results:
        print(f"  - {result.transcript_text}")

    # Cleanup
    await pipeline.db_manager.cleanup_session(session_id, remove_files=True)
    await pipeline.db_manager.close()
    print("\n✅ Pipeline test completed!")


if __name__ == "__main__":
    asyncio.run(main())
