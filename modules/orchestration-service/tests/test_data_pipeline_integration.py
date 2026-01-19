#!/usr/bin/env python3
"""
Data Pipeline Integration Tests

Comprehensive integration tests for the complete data pipeline covering:
- Database initialization and migration
- Audio chunk storage flow
- Transcription storage with speaker diarization
- Translation storage and linking
- Timeline queries
- Speaker statistics
- Full-text search functionality
- Error handling and edge cases

These tests use a real PostgreSQL database and test the complete
audio → transcription → translation → query flow.

Requirements:
- PostgreSQL database running (see POSTGRES_* env vars)
- Database schema initialized (database-init-complete.sql)

Author: LiveTranslate Team
Version: 1.0
"""

import os
import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.bot_session_manager import (
    create_bot_session_manager,
)
from pipeline.data_pipeline import (
    AudioChunkMetadata,
    TranscriptionDataPipeline,
    TranscriptionResult,
    TranslationResult,
)

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Database configuration from environment
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5433")),  # livetranslate-postgres container
    "database": os.getenv("POSTGRES_DB", "livetranslate_test"),
    "username": os.getenv("POSTGRES_USER", "livetranslate"),
    "password": os.getenv("POSTGRES_PASSWORD", "livetranslate_dev_password"),
}

AUDIO_STORAGE_PATH = os.getenv("TEST_AUDIO_STORAGE", "/tmp/livetranslate_test/audio")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest_asyncio.fixture(scope="module")
async def db_manager():
    """Create and initialize database manager."""
    manager = create_bot_session_manager(DB_CONFIG, AUDIO_STORAGE_PATH)

    # Initialize database
    success = await manager.initialize()
    assert success, "Failed to initialize database manager"

    yield manager

    # Cleanup
    await manager.close()


@pytest_asyncio.fixture(scope="module")
async def pipeline(db_manager):
    """Create data pipeline instance."""
    pipeline = TranscriptionDataPipeline(
        db_manager=db_manager,
        enable_speaker_tracking=True,
        enable_segment_continuity=True,
    )

    yield pipeline


@pytest_asyncio.fixture
async def test_session(db_manager):
    """Create a test session for each test."""
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    session_data = {
        "session_id": session_id,
        "bot_id": f"test_bot_{uuid.uuid4().hex[:8]}",
        "meeting_id": f"test_meeting_{uuid.uuid4().hex[:8]}",
        "meeting_title": "Integration Test Session",
        "status": "active",
        "target_languages": ["en", "es", "fr"],
    }

    created_session_id = await db_manager.create_bot_session(session_data)
    assert created_session_id == session_id

    yield session_id

    # Cleanup session after test
    await db_manager.cleanup_session(session_id, remove_files=True)


# ============================================================================
# TEST: DATABASE INITIALIZATION
# ============================================================================


@pytest.mark.asyncio
async def test_database_initialization(db_manager):
    """Test database connection and schema verification."""
    # Check that database is initialized
    assert db_manager.db_pool is not None

    # Verify core tables exist
    async with db_manager.db_pool.acquire() as conn:
        # Check sessions table
        sessions_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'bot_sessions'
                AND table_name = 'sessions'
            )
            """
        )
        assert sessions_exists, "Sessions table does not exist"

        # Check audio_files table
        audio_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'bot_sessions'
                AND table_name = 'audio_files'
            )
            """
        )
        assert audio_exists, "Audio files table does not exist"

        # Check transcripts table
        transcripts_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'bot_sessions'
                AND table_name = 'transcripts'
            )
            """
        )
        assert transcripts_exists, "Transcripts table does not exist"

        # Check translations table
        translations_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'bot_sessions'
                AND table_name = 'translations'
            )
            """
        )
        assert translations_exists, "Translations table does not exist"

        # Check speaker_identities table (enhancement)
        speaker_identities_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'bot_sessions'
                AND table_name = 'speaker_identities'
            )
            """
        )
        assert speaker_identities_exists, "Speaker identities table does not exist"


# ============================================================================
# TEST: AUDIO CHUNK STORAGE
# ============================================================================


@pytest.mark.asyncio
async def test_audio_chunk_storage(pipeline, test_session):
    """Test audio chunk storage with metadata."""
    # Create test audio data
    audio_bytes = b"fake_audio_data_" + os.urandom(1024)

    metadata = AudioChunkMetadata(
        duration_seconds=5.5,
        sample_rate=16000,
        channels=1,
        chunk_start_time=0.0,
        chunk_end_time=5.5,
        audio_quality_score=0.95,
        codec="pcm",
        bitrate=256000,
    )

    # Store audio chunk
    file_id = await pipeline.process_audio_chunk(
        session_id=test_session,
        audio_bytes=audio_bytes,
        file_format="wav",
        metadata=metadata,
    )

    assert file_id is not None, "Failed to store audio chunk"
    assert file_id.startswith("audio_"), "Invalid file_id format"

    # Verify audio file record
    audio_info = await pipeline.db_manager.audio_manager.get_audio_file_info(file_id)
    assert audio_info is not None, "Audio file record not found"
    assert audio_info.session_id == test_session
    assert audio_info.file_size == len(audio_bytes)
    assert audio_info.duration_seconds == 5.5
    assert audio_info.sample_rate == 16000
    assert audio_info.processing_status == "completed"


@pytest.mark.asyncio
async def test_multiple_audio_chunks(pipeline, test_session):
    """Test storing multiple audio chunks."""
    file_ids = []

    for i in range(3):
        audio_bytes = b"chunk_" + str(i).encode() + os.urandom(512)
        metadata = AudioChunkMetadata(
            duration_seconds=2.0,
            chunk_start_time=i * 2.0,
            chunk_end_time=(i + 1) * 2.0,
        )

        file_id = await pipeline.process_audio_chunk(test_session, audio_bytes, "wav", metadata)
        assert file_id is not None
        file_ids.append(file_id)

    # Verify all files stored
    assert len(file_ids) == 3
    assert len(set(file_ids)) == 3  # All unique

    # List session audio files
    audio_files = await pipeline.db_manager.audio_manager.list_session_audio_files(test_session)
    assert len(audio_files) == 3


# ============================================================================
# TEST: TRANSCRIPTION STORAGE
# ============================================================================


@pytest.mark.asyncio
async def test_transcription_storage(pipeline, test_session):
    """Test transcription result storage with speaker diarization."""
    # Create transcription result
    transcription = TranscriptionResult(
        text="Hello everyone, welcome to the meeting.",
        language="en",
        start_time=0.0,
        end_time=3.5,
        speaker="SPEAKER_00",
        speaker_name="John Doe",
        confidence=0.95,
        segment_index=0,
        is_final=True,
    )

    # Store transcription
    transcript_id = await pipeline.process_transcription_result(
        session_id=test_session,
        file_id=None,
        transcription=transcription,
        source_type="whisper_service",
    )

    assert transcript_id is not None, "Failed to store transcription"
    assert transcript_id.startswith("transcript_"), "Invalid transcript_id format"

    # Verify transcript record
    transcripts = await pipeline.db_manager.transcript_manager.get_session_transcripts(test_session)
    assert len(transcripts) == 1

    transcript = transcripts[0]
    assert transcript.transcript_id == transcript_id
    assert transcript.transcript_text == "Hello everyone, welcome to the meeting."
    assert transcript.language_code == "en"
    assert transcript.speaker_id == "SPEAKER_00"
    assert transcript.speaker_name == "John Doe"
    assert transcript.confidence_score == 0.95


@pytest.mark.asyncio
async def test_speaker_diarization_tracking(pipeline, test_session):
    """Test speaker diarization with multiple speakers."""
    speakers = [
        ("SPEAKER_00", "Alice"),
        ("SPEAKER_01", "Bob"),
        ("SPEAKER_00", "Alice"),
        ("SPEAKER_02", "Charlie"),
    ]

    transcript_ids = []

    for i, (speaker_id, speaker_name) in enumerate(speakers):
        transcription = TranscriptionResult(
            text=f"This is segment {i} from {speaker_name}.",
            language="en",
            start_time=i * 3.0,
            end_time=(i + 1) * 3.0,
            speaker=speaker_id,
            speaker_name=speaker_name,
            confidence=0.90 + (i * 0.01),
            segment_index=i,
        )

        transcript_id = await pipeline.process_transcription_result(
            test_session, None, transcription
        )
        assert transcript_id is not None
        transcript_ids.append(transcript_id)

    # Verify all transcripts stored
    transcripts = await pipeline.db_manager.transcript_manager.get_session_transcripts(test_session)
    assert len(transcripts) == 4

    # Verify speaker tracking
    async with pipeline.db_manager.db_pool.acquire() as conn:
        speaker_identities = await conn.fetch(
            """
            SELECT speaker_label, identified_name
            FROM bot_sessions.speaker_identities
            WHERE session_id = $1
            ORDER BY speaker_label
            """,
            test_session,
        )

        # Should have 3 unique speakers
        assert len(speaker_identities) == 3
        labels = [row["speaker_label"] for row in speaker_identities]
        assert "SPEAKER_00" in labels
        assert "SPEAKER_01" in labels
        assert "SPEAKER_02" in labels


# ============================================================================
# TEST: TRANSLATION STORAGE
# ============================================================================


@pytest.mark.asyncio
async def test_translation_storage(pipeline, test_session):
    """Test translation result storage and linking."""
    # First, create a transcript
    transcription = TranscriptionResult(
        text="Good morning everyone.",
        language="en",
        start_time=0.0,
        end_time=2.0,
        speaker="SPEAKER_00",
        confidence=0.95,
    )

    transcript_id = await pipeline.process_transcription_result(test_session, None, transcription)
    assert transcript_id is not None

    # Create translations
    translations = [
        TranslationResult(
            text="Buenos días a todos.",
            source_language="en",
            target_language="es",
            speaker="SPEAKER_00",
            confidence=0.92,
            translation_service="translation_service",
        ),
        TranslationResult(
            text="Bonjour à tous.",
            source_language="en",
            target_language="fr",
            speaker="SPEAKER_00",
            confidence=0.90,
            translation_service="translation_service",
        ),
    ]

    translation_ids = []
    for translation in translations:
        translation_id = await pipeline.process_translation_result(
            session_id=test_session,
            transcript_id=transcript_id,
            translation=translation,
            start_time=0.0,
            end_time=2.0,
        )
        assert translation_id is not None
        translation_ids.append(translation_id)

    # Verify translations stored
    assert len(translation_ids) == 2

    # Get translations from database
    stored_translations = await pipeline.db_manager.translation_manager.get_session_translations(
        test_session
    )
    assert len(stored_translations) == 2

    # Verify linking
    for trans in stored_translations:
        assert trans.source_transcript_id == transcript_id
        assert trans.source_language == "en"
        assert trans.target_language in ["es", "fr"]


# ============================================================================
# TEST: COMPLETE FLOW
# ============================================================================


@pytest.mark.asyncio
async def test_complete_pipeline_flow(pipeline, test_session):
    """Test complete audio → transcription → translation flow."""
    # Step 1: Store audio chunk
    audio_bytes = b"complete_flow_audio_" + os.urandom(2048)
    metadata = AudioChunkMetadata(
        duration_seconds=10.0,
        sample_rate=16000,
        channels=1,
        chunk_start_time=0.0,
        chunk_end_time=10.0,
    )

    file_id = await pipeline.process_audio_chunk(test_session, audio_bytes, "wav", metadata)
    assert file_id is not None

    # Step 2: Store multiple transcriptions with different speakers
    transcriptions = [
        TranscriptionResult(
            text="Hello, this is the first segment.",
            language="en",
            start_time=0.0,
            end_time=3.0,
            speaker="SPEAKER_00",
            speaker_name="Alice",
            confidence=0.95,
            segment_index=0,
        ),
        TranscriptionResult(
            text="Hi Alice, nice to meet you.",
            language="en",
            start_time=3.0,
            end_time=6.0,
            speaker="SPEAKER_01",
            speaker_name="Bob",
            confidence=0.93,
            segment_index=1,
        ),
        TranscriptionResult(
            text="Let's start the discussion.",
            language="en",
            start_time=6.0,
            end_time=9.0,
            speaker="SPEAKER_00",
            speaker_name="Alice",
            confidence=0.94,
            segment_index=2,
        ),
    ]

    transcript_ids = []
    for transcription in transcriptions:
        transcript_id = await pipeline.process_transcription_result(
            test_session, file_id, transcription
        )
        assert transcript_id is not None
        transcript_ids.append(transcript_id)

    # Step 3: Store translations for each transcript
    for i, transcript_id in enumerate(transcript_ids):
        # Spanish translation
        spanish_translation = TranslationResult(
            text=f"Spanish translation of segment {i}",
            source_language="en",
            target_language="es",
            speaker=transcriptions[i].speaker,
            speaker_name=transcriptions[i].speaker_name,
            confidence=0.90,
        )

        spanish_id = await pipeline.process_translation_result(
            test_session,
            transcript_id,
            spanish_translation,
            transcriptions[i].start_time,
            transcriptions[i].end_time,
        )
        assert spanish_id is not None

        # French translation
        french_translation = TranslationResult(
            text=f"French translation of segment {i}",
            source_language="en",
            target_language="fr",
            speaker=transcriptions[i].speaker,
            speaker_name=transcriptions[i].speaker_name,
            confidence=0.88,
        )

        french_id = await pipeline.process_translation_result(
            test_session,
            transcript_id,
            french_translation,
            transcriptions[i].start_time,
            transcriptions[i].end_time,
        )
        assert french_id is not None

    # Step 4: Verify complete data structure

    # Audio files
    audio_files = await pipeline.db_manager.audio_manager.list_session_audio_files(test_session)
    assert len(audio_files) == 1

    # Transcripts
    transcripts = await pipeline.db_manager.transcript_manager.get_session_transcripts(test_session)
    assert len(transcripts) == 3

    # Translations
    translations = await pipeline.db_manager.translation_manager.get_session_translations(
        test_session
    )
    assert len(translations) == 6  # 3 transcripts x 2 languages

    # Verify comprehensive session data
    comprehensive = await pipeline.db_manager.get_comprehensive_session_data(test_session)
    assert comprehensive is not None
    assert comprehensive["statistics"]["audio_files_count"] == 1
    assert comprehensive["statistics"]["transcripts_count"] == 3
    assert comprehensive["statistics"]["translations_count"] == 6


# ============================================================================
# TEST: TIMELINE QUERIES
# ============================================================================


@pytest.mark.asyncio
async def test_timeline_reconstruction(pipeline, test_session):
    """Test timeline reconstruction with transcripts and translations."""
    # Create test data
    transcriptions = [
        TranscriptionResult("First segment", "en", 0.0, 2.0, "SPEAKER_00", confidence=0.95),
        TranscriptionResult("Second segment", "en", 2.0, 4.0, "SPEAKER_01", confidence=0.93),
        TranscriptionResult("Third segment", "en", 4.0, 6.0, "SPEAKER_00", confidence=0.94),
    ]

    transcript_ids = []
    for trans in transcriptions:
        tid = await pipeline.process_transcription_result(test_session, None, trans)
        transcript_ids.append(tid)

        # Add translation
        translation = TranslationResult(
            text=f"Spanish: {trans.text}",
            source_language="en",
            target_language="es",
            speaker=trans.speaker,
            confidence=0.90,
        )
        await pipeline.process_translation_result(
            test_session, tid, translation, trans.start_time, trans.end_time
        )

    # Get timeline
    timeline = await pipeline.get_session_timeline(test_session)

    # Should have 6 entries (3 transcripts + 3 translations)
    assert len(timeline) == 6

    # Verify chronological order
    for i in range(len(timeline) - 1):
        assert timeline[i].timestamp <= timeline[i + 1].timestamp

    # Verify entry types
    transcript_entries = [e for e in timeline if e.entry_type == "transcript"]
    translation_entries = [e for e in timeline if e.entry_type == "translation"]
    assert len(transcript_entries) == 3
    assert len(translation_entries) == 3


@pytest.mark.asyncio
async def test_timeline_filtering(pipeline, test_session):
    """Test timeline filtering by time range, language, and speaker."""
    # Create test data with multiple speakers and languages
    test_data = [
        (0.0, 2.0, "SPEAKER_00", "en", "Hello"),
        (2.0, 4.0, "SPEAKER_01", "en", "Hi there"),
        (4.0, 6.0, "SPEAKER_00", "en", "How are you"),
        (6.0, 8.0, "SPEAKER_01", "en", "I'm fine"),
    ]

    for start, end, speaker, lang, text in test_data:
        trans = TranscriptionResult(text, lang, start, end, speaker, confidence=0.95)
        tid = await pipeline.process_transcription_result(test_session, None, trans)

        # Add Spanish translation
        translation = TranslationResult(f"ES: {text}", lang, "es", speaker, confidence=0.90)
        await pipeline.process_translation_result(test_session, tid, translation, start, end)

    # Test time range filter
    timeline_range = await pipeline.get_session_timeline(test_session, start_time=2.0, end_time=6.0)
    # Should have entries between 2.0 and 6.0 (2 transcripts + 2 translations = 4)
    assert len(timeline_range) == 4

    # Test language filter (only English transcripts)
    timeline_en = await pipeline.get_session_timeline(
        test_session, language_filter="en", include_translations=False
    )
    assert len(timeline_en) == 4
    assert all(e.language == "en" for e in timeline_en)

    # Test speaker filter
    timeline_speaker = await pipeline.get_speaker_timeline(
        test_session, "SPEAKER_00", include_translations=True
    )
    # SPEAKER_00 has 2 segments, each with 1 translation = 4 entries
    assert len(timeline_speaker) == 4
    assert all(e.speaker_id == "SPEAKER_00" for e in timeline_speaker)


# ============================================================================
# TEST: SPEAKER STATISTICS
# ============================================================================


@pytest.mark.asyncio
async def test_speaker_statistics(pipeline, test_session):
    """Test speaker statistics calculation."""
    # Create test data with varying speaking times
    speakers_data = [
        ("SPEAKER_00", "Alice", 5),  # 5 segments
        ("SPEAKER_01", "Bob", 3),  # 3 segments
        ("SPEAKER_02", "Charlie", 2),  # 2 segments
    ]

    for speaker_id, speaker_name, num_segments in speakers_data:
        for i in range(num_segments):
            start = i * 2.0
            end = start + 2.0

            trans = TranscriptionResult(
                text=f"{speaker_name} says something {i}",
                language="en",
                start_time=start,
                end_time=end,
                speaker=speaker_id,
                speaker_name=speaker_name,
                confidence=0.95,
            )

            tid = await pipeline.process_transcription_result(test_session, None, trans)

            # Add translation
            translation = TranslationResult(
                text=f"Spanish: {speaker_name} segment {i}",
                source_language="en",
                target_language="es",
                speaker=speaker_id,
                speaker_name=speaker_name,
                confidence=0.90,
            )
            await pipeline.process_translation_result(test_session, tid, translation, start, end)

    # Get speaker statistics
    stats = await pipeline.get_speaker_statistics(test_session)

    # Should have 3 speakers
    assert len(stats) == 3

    # Verify statistics for each speaker
    for stat in stats:
        expected_segments = dict(speakers_data)[stat.speaker_id][1]
        assert stat.total_segments == expected_segments
        assert stat.total_speaking_time == expected_segments * 2.0  # 2 seconds per segment
        assert stat.total_translations > 0


# ============================================================================
# TEST: FULL-TEXT SEARCH
# ============================================================================


@pytest.mark.asyncio
async def test_full_text_search(pipeline, test_session):
    """Test full-text search functionality."""
    # Create test data with searchable content
    test_phrases = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a great programming language",
        "Data pipeline integration testing is important",
        "Machine learning models require training data",
    ]

    for i, phrase in enumerate(test_phrases):
        trans = TranscriptionResult(
            text=phrase,
            language="en",
            start_time=i * 3.0,
            end_time=(i + 1) * 3.0,
            speaker="SPEAKER_00",
            confidence=0.95,
        )
        await pipeline.process_transcription_result(test_session, None, trans)

    # Test exact word search
    results = await pipeline.search_transcripts(test_session, "Python", use_fuzzy=False)
    assert len(results) >= 1
    assert any("Python" in r.transcript_text for r in results)

    # Test fuzzy search
    results_fuzzy = await pipeline.search_transcripts(test_session, "programming", use_fuzzy=True)
    assert len(results_fuzzy) >= 1

    # Test phrase search
    results_phrase = await pipeline.search_transcripts(
        test_session, "integration testing", use_fuzzy=False
    )
    assert len(results_phrase) >= 1


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_session_handling(pipeline):
    """Test handling of invalid session IDs."""
    fake_session = "nonexistent_session_12345"

    # Timeline should return empty list for invalid session
    timeline = await pipeline.get_session_timeline(fake_session)
    assert len(timeline) == 0

    # Speaker stats should return empty list
    stats = await pipeline.get_speaker_statistics(fake_session)
    assert len(stats) == 0

    # Search should return empty results
    results = await pipeline.search_transcripts(fake_session, "test")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_edge_cases(pipeline, test_session):
    """Test edge cases and boundary conditions."""
    # Empty audio bytes
    file_id = await pipeline.process_audio_chunk(test_session, b"", "wav")
    assert file_id is not None  # Should still create record

    # Very long transcript text
    long_text = "word " * 10000  # 10,000 words
    trans = TranscriptionResult(
        text=long_text,
        language="en",
        start_time=0.0,
        end_time=600.0,  # 10 minutes
        speaker="SPEAKER_00",
        confidence=0.95,
    )
    transcript_id = await pipeline.process_transcription_result(test_session, None, trans)
    assert transcript_id is not None

    # Zero duration segment
    zero_duration = TranscriptionResult(
        text="Zero duration",
        language="en",
        start_time=0.0,
        end_time=0.0,
        speaker="SPEAKER_00",
        confidence=0.95,
    )
    tid = await pipeline.process_transcription_result(test_session, None, zero_duration)
    assert tid is not None


# ============================================================================
# TEST: SEGMENT CONTINUITY
# ============================================================================


@pytest.mark.asyncio
async def test_segment_continuity(pipeline, test_session):
    """Test segment continuity tracking."""
    # Create sequential segments
    segments = [
        TranscriptionResult(
            f"Segment {i}", "en", i * 2.0, (i + 1) * 2.0, "SPEAKER_00", confidence=0.95
        )
        for i in range(3)
    ]

    transcript_ids = []
    for seg in segments:
        tid = await pipeline.process_transcription_result(test_session, None, seg)
        transcript_ids.append(tid)

    # Verify continuity links
    async with pipeline.db_manager.db_pool.acquire() as conn:
        # Check first segment (should have no previous, but have next)
        first = await conn.fetchrow(
            "SELECT previous_segment_id, next_segment_id FROM bot_sessions.transcripts WHERE transcript_id = $1",
            transcript_ids[0],
        )
        assert first["previous_segment_id"] is None
        assert first["next_segment_id"] == transcript_ids[1]

        # Check middle segment (should have both)
        middle = await conn.fetchrow(
            "SELECT previous_segment_id, next_segment_id FROM bot_sessions.transcripts WHERE transcript_id = $1",
            transcript_ids[1],
        )
        assert middle["previous_segment_id"] == transcript_ids[0]
        assert middle["next_segment_id"] == transcript_ids[2]

        # Check last segment (should have previous, but no next)
        last = await conn.fetchrow(
            "SELECT previous_segment_id, next_segment_id FROM bot_sessions.transcripts WHERE transcript_id = $1",
            transcript_ids[2],
        )
        assert last["previous_segment_id"] == transcript_ids[1]
        # next_segment_id might be None or empty depending on when test runs


# ============================================================================
# TEST: PERFORMANCE AND STATISTICS
# ============================================================================


@pytest.mark.asyncio
async def test_session_statistics_computation(pipeline, test_session):
    """Test automatic session statistics computation."""
    # Create comprehensive test data
    for i in range(5):
        # Audio
        audio_bytes = b"audio_" + os.urandom(1024)
        metadata = AudioChunkMetadata(duration_seconds=2.0)
        await pipeline.process_audio_chunk(test_session, audio_bytes, "wav", metadata)

        # Transcript
        trans = TranscriptionResult(
            f"Transcript {i}",
            "en",
            i * 2.0,
            (i + 1) * 2.0,
            "SPEAKER_00",
            confidence=0.95,
        )
        tid = await pipeline.process_transcription_result(test_session, None, trans)

        # Translation
        translation = TranslationResult(
            f"Translation {i}", "en", "es", "SPEAKER_00", confidence=0.90
        )
        await pipeline.process_translation_result(
            test_session, tid, translation, i * 2.0, (i + 1) * 2.0
        )

    # Trigger statistics update
    await pipeline.db_manager.db_pool.execute(
        "SELECT bot_sessions.update_session_statistics($1)", test_session
    )

    # Verify statistics
    async with pipeline.db_manager.db_pool.acquire() as conn:
        stats = await conn.fetchrow(
            "SELECT * FROM bot_sessions.session_statistics WHERE session_id = $1",
            test_session,
        )

        assert stats is not None
        assert stats["total_audio_files"] == 5
        assert stats["total_transcripts"] == 5
        assert stats["total_translations"] == 5


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
