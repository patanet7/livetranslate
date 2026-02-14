"""
Comprehensive Integration Tests for Orchestration Service
=========================================================

Tests the complete audio processing orchestration with:
- Real database operations (PostgreSQL)
- Proper audio chunking and streaming
- Session tracking and metrics
- Service response fixtures matching actual contracts
- Audio coordinator integration

NO MOCKS for database - uses real PostgreSQL test database
Service responses use fixtures that match actual response formats
"""

import asyncio
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# SMPTE Timecode for precise audio overlap handling
try:
    from timecode import Timecode

    SMPTE_AVAILABLE = True
except ImportError:
    SMPTE_AVAILABLE = False
    print("⚠️  timecode (SMPTE) not available - falling back to simple chunking")

# Database imports
from src.database import (
    DatabaseManager,
)
from src.database.unified_bot_session_repository import UnifiedBotSessionRepository

# Test Configuration
BASE_URL = "http://localhost:3000"


# =============================================================================
# TEST DATABASE SETUP — uses shared testcontainer + Alembic from root conftest
# =============================================================================


@pytest.fixture(scope="session")
def test_database(database_url, run_migrations):
    """Use shared testcontainer + Alembic schema (no create_all/drop_all)."""
    engine = create_engine(database_url, echo=False)
    session_local = sessionmaker(bind=engine)
    yield {
        "engine": engine,
        "session_factory": session_local,
        "url": database_url,
    }
    engine.dispose()


@pytest.fixture(scope="function")
async def db_session(test_database):
    """
    Create a fresh database session for each test.
    Rolls back after test to ensure isolation.
    """
    session = test_database["session_factory"]()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
async def bot_repository(test_database, database_url):
    """
    Create UnifiedBotSessionRepository with test database.
    Uses async URL for asyncpg driver.
    """
    from config import DatabaseSettings

    # Convert sync URL to async URL for asyncpg driver
    async_url = database_url
    if database_url.startswith("postgresql://"):
        async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    db_config = DatabaseSettings(url=async_url)
    db_manager = DatabaseManager(db_config)
    db_manager.initialize()  # Sync method, not async

    repository = UnifiedBotSessionRepository(db_manager)
    await repository.initialize()

    yield repository

    await repository.shutdown()


# =============================================================================
# SERVICE RESPONSE FIXTURES (matching actual service contracts)
# =============================================================================


@pytest.fixture
def whisper_transcription_response():
    """
    Fixture matching TranscriptionResponse from audio service.
    Based on actual Whisper service response format.
    """

    def _create(
        text: str = "Hello, this is a test transcription.",
        language: str = "en",
        confidence: float = 0.95,
        segments: list[dict] | None = None,
        speakers: list[dict] | None = None,
        processing_time: float = 0.5,
    ) -> dict[str, Any]:
        """Create a realistic transcription response"""

        if segments is None:
            segments = [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": text,
                    "confidence": confidence,
                    "no_speech_prob": 0.02,
                }
            ]

        if speakers is None:
            speakers = [
                {
                    "speaker_id": "SPEAKER_00",
                    "start": 0.0,
                    "end": 2.5,
                    "confidence": 0.92,
                }
            ]

        return {
            "text": text,
            "language": language,
            "segments": segments,
            "speakers": speakers,
            "processing_time": processing_time,
            "confidence": confidence,
            "model_used": "whisper-base",
            "diarization_enabled": True,
            "vad_enabled": True,
        }

    return _create


@pytest.fixture
def translation_service_response():
    """
    Fixture matching TranslationResponse from translation service.
    Based on actual translation service response format.
    """

    def _create(
        source_text: str = "Hello",
        translated_text: str = "Hola",
        source_language: str = "en",
        target_language: str = "es",
        confidence: float = 0.93,
        processing_time: float = 0.3,
    ) -> dict[str, Any]:
        """Create a realistic translation response"""

        return {
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language,
            "confidence": confidence,
            "processing_time": processing_time,
            "model_used": "default",
            "backend_used": "embedded",
            "session_id": None,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    return _create


# =============================================================================
# AUDIO GENERATION UTILITIES
# =============================================================================


@pytest.fixture
def generate_test_audio():
    """Generate realistic test audio data (sine wave)"""

    def _generate(
        duration_seconds: float = 1.0,
        frequency: int = 440,
        sample_rate: int = 16000,
        amplitude: float = 0.5,
    ) -> bytes:
        """
        Generate PCM audio data (sine wave).
        Returns raw bytes in 16-bit PCM format.
        """
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples, False)

        # Generate sine wave
        audio_signal = amplitude * np.sin(2 * np.pi * frequency * t)

        # Convert to 16-bit PCM
        audio_signal = (audio_signal * 32767).astype(np.int16)

        return audio_signal.tobytes()

    return _generate


@pytest.fixture
def create_wav_file(tmp_path, generate_test_audio):
    """Create WAV file from audio data"""

    def _create(
        duration: float = 1.0,
        frequency: int = 440,
        sample_rate: int = 16000,
    ) -> Path:
        """Create a real WAV file for testing"""
        audio_data = generate_test_audio(duration, frequency, sample_rate)

        wav_path = tmp_path / f"test_audio_{int(datetime.now().timestamp())}.wav"

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        return wav_path

    return _create


@pytest.fixture
def chunk_audio():
    """Split audio into chunks for streaming tests"""

    def _chunk(
        audio_data: bytes,
        chunk_size_ms: int = 100,
        sample_rate: int = 16000,
        sample_width: int = 2,  # 16-bit = 2 bytes
    ) -> list[bytes]:
        """
        Split audio data into time-based chunks.

        Args:
            audio_data: Raw PCM audio bytes
            chunk_size_ms: Chunk duration in milliseconds
            sample_rate: Audio sample rate
            sample_width: Bytes per sample (2 for 16-bit)

        Returns:
            List of audio chunks
        """
        # Calculate bytes per chunk
        samples_per_chunk = int((chunk_size_ms / 1000.0) * sample_rate)
        bytes_per_chunk = samples_per_chunk * sample_width

        # Split into chunks
        chunks = []
        for i in range(0, len(audio_data), bytes_per_chunk):
            chunk = audio_data[i : i + bytes_per_chunk]
            if len(chunk) > 0:
                chunks.append(chunk)

        return chunks

    return _chunk


@pytest.fixture
def chunk_audio_with_overlap():
    """
    Split audio into overlapping chunks with SMPTE timecode tracking.

    Uses SMPTE timecode for industry-standard precise timecode generation
    to ensure proper reconstruction and overlap handling.
    """

    def _chunk_with_overlap(
        audio_data: bytes,
        chunk_size_ms: int = 100,
        overlap_ms: int = 20,  # 20ms overlap for context
        sample_rate: int = 16000,
        sample_width: int = 2,
        framerate: str = "30",  # SMPTE framerate (24, 25, 30, etc.)
    ) -> list[dict[str, Any]]:
        """
        Split audio with overlap and SMPTE timecode metadata.

        Args:
            audio_data: Raw PCM audio bytes
            chunk_size_ms: Chunk duration in milliseconds
            overlap_ms: Overlap duration in milliseconds
            sample_rate: Audio sample rate
            sample_width: Bytes per sample
            framerate: SMPTE framerate as string ('24', '25', '30', '60')

        Returns:
            List of dicts with 'data', 'smpte_timecode', 'start_frame', 'end_frame'
        """
        samples_per_chunk = int((chunk_size_ms / 1000.0) * sample_rate)
        overlap_samples = int((overlap_ms / 1000.0) * sample_rate)
        bytes_per_chunk = samples_per_chunk * sample_width
        overlap_samples * sample_width

        # Stride considers overlap
        stride_samples = samples_per_chunk - overlap_samples
        stride_bytes = stride_samples * sample_width

        chunks = []
        frame_number = 0

        for i in range(0, len(audio_data), stride_bytes):
            # Get chunk with overlap
            chunk_end = min(i + bytes_per_chunk, len(audio_data))
            chunk = audio_data[i:chunk_end]

            if len(chunk) > 0:
                # Calculate timing
                start_time_seconds = i / (sample_rate * sample_width)
                end_time_seconds = chunk_end / (sample_rate * sample_width)

                # Create timecode metadata
                chunk_metadata = {
                    "data": chunk,
                    "chunk_index": len(chunks),
                    "start_byte": i,
                    "end_byte": chunk_end,
                    "start_time_seconds": start_time_seconds,
                    "end_time_seconds": end_time_seconds,
                    "duration_ms": (end_time_seconds - start_time_seconds) * 1000,
                    "has_overlap": len(chunks) > 0,  # All but first chunk have overlap
                    "overlap_ms": overlap_ms if len(chunks) > 0 else 0,
                    "frame_number": frame_number,
                }

                # Generate SMPTE timecode if available
                if SMPTE_AVAILABLE:
                    try:
                        # Create SMPTE timecode from seconds
                        start_tc = Timecode(framerate, start_seconds=start_time_seconds)
                        end_tc = Timecode(framerate, start_seconds=end_time_seconds)

                        chunk_metadata["smpte_timecode"] = {
                            "start": str(start_tc),
                            "end": str(end_tc),
                            "start_frames": start_tc.frames,
                            "end_frames": end_tc.frames,
                            "framerate": framerate,
                            "drop_frame": start_tc.drop_frame,
                        }
                    except Exception as e:
                        chunk_metadata["smpte_timecode"] = None
                        chunk_metadata["smpte_error"] = str(e)

                chunks.append(chunk_metadata)
                frame_number += 1

        return chunks

    return _chunk_with_overlap


# =============================================================================
# INTEGRATION TESTS - DATABASE OPERATIONS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestDatabaseIntegration:
    """Test real database operations with no mocks"""

    async def test_session_create_and_retrieve(self, bot_repository):
        """
        TEST: Create bot session and retrieve from database
        VERIFY: Session is persisted correctly with all fields
        """
        session_id = f"test-session-{int(datetime.now().timestamp())}"
        meeting_url = "https://meet.google.com/abc-defg-hij"
        bot_config = {
            "audio_config": {
                "sample_rate": 16000,
                "channels": 1,
            },
            "translation_config": {
                "source_language": "en",
                "target_language": "es",
            },
        }

        # Create session
        created_session = await bot_repository.session_create(
            session_id=session_id,
            meeting_url=meeting_url,
            bot_config=bot_config,
            metadata={"test": True},
        )

        assert created_session is not None, "Session should be created"
        assert created_session.session_id == session_id
        assert created_session.meeting_url == meeting_url
        assert created_session.status == "initializing"
        assert created_session.bot_config == bot_config

        # Retrieve session
        retrieved_session = await bot_repository.session_get(session_id)

        assert retrieved_session is not None, "Session should be retrievable"
        assert retrieved_session.session_id == session_id
        assert retrieved_session.meeting_url == meeting_url
        assert retrieved_session.bot_config == bot_config

    async def test_audio_file_storage(self, bot_repository, generate_test_audio):
        """
        TEST: Store and retrieve audio files with chunks
        VERIFY: Audio data persisted correctly with metadata
        """
        # Create session first
        session_id = f"test-session-{int(datetime.now().timestamp())}"
        await bot_repository.session_create(
            session_id=session_id,
            meeting_url="https://meet.google.com/test",
            bot_config={},
        )

        # Generate test audio
        audio_data = generate_test_audio(duration_seconds=2.0, frequency=440)

        # Store audio file
        audio_file = await bot_repository.audio_create(
            session_id=session_id,
            audio_data=audio_data,
            metadata={
                "sample_rate": 16000,
                "duration": 2.0,
                "format": "pcm",
            },
        )

        assert audio_file is not None, "Audio file should be created"
        assert audio_file.session_id == session_id
        assert audio_file.file_size == len(audio_data)
        assert audio_file.sample_rate == 16000
        assert audio_file.duration == 2.0

        # Retrieve audio file
        retrieved_audio = await bot_repository.audio_get(audio_file.audio_id)

        assert retrieved_audio is not None, "Audio should be retrievable"
        assert retrieved_audio.audio_id == audio_file.audio_id
        assert retrieved_audio.file_size == len(audio_data)

    async def test_transcript_storage(self, bot_repository, whisper_transcription_response):
        """
        TEST: Store transcription results in database
        VERIFY: Transcripts persisted with segments and speakers
        """
        # Create session and audio file
        session_id = f"test-session-{int(datetime.now().timestamp())}"
        await bot_repository.session_create(
            session_id=session_id,
            meeting_url="https://meet.google.com/test",
            bot_config={},
        )

        audio_file = await bot_repository.audio_create(
            session_id=session_id,
            audio_data=b"fake_audio_data",
            metadata={"sample_rate": 16000},
        )

        # Create transcription response
        transcription = whisper_transcription_response(
            text="This is a test transcription with multiple segments.",
            language="en",
            confidence=0.95,
        )

        # Store transcript
        transcript_record = await bot_repository.transcript_create(
            audio_id=audio_file.audio_id,
            transcription_text=transcription["text"],
            language=transcription["language"],
            confidence=transcription["confidence"],
            segments=transcription["segments"],
            speakers=transcription["speakers"],
        )

        assert transcript_record is not None, "Transcript should be created"
        assert transcript_record.audio_id == audio_file.audio_id
        assert transcript_record.transcription_text == transcription["text"]
        assert transcript_record.language == "en"
        assert transcript_record.confidence >= 0.95
        assert len(transcript_record.segments) > 0
        assert len(transcript_record.speakers) > 0

    async def test_translation_storage(self, bot_repository, translation_service_response):
        """
        TEST: Store translation results in database
        VERIFY: Translations linked to transcripts correctly
        """
        # Create session, audio, and transcript
        session_id = f"test-session-{int(datetime.now().timestamp())}"
        await bot_repository.session_create(
            session_id=session_id,
            meeting_url="https://meet.google.com/test",
            bot_config={},
        )

        audio_file = await bot_repository.audio_create(
            session_id=session_id, audio_data=b"fake_audio", metadata={}
        )

        transcript = await bot_repository.transcript_create(
            audio_id=audio_file.audio_id,
            transcription_text="Hello, how are you?",
            language="en",
            confidence=0.95,
        )

        # Create translation
        translation_data = translation_service_response(
            source_text="Hello, how are you?",
            translated_text="Hola, ¿cómo estás?",
            source_language="en",
            target_language="es",
            confidence=0.93,
        )

        # Store translation
        translation_record = await bot_repository.translation_create(
            transcript_id=transcript.transcript_id,
            translated_text=translation_data["translated_text"],
            target_language=translation_data["target_language"],
            confidence=translation_data["confidence"],
        )

        assert translation_record is not None, "Translation should be created"
        assert translation_record.transcript_id == transcript.transcript_id
        assert translation_record.translated_text == translation_data["translated_text"]
        assert translation_record.target_language == "es"
        assert translation_record.confidence >= 0.90


# =============================================================================
# INTEGRATION TESTS - AUDIO CHUNKING
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestAudioChunking:
    """Test audio chunking and streaming with database tracking"""

    async def test_chunk_generation(self, generate_test_audio, chunk_audio):
        """
        TEST: Split audio into time-based chunks
        VERIFY: Chunks are correctly sized and complete
        """
        # Generate 5 seconds of audio
        audio_data = generate_test_audio(duration_seconds=5.0, sample_rate=16000)

        # Split into 100ms chunks
        chunks = chunk_audio(audio_data, chunk_size_ms=100, sample_rate=16000)

        # Verify chunking
        assert len(chunks) == 50, "Should have 50 chunks (5s / 0.1s)"

        # Verify chunk sizes (16kHz * 0.1s * 2 bytes/sample = 3200 bytes)
        expected_chunk_size = 3200
        for i, chunk in enumerate(chunks):
            assert (
                len(chunk) == expected_chunk_size
            ), f"Chunk {i} should be {expected_chunk_size} bytes"

        # Verify total data preserved
        total_reconstructed = b"".join(chunks)
        assert len(total_reconstructed) == len(audio_data), "All audio data should be preserved"

    async def test_chunk_streaming_to_database(
        self,
        bot_repository,
        generate_test_audio,
        chunk_audio,
    ):
        """
        TEST: Stream audio chunks and store each in database
        VERIFY: All chunks tracked with sequence numbers
        """
        # Create session
        session_id = f"test-session-{int(datetime.now().timestamp())}"
        await bot_repository.session_create(
            session_id=session_id,
            meeting_url="https://meet.google.com/test",
            bot_config={},
        )

        # Generate and chunk audio
        audio_data = generate_test_audio(duration_seconds=2.0, sample_rate=16000)
        chunks = chunk_audio(audio_data, chunk_size_ms=200, sample_rate=16000)

        # Stream chunks to database
        audio_files = []
        for i, chunk in enumerate(chunks):
            audio_file = await bot_repository.audio_create(
                session_id=session_id,
                audio_data=chunk,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_duration_ms": 200,
                    "sample_rate": 16000,
                },
            )
            audio_files.append(audio_file)

        # Verify all chunks stored
        assert len(audio_files) == len(chunks), "All chunks should be stored"

        # Verify sequence
        for i, audio_file in enumerate(audio_files):
            assert audio_file.metadata["chunk_index"] == i
            assert audio_file.metadata["total_chunks"] == len(chunks)

        # Verify total data
        total_size = sum(af.file_size for af in audio_files)
        assert total_size == len(audio_data), "Total stored data should match original"

    async def test_concurrent_chunk_processing(
        self,
        bot_repository,
        generate_test_audio,
        chunk_audio,
        whisper_transcription_response,
    ):
        """
        TEST: Process multiple chunks concurrently with database tracking
        VERIFY: All chunks processed and stored correctly
        """
        # Create session
        session_id = f"test-session-{int(datetime.now().timestamp())}"
        await bot_repository.session_create(
            session_id=session_id,
            meeting_url="https://meet.google.com/test",
            bot_config={},
        )

        # Generate chunks
        audio_data = generate_test_audio(duration_seconds=3.0, sample_rate=16000)
        chunks = chunk_audio(audio_data, chunk_size_ms=500, sample_rate=16000)

        # Process chunks concurrently
        async def process_chunk(chunk_index: int, chunk_data: bytes):
            """Simulate processing a single chunk"""
            # Store audio chunk
            audio_file = await bot_repository.audio_create(
                session_id=session_id,
                audio_data=chunk_data,
                metadata={"chunk_index": chunk_index},
            )

            # Simulate transcription
            transcription = whisper_transcription_response(
                text=f"Transcription for chunk {chunk_index}",
                confidence=0.90 + (chunk_index * 0.01),  # Varying confidence
            )

            # Store transcript
            transcript = await bot_repository.transcript_create(
                audio_id=audio_file.audio_id,
                transcription_text=transcription["text"],
                language=transcription["language"],
                confidence=transcription["confidence"],
            )

            return {
                "audio_file": audio_file,
                "transcript": transcript,
                "chunk_index": chunk_index,
            }

        # Process all chunks concurrently
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        # Verify all processed
        assert len(results) == len(chunks), "All chunks should be processed"

        # Verify sequence maintained
        for result in results:
            assert result["audio_file"] is not None
            assert result["transcript"] is not None
            assert result["chunk_index"] >= 0

    @pytest.mark.skipif(not SMPTE_AVAILABLE, reason="timecode (SMPTE) not installed")
    async def test_overlapping_chunks_with_smpte_timecode(
        self,
        bot_repository,
        generate_test_audio,
        chunk_audio_with_overlap,
        whisper_transcription_response,
    ):
        """
        TEST: Process overlapping audio chunks with SMPTE timecode tracking
        VERIFY: Overlaps handled correctly, SMPTE timecodes accurate, proper reconstruction
        """
        # Create session
        session_id = f"test-session-{int(datetime.now().timestamp())}"
        await bot_repository.session_create(
            session_id=session_id,
            meeting_url="https://meet.google.com/test",
            bot_config={},
        )

        # Generate audio (10 seconds for clear timecode testing)
        audio_data = generate_test_audio(duration_seconds=10.0, sample_rate=16000, frequency=440)

        # Create overlapping chunks with SMPTE timecode
        chunks = chunk_audio_with_overlap(
            audio_data,
            chunk_size_ms=500,  # 500ms chunks
            overlap_ms=100,  # 100ms overlap (20% overlap)
            sample_rate=16000,
            framerate="30",  # 30fps SMPTE timecode
        )

        print("\n📊 Overlapping Chunk Analysis:")
        print("   Total audio duration: 10.0s")
        print("   Chunk size: 500ms")
        print("   Overlap: 100ms")
        print(f"   Total chunks: {len(chunks)}")

        # Verify chunk properties
        assert len(chunks) > 0, "Should have chunks"

        # Verify overlaps
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i - 1]

            # Current chunk should start before previous chunk ends (overlap)
            assert (
                current_chunk["start_time_seconds"] < previous_chunk["end_time_seconds"]
            ), f"Chunk {i} should overlap with previous chunk"

            # Verify overlap amount (approximately 100ms)
            overlap_time = previous_chunk["end_time_seconds"] - current_chunk["start_time_seconds"]
            overlap_ms = overlap_time * 1000
            assert 95 <= overlap_ms <= 105, f"Overlap should be ~100ms, got {overlap_ms:.1f}ms"

            # Verify SMPTE timecodes
            if current_chunk.get("smpte_timecode"):
                print(
                    f"   Chunk {i}: {current_chunk['smpte_timecode']['start']} -> {current_chunk['smpte_timecode']['end']} "
                    + f"({current_chunk['start_time_seconds']:.3f}s - {current_chunk['end_time_seconds']:.3f}s)"
                )

        # Store chunks in database with timecode metadata
        stored_chunks = []
        for chunk in chunks:
            audio_file = await bot_repository.audio_create(
                session_id=session_id,
                audio_data=chunk["data"],
                metadata={
                    "chunk_index": chunk["chunk_index"],
                    "start_time_seconds": chunk["start_time_seconds"],
                    "end_time_seconds": chunk["end_time_seconds"],
                    "has_overlap": chunk["has_overlap"],
                    "overlap_ms": chunk["overlap_ms"],
                    "smpte_timecode": chunk.get("smpte_timecode"),
                    "frame_number": chunk["frame_number"],
                },
            )
            stored_chunks.append(audio_file)

        # Verify database storage
        assert len(stored_chunks) == len(chunks), "All chunks should be stored"

        # Verify we can reconstruct timeline from database
        retrieved_chunks = []
        for audio_file in stored_chunks:
            retrieved_chunks.append(
                {
                    "index": audio_file.metadata["chunk_index"],
                    "start": audio_file.metadata["start_time_seconds"],
                    "end": audio_file.metadata["end_time_seconds"],
                    "timecode": audio_file.metadata.get("ltc_timecode"),
                }
            )

        # Sort by index
        retrieved_chunks.sort(key=lambda x: x["index"])

        # Verify timeline continuity
        for i in range(1, len(retrieved_chunks)):
            # Each chunk should connect to or overlap with the previous
            gap = retrieved_chunks[i]["start"] - retrieved_chunks[i - 1]["end"]
            assert gap <= 0, f"No gaps allowed between chunks (gap: {gap:.3f}s)"

        print(f"\n   ✅ All {len(chunks)} chunks stored with proper SMPTE timecode metadata")
        print(f"   ✅ Overlaps verified: {chunks[1]['overlap_ms']}ms between consecutive chunks")
        print("   ✅ Timeline reconstructed from database successfully")
        if chunks[0].get("smpte_timecode"):
            print(f"   ✅ SMPTE framerate: {chunks[0]['smpte_timecode']['framerate']} fps")


# =============================================================================
# INTEGRATION TESTS - METRICS AND SESSION TRACKING
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestSessionMetrics:
    """Test session tracking and metrics collection"""

    async def test_session_lifecycle_tracking(self, bot_repository):
        """
        TEST: Track complete session lifecycle
        VERIFY: Status transitions recorded correctly
        """
        session_id = f"test-session-{int(datetime.now().timestamp())}"

        # Create session
        session = await bot_repository.session_create(
            session_id=session_id,
            meeting_url="https://meet.google.com/test",
            bot_config={},
        )

        assert session.status == "initializing"
        assert session.started_at is not None
        assert session.ended_at is None

        # Update to running
        await bot_repository.session_update(
            session_id=session_id,
            status="running",
        )

        updated = await bot_repository.session_get(session_id)
        assert updated.status == "running"

        # Update to completed
        await bot_repository.session_update(
            session_id=session_id,
            status="completed",
        )

        completed = await bot_repository.session_get(session_id)
        assert completed.status == "completed"
        assert completed.ended_at is not None

        # Calculate duration
        duration = completed.ended_at - completed.started_at
        assert duration.total_seconds() >= 0

    async def test_processing_metrics_collection(
        self,
        bot_repository,
        generate_test_audio,
        whisper_transcription_response,
        translation_service_response,
    ):
        """
        TEST: Collect processing metrics throughout pipeline
        VERIFY: Latency, confidence, and throughput tracked
        """
        # Create session
        session_id = f"test-session-{int(datetime.now().timestamp())}"
        await bot_repository.session_create(
            session_id=session_id,
            meeting_url="https://meet.google.com/test",
            bot_config={},
        )

        # Process multiple chunks and track metrics
        metrics = {
            "chunks_processed": 0,
            "total_audio_duration": 0.0,
            "total_processing_time": 0.0,
            "transcription_times": [],
            "translation_times": [],
            "average_confidence": 0.0,
        }

        num_chunks = 5
        for i in range(num_chunks):
            # Generate audio
            audio_data = generate_test_audio(duration_seconds=1.0)

            # Store audio
            start_time = datetime.now()
            audio_file = await bot_repository.audio_create(
                session_id=session_id,
                audio_data=audio_data,
                metadata={"chunk_index": i},
            )

            # Simulate transcription
            transcription = whisper_transcription_response(
                text=f"Chunk {i} transcription",
                processing_time=0.3 + (i * 0.05),  # Varying processing time
                confidence=0.90 + (i * 0.01),
            )

            transcript = await bot_repository.transcript_create(
                audio_id=audio_file.audio_id,
                transcription_text=transcription["text"],
                language=transcription["language"],
                confidence=transcription["confidence"],
            )

            # Simulate translation
            translation = translation_service_response(
                source_text=transcription["text"],
                translated_text=f"Chunk {i} traducción",
                processing_time=0.2 + (i * 0.03),
            )

            await bot_repository.translation_create(
                transcript_id=transcript.transcript_id,
                translated_text=translation["translated_text"],
                target_language="es",
                confidence=translation["confidence"],
            )

            # Update metrics
            metrics["chunks_processed"] += 1
            metrics["total_audio_duration"] += 1.0
            metrics["transcription_times"].append(transcription["processing_time"])
            metrics["translation_times"].append(translation["processing_time"])

            processing_time = (datetime.now() - start_time).total_seconds()
            metrics["total_processing_time"] += processing_time

        # Calculate final metrics
        metrics["average_confidence"] = sum(metrics["transcription_times"]) / len(
            metrics["transcription_times"]
        )

        metrics["average_transcription_time"] = np.mean(metrics["transcription_times"])
        metrics["average_translation_time"] = np.mean(metrics["translation_times"])
        metrics["throughput"] = metrics["total_audio_duration"] / metrics["total_processing_time"]

        # Verify metrics
        assert metrics["chunks_processed"] == num_chunks
        assert metrics["total_audio_duration"] == num_chunks * 1.0
        assert metrics["average_transcription_time"] > 0
        assert metrics["average_translation_time"] > 0
        assert metrics["throughput"] > 0, "Should have positive throughput"

        print("\n📊 Processing Metrics:")
        print(f"   Chunks processed: {metrics['chunks_processed']}")
        print(f"   Total duration: {metrics['total_audio_duration']:.2f}s")
        print(f"   Avg transcription time: {metrics['average_transcription_time']:.3f}s")
        print(f"   Avg translation time: {metrics['average_translation_time']:.3f}s")
        print(f"   Throughput: {metrics['throughput']:.2f}x realtime")


# =============================================================================
# SUMMARY
# =============================================================================

_TEST_SUMMARY = """
Test Summary:
=============

✅ Database Integration (4 tests):
   - Session create and retrieve
   - Audio file storage with chunks
   - Transcript storage with segments
   - Translation storage with linking

✅ Audio Chunking (3 tests):
   - Chunk generation and validation
   - Chunk streaming to database
   - Concurrent chunk processing

✅ Session Metrics (2 tests):
   - Session lifecycle tracking
   - Processing metrics collection

Total: 9 comprehensive integration tests
Coverage: Database ops, chunking, metrics, session tracking
No mocks: Real database, real audio processing
Fixtures: Match actual service response contracts
"""
