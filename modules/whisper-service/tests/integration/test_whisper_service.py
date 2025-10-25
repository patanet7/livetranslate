#!/usr/bin/env python3
"""
Integration Tests for WhisperService

Tests transcription functionality with REAL models and REAL audio.
NO MOCKS - Full end-to-end testing of transcription pipeline.

Following TDD principle: Test current behavior BEFORE refactoring.
"""

import pytest
import pytest_asyncio
import asyncio
import numpy as np
import sys
from pathlib import Path
from typing import List

# Add src to path (adjusted for tests/integration/ location)
SRC_DIR = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from whisper_service import (
    WhisperService,
    TranscriptionRequest,
    TranscriptionResult,
    create_whisper_service
)

# Root directory for test fixtures
ROOT_DIR = Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def whisper_service():
    """Create WhisperService instance for tests (reuse across module)"""
    config = {
        "orchestration_mode": False,
        "models_dir": str(ROOT_DIR / ".models")
    }
    service = WhisperService(config=config)
    yield service
    # Cleanup after tests
    asyncio.run(service.shutdown())


@pytest.fixture
def jfk_audio():
    """Load JFK audio file for testing"""
    import soundfile as sf

    jfk_path = ROOT_DIR / "jfk.wav"
    if not jfk_path.exists():
        pytest.skip(f"JFK audio file not found: {jfk_path}")

    audio_data, sample_rate = sf.read(jfk_path, dtype='float32')

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    return audio_data, sample_rate


@pytest.fixture
def test_audio_1sec():
    """Generate 1 second of test audio (low-amplitude sine wave)"""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * frequency * t)

    return audio.astype(np.float32)


class TestWhisperServiceTranscription:
    """Test basic transcription functionality"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_jfk_audio(self, whisper_service, jfk_audio):
        """Test transcription of JFK audio file"""
        audio_data, sample_rate = jfk_audio

        request = TranscriptionRequest(
            audio_data=audio_data,
            model_name="base",  # Use base model for speed
            language="en",
            streaming=False,
            enable_vad=True
        )

        result = await whisper_service.transcribe(request)

        # Verify result structure
        assert isinstance(result, TranscriptionResult)
        assert result.text is not None
        assert len(result.text) > 0
        assert result.language in ["en", "english"]
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.processing_time > 0.0
        assert result.model_used == "base"
        assert result.device_used in ["cuda", "mps", "cpu"]

        # Check for JFK-related keywords
        text_lower = result.text.lower()
        jfk_keywords = ["country", "ask", "fellow"]
        found_keywords = [kw for kw in jfk_keywords if kw in text_lower]

        print(f"\n✅ Transcription: '{result.text}'")
        print(f"   Found keywords: {found_keywords}")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Processing time: {result.processing_time:.2f}s")

        # Should find at least one JFK keyword
        assert len(found_keywords) > 0, f"Expected JFK keywords, got: {result.text}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_with_beam_search(self, whisper_service, test_audio_1sec):
        """Test transcription with beam search"""
        request = TranscriptionRequest(
            audio_data=test_audio_1sec,
            model_name="base",
            language="en",
            beam_size=5,  # Use beam search
            temperature=0.0  # Deterministic
        )

        result = await whisper_service.transcribe(request)

        assert isinstance(result, TranscriptionResult)
        assert result.model_used == "base"
        print(f"\n✅ Beam search result: '{result.text}'")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_with_greedy_decoding(self, whisper_service, test_audio_1sec):
        """Test transcription with greedy decoding (beam_size=1)"""
        request = TranscriptionRequest(
            audio_data=test_audio_1sec,
            model_name="base",
            language="en",
            beam_size=1,  # Greedy decoding
            temperature=0.0
        )

        result = await whisper_service.transcribe(request)

        assert isinstance(result, TranscriptionResult)
        assert result.model_used == "base"
        print(f"\n✅ Greedy result: '{result.text}'")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcribe_with_domain_prompt(self, whisper_service, jfk_audio):
        """Test transcription with domain-specific prompt"""
        audio_data, sample_rate = jfk_audio

        request = TranscriptionRequest(
            audio_data=audio_data,
            model_name="base",
            language="en",
            initial_prompt="Political speech about American values.",
            domain="political"
        )

        result = await whisper_service.transcribe(request)

        assert isinstance(result, TranscriptionResult)
        assert len(result.text) > 0
        print(f"\n✅ With domain prompt: '{result.text}'")


class TestWhisperServiceStreaming:
    """Test streaming transcription functionality"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_streaming_basic(self, whisper_service, jfk_audio):
        """Test basic streaming transcription"""
        audio_data, sample_rate = jfk_audio

        request = TranscriptionRequest(
            audio_data=audio_data,
            model_name="base",
            language="en",
            streaming=True,
            enable_vad=True,
            session_id="test-stream-1"
        )

        results = []
        async for result in whisper_service.transcribe_stream(request):
            results.append(result)
            print(f"\n[Stream] Text: '{result.text[:60]}'")
            print(f"         Stable: '{result.stable_text[:40]}'")
            print(f"         Draft: {result.is_draft}, Final: {result.is_final}")

            # Limit results for testing
            if len(results) >= 3:
                break

        assert len(results) > 0
        # At least one result should have text
        assert any(len(r.text) > 0 for r in results)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_streaming_with_chunks(self, whisper_service):
        """Test streaming with multiple audio chunks"""
        session_id = "test-stream-chunks"

        # Create session
        whisper_service.create_session(session_id)

        # Add 3 chunks of audio
        for i in range(3):
            audio_chunk = np.random.randn(16000).astype(np.float32) * 0.01
            chunk_count = whisper_service.add_audio_chunk(
                audio_chunk,
                session_id=session_id,
                enable_vad_prefilter=False  # Accept all chunks for testing
            )
            print(f"\n[Chunk {i+1}] Buffer size: {chunk_count}")

        # Verify chunks were added
        duration = whisper_service._segments_len(session_id)
        assert duration > 0.0
        print(f"✅ Total duration: {duration:.2f}s")

        # Cleanup
        whisper_service.close_session(session_id)


class TestWhisperServiceSessions:
    """Test session management"""

    def test_create_session(self, whisper_service):
        """Test session creation"""
        session_id = "test-session-create"

        session = whisper_service.create_session(session_id)

        assert session is not None
        assert session["session_id"] == session_id
        assert "created_at" in session

    def test_get_session(self, whisper_service):
        """Test getting session information"""
        session_id = "test-session-get"

        # Create session
        whisper_service.create_session(session_id)

        # Get session
        session = whisper_service.get_session(session_id)

        assert session is not None
        assert session["session_id"] == session_id

    def test_close_session(self, whisper_service):
        """Test session closure and cleanup"""
        session_id = "test-session-close"

        # Create session and add audio
        whisper_service.create_session(session_id)
        audio_chunk = np.zeros(16000, dtype=np.float32)
        whisper_service.add_audio_chunk(audio_chunk, session_id=session_id, enable_vad_prefilter=False)

        # Verify audio was added
        assert session_id in whisper_service.session_audio_buffers

        # Close session
        closed_session = whisper_service.close_session(session_id)

        assert closed_session is not None

        # Verify cleanup
        assert session_id not in whisper_service.session_audio_buffers
        assert session_id not in whisper_service.session_vad_states
        assert session_id not in whisper_service.session_stability_trackers

    def test_multiple_sessions_isolated(self, whisper_service):
        """Test that multiple sessions are properly isolated"""
        session1 = "test-multi-1"
        session2 = "test-multi-2"

        # Create both sessions
        whisper_service.create_session(session1)
        whisper_service.create_session(session2)

        # Add different amounts of audio to each
        audio1 = np.zeros(16000, dtype=np.float32)
        audio2 = np.zeros(32000, dtype=np.float32)

        whisper_service.add_audio_chunk(audio1, session_id=session1, enable_vad_prefilter=False)
        whisper_service.add_audio_chunk(audio2, session_id=session2, enable_vad_prefilter=False)

        # Verify isolation
        duration1 = whisper_service._segments_len(session1)
        duration2 = whisper_service._segments_len(session2)

        assert 0.9 <= duration1 <= 1.1  # ~1 second
        assert 1.9 <= duration2 <= 2.1  # ~2 seconds

        # Cleanup
        whisper_service.close_session(session1)
        whisper_service.close_session(session2)


class TestWhisperServiceOrchestrationMode:
    """Test orchestration mode functionality"""

    @pytest.fixture
    def orchestration_service(self):
        """Create service in orchestration mode"""
        config = {
            "orchestration_mode": True,
            "models_dir": str(ROOT_DIR / ".models")
        }
        service = WhisperService(config=config)
        yield service
        asyncio.run(service.shutdown())

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_orchestration_chunk_processing(self, orchestration_service):
        """Test processing chunks in orchestration mode"""
        audio_chunk = np.random.randn(16000).astype(np.float32) * 0.01
        audio_bytes = audio_chunk.tobytes()

        result = await orchestration_service.process_orchestration_chunk(
            chunk_id="chunk-001",
            session_id="orch-session-1",
            audio_data=audio_bytes,
            chunk_metadata={
                "sequence": 1,
                "timestamp": 0.0,
                "duration": 1.0
            },
            model_name="base"
        )

        assert result is not None
        assert result["chunk_id"] == "chunk-001"
        assert result["session_id"] == "orch-session-1"
        assert "status" in result
        assert "text" in result
        assert "processing_time" in result

        print(f"\n✅ Orchestration result: {result['status']}")
        print(f"   Text: '{result.get('text', 'N/A')}'")

    def test_add_audio_chunk_blocked_in_orchestration_mode(self, orchestration_service):
        """Test that add_audio_chunk is blocked in orchestration mode"""
        audio_chunk = np.zeros(16000, dtype=np.float32)

        # Should return 0 (blocked)
        result = orchestration_service.add_audio_chunk(
            audio_chunk,
            session_id="test",
            enable_vad_prefilter=False
        )

        assert result == 0


class TestWhisperServiceUtilities:
    """Test utility methods"""

    def test_get_available_models(self, whisper_service):
        """Test getting list of available models"""
        models = whisper_service.get_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # Should include common models
        common_models = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
        assert any(model in models for model in common_models)

    def test_get_service_status(self, whisper_service):
        """Test getting service status"""
        status = whisper_service.get_service_status()

        assert isinstance(status, dict)
        assert "device" in status
        assert "loaded_models" in status
        assert "available_models" in status
        assert "streaming_active" in status
        assert "orchestration_mode" in status
        assert "sessions" in status

        print(f"\n✅ Service status:")
        print(f"   Device: {status['device']}")
        print(f"   Loaded models: {status['loaded_models']}")
        print(f"   Orchestration mode: {status['orchestration_mode']}")

    def test_clear_cache(self, whisper_service):
        """Test clearing model cache"""
        # Load a model first
        request = TranscriptionRequest(
            audio_data=np.zeros(16000, dtype=np.float32),
            model_name="base"
        )
        asyncio.run(whisper_service.transcribe(request))

        # Clear cache (should not raise exception)
        whisper_service.clear_cache()

        print(f"\n✅ Cache cleared successfully")

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test service shutdown"""
        config = {
            "orchestration_mode": False,
            "models_dir": str(ROOT_DIR / ".models")
        }
        service = WhisperService(config=config)

        # Add some audio to create sessions
        service.create_session("test-shutdown")
        audio_chunk = np.zeros(16000, dtype=np.float32)
        service.add_audio_chunk(audio_chunk, session_id="test-shutdown", enable_vad_prefilter=False)

        # Shutdown (should not raise exception)
        await service.shutdown()

        # Verify cleanup
        assert len(service.session_audio_buffers) == 0
        assert not service.streaming_active

        print(f"\n✅ Service shutdown successfully")


class TestWhisperServiceFactoryFunction:
    """Test factory function"""

    @pytest.mark.asyncio
    async def test_create_whisper_service(self):
        """Test factory function creates service correctly"""
        config = {
            "orchestration_mode": True,
            "models_dir": str(ROOT_DIR / ".models")
        }

        service = await create_whisper_service(config)

        assert isinstance(service, WhisperService)
        assert service.orchestration_mode is True

        await service.shutdown()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
