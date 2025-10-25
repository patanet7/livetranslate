#!/usr/bin/env python3
"""
Unit Tests for WhisperService Helper Methods

Tests helper methods that don't require model loading:
- _detect_hallucination()
- _find_stable_word_prefix()
- _calculate_text_stability_score()
- _segments_len()

Following TDD principle: Test current behavior BEFORE refactoring
NO MOCKS - Testing actual implementation behavior
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path (adjusted for tests/unit/ location)
SRC_DIR = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from whisper_service import WhisperService, TranscriptionRequest, TranscriptionResult


class TestHallucinationDetection:
    """Test hallucination detection logic"""

    def test_empty_text_is_hallucination(self):
        """Empty or whitespace-only text should be flagged"""
        service = WhisperService(config={"orchestration_mode": True})

        assert service._detect_hallucination("", 0.8) is True
        assert service._detect_hallucination("   ", 0.8) is True
        assert service._detect_hallucination("\n", 0.8) is True

    def test_short_text_is_hallucination(self):
        """Very short text (< 2 chars after strip) should be flagged"""
        service = WhisperService(config={"orchestration_mode": True})

        assert service._detect_hallucination("a", 0.8) is True
        assert service._detect_hallucination(" b ", 0.8) is True

    def test_repetitive_characters_are_hallucination(self):
        """Repetitive character patterns should be flagged"""
        service = WhisperService(config={"orchestration_mode": True})

        assert service._detect_hallucination("aaaa", 0.8) is True
        assert service._detect_hallucination("bbbb", 0.8) is True
        assert service._detect_hallucination("cccc hello world", 0.8) is True

    def test_whisper_artifacts_are_hallucination(self):
        """Known Whisper hallucination patterns should be flagged"""
        service = WhisperService(config={"orchestration_mode": True})

        assert service._detect_hallucination("mbc 뉴스", 0.8) is True
        assert service._detect_hallucination("김정진입니다", 0.8) is True
        assert service._detect_hallucination("thanks for watching our channel", 0.8) is True

    def test_excessive_word_repetition(self):
        """Text with >80% word repetition should be flagged"""
        service = WhisperService(config={"orchestration_mode": True})

        # 10 words, only 1 unique (10% unique = below 20% threshold)
        repetitive = "the the the the the the the the the the"
        assert service._detect_hallucination(repetitive, 0.8) is True

    def test_educational_content_not_hallucination(self):
        """Educational phrases should NOT be flagged"""
        service = WhisperService(config={"orchestration_mode": True})

        assert service._detect_hallucination("Let's practice this English phrase", 0.8) is False
        assert service._detect_hallucination("This is a language learning exercise", 0.8) is False
        assert service._detect_hallucination("Try to get in shape with vocabulary", 0.8) is False

    def test_normal_speech_not_hallucination(self):
        """Normal speech should NOT be flagged"""
        service = WhisperService(config={"orchestration_mode": True})

        assert service._detect_hallucination("Hello world, how are you today?", 0.8) is False
        assert service._detect_hallucination("The quick brown fox jumps over the lazy dog", 0.9) is False
        assert service._detect_hallucination("I would like to order a coffee please", 0.7) is False

    def test_single_character_repetition(self):
        """Text with very few unique characters should be flagged"""
        service = WhisperService(config={"orchestration_mode": True})

        # Long text but only 2 unique characters (excluding spaces)
        assert service._detect_hallucination("aaa bbb aaa bbb aaa", 0.8) is True


class TestStabilityTracking:
    """Test stability tracking helper methods"""

    def test_empty_history_returns_empty_prefix(self):
        """Empty history should return empty stable prefix"""
        service = WhisperService(config={"orchestration_mode": True})

        text_history: List[Tuple[str, float]] = []
        current_text = "hello world"

        stable_prefix = service._find_stable_word_prefix(text_history, current_text)
        assert stable_prefix == ""

    def test_single_item_history_returns_empty_prefix(self):
        """Single item in history should return empty prefix (need at least 2 for stability)"""
        service = WhisperService(config={"orchestration_mode": True})

        text_history = [("hello", 1.0)]
        current_text = "hello world"

        stable_prefix = service._find_stable_word_prefix(text_history, current_text)
        assert stable_prefix == ""

    def test_consistent_prefix_detected(self):
        """Consistent word prefix across multiple texts should be detected"""
        service = WhisperService(config={"orchestration_mode": True})

        text_history = [
            ("hello world", 1.0),
            ("hello world this", 2.0),
            ("hello world this is", 3.0),
        ]
        current_text = "hello world this is a test"

        stable_prefix = service._find_stable_word_prefix(text_history, current_text)

        # Should detect "hello world this is" as stable (appears in at least 2 recent texts)
        assert "hello world" in stable_prefix
        assert "hello world this" in stable_prefix

    def test_unstable_words_not_included(self):
        """Words that change position should not be in stable prefix"""
        service = WhisperService(config={"orchestration_mode": True})

        text_history = [
            ("hello world", 1.0),
            ("hello there", 2.0),
            ("hello friend", 3.0),
        ]
        current_text = "hello world again"

        stable_prefix = service._find_stable_word_prefix(text_history, current_text)

        # Only "hello" should be stable (consistent position)
        # "world" appears but not in consistent position
        assert stable_prefix == "hello"

    def test_empty_current_text_returns_empty(self):
        """Empty current text should return empty stable prefix"""
        service = WhisperService(config={"orchestration_mode": True})

        text_history = [("hello", 1.0), ("hello world", 2.0)]
        current_text = ""

        stable_prefix = service._find_stable_word_prefix(text_history, current_text)
        assert stable_prefix == ""

    def test_stability_score_calculation(self):
        """Test stability score calculation from text history"""
        service = WhisperService(config={"orchestration_mode": True})

        text_history = [
            ("hello world", 1.0),
            ("hello world this", 2.0),
            ("hello world this is", 3.0),
        ]
        stable_prefix = "hello world"

        score = service._calculate_text_stability_score(text_history, stable_prefix)

        # Score should be between 0.0 and 1.0
        assert 0.0 <= score <= 1.0
        # High consistency should result in higher score
        assert score > 0.5

    def test_stability_score_empty_history(self):
        """Empty history should return 0 stability score"""
        service = WhisperService(config={"orchestration_mode": True})

        text_history: List[Tuple[str, float]] = []
        stable_prefix = ""

        score = service._calculate_text_stability_score(text_history, stable_prefix)
        assert score == 0.0


class TestSessionBuffering:
    """Test session-specific audio buffering"""

    def test_segments_len_empty_session(self):
        """Empty session should return 0 duration"""
        service = WhisperService(config={"orchestration_mode": False})

        duration = service._segments_len("test-session")
        assert duration == 0.0

    def test_segments_len_with_audio(self):
        """Test duration calculation from audio segments"""
        service = WhisperService(config={"orchestration_mode": False})

        session_id = "test-session-audio"

        # Add 1 second of audio (16000 samples @ 16kHz)
        audio_chunk = np.zeros(16000, dtype=np.float32)
        service.add_audio_chunk(audio_chunk, session_id=session_id, enable_vad_prefilter=False)

        duration = service._segments_len(session_id)

        # Should be approximately 1.0 second
        assert 0.9 <= duration <= 1.1

    def test_segments_len_multiple_chunks(self):
        """Test duration calculation with multiple chunks"""
        service = WhisperService(config={"orchestration_mode": False})

        session_id = "test-session-multi"

        # Add 3 x 1 second chunks
        for _ in range(3):
            audio_chunk = np.zeros(16000, dtype=np.float32)
            service.add_audio_chunk(audio_chunk, session_id=session_id, enable_vad_prefilter=False)

        duration = service._segments_len(session_id)

        # Should be approximately 3.0 seconds
        assert 2.9 <= duration <= 3.1


class TestDataClasses:
    """Test TranscriptionRequest and TranscriptionResult dataclasses"""

    def test_transcription_request_defaults(self):
        """Test TranscriptionRequest default values"""
        audio = np.zeros(16000, dtype=np.float32)
        request = TranscriptionRequest(audio_data=audio)

        assert request.model_name == "whisper-large-v3"
        assert request.language is None
        assert request.streaming is False
        assert request.sample_rate == 16000
        assert request.enable_vad is True
        assert request.beam_size == 5
        assert request.temperature == 0.0
        assert request.streaming_policy == "alignatt"
        assert request.task == "transcribe"
        assert request.target_language == "en"

    def test_transcription_request_custom_values(self):
        """Test TranscriptionRequest with custom values"""
        audio = np.zeros(16000, dtype=np.float32)
        request = TranscriptionRequest(
            audio_data=audio,
            model_name="base",
            language="en",
            streaming=True,
            beam_size=1,
            session_id="test-123"
        )

        assert request.model_name == "base"
        assert request.language == "en"
        assert request.streaming is True
        assert request.beam_size == 1
        assert request.session_id == "test-123"

    def test_transcription_result_initialization(self):
        """Test TranscriptionResult initialization"""
        result = TranscriptionResult(
            text="Hello world",
            segments=[],
            language="en",
            confidence_score=0.95,
            processing_time=1.5,
            model_used="base",
            device_used="cpu"
        )

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence_score == 0.95
        assert result.processing_time == 1.5
        assert result.model_used == "base"
        assert result.device_used == "cpu"
        assert result.timestamp is not None  # Auto-generated

    def test_transcription_result_phase3_fields(self):
        """Test Phase 3 stability tracking fields"""
        result = TranscriptionResult(
            text="Hello world this is a test",
            segments=[],
            language="en",
            confidence_score=0.9,
            processing_time=1.0,
            model_used="base",
            device_used="cpu",
            stable_text="Hello world",
            unstable_text="this is a test",
            is_draft=True,
            is_final=False,
            should_translate=True,
            translation_mode="draft",
            stability_score=0.85
        )

        assert result.stable_text == "Hello world"
        assert result.unstable_text == "this is a test"
        assert result.is_draft is True
        assert result.is_final is False
        assert result.should_translate is True
        assert result.translation_mode == "draft"
        assert result.stability_score == 0.85


class TestWhisperServiceInitialization:
    """Test WhisperService initialization"""

    def test_initialization_default_config(self):
        """Test service initialization with default config"""
        service = WhisperService(config={"orchestration_mode": True})

        assert service.orchestration_mode is True
        assert service.model_manager is not None
        assert service.session_manager is not None
        assert not service.streaming_active
        assert len(service.session_audio_buffers) == 0

    def test_initialization_custom_config(self):
        """Test service initialization with custom config"""
        config = {
            "orchestration_mode": False,
            "inference_interval": 5.0,
            "models_dir": ".models"
        }
        service = WhisperService(config=config)

        assert service.orchestration_mode is False
        assert service.inference_interval == 5.0

    def test_service_status(self):
        """Test get_service_status returns correct information"""
        service = WhisperService(config={"orchestration_mode": True})

        status = service.get_service_status()

        assert "device" in status
        assert "loaded_models" in status
        assert "available_models" in status
        assert "streaming_active" in status
        assert "orchestration_mode" in status
        assert status["orchestration_mode"] is True
        assert status["streaming_active"] is False


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
