#!/usr/bin/env python3
"""
REAL Speech Integration Tests - Simple and Fast

Tests using actual speech audio (JFK, Chinese) to verify
transcription quality with REAL Whisper models.

These tests are simpler and faster than the comprehensive test suite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from whisper_service import TranscriptionRequest, WhisperService

# Mark as slow integration tests
pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.asyncio]


class TestJFKSpeech:
    """Test English speech transcription with JFK audio"""

    @pytest.fixture(scope="class")
    def service(self):
        """Create WhisperService for testing"""
        config = {
            "models_dir": ".models/pytorch",
            "device": "cpu",  # Use CPU for consistency
            "model_name": "tiny",  # Fast model for testing
        }
        return WhisperService(config=config)

    async def test_jfk_transcription_contains_famous_quote(self, service, jfk_audio):
        """Test that JFK audio transcribes the famous quote"""
        audio, sr = jfk_audio

        print(f"\n🎤 Transcribing JFK speech ({len(audio)/sr:.1f}s at {sr}Hz)...")

        # Create transcription request
        request = TranscriptionRequest(
            audio_data=audio, model_name="tiny", language="en", sample_rate=sr
        )

        # Transcribe
        result = await service.transcribe(request)

        assert result is not None
        assert hasattr(result, "text")

        text = result.text.lower()
        print(f"📝 Transcription: {result.text}")

        # Check for key phrases from the quote
        # "Ask not what your country can do for you; ask what you can do for your country"
        key_phrases = ["ask not", "country", "can do"]

        found_phrases = [phrase for phrase in key_phrases if phrase in text]

        print(f"✅ Found {len(found_phrases)}/{len(key_phrases)} key phrases: {found_phrases}")

        # Should find at least 2 out of 3 key phrases
        assert len(found_phrases) >= 2, f"Only found {found_phrases} in: {text}"

    async def test_jfk_language_detected_as_english(self, service, jfk_audio):
        """Test that JFK audio is detected as English"""
        audio, sr = jfk_audio

        print("\n🔍 Testing language detection...")

        request = TranscriptionRequest(audio_data=audio, model_name="tiny", sample_rate=sr)

        result = await service.transcribe(request)

        # Check if language is in result
        if hasattr(result, "language") and result.language:
            detected_lang = result.language
            print(f"✅ Detected language: {detected_lang}")
            assert (
                detected_lang == "en" or detected_lang == "english"
            ), f"Expected English, got {detected_lang}"
        else:
            print("⚠️  No language field in result (may not be supported)")


class TestChineseSpeech:
    """Test Chinese speech transcription"""

    @pytest.fixture(scope="class")
    def service(self):
        """Create WhisperService for testing"""
        config = {"models_dir": ".models/pytorch", "device": "cpu", "model_name": "tiny"}
        return WhisperService(config=config)

    async def test_chinese_transcription_produces_text(self, service, chinese_audio_1):
        """Test that Chinese audio produces transcribed text"""
        audio, sr = chinese_audio_1

        print(f"\n🎤 Transcribing Chinese speech ({len(audio)/sr:.1f}s at {sr}Hz)...")

        # Transcribe with language hint
        request = TranscriptionRequest(
            audio_data=audio, model_name="tiny", language="zh", sample_rate=sr
        )

        result = await service.transcribe(request)

        assert result is not None
        assert hasattr(result, "text")
        assert len(result.text) > 0, "Transcription should not be empty"

        print(f"📝 Transcription: {result.text}")
        print(f"✅ Produced {len(result.text)} characters")

    async def test_chinese_transcription_contains_chinese_characters(
        self, service, chinese_audio_1
    ):
        """Test that Chinese transcription contains Chinese characters"""
        audio, sr = chinese_audio_1

        request = TranscriptionRequest(
            audio_data=audio, model_name="tiny", language="zh", sample_rate=sr
        )

        result = await service.transcribe(request)

        text = result.text

        # Check for Chinese characters (Unicode range)
        has_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)

        print(f"📝 Text: {text}")
        print(f"✅ Contains Chinese characters: {has_chinese}")

        assert has_chinese, f"Expected Chinese characters in: {text}"


class TestMultiLanguage:
    """Test multi-language capabilities"""

    @pytest.fixture(scope="class")
    def service(self):
        """Create WhisperService for testing"""
        config = {"models_dir": ".models/pytorch", "device": "cpu", "model_name": "tiny"}
        return WhisperService(config=config)

    async def test_english_and_chinese_in_same_session(self, service, jfk_audio, chinese_audio_1):
        """Test transcribing different languages in same session"""
        print("\n🌐 Testing multi-language session...")

        # Transcribe English
        jfk, sr_en = jfk_audio
        request_en = TranscriptionRequest(
            audio_data=jfk, model_name="tiny", language="en", sample_rate=sr_en
        )
        result_en = await service.transcribe(request_en)

        assert hasattr(result_en, "text")
        print(f"📝 English: {result_en.text[:50]}...")

        # Transcribe Chinese
        chinese, sr_zh = chinese_audio_1
        request_zh = TranscriptionRequest(
            audio_data=chinese, model_name="tiny", language="zh", sample_rate=sr_zh
        )
        result_zh = await service.transcribe(request_zh)

        assert hasattr(result_zh, "text")
        print(f"📝 Chinese: {result_zh.text[:50]}...")

        # Both should succeed
        assert len(result_en.text) > 0
        assert len(result_zh.text) > 0

        print("✅ Successfully transcribed both languages")


# Usage:
_USAGE_INSTRUCTIONS = """
To run these tests:

# Run all real speech tests
pytest tests/integration/test_real_speech.py -v

# Run just JFK tests
pytest tests/integration/test_real_speech.py::TestJFKSpeech -v

# Run just Chinese tests
pytest tests/integration/test_real_speech.py::TestChineseSpeech -v

# Run with output visible
pytest tests/integration/test_real_speech.py -v -s

# Skip slow tests
pytest tests/integration/ -v -m "not slow"
"""
