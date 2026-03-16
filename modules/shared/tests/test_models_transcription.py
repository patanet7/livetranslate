"""Behavioral tests for TranscriptionResult, Segment, and ModelInfo models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from livetranslate_common.models.transcription import ModelInfo, Segment, TranscriptionResult


class TestSegment:
    def test_segment_creation(self) -> None:
        seg = Segment(text="Hello world", start_ms=0, end_ms=1500, confidence=0.95)
        assert seg.text == "Hello world"
        assert seg.start_ms == 0
        assert seg.end_ms == 1500
        assert seg.confidence == 0.95
        assert seg.speaker_id is None

    def test_segment_with_speaker(self) -> None:
        seg = Segment(
            text="Good morning",
            start_ms=500,
            end_ms=2000,
            confidence=0.88,
            speaker_id="SPEAKER_01",
        )
        assert seg.speaker_id == "SPEAKER_01"

    def test_segment_duration_ms(self) -> None:
        seg = Segment(text="Test", start_ms=1000, end_ms=3500, confidence=0.9)
        assert seg.duration_ms == 2500

    def test_segment_confidence_bounds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Segment(text="Bad", start_ms=0, end_ms=100, confidence=1.5)

        with pytest.raises(ValidationError):
            Segment(text="Bad", start_ms=0, end_ms=100, confidence=-0.1)

    def test_segment_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError, match="end_ms"):
            Segment(text="Bad", start_ms=500, end_ms=100, confidence=0.9)


class TestTranscriptionResult:
    def test_minimal_result(self) -> None:
        result = TranscriptionResult(text="Hello", language="en", confidence=0.9)
        assert result.text == "Hello"
        assert result.language == "en"
        assert result.confidence == 0.9
        assert result.segments == []
        assert result.stable_text == ""
        assert result.unstable_text == ""
        assert result.is_final is False
        assert result.is_draft is True
        assert result.speaker_id is None
        assert result.should_translate is False
        assert result.context_text == ""

    def test_final_result_with_segments(self) -> None:
        seg = Segment(text="Hello", start_ms=0, end_ms=500, confidence=0.97)
        result = TranscriptionResult(
            text="Hello there",
            language="en",
            confidence=0.95,
            segments=[seg],
            stable_text="Hello there",
            unstable_text="",
            is_final=True,
            is_draft=False,
            speaker_id="SPEAKER_00",
            should_translate=True,
            context_text="Previous sentence.",
        )
        assert result.is_final is True
        assert result.is_draft is False
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello"
        assert result.speaker_id == "SPEAKER_00"
        assert result.should_translate is True
        assert result.context_text == "Previous sentence."

    def test_roundtrip_json(self) -> None:
        seg = Segment(text="Round trip", start_ms=100, end_ms=900, confidence=0.85)
        original = TranscriptionResult(
            text="Round trip",
            language="fr",
            confidence=0.85,
            segments=[seg],
            is_final=True,
        )
        json_str = original.model_dump_json()
        restored = TranscriptionResult.model_validate_json(json_str)
        assert restored.text == original.text
        assert restored.language == original.language
        assert restored.confidence == original.confidence
        assert restored.is_final == original.is_final
        assert len(restored.segments) == 1
        assert restored.segments[0].text == "Round trip"


class TestModelInfo:
    def test_model_info(self) -> None:
        info = ModelInfo(
            name="whisper-base",
            backend="faster-whisper",
            languages=["en", "fr", "de", "es"],
            vram_mb=1024,
            compute_type="int8",
        )
        assert info.name == "whisper-base"
        assert info.backend == "faster-whisper"
        assert "en" in info.languages
        assert info.vram_mb == 1024
        assert info.compute_type == "int8"


class TestTranscriptionResultNoSpeechProbDefault:
    """Verify no_speech_prob field defaults and contract."""

    def test_no_speech_prob_defaults_to_none(self) -> None:
        """TranscriptionResult created without no_speech_prob must have it as None."""
        result = TranscriptionResult(text="x", language="en", confidence=0.5)
        assert result.no_speech_prob is None

    def test_no_speech_prob_can_be_set_to_float(self) -> None:
        """no_speech_prob must accept a float value."""
        result = TranscriptionResult(
            text="hello", language="en", confidence=0.9, no_speech_prob=0.2
        )
        assert result.no_speech_prob == pytest.approx(0.2)

    def test_no_speech_prob_survives_roundtrip(self) -> None:
        """no_speech_prob must be preserved through JSON serialisation round-trip."""
        result = TranscriptionResult(
            text="hello", language="en", confidence=0.9, no_speech_prob=0.75
        )
        restored = TranscriptionResult.model_validate_json(result.model_dump_json())
        assert restored.no_speech_prob == pytest.approx(0.75)

    def test_no_speech_prob_none_survives_roundtrip(self) -> None:
        """no_speech_prob=None must survive JSON serialisation round-trip."""
        result = TranscriptionResult(text="hi", language="en", confidence=0.8)
        restored = TranscriptionResult.model_validate_json(result.model_dump_json())
        assert restored.no_speech_prob is None


class TestCompressionRatioField:
    """Verify compression_ratio field defaults and contract."""

    def test_compression_ratio_defaults_to_none(self) -> None:
        result = TranscriptionResult(text="x", language="en", confidence=0.5)
        assert result.compression_ratio is None

    def test_compression_ratio_can_be_set_to_float(self) -> None:
        result = TranscriptionResult(
            text="hello", language="en", confidence=0.9, compression_ratio=2.4
        )
        assert result.compression_ratio == pytest.approx(2.4)

    def test_compression_ratio_survives_roundtrip(self) -> None:
        result = TranscriptionResult(
            text="hello", language="en", confidence=0.9, compression_ratio=1.8
        )
        restored = TranscriptionResult.model_validate_json(result.model_dump_json())
        assert restored.compression_ratio == pytest.approx(1.8)

    def test_compression_ratio_none_survives_roundtrip(self) -> None:
        result = TranscriptionResult(text="hi", language="en", confidence=0.8)
        restored = TranscriptionResult.model_validate_json(result.model_dump_json())
        assert restored.compression_ratio is None
