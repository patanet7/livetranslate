"""Cross-service contract round-trip tests.

Validates that Pydantic models used across service boundaries can be
transformed into each other without data loss. All tests use real model
instances -- no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from livetranslate_common.models.audio import AudioChunk
from livetranslate_common.models.transcription import Segment, TranscriptionResult
from livetranslate_common.models.translation import TranslationRequest, TranslationResponse
from livetranslate_common.models.ws_messages import SegmentMessage, TranslationMessage


class TestTranscriptionToSegmentMessage:
    """Verify TranscriptionResult fields map correctly to SegmentMessage."""

    def _build_transcription_result(
        self,
        *,
        segments: list[Segment] | None = None,
        speaker_id: str | None = "speaker_0",
    ) -> TranscriptionResult:
        if segments is None:
            segments = [Segment(text="Hello world", start_ms=0, end_ms=1500, confidence=0.95)]
        return TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            segments=segments,
            stable_text="Hello world",
            unstable_text="",
            is_final=True,
            speaker_id=speaker_id,
        )

    def _transform_to_segment_message(self, result: TranscriptionResult, segment_id: int) -> SegmentMessage:
        """Perform the same transformation the orchestration service applies."""
        data = result.model_dump(
            include={"text", "language", "confidence", "is_final", "segments", "stable_text", "unstable_text", "speaker_id"},
        )
        data["start_ms"] = result.segments[0].start_ms if result.segments else None
        data["end_ms"] = result.segments[-1].end_ms if result.segments else None
        data["segment_id"] = segment_id
        # segments is not a SegmentMessage field; remove it before validation
        data.pop("segments", None)
        return SegmentMessage.model_validate(data)

    def test_transcription_result_to_segment_message_roundtrip(self) -> None:
        result = self._build_transcription_result()
        msg = self._transform_to_segment_message(result, segment_id=1)

        assert msg.segment_id == 1
        assert msg.text == result.text
        assert msg.language == result.language
        assert msg.confidence == result.confidence
        assert msg.stable_text == result.stable_text
        assert msg.unstable_text == result.unstable_text
        assert msg.is_final == result.is_final
        assert msg.speaker_id == result.speaker_id
        assert msg.start_ms == result.segments[0].start_ms
        assert msg.end_ms == result.segments[-1].end_ms
        assert msg.type == "segment"

    def test_empty_segments_produce_none_timestamps(self) -> None:
        result = self._build_transcription_result(segments=[])  # explicitly empty
        msg = self._transform_to_segment_message(result, segment_id=2)

        assert msg.start_ms is None
        assert msg.end_ms is None
        assert msg.text == "Hello world"

    def test_segment_timing_extraction_uses_first_and_last(self) -> None:
        segments = [
            Segment(text="Hello", start_ms=100, end_ms=500, confidence=0.9),
            Segment(text="beautiful", start_ms=500, end_ms=1200, confidence=0.85),
            Segment(text="world", start_ms=1200, end_ms=2000, confidence=0.92),
        ]
        result = self._build_transcription_result(segments=segments)
        msg = self._transform_to_segment_message(result, segment_id=3)

        assert msg.start_ms == 100
        assert msg.end_ms == 2000

    def test_no_speaker_id_passes_validation(self) -> None:
        result = self._build_transcription_result(speaker_id=None)
        msg = self._transform_to_segment_message(result, segment_id=4)

        assert msg.speaker_id is None

    @pytest.mark.parametrize("confidence", [0.0, 1.0])
    def test_confidence_boundary_values(self, confidence: float) -> None:
        result = TranscriptionResult(
            text="test",
            language="en",
            confidence=confidence,
            segments=[Segment(text="test", start_ms=0, end_ms=100, confidence=confidence)],
            stable_text="test",
            unstable_text="",
            is_final=True,
        )
        msg = self._transform_to_segment_message(result, segment_id=5)

        assert msg.confidence == confidence


class TestTranslationResponseToMessage:
    """Verify TranslationResponse maps correctly to TranslationMessage."""

    def test_translation_response_to_translation_message(self) -> None:
        response = TranslationResponse(
            translated_text="Hola mundo",
            source_language="en",
            target_language="es",
            model_used="qwen3.5:7b",
            latency_ms=150.0,
        )
        msg = TranslationMessage(
            text=response.translated_text,
            source_lang=response.source_language,
            target_lang=response.target_language,
            transcript_id=42,
            context_used=3,
        )

        assert msg.text == response.translated_text
        assert msg.text == "Hola mundo"
        assert msg.source_lang == response.source_language
        assert msg.target_lang == response.target_language
        assert msg.transcript_id == 42
        assert msg.context_used == 3
        assert msg.type == "translation"

    def test_field_name_mapping_is_intentional(self) -> None:
        """The field names differ between service boundary models on purpose:
        TranslationResponse.translated_text -> TranslationMessage.text
        TranslationResponse.source_language -> TranslationMessage.source_lang
        TranslationResponse.target_language -> TranslationMessage.target_lang
        """
        response_fields = set(TranslationResponse.model_fields.keys())
        message_fields = set(TranslationMessage.model_fields.keys())

        assert "translated_text" in response_fields
        assert "translated_text" not in message_fields
        assert "text" in message_fields

        assert "source_language" in response_fields
        assert "source_lang" in message_fields

        assert "target_language" in response_fields
        assert "target_lang" in message_fields


class TestAudioChunkBinaryRoundtrip:
    """Verify PCM audio data survives byte serialisation round-trips."""

    def test_float32_array_roundtrip(self) -> None:
        original = np.array([0.1, -0.5, 0.99, 0.0], dtype=np.float32)
        raw_bytes = original.tobytes()

        chunk = AudioChunk(
            data=raw_bytes,
            timestamp_ms=1000,
            sequence_number=0,
            source_id="mic-0",
        )

        restored = np.frombuffer(chunk.data, dtype=np.float32)
        np.testing.assert_array_almost_equal(original, restored)

    def test_empty_audio_chunk_roundtrip(self) -> None:
        original = np.array([], dtype=np.float32)
        raw_bytes = original.tobytes()

        chunk = AudioChunk(
            data=raw_bytes,
            timestamp_ms=0,
            sequence_number=0,
            source_id="test",
        )

        restored = np.frombuffer(chunk.data, dtype=np.float32)
        assert len(restored) == 0

    def test_large_audio_chunk_preserves_all_samples(self) -> None:
        sample_rate = 16000
        duration_s = 1.0
        num_samples = int(sample_rate * duration_s)
        original = np.random.default_rng(42).uniform(-1.0, 1.0, num_samples).astype(np.float32)

        chunk = AudioChunk(
            data=original.tobytes(),
            timestamp_ms=500,
            sequence_number=1,
            source_id="loopback",
        )

        restored = np.frombuffer(chunk.data, dtype=np.float32)
        np.testing.assert_array_equal(original, restored)
        assert len(restored) == num_samples


class TestTranslationRequestValidation:
    """Verify TranslationRequest constraints match what services expect."""

    def test_request_round_trips_through_model_dump(self) -> None:
        request = TranslationRequest(
            text="Hello world",
            source_language="en",
            target_language="es",
            context_window_size=5,
            max_context_tokens=500,
        )
        data = request.model_dump()
        restored = TranslationRequest.model_validate(data)

        assert restored.text == request.text
        assert restored.source_language == request.source_language
        assert restored.target_language == request.target_language
        assert restored.context_window_size == request.context_window_size
        assert restored.max_context_tokens == request.max_context_tokens

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(Exception):
            TranslationRequest(
                text="",
                source_language="en",
                target_language="es",
            )
