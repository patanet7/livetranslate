#!/usr/bin/env python3
"""
Unit Tests for Audio Models

Tests for all audio processing data models including validation,
serialization, and factory functions.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.audio.models import (
    AudioChunkingConfig,
    AudioChunkMetadata,
    AudioFormat,
    AudioStreamingSession,
    CorrelationType,
    ProcessingResult,
    ProcessingStatus,
    QualityMetrics,
    SourceType,
    SpeakerCorrelation,
    create_audio_chunk_metadata,
    create_processing_result,
    create_speaker_correlation,
    get_default_chunking_config,
)


class TestAudioChunkMetadata:
    """Test AudioChunkMetadata model."""

    def test_valid_chunk_metadata_creation(self):
        """Test creating valid chunk metadata."""
        metadata = create_audio_chunk_metadata(
            session_id="test_session",
            file_path="/test/path.wav",
            file_size=64000,
            duration_seconds=3.0,
            chunk_sequence=1,
            chunk_start_time=0.0,
            audio_quality_score=0.85,
        )

        assert metadata.session_id == "test_session"
        assert metadata.file_path == "/test/path.wav"
        assert metadata.file_size == 64000
        assert metadata.duration_seconds == 3.0
        assert metadata.chunk_sequence == 1
        assert metadata.chunk_start_time == 0.0
        assert metadata.chunk_end_time == 3.0
        assert metadata.audio_quality_score == 0.85
        assert metadata.file_format == AudioFormat.WAV
        assert metadata.processing_status == ProcessingStatus.PENDING

    def test_chunk_metadata_time_validation(self):
        """Test chunk metadata time validation."""
        # Test invalid time order
        with pytest.raises(ValidationError):
            AudioChunkMetadata(
                session_id="test",
                file_path="/test.wav",
                file_name="test.wav",
                file_size=1000,
                duration_seconds=3.0,
                chunk_sequence=1,
                chunk_start_time=5.0,  # Start after end
                chunk_end_time=2.0,  # End before start
            )

    def test_chunk_metadata_duration_consistency(self):
        """Test duration consistency validation."""
        # Test inconsistent duration
        with pytest.raises(ValidationError):
            AudioChunkMetadata(
                session_id="test",
                file_path="/test.wav",
                file_name="test.wav",
                file_size=1000,
                duration_seconds=5.0,  # Doesn't match timing
                chunk_sequence=1,
                chunk_start_time=0.0,
                chunk_end_time=3.0,  # Should be 3.0 duration
            )

    def test_chunk_metadata_quality_score_validation(self):
        """Test quality score validation."""
        # Test invalid quality score
        with pytest.raises(ValidationError):
            AudioChunkMetadata(
                session_id="test",
                file_path="/test.wav",
                file_name="test.wav",
                file_size=1000,
                duration_seconds=3.0,
                chunk_sequence=1,
                chunk_start_time=0.0,
                chunk_end_time=3.0,
                audio_quality_score=1.5,  # > 1.0
            )

    def test_chunk_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = create_audio_chunk_metadata(
            session_id="test_session",
            file_path="/test/path.wav",
            file_size=64000,
            duration_seconds=3.0,
            chunk_sequence=1,
            chunk_start_time=0.0,
        )

        # Test JSON serialization
        json_data = metadata.json()
        assert "session_id" in json_data
        assert "test_session" in json_data

        # Test dict conversion
        dict_data = metadata.model_dump()
        assert dict_data["session_id"] == "test_session"
        assert dict_data["file_size"] == 64000


class TestSpeakerCorrelation:
    """Test SpeakerCorrelation model."""

    def test_valid_speaker_correlation_creation(self):
        """Test creating valid speaker correlation."""
        correlation = create_speaker_correlation(
            session_id="test_session",
            whisper_speaker_id="speaker_0",
            correlation_confidence=0.9,
            correlation_type=CorrelationType.TEMPORAL,
            start_timestamp=0.0,
            end_timestamp=3.0,
            google_meet_speaker_name="John Doe",
        )

        assert correlation.session_id == "test_session"
        assert correlation.whisper_speaker_id == "speaker_0"
        assert correlation.correlation_confidence == 0.9
        assert correlation.correlation_type == CorrelationType.TEMPORAL
        assert correlation.start_timestamp == 0.0
        assert correlation.end_timestamp == 3.0
        assert correlation.google_meet_speaker_name == "John Doe"

    def test_speaker_correlation_confidence_validation(self):
        """Test correlation confidence validation."""
        # Test invalid confidence score
        with pytest.raises(ValidationError):
            SpeakerCorrelation(
                session_id="test",
                whisper_speaker_id="speaker_0",
                correlation_confidence=1.5,  # > 1.0
                correlation_type=CorrelationType.EXACT,
                correlation_method="test",
                start_timestamp=0.0,
                end_timestamp=3.0,
            )

    def test_speaker_correlation_timestamp_validation(self):
        """Test timestamp validation."""
        # Test invalid timestamp order
        with pytest.raises(ValidationError):
            SpeakerCorrelation(
                session_id="test",
                whisper_speaker_id="speaker_0",
                correlation_confidence=0.9,
                correlation_type=CorrelationType.EXACT,
                correlation_method="test",
                start_timestamp=5.0,  # After end
                end_timestamp=2.0,  # Before start
            )


class TestProcessingResult:
    """Test ProcessingResult model."""

    def test_valid_processing_result_creation(self):
        """Test creating valid processing result."""
        result = create_processing_result(
            chunk_id="chunk_123",
            session_id="session_456",
            processing_stage="transcription",
            status=ProcessingStatus.COMPLETED,
            processing_time_ms=150.0,
            quality_score=0.85,
        )

        assert result.chunk_id == "chunk_123"
        assert result.session_id == "session_456"
        assert result.processing_stage == "transcription"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.processing_time_ms == 150.0
        assert result.quality_score == 0.85

    def test_processing_result_completion_time_validation(self):
        """Test completion time validation."""
        # Test invalid completion time
        with pytest.raises(ValidationError):
            ProcessingResult(
                chunk_id="test",
                session_id="test",
                processing_stage="test",
                status=ProcessingStatus.COMPLETED,
                started_at=datetime.now(UTC),
                completed_at=datetime(2020, 1, 1, tzinfo=UTC),  # Before start
            )


class TestAudioChunkingConfig:
    """Test AudioChunkingConfig model."""

    def test_default_chunking_config(self):
        """Test default chunking configuration."""
        config = get_default_chunking_config()

        assert config.chunk_duration > 0
        assert config.overlap_duration >= 0
        assert config.overlap_duration < config.chunk_duration
        assert config.buffer_duration > config.chunk_duration
        assert 0.0 <= config.min_quality_threshold <= 1.0
        assert config.max_concurrent_chunks > 0

    def test_chunking_config_validation(self):
        """Test chunking configuration validation."""
        # Test valid configuration
        config = AudioChunkingConfig(
            chunk_duration=3.0,
            overlap_duration=0.5,
            processing_interval=2.5,
            buffer_duration=10.0,
        )

        assert config.chunk_duration == 3.0
        assert config.overlap_duration == 0.5
        assert config.processing_interval == 2.5
        assert config.buffer_duration == 10.0

    def test_chunking_config_timing_validation(self):
        """Test timing parameter validation."""
        # Test overlap >= chunk_duration (should fail)
        with pytest.raises(ValidationError):
            AudioChunkingConfig(
                chunk_duration=3.0,
                overlap_duration=4.0,  # Longer than chunk
            )

    def test_chunking_config_range_validation(self):
        """Test parameter range validation."""
        # Test invalid ranges
        with pytest.raises(ValidationError):
            AudioChunkingConfig(chunk_duration=0.1)  # Too short

        with pytest.raises(ValidationError):
            AudioChunkingConfig(overlap_duration=-1.0)  # Negative

        with pytest.raises(ValidationError):
            AudioChunkingConfig(min_quality_threshold=1.5)  # > 1.0


class TestQualityMetrics:
    """Test QualityMetrics model."""

    def test_valid_quality_metrics_creation(self):
        """Test creating valid quality metrics."""
        metrics = QualityMetrics(
            rms_level=0.1,
            peak_level=0.5,
            signal_to_noise_ratio=15.0,
            zero_crossing_rate=0.05,
            voice_activity_detected=True,
            voice_activity_confidence=0.8,
            speaking_time_ratio=0.7,
            clipping_detected=False,
            distortion_level=0.1,
            noise_level=0.2,
            overall_quality_score=0.75,
            quality_factors={"clarity": 0.8, "noise": 0.7},
            analysis_method="comprehensive",
        )

        assert metrics.rms_level == 0.1
        assert metrics.peak_level == 0.5
        assert metrics.signal_to_noise_ratio == 15.0
        assert metrics.voice_activity_detected
        assert metrics.overall_quality_score == 0.75

    def test_quality_metrics_range_validation(self):
        """Test quality metrics range validation."""
        # Test invalid ranges
        with pytest.raises(ValidationError):
            QualityMetrics(
                rms_level=-0.1,  # Negative
                peak_level=0.5,
                signal_to_noise_ratio=15.0,
                zero_crossing_rate=0.05,
                voice_activity_detected=True,
                overall_quality_score=0.75,
            )

        with pytest.raises(ValidationError):
            QualityMetrics(
                rms_level=0.1,
                peak_level=1.5,  # > 1.0
                signal_to_noise_ratio=15.0,
                zero_crossing_rate=0.05,
                voice_activity_detected=True,
                overall_quality_score=0.75,
            )


class TestAudioStreamingSession:
    """Test AudioStreamingSession model."""

    def test_valid_streaming_session_creation(self):
        """Test creating valid streaming session."""
        chunking_config = get_default_chunking_config()

        session = AudioStreamingSession(
            bot_session_id="bot_123",
            source_type=SourceType.BOT_AUDIO,
            chunk_config=chunking_config,
            target_languages=["en", "es", "fr"],
            real_time_processing=True,
            speaker_correlation_enabled=True,
        )

        assert session.bot_session_id == "bot_123"
        assert session.source_type == SourceType.BOT_AUDIO
        assert session.target_languages == ["en", "es", "fr"]
        assert session.real_time_processing
        assert session.speaker_correlation_enabled
        assert session.stream_status == "initialized"
        assert session.chunks_processed == 0

    def test_streaming_session_status_updates(self):
        """Test streaming session status updates."""
        chunking_config = get_default_chunking_config()

        session = AudioStreamingSession(
            bot_session_id="bot_123",
            source_type=SourceType.BOT_AUDIO,
            chunk_config=chunking_config,
        )

        # Update status
        session.stream_status = "active"
        session.chunks_processed = 10
        session.total_duration = 30.0
        session.average_processing_time = 25.5

        assert session.stream_status == "active"
        assert session.chunks_processed == 10
        assert session.total_duration == 30.0
        assert session.average_processing_time == 25.5


class TestEnumValidation:
    """Test enum validation in models."""

    def test_audio_format_validation(self):
        """Test AudioFormat enum validation."""
        # Valid formats
        for format_type in AudioFormat:
            metadata = AudioChunkMetadata(
                session_id="test",
                file_path="/test.wav",
                file_name="test.wav",
                file_size=1000,
                file_format=format_type,
                duration_seconds=3.0,
                chunk_sequence=1,
                chunk_start_time=0.0,
                chunk_end_time=3.0,
                source_type=SourceType.MANUAL_UPLOAD,
            )
            assert metadata.file_format == format_type

    def test_processing_status_validation(self):
        """Test ProcessingStatus enum validation."""
        # Valid statuses
        for status in ProcessingStatus:
            result = ProcessingResult(
                chunk_id="test",
                session_id="test",
                processing_stage="test",
                status=status,
            )
            assert result.status == status

    def test_source_type_validation(self):
        """Test SourceType enum validation."""
        # Valid source types
        for source_type in SourceType:
            metadata = AudioChunkMetadata(
                session_id="test",
                file_path="/test.wav",
                file_name="test.wav",
                file_size=1000,
                duration_seconds=3.0,
                chunk_sequence=1,
                chunk_start_time=0.0,
                chunk_end_time=3.0,
                source_type=source_type,
            )
            assert metadata.source_type == source_type


class TestFactoryFunctions:
    """Test factory functions for creating models."""

    def test_create_audio_chunk_metadata_factory(self):
        """Test audio chunk metadata factory function."""
        metadata = create_audio_chunk_metadata(
            session_id="factory_test",
            file_path="/factory/test.wav",
            file_size=32000,
            duration_seconds=2.0,
            chunk_sequence=5,
            chunk_start_time=10.0,
            audio_quality_score=0.9,
            sample_rate=22050,
        )

        assert metadata.session_id == "factory_test"
        assert metadata.file_path == "/factory/test.wav"
        assert metadata.file_size == 32000
        assert metadata.duration_seconds == 2.0
        assert metadata.chunk_sequence == 5
        assert metadata.chunk_start_time == 10.0
        assert metadata.chunk_end_time == 12.0
        assert metadata.audio_quality_score == 0.9
        assert metadata.sample_rate == 22050
        assert metadata.file_name == "test.wav"  # Extracted from path

    def test_create_speaker_correlation_factory(self):
        """Test speaker correlation factory function."""
        correlation = create_speaker_correlation(
            session_id="correlation_test",
            whisper_speaker_id="spk_1",
            correlation_confidence=0.95,
            correlation_type=CorrelationType.EXACT,
            start_timestamp=5.0,
            end_timestamp=10.0,
            google_meet_speaker_id="gmeet_speaker_456",
            text_similarity_score=0.88,
        )

        assert correlation.session_id == "correlation_test"
        assert correlation.whisper_speaker_id == "spk_1"
        assert correlation.correlation_confidence == 0.95
        assert correlation.correlation_type == CorrelationType.EXACT
        assert correlation.start_timestamp == 5.0
        assert correlation.end_timestamp == 10.0
        assert correlation.google_meet_speaker_id == "gmeet_speaker_456"
        assert correlation.text_similarity_score == 0.88
        assert correlation.correlation_method == "temporal_alignment"  # Default

    def test_create_processing_result_factory(self):
        """Test processing result factory function."""
        result = create_processing_result(
            chunk_id="result_test_chunk",
            session_id="result_test_session",
            processing_stage="enhancement",
            status=ProcessingStatus.COMPLETED,
            processing_time_ms=123.45,
            confidence_score=0.87,
            input_chunk_ids=["input_1", "input_2"],
            output_chunk_ids=["output_1"],
        )

        assert result.chunk_id == "result_test_chunk"
        assert result.session_id == "result_test_session"
        assert result.processing_stage == "enhancement"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.processing_time_ms == 123.45
        assert result.confidence_score == 0.87
        assert result.input_chunk_ids == ["input_1", "input_2"]
        assert result.output_chunk_ids == ["output_1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
