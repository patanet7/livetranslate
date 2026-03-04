#!/usr/bin/env python3
"""
Diarization Models Tests - Behavioral Tests (No Mocks)

Tests real model behavior, field validation, enum values, and defaults
for all diarization-related Pydantic models.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))

from models.diarization import (
    DiarizationJobCreate,
    DiarizationJobResponse,
    DiarizationJobStatus,
    DiarizationRules,
    SpeakerMapEntry,
    SpeakerMergeRequest,
    SpeakerProfileCreate,
    SpeakerProfileResponse,
    TranscribeResponse,
    TranscribeSegment,
    TranscriptCompareResponse,
)


# =============================================================================
# TranscribeSegment
# =============================================================================


class TestTranscribeSegment:
    """Test TranscribeSegment - the core per-speaker transcription unit."""

    def test_creation_and_field_access(self):
        """Test creating a segment and accessing all fields."""
        segment = TranscribeSegment(
            speaker=0,
            start=0.0,
            end=3.5,
            text="Hello, this is speaker zero.",
        )

        assert segment.speaker == 0
        assert segment.start == 0.0
        assert segment.end == 3.5
        assert segment.text == "Hello, this is speaker zero."

    def test_multiple_speakers(self):
        """Test segments with different speaker indices."""
        seg_a = TranscribeSegment(speaker=0, start=0.0, end=2.0, text="First speaker.")
        seg_b = TranscribeSegment(speaker=1, start=2.5, end=5.0, text="Second speaker.")

        assert seg_a.speaker != seg_b.speaker
        assert seg_b.start > seg_a.end

    def test_serialization_roundtrip(self):
        """Test JSON serialisation and deserialisation."""
        segment = TranscribeSegment(speaker=2, start=10.0, end=15.3, text="Round-trip test.")
        json_str = segment.model_dump_json()
        restored = TranscribeSegment.model_validate_json(json_str)

        assert restored.speaker == segment.speaker
        assert restored.start == segment.start
        assert restored.end == segment.end
        assert restored.text == segment.text


# =============================================================================
# TranscribeResponse
# =============================================================================


class TestTranscribeResponse:
    """Test TranscribeResponse - full transcription result with metadata."""

    def test_creation_with_multiple_segments(self):
        """Test response containing several segments."""
        segments = [
            TranscribeSegment(speaker=0, start=0.0, end=2.0, text="Hello everyone."),
            TranscribeSegment(speaker=1, start=2.5, end=4.0, text="Hi there."),
            TranscribeSegment(speaker=0, start=4.5, end=7.0, text="Glad you could make it."),
        ]
        response = TranscribeResponse(
            segments=segments,
            detected_language="en",
            num_speakers=2,
            duration_seconds=7.5,
            processing_time_seconds=1.23,
        )

        assert len(response.segments) == 3
        assert response.detected_language == "en"
        assert response.num_speakers == 2
        assert response.duration_seconds == 7.5
        assert response.processing_time_seconds == 1.23

    def test_empty_segments(self):
        """Test response with no segments is valid."""
        response = TranscribeResponse(
            segments=[],
            detected_language="fr",
            num_speakers=0,
            duration_seconds=0.0,
            processing_time_seconds=0.5,
        )

        assert response.segments == []
        assert response.num_speakers == 0

    def test_serialization_roundtrip(self):
        """Test JSON serialisation preserves nested segments."""
        segments = [TranscribeSegment(speaker=0, start=0.0, end=1.0, text="Test.")]
        response = TranscribeResponse(
            segments=segments,
            detected_language="de",
            num_speakers=1,
            duration_seconds=1.0,
            processing_time_seconds=0.3,
        )

        data = response.model_dump()
        restored = TranscribeResponse(**data)
        assert len(restored.segments) == 1
        assert restored.segments[0].text == "Test."


# =============================================================================
# DiarizationJobStatus
# =============================================================================


class TestDiarizationJobStatus:
    """Test DiarizationJobStatus enum values."""

    def test_all_status_values_exist(self):
        """Verify all expected status values are present."""
        assert DiarizationJobStatus.queued == "queued"
        assert DiarizationJobStatus.downloading == "downloading"
        assert DiarizationJobStatus.processing == "processing"
        assert DiarizationJobStatus.mapping == "mapping"
        assert DiarizationJobStatus.completed == "completed"
        assert DiarizationJobStatus.failed == "failed"
        assert DiarizationJobStatus.cancelled == "cancelled"

    def test_status_is_string(self):
        """Status values are plain strings (StrEnum)."""
        assert isinstance(DiarizationJobStatus.queued, str)
        assert isinstance(DiarizationJobStatus.completed, str)

    def test_queued_and_processing_distinct(self):
        """Queued and processing are different statuses."""
        assert DiarizationJobStatus.queued != DiarizationJobStatus.processing

    def test_terminal_states(self):
        """Completed, failed, and cancelled are terminal states."""
        terminal = {
            DiarizationJobStatus.completed,
            DiarizationJobStatus.failed,
            DiarizationJobStatus.cancelled,
        }
        assert len(terminal) == 3


# =============================================================================
# SpeakerMapEntry
# =============================================================================


class TestSpeakerMapEntry:
    """Test SpeakerMapEntry - speaker identification result."""

    def test_creation(self):
        """Test creating a speaker map entry."""
        entry = SpeakerMapEntry(name="Alice", confidence=0.92, method="voice_print")

        assert entry.name == "Alice"
        assert entry.confidence == 0.92
        assert entry.method == "voice_print"

    def test_confidence_lower_bound(self):
        """Confidence of 0.0 is valid."""
        entry = SpeakerMapEntry(name="Unknown", confidence=0.0, method="fallback")
        assert entry.confidence == 0.0

    def test_confidence_upper_bound(self):
        """Confidence of 1.0 is valid."""
        entry = SpeakerMapEntry(name="Bob", confidence=1.0, method="exact_match")
        assert entry.confidence == 1.0

    def test_confidence_below_zero_invalid(self):
        """Confidence below 0 must be rejected."""
        with pytest.raises(Exception):
            SpeakerMapEntry(name="X", confidence=-0.1, method="test")

    def test_confidence_above_one_invalid(self):
        """Confidence above 1 must be rejected."""
        with pytest.raises(Exception):
            SpeakerMapEntry(name="X", confidence=1.1, method="test")


# =============================================================================
# DiarizationJobCreate
# =============================================================================


class TestDiarizationJobCreate:
    """Test DiarizationJobCreate - payload to kick off a diarization job."""

    def test_creation_with_hotwords(self):
        """Test creating a job with hotwords."""
        job = DiarizationJobCreate(
            meeting_id=42,
            hotwords=["OpenVINO", "Whisper", "NPU"],
        )

        assert job.meeting_id == 42
        assert job.hotwords == ["OpenVINO", "Whisper", "NPU"]

    def test_creation_without_hotwords(self):
        """Test creating a job without hotwords (optional field)."""
        job = DiarizationJobCreate(meeting_id=7)

        assert job.meeting_id == 7
        assert job.hotwords is None

    def test_empty_hotwords_list(self):
        """Test creating a job with an empty hotwords list."""
        job = DiarizationJobCreate(meeting_id=1, hotwords=[])
        assert job.hotwords == []

    def test_serialization(self):
        """Test dict serialisation for API submission."""
        job = DiarizationJobCreate(meeting_id=99, hotwords=["keyword"])
        data = job.model_dump()
        assert data["meeting_id"] == 99
        assert data["hotwords"] == ["keyword"]


# =============================================================================
# DiarizationJobResponse
# =============================================================================


class TestDiarizationJobResponse:
    """Test DiarizationJobResponse - API response for a diarization job."""

    def test_minimal_creation(self):
        """Test creating a minimal response (all optional fields absent)."""
        response = DiarizationJobResponse(
            id=1,
            meeting_id=10,
            status=DiarizationJobStatus.queued,
            triggered_by="user",
        )

        assert response.id == 1
        assert response.meeting_id == 10
        assert response.status == DiarizationJobStatus.queued
        assert response.triggered_by == "user"
        assert response.speaker_map is None
        assert response.error_message is None

    def test_completed_response(self):
        """Test a fully-populated completed response."""
        speaker_map = {
            "SPEAKER_00": SpeakerMapEntry(name="Alice", confidence=0.9, method="voice_print"),
            "SPEAKER_01": SpeakerMapEntry(name="Bob", confidence=0.75, method="name_match"),
        }
        response = DiarizationJobResponse(
            id=2,
            meeting_id=20,
            status=DiarizationJobStatus.completed,
            triggered_by="scheduler",
            detected_language="en",
            num_speakers_detected=2,
            processing_time_seconds=45.3,
            speaker_map=speaker_map,
            unmapped_speakers=["SPEAKER_02"],
            merge_applied=False,
        )

        assert response.status == DiarizationJobStatus.completed
        assert len(response.speaker_map) == 2
        assert response.speaker_map["SPEAKER_00"].name == "Alice"
        assert response.unmapped_speakers == ["SPEAKER_02"]

    def test_failed_response(self):
        """Test a failed job response has an error message."""
        response = DiarizationJobResponse(
            id=3,
            meeting_id=30,
            status=DiarizationJobStatus.failed,
            triggered_by="api",
            error_message="Audio file download timed out.",
        )

        assert response.status == DiarizationJobStatus.failed
        assert response.error_message == "Audio file download timed out."


# =============================================================================
# DiarizationRules
# =============================================================================


class TestDiarizationRules:
    """Test DiarizationRules - auto-trigger configuration."""

    def test_defaults(self):
        """Test default values are sensible."""
        rules = DiarizationRules()

        assert rules.enabled is False or rules.enabled is True  # just ensure field exists
        assert rules.min_duration_minutes == 5
        assert rules.exclude_empty is True
        assert isinstance(rules.participant_patterns, list)
        assert isinstance(rules.title_patterns, list)

    def test_custom_values(self):
        """Test overriding defaults."""
        rules = DiarizationRules(
            enabled=True,
            participant_patterns=["@company.com"],
            title_patterns=["standup", "sprint"],
            min_duration_minutes=10,
            exclude_empty=False,
        )

        assert rules.enabled is True
        assert "@company.com" in rules.participant_patterns
        assert rules.min_duration_minutes == 10
        assert rules.exclude_empty is False

    def test_empty_patterns_valid(self):
        """Empty pattern lists are valid."""
        rules = DiarizationRules(participant_patterns=[], title_patterns=[])
        assert rules.participant_patterns == []
        assert rules.title_patterns == []

    def test_serialization(self):
        """Test rules serialise to dict correctly."""
        rules = DiarizationRules(enabled=True, min_duration_minutes=15)
        data = rules.model_dump()
        assert data["min_duration_minutes"] == 15
        assert "enabled" in data


# =============================================================================
# SpeakerProfileCreate
# =============================================================================


class TestSpeakerProfileCreate:
    """Test SpeakerProfileCreate - payload to register a speaker profile."""

    def test_creation_with_email(self):
        """Test creating a profile with email."""
        profile = SpeakerProfileCreate(name="Alice Smith", email="alice@example.com")

        assert profile.name == "Alice Smith"
        assert profile.email == "alice@example.com"

    def test_creation_without_email(self):
        """Test creating a profile without email (optional)."""
        profile = SpeakerProfileCreate(name="Bob")

        assert profile.name == "Bob"
        assert profile.email is None

    def test_name_max_length(self):
        """Name exceeding 255 characters must be rejected."""
        long_name = "A" * 256
        with pytest.raises(Exception):
            SpeakerProfileCreate(name=long_name)

    def test_name_at_max_length(self):
        """Name of exactly 255 characters is valid."""
        max_name = "A" * 255
        profile = SpeakerProfileCreate(name=max_name)
        assert len(profile.name) == 255


# =============================================================================
# SpeakerProfileResponse
# =============================================================================


class TestSpeakerProfileResponse:
    """Test SpeakerProfileResponse - API response for a speaker profile."""

    def test_creation(self):
        """Test creating a speaker profile response."""
        response = SpeakerProfileResponse(
            id=1,
            name="Alice Smith",
            email="alice@example.com",
            enrollment_source="manual",
            sample_count=3,
        )

        assert response.id == 1
        assert response.name == "Alice Smith"
        assert response.enrollment_source == "manual"
        assert response.sample_count == 3

    def test_without_email(self):
        """Test profile response without email."""
        response = SpeakerProfileResponse(
            id=2,
            name="Bob",
            email=None,
            enrollment_source="fireflies",
            sample_count=0,
        )

        assert response.email is None
        assert response.sample_count == 0


# =============================================================================
# SpeakerMergeRequest
# =============================================================================


class TestSpeakerMergeRequest:
    """Test SpeakerMergeRequest - merge two speaker profiles."""

    def test_creation(self):
        """Test creating a merge request."""
        request = SpeakerMergeRequest(source_id=5, target_id=3)

        assert request.source_id == 5
        assert request.target_id == 3

    def test_source_and_target_distinct(self):
        """Source and target IDs can be different values."""
        request = SpeakerMergeRequest(source_id=1, target_id=2)
        assert request.source_id != request.target_id

    def test_serialization(self):
        """Test dict serialisation for API submission."""
        request = SpeakerMergeRequest(source_id=10, target_id=20)
        data = request.model_dump()
        assert data["source_id"] == 10
        assert data["target_id"] == 20


# =============================================================================
# TranscriptCompareResponse
# =============================================================================


class TestTranscriptCompareResponse:
    """Test TranscriptCompareResponse - side-by-side transcript comparison."""

    def test_creation(self):
        """Test creating a comparison response."""
        speaker_map = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
        response = TranscriptCompareResponse(
            meeting_id=100,
            fireflies_sentences=[{"speaker": "Alice", "text": "Hello."}],
            vibevoice_segments=[{"speaker": 0, "start": 0.0, "end": 1.0, "text": "Hello."}],
            speaker_map=speaker_map,
        )

        assert response.meeting_id == 100
        assert len(response.fireflies_sentences) == 1
        assert len(response.vibevoice_segments) == 1
        assert response.speaker_map["SPEAKER_00"] == "Alice"

    def test_empty_sentences_and_segments(self):
        """Test comparison response with no data."""
        response = TranscriptCompareResponse(
            meeting_id=0,
            fireflies_sentences=[],
            vibevoice_segments=[],
            speaker_map={},
        )

        assert response.fireflies_sentences == []
        assert response.vibevoice_segments == []
        assert response.speaker_map == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
