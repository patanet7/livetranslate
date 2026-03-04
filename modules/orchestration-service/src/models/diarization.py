"""
Offline Diarization Models

Pydantic v2 data models for the offline speaker diarization pipeline.

Covers:
- Transcription segments produced by VibeVoice-ASR
- Diarization job lifecycle (create / response / status)
- Speaker profile management (create / response / merge)
- Auto-trigger rules configuration
- Transcript comparison between Fireflies and VibeVoice-ASR
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import ConfigDict, Field

from .base import BaseModel


# =============================================================================
# Enums
# =============================================================================


class DiarizationJobStatus(StrEnum):
    """Lifecycle states for an offline diarization job."""

    queued = "queued"
    downloading = "downloading"
    processing = "processing"
    mapping = "mapping"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


# =============================================================================
# Transcription Models
# =============================================================================


class TranscribeSegment(BaseModel):
    """A single transcribed segment attributed to one speaker.

    Produced by the VibeVoice-ASR / Whisper diarization pipeline.
    Each segment corresponds to a contiguous speech region from one speaker.
    """

    speaker: int = Field(description="Zero-based speaker index (e.g. 0, 1, 2)")
    start: float = Field(description="Segment start position in seconds")
    end: float = Field(description="Segment end position in seconds")
    text: str = Field(description="Transcribed text for this segment")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "speaker": 0,
                "start": 0.0,
                "end": 3.5,
                "text": "Hello, glad to be here today.",
            }
        }
    )


class TranscribeResponse(BaseModel):
    """Full transcription result returned by the VibeVoice-ASR service.

    Contains all diarized segments together with high-level metadata about
    the audio file that was processed.
    """

    segments: list[TranscribeSegment] = Field(
        description="Ordered list of diarized transcription segments"
    )
    detected_language: str = Field(
        description="BCP-47 language code detected in the audio (e.g. 'en', 'fr')"
    )
    num_speakers: int = Field(description="Number of distinct speakers detected")
    duration_seconds: float = Field(description="Total audio duration in seconds")
    processing_time_seconds: float = Field(
        description="Wall-clock time taken to transcribe and diarize, in seconds"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "segments": [
                    {"speaker": 0, "start": 0.0, "end": 2.5, "text": "Hello everyone."},
                    {"speaker": 1, "start": 3.0, "end": 5.0, "text": "Hi, good to see you."},
                ],
                "detected_language": "en",
                "num_speakers": 2,
                "duration_seconds": 60.0,
                "processing_time_seconds": 4.2,
            }
        }
    )


# =============================================================================
# Speaker Map / Profile Models
# =============================================================================


class SpeakerMapEntry(BaseModel):
    """Mapping of a diarization speaker ID to a human-readable name.

    Produced during the mapping phase of a diarization job.  The confidence
    score indicates how certain the system is about the identity assignment.
    """

    name: str = Field(description="Human-readable name assigned to the speaker")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the identity assignment, in the range [0, 1]",
    )
    method: str = Field(
        description=(
            "Method used to determine speaker identity "
            "(e.g. 'voice_print', 'name_match', 'participant_list', 'fallback')"
        )
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Alice Smith",
                "confidence": 0.92,
                "method": "voice_print",
            }
        }
    )


class SpeakerProfileCreate(BaseModel):
    """Payload for registering a new speaker profile.

    Speaker profiles are used during the mapping phase to match diarized
    speaker IDs to known participants.
    """

    name: str = Field(max_length=255, description="Display name for the speaker (max 255 chars)")
    email: str | None = Field(
        default=None,
        description="Optional email address used for cross-referencing meeting participants",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Alice Smith",
                "email": "alice@company.com",
            }
        }
    )


class SpeakerProfileResponse(BaseModel):
    """API response representing a persisted speaker profile."""

    id: int = Field(description="Database primary key for the speaker profile")
    name: str = Field(description="Display name for the speaker")
    email: str | None = Field(
        default=None, description="Email address associated with the speaker"
    )
    enrollment_source: str = Field(
        description=(
            "How the profile was created "
            "(e.g. 'manual', 'fireflies', 'import', 'auto_detected')"
        )
    )
    sample_count: int = Field(
        description="Number of audio samples enrolled for voice-print matching"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "name": "Alice Smith",
                "email": "alice@company.com",
                "enrollment_source": "manual",
                "sample_count": 5,
            }
        }
    )


class SpeakerMergeRequest(BaseModel):
    """Request to merge two speaker profiles into one.

    The source profile is merged into the target profile.  All historical
    references to the source are re-pointed to the target, and the source
    profile is then deleted.
    """

    source_id: int = Field(description="ID of the speaker profile to be merged (will be deleted)")
    target_id: int = Field(
        description="ID of the speaker profile to merge into (will be retained)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_id": 5,
                "target_id": 3,
            }
        }
    )


# =============================================================================
# Job Models
# =============================================================================


class DiarizationJobCreate(BaseModel):
    """Payload to trigger an offline diarization job for a meeting.

    The orchestration service will download the meeting audio, run
    VibeVoice-ASR diarization, and then attempt speaker mapping.
    """

    meeting_id: int = Field(description="Database ID of the meeting to diarize")
    hotwords: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of domain-specific hotwords to bias recognition "
            "(e.g. product names, technical terms)"
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "meeting_id": 42,
                "hotwords": ["OpenVINO", "VibeVoice", "NPU"],
            }
        }
    )


class DiarizationJobResponse(BaseModel):
    """API response describing a diarization job and its current state.

    Returned by both the job-creation endpoint and status-poll endpoints.
    Optional fields are populated only when the job reaches the relevant phase.
    """

    id: int = Field(description="Database primary key for the diarization job")
    meeting_id: int = Field(description="Database ID of the associated meeting")
    status: DiarizationJobStatus = Field(description="Current lifecycle status of the job")
    triggered_by: str = Field(
        description="Identity of the actor that triggered the job (e.g. 'user', 'scheduler', 'api')"
    )

    # Populated once audio analysis completes
    detected_language: str | None = Field(
        default=None,
        description="BCP-47 language code detected in the meeting audio",
    )
    num_speakers_detected: int | None = Field(
        default=None, description="Number of distinct speakers found in the audio"
    )
    processing_time_seconds: float | None = Field(
        default=None, description="Total wall-clock processing time in seconds"
    )

    # Populated once speaker mapping completes
    speaker_map: dict[str, SpeakerMapEntry] | None = Field(
        default=None,
        description="Mapping of diarization speaker IDs to named SpeakerMapEntry objects",
    )
    unmapped_speakers: list[str] | None = Field(
        default=None,
        description="Speaker IDs that could not be mapped to a known profile",
    )
    merge_applied: bool | None = Field(
        default=None,
        description="Whether a speaker-merge operation was applied during mapping",
    )

    # Error information
    error_message: str | None = Field(
        default=None, description="Human-readable error description if the job failed"
    )

    # Timestamps
    created_at: datetime | None = Field(default=None, description="UTC timestamp when job was created")
    completed_at: datetime | None = Field(
        default=None, description="UTC timestamp when job reached a terminal state"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "meeting_id": 42,
                "status": "completed",
                "triggered_by": "user",
                "detected_language": "en",
                "num_speakers_detected": 3,
                "processing_time_seconds": 87.4,
                "speaker_map": {
                    "SPEAKER_00": {
                        "name": "Alice Smith",
                        "confidence": 0.92,
                        "method": "voice_print",
                    }
                },
                "unmapped_speakers": ["SPEAKER_02"],
                "merge_applied": False,
                "error_message": None,
                "created_at": "2026-03-04T10:00:00Z",
                "completed_at": "2026-03-04T10:01:27Z",
            }
        }
    )


# =============================================================================
# Auto-Trigger Rules
# =============================================================================


class DiarizationRules(BaseModel):
    """Configuration rules that control automatic diarization triggering.

    When a meeting completes, the orchestration service evaluates these rules
    to decide whether to enqueue an offline diarization job automatically.
    """

    enabled: bool = Field(
        default=False,
        description="Whether automatic diarization triggering is enabled",
    )
    participant_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Glob/regex patterns matched against participant email addresses; "
            "a meeting is eligible when at least one participant matches"
        ),
    )
    title_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Glob/regex patterns matched against the meeting title; "
            "a meeting is eligible when the title matches at least one pattern"
        ),
    )
    min_duration_minutes: int = Field(
        default=5,
        description="Meetings shorter than this duration (in minutes) are skipped",
    )
    exclude_empty: bool = Field(
        default=True,
        description="When True, meetings with no transcript sentences are skipped",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "participant_patterns": ["@company.com"],
                "title_patterns": ["standup", "sprint review"],
                "min_duration_minutes": 5,
                "exclude_empty": True,
            }
        }
    )


# =============================================================================
# Comparison / Reconciliation Models
# =============================================================================


class TranscriptCompareResponse(BaseModel):
    """Side-by-side comparison of Fireflies and VibeVoice-ASR transcripts.

    Used by the reconciliation UI to review discrepancies between the two
    sources and confirm or adjust speaker mappings.
    """

    meeting_id: int = Field(description="Database ID of the meeting being compared")
    fireflies_sentences: list[dict[str, Any]] = Field(
        description="Ordered list of sentence objects from Fireflies (raw API format)"
    )
    vibevoice_segments: list[dict[str, Any]] = Field(
        description="Ordered list of segment dicts from VibeVoice-ASR diarization"
    )
    speaker_map: dict[str, Any] = Field(
        description=(
            "Current speaker ID to name mapping; values may be strings or SpeakerMapEntry dicts"
        )
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "meeting_id": 42,
                "fireflies_sentences": [
                    {"speaker_name": "Alice", "text": "Hello everyone.", "start_time": 0.0}
                ],
                "vibevoice_segments": [
                    {"speaker": 0, "start": 0.0, "end": 2.0, "text": "Hello everyone."}
                ],
                "speaker_map": {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
            }
        }
    )
