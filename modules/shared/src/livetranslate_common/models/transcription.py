"""Transcription-related shared Pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Segment(BaseModel):
    """A single timed segment within a transcription result.

    Args:
        text: The transcribed text for this segment.
        start_ms: Segment start time in milliseconds.
        end_ms: Segment end time in milliseconds.
        confidence: Recognition confidence in [0.0, 1.0].
        speaker_id: Optional speaker diarization identifier.
    """

    text: str
    start_ms: int
    end_ms: int
    confidence: float = Field(ge=0.0, le=1.0)
    speaker_id: str | None = None

    @model_validator(mode="after")
    def check_end_after_start(self) -> Segment:
        if self.end_ms < self.start_ms:
            msg = f"end_ms ({self.end_ms}) must be >= start_ms ({self.start_ms})"
            raise ValueError(msg)
        return self

    @property
    def duration_ms(self) -> int:
        """Duration of this segment in milliseconds."""
        return self.end_ms - self.start_ms


class TranscriptionResult(BaseModel):
    """Complete result from a transcription operation.

    Args:
        text: Full transcribed text.
        language: BCP-47 language tag detected or specified.
        confidence: Overall confidence score in [0.0, 1.0].
        segments: List of timed sub-segments.
        stable_text: The portion of text considered stable (finalized).
        unstable_text: The portion of text still being refined.
        is_final: True when the segment text ends at a sentence boundary
            (punctuation). WARNING: Does NOT mean "last segment" or "will not
            be updated." A segment with is_final=False can still be the
            definitive transcription for its audio window.
            See ARCHITECTURE.md Draft/Final Protocol.
        is_draft: True for first-pass VAC snapshot (non-destructive, stride/2
            audio). Draft and final segments share the same segment_id. The
            final is a second-pass with the full audio stride -- same model,
            more audio, usually longer/more accurate text. The frontend
            replaces the draft in-place when the final arrives.
        speaker_id: Optional speaker identifier for the result.
        should_translate: Whether this result should be sent for translation.
        context_text: Prior context text used to condition the model.
    """

    text: str
    language: str
    confidence: float = Field(ge=0.0, le=1.0)
    segments: list[Segment] = Field(default_factory=list)
    stable_text: str = ""
    unstable_text: str = ""
    is_final: bool = False
    is_draft: bool = True
    speaker_id: str | None = None
    should_translate: bool = False
    context_text: str = ""
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class ModelInfo(BaseModel):
    """Metadata describing an available transcription model.

    Args:
        name: Model identifier (e.g. "whisper-base").
        backend: Inference backend (e.g. "faster-whisper", "openvino").
        languages: Supported BCP-47 language tags.
        vram_mb: VRAM requirement in megabytes.
        compute_type: Quantization/compute type (e.g. "int8", "float16").
    """

    name: str
    backend: str
    languages: list[str]
    vram_mb: int
    compute_type: str
