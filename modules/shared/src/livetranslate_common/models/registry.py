"""Model registry configuration types for backend selection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BackendConfig(BaseModel):
    """Configuration entry for a single transcription backend/model combination.

    The ``stride_s`` field is constrained to equal
    ``chunk_duration_s - overlap_s``; violating this invariant raises a
    ``ValueError`` at construction time.

    Args:
        backend: Backend identifier (e.g. "faster-whisper", "openvino").
        model: Model identifier (e.g. "whisper-base", "whisper-large-v3").
        compute_type: Quantization type (e.g. "int8", "float16", "auto").
        chunk_duration_s: Audio chunk size in seconds (must be > 0).
        stride_s: Sliding window stride in seconds (must be > 0).
        overlap_s: Overlap between consecutive chunks in seconds (>= 0).
        vad_threshold: Voice Activity Detection sensitivity in [0.0, 1.0].
        beam_size: Beam search width (must be >= 1).
        prebuffer_s: Pre-roll buffer duration in seconds (>= 0).
        batch_profile: Optimisation profile — "realtime" or "batch".
    """

    backend: str
    model: str
    compute_type: str
    chunk_duration_s: float = Field(gt=0)
    stride_s: float = Field(gt=0)
    overlap_s: float = Field(ge=0)
    vad_threshold: float = Field(ge=0.0, le=1.0)
    beam_size: int = Field(ge=1)
    prebuffer_s: float = Field(ge=0)
    batch_profile: Literal["realtime", "batch"] = "realtime"

    @model_validator(mode="after")
    def check_stride_overlap_consistency(self) -> "BackendConfig":
        """Enforce stride_s == chunk_duration_s - overlap_s."""
        expected = self.chunk_duration_s - self.overlap_s
        if abs(self.stride_s - expected) > 1e-6:
            raise ValueError(
                f"stride_s ({self.stride_s}) must equal "
                f"chunk_duration_s ({self.chunk_duration_s}) - overlap_s ({self.overlap_s}) = {expected}"
            )
        return self
