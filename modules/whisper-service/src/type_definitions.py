#!/usr/bin/env python3
"""
Type Definitions for Whisper Service

Provides TypedDict classes and type aliases for better type safety and IDE support.

Usage:
    from types import ProcessResult, SessionSegment, LIDProbs

    def process(audio: np.ndarray) -> ProcessResult:
        return {
            'text': 'transcription',
            'language': 'en',
            'is_final': True,
            ...
        }
"""

from typing import Any, TypedDict

import numpy as np

# ============================================================================
# VAD Types
# ============================================================================


class VADResult(TypedDict, total=False):
    """
    Result from VAD detection.

    Can contain 'start', 'end', or both keys depending on detection state.
    """

    start: float  # Speech start timestamp in seconds
    end: float  # Speech end timestamp in seconds


# ============================================================================
# Language ID Types
# ============================================================================


class LIDProbs(TypedDict):
    """Language ID probabilities for each language"""

    # Dynamic keys based on target languages
    # Example: {'en': 0.7, 'zh': 0.3}


class LIDFrame(TypedDict):
    """Frame-level LID detection result"""

    timestamp: float
    language: str
    probabilities: LIDProbs
    confidence: float


class SmoothedLIDResult(TypedDict):
    """Result from LID smoothing (Viterbi/median)"""

    smoothed_probabilities: LIDProbs
    smoothed_language: str
    timestamp: float


class SwitchEvent(TypedDict):
    """Language switch event"""

    from_language: str
    to_language: str
    timestamp: float
    confidence_margin: float
    dwell_frames: int
    dwell_duration_ms: float


# ============================================================================
# Session and Segment Types
# ============================================================================


class SessionSegment(TypedDict):
    """Transcription segment from a single session"""

    text: str
    language: str
    start: float
    end: float
    is_final: bool
    confidence: float


class Statistics(TypedDict):
    """Session statistics"""

    current_language: str | None
    total_sessions: int
    total_switches: int
    total_audio_seconds: float
    session_duration: float
    lid_stats: dict[str, Any]
    sustained_detector_stats: dict[str, Any]
    smoother_stats: dict[str, Any]


# ============================================================================
# Process Result Types
# ============================================================================


class ProcessResult(TypedDict):
    """
    Result from processing an audio chunk.

    Complete result including transcription, language detection, and metadata.
    """

    # Transcription output
    text: str
    language: str
    is_final: bool

    # Session segments
    segments: list[SessionSegment]

    # Language switching
    switch_detected: bool
    current_language: str | None
    candidate_language: str | None

    # Chunk tracking
    chunk_id: int
    chunks_since_output: int
    silence_detected: bool

    # Statistics
    statistics: Statistics


class FinalizeResult(TypedDict):
    """Result from finalize() call"""

    text: str
    language: str | None
    is_final: bool
    segments: list[SessionSegment]
    switch_detected: bool
    finalized: bool


# ============================================================================
# Audio Types
# ============================================================================


class AudioStats(TypedDict):
    """Audio chunk statistics"""

    rms: float
    max_amplitude: float
    duration_seconds: float
    sample_count: int


class AudioChunk(TypedDict):
    """Audio chunk with metadata"""

    data: np.ndarray
    timestamp: float
    stats: AudioStats


# ============================================================================
# Whisper Model Types
# ============================================================================


class WhisperMetadata(TypedDict, total=False):
    """Metadata from Whisper inference"""

    progress: list[dict[str, Any]]
    completed: bool


class TranscriptionResult(TypedDict):
    """Raw transcription result from Whisper"""

    token_ids: list[int]
    text: str
    metadata: WhisperMetadata | None


# ============================================================================
# Configuration Types
# ============================================================================


class VADConfigDict(TypedDict):
    """VAD configuration dictionary"""

    threshold: float
    sampling_rate: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    silence_threshold_chunks: int


class LIDConfigDict(TypedDict):
    """LID configuration dictionary"""

    lid_hop_ms: int
    smoothing_enabled: bool
    smoothing_window_size: int
    confidence_margin: float
    min_dwell_frames: int
    min_dwell_ms: float
    viterbi_transition_cost: float
    viterbi_window_size: int


class WhisperConfigDict(TypedDict):
    """Whisper configuration dictionary"""

    model_path: str
    models_dir: str | None
    decoder_type: str
    beam_size: int
    online_chunk_size: float
    sampling_rate: int
    target_languages: list[str]
    audio_min_len: float
    n_mels: int


# ============================================================================
# Detector Statistics Types
# ============================================================================


class SustainedDetectorStats(TypedDict):
    """Statistics from sustained language detector"""

    current_language: str | None
    candidate_language: str | None
    total_switches: int
    false_positives_prevented: int
    candidate_frames: int
    candidate_progress: str


class SmootherStats(TypedDict):
    """Statistics from LID smoother"""

    frames_processed: int
    smoothing_method: str
    window_size: int


class LIDDetectorStats(TypedDict):
    """Statistics from frame-level LID detector"""

    frames_processed: int
    detections_by_language: dict[str, int]
    average_confidence: float
