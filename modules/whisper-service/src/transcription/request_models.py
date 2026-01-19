#!/usr/bin/env python3
"""
Transcription Request and Response Models

Data structures for transcription requests and results.
Extracted from whisper_service.py for better modularity.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class TranscriptionRequest:
    """Transcription request data structure - Phase 2/3/4 Enhanced"""

    audio_data: np.ndarray | bytes
    model_name: str = "whisper-large-v3"  # Phase 2: Default to Large-v3
    language: str | None = None
    session_id: str | None = None
    streaming: bool = False
    enhanced: bool = False
    sample_rate: int = 16000
    enable_vad: bool = True
    timestamp_mode: str = "word"  # word, segment, none

    # Phase 2: Beam Search parameters
    beam_size: int = 5  # Beam width: 1=greedy, 5=quality (default), 10=max quality
    temperature: float = 0.0  # Sampling temperature (0.0 = deterministic)

    # Phase 2: In-Domain Prompting
    initial_prompt: str | None = None  # Domain-specific prompt or terminology
    domain: str | None = None  # Domain hint: "medical", "legal", "technical", etc.
    custom_terms: list[str] | None = None  # Custom terminology to inject

    # Phase 2: Context Carryover
    previous_context: str | None = None  # Previous output for continuity (max 223 tokens)

    # Phase 2: AlignAtt Streaming Policy
    streaming_policy: str = "alignatt"  # "alignatt" (SimulStreaming) or "fixed" (traditional)
    frame_threshold_offset: int = 10  # AlignAtt: frames to reserve for streaming

    # Phase 4: Translation Configuration
    task: str = "transcribe"  # "transcribe" (same lang) or "translate" (to English ONLY)
    target_language: str = (
        "en"  # Target language for translation (used by external service if not English)
    )

    # Phase 5: Code-Switching Support
    enable_code_switching: bool = (
        False  # Enable intra-sentence multilingual support (e.g., "我想要 a coffee please")
    )

    # Phase 5 Enhancement: Advanced Code-Switching Configuration
    sliding_lid_window: float | None = (
        None  # Sliding window for language detection (seconds, default: 0.9)
    )
    sustained_lang_duration: float | None = (
        None  # Duration before SOT reset (seconds, default: 3.0)
    )
    sustained_lang_min_silence: float | None = (
        None  # Min silence for SOT reset (seconds, default: 0.25)
    )
    soft_bias_enabled: bool | None = None  # Enable soft bias token injection (default: False)
    token_dedup_enabled: bool | None = None  # Enable token deduplication (default: True)
    confidence_threshold: float | None = None  # Threshold for n-best rescoring (default: 0.6)

    # VAD Configuration
    vad_threshold: float | None = None  # VAD threshold (default: 0.5)
    vad_min_speech_ms: int | None = None  # Min speech duration (default: 120ms)
    vad_min_silence_ms: int | None = None  # Min silence duration (default: 250ms)


@dataclass
class TranscriptionResult:
    """
    Transcription result data structure.

    Phase 3 Enhancement: Stability Tracking for Draft/Final Emission
    - Separates stable (confirmed, black in UI) vs unstable (uncertain, grey in UI)
    - Enables incremental MT updates (only translate stable tokens)
    - Supports draft/final emission protocol
    """

    # Original fields (backward compatible)
    text: str
    segments: list[dict]
    language: str
    confidence_score: float
    processing_time: float
    model_used: str
    device_used: str
    session_id: str | None = None
    timestamp: str = None

    # Phase 3: Stability Tracking - Text representations
    stable_text: str = ""  # Only stable prefix (black in UI)
    unstable_text: str = ""  # Only unstable tail (grey in UI)

    # Phase 3: Stability Tracking - Token-level data
    stable_tokens: list[Any] = None  # TokenState list - confirmed tokens → send to MT
    unstable_tokens: list[Any] = None  # TokenState list - uncertain tokens → hold back

    # Phase 3: Emission metadata
    is_final: bool = False  # True = segment boundary reached
    is_draft: bool = False  # True = incremental update
    is_forced: bool = False  # True = forced by max_latency

    # Phase 3: Translation integration
    should_translate: bool = False  # True if enough stable text for MT
    translation_mode: str = "none"  # "draft", "final", or "none"

    # Phase 3: Timestamps
    stable_end_time: float = 0.0  # Time of last stable token
    segment_start_time: float = 0.0
    segment_end_time: float = 0.0

    # Phase 3: Confidence metrics
    stability_score: float = 0.0  # Avg confidence of stable tokens

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        # Initialize lists if None
        if self.stable_tokens is None:
            self.stable_tokens = []
        if self.unstable_tokens is None:
            self.unstable_tokens = []
