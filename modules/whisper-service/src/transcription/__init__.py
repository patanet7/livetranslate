"""
Transcription Module

Contains transcription-related data structures and processing logic.

CLEANUP: Removed SimpleAudioBufferManager import (unused dead code).
Buffering is now handled by VACOnlineASRProcessor's internal buffers.
"""

from .request_models import TranscriptionRequest, TranscriptionResult
from .text_analysis import (
    detect_hallucination,
    find_stable_word_prefix,
    calculate_text_stability_score
)
from .result_parser import parse_whisper_result
from .domain_prompt_helper import prepare_domain_prompt

__all__ = [
    "TranscriptionRequest",
    "TranscriptionResult",
    "detect_hallucination",
    "find_stable_word_prefix",
    "calculate_text_stability_score",
    "parse_whisper_result",
    "prepare_domain_prompt",
]
