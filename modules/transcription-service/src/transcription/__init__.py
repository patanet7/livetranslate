"""
Transcription Module

Contains transcription-related data structures and processing logic.

CLEANUP: Removed SimpleAudioBufferManager import (unused dead code).
Buffering is now handled by VACOnlineASRProcessor's internal buffers.
"""

from .domain_prompt_helper import prepare_domain_prompt
from .request_models import TranscriptionRequest, TranscriptionResult
from .result_parser import parse_whisper_result
from .text_analysis import (
    calculate_text_stability_score,
    detect_hallucination,
    find_stable_word_prefix,
)

__all__ = [
    "TranscriptionRequest",
    "TranscriptionResult",
    "calculate_text_stability_score",
    "detect_hallucination",
    "find_stable_word_prefix",
    "parse_whisper_result",
    "prepare_domain_prompt",
]
