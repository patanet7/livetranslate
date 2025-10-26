"""
Transcription Module

Contains transcription-related data structures and processing logic.
"""

from .request_models import TranscriptionRequest, TranscriptionResult
from .buffer_manager import SimpleAudioBufferManager
from .text_analysis import (
    detect_hallucination,
    find_stable_word_prefix,
    calculate_text_stability_score
)
from .result_parser import parse_whisper_result

__all__ = [
    "TranscriptionRequest",
    "TranscriptionResult",
    "SimpleAudioBufferManager",
    "detect_hallucination",
    "find_stable_word_prefix",
    "calculate_text_stability_score",
    "parse_whisper_result",
]
