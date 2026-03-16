"""
Transcription Module

Contains transcription-related data structures and processing logic.

CLEANUP: Removed SimpleAudioBufferManager import (unused dead code).
Buffering is now handled by VACOnlineASRProcessor's internal buffers.
"""

from .domain_prompt_helper import prepare_domain_prompt
from .hallucination_filter import HallucinationFilter
from .request_models import TranscriptionRequest, TranscriptionResult


def detect_hallucination(text: str, **kwargs) -> bool:
    """Backward-compat stub — legacy whisper_service imports this.

    The real logic now lives in HallucinationFilter.should_suppress().
    This stub always returns False (never suppresses) so legacy code
    doesn't crash. Remove when _legacy/ is deleted.
    """
    return False
from .result_parser import parse_whisper_result
from .text_analysis import (
    calculate_text_stability_score,
    find_stable_word_prefix,
)

__all__ = [
    "HallucinationFilter",
    "TranscriptionRequest",
    "TranscriptionResult",
    "calculate_text_stability_score",
    "find_stable_word_prefix",
    "parse_whisper_result",
    "prepare_domain_prompt",
]
