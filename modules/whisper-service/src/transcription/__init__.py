"""
Transcription Module

Contains transcription-related data structures and processing logic.
"""

from .request_models import TranscriptionRequest, TranscriptionResult
from .buffer_manager import SimpleAudioBufferManager

__all__ = [
    "TranscriptionRequest",
    "TranscriptionResult",
    "SimpleAudioBufferManager",
]
