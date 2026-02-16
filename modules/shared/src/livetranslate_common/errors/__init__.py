"""Structured error hierarchy for LiveTranslate services."""

from livetranslate_common.errors.exceptions import (
    AudioProcessingError,
    LiveTranslateError,
    ServiceUnavailableError,
    ValidationError,
)

__all__ = [
    "AudioProcessingError",
    "LiveTranslateError",
    "ServiceUnavailableError",
    "ValidationError",
]
