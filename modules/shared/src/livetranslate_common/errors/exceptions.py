"""Structured exception hierarchy for LiveTranslate services."""

from typing import Any


class LiveTranslateError(Exception):
    """Base exception for all LiveTranslate application errors.

    Attributes:
        status_code: HTTP status code to return when this error is raised in a handler.
        error_code: Machine-readable error identifier for clients.
        context: Arbitrary key-value pairs providing additional error context.
    """

    status_code: int = 500

    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", **context: Any) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.context = context


class ServiceUnavailableError(LiveTranslateError):
    """A downstream service is unreachable or unhealthy."""

    status_code: int = 503

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message, error_code="SERVICE_UNAVAILABLE", **context)


class ValidationError(LiveTranslateError):
    """Request data failed validation."""

    status_code: int = 422

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message, error_code="VALIDATION_ERROR", **context)


class AudioProcessingError(LiveTranslateError):
    """An error during audio processing."""

    status_code: int = 500

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message, error_code="AUDIO_PROCESSING_ERROR", **context)
