"""LiveTranslate Common - Shared utilities for all LiveTranslate services."""

__version__ = "0.1.0"

from livetranslate_common.config import ServiceSettings
from livetranslate_common.errors import (
    AudioProcessingError,
    LiveTranslateError,
    ServiceUnavailableError,
    ValidationError,
)
from livetranslate_common.errors.handlers import register_error_handlers
from livetranslate_common.health import create_health_router
from livetranslate_common.logging import get_logger, log_performance, setup_logging
from livetranslate_common.middleware import RequestIDMiddleware, RequestLoggingMiddleware

__all__ = [
    "AudioProcessingError",
    "LiveTranslateError",
    "RequestIDMiddleware",
    "RequestLoggingMiddleware",
    "ServiceSettings",
    "ServiceUnavailableError",
    "ValidationError",
    "create_health_router",
    "get_logger",
    "log_performance",
    "register_error_handlers",
    "setup_logging",
]
