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
from livetranslate_common.models import (
    AudioChunk,
    BackendConfig,
    BackendSwitchedMessage,
    ConfigMessage,
    ConnectedMessage,
    EndMessage,
    EndMeetingMessage,
    EndSessionMessage,
    InterimMessage,
    LanguageDetectedMessage,
    MeetingAudioStream,
    MeetingStartedMessage,
    ModelInfo,
    PROTOCOL_VERSION,
    PromoteToMeetingMessage,
    RecordingStatusMessage,
    Segment,
    SegmentMessage,
    ServiceStatusMessage,
    StartSessionMessage,
    TranscriptionResult,
    TranslationContext,
    TranslationMessage,
    TranslationRequest,
    TranslationResponse,
    parse_ws_message,
)

__all__ = [
    # infrastructure
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
    # shared models — transcription
    "ModelInfo",
    "Segment",
    "TranscriptionResult",
    # shared models — audio
    "AudioChunk",
    "MeetingAudioStream",
    # shared models — translation
    "TranslationContext",
    "TranslationRequest",
    "TranslationResponse",
    # shared models — WebSocket messages
    "PROTOCOL_VERSION",
    "BackendSwitchedMessage",
    "ConfigMessage",
    "ConnectedMessage",
    "EndMessage",
    "EndMeetingMessage",
    "EndSessionMessage",
    "InterimMessage",
    "LanguageDetectedMessage",
    "MeetingStartedMessage",
    "PromoteToMeetingMessage",
    "RecordingStatusMessage",
    "SegmentMessage",
    "ServiceStatusMessage",
    "StartSessionMessage",
    "TranslationMessage",
    "parse_ws_message",
    # shared models — registry
    "BackendConfig",
]
