"""
Pydantic models for request/response validation

Modern data validation with automatic OpenAPI schema generation
"""

from .audio import (
    AudioConfiguration,
    AudioProcessingRequest,
    AudioProcessingResponse,
    AudioStats,
)
from .base import BaseModel, TimestampMixin
from .bot import BotConfiguration, BotResponse, BotSpawnRequest, BotStats, BotStatus
from .config import ConfigResponse, ConfigUpdate
from .fireflies import (
    # Constants
    FIREFLIES_SOURCE_TYPE,
    TRANSCRIPT_SOURCE_TYPES,
    ActiveMeetingsResponse,
    CaptionBroadcast,
    # Caption Output
    CaptionEntry,
    # API Models
    FirefliesChunk,
    FirefliesConnectionStatus,
    # API Request/Response
    FirefliesConnectRequest,
    FirefliesConnectResponse,
    FirefliesEvent,
    # Enums
    FirefliesEventType,
    FirefliesMeeting,
    FirefliesSession,
    # Session Models
    FirefliesSessionConfig,
    FirefliesStatusResponse,
    MeetingState,
    # Sentence Aggregation
    SpeakerBuffer,
    # Translation
    TranslationContext,
    TranslationResult,
    TranslationUnit,
)
from .system import ErrorResponse, ServiceHealth, SystemStatus
from .websocket import ConnectionStats, WebSocketMessage, WebSocketResponse

__all__ = [
    # Fireflies models
    "FIREFLIES_SOURCE_TYPE",
    "TRANSCRIPT_SOURCE_TYPES",
    "ActiveMeetingsResponse",
    "AudioConfiguration",
    # Audio models
    "AudioProcessingRequest",
    "AudioProcessingResponse",
    "AudioStats",
    # Base models
    "BaseModel",
    "BotConfiguration",
    "BotResponse",
    # Bot models
    "BotSpawnRequest",
    "BotStats",
    "BotStatus",
    "CaptionBroadcast",
    "CaptionEntry",
    "ConfigResponse",
    # Configuration models
    "ConfigUpdate",
    "ConnectionStats",
    "ErrorResponse",
    "FirefliesChunk",
    "FirefliesConnectRequest",
    "FirefliesConnectResponse",
    "FirefliesConnectionStatus",
    "FirefliesEvent",
    "FirefliesEventType",
    "FirefliesMeeting",
    "FirefliesSession",
    "FirefliesSessionConfig",
    "FirefliesStatusResponse",
    "MeetingState",
    "ServiceHealth",
    "SpeakerBuffer",
    # System models
    "SystemStatus",
    "TimestampMixin",
    "TranslationContext",
    "TranslationResult",
    "TranslationUnit",
    # WebSocket models
    "WebSocketMessage",
    "WebSocketResponse",
]
