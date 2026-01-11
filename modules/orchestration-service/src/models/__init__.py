"""
Pydantic models for request/response validation

Modern data validation with automatic OpenAPI schema generation
"""

from .base import BaseModel, TimestampMixin
from .system import SystemStatus, ServiceHealth, ErrorResponse
from .config import ConfigUpdate, ConfigResponse
from .audio import (
    AudioProcessingRequest,
    AudioProcessingResponse,
    AudioConfiguration,
    AudioStats,
)
from .bot import BotSpawnRequest, BotResponse, BotStatus, BotStats, BotConfiguration
from .websocket import WebSocketMessage, WebSocketResponse, ConnectionStats
from .fireflies import (
    # Constants
    FIREFLIES_SOURCE_TYPE,
    TRANSCRIPT_SOURCE_TYPES,
    # Enums
    FirefliesEventType,
    FirefliesConnectionStatus,
    MeetingState,
    # API Models
    FirefliesChunk,
    FirefliesEvent,
    FirefliesMeeting,
    # Session Models
    FirefliesSessionConfig,
    FirefliesSession,
    # Sentence Aggregation
    SpeakerBuffer,
    TranslationUnit,
    # Translation
    TranslationContext,
    TranslationResult,
    # Caption Output
    CaptionEntry,
    CaptionBroadcast,
    # API Request/Response
    FirefliesConnectRequest,
    FirefliesConnectResponse,
    FirefliesStatusResponse,
    ActiveMeetingsResponse,
)

__all__ = [
    # Base models
    "BaseModel",
    "TimestampMixin",
    # System models
    "SystemStatus",
    "ServiceHealth",
    "ErrorResponse",
    # Configuration models
    "ConfigUpdate",
    "ConfigResponse",
    # Audio models
    "AudioProcessingRequest",
    "AudioProcessingResponse",
    "AudioConfiguration",
    "AudioStats",
    # Bot models
    "BotSpawnRequest",
    "BotResponse",
    "BotStatus",
    "BotStats",
    "BotConfiguration",
    # WebSocket models
    "WebSocketMessage",
    "WebSocketResponse",
    "ConnectionStats",
    # Fireflies models
    "FIREFLIES_SOURCE_TYPE",
    "TRANSCRIPT_SOURCE_TYPES",
    "FirefliesEventType",
    "FirefliesConnectionStatus",
    "MeetingState",
    "FirefliesChunk",
    "FirefliesEvent",
    "FirefliesMeeting",
    "FirefliesSessionConfig",
    "FirefliesSession",
    "SpeakerBuffer",
    "TranslationUnit",
    "TranslationContext",
    "TranslationResult",
    "CaptionEntry",
    "CaptionBroadcast",
    "FirefliesConnectRequest",
    "FirefliesConnectResponse",
    "FirefliesStatusResponse",
    "ActiveMeetingsResponse",
]
