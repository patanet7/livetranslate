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
]
