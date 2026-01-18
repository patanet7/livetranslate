"""
WebSocket-related Pydantic models
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from .base import BaseModel, ResponseMixin, TimestampMixin


class MessageType(str, Enum):
    """WebSocket message types"""

    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"

    # Session management
    CREATE_SESSION = "create_session"
    JOIN_SESSION = "join_session"
    LEAVE_SESSION = "leave_session"
    SESSION_UPDATE = "session_update"

    # Audio processing
    AUDIO_START = "audio_start"
    AUDIO_CHUNK = "audio_chunk"
    AUDIO_END = "audio_end"
    AUDIO_RESULT = "audio_result"

    # Transcription
    TRANSCRIPTION_START = "transcription_start"
    TRANSCRIPTION_PARTIAL = "transcription_partial"
    TRANSCRIPTION_FINAL = "transcription_final"

    # Translation
    TRANSLATION_REQUEST = "translation_request"
    TRANSLATION_RESULT = "translation_result"

    # Bot management
    BOT_SPAWN = "bot_spawn"
    BOT_STATUS = "bot_status"
    BOT_TERMINATE = "bot_terminate"
    BOT_UPDATE = "bot_update"

    # System notifications
    SYSTEM_NOTIFICATION = "system_notification"
    SERVICE_STATUS = "service_status"
    HEALTH_UPDATE = "health_update"

    # Broadcasting
    BROADCAST = "broadcast"
    USER_MESSAGE = "user_message"


class ConnectionStatus(str, Enum):
    """WebSocket connection status"""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class SessionRole(str, Enum):
    """User role in session"""

    OWNER = "owner"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""

    type: MessageType = Field(description="Message type")
    data: dict[str, Any] = Field(default_factory=dict, description="Message data payload")
    session_id: str | None = Field(default=None, description="Session identifier")
    user_id: str | None = Field(default=None, description="User identifier")
    request_id: str | None = Field(default=None, description="Request identifier for correlation")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Message timestamp"
    )
    priority: int = Field(
        default=0, description="Message priority (higher = more urgent)", ge=0, le=10
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "audio_result",
                "data": {
                    "transcription": "Hello, this is a test.",
                    "confidence": 0.95,
                    "language": "en",
                },
                "session_id": "session_abc123",
                "user_id": "user_def456",
                "request_id": "req_ghi789",
                "timestamp": "2024-01-15T10:30:00Z",
                "priority": 5,
            }
        }
    )


class WebSocketResponse(ResponseMixin, TimestampMixin):
    """WebSocket response message"""

    type: MessageType = Field(description="Response message type")
    request_id: str | None = Field(default=None, description="Original request ID")
    data: dict[str, Any] = Field(default_factory=dict, description="Response data")
    error_code: str | None = Field(default=None, description="Error code if applicable")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Session created successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "type": "create_session",
                "request_id": "req_abc123",
                "data": {"session_id": "session_def456", "participants": 1},
            }
        }
    )


class ConnectionInfo(BaseModel):
    """WebSocket connection information"""

    connection_id: str = Field(
        description="Connection identifier", json_schema_extra={"example": "conn_abc123def456"}
    )
    user_id: str | None = Field(default=None, description="Associated user ID")
    session_id: str | None = Field(default=None, description="Associated session ID")
    status: ConnectionStatus = Field(description="Connection status")
    connected_at: datetime = Field(description="Connection timestamp")
    last_activity: datetime = Field(description="Last activity timestamp")
    ip_address: str = Field(
        description="Client IP address", json_schema_extra={"example": "192.168.1.100"}
    )
    user_agent: str | None = Field(default=None, description="Client user agent")
    protocol_version: str = Field(default="13", description="WebSocket protocol version")

    # Statistics
    messages_sent: int = Field(default=0, description="Number of messages sent")
    messages_received: int = Field(default=0, description="Number of messages received")
    bytes_sent: int = Field(default=0, description="Total bytes sent")
    bytes_received: int = Field(default=0, description="Total bytes received")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "connection_id": "conn_abc123def456",
                "user_id": "user_123",
                "session_id": "session_456",
                "status": "authenticated",
                "connected_at": "2024-01-15T10:00:00Z",
                "last_activity": "2024-01-15T10:30:00Z",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0...",
                "protocol_version": "13",
                "messages_sent": 25,
                "messages_received": 30,
                "bytes_sent": 5120,
                "bytes_received": 7680,
            }
        }
    )


class SessionInfo(BaseModel):
    """WebSocket session information"""

    session_id: str = Field(
        description="Session identifier", json_schema_extra={"example": "session_abc123def456"}
    )
    owner_id: str = Field(description="Session owner user ID")
    created_at: datetime = Field(description="Session creation timestamp")
    last_activity: datetime = Field(description="Last session activity")
    participant_count: int = Field(
        description="Number of participants", json_schema_extra={"example": 3}
    )
    participants: list[dict[str, Any]] = Field(
        description="Session participants",
        json_schema_extra={
            "example": [
                {"user_id": "user_1", "role": "owner", "joined_at": "2024-01-15T10:00:00Z"},
                {
                    "user_id": "user_2",
                    "role": "participant",
                    "joined_at": "2024-01-15T10:05:00Z",
                },
            ]
        },
    )
    configuration: dict[str, Any] = Field(default_factory=dict, description="Session configuration")

    # Session statistics
    total_messages: int = Field(default=0, description="Total messages in session")
    audio_chunks_processed: int = Field(default=0, description="Audio chunks processed in session")
    transcriptions_generated: int = Field(
        default=0, description="Transcriptions generated in session"
    )
    translations_generated: int = Field(default=0, description="Translations generated in session")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "session_abc123def456",
                "owner_id": "user_owner123",
                "created_at": "2024-01-15T10:00:00Z",
                "last_activity": "2024-01-15T10:30:00Z",
                "participant_count": 3,
                "participants": [
                    {
                        "user_id": "user_owner123",
                        "role": "owner",
                        "joined_at": "2024-01-15T10:00:00Z",
                    },
                    {
                        "user_id": "user_participant456",
                        "role": "participant",
                        "joined_at": "2024-01-15T10:05:00Z",
                    },
                ],
                "configuration": {
                    "audio_enabled": True,
                    "translation_enabled": True,
                    "languages": ["en", "es"],
                },
                "total_messages": 150,
                "audio_chunks_processed": 45,
                "transcriptions_generated": 12,
                "translations_generated": 24,
            }
        }
    )


class ConnectionStats(BaseModel):
    """WebSocket connection statistics"""

    total_connections: int = Field(
        description="Total connections ever made", json_schema_extra={"example": 2500}
    )
    active_connections: int = Field(
        description="Currently active connections", json_schema_extra={"example": 150}
    )
    authenticated_connections: int = Field(
        description="Currently authenticated connections", json_schema_extra={"example": 125}
    )
    peak_connections: int = Field(
        description="Peak concurrent connections", json_schema_extra={"example": 300}
    )

    # Session statistics
    total_sessions: int = Field(
        description="Total sessions created", json_schema_extra={"example": 450}
    )
    active_sessions: int = Field(
        description="Currently active sessions", json_schema_extra={"example": 25}
    )
    average_session_duration_minutes: float = Field(
        description="Average session duration in minutes", json_schema_extra={"example": 35.7}
    )

    # Message statistics
    total_messages_sent: int = Field(
        description="Total messages sent to clients", json_schema_extra={"example": 125000}
    )
    total_messages_received: int = Field(
        description="Total messages received from clients", json_schema_extra={"example": 98000}
    )
    messages_per_second: float = Field(
        description="Current messages per second", json_schema_extra={"example": 15.3}
    )

    # Bandwidth statistics
    total_bytes_sent: int = Field(
        description="Total bytes sent",
        json_schema_extra={"example": 52428800},  # 50 MB
    )
    total_bytes_received: int = Field(
        description="Total bytes received",
        json_schema_extra={"example": 41943040},  # 40 MB
    )
    bandwidth_usage_mbps: float = Field(
        description="Current bandwidth usage in Mbps", json_schema_extra={"example": 2.5}
    )

    # Error statistics
    connection_errors: int = Field(
        description="Connection errors", json_schema_extra={"example": 25}
    )
    message_errors: int = Field(
        description="Message processing errors", json_schema_extra={"example": 12}
    )
    timeout_errors: int = Field(description="Timeout errors", json_schema_extra={"example": 8})

    # Performance metrics
    average_response_time_ms: float = Field(
        description="Average response time in milliseconds", json_schema_extra={"example": 45.2}
    )
    connection_pool_usage_percent: float = Field(
        description="Connection pool usage percentage",
        ge=0.0,
        le=100.0,
        json_schema_extra={"example": 15.0},
    )

    # Recent activity
    last_connection_at: datetime | None = Field(
        default=None, description="Last connection timestamp"
    )
    last_disconnection_at: datetime | None = Field(
        default=None, description="Last disconnection timestamp"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_connections": 2500,
                "active_connections": 150,
                "authenticated_connections": 125,
                "peak_connections": 300,
                "total_sessions": 450,
                "active_sessions": 25,
                "average_session_duration_minutes": 35.7,
                "total_messages_sent": 125000,
                "total_messages_received": 98000,
                "messages_per_second": 15.3,
                "total_bytes_sent": 52428800,
                "total_bytes_received": 41943040,
                "bandwidth_usage_mbps": 2.5,
                "connection_errors": 25,
                "message_errors": 12,
                "timeout_errors": 8,
                "average_response_time_ms": 45.2,
                "connection_pool_usage_percent": 15.0,
                "last_connection_at": "2024-01-15T10:30:00Z",
                "last_disconnection_at": "2024-01-15T10:28:00Z",
            }
        }
    )


class WebSocketEvent(BaseModel):
    """WebSocket event for logging/analytics"""

    event_type: str = Field(
        description="Event type", json_schema_extra={"example": "connection_established"}
    )
    connection_id: str = Field(description="Connection identifier")
    user_id: str | None = Field(default=None, description="User identifier")
    session_id: str | None = Field(default=None, description="Session identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Event timestamp"
    )
    data: dict[str, Any] = Field(default_factory=dict, description="Event data")
    ip_address: str | None = Field(default=None, description="Client IP address")
    user_agent: str | None = Field(default=None, description="Client user agent")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event_type": "message_received",
                "connection_id": "conn_abc123",
                "user_id": "user_def456",
                "session_id": "session_ghi789",
                "timestamp": "2024-01-15T10:30:00Z",
                "data": {"message_type": "audio_chunk", "size_bytes": 2048},
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0...",
            }
        }
    )


class BroadcastMessage(BaseModel):
    """Message for broadcasting to multiple connections"""

    message: WebSocketMessage = Field(description="Message to broadcast")
    target_type: str = Field(
        description="Target type for broadcast", json_schema_extra={"example": "session"}
    )
    target_ids: list[str] = Field(
        description="Target identifiers", json_schema_extra={"example": ["session_1", "session_2"]}
    )
    exclude_connection_ids: list[str] = Field(
        default_factory=list, description="Connection IDs to exclude from broadcast"
    )
    delivery_method: str = Field(
        default="best_effort",
        description="Delivery method",
        json_schema_extra={"example": "best_effort"},
    )

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v, info=None):
        """Validate target type"""
        valid_types = ["all", "session", "user", "role", "connection"]
        if v not in valid_types:
            raise ValueError(f"Target type must be one of: {valid_types}")
        return v

    @field_validator("delivery_method")
    @classmethod
    def validate_delivery_method(cls, v, info=None):
        """Validate delivery method"""
        valid_methods = ["best_effort", "guaranteed", "priority"]
        if v not in valid_methods:
            raise ValueError(f"Delivery method must be one of: {valid_methods}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": {
                    "type": "system_notification",
                    "data": {
                        "title": "System Maintenance",
                        "body": "Scheduled maintenance in 10 minutes",
                    },
                },
                "target_type": "all",
                "target_ids": [],
                "exclude_connection_ids": ["conn_admin123"],
                "delivery_method": "guaranteed",
            }
        }
    )
