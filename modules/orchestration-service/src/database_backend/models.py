"""
SQLAlchemy database models for the orchestration service

Comprehensive models for users, sessions, bots, audio processing, and metrics
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func

from .base import Base


class TimestampMixin:
    """Mixin for created/updated timestamps"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDMixin:
    """Mixin for UUID primary keys"""

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
    )


class User(Base, UUIDMixin, TimestampMixin):
    """User model for authentication and session management"""

    __tablename__ = "users"

    # Basic user information
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    username: Mapped[Optional[str]] = mapped_column(
        String(100), unique=True, nullable=True, index=True
    )
    full_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Authentication
    hashed_password: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Profile
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)

    # Metadata
    metadata_: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Last activity tracking
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_active_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    sessions: Mapped[List["Session"]] = relationship(
        "Session", back_populates="owner", lazy="dynamic"
    )
    bot_instances: Mapped[List["BotInstance"]] = relationship(
        "BotInstance", back_populates="owner", lazy="dynamic"
    )

    # Constraints
    __table_args__ = (
        Index("idx_user_email", "email"),
        Index("idx_user_username", "username"),
        Index("idx_user_active", "is_active"),
        CheckConstraint(
            "email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'",
            name="valid_email",
        ),
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class Session(Base, UUIDMixin, TimestampMixin):
    """WebSocket session model"""

    __tablename__ = "sessions"

    # Session identification
    session_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    session_type: Mapped[str] = mapped_column(
        String(50), default="websocket", nullable=False
    )

    # Owner relationship
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    owner: Mapped["User"] = relationship("User", back_populates="sessions")

    # Session status
    status: Mapped[str] = mapped_column(
        String(20), default="active", nullable=False, index=True
    )
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Configuration
    configuration: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Statistics
    participant_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_participants: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    total_messages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Duration tracking
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    audio_sessions: Mapped[List["AudioSession"]] = relationship(
        "AudioSession", back_populates="session", lazy="dynamic"
    )
    translation_sessions: Mapped[List["TranslationSession"]] = relationship(
        "TranslationSession", back_populates="session", lazy="dynamic"
    )
    bot_instances: Mapped[List["BotInstance"]] = relationship(
        "BotInstance", back_populates="session", lazy="dynamic"
    )

    # Constraints
    __table_args__ = (
        Index("idx_session_owner", "owner_id"),
        Index("idx_session_status", "status"),
        Index("idx_session_started", "started_at"),
        CheckConstraint(
            "status IN ('active', 'inactive', 'ended')", name="valid_session_status"
        ),
        CheckConstraint("participant_count >= 0", name="non_negative_participants"),
        CheckConstraint("max_participants > 0", name="positive_max_participants"),
    )

    @hybrid_property
    def duration_minutes(self) -> Optional[float]:
        """Calculate session duration in minutes"""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() / 60
        return (datetime.utcnow() - self.started_at).total_seconds() / 60

    @hybrid_property
    def is_active(self) -> bool:
        """Check if session is currently active"""
        return self.status == "active" and self.ended_at is None

    def __repr__(self):
        return (
            f"<Session(id={self.id}, owner_id={self.owner_id}, status={self.status})>"
        )


class BotInstance(Base, UUIDMixin, TimestampMixin):
    """Bot instance model for Google Meet bot management"""

    __tablename__ = "bot_instances"

    # Bot identification
    bot_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    bot_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Ownership
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    owner: Mapped["User"] = relationship("User", back_populates="bot_instances")

    # Session association
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions.id"), nullable=True, index=True
    )
    session: Mapped[Optional["Session"]] = relationship(
        "Session", back_populates="bot_instances"
    )

    # Bot status
    status: Mapped[str] = mapped_column(
        String(20), default="spawning", nullable=False, index=True
    )
    priority: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)

    # Meeting information
    meeting_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    meeting_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    meeting_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    platform: Mapped[str] = mapped_column(
        String(50), default="google_meet", nullable=False
    )

    # Configuration
    configuration: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Performance metrics
    cpu_usage_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    memory_usage_mb: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    network_bytes_sent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    network_bytes_received: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    # Statistics
    total_audio_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_transcriptions: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    total_translations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Timing
    spawned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    terminated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Error tracking
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_error_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Constraints
    __table_args__ = (
        Index("idx_bot_id", "bot_id"),
        Index("idx_bot_owner", "owner_id"),
        Index("idx_bot_session", "session_id"),
        Index("idx_bot_status", "status"),
        Index("idx_bot_meeting", "meeting_id"),
        Index("idx_bot_spawned", "spawned_at"),
        CheckConstraint(
            "status IN ('spawning', 'joining', 'active', 'recording', 'processing', 'error', 'terminating', 'terminated')",
            name="valid_bot_status",
        ),
        CheckConstraint(
            "priority IN ('low', 'medium', 'high', 'critical')",
            name="valid_bot_priority",
        ),
        CheckConstraint(
            "cpu_usage_percent >= 0 AND cpu_usage_percent <= 100",
            name="valid_cpu_usage",
        ),
        CheckConstraint("memory_usage_mb >= 0", name="non_negative_memory"),
        CheckConstraint("error_count >= 0", name="non_negative_errors"),
    )

    @hybrid_property
    def uptime_minutes(self) -> Optional[float]:
        """Calculate bot uptime in minutes"""
        end_time = self.terminated_at or datetime.utcnow()
        return (end_time - self.spawned_at).total_seconds() / 60

    @hybrid_property
    def is_active(self) -> bool:
        """Check if bot is currently active"""
        return (
            self.status in ("active", "recording", "processing")
            and self.terminated_at is None
        )

    def __repr__(self):
        return (
            f"<BotInstance(id={self.id}, bot_id={self.bot_id}, status={self.status})>"
        )


class AudioSession(Base, UUIDMixin, TimestampMixin):
    """Audio processing session model"""

    __tablename__ = "audio_sessions"

    # Session association
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions.id"), nullable=True, index=True
    )
    session: Mapped[Optional["Session"]] = relationship(
        "Session", back_populates="audio_sessions"
    )

    # Bot association
    bot_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("bot_instances.id"), nullable=True, index=True
    )

    # Audio configuration
    sample_rate: Mapped[int] = mapped_column(Integer, default=16000, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    format: Mapped[str] = mapped_column(String(20), default="wav", nullable=False)

    # Processing configuration
    processing_config: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    enabled_stages: Mapped[List[str]] = mapped_column(
        JSON, default=list, nullable=False
    )

    # Statistics
    total_chunks_processed: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    total_duration_seconds: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )
    average_processing_time_ms: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )
    average_quality_score: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )

    # Error tracking
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default="active", nullable=False, index=True
    )

    # Constraints
    __table_args__ = (
        Index("idx_audio_session", "session_id"),
        Index("idx_audio_bot", "bot_id"),
        Index("idx_audio_status", "status"),
        CheckConstraint("sample_rate > 0", name="positive_sample_rate"),
        CheckConstraint("channels > 0", name="positive_channels"),
        CheckConstraint("total_duration_seconds >= 0", name="non_negative_duration"),
        CheckConstraint(
            "average_quality_score >= 0 AND average_quality_score <= 1",
            name="valid_quality_score",
        ),
        CheckConstraint(
            "status IN ('active', 'paused', 'completed', 'error')",
            name="valid_audio_status",
        ),
    )

    def __repr__(self):
        return f"<AudioSession(id={self.id}, session_id={self.session_id}, status={self.status})>"


class TranslationSession(Base, UUIDMixin, TimestampMixin):
    """Translation session model"""

    __tablename__ = "translation_sessions"

    # Session association
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions.id"), nullable=True, index=True
    )
    session: Mapped[Optional["Session"]] = relationship(
        "Session", back_populates="translation_sessions"
    )

    # Bot association
    bot_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("bot_instances.id"), nullable=True, index=True
    )

    # Language configuration
    source_language: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True
    )  # Auto-detect if None
    target_languages: Mapped[List[str]] = mapped_column(
        JSON, default=list, nullable=False
    )

    # Translation configuration
    translation_config: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    quality_setting: Mapped[str] = mapped_column(
        String(20), default="balanced", nullable=False
    )

    # Statistics
    total_translations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_characters_translated: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    average_confidence_score: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )
    average_translation_time_ms: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )

    # Language distribution
    language_distribution: Mapped[Dict[str, int]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Error tracking
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default="active", nullable=False, index=True
    )

    # Constraints
    __table_args__ = (
        Index("idx_translation_session", "session_id"),
        Index("idx_translation_bot", "bot_id"),
        Index("idx_translation_status", "status"),
        CheckConstraint("total_translations >= 0", name="non_negative_translations"),
        CheckConstraint(
            "total_characters_translated >= 0", name="non_negative_characters"
        ),
        CheckConstraint(
            "average_confidence_score >= 0 AND average_confidence_score <= 1",
            name="valid_confidence",
        ),
        CheckConstraint(
            "quality_setting IN ('fast', 'balanced', 'accurate')",
            name="valid_quality_setting",
        ),
        CheckConstraint(
            "status IN ('active', 'paused', 'completed', 'error')",
            name="valid_translation_status",
        ),
    )

    def __repr__(self):
        return f"<TranslationSession(id={self.id}, session_id={self.session_id}, status={self.status})>"


class SystemMetrics(Base, UUIDMixin, TimestampMixin):
    """System metrics and performance data"""

    __tablename__ = "system_metrics"

    # Metric identification
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # counter, gauge, histogram

    # Metric value
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Labels and tags
    labels: Mapped[Dict[str, str]] = mapped_column(JSON, default=dict, nullable=False)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)

    # Source information
    source_service: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    source_instance: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Timestamp (for time-series data)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now(), index=True
    )

    # Additional metadata
    metadata_: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Constraints
    __table_args__ = (
        Index("idx_metric_name", "metric_name"),
        Index("idx_metric_type", "metric_type"),
        Index("idx_metric_service", "source_service"),
        Index("idx_metric_timestamp", "timestamp"),
        Index("idx_metric_composite", "metric_name", "source_service", "timestamp"),
        CheckConstraint(
            "metric_type IN ('counter', 'gauge', 'histogram', 'summary')",
            name="valid_metric_type",
        ),
    )

    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, metric_name={self.metric_name}, value={self.value})>"


# Additional utility tables for advanced features


class WebSocketConnection(Base, UUIDMixin, TimestampMixin):
    """WebSocket connection tracking"""

    __tablename__ = "websocket_connections"

    # Connection identification
    connection_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # User association
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions.id"), nullable=True, index=True
    )

    # Connection details
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)  # IPv6 support
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status and timing
    status: Mapped[str] = mapped_column(
        String(20), default="connected", nullable=False, index=True
    )
    connected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    disconnected_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )

    # Statistics
    messages_sent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    messages_received: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    bytes_sent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    bytes_received: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Constraints
    __table_args__ = (
        Index("idx_ws_connection", "connection_id"),
        Index("idx_ws_user", "user_id"),
        Index("idx_ws_session", "session_id"),
        Index("idx_ws_status", "status"),
        Index("idx_ws_connected", "connected_at"),
        CheckConstraint(
            "status IN ('connecting', 'connected', 'disconnecting', 'disconnected')",
            name="valid_ws_status",
        ),
    )

    @hybrid_property
    def duration_minutes(self) -> Optional[float]:
        """Calculate connection duration in minutes"""
        end_time = self.disconnected_at or datetime.utcnow()
        return (end_time - self.connected_at).total_seconds() / 60

    def __repr__(self):
        return f"<WebSocketConnection(id={self.id}, connection_id={self.connection_id}, status={self.status})>"
