"""
Chat History Database Models

SQLAlchemy models for user conversation persistence and retrieval.
Follows Vexa patterns with user-centric scoping and full-text search capabilities.
"""

from typing import Dict, Any
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    Index,
    event,
    select,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

# Import shared Base from base.py to ensure all models share the same MetaData
from .base import Base


class User(Base):
    """
    User model for multi-tenant chat system

    Follows Vexa pattern with email-based authentication and API tokens.
    Each user can have multiple conversation sessions.
    """

    __tablename__ = "users"

    # Primary key
    user_id = Column(
        String(255), primary_key=True, index=True
    )  # External user ID (e.g., from OAuth)

    # User information
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    image_url = Column(Text, nullable=True)

    # Configuration
    max_concurrent_sessions = Column(
        Integer, nullable=False, server_default="10", default=10
    )
    preferred_language = Column(String(10), nullable=True, default="en")

    # User preferences (JSONB for flexible storage)
    preferences = Column(
        JSONB, nullable=False, server_default=func.jsonb("{}"), default=lambda: {}
    )

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )
    last_active_at = Column(DateTime, nullable=True)

    # Account status
    is_active = Column(Boolean, nullable=False, default=True)

    # Relationships
    conversation_sessions = relationship(
        "ConversationSession", back_populates="user", cascade="all, delete-orphan"
    )
    api_tokens = relationship(
        "APIToken", back_populates="user", cascade="all, delete-orphan"
    )

    # Indexes for efficient querying (email already indexed via Column definition)
    __table_args__ = (
        Index("ix_users_created_at", "created_at"),
        Index("ix_users_last_active", "last_active_at"),
        Index("ix_users_preferences_gin", "preferences", postgresql_using="gin"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "image_url": self.image_url,
            "preferred_language": self.preferred_language,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active_at": self.last_active_at.isoformat()
            if self.last_active_at
            else None,
            "is_active": self.is_active,
        }


class APIToken(Base):
    """
    API token model for user authentication

    Follows Vexa pattern for token-based API access.
    """

    __tablename__ = "api_tokens"

    # Primary key
    token_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Token information
    token = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(
        String(255), ForeignKey("users.user_id"), nullable=False, index=True
    )

    # Token metadata
    name = Column(String(100), nullable=True)  # User-friendly name for token
    scopes = Column(JSONB, nullable=False, default=lambda: ["read", "write"])

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    expires_at = Column(DateTime, nullable=True)  # Null = never expires
    last_used_at = Column(DateTime, nullable=True)

    # Status
    is_active = Column(Boolean, nullable=False, default=True)

    # Relationships
    user = relationship("User", back_populates="api_tokens")

    # Indexes (token and user_id already indexed via Column definitions)
    __table_args__ = (Index("ix_api_tokens_expires_at", "expires_at"),)


class ConversationSession(Base):
    """
    Conversation session model

    Represents a single conversation thread between user and assistant.
    Each session can contain multiple messages and span multiple interactions.
    """

    __tablename__ = "conversation_sessions"

    # Primary key
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # User scoping (CRITICAL for multi-tenant isolation)
    user_id = Column(
        String(255), ForeignKey("users.user_id"), nullable=False, index=True
    )

    # Session metadata
    session_type = Column(
        String(50), nullable=False, default="user_chat"
    )  # user_chat, bot_meeting, etc.
    session_title = Column(
        String(500), nullable=True
    )  # Auto-generated or user-provided

    # Session lifecycle
    started_at = Column(DateTime, nullable=False, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)
    last_message_at = Column(DateTime, nullable=True)

    # Session configuration
    target_languages = Column(JSONB, nullable=True)  # Languages for translation
    enable_translation = Column(Boolean, nullable=False, default=False)

    # Session statistics (denormalized for performance)
    message_count = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)  # Estimated token count

    # Session metadata (flexible JSONB storage)
    session_metadata = Column(JSONB, nullable=True)

    # Relationships
    user = relationship("User", back_populates="conversation_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.sequence_number",
    )

    # Indexes for efficient querying (user_id already indexed via Column definition)
    __table_args__ = (
        Index("ix_conv_sessions_started_at", "started_at"),
        Index("ix_conv_sessions_last_message_at", "last_message_at"),
        Index("ix_conv_sessions_session_type", "session_type"),
        Index(
            "ix_conv_sessions_user_started", "user_id", "started_at"
        ),  # Composite for user queries
        Index(
            "ix_conv_sessions_metadata_gin", "session_metadata", postgresql_using="gin"
        ),
    )

    def to_dict(self, include_messages: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        result = {
            "session_id": str(self.session_id),
            "user_id": self.user_id,
            "session_type": self.session_type,
            "session_title": self.session_title,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "last_message_at": self.last_message_at.isoformat()
            if self.last_message_at
            else None,
            "message_count": self.message_count,
            "enable_translation": self.enable_translation,
            "target_languages": self.target_languages,
        }

        if include_messages and self.messages:
            result["messages"] = [msg.to_dict() for msg in self.messages]

        return result


class ChatMessage(Base):
    """
    Chat message model

    Represents individual messages in a conversation.
    Supports multi-language content and translations via JSONB.
    """

    __tablename__ = "chat_messages"

    # Primary key
    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Session scoping
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.session_id"),
        nullable=False,
        index=True,
    )

    # Message ordering (auto-incrementing within session)
    sequence_number = Column(Integer, nullable=False)

    # Message role
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'

    # Message content
    content = Column(Text, nullable=False)
    original_language = Column(String(10), nullable=True, default="en")

    # Translations (JSONB for flexibility)
    # Format: {"es": "Hola mundo", "fr": "Bonjour le monde"}
    translated_content = Column(JSONB, nullable=True)

    # Message metadata
    timestamp = Column(DateTime, nullable=False, server_default=func.now())
    edited_at = Column(DateTime, nullable=True)

    # Quality metrics
    confidence = Column(Integer, nullable=True)  # 0-100 confidence score
    token_count = Column(Integer, nullable=True)  # Estimated tokens

    # Metadata (flexible JSONB storage)
    message_metadata = Column(JSONB, nullable=True)

    # Relationships
    session = relationship("ConversationSession", back_populates="messages")

    # Indexes for efficient querying (session_id already indexed via Column definition)
    __table_args__ = (
        Index("ix_chat_messages_timestamp", "timestamp"),
        Index("ix_chat_messages_role", "role"),
        Index(
            "ix_chat_messages_sequence", "session_id", "sequence_number"
        ),  # Composite for ordering
        Index(
            "ix_chat_messages_content_fulltext",
            "content",
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
        ),  # Full-text search
        Index(
            "ix_chat_messages_translated_gin",
            "translated_content",
            postgresql_using="gin",
        ),
    )

    def to_dict(self, include_translations: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        result = {
            "message_id": str(self.message_id),
            "session_id": str(self.session_id),
            "sequence_number": self.sequence_number,
            "role": self.role,
            "content": self.content,
            "original_language": self.original_language,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence,
        }

        if include_translations and self.translated_content:
            result["translated_content"] = self.translated_content

        return result


class ConversationStatistics(Base):
    """
    Aggregated conversation statistics

    Denormalized statistics for efficient dashboard queries.
    Updated via triggers or background jobs.
    """

    __tablename__ = "conversation_statistics"

    # Primary key
    statistics_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.session_id"),
        nullable=False,
        unique=True,
    )

    # Message statistics
    total_messages = Column(Integer, nullable=False, default=0)
    user_messages = Column(Integer, nullable=False, default=0)
    assistant_messages = Column(Integer, nullable=False, default=0)

    # Content statistics
    total_characters = Column(Integer, nullable=False, default=0)
    total_words = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)

    # Language statistics
    languages_used = Column(JSONB, nullable=True)  # List of language codes
    translation_count = Column(Integer, nullable=False, default=0)

    # Timing statistics
    duration_seconds = Column(Integer, nullable=True)  # ended_at - started_at
    avg_response_time = Column(
        Integer, nullable=True
    )  # Average assistant response time

    # Quality metrics
    average_confidence = Column(Integer, nullable=True)  # 0-100

    # Timestamps
    calculated_at = Column(DateTime, nullable=False, server_default=func.now())

    # Relationships
    session = relationship("ConversationSession")

    # Indexes
    __table_args__ = (
        Index("ix_conv_stats_session_id", "session_id"),
        Index("ix_conv_stats_calculated_at", "calculated_at"),
    )


# ============================================
# Event Listeners for Auto-populating Fields
# ============================================

# Track sequence numbers during session flush to handle batch inserts
_session_sequence_counters = {}


@event.listens_for(ChatMessage, "before_insert")
def set_message_sequence_number(mapper, connection, target):
    """
    Auto-increment sequence_number for new messages within a session.
    This mimics the PostgreSQL trigger behavior for compatibility.
    Handles batch inserts correctly by tracking in-memory counters.
    """
    if target.sequence_number is None:
        session_id = str(target.session_id)

        # Get current counter for this session
        if session_id not in _session_sequence_counters:
            # Query for max sequence_number in this session from database
            result = connection.execute(
                select(func.coalesce(func.max(ChatMessage.sequence_number), 0)).where(
                    ChatMessage.session_id == target.session_id
                )
            )
            max_seq = result.scalar() or 0
            _session_sequence_counters[session_id] = max_seq

        # Increment and assign
        _session_sequence_counters[session_id] += 1
        target.sequence_number = _session_sequence_counters[session_id]


@event.listens_for(ChatMessage, "after_insert")
def cleanup_sequence_counter(mapper, connection, target):
    """Clean up sequence counter after successful insert"""
    # We keep the counter during the transaction, but this could be optimized
    pass


# Listen to session events to clear counters after commit/rollback
from sqlalchemy.orm import Session as SQLAlchemySession


@event.listens_for(SQLAlchemySession, "after_commit")
def clear_sequence_counters_on_commit(session):
    """Clear sequence counters after successful commit"""
    _session_sequence_counters.clear()


@event.listens_for(SQLAlchemySession, "after_rollback")
def clear_sequence_counters_on_rollback(session):
    """Clear sequence counters after rollback"""
    _session_sequence_counters.clear()
