"""
Database Models

SQLAlchemy models for the orchestration service database.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
    JSON,
    Float,
    ForeignKey,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class BotSession(Base):
    """Bot session model"""

    __tablename__ = "bot_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id = Column(String(255), nullable=False, index=True)
    meeting_id = Column(String(255), nullable=False, index=True)
    meeting_title = Column(String(500), nullable=True)
    meeting_uri = Column(Text, nullable=True)
    bot_type = Column(String(50), nullable=False, default="google_meet")

    # Session lifecycle
    status = Column(String(50), nullable=False, default="pending")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)

    # Configuration
    target_languages = Column(JSON, nullable=True)
    enable_translation = Column(Boolean, nullable=False, default=True)
    enable_transcription = Column(Boolean, nullable=False, default=True)
    enable_virtual_webcam = Column(Boolean, nullable=False, default=False)
    audio_storage_enabled = Column(Boolean, nullable=False, default=True)

    # Metadata
    session_metadata = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    audio_files = relationship(
        "AudioFile", back_populates="session", cascade="all, delete-orphan"
    )
    transcripts = relationship(
        "Transcript", back_populates="session", cascade="all, delete-orphan"
    )
    translations = relationship(
        "Translation", back_populates="session", cascade="all, delete-orphan"
    )
    correlations = relationship(
        "Correlation", back_populates="session", cascade="all, delete-orphan"
    )
    participants = relationship(
        "Participant", back_populates="session", cascade="all, delete-orphan"
    )
    events = relationship(
        "SessionEvent", back_populates="session", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_bot_sessions_status", "status"),
        Index("idx_bot_sessions_created_at", "created_at"),
        Index("idx_bot_sessions_meeting_id", "meeting_id"),
        Index("idx_bot_sessions_bot_id", "bot_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": str(self.session_id),
            "bot_id": self.bot_id,
            "meeting_id": self.meeting_id,
            "meeting_title": self.meeting_title,
            "meeting_uri": self.meeting_uri,
            "bot_type": self.bot_type,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "target_languages": self.target_languages,
            "enable_translation": self.enable_translation,
            "enable_transcription": self.enable_transcription,
            "enable_virtual_webcam": self.enable_virtual_webcam,
            "audio_storage_enabled": self.audio_storage_enabled,
            "metadata": self.session_metadata,
            "error_message": self.error_message,
        }


class AudioFile(Base):
    """Audio file model"""

    __tablename__ = "audio_files"

    file_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False
    )

    # File information
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False)
    mime_type = Column(String(100), nullable=False)

    # Audio properties
    duration = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    bit_depth = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

    # Metadata
    session_metadata = Column(JSON, nullable=True)

    # Relationships
    session = relationship("BotSession", back_populates="audio_files")

    # Indexes
    __table_args__ = (
        Index("idx_audio_files_session_id", "session_id"),
        Index("idx_audio_files_created_at", "created_at"),
        Index("idx_audio_files_start_time", "start_time"),
        Index("idx_audio_files_hash", "file_hash"),
    )


class Transcript(Base):
    """Transcript model"""

    __tablename__ = "transcripts"

    transcript_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False
    )

    # Transcript content
    text = Column(Text, nullable=False)
    language = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=True)

    # Source information
    source = Column(String(50), nullable=False)  # 'google_meet', 'whisper', etc.
    audio_file_id = Column(
        UUID(as_uuid=True), ForeignKey("audio_files.file_id"), nullable=True
    )

    # Speaker information
    speaker_id = Column(String(100), nullable=True)
    speaker_name = Column(String(255), nullable=True)

    # Timing information
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    start_offset = Column(Float, nullable=True)  # Offset within audio file
    end_offset = Column(Float, nullable=True)

    # Metadata
    session_metadata = Column(JSON, nullable=True)

    # Relationships
    session = relationship("BotSession", back_populates="transcripts")
    audio_file = relationship("AudioFile")

    # Indexes
    __table_args__ = (
        Index("idx_transcripts_session_id", "session_id"),
        Index("idx_transcripts_start_time", "start_time"),
        Index("idx_transcripts_speaker_id", "speaker_id"),
        Index("idx_transcripts_language", "language"),
        Index("idx_transcripts_source", "source"),
    )


class Translation(Base):
    """Translation model"""

    __tablename__ = "translations"

    translation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False
    )
    transcript_id = Column(
        UUID(as_uuid=True), ForeignKey("transcripts.transcript_id"), nullable=True
    )

    # Translation content
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    source_language = Column(String(10), nullable=False)
    target_language = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=True)

    # Speaker information
    speaker_id = Column(String(100), nullable=True)
    speaker_name = Column(String(255), nullable=True)

    # Timing information
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)

    # Quality metrics
    quality_score = Column(Float, nullable=True)
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)

    # Metadata
    session_metadata = Column(JSON, nullable=True)

    # Relationships
    session = relationship("BotSession", back_populates="translations")
    transcript = relationship("Transcript")

    # Indexes
    __table_args__ = (
        Index("idx_translations_session_id", "session_id"),
        Index("idx_translations_start_time", "start_time"),
        Index("idx_translations_speaker_id", "speaker_id"),
        Index("idx_translations_source_language", "source_language"),
        Index("idx_translations_target_language", "target_language"),
    )


class Correlation(Base):
    """Correlation model for time synchronization"""

    __tablename__ = "correlations"

    correlation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False
    )

    # External source (e.g., Google Meet captions)
    external_source = Column(String(50), nullable=False)
    external_text = Column(Text, nullable=False)
    external_timestamp = Column(DateTime, nullable=False)

    # Internal source (e.g., Whisper transcription)
    internal_source = Column(String(50), nullable=False)
    internal_text = Column(Text, nullable=False)
    internal_timestamp = Column(DateTime, nullable=False)

    # Correlation metrics
    similarity_score = Column(Float, nullable=True)
    time_offset = Column(Float, nullable=True)  # Seconds
    confidence = Column(Float, nullable=True)

    # Metadata
    session_metadata = Column(JSON, nullable=True)

    # Relationships
    session = relationship("BotSession", back_populates="correlations")

    # Indexes
    __table_args__ = (
        Index("idx_correlations_session_id", "session_id"),
        Index("idx_correlations_external_timestamp", "external_timestamp"),
        Index("idx_correlations_internal_timestamp", "internal_timestamp"),
        Index("idx_correlations_similarity_score", "similarity_score"),
    )


class Participant(Base):
    """Meeting participant model"""

    __tablename__ = "participants"

    participant_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False
    )

    # Participant information
    external_id = Column(String(255), nullable=True)  # External system ID
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)

    # Speaker mapping
    speaker_id = Column(String(100), nullable=True, index=True)
    speaker_embedding = Column(JSON, nullable=True)  # Voice embedding data

    # Participation metrics
    joined_at = Column(DateTime, nullable=True)
    left_at = Column(DateTime, nullable=True)
    speaking_time = Column(Float, nullable=True)  # Seconds
    word_count = Column(Integer, nullable=True)

    # Metadata
    session_metadata = Column(JSON, nullable=True)

    # Relationships
    session = relationship("BotSession", back_populates="participants")

    # Indexes
    __table_args__ = (
        Index("idx_participants_session_id", "session_id"),
        Index("idx_participants_speaker_id", "speaker_id"),
        Index("idx_participants_external_id", "external_id"),
        Index("idx_participants_joined_at", "joined_at"),
    )


class SessionEvent(Base):
    """Session event model for debugging and analytics"""

    __tablename__ = "session_events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False
    )

    # Event information
    event_type = Column(String(50), nullable=False)
    event_name = Column(String(255), nullable=False)
    event_data = Column(JSON, nullable=True)

    # Timing
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Severity
    severity = Column(
        String(20), nullable=False, default="info"
    )  # debug, info, warning, error

    # Source
    source = Column(String(100), nullable=False)

    # Relationships
    session = relationship("BotSession", back_populates="events")

    # Indexes
    __table_args__ = (
        Index("idx_session_events_session_id", "session_id"),
        Index("idx_session_events_timestamp", "timestamp"),
        Index("idx_session_events_event_type", "event_type"),
        Index("idx_session_events_severity", "severity"),
    )


class SessionStatistics(Base):
    """Aggregated session statistics"""

    __tablename__ = "session_statistics"

    statistics_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("bot_sessions.session_id"),
        nullable=False,
        unique=True,
    )

    # Basic metrics
    duration = Column(Float, nullable=True)  # Session duration in seconds
    total_participants = Column(Integer, nullable=False, default=0)
    total_audio_files = Column(Integer, nullable=False, default=0)
    total_transcripts = Column(Integer, nullable=False, default=0)
    total_translations = Column(Integer, nullable=False, default=0)
    total_correlations = Column(Integer, nullable=False, default=0)

    # Audio metrics
    total_audio_duration = Column(Float, nullable=True)
    total_audio_size = Column(Integer, nullable=True)
    audio_quality_score = Column(Float, nullable=True)

    # Transcription metrics
    total_words = Column(Integer, nullable=False, default=0)
    total_characters = Column(Integer, nullable=False, default=0)
    average_confidence = Column(Float, nullable=True)
    languages_detected = Column(JSON, nullable=True)

    # Translation metrics
    translation_quality_score = Column(Float, nullable=True)
    translation_coverage = Column(
        Float, nullable=True
    )  # Percentage of transcripts translated

    # Speaker metrics
    unique_speakers = Column(Integer, nullable=False, default=0)
    speaker_distribution = Column(JSON, nullable=True)  # Speaking time distribution

    # Timing metrics
    processing_time = Column(Float, nullable=True)
    correlation_accuracy = Column(Float, nullable=True)

    # Timestamps
    calculated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    session = relationship("BotSession")

    # Indexes
    __table_args__ = (
        Index("idx_session_statistics_session_id", "session_id"),
        Index("idx_session_statistics_calculated_at", "calculated_at"),
    )

