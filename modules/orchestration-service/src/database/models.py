"""
Database Models

SQLAlchemy models for the orchestration service database.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

# Import shared Base from base.py to ensure all models share the same MetaData
from .base import Base


def utc_now() -> datetime:
    """Return current UTC time (timezone-naive for TIMESTAMP WITHOUT TIME ZONE columns)."""
    return datetime.now(UTC).replace(tzinfo=None)


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
    created_at = Column(DateTime, nullable=False, default=utc_now)
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
    audio_files = relationship("AudioFile", back_populates="session", cascade="all, delete-orphan")
    transcripts = relationship("Transcript", back_populates="session", cascade="all, delete-orphan")
    translations = relationship(
        "Translation", back_populates="session", cascade="all, delete-orphan"
    )
    correlations = relationship(
        "Correlation", back_populates="session", cascade="all, delete-orphan"
    )
    participants = relationship(
        "Participant", back_populates="session", cascade="all, delete-orphan"
    )
    events = relationship("SessionEvent", back_populates="session", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_bot_sessions_status", "status"),
        Index("idx_bot_sessions_created_at", "created_at"),
        Index("idx_bot_sessions_meeting_id", "meeting_id"),
        Index("idx_bot_sessions_bot_id", "bot_id"),
    )

    def to_dict(self) -> dict[str, Any]:
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
    session_id = Column(UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False)

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
    created_at = Column(DateTime, nullable=False, default=utc_now)
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
    session_id = Column(UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False)

    # Transcript content
    text = Column(Text, nullable=False)
    language = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=True)

    # Source information
    source = Column(String(50), nullable=False)  # 'google_meet', 'whisper', etc.
    audio_file_id = Column(UUID(as_uuid=True), ForeignKey("audio_files.file_id"), nullable=True)

    # Speaker information
    speaker_id = Column(String(100), nullable=True)
    speaker_name = Column(String(255), nullable=True)

    # Timing information
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
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
    session_id = Column(UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False)
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
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

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
    session_id = Column(UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False)

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
    session_id = Column(UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False)

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
        UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=True
    )  # Nullable to support dashboard settings not tied to a session

    # Event information
    event_type = Column(String(50), nullable=False)
    event_name = Column(String(255), nullable=False)
    event_data = Column(JSON, nullable=True)

    # Timing
    timestamp = Column(DateTime, nullable=False, default=utc_now)

    # Severity
    severity = Column(String(20), nullable=False, default="info")  # debug, info, warning, error

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
    translation_coverage = Column(Float, nullable=True)  # Percentage of transcripts translated

    # Speaker metrics
    unique_speakers = Column(Integer, nullable=False, default=0)
    speaker_distribution = Column(JSON, nullable=True)  # Speaking time distribution

    # Timing metrics
    processing_time = Column(Float, nullable=True)
    correlation_accuracy = Column(Float, nullable=True)

    # Timestamps
    calculated_at = Column(DateTime, nullable=False, default=utc_now)

    # Relationships
    session = relationship("BotSession")

    # Indexes
    __table_args__ = (
        Index("idx_session_statistics_session_id", "session_id"),
        Index("idx_session_statistics_calculated_at", "calculated_at"),
    )


# =============================================================================
# Glossary Models (for Fireflies Integration)
# =============================================================================


class Glossary(Base):
    """
    Glossary collection for translation term consistency.

    Used by Fireflies integration to ensure consistent translation of
    domain-specific terms, proper nouns, and technical vocabulary.
    """

    __tablename__ = "glossaries"

    glossary_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Glossary identification
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Domain categorization
    domain = Column(String(100), nullable=True, index=True)  # e.g., 'medical', 'legal', 'tech'

    # Language settings
    source_language = Column(String(10), nullable=False, default="en")
    target_languages = Column(JSON, nullable=False)  # List of target language codes

    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    is_default = Column(Boolean, nullable=False, default=False)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)
    created_by = Column(String(255), nullable=True)

    # Entry count (denormalized for performance)
    entry_count = Column(Integer, nullable=False, default=0)

    # Relationships
    entries = relationship("GlossaryEntry", back_populates="glossary", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_glossaries_name", "name"),
        Index("idx_glossaries_domain", "domain"),
        Index("idx_glossaries_is_active", "is_active"),
        Index("idx_glossaries_source_language", "source_language"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "glossary_id": str(self.glossary_id),
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "source_language": self.source_language,
            "target_languages": self.target_languages,
            "is_active": self.is_active,
            "is_default": self.is_default,
            "entry_count": self.entry_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class GlossaryEntry(Base):
    """
    Individual glossary entry mapping source term to translations.

    Supports multiple target language translations per source term.
    """

    __tablename__ = "glossary_entries"

    entry_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    glossary_id = Column(UUID(as_uuid=True), ForeignKey("glossaries.glossary_id"), nullable=False)

    # Source term
    source_term = Column(String(500), nullable=False)
    source_term_normalized = Column(
        String(500), nullable=False, index=True
    )  # Lowercase for matching

    # Target language translations (JSON: {"es": "término", "fr": "terme"})
    translations = Column(JSON, nullable=False)

    # Context and notes
    context = Column(Text, nullable=True)  # Usage context or example
    notes = Column(Text, nullable=True)  # Internal notes

    # Whisper prompting fields (for transcription accuracy)
    phonetic = Column(String(255), nullable=True)  # Pronunciation hint for Whisper
    common_context = Column(Text, nullable=True)  # Common usage context for prompting

    # Matching settings
    case_sensitive = Column(Boolean, nullable=False, default=False)
    match_whole_word = Column(Boolean, nullable=False, default=True)

    # Priority (higher = more important, used when terms conflict)
    priority = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)

    # Relationships
    glossary = relationship("Glossary", back_populates="entries")

    # Indexes
    __table_args__ = (
        Index("idx_glossary_entries_glossary_id", "glossary_id"),
        Index("idx_glossary_entries_source_term", "source_term_normalized"),
        Index("idx_glossary_entries_priority", "priority"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entry_id": str(self.entry_id),
            "glossary_id": str(self.glossary_id),
            "source_term": self.source_term,
            "translations": self.translations,
            "context": self.context,
            "notes": self.notes,
            "phonetic": self.phonetic,
            "common_context": self.common_context,
            "case_sensitive": self.case_sensitive,
            "match_whole_word": self.match_whole_word,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def get_translation(self, target_language: str) -> str | None:
        """Get translation for a specific target language"""
        if self.translations and target_language in self.translations:
            translation: str = self.translations[target_language]
            return translation
        return None


# =============================================================================
# Meeting Intelligence Models
# =============================================================================


class MeetingNote(Base):
    """Real-time meeting notes (auto-generated, manual, or annotated)."""

    __tablename__ = "meeting_notes"

    note_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False)

    # Note classification
    note_type = Column(String(50), nullable=False, default="manual")  # auto, manual, annotation

    # Content
    content = Column(Text, nullable=False)
    prompt_used = Column(Text, nullable=True)  # Prompt that generated it (null for manual)
    context_sentences = Column(JSON, nullable=True)  # Transcript sentences used as context
    speaker_name = Column(String(255), nullable=True)  # If note is about a specific speaker

    # Transcript range
    transcript_range_start = Column(Float, nullable=True)
    transcript_range_end = Column(Float, nullable=True)

    # LLM metadata
    llm_backend = Column(String(100), nullable=True)
    llm_model = Column(String(255), nullable=True)
    processing_time_ms = Column(Float, nullable=True)

    # Metadata
    note_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)

    # Relationships
    session = relationship("BotSession")

    # Indexes
    __table_args__ = (
        Index("idx_meeting_notes_session_id", "session_id"),
        Index("idx_meeting_notes_note_type", "note_type"),
        Index("idx_meeting_notes_created_at", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "note_id": str(self.note_id),
            "session_id": str(self.session_id),
            "note_type": self.note_type,
            "content": self.content,
            "prompt_used": self.prompt_used,
            "context_sentences": self.context_sentences,
            "speaker_name": self.speaker_name,
            "transcript_range_start": self.transcript_range_start,
            "transcript_range_end": self.transcript_range_end,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.note_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MeetingInsight(Base):
    """Post-meeting generated insights."""

    __tablename__ = "meeting_insights"

    insight_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False)
    template_id = Column(
        UUID(as_uuid=True),
        ForeignKey("insight_prompt_templates.template_id"),
        nullable=True,
    )

    # Insight classification
    insight_type = Column(
        String(50), nullable=False, default="summary"
    )  # summary, action_items, decisions, custom
    title = Column(String(500), nullable=False)

    # Content
    content = Column(Text, nullable=False)
    prompt_used = Column(Text, nullable=True)  # Full prompt sent to LLM
    transcript_length = Column(Integer, nullable=True)  # Chars of transcript used

    # LLM metadata
    llm_backend = Column(String(100), nullable=True)
    llm_model = Column(String(255), nullable=True)
    processing_time_ms = Column(Float, nullable=True)

    # Metadata (speakers, duration, etc.)
    insight_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)

    # Relationships
    session = relationship("BotSession")
    template = relationship("InsightPromptTemplate")

    # Indexes
    __table_args__ = (
        Index("idx_meeting_insights_session_id", "session_id"),
        Index("idx_meeting_insights_insight_type", "insight_type"),
        Index("idx_meeting_insights_created_at", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "insight_id": str(self.insight_id),
            "session_id": str(self.session_id),
            "template_id": str(self.template_id) if self.template_id else None,
            "insight_type": self.insight_type,
            "title": self.title,
            "content": self.content,
            "prompt_used": self.prompt_used,
            "transcript_length": self.transcript_length,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.insight_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class InsightPromptTemplate(Base):
    """Configurable prompt templates for insight generation."""

    __tablename__ = "insight_prompt_templates"

    template_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Template identification
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    category = Column(
        String(100), nullable=False, default="custom"
    )  # summary, analysis, action_items, custom

    # Prompt content
    prompt_template = Column(Text, nullable=False)  # With {transcript}, {speakers}, etc.
    system_prompt = Column(Text, nullable=True)  # Optional system message override
    expected_output_format = Column(
        String(50), nullable=False, default="markdown"
    )  # text, markdown, json, bullets

    # LLM defaults
    default_llm_backend = Column(String(100), nullable=True)  # Uses session default if null
    default_temperature = Column(Float, nullable=False, default=0.3)
    default_max_tokens = Column(Integer, nullable=False, default=1024)

    # Status
    is_builtin = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)

    # Metadata
    template_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)

    # Indexes
    __table_args__ = (
        Index("idx_insight_templates_name", "name"),
        Index("idx_insight_templates_category", "category"),
        Index("idx_insight_templates_is_active", "is_active"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "template_id": str(self.template_id),
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "prompt_template": self.prompt_template,
            "system_prompt": self.system_prompt,
            "expected_output_format": self.expected_output_format,
            "default_llm_backend": self.default_llm_backend,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "is_builtin": self.is_builtin,
            "is_active": self.is_active,
            "metadata": self.template_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentConversation(Base):
    """Chat sessions about a meeting."""

    __tablename__ = "agent_conversations"

    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("bot_sessions.session_id"), nullable=False)

    # Conversation details
    title = Column(String(500), nullable=True)
    status = Column(String(50), nullable=False, default="active")  # active, closed

    # System context (system prompt with transcript context)
    system_context = Column(Text, nullable=True)

    # Metadata
    conversation_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)

    # Relationships
    session = relationship("BotSession")
    messages = relationship(
        "AgentMessage", back_populates="conversation", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_agent_conversations_session_id", "session_id"),
        Index("idx_agent_conversations_status", "status"),
        Index("idx_agent_conversations_created_at", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "conversation_id": str(self.conversation_id),
            "session_id": str(self.session_id),
            "title": self.title,
            "status": self.status,
            "system_context": self.system_context,
            "metadata": self.conversation_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentMessage(Base):
    """Individual messages in agent chat."""

    __tablename__ = "agent_messages"

    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_conversations.conversation_id"),
        nullable=False,
    )

    # Message content
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)

    # LLM metadata (null for user messages)
    llm_backend = Column(String(100), nullable=True)
    llm_model = Column(String(255), nullable=True)
    processing_time_ms = Column(Float, nullable=True)

    # Suggestions (null for user messages)
    suggested_queries = Column(JSON, nullable=True)

    # Metadata
    message_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=utc_now)

    # Relationships
    conversation = relationship("AgentConversation", back_populates="messages")

    # Indexes
    __table_args__ = (
        Index("idx_agent_messages_conversation_id", "conversation_id"),
        Index("idx_agent_messages_role", "role"),
        Index("idx_agent_messages_created_at", "created_at"),
    )
