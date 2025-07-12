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


# Database utility functions
class DatabaseManager:
    """Database management utilities"""

    def __init__(self, session: Session):
        self.session = session

    async def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new bot session"""
        bot_session = BotSession(
            bot_id=session_data["bot_id"],
            meeting_id=session_data["meeting_id"],
            meeting_title=session_data.get("meeting_title"),
            meeting_uri=session_data.get("meeting_uri"),
            bot_type=session_data.get("bot_type", "google_meet"),
            target_languages=session_data.get("target_languages", ["en"]),
            enable_translation=session_data.get("enable_translation", True),
            enable_transcription=session_data.get("enable_transcription", True),
            enable_virtual_webcam=session_data.get("enable_virtual_webcam", False),
            audio_storage_enabled=session_data.get("audio_storage_enabled", True),
            session_metadata=session_data.get("metadata", {}),
        )

        self.session.add(bot_session)
        await self.session.commit()

        return str(bot_session.session_id)

    async def store_audio_file(self, audio_data: Dict[str, Any]) -> str:
        """Store audio file information"""
        audio_file = AudioFile(
            session_id=audio_data["session_id"],
            filename=audio_data["filename"],
            file_path=audio_data["file_path"],
            file_size=audio_data["file_size"],
            file_hash=audio_data["file_hash"],
            mime_type=audio_data["mime_type"],
            duration=audio_data.get("duration"),
            sample_rate=audio_data.get("sample_rate"),
            channels=audio_data.get("channels"),
            bit_depth=audio_data.get("bit_depth"),
            start_time=audio_data.get("start_time"),
            end_time=audio_data.get("end_time"),
            metadata=audio_data.get("metadata", {}),
        )

        self.session.add(audio_file)
        await self.session.commit()

        return str(audio_file.file_id)

    async def store_transcript(self, transcript_data: Dict[str, Any]) -> str:
        """Store transcript information"""
        transcript = Transcript(
            session_id=transcript_data["session_id"],
            text=transcript_data["text"],
            language=transcript_data["language"],
            confidence=transcript_data.get("confidence"),
            source=transcript_data["source"],
            audio_file_id=transcript_data.get("audio_file_id"),
            speaker_id=transcript_data.get("speaker_id"),
            speaker_name=transcript_data.get("speaker_name"),
            start_time=transcript_data["start_time"],
            end_time=transcript_data["end_time"],
            start_offset=transcript_data.get("start_offset"),
            end_offset=transcript_data.get("end_offset"),
            metadata=transcript_data.get("metadata", {}),
        )

        self.session.add(transcript)
        await self.session.commit()

        return str(transcript.transcript_id)

    async def store_translation(self, translation_data: Dict[str, Any]) -> str:
        """Store translation information"""
        translation = Translation(
            session_id=translation_data["session_id"],
            transcript_id=translation_data.get("transcript_id"),
            original_text=translation_data["original_text"],
            translated_text=translation_data["translated_text"],
            source_language=translation_data["source_language"],
            target_language=translation_data["target_language"],
            confidence=translation_data.get("confidence"),
            speaker_id=translation_data.get("speaker_id"),
            speaker_name=translation_data.get("speaker_name"),
            start_time=translation_data["start_time"],
            end_time=translation_data["end_time"],
            quality_score=translation_data.get("quality_score"),
            word_count=translation_data.get("word_count"),
            character_count=translation_data.get("character_count"),
            metadata=translation_data.get("metadata", {}),
        )

        self.session.add(translation)
        await self.session.commit()

        return str(translation.translation_id)

    async def store_correlation(self, correlation_data: Dict[str, Any]) -> str:
        """Store correlation information"""
        correlation = Correlation(
            session_id=correlation_data["session_id"],
            external_source=correlation_data["external_source"],
            external_text=correlation_data["external_text"],
            external_timestamp=correlation_data["external_timestamp"],
            internal_source=correlation_data["internal_source"],
            internal_text=correlation_data["internal_text"],
            internal_timestamp=correlation_data["internal_timestamp"],
            similarity_score=correlation_data.get("similarity_score"),
            time_offset=correlation_data.get("time_offset"),
            confidence=correlation_data.get("confidence"),
            metadata=correlation_data.get("metadata", {}),
        )

        self.session.add(correlation)
        await self.session.commit()

        return str(correlation.correlation_id)

    async def get_session_comprehensive_data(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive session data"""
        bot_session = (
            self.session.query(BotSession)
            .filter(BotSession.session_id == session_id)
            .first()
        )

        if not bot_session:
            return None

        # Get related data
        audio_files = (
            self.session.query(AudioFile)
            .filter(AudioFile.session_id == session_id)
            .all()
        )

        transcripts = (
            self.session.query(Transcript)
            .filter(Transcript.session_id == session_id)
            .all()
        )

        translations = (
            self.session.query(Translation)
            .filter(Translation.session_id == session_id)
            .all()
        )

        correlations = (
            self.session.query(Correlation)
            .filter(Correlation.session_id == session_id)
            .all()
        )

        participants = (
            self.session.query(Participant)
            .filter(Participant.session_id == session_id)
            .all()
        )

        # Calculate statistics
        statistics = self._calculate_session_statistics(
            bot_session,
            audio_files,
            transcripts,
            translations,
            correlations,
            participants,
        )

        return {
            "session": bot_session.to_dict(),
            "audio_files": [af.to_dict() for af in audio_files],
            "transcripts": [t.to_dict() for t in transcripts],
            "translations": [t.to_dict() for t in translations],
            "correlations": [c.to_dict() for c in correlations],
            "participants": [p.to_dict() for p in participants],
            "statistics": statistics,
        }

    def _calculate_session_statistics(
        self,
        session,
        audio_files,
        transcripts,
        translations,
        correlations,
        participants,
    ):
        """Calculate session statistics"""
        duration = None
        if session.started_at and session.ended_at:
            duration = (session.ended_at - session.started_at).total_seconds()

        total_audio_duration = sum(af.duration for af in audio_files if af.duration)
        total_audio_size = sum(af.file_size for af in audio_files)

        total_words = sum(t.word_count for t in translations if t.word_count)
        total_characters = sum(
            t.character_count for t in translations if t.character_count
        )

        average_confidence = None
        if transcripts:
            confidences = [t.confidence for t in transcripts if t.confidence]
            if confidences:
                average_confidence = sum(confidences) / len(confidences)

        languages_detected = list(set(t.language for t in transcripts))

        return {
            "duration": duration,
            "total_participants": len(participants),
            "total_audio_files": len(audio_files),
            "total_transcripts": len(transcripts),
            "total_translations": len(translations),
            "total_correlations": len(correlations),
            "total_audio_duration": total_audio_duration,
            "total_audio_size": total_audio_size,
            "total_words": total_words,
            "total_characters": total_characters,
            "average_confidence": average_confidence,
            "languages_detected": languages_detected,
            "unique_speakers": len(
                set(p.speaker_id for p in participants if p.speaker_id)
            ),
        }
