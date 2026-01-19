"""
Fireflies.ai Integration Models

Data models for Fireflies realtime transcription API integration.
Handles WebSocket events, transcript chunks, and session management.

Integrates with existing bot_sessions database schema:
- sessions table: Main session record (source_type='fireflies')
- transcripts table: Stores Fireflies chunks (source_type='fireflies')
- translations table: Stores translated output
- speaker_identities: Maps Fireflies speaker names

Reference: https://docs.fireflies.ai/realtime-api/event-schema
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field

from .base import BaseModel, ResponseMixin, TimestampMixin

# =============================================================================
# Constants for Database Integration
# =============================================================================

# Source type for Fireflies in transcripts table
FIREFLIES_SOURCE_TYPE = "fireflies"

# Valid source types for transcripts table
TRANSCRIPT_SOURCE_TYPES = ["google_meet", "whisper_service", "manual", "fireflies"]


# =============================================================================
# Enums
# =============================================================================


class FirefliesEventType(str, Enum):
    """Fireflies WebSocket event types"""

    # Authentication events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILED = "auth.failed"

    # Connection events
    CONNECTION_ESTABLISHED = "connection.established"
    CONNECTION_ERROR = "connection.error"

    # Transcription events
    TRANSCRIPTION_BROADCAST = "transcription.broadcast"


class FirefliesConnectionStatus(str, Enum):
    """Fireflies connection status"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class MeetingState(str, Enum):
    """Fireflies meeting state"""

    ACTIVE = "active"
    PAUSED = "paused"


# =============================================================================
# Fireflies API Models
# =============================================================================


class FirefliesChunk(BaseModel):
    """
    Incoming transcript chunk from Fireflies realtime API.

    This is the primary data unit received via WebSocket.
    chunk_id is used for deduplication (same chunk_id = update to previous).
    """

    transcript_id: str = Field(description="Unique identifier for the transcript/meeting")
    chunk_id: str = Field(
        description="Unique segment identifier; same chunk_id = update to previous"
    )
    text: str = Field(description="Transcribed text content for the segment")
    speaker_name: str = Field(description="Speaker identification for the segment")
    start_time: float = Field(description="Segment start position in seconds")
    end_time: float = Field(description="Segment end position in seconds")

    @property
    def duration_ms(self) -> float:
        """Duration of this chunk in milliseconds"""
        return (self.end_time - self.start_time) * 1000

    @property
    def word_count(self) -> int:
        """Approximate word count in this chunk"""
        return len(self.text.split())

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transcript_id": "abc123",
                "chunk_id": "chunk_001",
                "text": "Hello, this is a test transcription.",
                "speaker_name": "Alice",
                "start_time": 0.0,
                "end_time": 2.5,
            }
        }
    )


class FirefliesEvent(BaseModel):
    """Generic Fireflies WebSocket event wrapper"""

    event_type: FirefliesEventType = Field(description="Type of event received")
    data: dict[str, Any] | None = Field(default=None, description="Event payload data")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Event timestamp"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event_type": "transcription.broadcast",
                "data": {
                    "transcript_id": "abc123",
                    "chunk_id": "chunk_001",
                    "text": "Hello world",
                    "speaker_name": "Alice",
                    "start_time": 0.0,
                    "end_time": 1.25,
                },
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    )


class FirefliesMeeting(BaseModel):
    """Active meeting from Fireflies GraphQL API"""

    id: str = Field(description="Meeting ID (used for WebSocket connection)")
    title: str | None = Field(default=None, description="Meeting title")
    organizer_email: str | None = Field(default=None, description="Organizer email address")
    meeting_link: str | None = Field(default=None, description="Original meeting link")
    start_time: datetime | None = Field(default=None, description="Meeting start time")
    end_time: datetime | None = Field(default=None, description="Meeting end time")
    privacy: str | None = Field(default=None, description="Privacy setting")
    state: MeetingState = Field(default=MeetingState.ACTIVE, description="Meeting state")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "meeting-id-12345",
                "title": "Team Standup",
                "organizer_email": "user@example.com",
                "meeting_link": "https://zoom.us/j/123456789",
                "start_time": "2024-01-15T10:00:00Z",
                "state": "active",
            }
        }
    )


# =============================================================================
# Session Management Models
# =============================================================================


class FirefliesSessionConfig(BaseModel):
    """Configuration for a Fireflies session"""

    # Fireflies API
    api_key: str = Field(description="Fireflies API key")
    transcript_id: str = Field(description="Transcript ID to connect to (from active_meetings)")

    # Target languages for translation
    target_languages: list[str] = Field(
        default_factory=lambda: ["es"], description="Target languages for translation"
    )

    # Translation model/service to use
    translation_model: str | None = Field(
        default=None,
        description="Translation model/service to use (ollama, groq, etc.)",
    )

    # Sentence aggregation settings
    pause_threshold_ms: float = Field(
        default=800.0,
        description="Pause duration (ms) that indicates sentence boundary",
    )
    max_buffer_words: int = Field(
        default=30, description="Maximum words to buffer before forcing translation"
    )
    max_buffer_seconds: float = Field(
        default=5.0, description="Maximum seconds to buffer before forcing translation"
    )
    min_words_for_translation: int = Field(
        default=3, description="Minimum words required before translating"
    )
    use_nlp_boundary_detection: bool = Field(
        default=True, description="Use spaCy for sentence boundary detection"
    )

    # Translation context
    context_window_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of previous sentences for translation context",
    )
    include_cross_speaker_context: bool = Field(
        default=True, description="Include other speakers' sentences in context"
    )

    # Glossary
    glossary_id: str | None = Field(default=None, description="Glossary ID to use for this session")
    domain: str | None = Field(
        default=None,
        description="Domain for glossary filtering (e.g., 'medical', 'legal')",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_key": "ff-api-key-xxxxx",
                "transcript_id": "abc123",
                "target_languages": ["es", "fr", "de"],
                "pause_threshold_ms": 800.0,
                "max_buffer_words": 30,
                "context_window_size": 3,
                "glossary_id": "glossary-tech-001",
                "domain": "technology",
            }
        }
    )


class FirefliesSession(TimestampMixin):
    """
    Active Fireflies session state.

    Tracks connection status, chunk processing, and statistics.
    """

    session_id: str = Field(description="Internal session ID")
    fireflies_transcript_id: str = Field(description="Fireflies transcript ID")
    fireflies_meeting_id: str | None = Field(default=None, description="Fireflies meeting ID")

    # Connection state
    connection_status: FirefliesConnectionStatus = Field(
        default=FirefliesConnectionStatus.DISCONNECTED,
        description="Current connection status",
    )
    connected_at: datetime | None = Field(default=None, description="Connection timestamp")
    last_chunk_time: datetime | None = Field(
        default=None, description="Last chunk received timestamp"
    )
    last_chunk_id: str | None = Field(default=None, description="Last processed chunk ID")

    # Configuration
    config: FirefliesSessionConfig | None = Field(default=None, description="Session configuration")

    # Statistics
    chunks_received: int = Field(default=0, description="Total chunks received")
    sentences_produced: int = Field(default=0, description="Complete sentences produced")
    translations_completed: int = Field(default=0, description="Translations completed")
    speakers_detected: list[str] = Field(
        default_factory=list, description="Unique speakers detected"
    )

    # Error tracking
    error_count: int = Field(default=0, description="Error count")
    last_error: str | None = Field(default=None, description="Last error message")
    reconnection_attempts: int = Field(default=0, description="Reconnection attempts")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "session-uuid-12345",
                "fireflies_transcript_id": "abc123",
                "connection_status": "connected",
                "connected_at": "2024-01-15T10:00:00Z",
                "chunks_received": 150,
                "sentences_produced": 45,
                "translations_completed": 135,
                "speakers_detected": ["Alice", "Bob", "Charlie"],
            }
        }
    )


# =============================================================================
# Sentence Aggregation Models
# =============================================================================


class SpeakerBuffer(BaseModel):
    """
    Buffer for accumulating chunks from a single speaker.

    Used by SentenceAggregator to build complete sentences.
    """

    speaker_name: str = Field(description="Speaker this buffer belongs to")
    chunks: list[FirefliesChunk] = Field(default_factory=list, description="Accumulated chunks")
    buffer_start_time: float | None = Field(
        default=None, description="Start time of first chunk in buffer"
    )
    word_count: int = Field(default=0, description="Total words in buffer")

    def add(self, chunk: FirefliesChunk) -> None:
        """Add a chunk to the buffer"""
        if not self.chunks:
            self.buffer_start_time = chunk.start_time
        self.chunks.append(chunk)
        self.word_count += chunk.word_count

    def get_text(self) -> str:
        """Get concatenated text from all chunks"""
        return " ".join(c.text for c in self.chunks)

    def get_end_time(self) -> float | None:
        """Get end time of last chunk"""
        return self.chunks[-1].end_time if self.chunks else None

    def clear(self) -> None:
        """Clear the buffer"""
        self.chunks = []
        self.buffer_start_time = None
        self.word_count = 0

    def reset_to(
        self, text: str, start_time: float, end_time: float, transcript_id: str = ""
    ) -> None:
        """Reset buffer with remaining text after extraction"""
        # Save transcript_id before clearing
        saved_transcript_id = self.chunks[0].transcript_id if self.chunks else transcript_id
        self.chunks = []
        if text.strip():
            # Create a synthetic chunk for the remainder
            remainder_chunk = FirefliesChunk(
                transcript_id=saved_transcript_id,
                chunk_id=f"remainder_{datetime.now(UTC).timestamp()}",
                text=text.strip(),
                speaker_name=self.speaker_name,
                start_time=start_time,
                end_time=end_time,
            )
            self.chunks = [remainder_chunk]
            self.buffer_start_time = start_time
            self.word_count = remainder_chunk.word_count
        else:
            self.buffer_start_time = None
            self.word_count = 0

    @property
    def last_chunk(self) -> FirefliesChunk | None:
        """Get the last chunk in the buffer"""
        return self.chunks[-1] if self.chunks else None

    @property
    def duration_seconds(self) -> float:
        """Get total buffer duration in seconds"""
        if not self.chunks or self.buffer_start_time is None:
            return 0.0
        end_time = self.get_end_time() or self.buffer_start_time
        return end_time - self.buffer_start_time


class TranslationUnit(BaseModel):
    """
    A complete sentence ready for translation.

    Produced by SentenceAggregator, consumed by RollingWindowTranslator.
    """

    text: str = Field(description="Complete sentence text")
    speaker_name: str = Field(description="Speaker who said this sentence")
    start_time: float = Field(description="Sentence start time in seconds")
    end_time: float = Field(description="Sentence end time in seconds")
    session_id: str = Field(description="Session this sentence belongs to")
    transcript_id: str = Field(description="Fireflies transcript ID")
    chunk_ids: list[str] = Field(
        default_factory=list, description="Source chunk IDs that formed this sentence"
    )
    boundary_type: str = Field(
        default="unknown",
        description="How boundary was detected: punctuation, pause, speaker_change, nlp, forced",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When this unit was created"
    )

    @property
    def word_count(self) -> int:
        """Word count of the sentence"""
        return len(self.text.split())

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds"""
        return (self.end_time - self.start_time) * 1000

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "I think we should move forward with the implementation.",
                "speaker_name": "Alice",
                "start_time": 10.5,
                "end_time": 14.2,
                "session_id": "session-uuid-12345",
                "transcript_id": "abc123",
                "chunk_ids": ["chunk_010", "chunk_011", "chunk_012"],
                "boundary_type": "punctuation",
            }
        }
    )


# =============================================================================
# Translation Models
# =============================================================================


class TranslationContext(BaseModel):
    """Context for translation including rolling window and glossary"""

    previous_sentences: list[str] = Field(
        default_factory=list,
        description="Previous sentences for context (rolling window)",
    )
    glossary: dict[str, str] = Field(
        default_factory=dict, description="Glossary terms: source -> target"
    )
    target_language: str = Field(description="Target language code")
    source_language: str = Field(default="en", description="Source language code")

    def format_context_window(self) -> str:
        """Format previous sentences for prompt injection"""
        if not self.previous_sentences:
            return "(No previous context)"
        return "\n".join(self.previous_sentences)

    def format_glossary(self) -> str:
        """Format glossary for prompt injection"""
        if not self.glossary:
            return "(No glossary terms)"
        return "\n".join(f"- {src} -> {tgt}" for src, tgt in self.glossary.items())


class TranslationResult(TimestampMixin):
    """Result of translating a sentence"""

    original: str = Field(description="Original sentence")
    translated: str = Field(description="Translated sentence")
    speaker_name: str = Field(description="Speaker who said this")
    source_language: str = Field(description="Source language code")
    target_language: str = Field(description="Target language code")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Translation confidence score"
    )
    context_sentences_used: int = Field(default=0, description="Number of context sentences used")
    glossary_terms_applied: list[str] = Field(
        default_factory=list, description="Glossary terms that were applied"
    )

    # Timing
    translation_time_ms: float = Field(
        default=0.0, description="Time taken to translate in milliseconds"
    )

    # Source reference
    session_id: str | None = Field(default=None, description="Session ID")
    translation_unit_id: str | None = Field(default=None, description="Source translation unit ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "original": "I think we should move forward with the implementation.",
                "translated": "Creo que deberíamos seguir adelante con la implementación.",
                "speaker_name": "Alice",
                "source_language": "en",
                "target_language": "es",
                "confidence": 0.95,
                "context_sentences_used": 3,
                "glossary_terms_applied": ["implementation"],
                "translation_time_ms": 125.5,
            }
        }
    )


# =============================================================================
# Caption Output Models
# =============================================================================


class CaptionEntry(BaseModel):
    """A caption entry for display"""

    id: str = Field(description="Unique caption ID")
    original_text: str | None = Field(default=None, description="Original text (if showing both)")
    translated_text: str = Field(description="Translated text to display")
    speaker_name: str = Field(description="Speaker name")
    speaker_color: str | None = Field(default=None, description="Color for speaker (hex)")
    target_language: str = Field(description="Target language code")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Caption timestamp"
    )
    duration_seconds: float = Field(default=8.0, description="How long to display this caption")
    confidence: float = Field(default=1.0, description="Translation confidence")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "caption-uuid-12345",
                "original_text": "Hello, how are you?",
                "translated_text": "Hola, ¿cómo estás?",
                "speaker_name": "Alice",
                "speaker_color": "#4CAF50",
                "target_language": "es",
                "duration_seconds": 8.0,
                "confidence": 0.95,
            }
        }
    )


class CaptionBroadcast(BaseModel):
    """Caption broadcast message for WebSocket clients"""

    session_id: str = Field(description="Session ID")
    captions: list[CaptionEntry] = Field(description="Current captions to display")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Broadcast timestamp"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "session-uuid-12345",
                "captions": [
                    {
                        "id": "caption-001",
                        "translated_text": "Hola, ¿cómo estás?",
                        "speaker_name": "Alice",
                        "target_language": "es",
                    }
                ],
            }
        }
    )


# =============================================================================
# API Request/Response Models
# =============================================================================


class FirefliesConnectRequest(BaseModel):
    """Request to connect to Fireflies realtime API"""

    api_key: str = Field(description="Fireflies API key")
    transcript_id: str = Field(description="Transcript ID to connect to")
    target_languages: list[str] = Field(
        default_factory=lambda: ["es"], description="Target languages for translation"
    )
    glossary_id: str | None = Field(default=None, description="Glossary to use")
    domain: str | None = Field(default=None, description="Domain for glossary filtering")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_key": "ff-api-key-xxxxx",
                "transcript_id": "abc123",
                "target_languages": ["es", "fr"],
                "glossary_id": "glossary-tech-001",
            }
        }
    )


class FirefliesConnectResponse(ResponseMixin):
    """Response after connecting to Fireflies"""

    session_id: str = Field(description="Created session ID")
    connection_status: FirefliesConnectionStatus = Field(description="Connection status")
    transcript_id: str = Field(description="Connected transcript ID")
    meeting_info: FirefliesMeeting | None = Field(
        default=None, description="Meeting information if available"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Connected to Fireflies realtime API",
                "session_id": "session-uuid-12345",
                "connection_status": "connected",
                "transcript_id": "abc123",
            }
        }
    )


class FirefliesStatusResponse(ResponseMixin):
    """Status of a Fireflies session"""

    session: FirefliesSession | None = Field(default=None, description="Session details")
    is_connected: bool = Field(description="Whether currently connected")
    uptime_seconds: float | None = Field(default=None, description="Connection uptime")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "is_connected": True,
                "uptime_seconds": 3600.5,
                "session": {
                    "session_id": "session-uuid-12345",
                    "connection_status": "connected",
                    "chunks_received": 150,
                },
            }
        }
    )


class ActiveMeetingsResponse(ResponseMixin):
    """Response with active meetings from Fireflies"""

    meetings: list[FirefliesMeeting] = Field(
        default_factory=list, description="List of active meetings"
    )
    count: int = Field(default=0, description="Number of meetings")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "count": 2,
                "meetings": [
                    {
                        "id": "meeting-001",
                        "title": "Team Standup",
                        "state": "active",
                    },
                    {
                        "id": "meeting-002",
                        "title": "Planning Session",
                        "state": "active",
                    },
                ],
            }
        }
    )
