#!/usr/bin/env python3
"""
Fireflies Models Tests - ACTUAL TESTS (No Mocks)

Tests the real model behavior, validation, and computed properties.
"""

import sys
from pathlib import Path
import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))

from models.fireflies import (
    FIREFLIES_SOURCE_TYPE,
    FirefliesEventType,
    FirefliesConnectionStatus,
    MeetingState,
    FirefliesChunk,
    FirefliesMeeting,
    FirefliesSessionConfig,
    FirefliesSession,
    SpeakerBuffer,
    TranslationUnit,
    TranslationContext,
    TranslationResult,
    CaptionEntry,
    CaptionBroadcast,
    FirefliesConnectRequest,
    FirefliesConnectResponse,
    ActiveMeetingsResponse,
)


class TestFirefliesChunk:
    """Test FirefliesChunk - the core transcript unit."""

    def test_chunk_creation_and_properties(self):
        """Test creating chunk and computed properties."""
        chunk = FirefliesChunk(
            transcript_id="abc123",
            chunk_id="chunk_001",
            text="Hello world how are you",
            speaker_name="Alice",
            start_time=1.0,
            end_time=3.5,
        )

        assert chunk.transcript_id == "abc123"
        assert chunk.chunk_id == "chunk_001"
        assert chunk.text == "Hello world how are you"
        assert chunk.speaker_name == "Alice"
        assert chunk.duration_ms == 2500.0  # (3.5 - 1.0) * 1000
        assert chunk.word_count == 5

    def test_chunk_serialization(self):
        """Test chunk JSON serialization round-trip."""
        chunk = FirefliesChunk(
            transcript_id="abc123",
            chunk_id="chunk_001",
            text="Test",
            speaker_name="Bob",
            start_time=0.0,
            end_time=1.0,
        )

        # To dict and back
        data = chunk.model_dump()
        restored = FirefliesChunk(**data)
        assert restored.chunk_id == chunk.chunk_id

        # To JSON and back
        json_str = chunk.model_dump_json()
        restored2 = FirefliesChunk.model_validate_json(json_str)
        assert restored2.text == chunk.text


class TestSpeakerBuffer:
    """Test SpeakerBuffer - accumulating chunks for sentence building."""

    def test_buffer_accumulation(self):
        """Test adding chunks and getting combined text."""
        buffer = SpeakerBuffer(speaker_name="Alice")

        # Add chunks
        buffer.add(
            FirefliesChunk(
                transcript_id="t1",
                chunk_id="c1",
                text="Hello",
                speaker_name="Alice",
                start_time=0.0,
                end_time=0.5,
            )
        )
        buffer.add(
            FirefliesChunk(
                transcript_id="t1",
                chunk_id="c2",
                text="world",
                speaker_name="Alice",
                start_time=0.5,
                end_time=1.0,
            )
        )
        buffer.add(
            FirefliesChunk(
                transcript_id="t1",
                chunk_id="c3",
                text="how are you",
                speaker_name="Alice",
                start_time=1.0,
                end_time=2.0,
            )
        )

        assert buffer.get_text() == "Hello world how are you"
        assert buffer.word_count == 5
        assert buffer.duration_seconds == 2.0
        assert buffer.buffer_start_time == 0.0
        assert buffer.get_end_time() == 2.0
        assert len(buffer.chunks) == 3

    def test_buffer_clear_and_reset(self):
        """Test clearing buffer and resetting with remainder."""
        buffer = SpeakerBuffer(speaker_name="Alice")

        buffer.add(
            FirefliesChunk(
                transcript_id="t1",
                chunk_id="c1",
                text="Hello world. How are you",
                speaker_name="Alice",
                start_time=0.0,
                end_time=3.0,
            )
        )

        # Clear
        buffer.clear()
        assert len(buffer.chunks) == 0
        assert buffer.word_count == 0

        # Add again and reset to remainder
        buffer.add(
            FirefliesChunk(
                transcript_id="t1",
                chunk_id="c2",
                text="First sentence. Second part",
                speaker_name="Alice",
                start_time=0.0,
                end_time=3.0,
            )
        )

        buffer.reset_to("Second part", 1.5, 3.0, "t1")
        assert buffer.get_text() == "Second part"
        assert buffer.word_count == 2


class TestTranslationUnit:
    """Test TranslationUnit - sentence ready for translation."""

    def test_unit_creation(self):
        """Test creating translation unit."""
        unit = TranslationUnit(
            text="Hello, how are you today?",
            speaker_name="Alice",
            start_time=0.0,
            end_time=2.5,
            session_id="session-001",
            transcript_id="abc123",
            chunk_ids=["c1", "c2", "c3"],
            boundary_type="punctuation",
        )

        assert unit.word_count == 5
        assert unit.duration_ms == 2500.0
        assert unit.boundary_type == "punctuation"


class TestTranslationContext:
    """Test TranslationContext - context for translation."""

    def test_context_formatting(self):
        """Test context window and glossary formatting."""
        context = TranslationContext(
            previous_sentences=["First sentence.", "Second sentence."],
            glossary={"API": "API", "backend": "servidor"},
            target_language="es",
        )

        window = context.format_context_window()
        assert "First sentence." in window
        assert "Second sentence." in window

        glossary = context.format_glossary()
        assert "API -> API" in glossary
        assert "backend -> servidor" in glossary

    def test_empty_context(self):
        """Test empty context formatting."""
        context = TranslationContext(target_language="es")

        assert context.format_context_window() == "(No previous context)"
        assert context.format_glossary() == "(No glossary terms)"


class TestTranslationResult:
    """Test TranslationResult - translation output."""

    def test_result_validation(self):
        """Test result creation and confidence bounds."""
        result = TranslationResult(
            original="Hello",
            translated="Hola",
            speaker_name="Alice",
            source_language="en",
            target_language="es",
            confidence=0.95,
        )

        assert result.confidence == 0.95

        # Test bounds
        with pytest.raises(ValueError):
            TranslationResult(
                original="Hi",
                translated="Hola",
                speaker_name="A",
                source_language="en",
                target_language="es",
                confidence=1.5,
            )


class TestFirefliesSessionConfig:
    """Test FirefliesSessionConfig - session configuration."""

    def test_config_defaults(self):
        """Test config with defaults."""
        config = FirefliesSessionConfig(
            api_key="test-key",
            transcript_id="abc123",
        )

        assert config.target_languages == ["es"]
        assert config.pause_threshold_ms == 800.0
        assert config.max_buffer_words == 30
        assert config.context_window_size == 3

    def test_config_validation(self):
        """Test config validation."""
        # Valid range
        config = FirefliesSessionConfig(
            api_key="key", transcript_id="t1", context_window_size=5
        )
        assert config.context_window_size == 5

        # Invalid range
        with pytest.raises(ValueError):
            FirefliesSessionConfig(
                api_key="key", transcript_id="t1", context_window_size=15
            )


class TestFirefliesSession:
    """Test FirefliesSession - session state tracking."""

    def test_session_state(self):
        """Test session state management."""
        session = FirefliesSession(
            session_id="s1",
            fireflies_transcript_id="t1",
        )

        assert session.connection_status == FirefliesConnectionStatus.DISCONNECTED
        assert session.chunks_received == 0
        assert session.speakers_detected == []

        # Update state
        session.connection_status = FirefliesConnectionStatus.CONNECTED
        session.chunks_received = 10
        session.speakers_detected = ["Alice", "Bob"]

        assert session.connection_status == FirefliesConnectionStatus.CONNECTED
        assert session.chunks_received == 10
        assert len(session.speakers_detected) == 2


class TestCaptionModels:
    """Test caption-related models."""

    def test_caption_entry(self):
        """Test caption entry creation."""
        caption = CaptionEntry(
            id="c1",
            original_text="Hello",
            translated_text="Hola",
            speaker_name="Alice",
            speaker_color="#FF0000",
            target_language="es",
        )

        assert caption.duration_seconds == 8.0  # Default
        assert caption.confidence == 1.0  # Default

    def test_caption_broadcast(self):
        """Test caption broadcast creation."""
        captions = [
            CaptionEntry(
                id="c1", translated_text="Hola", speaker_name="A", target_language="es"
            ),
            CaptionEntry(
                id="c2", translated_text="Mundo", speaker_name="B", target_language="es"
            ),
        ]

        broadcast = CaptionBroadcast(session_id="s1", captions=captions)
        assert len(broadcast.captions) == 2


class TestEnumsAndConstants:
    """Test enums and constants."""

    def test_source_type(self):
        """Test FIREFLIES_SOURCE_TYPE constant."""
        assert FIREFLIES_SOURCE_TYPE == "fireflies"

    def test_event_types(self):
        """Test event type enum values."""
        assert FirefliesEventType.AUTH_SUCCESS.value == "auth.success"
        assert (
            FirefliesEventType.TRANSCRIPTION_BROADCAST.value
            == "transcription.broadcast"
        )

    def test_connection_status(self):
        """Test connection status enum."""
        assert FirefliesConnectionStatus.CONNECTED.value == "connected"
        assert FirefliesConnectionStatus.DISCONNECTED.value == "disconnected"

    def test_meeting_state(self):
        """Test meeting state enum."""
        assert MeetingState.ACTIVE.value == "active"
        assert MeetingState.PAUSED.value == "paused"


class TestAPIModels:
    """Test API request/response models."""

    def test_connect_request(self):
        """Test connect request model."""
        request = FirefliesConnectRequest(
            api_key="key",
            transcript_id="t1",
            target_languages=["es", "fr"],
        )
        assert request.target_languages == ["es", "fr"]

    def test_connect_response(self):
        """Test connect response model."""
        response = FirefliesConnectResponse(
            success=True,
            message="Connected",
            session_id="s1",
            connection_status=FirefliesConnectionStatus.CONNECTED,
            transcript_id="t1",
        )
        assert response.success is True

    def test_active_meetings_response(self):
        """Test active meetings response."""
        meetings = [FirefliesMeeting(id="m1", title="Test")]
        response = ActiveMeetingsResponse(success=True, meetings=meetings, count=1)
        assert response.count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
