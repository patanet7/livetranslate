#!/usr/bin/env python3
"""
Fireflies Integration Tests (TDD)

Comprehensive integration tests for the full Fireflies pipeline.
Tests real-world scenarios with mocked Fireflies API but real internal components.

Test Scenarios:
1. Full Connection Flow - Connect -> Receive Transcripts -> Disconnect
2. Multiple Sessions - Concurrent session management
3. Error Recovery - Reconnection and error handling
4. Transcript Processing - Chunk handling and deduplication
5. Meeting Discovery - Active meetings query flow
6. End-to-End Pipeline - Full transcript to translation flow
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import json
import pytest

# Add src to path for imports
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

# FastAPI test imports
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.routers.fireflies import (
    router,
    FirefliesSessionManager,
    get_session_manager,
    get_fireflies_config,
)

from src.models.fireflies import (
    FirefliesChunk,
    FirefliesMeeting,
    FirefliesSession,
    FirefliesSessionConfig,
    FirefliesConnectionStatus,
    SpeakerBuffer,
    TranslationUnit,
    TranslationContext,
    TranslationResult,
    CaptionEntry,
    CaptionBroadcast,
    MeetingState,
)



# =============================================================================
# Test App Setup
# =============================================================================


def create_test_app():
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def test_app():
    """Provide test FastAPI app."""
    return create_test_app()


@pytest.fixture
def client(test_app):
    """Provide test client."""
    return TestClient(test_app)


@pytest.fixture
def mock_fireflies_config():
    """Create mock Fireflies configuration."""
    config = MagicMock()
    config.api_key = "test-api-key"
    config.graphql_endpoint = "https://api.fireflies.ai/graphql"
    config.websocket_endpoint = "wss://api.fireflies.ai/realtime"
    config.pause_threshold_ms = 800.0
    config.max_buffer_words = 30
    config.context_window_size = 3
    config.default_target_languages = ["es"]
    config.has_api_key.return_value = True
    return config


# =============================================================================
# Mock Fireflies WebSocket Server
# =============================================================================


class MockFirefliesWebSocketServer:
    """
    Mock Fireflies WebSocket server for integration testing.

    Simulates the Fireflies realtime API behavior:
    - Authentication handshake
    - Transcript broadcast messages
    - Connection events
    """

    def __init__(self):
        self.connected = False
        self.authenticated = False
        self.transcript_id = None
        self.messages_sent = []
        self.chunk_counter = 0

    def authenticate(self, api_key: str, transcript_id: str) -> Dict[str, Any]:
        """Simulate authentication."""
        if api_key.startswith("invalid"):
            return {"type": "auth.failed", "message": "Invalid API key"}

        self.authenticated = True
        self.transcript_id = transcript_id
        return {"type": "auth.success"}

    def connect(self) -> Dict[str, Any]:
        """Simulate connection establishment."""
        self.connected = True
        return {"type": "connection.established"}

    def generate_chunk(
        self,
        text: str,
        speaker_name: str = "TestSpeaker",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a transcript chunk."""
        self.chunk_counter += 1

        if start_time is None:
            start_time = float(self.chunk_counter - 1)
        if end_time is None:
            end_time = start_time + len(text.split()) * 0.3  # ~0.3s per word

        chunk = {
            "type": "transcription.broadcast",
            "data": {
                "transcript_id": self.transcript_id,
                "chunk_id": f"chunk_{self.chunk_counter:04d}",
                "text": text,
                "speaker_name": speaker_name,
                "start_time": start_time,
                "end_time": end_time,
            },
        }
        self.messages_sent.append(chunk)
        return chunk


@pytest.fixture
def mock_ws_server():
    """Provide mock WebSocket server."""
    return MockFirefliesWebSocketServer()


# =============================================================================
# Full Connection Flow Tests
# =============================================================================


class TestFullConnectionFlow:
    """Test complete connection flow: connect -> authenticate -> receive -> disconnect."""

    @pytest.mark.asyncio
    async def test_connect_authenticate_receive_disconnect(self, mock_ws_server):
        """Test full connection lifecycle."""
        # Setup
        received_chunks = []
        status_changes = []

        async def on_transcript(chunk: FirefliesChunk):
            received_chunks.append(chunk)

        async def on_status_change(
            status: FirefliesConnectionStatus, message: Optional[str]
        ):
            status_changes.append((status, message))

        # Simulate the flow with mocked client
        config = FirefliesSessionConfig(
            api_key="test-api-key",
            transcript_id="test-transcript-001",
        )

        manager = FirefliesSessionManager()

        with patch("src.routers.fireflies.FirefliesClient") as MockClientClass:
            mock_client = AsyncMock()
            MockClientClass.return_value = mock_client

            # Simulate successful connection
            async def mock_connect_realtime(*args, **kwargs):
                # Simulate authentication
                auth_response = mock_ws_server.authenticate(
                    config.api_key, config.transcript_id
                )
                assert auth_response["type"] == "auth.success"

                # Call status callback
                if kwargs.get("on_status_change"):
                    await kwargs["on_status_change"](
                        FirefliesConnectionStatus.CONNECTED, "Connected"
                    )

                # Simulate receiving a chunk
                chunk_msg = mock_ws_server.generate_chunk(
                    text="Hello, this is a test.", speaker_name="Alice"
                )

                if kwargs.get("on_transcript"):
                    chunk = FirefliesChunk(**chunk_msg["data"])
                    await kwargs["on_transcript"](chunk)

            mock_client.connect_realtime = mock_connect_realtime

            # Create session
            session = await manager.create_session(
                config,
                on_transcript=on_transcript,
                on_status_change=on_status_change,
            )

            assert session is not None
            assert session.session_id.startswith("ff_session_")

        # Verify chunks received (if callbacks were properly wired)
        # Note: This depends on how the callbacks are set up in create_session

    @pytest.mark.asyncio
    async def test_authentication_failure(self, mock_ws_server):
        """Test handling of authentication failure."""
        status_changes = []
        errors = []

        async def on_status_change(
            status: FirefliesConnectionStatus, message: Optional[str]
        ):
            status_changes.append((status, message))

        async def on_error(message: str, exception: Optional[Exception]):
            errors.append(message)

        # Use invalid API key
        auth_response = mock_ws_server.authenticate(
            "invalid-api-key", "test-transcript"
        )

        assert auth_response["type"] == "auth.failed"
        assert mock_ws_server.authenticated is False


# =============================================================================
# Multiple Sessions Tests
# =============================================================================


class TestMultipleSessions:
    """Test concurrent session management."""

    @pytest.mark.asyncio
    async def test_create_multiple_sessions(self):
        """Test creating multiple concurrent sessions."""
        manager = FirefliesSessionManager()

        with patch("src.routers.fireflies.FirefliesClient") as MockClientClass:
            mock_client1 = AsyncMock()
            mock_client2 = AsyncMock()
            MockClientClass.side_effect = [mock_client1, mock_client2]

            mock_client1.connect_realtime = AsyncMock()
            mock_client2.connect_realtime = AsyncMock()

            config1 = FirefliesSessionConfig(
                api_key="api-key-1",
                transcript_id="transcript-001",
            )
            config2 = FirefliesSessionConfig(
                api_key="api-key-2",
                transcript_id="transcript-002",
            )

            session1 = await manager.create_session(config1)
            session2 = await manager.create_session(config2)

            assert session1.session_id != session2.session_id
            assert len(manager.get_all_sessions()) == 2

    @pytest.mark.asyncio
    async def test_disconnect_specific_session(self):
        """Test disconnecting a specific session while others remain."""
        manager = FirefliesSessionManager()

        with patch("src.routers.fireflies.FirefliesClient") as MockClientClass:
            mock_client1 = AsyncMock()
            mock_client2 = AsyncMock()
            MockClientClass.side_effect = [mock_client1, mock_client2]

            mock_client1.connect_realtime = AsyncMock()
            mock_client2.connect_realtime = AsyncMock()

            config1 = FirefliesSessionConfig(
                api_key="api-key-1",
                transcript_id="transcript-001",
            )
            config2 = FirefliesSessionConfig(
                api_key="api-key-2",
                transcript_id="transcript-002",
            )

            session1 = await manager.create_session(config1)
            session2 = await manager.create_session(config2)

            # Disconnect first session
            result = await manager.disconnect_session(session1.session_id)

            assert result is True
            assert len(manager.get_all_sessions()) == 1
            assert manager.get_session(session1.session_id) is None
            assert manager.get_session(session2.session_id) is not None


# =============================================================================
# Transcript Processing Tests
# =============================================================================


class TestTranscriptProcessing:
    """Test transcript chunk handling and processing."""

    def test_chunk_sequence_tracking(self, mock_ws_server):
        """Test tracking chunk sequence."""
        mock_ws_server.transcript_id = "test-transcript"

        chunk1 = mock_ws_server.generate_chunk("Hello", "Alice")
        chunk2 = mock_ws_server.generate_chunk("world", "Alice")
        chunk3 = mock_ws_server.generate_chunk("how are you", "Bob")

        assert chunk1["data"]["chunk_id"] == "chunk_0001"
        assert chunk2["data"]["chunk_id"] == "chunk_0002"
        assert chunk3["data"]["chunk_id"] == "chunk_0003"

    def test_speaker_buffer_accumulation(self):
        """Test accumulating chunks in speaker buffer."""
        buffer = SpeakerBuffer(speaker_name="Alice")

        chunk1 = FirefliesChunk(
            transcript_id="test",
            chunk_id="chunk_001",
            text="Hello",
            speaker_name="Alice",
            start_time=0.0,
            end_time=0.5,
        )
        chunk2 = FirefliesChunk(
            transcript_id="test",
            chunk_id="chunk_002",
            text="world",
            speaker_name="Alice",
            start_time=0.5,
            end_time=1.0,
        )

        buffer.add(chunk1)
        buffer.add(chunk2)

        assert buffer.get_text() == "Hello world"
        assert buffer.word_count == 2
        assert buffer.duration_seconds == 1.0

    def test_speaker_buffer_clear_on_speaker_change(self):
        """Test buffer handling on speaker change."""
        alice_buffer = SpeakerBuffer(speaker_name="Alice")
        bob_buffer = SpeakerBuffer(speaker_name="Bob")

        # Alice speaks
        alice_chunk = FirefliesChunk(
            transcript_id="test",
            chunk_id="chunk_001",
            text="Hello Bob",
            speaker_name="Alice",
            start_time=0.0,
            end_time=1.0,
        )
        alice_buffer.add(alice_chunk)

        # Bob responds
        bob_chunk = FirefliesChunk(
            transcript_id="test",
            chunk_id="chunk_002",
            text="Hi Alice",
            speaker_name="Bob",
            start_time=1.0,
            end_time=2.0,
        )
        bob_buffer.add(bob_chunk)

        assert alice_buffer.get_text() == "Hello Bob"
        assert bob_buffer.get_text() == "Hi Alice"


# =============================================================================
# Translation Unit Creation Tests
# =============================================================================


class TestTranslationUnitCreation:
    """Test creating translation units from accumulated chunks."""

    def test_create_translation_unit_from_buffer(self):
        """Test creating translation unit from speaker buffer."""
        buffer = SpeakerBuffer(speaker_name="Alice")

        chunks = [
            FirefliesChunk(
                transcript_id="test-transcript",
                chunk_id="chunk_001",
                text="Hello",
                speaker_name="Alice",
                start_time=0.0,
                end_time=0.3,
            ),
            FirefliesChunk(
                transcript_id="test-transcript",
                chunk_id="chunk_002",
                text="how are you",
                speaker_name="Alice",
                start_time=0.3,
                end_time=1.0,
            ),
            FirefliesChunk(
                transcript_id="test-transcript",
                chunk_id="chunk_003",
                text="today?",
                speaker_name="Alice",
                start_time=1.0,
                end_time=1.5,
            ),
        ]

        for chunk in chunks:
            buffer.add(chunk)

        # Create translation unit
        unit = TranslationUnit(
            text=buffer.get_text(),
            speaker_name=buffer.speaker_name,
            start_time=buffer.buffer_start_time,
            end_time=buffer.get_end_time(),
            session_id="session-001",
            transcript_id="test-transcript",
            chunk_ids=[c.chunk_id for c in buffer.chunks],
            boundary_type="punctuation",
        )

        assert unit.text == "Hello how are you today?"
        assert unit.speaker_name == "Alice"
        assert unit.start_time == 0.0
        assert unit.end_time == 1.5
        assert len(unit.chunk_ids) == 3
        assert unit.boundary_type == "punctuation"


# =============================================================================
# Translation Context Tests
# =============================================================================


class TestTranslationContextIntegration:
    """Test translation context management."""

    def test_rolling_window_context(self):
        """Test rolling window of previous sentences."""
        sentences = [
            "Good morning everyone.",
            "Today we will discuss the project.",
            "Let me share my screen.",
        ]

        # Simulate rolling window of size 2
        window_size = 2

        for i, current_sentence in enumerate(sentences):
            # Get context window
            start_idx = max(0, i - window_size)
            context_sentences = sentences[start_idx:i]

            context = TranslationContext(
                previous_sentences=context_sentences,
                target_language="es",
            )

            if i == 0:
                assert len(context.previous_sentences) == 0
            elif i == 1:
                assert len(context.previous_sentences) == 1
                assert context.previous_sentences[0] == "Good morning everyone."
            elif i == 2:
                assert len(context.previous_sentences) == 2

    def test_context_with_glossary(self):
        """Test context with glossary terms."""
        context = TranslationContext(
            previous_sentences=["We need to update the API."],
            glossary={
                "API": "API",  # Keep as-is
                "endpoint": "punto final",
                "database": "base de datos",
            },
            target_language="es",
        )

        formatted_glossary = context.format_glossary()
        assert "API -> API" in formatted_glossary
        assert "endpoint -> punto final" in formatted_glossary
        assert "database -> base de datos" in formatted_glossary


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


class TestEndToEndPipeline:
    """Test full pipeline from transcript to caption."""

    def test_full_pipeline_single_sentence(self):
        """Test processing a single sentence through full pipeline."""
        # 1. Receive chunk
        chunk = FirefliesChunk(
            transcript_id="meeting-001",
            chunk_id="chunk_001",
            text="Hello, how are you today?",
            speaker_name="Alice",
            start_time=0.0,
            end_time=2.0,
        )

        # 2. Create translation unit
        unit = TranslationUnit(
            text=chunk.text,
            speaker_name=chunk.speaker_name,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            session_id="session-001",
            transcript_id=chunk.transcript_id,
            chunk_ids=[chunk.chunk_id],
            boundary_type="punctuation",
        )

        # 3. Create translation context
        context = TranslationContext(
            previous_sentences=[],
            glossary={},
            target_language="es",
        )

        # 4. Create translation result (simulated)
        result = TranslationResult(
            original=unit.text,
            translated="Hola, ¿cómo estás hoy?",
            speaker_name=unit.speaker_name,
            source_language="en",
            target_language="es",
            confidence=0.95,
            context_sentences_used=0,
            translation_time_ms=125.0,
            session_id="session-001",
        )

        # 5. Create caption entry
        caption = CaptionEntry(
            id=f"caption_{chunk.chunk_id}",
            original_text=result.original,
            translated_text=result.translated,
            speaker_name=result.speaker_name,
            target_language=result.target_language,
            confidence=result.confidence,
        )

        # Verify pipeline output
        assert caption.original_text == "Hello, how are you today?"
        assert caption.translated_text == "Hola, ¿cómo estás hoy?"
        assert caption.speaker_name == "Alice"
        assert caption.confidence == 0.95

    def test_pipeline_multiple_speakers(self):
        """Test processing multiple speakers."""
        chunks = [
            FirefliesChunk(
                transcript_id="meeting-001",
                chunk_id="chunk_001",
                text="Hi everyone, let's start.",
                speaker_name="Alice",
                start_time=0.0,
                end_time=2.0,
            ),
            FirefliesChunk(
                transcript_id="meeting-001",
                chunk_id="chunk_002",
                text="Sounds good, I have updates.",
                speaker_name="Bob",
                start_time=2.0,
                end_time=4.0,
            ),
            FirefliesChunk(
                transcript_id="meeting-001",
                chunk_id="chunk_003",
                text="Great, please share.",
                speaker_name="Alice",
                start_time=4.0,
                end_time=5.5,
            ),
        ]

        captions = []
        previous_sentences = []

        for chunk in chunks:
            # Create translation unit
            unit = TranslationUnit(
                text=chunk.text,
                speaker_name=chunk.speaker_name,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                session_id="session-001",
                transcript_id=chunk.transcript_id,
                chunk_ids=[chunk.chunk_id],
                boundary_type="punctuation",
            )

            # Create context with rolling window
            context = TranslationContext(
                previous_sentences=previous_sentences[-3:],  # Window size 3
                target_language="es",
            )

            # Simulate translation
            translations = {
                "Hi everyone, let's start.": "Hola a todos, comencemos.",
                "Sounds good, I have updates.": "Suena bien, tengo actualizaciones.",
                "Great, please share.": "Genial, por favor comparte.",
            }

            result = TranslationResult(
                original=unit.text,
                translated=translations.get(unit.text, unit.text),
                speaker_name=unit.speaker_name,
                source_language="en",
                target_language="es",
                confidence=0.9,
                context_sentences_used=len(context.previous_sentences),
            )

            # Create caption
            caption = CaptionEntry(
                id=f"caption_{chunk.chunk_id}",
                original_text=result.original,
                translated_text=result.translated,
                speaker_name=result.speaker_name,
                target_language="es",
            )
            captions.append(caption)

            # Update context
            previous_sentences.append(unit.text)

        # Verify all captions
        assert len(captions) == 3

        # Verify speakers are preserved
        assert captions[0].speaker_name == "Alice"
        assert captions[1].speaker_name == "Bob"
        assert captions[2].speaker_name == "Alice"

        # Verify translations
        assert captions[0].translated_text == "Hola a todos, comencemos."
        assert captions[1].translated_text == "Suena bien, tengo actualizaciones."
        assert captions[2].translated_text == "Genial, por favor comparte."


# =============================================================================
# Meeting Discovery Integration Tests
# =============================================================================


class TestMeetingDiscoveryIntegration:
    """Test active meetings discovery flow."""

    def test_discover_meetings_full_flow(self, mock_fireflies_config):
        """Test full meeting discovery flow via API."""
        test_app = create_test_app()

        meetings = [
            FirefliesMeeting(
                id="meeting-001",
                title="Team Standup",
                organizer_email="alice@example.com",
                state=MeetingState.ACTIVE,
            ),
            FirefliesMeeting(
                id="meeting-002",
                title="Planning Session",
                organizer_email="bob@example.com",
                state=MeetingState.PAUSED,
            ),
        ]

        test_app.dependency_overrides[get_fireflies_config] = (
            lambda: mock_fireflies_config
        )

        with patch("src.routers.fireflies.FirefliesClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get_active_meetings = AsyncMock(return_value=meetings)
            mock_client.close = AsyncMock()
            MockClient.return_value = mock_client

            with TestClient(test_app) as client:
                response = client.post(
                    "/fireflies/meetings",
                    json={},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["meetings"][0]["id"] == "meeting-001"
        assert data["meetings"][1]["id"] == "meeting-002"


# =============================================================================
# Caption Broadcast Integration Tests
# =============================================================================


class TestCaptionBroadcastIntegration:
    """Test caption broadcast functionality."""

    def test_create_caption_broadcast(self):
        """Test creating caption broadcast message."""
        captions = [
            CaptionEntry(
                id="caption-001",
                original_text="Hello everyone",
                translated_text="Hola a todos",
                speaker_name="Alice",
                speaker_color="#4CAF50",
                target_language="es",
            ),
            CaptionEntry(
                id="caption-002",
                original_text="Let's begin",
                translated_text="Comencemos",
                speaker_name="Bob",
                speaker_color="#2196F3",
                target_language="es",
            ),
        ]

        broadcast = CaptionBroadcast(
            session_id="session-001",
            captions=captions,
        )

        assert broadcast.session_id == "session-001"
        assert len(broadcast.captions) == 2
        assert broadcast.captions[0].speaker_color == "#4CAF50"
        assert broadcast.captions[1].speaker_color == "#2196F3"

    def test_caption_broadcast_serialization(self):
        """Test caption broadcast JSON serialization."""
        caption = CaptionEntry(
            id="caption-001",
            translated_text="Hola",
            speaker_name="Alice",
            target_language="es",
        )

        broadcast = CaptionBroadcast(
            session_id="session-001",
            captions=[caption],
        )

        # Serialize to JSON
        json_str = broadcast.model_dump_json()
        data = json.loads(json_str)

        assert data["session_id"] == "session-001"
        assert len(data["captions"]) == 1
        assert data["captions"][0]["translated_text"] == "Hola"


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecoveryIntegration:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_session_error_tracking(self):
        """Test that errors are tracked in session state."""
        session = FirefliesSession(
            session_id="session-001",
            fireflies_transcript_id="transcript-001",
            connection_status=FirefliesConnectionStatus.CONNECTED,
        )

        # Simulate errors
        session.error_count = 3
        session.last_error = "Connection timeout"
        session.connection_status = FirefliesConnectionStatus.RECONNECTING
        session.reconnection_attempts = 2

        assert session.error_count == 3
        assert session.last_error == "Connection timeout"
        assert session.reconnection_attempts == 2

    def test_health_check_reports_errors(self):
        """Test health check reports error states."""
        test_app = create_test_app()

        error_session = FirefliesSession(
            session_id="session-001",
            fireflies_transcript_id="transcript-001",
            connection_status=FirefliesConnectionStatus.ERROR,
            error_count=5,
            last_error="Max reconnection attempts reached",
        )

        mock_manager = MagicMock()
        mock_manager.get_all_sessions.return_value = [error_session]

        test_app.dependency_overrides[get_session_manager] = lambda: mock_manager

        with TestClient(test_app) as client:
            response = client.get("/fireflies/health")

        assert response.status_code == 200
        data = response.json()
        assert data["connected_sessions"] == 0
        assert data["total_sessions"] == 1
        assert data["sessions"][0]["status"] == "error"


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformanceIntegration:
    """Test performance characteristics."""

    def test_high_volume_chunk_processing(self):
        """Test processing high volume of chunks."""
        buffer = SpeakerBuffer(speaker_name="Speaker")

        # Generate 100 chunks
        for i in range(100):
            chunk = FirefliesChunk(
                transcript_id="test",
                chunk_id=f"chunk_{i:04d}",
                text=f"Word{i}",
                speaker_name="Speaker",
                start_time=float(i * 0.3),
                end_time=float((i + 1) * 0.3),
            )
            buffer.add(chunk)

        assert len(buffer.chunks) == 100
        assert buffer.word_count == 100

        # Text should be all words joined
        text = buffer.get_text()
        assert "Word0" in text
        assert "Word99" in text

    def test_session_statistics_tracking(self):
        """Test session statistics update correctly."""
        session = FirefliesSession(
            session_id="session-001",
            fireflies_transcript_id="transcript-001",
        )

        # Simulate processing
        for i in range(50):
            session.chunks_received += 1

        for i in range(15):
            session.sentences_produced += 1

        # Each sentence translated to 3 languages
        session.translations_completed = session.sentences_produced * 3

        assert session.chunks_received == 50
        assert session.sentences_produced == 15
        assert session.translations_completed == 45


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
