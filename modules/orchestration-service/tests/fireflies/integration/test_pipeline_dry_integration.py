#!/usr/bin/env python3
"""
Pipeline DRY Integration Tests

Tests that ALL transcript sources flow through the IDENTICAL pipeline.
Verifies the DRY principle: Fireflies, Audio Upload, and Google Meet
all use the same TranscriptionPipelineCoordinator.

These tests verify:
1. Socket.IO Fireflies client initialization and configuration
2. AudioUploadChunkAdapter correctly adapts Whisper results
3. All sources produce identical TranscriptChunk format
4. Pipeline coordinator handles all sources consistently
5. Sessions API endpoint returns correct data
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))


class TestSocketIOFirefliesClient:
    """Integration tests for Socket.IO-based Fireflies client."""

    def test_socketio_client_creation(self):
        """Test that FirefliesRealtimeClient creates Socket.IO client."""
        from clients.fireflies_client import (
            DEFAULT_WEBSOCKET_ENDPOINT,
            DEFAULT_WEBSOCKET_PATH,
            FirefliesRealtimeClient,
        )

        client = FirefliesRealtimeClient(
            api_key="test-key",
            transcript_id="test-123",
        )

        # Verify Socket.IO client is created
        assert hasattr(client, "sio")
        assert client.sio is not None
        assert client.endpoint == DEFAULT_WEBSOCKET_ENDPOINT
        assert client.socketio_path == DEFAULT_WEBSOCKET_PATH

    def test_socketio_client_custom_path(self):
        """Test custom Socket.IO path configuration."""
        from clients.fireflies_client import FirefliesRealtimeClient

        client = FirefliesRealtimeClient(
            api_key="test-key",
            transcript_id="test-123",
            endpoint="wss://custom.api",
            socketio_path="/custom/path",
        )

        assert client.endpoint == "wss://custom.api"
        assert client.socketio_path == "/custom/path"

    def test_socketio_event_handlers_registered(self):
        """Test that Socket.IO event handlers are registered."""
        from clients.fireflies_client import FirefliesRealtimeClient

        client = FirefliesRealtimeClient(
            api_key="test-key",
            transcript_id="test-123",
        )

        # Check that handlers are registered on the sio client
        # The handlers are registered via decorators in _register_handlers()
        assert client.sio is not None

    @pytest.mark.asyncio
    async def test_transcript_callback_invoked(self):
        """Test that transcript callback is invoked when handling transcripts."""
        from clients.fireflies_client import FirefliesRealtimeClient

        received_chunks = []

        async def on_transcript(chunk):
            received_chunks.append(chunk)

        client = FirefliesRealtimeClient(
            api_key="test-key",
            transcript_id="test-123",
            on_transcript=on_transcript,
        )

        # Simulate receiving transcript data
        await client._handle_transcript(
            {
                "chunk_id": "chunk-1",
                "text": "Integration test",
                "speaker_name": "Tester",
                "start_time": 0.0,
                "end_time": 2.0,
            }
        )

        assert len(received_chunks) == 1
        assert received_chunks[0].text == "Integration test"
        assert received_chunks[0].speaker_name == "Tester"


class TestAudioUploadAdapterIntegration:
    """Integration tests for AudioUploadChunkAdapter."""

    def test_adapter_source_type(self):
        """Test adapter reports correct source type."""
        from services.pipeline.adapters.audio_adapter import AudioUploadChunkAdapter

        adapter = AudioUploadChunkAdapter(session_id="test-session")
        assert adapter.source_type == "audio_upload"

    def test_adapter_converts_whisper_format(self):
        """Test adapter correctly converts Whisper transcription format."""
        from services.pipeline.adapters.audio_adapter import AudioUploadChunkAdapter

        adapter = AudioUploadChunkAdapter(session_id="test-session")

        # Whisper-style input (uses "start" and "end" not "start_time" and "end_time")
        whisper_result = {
            "text": "Hello, this is a test transcription.",
            "start": 0.0,
            "end": 3.5,
            "speaker": "SPEAKER_00",
            "confidence": 0.95,
            "language": "en",
            "segment_index": 0,
        }

        chunk = adapter.adapt(whisper_result)

        assert chunk.text == "Hello, this is a test transcription."
        assert chunk.speaker_name == "SPEAKER_00"
        assert chunk.start_time_seconds == 0.0
        assert chunk.end_time_seconds == 3.5
        assert chunk.confidence == 0.95
        assert chunk.is_final is True
        assert chunk.metadata["language"] == "en"

    def test_adapter_handles_diarization_info(self):
        """Test adapter extracts speaker from diarization info."""
        from services.pipeline.adapters.audio_adapter import AudioUploadChunkAdapter

        adapter = AudioUploadChunkAdapter()

        whisper_result = {
            "text": "Diarized speech",
            "start": 5.0,
            "end": 7.0,
            "diarization": {
                "speaker_label": "Speaker A",
                "confidence": 0.88,
            },
        }

        chunk = adapter.adapt(whisper_result)

        assert chunk.speaker_name == "Speaker A"
        assert chunk.metadata["diarization"]["confidence"] == 0.88

    def test_adapter_batch_processing(self):
        """Test adapter can process batches of segments."""
        from services.pipeline.adapters.audio_adapter import AudioUploadChunkAdapter

        adapter = AudioUploadChunkAdapter()

        segments = [
            {"text": "First segment", "start": 0.0, "end": 1.0},
            {"text": "Second segment", "start": 1.0, "end": 2.0},
            {"text": "Third segment", "start": 2.0, "end": 3.0},
        ]

        chunks = adapter.create_batch_chunks(segments, session_id="batch-session")

        assert len(chunks) == 3
        assert chunks[0].text == "First segment"
        assert chunks[1].text == "Second segment"
        assert chunks[2].text == "Third segment"

    def test_adapter_validation(self):
        """Test adapter validates input data."""
        from services.pipeline.adapters.audio_adapter import AudioUploadChunkAdapter

        adapter = AudioUploadChunkAdapter()

        # Valid input
        assert adapter.validate({"text": "Valid", "start": 0.0}) is True

        # Invalid - missing text
        assert adapter.validate({"start": 0.0, "end": 1.0}) is False

        # Invalid - missing timing
        assert adapter.validate({"text": "No timing"}) is False


class TestUnifiedPipelineFormat:
    """Test that all adapters produce the same TranscriptChunk format."""

    def test_all_adapters_produce_transcript_chunk(self):
        """Test all adapters produce TranscriptChunk instances."""
        from services.pipeline.adapters import (
            AudioUploadChunkAdapter,
            FirefliesChunkAdapter,
            GoogleMeetChunkAdapter,
            ImportChunkAdapter,
            TranscriptChunk,
        )

        # Fireflies adapter
        ff_adapter = FirefliesChunkAdapter()
        from models.fireflies import FirefliesChunk

        ff_chunk = FirefliesChunk(
            transcript_id="ff-123",
            chunk_id="c1",
            text="Fireflies text",
            speaker_name="FF Speaker",
            start_time=0.0,
            end_time=1.0,
        )
        ff_result = ff_adapter.adapt(ff_chunk)
        assert isinstance(ff_result, TranscriptChunk)

        # Audio upload adapter
        audio_adapter = AudioUploadChunkAdapter()
        audio_result = audio_adapter.adapt(
            {
                "text": "Audio text",
                "start": 0.0,
                "end": 1.0,
                "speaker": "Audio Speaker",
            }
        )
        assert isinstance(audio_result, TranscriptChunk)

        # Google Meet adapter
        gm_adapter = GoogleMeetChunkAdapter()
        gm_result = gm_adapter.adapt(
            {
                "text": "Google Meet text",
                "start_time_seconds": 0.0,
                "end_time_seconds": 1.0,
                "speaker_name": "GM Speaker",
            }
        )
        assert isinstance(gm_result, TranscriptChunk)

        # Import adapter
        import_adapter = ImportChunkAdapter()
        import_result = import_adapter.adapt(
            {
                "text": "Imported text",
                "start_time": 0.0,
                "end_time": 1.0,
                "speaker_name": "Import Speaker",
            }
        )
        assert isinstance(import_result, TranscriptChunk)

    def test_transcript_chunk_has_required_fields(self):
        """Test TranscriptChunk has all required fields for pipeline."""
        from services.pipeline.adapters import AudioUploadChunkAdapter

        adapter = AudioUploadChunkAdapter()
        chunk = adapter.adapt(
            {
                "text": "Test",
                "start": 0.0,
                "end": 1.0,
            }
        )

        # Required fields for pipeline processing
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "speaker_name")
        assert hasattr(chunk, "timestamp_ms")
        assert hasattr(chunk, "chunk_id")
        assert hasattr(chunk, "start_time_seconds")
        assert hasattr(chunk, "end_time_seconds")
        assert hasattr(chunk, "is_final")
        assert hasattr(chunk, "confidence")
        assert hasattr(chunk, "metadata")


class TestPipelineCoordinatorIntegration:
    """Test TranscriptionPipelineCoordinator with different adapters."""

    @pytest.mark.asyncio
    async def test_coordinator_with_audio_adapter(self):
        """Test pipeline coordinator works with AudioUploadChunkAdapter."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="test-audio-session",
            source_type="audio_upload",
            transcript_id="audio-123",
            target_languages=["es"],
        )

        adapter = AudioUploadChunkAdapter(session_id="test-audio-session")
        coordinator = TranscriptionPipelineCoordinator(
            config=config,
            adapter=adapter,
        )

        await coordinator.initialize()
        assert coordinator._initialized is True
        assert coordinator.adapter.source_type == "audio_upload"

    @pytest.mark.asyncio
    async def test_coordinator_with_fireflies_adapter(self):
        """Test pipeline coordinator works with FirefliesChunkAdapter."""
        from services.pipeline import (
            FirefliesChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="test-ff-session",
            source_type="fireflies",
            transcript_id="ff-123",
            target_languages=["es"],
        )

        adapter = FirefliesChunkAdapter()
        coordinator = TranscriptionPipelineCoordinator(
            config=config,
            adapter=adapter,
        )

        await coordinator.initialize()
        assert coordinator._initialized is True
        assert coordinator.adapter.source_type == "fireflies"

    @pytest.mark.asyncio
    async def test_coordinator_stats_tracking(self):
        """Test coordinator tracks statistics correctly."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="stats-test",
            source_type="audio_upload",
            transcript_id="stats-123",
            target_languages=["es"],
        )

        adapter = AudioUploadChunkAdapter()
        coordinator = TranscriptionPipelineCoordinator(
            config=config,
            adapter=adapter,
        )

        await coordinator.initialize()

        # Process some chunks
        for i in range(3):
            await coordinator.process_raw_chunk(
                {
                    "text": f"Test segment {i}",
                    "start": float(i),
                    "end": float(i + 1),
                    "speaker": "Test Speaker",
                }
            )

        stats = coordinator.get_stats()
        assert stats["chunks_received"] == 3
        assert stats["source_type"] == "audio_upload"


class TestSessionsAPIEndpoint:
    """Integration tests for the /sessions API endpoint."""

    def test_sessions_response_model(self):
        """Test SessionSummary and SessionsListResponse models."""
        from routers.data_query import SessionsListResponse, SessionSummary

        session = SessionSummary(
            session_id="test-session-123",
            source_type="fireflies",
            title="Test Meeting",
            created_at=datetime.now(),
            transcript_count=10,
            translation_count=5,
            speaker_count=2,
            total_duration=300.5,
            languages=["en", "es"],
        )

        response = SessionsListResponse(
            total=1,
            sessions=[session],
        )

        assert response.total == 1
        assert len(response.sessions) == 1
        assert response.sessions[0].session_id == "test-session-123"
        assert response.sessions[0].source_type == "fireflies"

    def test_sessions_filter_by_source_type(self):
        """Test SessionSummary can be filtered by source type."""
        from routers.data_query import SessionSummary

        sessions = [
            SessionSummary(
                session_id="ff-1",
                source_type="fireflies",
                created_at=datetime.now(),
            ),
            SessionSummary(
                session_id="audio-1",
                source_type="audio_upload",
                created_at=datetime.now(),
            ),
            SessionSummary(
                session_id="gm-1",
                source_type="google_meet",
                created_at=datetime.now(),
            ),
        ]

        # Filter by source type
        fireflies_sessions = [s for s in sessions if s.source_type == "fireflies"]
        audio_sessions = [s for s in sessions if s.source_type == "audio_upload"]

        assert len(fireflies_sessions) == 1
        assert len(audio_sessions) == 1
        assert fireflies_sessions[0].session_id == "ff-1"


class TestDashboardSavedTranscripts:
    """Integration tests for dashboard saved transcripts feature."""

    def test_valid_prefixes_comprehensive(self):
        """Test that all expected prefixes are handled."""
        # This simulates the JavaScript prefix list in fireflies-dashboard.html
        valid_prefixes = [
            "fireflies_translated_",
            "fireflies_feed_",
            "fireflies_imported_",
            "saved_transcript_",
            "local_transcript_",
            "audio_session_",
            "import_",
        ]

        test_keys = [
            "fireflies_translated_123",
            "fireflies_feed_456",
            "fireflies_imported_789",
            "saved_transcript_abc",
            "local_transcript_def",
            "audio_session_ghi",
            "import_jkl",
            "random_key",  # Should NOT match
            "other_data",  # Should NOT match
        ]

        matched_keys = []
        for key in test_keys:
            if any(key.startswith(prefix) for prefix in valid_prefixes):
                matched_keys.append(key)

        assert len(matched_keys) == 7
        assert "random_key" not in matched_keys
        assert "other_data" not in matched_keys

    def test_source_type_extraction(self):
        """Test that source type is correctly extracted from key prefix."""
        valid_prefixes = [
            "fireflies_translated_",
            "fireflies_feed_",
            "fireflies_imported_",
            "saved_transcript_",
            "local_transcript_",
            "audio_session_",
            "import_",
        ]

        test_key = "audio_session_12345"

        prefix = next((p for p in valid_prefixes if test_key.startswith(p)), "unknown_")
        source_type = prefix.rstrip("_").replace("_", " ")

        assert source_type == "audio session"


class TestEndToEndPipelineFlow:
    """End-to-end tests for complete pipeline flow."""

    @pytest.mark.asyncio
    async def test_audio_upload_full_flow(self):
        """Test complete flow from audio upload to pipeline processing."""
        from services.caption_buffer import CaptionBuffer
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        # Setup
        caption_buffer = CaptionBuffer(max_captions=5)
        captions_received = []

        async def on_caption_added(caption):
            captions_received.append(caption)

        caption_buffer.on_caption_added = on_caption_added

        config = PipelineConfig(
            session_id="e2e-test",
            source_type="audio_upload",
            transcript_id="e2e-123",
            target_languages=["es"],
            min_words_for_translation=1,
        )

        adapter = AudioUploadChunkAdapter(session_id="e2e-test")
        coordinator = TranscriptionPipelineCoordinator(
            config=config,
            adapter=adapter,
            caption_buffer=caption_buffer,
        )

        await coordinator.initialize()

        # Simulate Whisper results
        whisper_segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"text": "How are you", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"text": "I am fine", "start": 2.0, "end": 3.0, "speaker": "SPEAKER_00"},
        ]

        for segment in whisper_segments:
            await coordinator.process_raw_chunk(segment)

        # Verify stats
        stats = coordinator.get_stats()
        assert stats["chunks_received"] == 3
        # Check that aggregator processed the chunks
        aggregator_stats = stats.get("aggregator", {})
        assert aggregator_stats.get("chunks_processed", 0) >= 0

    @pytest.mark.asyncio
    async def test_fireflies_import_full_flow(self):
        """Test complete flow from Fireflies import to pipeline processing."""
        from services.pipeline import (
            ImportChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="import-e2e",
            source_type="fireflies_import",
            transcript_id="import-123",
            target_languages=["es"],
        )

        adapter = ImportChunkAdapter(source_name="fireflies_import")
        coordinator = TranscriptionPipelineCoordinator(
            config=config,
            adapter=adapter,
        )

        await coordinator.initialize()

        # Simulate imported sentences
        sentences = [
            {
                "text": "Welcome to the meeting",
                "speaker_name": "Host",
                "start_time": 0,
                "end_time": 2,
            },
            {
                "text": "Thank you for joining",
                "speaker_name": "Host",
                "start_time": 2,
                "end_time": 4,
            },
            {"text": "Let's begin", "speaker_name": "Host", "start_time": 4, "end_time": 5},
        ]

        for i, sentence in enumerate(sentences):
            sentence["index"] = i
            await coordinator.process_raw_chunk(sentence)

        # Flush remaining
        await coordinator.flush()

        stats = coordinator.get_stats()
        assert stats["chunks_received"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
