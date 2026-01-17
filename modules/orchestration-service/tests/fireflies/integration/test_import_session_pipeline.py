"""
Integration tests for Import Session Pipeline.

Tests the session-based import functionality to ensure imported transcripts
flow through the same pipeline as live data (DRY principle).

These tests verify:
1. Session creation and configuration
2. Pipeline coordinator initialization
3. Chunk processing through the coordinator
4. Database storage (when available)
5. Translation integration (when available)
6. Session finalization and stats
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from routers.fireflies import FirefliesSessionManager
from services.pipeline import (
    TranscriptionPipelineCoordinator,
    PipelineConfig,
    ImportChunkAdapter,
)
from services.caption_buffer import CaptionBuffer


class TestFirefliesSessionManagerImportSession:
    """Test FirefliesSessionManager.create_import_session()"""

    @pytest.fixture
    def session_manager(self):
        """Create a fresh session manager for each test."""
        return FirefliesSessionManager()

    @pytest.fixture
    def sample_sentences(self):
        """Sample sentences for testing."""
        return [
            {
                "text": "Good morning everyone.",
                "speaker_name": "Alice",
                "start_time": 0.0,
                "end_time": 2.0,
            },
            {
                "text": "Thanks for joining the meeting today.",
                "speaker_name": "Alice",
                "start_time": 2.5,
                "end_time": 5.0,
            },
            {
                "text": "Let's discuss the project updates.",
                "speaker_name": "Bob",
                "start_time": 5.5,
                "end_time": 8.0,
            },
            {
                "text": "I've completed the API integration.",
                "speaker_name": "Bob",
                "start_time": 8.5,
                "end_time": 11.0,
            },
            {
                "text": "That's great progress!",
                "speaker_name": "Alice",
                "start_time": 11.5,
                "end_time": 13.0,
            },
        ]

    @pytest.mark.asyncio
    async def test_create_import_session_returns_session_id_and_coordinator(
        self, session_manager
    ):
        """Create import session returns session_id and coordinator."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        assert session_id is not None
        assert session_id.startswith("import_")
        assert isinstance(coordinator, TranscriptionPipelineCoordinator)

    @pytest.mark.asyncio
    async def test_create_import_session_initializes_coordinator(self, session_manager):
        """Coordinator is properly initialized after session creation."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Coordinator should be initialized
        assert coordinator._initialized is True

    @pytest.mark.asyncio
    async def test_create_import_session_stores_in_manager(self, session_manager):
        """Session is stored in manager's internal state."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Should be retrievable from manager
        stored_coordinator = session_manager.get_coordinator(session_id)
        assert stored_coordinator is coordinator

        # Caption buffer should also be stored
        caption_buffer = session_manager.get_caption_buffer(session_id)
        assert isinstance(caption_buffer, CaptionBuffer)

    @pytest.mark.asyncio
    async def test_create_import_session_uses_import_adapter(self, session_manager):
        """Session uses ImportChunkAdapter."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        assert isinstance(coordinator.adapter, ImportChunkAdapter)
        assert coordinator.adapter.source_type == "fireflies_import"

    @pytest.mark.asyncio
    async def test_create_import_session_configures_pipeline_correctly(
        self, session_manager
    ):
        """Pipeline config is set correctly for imports."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es", "fr"],
            glossary_id="glossary_001",
            domain="technology",
        )

        config = coordinator.config
        assert config.session_id == session_id
        assert config.source_type == "fireflies_import"
        assert config.transcript_id == "ff_test_123"
        assert config.target_languages == ["es", "fr"]
        assert config.glossary_id == "glossary_001"
        assert config.domain == "technology"

    @pytest.mark.asyncio
    async def test_create_import_session_relaxed_aggregation_settings(
        self, session_manager
    ):
        """Import sessions have relaxed aggregation settings since sentences are pre-formed."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        config = coordinator.config
        # Should have relaxed settings for pre-formed sentences
        assert config.pause_threshold_ms == 100  # Low threshold
        assert config.max_words_per_sentence >= 1000  # High limit
        assert config.min_words_for_translation == 1  # Translate everything

    @pytest.mark.asyncio
    async def test_create_multiple_import_sessions(self, session_manager):
        """Can create multiple import sessions simultaneously."""
        session_id_1, coordinator_1 = await session_manager.create_import_session(
            transcript_id="ff_test_1",
            transcript_title="Meeting 1",
            target_languages=["es"],
        )

        session_id_2, coordinator_2 = await session_manager.create_import_session(
            transcript_id="ff_test_2",
            transcript_title="Meeting 2",
            target_languages=["fr"],
        )

        assert session_id_1 != session_id_2
        assert coordinator_1 is not coordinator_2
        assert session_manager.get_coordinator(session_id_1) is coordinator_1
        assert session_manager.get_coordinator(session_id_2) is coordinator_2


class TestImportSessionChunkProcessing:
    """Test processing chunks through the import session pipeline."""

    @pytest.fixture
    def session_manager(self):
        return FirefliesSessionManager()

    @pytest.fixture
    def sample_sentences(self):
        return [
            {
                "text": "Hello everyone.",
                "speaker_name": "Alice",
                "start_time": 0.0,
                "end_time": 1.5,
                "index": 0,
            },
            {
                "text": "Welcome to the meeting.",
                "speaker_name": "Alice",
                "start_time": 2.0,
                "end_time": 4.0,
                "index": 1,
            },
            {
                "text": "Let me share my screen.",
                "speaker_name": "Bob",
                "start_time": 4.5,
                "end_time": 6.5,
                "index": 2,
            },
        ]

    @pytest.mark.asyncio
    async def test_process_single_chunk(self, session_manager, sample_sentences):
        """Process a single chunk through the coordinator."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Process first sentence
        await coordinator.process_raw_chunk(sample_sentences[0])

        # Check stats updated
        stats = coordinator.get_stats()
        assert stats["chunks_received"] == 1

    @pytest.mark.asyncio
    async def test_process_multiple_chunks(self, session_manager, sample_sentences):
        """Process multiple chunks through the coordinator."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Process all sentences
        for sentence in sample_sentences:
            await coordinator.process_raw_chunk(sentence)

        stats = coordinator.get_stats()
        assert stats["chunks_received"] == 3

    @pytest.mark.asyncio
    async def test_process_tracks_speakers(self, session_manager, sample_sentences):
        """Processing tracks unique speakers."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        for sentence in sample_sentences:
            await coordinator.process_raw_chunk(sentence)

        stats = coordinator.get_stats()
        # Should have detected both Alice and Bob
        # Stats use "speaker_names" field
        speaker_names = stats.get("speaker_names", [])
        assert len(speaker_names) == 2
        assert "Alice" in speaker_names
        assert "Bob" in speaker_names

    @pytest.mark.asyncio
    async def test_process_chunk_with_transcript_id(self, session_manager):
        """Chunks preserve transcript_id through processing."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        sentence = {
            "text": "Test sentence.",
            "speaker_name": "Alice",
            "start_time": 0.0,
            "end_time": 1.0,
            "transcript_id": "ff_test_123",
            "index": 0,
        }

        await coordinator.process_raw_chunk(sentence)

        # Stats should include source metadata
        stats = coordinator.get_stats()
        assert stats.get("source_type") == "fireflies_import"


class TestImportSessionFinalization:
    """Test session finalization and cleanup."""

    @pytest.fixture
    def session_manager(self):
        return FirefliesSessionManager()

    @pytest.mark.asyncio
    async def test_finalize_import_session_returns_stats(self, session_manager):
        """Finalize returns pipeline statistics."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Process some chunks
        sentences = [
            {"text": "Hello.", "speaker_name": "Alice", "start_time": 0.0, "end_time": 1.0, "index": 0},
            {"text": "Hi.", "speaker_name": "Bob", "start_time": 1.0, "end_time": 2.0, "index": 1},
        ]
        for s in sentences:
            await coordinator.process_raw_chunk(s)

        # Finalize
        stats = await session_manager.finalize_import_session(session_id)

        assert isinstance(stats, dict)
        assert "chunks_received" in stats
        assert stats["chunks_received"] == 2

    @pytest.mark.asyncio
    async def test_finalize_import_session_cleans_up(self, session_manager):
        """Finalize cleans up session from manager."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        await session_manager.finalize_import_session(session_id)

        # Should be removed from manager
        assert session_manager.get_coordinator(session_id) is None
        assert session_manager.get_caption_buffer(session_id) is None

    @pytest.mark.asyncio
    async def test_finalize_nonexistent_session_returns_error(self, session_manager):
        """Finalize nonexistent session returns error dict."""
        stats = await session_manager.finalize_import_session("nonexistent_session")

        assert "error" in stats
        assert "not found" in stats["error"]

    @pytest.mark.asyncio
    async def test_finalize_flushes_pending_content(self, session_manager):
        """Finalize flushes any pending buffered content."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Mock flush to track calls
        flush_called = False
        original_flush = coordinator.flush

        async def mock_flush():
            nonlocal flush_called
            flush_called = True
            await original_flush()

        coordinator.flush = mock_flush

        await session_manager.finalize_import_session(session_id)

        assert flush_called is True


class TestImportSessionWithMockTranslation:
    """Test import session with mocked translation service."""

    @pytest.fixture
    def session_manager(self):
        return FirefliesSessionManager()

    @pytest.fixture
    def mock_translation_client(self):
        """Create mock translation client."""
        client = MagicMock()
        client.translate = AsyncMock(return_value={
            "translated_text": "Hola a todos.",
            "source_language": "en",
            "target_language": "es",
            "confidence": 0.95,
        })
        return client

    @pytest.mark.asyncio
    async def test_import_with_translation_client(
        self, session_manager, mock_translation_client
    ):
        """Import session works with translation client."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
            translation_client=mock_translation_client,
        )

        # Coordinator should have translation client
        assert coordinator.translation_client is mock_translation_client

        # Process a chunk
        sentence = {
            "text": "Hello everyone.",
            "speaker_name": "Alice",
            "start_time": 0.0,
            "end_time": 1.5,
            "index": 0,
        }
        await coordinator.process_raw_chunk(sentence)

        # Finalize
        stats = await session_manager.finalize_import_session(session_id)

        assert stats["chunks_received"] == 1


class TestImportSessionWithMockDatabase:
    """Test import session with mocked database manager."""

    @pytest.fixture
    def session_manager(self):
        return FirefliesSessionManager()

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        manager = MagicMock()
        manager.transcript_manager = MagicMock()
        manager.transcript_manager.store_transcript = AsyncMock(return_value="transcript_123")
        manager.translation_manager = MagicMock()
        manager.translation_manager.store_translation = AsyncMock(return_value="translation_456")
        return manager

    @pytest.mark.asyncio
    async def test_import_with_db_manager(self, session_manager, mock_db_manager):
        """Import session works with database manager."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
            db_manager=mock_db_manager,
        )

        # Coordinator should have db manager
        assert coordinator.db_manager is mock_db_manager


class TestImportVsLiveSessionParity:
    """Test that import sessions behave like live sessions (DRY)."""

    @pytest.fixture
    def session_manager(self):
        return FirefliesSessionManager()

    @pytest.mark.asyncio
    async def test_import_session_has_same_components_as_live(self, session_manager):
        """Import session has same pipeline components as live session would."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Should have same internal components
        assert coordinator._sentence_aggregator is not None
        assert coordinator.caption_buffer is not None
        assert coordinator.config is not None

    @pytest.mark.asyncio
    async def test_import_session_uses_same_pipeline_config_class(self, session_manager):
        """Import session uses same PipelineConfig class as live sessions."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        assert isinstance(coordinator.config, PipelineConfig)

    @pytest.mark.asyncio
    async def test_import_session_produces_same_stats_format(self, session_manager):
        """Import session stats have same format as live sessions."""
        session_id, coordinator = await session_manager.create_import_session(
            transcript_id="ff_test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Process a chunk
        await coordinator.process_raw_chunk({
            "text": "Test.",
            "speaker_name": "Alice",
            "start_time": 0.0,
            "end_time": 1.0,
            "index": 0,
        })

        stats = coordinator.get_stats()

        # Should have standard stat fields
        expected_fields = [
            "chunks_received",
            "sentences_produced",
            "source_type",
            "session_id",
            "initialized",
        ]
        for field in expected_fields:
            assert field in stats, f"Missing expected field: {field}"
