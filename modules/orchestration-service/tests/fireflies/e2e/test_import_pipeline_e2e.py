"""
End-to-end tests for Fireflies Import Pipeline.

Tests the complete import flow from API request through the session-based
pipeline, ensuring:
1. API endpoint accepts import requests
2. Sessions are created properly
3. Sentences flow through the coordinator pipeline
4. Translations are processed (when service available)
5. Data is stored consistently
6. Response includes proper statistics

These tests use the Fireflies mock server for transcript data.
"""

import pytest
import asyncio
import logging
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_transcript_data():
    """Sample transcript data as returned by Fireflies API."""
    return {
        "id": "test_transcript_001",
        "title": "Product Planning Meeting",
        "dateString": "2026-01-15",
        "duration": 1800,  # 30 minutes
        "sentences": [
            {
                "text": "Good morning everyone, thanks for joining.",
                "speaker_name": "Sarah Chen",
                "speaker_id": "speaker_1",
                "start_time": 0.0,
                "end_time": 3.5,
            },
            {
                "text": "Today we'll be discussing our Q2 roadmap.",
                "speaker_name": "Sarah Chen",
                "speaker_id": "speaker_1",
                "start_time": 4.0,
                "end_time": 7.5,
            },
            {
                "text": "I have some updates from the engineering team.",
                "speaker_name": "Mike Johnson",
                "speaker_id": "speaker_2",
                "start_time": 8.0,
                "end_time": 11.0,
            },
            {
                "text": "We've completed the API integration last week.",
                "speaker_name": "Mike Johnson",
                "speaker_id": "speaker_2",
                "start_time": 11.5,
                "end_time": 15.0,
            },
            {
                "text": "That's excellent progress Mike!",
                "speaker_name": "Sarah Chen",
                "speaker_id": "speaker_1",
                "start_time": 15.5,
                "end_time": 18.0,
            },
            {
                "text": "What about the machine learning features?",
                "speaker_name": "Sarah Chen",
                "speaker_id": "speaker_1",
                "start_time": 18.5,
                "end_time": 21.5,
            },
            {
                "text": "The ML pipeline is 80% complete.",
                "speaker_name": "Lisa Park",
                "speaker_id": "speaker_3",
                "start_time": 22.0,
                "end_time": 25.0,
            },
            {
                "text": "We're running final tests on the model accuracy.",
                "speaker_name": "Lisa Park",
                "speaker_id": "speaker_3",
                "start_time": 25.5,
                "end_time": 29.0,
            },
            {
                "text": "When do you expect it to be production ready?",
                "speaker_name": "Sarah Chen",
                "speaker_id": "speaker_1",
                "start_time": 29.5,
                "end_time": 33.0,
            },
            {
                "text": "By the end of next week at the latest.",
                "speaker_name": "Lisa Park",
                "speaker_id": "speaker_3",
                "start_time": 33.5,
                "end_time": 36.5,
            },
        ],
    }


@pytest.fixture
def large_transcript_data():
    """Large transcript for stress testing."""
    sentences = []
    speakers = ["Alice", "Bob", "Charlie", "Diana"]

    for i in range(100):
        sentences.append({
            "text": f"This is sentence number {i + 1} in our test transcript.",
            "speaker_name": speakers[i % len(speakers)],
            "speaker_id": f"speaker_{i % len(speakers)}",
            "start_time": i * 3.0,
            "end_time": (i * 3.0) + 2.5,
        })

    return {
        "id": "large_transcript_001",
        "title": "Extended Meeting",
        "dateString": "2026-01-15",
        "duration": 300,
        "sentences": sentences,
    }


# =============================================================================
# Unit Tests for Import API (Direct Function Testing)
# =============================================================================


class TestImportTranscriptAPIUnit:
    """Unit tests for the import API endpoint logic."""

    @pytest.mark.asyncio
    async def test_import_creates_session_with_correct_config(
        self, sample_transcript_data
    ):
        """Import creates session with correct configuration."""
        from routers.fireflies import (
            FirefliesSessionManager,
            ImportTranscriptRequest,
        )

        manager = FirefliesSessionManager()

        # Create import session
        session_id, coordinator = await manager.create_import_session(
            transcript_id=sample_transcript_data["id"],
            transcript_title=sample_transcript_data["title"],
            target_languages=["es"],
            glossary_id="test_glossary",
            domain="technology",
        )

        # Verify configuration
        assert coordinator.config.transcript_id == "test_transcript_001"
        assert coordinator.config.target_languages == ["es"]
        assert coordinator.config.glossary_id == "test_glossary"
        assert coordinator.config.domain == "technology"
        assert coordinator.config.source_type == "fireflies_import"

        # Clean up
        await manager.finalize_import_session(session_id)

    @pytest.mark.asyncio
    async def test_import_processes_all_sentences(self, sample_transcript_data):
        """Import processes all sentences through pipeline."""
        from routers.fireflies import FirefliesSessionManager

        manager = FirefliesSessionManager()

        session_id, coordinator = await manager.create_import_session(
            transcript_id=sample_transcript_data["id"],
            transcript_title=sample_transcript_data["title"],
            target_languages=["es"],
        )

        # Process all sentences
        for i, sentence in enumerate(sample_transcript_data["sentences"]):
            sentence["index"] = i
            await coordinator.process_raw_chunk(sentence)

        stats = coordinator.get_stats()
        assert stats["chunks_received"] == 10

        # Clean up
        await manager.finalize_import_session(session_id)

    @pytest.mark.asyncio
    async def test_import_tracks_unique_speakers(self, sample_transcript_data):
        """Import correctly tracks unique speakers."""
        from routers.fireflies import FirefliesSessionManager

        manager = FirefliesSessionManager()

        session_id, coordinator = await manager.create_import_session(
            transcript_id=sample_transcript_data["id"],
            transcript_title=sample_transcript_data["title"],
            target_languages=["es"],
        )

        # Process all sentences
        for i, sentence in enumerate(sample_transcript_data["sentences"]):
            sentence["index"] = i
            await coordinator.process_raw_chunk(sentence)

        stats = coordinator.get_stats()

        # Should have 3 unique speakers: Sarah Chen, Mike Johnson, Lisa Park
        # Stats use "speaker_names" field
        speakers = stats.get("speaker_names", [])
        assert len(speakers) == 3
        assert "Sarah Chen" in speakers
        assert "Mike Johnson" in speakers
        assert "Lisa Park" in speakers

        # Clean up
        await manager.finalize_import_session(session_id)

    @pytest.mark.asyncio
    async def test_import_handles_large_transcripts(self, large_transcript_data):
        """Import handles large transcripts efficiently."""
        from routers.fireflies import FirefliesSessionManager

        manager = FirefliesSessionManager()

        session_id, coordinator = await manager.create_import_session(
            transcript_id=large_transcript_data["id"],
            transcript_title=large_transcript_data["title"],
            target_languages=["es"],
        )

        # Process all 100 sentences
        for i, sentence in enumerate(large_transcript_data["sentences"]):
            sentence["index"] = i
            await coordinator.process_raw_chunk(sentence)

        stats = coordinator.get_stats()
        assert stats["chunks_received"] == 100

        # Should have 4 unique speakers (speaker_names field)
        speakers = stats.get("speaker_names", [])
        assert len(speakers) == 4

        # Clean up
        final_stats = await manager.finalize_import_session(session_id)
        assert final_stats["chunks_received"] == 100


class TestImportTranscriptAPIIntegration:
    """Integration tests for the import API endpoint."""

    @pytest.fixture
    def mock_fireflies_client(self, sample_transcript_data):
        """Mock Fireflies client that returns sample data."""
        client = MagicMock()
        client.get_transcript_detail = AsyncMock(return_value=sample_transcript_data)
        client.close = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_import_endpoint_success_flow(
        self, sample_transcript_data, mock_fireflies_client
    ):
        """Test successful import flow."""
        from routers.fireflies import (
            import_transcript_to_db,
            ImportTranscriptRequest,
            FirefliesSessionManager,
            get_session_manager,
        )
        from config import FirefliesSettings

        # Create request
        request = ImportTranscriptRequest(
            api_key="test_api_key",
            include_translations=False,  # Skip translation for this test
            target_language="es",
        )

        # Mock dependencies
        ff_config = FirefliesSettings(api_key="default_key")

        with patch("routers.fireflies.FirefliesClient", return_value=mock_fireflies_client), \
             patch("routers.fireflies.get_data_pipeline", return_value=None):

            result = await import_transcript_to_db(
                transcript_id="test_transcript_001",
                request=request,
                manager=get_session_manager(),
                ff_config=ff_config,
            )

        # Verify response
        assert result["success"] is True
        assert result["transcript_id"] == "test_transcript_001"
        assert result["title"] == "Product Planning Meeting"
        assert result["total_sentences"] == 10
        assert result["processed"] == 10
        assert result["errors"] == 0
        assert result["target_language"] == "es"
        assert "session_id" in result
        assert result["session_id"].startswith("import_")

    @pytest.mark.asyncio
    async def test_import_endpoint_missing_api_key(self):
        """Test import fails without API key."""
        from routers.fireflies import (
            import_transcript_to_db,
            ImportTranscriptRequest,
            get_session_manager,
        )
        from fastapi import HTTPException
        from unittest.mock import MagicMock

        request = ImportTranscriptRequest(
            api_key=None,  # No API key
            include_translations=False,
        )

        # Create mock config with no API key using has_api_key() pattern
        mock_ff_config = MagicMock()
        mock_ff_config.api_key = None

        with pytest.raises(HTTPException) as exc_info:
            await import_transcript_to_db(
                transcript_id="test_transcript",
                request=request,
                manager=get_session_manager(),
                ff_config=mock_ff_config,
            )

        assert exc_info.value.status_code == 400
        assert "API key required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_import_endpoint_transcript_not_found(self, mock_fireflies_client):
        """Test import handles transcript not found."""
        from routers.fireflies import (
            import_transcript_to_db,
            ImportTranscriptRequest,
            get_session_manager,
        )
        from config import FirefliesSettings
        from fastapi import HTTPException

        # Mock client to return None (not found)
        mock_fireflies_client.get_transcript_detail = AsyncMock(return_value=None)

        request = ImportTranscriptRequest(
            api_key="test_key",
            include_translations=False,
        )

        ff_config = FirefliesSettings(api_key="default_key")

        with patch("routers.fireflies.FirefliesClient", return_value=mock_fireflies_client):
            with pytest.raises(HTTPException) as exc_info:
                await import_transcript_to_db(
                    transcript_id="nonexistent",
                    request=request,
                    manager=get_session_manager(),
                    ff_config=ff_config,
                )

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_import_endpoint_empty_transcript(self, mock_fireflies_client):
        """Test import handles empty transcript."""
        from routers.fireflies import (
            import_transcript_to_db,
            ImportTranscriptRequest,
            get_session_manager,
        )
        from config import FirefliesSettings
        from fastapi import HTTPException

        # Mock client to return empty transcript
        mock_fireflies_client.get_transcript_detail = AsyncMock(return_value={
            "id": "empty_transcript",
            "title": "Empty Meeting",
            "sentences": [],
        })

        request = ImportTranscriptRequest(
            api_key="test_key",
            include_translations=False,
        )

        ff_config = FirefliesSettings(api_key="default_key")

        with patch("routers.fireflies.FirefliesClient", return_value=mock_fireflies_client):
            with pytest.raises(HTTPException) as exc_info:
                await import_transcript_to_db(
                    transcript_id="empty_transcript",
                    request=request,
                    manager=get_session_manager(),
                    ff_config=ff_config,
                )

        assert exc_info.value.status_code == 400
        assert "no sentences" in str(exc_info.value.detail)


class TestImportWithTranslation:
    """Test import flow with translation service integration."""

    @pytest.fixture
    def mock_translation_result(self):
        """Mock translation result."""
        return {
            "translated_text": "Buenos días a todos, gracias por unirse.",
            "source_language": "en",
            "target_language": "es",
            "confidence": 0.92,
            "processing_time_ms": 150,
        }

    @pytest.mark.asyncio
    async def test_import_with_translation_enabled(
        self, sample_transcript_data, mock_translation_result
    ):
        """Test import with translation service enabled."""
        from routers.fireflies import FirefliesSessionManager

        manager = FirefliesSessionManager()

        # Mock translation client
        mock_translation_client = MagicMock()
        mock_translation_client.translate = AsyncMock(return_value=mock_translation_result)

        session_id, coordinator = await manager.create_import_session(
            transcript_id=sample_transcript_data["id"],
            transcript_title=sample_transcript_data["title"],
            target_languages=["es"],
            translation_client=mock_translation_client,
        )

        # Coordinator should have translation client configured
        assert coordinator.translation_client is mock_translation_client

        # Process a few sentences
        for i, sentence in enumerate(sample_transcript_data["sentences"][:3]):
            sentence["index"] = i
            await coordinator.process_raw_chunk(sentence)

        stats = coordinator.get_stats()
        assert stats["chunks_received"] == 3

        # Clean up
        await manager.finalize_import_session(session_id)


class TestImportSessionDRYParity:
    """Test that import sessions maintain DRY parity with live sessions."""

    @pytest.mark.asyncio
    async def test_import_uses_same_coordinator_class(self, sample_transcript_data):
        """Import uses TranscriptionPipelineCoordinator like live sessions."""
        from routers.fireflies import FirefliesSessionManager
        from services.pipeline import TranscriptionPipelineCoordinator

        manager = FirefliesSessionManager()

        session_id, coordinator = await manager.create_import_session(
            transcript_id=sample_transcript_data["id"],
            transcript_title=sample_transcript_data["title"],
            target_languages=["es"],
        )

        assert isinstance(coordinator, TranscriptionPipelineCoordinator)

        # Clean up
        await manager.finalize_import_session(session_id)

    @pytest.mark.asyncio
    async def test_import_uses_same_caption_buffer_class(self, sample_transcript_data):
        """Import uses CaptionBuffer like live sessions."""
        from routers.fireflies import FirefliesSessionManager
        from services.caption_buffer import CaptionBuffer

        manager = FirefliesSessionManager()

        session_id, coordinator = await manager.create_import_session(
            transcript_id=sample_transcript_data["id"],
            transcript_title=sample_transcript_data["title"],
            target_languages=["es"],
        )

        caption_buffer = manager.get_caption_buffer(session_id)
        assert isinstance(caption_buffer, CaptionBuffer)

        # Clean up
        await manager.finalize_import_session(session_id)

    @pytest.mark.asyncio
    async def test_import_stats_format_matches_live(self, sample_transcript_data):
        """Import session stats have same format as live sessions."""
        from routers.fireflies import FirefliesSessionManager

        manager = FirefliesSessionManager()

        session_id, coordinator = await manager.create_import_session(
            transcript_id=sample_transcript_data["id"],
            transcript_title=sample_transcript_data["title"],
            target_languages=["es"],
        )

        # Process some data
        sentence = sample_transcript_data["sentences"][0]
        sentence["index"] = 0
        await coordinator.process_raw_chunk(sentence)

        stats = coordinator.get_stats()

        # Verify expected fields are present (same as live session stats)
        expected_fields = [
            "chunks_received",
            "sentences_produced",
            "translations_completed",
            "translations_failed",
            "captions_displayed",
            "errors",
            "speaker_names",  # Changed from speakers_detected
            "source_type",
            "session_id",
            "initialized",
        ]

        for field in expected_fields:
            assert field in stats, f"Missing field: {field}"

        # Clean up
        await manager.finalize_import_session(session_id)


class TestImportErrorHandling:
    """Test error handling in import pipeline."""

    @pytest.mark.asyncio
    async def test_import_continues_on_individual_chunk_error(self):
        """Import continues processing even if individual chunks fail."""
        from routers.fireflies import FirefliesSessionManager

        manager = FirefliesSessionManager()

        session_id, coordinator = await manager.create_import_session(
            transcript_id="test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Process valid chunks with one invalid
        sentences = [
            {"text": "Valid sentence 1.", "speaker_name": "Alice", "start_time": 0.0, "end_time": 1.0, "index": 0},
            {"text": "", "speaker_name": "Bob", "start_time": 1.0, "end_time": 2.0, "index": 1},  # Empty text
            {"text": "Valid sentence 2.", "speaker_name": "Alice", "start_time": 2.0, "end_time": 3.0, "index": 2},
        ]

        errors = 0
        for sentence in sentences:
            try:
                await coordinator.process_raw_chunk(sentence)
            except Exception:
                errors += 1

        # Should have processed valid chunks
        stats = coordinator.get_stats()
        assert stats["chunks_received"] >= 2

        # Clean up
        await manager.finalize_import_session(session_id)

    @pytest.mark.asyncio
    async def test_import_session_cleanup_on_error(self):
        """Session is properly cleaned up even on errors."""
        from routers.fireflies import FirefliesSessionManager

        manager = FirefliesSessionManager()

        session_id, coordinator = await manager.create_import_session(
            transcript_id="test_123",
            transcript_title="Test Meeting",
            target_languages=["es"],
        )

        # Verify session exists
        assert manager.get_coordinator(session_id) is not None

        # Finalize (even if processing failed)
        await manager.finalize_import_session(session_id)

        # Session should be cleaned up
        assert manager.get_coordinator(session_id) is None
        assert manager.get_caption_buffer(session_id) is None
