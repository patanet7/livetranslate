#!/usr/bin/env python3
"""
Unit Tests for Production Pipeline Fixes

Tests the 5 critical production-readiness fixes:
1. NULL safety in timeline queries
2. Cache eviction strategy (LRU)
3. Database connection pooling
4. Transaction support
5. Rate limiting / backpressure

Author: LiveTranslate Team
Version: 1.0
"""

import os
import sys
import pytest
import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from collections import OrderedDict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pipeline.data_pipeline import (
    TranscriptionDataPipeline,
    AudioChunkMetadata,
    TranscriptionResult,
    TranslationResult,
    create_data_pipeline,
)
from database.bot_session_manager import (
    BotSessionDatabaseManager,
    DatabaseConfig,
    TranslationRecord,
)


# ============================================================================
# TEST: NULL SAFETY IN TIMELINE QUERIES
# ============================================================================


@pytest.mark.asyncio
async def test_null_timestamp_filtering():
    """Test that NULL timestamps don't cause TypeErrors in timeline filtering."""
    # Create mock database manager
    db_manager = Mock()
    db_manager.db_pool = Mock()

    # Create pipeline
    pipeline = TranscriptionDataPipeline(
        db_manager=db_manager,
        enable_speaker_tracking=False,
        enable_segment_continuity=False,
    )

    # Mock translation manager to return translations with NULL timestamps
    mock_translations = [
        # Valid translation
        TranslationRecord(
            translation_id="trans_1",
            session_id="session_1",
            source_transcript_id="transcript_1",
            translated_text="Valid translation",
            source_language="en",
            target_language="es",
            translation_confidence=0.9,
            translation_service="test",
            speaker_id="SPEAKER_00",
            speaker_name="Test",
            start_timestamp=1.0,
            end_timestamp=3.0,
            processing_metadata={},
        ),
        # NULL start timestamp
        TranslationRecord(
            translation_id="trans_2",
            session_id="session_1",
            source_transcript_id="transcript_2",
            translated_text="NULL start timestamp",
            source_language="en",
            target_language="es",
            translation_confidence=0.9,
            translation_service="test",
            speaker_id="SPEAKER_00",
            speaker_name="Test",
            start_timestamp=None,  # NULL
            end_timestamp=5.0,
            processing_metadata={},
        ),
        # NULL end timestamp
        TranslationRecord(
            translation_id="trans_3",
            session_id="session_1",
            source_transcript_id="transcript_3",
            translated_text="NULL end timestamp",
            source_language="en",
            target_language="es",
            translation_confidence=0.9,
            translation_service="test",
            speaker_id="SPEAKER_00",
            speaker_name="Test",
            start_timestamp=6.0,
            end_timestamp=None,  # NULL
            processing_metadata={},
        ),
        # Both NULL
        TranslationRecord(
            translation_id="trans_4",
            session_id="session_1",
            source_transcript_id="transcript_4",
            translated_text="Both NULL",
            source_language="en",
            target_language="es",
            translation_confidence=0.9,
            translation_service="test",
            speaker_id="SPEAKER_00",
            speaker_name="Test",
            start_timestamp=None,  # NULL
            end_timestamp=None,  # NULL
            processing_metadata={},
        ),
    ]

    # Mock the translation manager methods
    async def mock_get_translations(session_id):
        return mock_translations

    db_manager.translation_manager = Mock()
    db_manager.translation_manager.get_session_translations = mock_get_translations

    # Mock the transcript manager to return empty list
    async def mock_get_transcripts(session_id):
        return []

    db_manager.transcript_manager = Mock()
    db_manager.transcript_manager.get_session_transcripts = mock_get_transcripts

    # Test: Should NOT raise TypeError even with NULL timestamps
    try:
        timeline = await pipeline.get_session_timeline(
            "session_1",
            start_time=0.0,
            end_time=10.0,
            include_translations=True,
        )
        # Success! No TypeError was raised
        assert True, "NULL safety works correctly"
    except TypeError as e:
        pytest.fail(f"NULL timestamps caused TypeError: {e}")

    # Verify that entries with NULL timestamps are handled gracefully
    assert isinstance(timeline, list), "Timeline should be a list"


@pytest.mark.asyncio
async def test_null_duration_calculation():
    """Test that NULL timestamps result in safe duration calculation."""
    db_manager = Mock()
    db_manager.db_pool = Mock()

    pipeline = TranscriptionDataPipeline(
        db_manager=db_manager,
        enable_speaker_tracking=False,
        enable_segment_continuity=False,
    )

    # Translation with both NULL timestamps
    mock_translation = TranslationRecord(
        translation_id="trans_1",
        session_id="session_1",
        source_transcript_id="transcript_1",
        translated_text="NULL timestamps",
        source_language="en",
        target_language="es",
        translation_confidence=0.9,
        translation_service="test",
        speaker_id=None,
        speaker_name=None,
        start_timestamp=None,
        end_timestamp=None,
        processing_metadata={},
    )

    async def mock_get_translations(session_id):
        return [mock_translation]

    async def mock_get_transcripts(session_id):
        return []

    db_manager.translation_manager = Mock()
    db_manager.translation_manager.get_session_translations = mock_get_translations
    db_manager.transcript_manager = Mock()
    db_manager.transcript_manager.get_session_transcripts = mock_get_transcripts

    # Should not raise error, duration should be 0.0 for NULL timestamps
    timeline = await pipeline.get_session_timeline("session_1")

    assert len(timeline) == 1
    assert timeline[0].duration == 0.0, "Duration should be 0.0 for NULL timestamps"
    assert timeline[0].timestamp == 0.0, "Timestamp should default to 0.0 for NULL"


# ============================================================================
# TEST: CACHE EVICTION STRATEGY
# ============================================================================


@pytest.mark.asyncio
async def test_cache_lru_eviction():
    """Test that cache properly evicts oldest entries when full."""
    db_manager = Mock()
    db_manager.db_pool = AsyncMock()

    # Create pipeline with small cache size for testing
    pipeline = TranscriptionDataPipeline(
        db_manager=db_manager,
        enable_speaker_tracking=False,
        enable_segment_continuity=True,
        max_cache_size=3,  # Small cache for testing
    )

    # Mock database connection
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    db_manager.db_pool.acquire = AsyncMock(return_value=mock_conn)
    db_manager.db_pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
    db_manager.db_pool.acquire().__aexit__ = AsyncMock()

    # Add entries to cache
    await pipeline._update_segment_continuity("session_1", "transcript_1")
    await pipeline._update_segment_continuity("session_2", "transcript_2")
    await pipeline._update_segment_continuity("session_3", "transcript_3")

    # Cache should be full
    assert len(pipeline._segment_cache) == 3
    assert pipeline._cache_evictions == 0

    # Add one more - should evict oldest (session_1)
    await pipeline._update_segment_continuity("session_4", "transcript_4")

    assert len(pipeline._segment_cache) == 3, "Cache size should stay at max"
    assert pipeline._cache_evictions == 1, "Should have evicted one entry"
    assert "session_1" not in pipeline._segment_cache, "Oldest entry should be evicted"
    assert "session_4" in pipeline._segment_cache, "New entry should be added"


@pytest.mark.asyncio
async def test_cache_statistics():
    """Test cache statistics tracking."""
    db_manager = Mock()
    db_manager.db_pool = AsyncMock()

    pipeline = TranscriptionDataPipeline(
        db_manager=db_manager,
        enable_speaker_tracking=False,
        enable_segment_continuity=True,
        max_cache_size=10,
    )

    # Mock database connection
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    db_manager.db_pool.acquire = AsyncMock(return_value=mock_conn)
    db_manager.db_pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
    db_manager.db_pool.acquire().__aexit__ = AsyncMock()

    # Add some entries
    await pipeline._update_segment_continuity("session_1", "transcript_1")
    await pipeline._update_segment_continuity("session_1", "transcript_2")  # Cache hit
    await pipeline._update_segment_continuity("session_2", "transcript_3")

    stats = pipeline.get_cache_statistics()

    assert stats["cache_size"] == 2
    assert stats["max_cache_size"] == 10
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 2
    assert 0.0 <= stats["hit_rate"] <= 1.0


@pytest.mark.asyncio
async def test_clear_session_cache():
    """Test clearing cache for specific session."""
    db_manager = Mock()
    db_manager.db_pool = AsyncMock()

    pipeline = TranscriptionDataPipeline(
        db_manager=db_manager,
        enable_speaker_tracking=False,
        enable_segment_continuity=True,
    )

    # Mock database connection
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    db_manager.db_pool.acquire = AsyncMock(return_value=mock_conn)
    db_manager.db_pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
    db_manager.db_pool.acquire().__aexit__ = AsyncMock()

    # Add entries
    await pipeline._update_segment_continuity("session_1", "transcript_1")
    await pipeline._update_segment_continuity("session_2", "transcript_2")

    assert len(pipeline._segment_cache) == 2

    # Clear session_1
    await pipeline.clear_session_cache("session_1")

    assert len(pipeline._segment_cache) == 1
    assert "session_1" not in pipeline._segment_cache
    assert "session_2" in pipeline._segment_cache


# ============================================================================
# TEST: DATABASE CONNECTION POOLING
# ============================================================================


def test_database_config_defaults():
    """Test DatabaseConfig has proper connection pool defaults."""
    config = DatabaseConfig()

    assert config.min_connections == 5
    assert config.max_connections == 20
    assert config.connection_timeout == 30.0
    assert config.command_timeout == 60.0
    assert config.max_queries == 50000
    assert config.max_inactive_connection_lifetime == 300.0


def test_database_config_custom_values():
    """Test DatabaseConfig accepts custom pool settings."""
    config = DatabaseConfig(
        min_connections=10,
        max_connections=50,
        connection_timeout=60.0,
        command_timeout=120.0,
    )

    assert config.min_connections == 10
    assert config.max_connections == 50
    assert config.connection_timeout == 60.0
    assert config.command_timeout == 120.0


# ============================================================================
# TEST: TRANSACTION SUPPORT
# ============================================================================


@pytest.mark.asyncio
async def test_transaction_context_manager():
    """Test transaction context manager commits on success."""
    db_manager = Mock()
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_transaction = AsyncMock()

    # Setup mock chain
    mock_conn.transaction = Mock(return_value=mock_transaction)
    mock_transaction.__aenter__ = AsyncMock()
    mock_transaction.__aexit__ = AsyncMock()

    mock_pool.acquire = Mock(return_value=mock_conn)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock()

    db_manager.db_pool = mock_pool

    pipeline = TranscriptionDataPipeline(
        db_manager=db_manager,
        enable_speaker_tracking=False,
        enable_segment_continuity=False,
    )

    # Use transaction context manager
    async with pipeline.transaction() as conn:
        assert conn == mock_conn

    # Verify transaction was entered and exited (commit)
    mock_transaction.__aenter__.assert_called_once()
    mock_transaction.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_process_complete_segment_atomic():
    """Test process_complete_segment atomicity."""
    db_manager = Mock()
    db_manager.db_pool = AsyncMock()

    pipeline = TranscriptionDataPipeline(
        db_manager=db_manager,
        enable_speaker_tracking=False,
        enable_segment_continuity=False,
    )

    # Mock successful operations
    pipeline.process_audio_chunk = AsyncMock(return_value="audio_123")
    pipeline.process_transcription_result = AsyncMock(return_value="transcript_123")
    pipeline.process_translation_result = AsyncMock(return_value="translation_123")

    # Mock transaction
    mock_conn = AsyncMock()
    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock()
    mock_transaction.__aexit__ = AsyncMock()
    mock_conn.transaction = Mock(return_value=mock_transaction)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock()

    db_manager.db_pool.acquire = Mock(return_value=mock_conn)

    # Test successful complete segment processing
    transcription = TranscriptionResult(
        text="Test",
        language="en",
        start_time=0.0,
        end_time=2.0,
        confidence=0.95,
    )

    translations = [
        TranslationResult(
            text="Prueba",
            source_language="en",
            target_language="es",
            confidence=0.90,
        )
    ]

    result = await pipeline.process_complete_segment(
        session_id="session_1",
        audio_bytes=b"test_audio",
        transcription=transcription,
        translations=translations,
    )

    assert result is not None
    assert result["file_id"] == "audio_123"
    assert result["transcript_id"] == "transcript_123"
    assert len(result["translation_ids"]) == 1


@pytest.mark.asyncio
async def test_process_complete_segment_rollback():
    """Test process_complete_segment rolls back on failure."""
    db_manager = Mock()
    db_manager.db_pool = AsyncMock()

    pipeline = TranscriptionDataPipeline(
        db_manager=db_manager,
        enable_speaker_tracking=False,
        enable_segment_continuity=False,
    )

    # Mock audio succeeds, transcription fails
    pipeline.process_audio_chunk = AsyncMock(return_value="audio_123")
    pipeline.process_transcription_result = AsyncMock(return_value=None)  # Fails

    # Mock transaction
    mock_conn = AsyncMock()
    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock()
    mock_transaction.__aexit__ = AsyncMock()
    mock_conn.transaction = Mock(return_value=mock_transaction)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock()

    db_manager.db_pool.acquire = Mock(return_value=mock_conn)

    transcription = TranscriptionResult(
        text="Test",
        language="en",
        start_time=0.0,
        end_time=2.0,
        confidence=0.95,
    )

    result = await pipeline.process_complete_segment(
        session_id="session_1",
        audio_bytes=b"test_audio",
        transcription=transcription,
        translations=[],
    )

    # Should return None on failure (transaction rolled back)
    assert result is None


# ============================================================================
# TEST: RATE LIMITING / BACKPRESSURE
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limit_statistics():
    """Test rate limiting statistics tracking."""
    from bot.bot_manager import GoogleMeetBotManager

    config = {
        "max_concurrent_db_operations": 10,
        "db_operation_timeout": 5.0,
    }

    manager = GoogleMeetBotManager(config=config)

    stats = manager.get_rate_limit_statistics()

    assert stats["max_concurrent_operations"] == 10
    assert stats["current_queue_depth"] == 0
    assert stats["operations_completed"] == 0
    assert stats["operations_rejected"] == 0
    assert stats["rejection_rate"] == 0.0


@pytest.mark.asyncio
async def test_rate_limited_operation_success():
    """Test successful rate-limited operation."""
    from bot.bot_manager import GoogleMeetBotManager

    config = {
        "max_concurrent_db_operations": 10,
        "db_operation_timeout": 5.0,
    }

    manager = GoogleMeetBotManager(config=config)

    # Mock operation that succeeds
    async def mock_operation():
        await asyncio.sleep(0.01)
        return "success"

    result = await manager._rate_limited_db_operation(
        mock_operation, "test_operation"
    )

    assert result == "success"
    assert manager._db_operations_completed == 1
    assert manager._db_operations_rejected == 0


@pytest.mark.asyncio
async def test_rate_limited_operation_timeout():
    """Test rate-limited operation timeout."""
    from bot.bot_manager import GoogleMeetBotManager

    config = {
        "max_concurrent_db_operations": 1,  # Only allow 1 concurrent operation
        "db_operation_timeout": 0.1,  # Very short timeout
    }

    manager = GoogleMeetBotManager(config=config)

    # Mock slow operation that holds the semaphore
    async def slow_operation():
        await asyncio.sleep(1.0)  # Longer than timeout
        return "slow"

    # Start first operation (holds semaphore)
    task1 = asyncio.create_task(
        manager._rate_limited_db_operation(slow_operation, "slow_op")
    )

    # Give it time to acquire semaphore
    await asyncio.sleep(0.05)

    # Start second operation (should timeout)
    result2 = await manager._rate_limited_db_operation(
        slow_operation, "timeout_op"
    )

    # Second operation should timeout and return None
    assert result2 is None
    assert manager._db_operations_rejected >= 1

    # Cleanup
    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
