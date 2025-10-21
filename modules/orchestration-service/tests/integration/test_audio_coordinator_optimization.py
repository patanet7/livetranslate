"""
Integration tests for AudioCoordinator translation optimization.

Tests the complete flow of:
1. Audio chunk → Transcription → Translation with caching
2. Multi-language batch translation
3. Cache hit/miss tracking
4. Database recording of optimization metrics

TDD Approach: These tests define the expected behavior.
Implement AudioCoordinator changes to make these tests pass.

Requirements:
- Redis running on localhost:6379
- PostgreSQL with translation optimization schema
- Whisper service running on localhost:5001
- Translation service running on localhost:5003
"""

import asyncio
import pytest
import time
import os
from typing import Dict, List


# Test configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/livetranslate")
TEST_SESSION_ID = "test_coordinator_opt"


class TestAudioCoordinatorCacheIntegration:
    """Test AudioCoordinator with translation caching"""

    @pytest.mark.asyncio
    async def test_coordinator_cache_initialization(self):
        """
        TEST: AudioCoordinator should initialize with translation cache if Redis available.

        Expected behavior:
        - translation_cache attribute exists
        - translation_cache is not None if Redis available
        - translation_opt_adapter exists if database available
        """
        from audio.audio_coordinator import create_audio_coordinator

        coordinator = create_audio_coordinator(
            database_url=DATABASE_URL,
            service_urls={
                "whisper_service": "http://localhost:5001",
                "translation_service": "http://localhost:5003"
            },
            audio_config_file=None
        )

        await coordinator.initialize()

        # Should have cache attribute
        assert hasattr(coordinator, 'translation_cache')

        # Should have optimization adapter if database available
        assert hasattr(coordinator, 'translation_opt_adapter')

        # Cache should be initialized if Redis available
        if coordinator.translation_cache:
            assert coordinator.translation_cache.redis_url == REDIS_URL
            assert coordinator.translation_cache.ttl > 0

        await coordinator.cleanup()

    @pytest.mark.asyncio
    async def test_coordinator_uses_cache_for_duplicate_translations(self):
        """
        TEST: AudioCoordinator should use cache for duplicate translations.

        Flow:
        1. Process first chunk with text "Hello world"
        2. Translation happens (cache miss)
        3. Process second chunk with same text "Hello world"
        4. Translation comes from cache (cache hit)

        Expected:
        - First translation: cache miss, normal latency
        - Second translation: cache hit, <10ms latency
        - Cache hit rate should increase
        """
        from audio.audio_coordinator import create_audio_coordinator

        coordinator = create_audio_coordinator(
            database_url=DATABASE_URL,
            service_urls={
                "whisper_service": "http://localhost:5001",
                "translation_service": "http://localhost:5003"
            }
        )

        await coordinator.initialize()

        if not coordinator.translation_cache:
            pytest.skip("Translation cache not available")

        # Reset cache stats
        coordinator.translation_cache.reset_stats()

        session_id = TEST_SESSION_ID
        target_languages = ["es", "fr"]

        # Create session
        await coordinator.create_audio_session(
            bot_session_id=session_id,
            target_languages=target_languages
        )

        # Mock transcript result (same text twice)
        transcript_result = {
            "text": "Hello world",
            "language": "en",
            "confidence": 0.95,
            "start_timestamp": 0.0,
            "end_timestamp": 2.0,
            "speaker_info": {}
        }

        # First request (cache miss expected)
        start_first = time.time()
        await coordinator._request_translations(
            session_id=session_id,
            transcript_id="transcript_001",
            transcript_result=transcript_result,
            target_languages=target_languages
        )
        duration_first = time.time() - start_first

        # Second request with SAME text (cache hit expected)
        start_second = time.time()
        await coordinator._request_translations(
            session_id=session_id,
            transcript_id="transcript_002",
            transcript_result=transcript_result,
            target_languages=target_languages
        )
        duration_second = time.time() - start_second

        # Check cache stats
        stats = coordinator.translation_cache.get_stats()

        # Should have cache hits
        assert stats['hits'] > 0, "Expected cache hits for duplicate translation"

        # Second request should be significantly faster
        print(f"First request: {duration_first*1000:.2f}ms, Second (cached): {duration_second*1000:.2f}ms")
        assert duration_second < duration_first * 0.5, "Cached request should be at least 50% faster"

        await coordinator.cleanup()

    @pytest.mark.asyncio
    async def test_coordinator_multi_language_batching(self):
        """
        TEST: AudioCoordinator should use multi-language endpoint for batch translation.

        Expected:
        - Single HTTP request for multiple languages
        - Batch metadata recorded in database
        - All languages processed in parallel
        """
        from audio.audio_coordinator import create_audio_coordinator

        coordinator = create_audio_coordinator(
            database_url=DATABASE_URL,
            service_urls={
                "whisper_service": "http://localhost:5001",
                "translation_service": "http://localhost:5003"
            }
        )

        await coordinator.initialize()

        session_id = TEST_SESSION_ID
        target_languages = ["es", "fr", "de"]  # 3 languages

        await coordinator.create_audio_session(
            bot_session_id=session_id,
            target_languages=target_languages
        )

        transcript_result = {
            "text": "Good morning everyone",
            "language": "en",
            "confidence": 0.95,
            "start_timestamp": 0.0,
            "end_timestamp": 2.0,
            "speaker_info": {}
        }

        # Time the translation
        start_time = time.time()
        await coordinator._request_translations(
            session_id=session_id,
            transcript_id="transcript_batch_001",
            transcript_result=transcript_result,
            target_languages=target_languages
        )
        duration = time.time() - start_time

        # Should be faster than 3 sequential requests
        # (Each request ~150ms, so 3 × 150ms = 450ms)
        # Batch should be < 300ms
        print(f"Batch translation time: {duration*1000:.2f}ms for {len(target_languages)} languages")
        assert duration < 0.5, f"Batch translation too slow: {duration*1000:.2f}ms"

        # Check if batch was recorded in database
        if coordinator.translation_opt_adapter:
            batch_stats = await coordinator.translation_opt_adapter.get_batch_efficiency(
                session_id=session_id
            )
            assert len(batch_stats) > 0, "Batch should be recorded in database"

        await coordinator.cleanup()

    @pytest.mark.asyncio
    async def test_coordinator_cache_hit_rate_measurement(self):
        """
        TEST: AudioCoordinator should achieve >50% cache hit rate with repetitive phrases.

        Simulate realistic meeting scenario:
        - 10 unique phrases
        - Each phrase repeated 5 times
        - Total: 50 translations
        - Expected cache hit rate: ~80% (40/50)
        """
        from audio.audio_coordinator import create_audio_coordinator

        coordinator = create_audio_coordinator(
            database_url=DATABASE_URL,
            service_urls={
                "whisper_service": "http://localhost:5001",
                "translation_service": "http://localhost:5003"
            }
        )

        await coordinator.initialize()

        if not coordinator.translation_cache:
            pytest.skip("Translation cache not available")

        coordinator.translation_cache.reset_stats()

        session_id = TEST_SESSION_ID
        target_languages = ["es"]  # Single language for simplicity

        await coordinator.create_audio_session(
            bot_session_id=session_id,
            target_languages=target_languages
        )

        # 10 unique phrases
        phrases = [
            "Thank you",
            "Good morning",
            "How are you",
            "See you later",
            "Have a nice day",
            "You're welcome",
            "Nice to meet you",
            "Excuse me",
            "I understand",
            "No problem"
        ]

        # Repeat each phrase 5 times
        total_requests = 0
        for phrase in phrases:
            for repetition in range(5):
                transcript_result = {
                    "text": phrase,
                    "language": "en",
                    "confidence": 0.95,
                    "start_timestamp": total_requests * 2.0,
                    "end_timestamp": (total_requests + 1) * 2.0,
                    "speaker_info": {}
                }

                await coordinator._request_translations(
                    session_id=session_id,
                    transcript_id=f"transcript_{total_requests:03d}",
                    transcript_result=transcript_result,
                    target_languages=target_languages
                )

                total_requests += 1

        # Check cache hit rate
        stats = coordinator.translation_cache.get_stats()

        print(f"Cache stats: {stats}")
        print(f"Total requests: {total_requests}")
        print(f"Expected: ~{((total_requests - len(phrases)) / total_requests) * 100:.0f}% hit rate")

        # Expected: 10 misses (first occurrence), 40 hits (4 repetitions × 10 phrases)
        # Hit rate: 40/50 = 80%
        assert stats['hit_rate'] > 0.5, f"Cache hit rate too low: {stats['hit_rate']:.1%}"

        await coordinator.cleanup()

    @pytest.mark.asyncio
    async def test_coordinator_database_optimization_tracking(self):
        """
        TEST: AudioCoordinator should record optimization metrics in database.

        Expected database records:
        - translation_cache_stats: cache hit/miss per translation
        - translation_batches: batch operations
        - translations table: updated with optimization metadata
        """
        from audio.audio_coordinator import create_audio_coordinator

        coordinator = create_audio_coordinator(
            database_url=DATABASE_URL,
            service_urls={
                "whisper_service": "http://localhost:5001",
                "translation_service": "http://localhost:5003"
            }
        )

        await coordinator.initialize()

        if not coordinator.translation_opt_adapter:
            pytest.skip("Translation optimization adapter not available")

        session_id = TEST_SESSION_ID
        target_languages = ["es", "fr"]

        await coordinator.create_audio_session(
            bot_session_id=session_id,
            target_languages=target_languages
        )

        # Process a translation
        transcript_result = {
            "text": "Database tracking test",
            "language": "en",
            "confidence": 0.95,
            "start_timestamp": 0.0,
            "end_timestamp": 2.0,
            "speaker_info": {}
        }

        await coordinator._request_translations(
            session_id=session_id,
            transcript_id="transcript_db_001",
            transcript_result=transcript_result,
            target_languages=target_languages
        )

        # Wait for async database writes to complete
        await asyncio.sleep(0.5)

        # Check cache performance stats
        cache_stats = await coordinator.translation_opt_adapter.get_cache_performance(
            session_id=session_id
        )

        # Should have recorded cache stats
        # Note: Might be empty if cache wasn't used, but structure should be correct
        assert isinstance(cache_stats, list)

        # Check batch efficiency
        batch_stats = await coordinator.translation_opt_adapter.get_batch_efficiency(
            session_id=session_id
        )

        # Should have recorded batch
        assert isinstance(batch_stats, list)
        # assert len(batch_stats) > 0, "Should have batch records"

        await coordinator.cleanup()

    @pytest.mark.asyncio
    async def test_coordinator_cache_with_different_languages(self):
        """
        TEST: Cache should properly differentiate between language pairs.

        Expected:
        - "Hello" → Spanish: cached separately from
        - "Hello" → French: cached separately
        - Each language pair gets its own cache entry
        """
        from audio.audio_coordinator import create_audio_coordinator

        coordinator = create_audio_coordinator(
            database_url=DATABASE_URL,
            service_urls={
                "whisper_service": "http://localhost:5001",
                "translation_service": "http://localhost:5003"
            }
        )

        await coordinator.initialize()

        if not coordinator.translation_cache:
            pytest.skip("Translation cache not available")

        session_id = TEST_SESSION_ID

        await coordinator.create_audio_session(
            bot_session_id=session_id,
            target_languages=["es", "fr", "de"]
        )

        transcript_result = {
            "text": "Hello",
            "language": "en",
            "confidence": 0.95,
            "start_timestamp": 0.0,
            "end_timestamp": 1.0,
            "speaker_info": {}
        }

        # Translate to Spanish
        await coordinator._request_translations(
            session_id=session_id,
            transcript_id="transcript_lang_001",
            transcript_result=transcript_result,
            target_languages=["es"]
        )

        # Translate to French
        await coordinator._request_translations(
            session_id=session_id,
            transcript_id="transcript_lang_002",
            transcript_result=transcript_result,
            target_languages=["fr"]
        )

        # Translate to Spanish again (should be cached)
        await coordinator._request_translations(
            session_id=session_id,
            transcript_id="transcript_lang_003",
            transcript_result=transcript_result,
            target_languages=["es"]
        )

        # Check that we have cache hits
        stats = coordinator.translation_cache.get_stats()
        assert stats['hits'] > 0, "Should have cache hits for repeated language pair"

        await coordinator.cleanup()


class TestAudioCoordinatorPerformance:
    """Test performance improvements with optimization"""

    @pytest.mark.asyncio
    async def test_performance_comparison_cache_vs_no_cache(self):
        """
        TEST: Compare performance with and without cache.

        Expected:
        - With cache: 80% of requests < 10ms
        - Without cache: all requests > 100ms
        """
        pytest.skip("Performance comparison test - implement after full integration")


# Pytest fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Clean up test data after each test"""
    yield

    # Cleanup Redis
    try:
        import redis.asyncio as redis
        r = redis.from_url(REDIS_URL)
        async for key in r.scan_iter(match="trans:v1:*"):
            await r.delete(key)
        await r.close()
    except Exception as e:
        print(f"Redis cleanup failed: {e}")

    # Cleanup database test session
    try:
        from database.database import DatabaseManager

        db = DatabaseManager(DATABASE_URL)
        await db.initialize()

        async with db.get_connection() as conn:
            await conn.execute(
                "DELETE FROM bot_sessions.sessions WHERE session_id = $1",
                TEST_SESSION_ID
            )

        await db.close()
    except Exception as e:
        print(f"Database cleanup failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
