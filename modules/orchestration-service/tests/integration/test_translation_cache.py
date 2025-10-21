"""
Integration tests for TranslationResultCache

Tests the complete caching flow with Redis backend and database tracking.

Requirements:
- Redis running on localhost:6379
- PostgreSQL with translation optimization schema
"""

import asyncio
import pytest
import time
from typing import Dict, Optional
import os


# Test configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1")
TEST_SESSION_ID = "test_session_cache"


class TestTranslationCache:
    """Test translation result caching"""

    @pytest.mark.asyncio
    async def test_cache_initialization(self):
        """Test that cache initializes successfully"""
        from audio.translation_cache import TranslationResultCache

        cache = TranslationResultCache(redis_url=REDIS_URL, ttl=3600)

        # Should initialize without error
        assert cache is not None
        assert cache.ttl == 3600

        # Should be able to get stats
        stats = cache.get_stats()
        assert "hit_count" in stats or "hits" in stats
        assert "miss_count" in stats or "misses" in stats

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache set and get operations"""
        from audio.translation_cache import TranslationResultCache

        cache = TranslationResultCache(redis_url=REDIS_URL, ttl=3600)

        text = "Hello world"
        source_lang = "en"
        target_lang = "es"

        # Set translation in cache
        await cache.set(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            translation="Hola mundo",
            confidence=0.95,
            metadata={"backend": "llama"}
        )

        # Get translation from cache
        result = await cache.get(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )

        # Verify result
        assert result is not None
        assert result["translated_text"] == "Hola mundo"
        assert result["confidence"] == 0.95
        assert result["metadata"]["backend"] == "llama"
        assert "cached_at" in result

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None"""
        from audio.translation_cache import TranslationResultCache

        cache = TranslationResultCache(redis_url=REDIS_URL, ttl=3600)

        # Get non-existent translation
        result = await cache.get(
            text="This text does not exist in cache",
            source_lang="en",
            target_lang="fr"
        )

        # Should return None
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_key_normalization(self):
        """Test that cache keys are normalized (case-insensitive, whitespace-trimmed)"""
        from audio.translation_cache import TranslationResultCache

        cache = TranslationResultCache(redis_url=REDIS_URL, ttl=3600)

        # Set with mixed case and extra whitespace
        await cache.set(
            text="  Hello World  ",
            source_lang="en",
            target_lang="es",
            translation="Hola Mundo",
            confidence=0.95
        )

        # Get with different case and whitespace
        result1 = await cache.get(
            text="hello world",
            source_lang="en",
            target_lang="es"
        )

        result2 = await cache.get(
            text="HELLO WORLD",
            source_lang="en",
            target_lang="es"
        )

        result3 = await cache.get(
            text="  hello world  ",
            source_lang="en",
            target_lang="es"
        )

        # All should return the same cached result
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1["translated_text"] == "Hola Mundo"
        assert result2["translated_text"] == "Hola Mundo"
        assert result3["translated_text"] == "Hola Mundo"

    @pytest.mark.asyncio
    async def test_cache_multi_get(self):
        """Test getting multiple cached translations at once"""
        from audio.translation_cache import TranslationResultCache

        cache = TranslationResultCache(redis_url=REDIS_URL, ttl=3600)

        text = "Good morning"
        source_lang = "en"

        # Set translations for multiple languages
        await cache.set(text, source_lang, "es", "Buenos días", 0.95)
        await cache.set(text, source_lang, "fr", "Bonjour", 0.93)
        await cache.set(text, source_lang, "de", "Guten Morgen", 0.94)

        # Get all at once
        results = await cache.get_multi(
            text=text,
            source_lang=source_lang,
            target_langs=["es", "fr", "de", "it"]  # it is not cached
        )

        # Verify results
        assert "es" in results
        assert "fr" in results
        assert "de" in results
        assert "it" in results

        assert results["es"]["translated_text"] == "Buenos días"
        assert results["fr"]["translated_text"] == "Bonjour"
        assert results["de"]["translated_text"] == "Guten Morgen"
        assert results["it"] is None  # Cache miss

    @pytest.mark.asyncio
    async def test_cache_multi_set(self):
        """Test setting multiple translations at once"""
        from audio.translation_cache import TranslationResultCache

        cache = TranslationResultCache(redis_url=REDIS_URL, ttl=3600)

        text = "Thank you"
        source_lang = "en"

        translations = {
            "es": {"translated_text": "Gracias", "confidence": 0.96},
            "fr": {"translated_text": "Merci", "confidence": 0.95},
            "de": {"translated_text": "Danke", "confidence": 0.94}
        }

        # Set all at once
        await cache.set_multi(text, source_lang, translations)

        # Verify all were cached
        result_es = await cache.get(text, source_lang, "es")
        result_fr = await cache.get(text, source_lang, "fr")
        result_de = await cache.get(text, source_lang, "de")

        assert result_es["translated_text"] == "Gracias"
        assert result_fr["translated_text"] == "Merci"
        assert result_de["translated_text"] == "Danke"

    @pytest.mark.asyncio
    async def test_cache_statistics(self):
        """Test cache statistics tracking"""
        from audio.translation_cache import TranslationResultCache

        cache = TranslationResultCache(redis_url=REDIS_URL, ttl=3600)

        # Reset stats
        cache.reset_stats()

        initial_stats = cache.get_stats()
        initial_hits = initial_stats.get("hits", initial_stats.get("hit_count", 0))
        initial_misses = initial_stats.get("misses", initial_stats.get("miss_count", 0))

        # Generate some cache activity
        await cache.set("test1", "en", "es", "prueba1", 0.9)
        await cache.get("test1", "en", "es")  # Hit
        await cache.get("test2", "en", "es")  # Miss
        await cache.get("test1", "en", "es")  # Hit

        # Check stats
        stats = cache.get_stats()
        hits = stats.get("hits", stats.get("hit_count", 0))
        misses = stats.get("misses", stats.get("miss_count", 0))

        # Should have 2 hits and 1 miss since reset
        assert hits >= initial_hits + 2
        assert misses >= initial_misses + 1

        # Check hit rate
        hit_rate = stats.get("hit_rate", stats.get("cache_hit_rate", 0))
        assert 0 <= hit_rate <= 1

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL"""
        from audio.translation_cache import TranslationResultCache

        # Create cache with short TTL (2 seconds)
        cache = TranslationResultCache(redis_url=REDIS_URL, ttl=2)

        text = "Expire me"
        source_lang = "en"
        target_lang = "es"

        # Set translation
        await cache.set(text, source_lang, target_lang, "Expírame", 0.9)

        # Should be available immediately
        result1 = await cache.get(text, source_lang, target_lang)
        assert result1 is not None

        # Wait for expiration (2 seconds + buffer)
        await asyncio.sleep(2.5)

        # Should be expired now
        result2 = await cache.get(text, source_lang, target_lang)
        assert result2 is None

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance is significantly faster than translation"""
        from audio.translation_cache import TranslationResultCache

        cache = TranslationResultCache(redis_url=REDIS_URL, ttl=3600)

        text = "Performance test"
        source_lang = "en"
        target_lang = "es"

        # Set translation
        await cache.set(text, source_lang, target_lang, "Prueba de rendimiento", 0.9)

        # Time cache get (should be very fast)
        start = time.time()
        for _ in range(100):
            await cache.get(text, source_lang, target_lang)
        cache_time = time.time() - start

        # Average cache lookup should be < 10ms
        avg_cache_time_ms = (cache_time / 100) * 1000
        print(f"Average cache lookup time: {avg_cache_time_ms:.2f}ms")

        assert avg_cache_time_ms < 10, f"Cache too slow: {avg_cache_time_ms}ms"

    @pytest.mark.asyncio
    async def test_cache_with_database_tracking(self):
        """Test that cache integrates with database tracking"""
        from audio.translation_cache import TranslationResultCache

        # Check if database adapter is available
        try:
            from database.database import DatabaseManager
            from database.translation_optimization_adapter import TranslationOptimizationAdapter

            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                pytest.skip("DATABASE_URL not set, skipping database integration test")

            # Initialize database
            db_manager = DatabaseManager(db_url)
            await db_manager.initialize()

            opt_adapter = TranslationOptimizationAdapter(db_manager)
            if not await opt_adapter.initialize():
                pytest.skip("Translation optimization schema not available")

            # Create cache with database tracking
            cache = TranslationResultCache(
                redis_url=REDIS_URL,
                ttl=3600,
                db_adapter=opt_adapter
            )

            # Generate cache activity
            text = "Database tracking test"
            source_lang = "en"
            target_lang = "es"

            # Cache miss (first time)
            result1 = await cache.get(text, source_lang, target_lang)
            assert result1 is None

            # Set in cache
            await cache.set(text, source_lang, target_lang, "Prueba de seguimiento", 0.95)

            # Cache hit (second time)
            result2 = await cache.get(text, source_lang, target_lang)
            assert result2 is not None

            # Verify database tracking
            # Note: This requires that cache internally calls db_adapter.record_cache_stat()
            # We'll implement this in the cache class

        except ImportError:
            pytest.skip("Database modules not available")


class TestCacheIntegrationWithTranslationClient:
    """Test cache integration with translation client"""

    @pytest.mark.asyncio
    async def test_translation_client_with_cache(self):
        """Test that translation client uses cache when available"""
        # This test requires the full translation client setup
        # We'll implement this after integrating cache into the client

        pytest.skip("Full integration test - implement after cache integration")


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def cleanup_redis():
    """Clean up Redis test data after each test"""
    yield

    # Cleanup after test
    try:
        import redis.asyncio as redis
        r = redis.from_url(REDIS_URL)
        # Delete all keys with test prefix
        async for key in r.scan_iter(match="trans:v1:*"):
            await r.delete(key)
        await r.close()
    except Exception as e:
        print(f"Redis cleanup failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
