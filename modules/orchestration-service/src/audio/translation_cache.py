"""
Translation Result Cache with Redis Backend

Provides high-performance caching for translation results to reduce
redundant API calls and improve response times.

Key Features:
- Redis-backed storage with configurable TTL
- Intelligent cache key generation (normalized, case-insensitive)
- Multi-language batch operations
- Performance statistics tracking
- Database integration for analytics
- Thread-safe operations
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class TranslationResultCache:
    """
    High-performance translation result cache using Redis.

    Provides:
    - Fast cache lookups (<5ms typical)
    - Intelligent key normalization
    - Batch operations for multi-language translations
    - Hit/miss statistics tracking
    - Optional database integration for analytics
    """

    def __init__(
        self,
        redis_url: str,
        ttl: int = 3600,
        db_adapter: Any | None = None,
        session_id: str | None = None,
    ):
        """
        Initialize translation cache.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/1)
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
            db_adapter: Optional TranslationOptimizationAdapter for database tracking
            session_id: Optional session ID for database tracking
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self.db_adapter = db_adapter
        self.session_id = session_id

        # Statistics tracking
        self.hit_count = 0
        self.miss_count = 0
        self._stats_lock = asyncio.Lock()

        # Background task tracking (prevents garbage collection and enables cleanup)
        self._background_tasks: set[asyncio.Task] = set()

        # Redis connection (lazy initialization)
        self._redis: redis.Redis | None = None

        logger.info(
            f"Translation cache initialized: TTL={ttl}s, DB tracking={'enabled' if db_adapter else 'disabled'}"
        )

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle JSON encoding/decoding
            )
        return self._redis

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Translation cache closed")

    # =========================================================================
    # CACHE KEY GENERATION
    # =========================================================================

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for cache key generation.

        Normalization:
        - Convert to lowercase
        - Strip leading/trailing whitespace
        - This ensures "Hello World", "hello world", "  HELLO WORLD  " all match

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        return text.lower().strip()

    @staticmethod
    def _generate_cache_key(text: str, source_lang: str, target_lang: str) -> str:
        """
        Generate deterministic cache key from text and language pair.

        Key format: trans:v1:<md5_hash>

        Args:
            text: Source text (will be normalized)
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Cache key string
        """
        normalized_text = TranslationResultCache._normalize_text(text)
        content = f"{source_lang}:{target_lang}:{normalized_text}"
        hash_val = hashlib.md5(content.encode()).hexdigest()
        return f"trans:v1:{hash_val}"

    # =========================================================================
    # SINGLE TRANSLATION OPERATIONS
    # =========================================================================

    async def get(self, text: str, source_lang: str, target_lang: str) -> dict[str, Any] | None:
        """
        Get cached translation if exists.

        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translation result dict or None if not cached
        """
        cache_key = self._generate_cache_key(text, source_lang, target_lang)

        start_time = time.time()

        try:
            r = await self._get_redis()
            cached_data = await r.get(cache_key)

            latency_ms = (time.time() - start_time) * 1000

            if cached_data:
                # Cache hit
                async with self._stats_lock:
                    self.hit_count += 1

                result = json.loads(cached_data)

                logger.debug(
                    f"Cache HIT: {source_lang}→{target_lang} "
                    f"'{text[:50]}...' ({latency_ms:.2f}ms)"
                )

                # Track in database if adapter available
                if self.db_adapter and self.session_id:
                    task = asyncio.create_task(
                        self._record_cache_hit(text, source_lang, target_lang, latency_ms, result)
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

                return result

            else:
                # Cache miss
                async with self._stats_lock:
                    self.miss_count += 1

                logger.debug(
                    f"Cache MISS: {source_lang}→{target_lang} "
                    f"'{text[:50]}...' ({latency_ms:.2f}ms)"
                )

                # Track in database if adapter available
                if self.db_adapter and self.session_id:
                    task = asyncio.create_task(
                        self._record_cache_miss(text, source_lang, target_lang, latency_ms)
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

                return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translation: str,
        confidence: float,
        metadata: dict | None = None,
    ):
        """
        Store translation in cache.

        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code
            translation: Translated text
            confidence: Translation confidence score
            metadata: Optional metadata dict
        """
        cache_key = self._generate_cache_key(text, source_lang, target_lang)

        cache_data = {
            "translated_text": translation,
            "confidence": confidence,
            "metadata": metadata or {},
            "cached_at": datetime.now(UTC).isoformat(),
        }

        try:
            r = await self._get_redis()
            await r.setex(cache_key, self.ttl, json.dumps(cache_data))

            logger.debug(
                f"Cache SET: {source_lang}→{target_lang} " f"'{text[:50]}...' (TTL={self.ttl}s)"
            )

        except Exception as e:
            logger.error(f"Cache set error: {e}")

    # =========================================================================
    # MULTI-LANGUAGE BATCH OPERATIONS
    # =========================================================================

    async def get_multi(
        self, text: str, source_lang: str, target_langs: list[str]
    ) -> dict[str, dict[str, Any] | None]:
        """
        Get cached translations for multiple target languages at once.

        Uses Redis pipeline for efficiency.

        Args:
            text: Source text
            source_lang: Source language code
            target_langs: List of target language codes

        Returns:
            Dict mapping language code to translation result (or None if not cached)
        """
        cache_keys = [self._generate_cache_key(text, source_lang, lang) for lang in target_langs]

        start_time = time.time()

        try:
            r = await self._get_redis()

            # Use pipeline for efficient batch get
            pipe = r.pipeline()
            for key in cache_keys:
                pipe.get(key)

            results = await pipe.execute()

            latency_ms = (time.time() - start_time) * 1000

            # Parse results
            cached_translations = {}
            hits = 0
            misses = 0

            for i, target_lang in enumerate(target_langs):
                if results[i]:
                    cached_translations[target_lang] = json.loads(results[i])
                    hits += 1
                else:
                    cached_translations[target_lang] = None
                    misses += 1

            # Update stats
            async with self._stats_lock:
                self.hit_count += hits
                self.miss_count += misses

            logger.info(
                f"Cache multi-get: {len(target_langs)} languages, "
                f"{hits} hits, {misses} misses ({latency_ms:.2f}ms)"
            )

            return cached_translations

        except Exception as e:
            logger.error(f"Cache multi-get error: {e}")
            # Return all None on error
            return dict.fromkeys(target_langs)

    async def set_multi(self, text: str, source_lang: str, translations: dict[str, dict[str, Any]]):
        """
        Store multiple translations at once.

        Uses Redis pipeline for efficiency.

        Args:
            text: Source text
            source_lang: Source language code
            translations: Dict mapping language code to translation data
                         Format: {lang: {"translated_text": str, "confidence": float, ...}}
        """
        try:
            r = await self._get_redis()

            # Use pipeline for efficient batch set
            pipe = r.pipeline()

            for target_lang, translation_data in translations.items():
                cache_key = self._generate_cache_key(text, source_lang, target_lang)

                cache_data = {
                    "translated_text": translation_data.get("translated_text", ""),
                    "confidence": translation_data.get("confidence", 0.0),
                    "metadata": translation_data.get("metadata", {}),
                    "cached_at": datetime.now(UTC).isoformat(),
                }

                pipe.setex(cache_key, self.ttl, json.dumps(cache_data))

            await pipe.execute()

            logger.info(
                f"Cache multi-set: {len(translations)} languages "
                f"for '{text[:50]}...' (TTL={self.ttl}s)"
            )

        except Exception as e:
            logger.error(f"Cache multi-set error: {e}")

    # =========================================================================
    # STATISTICS & MONITORING
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, total, hit_rate, efficiency_gain
        """
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "hits": self.hit_count,
            "hit_count": self.hit_count,  # Alias for compatibility
            "misses": self.miss_count,
            "miss_count": self.miss_count,  # Alias
            "total_requests": total,
            "hit_rate": hit_rate,
            "cache_hit_rate": hit_rate,  # Alias
            "efficiency_gain": f"{hit_rate * 100:.1f}%",
        }

    def reset_stats(self):
        """Reset cache statistics counters."""
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache statistics reset")

    # =========================================================================
    # DATABASE TRACKING (OPTIONAL)
    # =========================================================================

    async def _record_cache_hit(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        latency_ms: float,
        result: dict[str, Any],
    ):
        """Record cache hit in database for analytics."""
        if not self.db_adapter or not self.session_id:
            return

        try:
            await self.db_adapter.record_cache_stat(
                session_id=self.session_id,
                text=text,
                source_language=source_lang,
                target_language=target_lang,
                was_cache_hit=True,
                cache_latency_ms=latency_ms,
                translation_latency_ms=None,
                model_used=result.get("metadata", {}).get("model_used"),
                translation_service=result.get("metadata", {}).get("backend_used"),
                confidence=result.get("confidence"),
                quality=result.get("metadata", {}).get("quality_score"),
            )
        except Exception as e:
            logger.error(f"Failed to record cache hit in database: {e}")

    async def _record_cache_miss(
        self, text: str, source_lang: str, target_lang: str, latency_ms: float
    ):
        """Record cache miss in database for analytics."""
        if not self.db_adapter or not self.session_id:
            return

        try:
            await self.db_adapter.record_cache_stat(
                session_id=self.session_id,
                text=text,
                source_language=source_lang,
                target_language=target_lang,
                was_cache_hit=False,
                cache_latency_ms=latency_ms,
                translation_latency_ms=None,  # Will be filled when translation completes
            )
        except Exception as e:
            logger.error(f"Failed to record cache miss in database: {e}")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def invalidate(self, text: str, source_lang: str, target_lang: str) -> bool:
        """
        Invalidate (delete) a cached translation.

        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            True if deleted, False otherwise
        """
        cache_key = self._generate_cache_key(text, source_lang, target_lang)

        try:
            r = await self._get_redis()
            result = await r.delete(cache_key)

            if result > 0:
                logger.info(f"Cache invalidated: {source_lang}→{target_lang}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "trans:v1:*es*")

        Returns:
            Number of keys deleted
        """
        try:
            r = await self._get_redis()
            deleted_count = 0

            async for key in r.scan_iter(match=pattern):
                await r.delete(key)
                deleted_count += 1

            logger.info(f"Cache invalidated {deleted_count} keys matching '{pattern}'")
            return deleted_count

        except Exception as e:
            logger.error(f"Cache invalidate pattern error: {e}")
            return 0

    async def get_cache_size(self) -> int:
        """
        Get approximate number of cached translations.

        Returns:
            Count of cache keys
        """
        try:
            r = await self._get_redis()
            count = 0

            async for _ in r.scan_iter(match="trans:v1:*"):
                count += 1

            return count

        except Exception as e:
            logger.error(f"Failed to get cache size: {e}")
            return 0

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"TranslationResultCache("
            f"ttl={self.ttl}s, "
            f"hits={stats['hits']}, "
            f"misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )
