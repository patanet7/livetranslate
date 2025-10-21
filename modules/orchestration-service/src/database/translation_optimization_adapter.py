"""
Translation Optimization Database Adapter

Provides database operations for translation caching, batching, and performance tracking.
Uses the translation optimization schema (migration-translation-optimization.sql).
"""

import asyncio
import logging
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import uuid

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class TranslationOptimizationAdapter:
    """
    Database adapter for translation optimization features.

    Provides methods to:
    - Record cache statistics
    - Track translation batches
    - Update model performance metrics
    - Manage translation context
    """

    def __init__(self, database_manager: DatabaseManager):
        """Initialize with database manager."""
        self.db = database_manager

    async def initialize(self) -> bool:
        """Initialize the adapter and verify schema."""
        try:
            # Verify that optimization tables exist
            async with self.db.get_connection() as conn:
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'bot_sessions'
                        AND table_name = 'translation_cache_stats'
                    )
                """)

                if not result:
                    logger.warning(
                        "Translation optimization tables not found. "
                        "Run migration-translation-optimization.sql first."
                    )
                    return False

                logger.info("Translation optimization adapter initialized")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize translation optimization adapter: {e}")
            return False

    # =========================================================================
    # CACHE STATISTICS METHODS
    # =========================================================================

    @staticmethod
    def generate_cache_key(text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key hash from text and language pair."""
        # Normalize text (lowercase, strip whitespace)
        normalized_text = text.lower().strip()
        content = f"{source_lang}:{target_lang}:{normalized_text}"
        return hashlib.md5(content.encode()).hexdigest()

    async def record_cache_stat(
        self,
        session_id: str,
        text: str,
        source_language: str,
        target_language: str,
        was_cache_hit: bool,
        cache_latency_ms: float,
        translation_latency_ms: Optional[float] = None,
        model_used: Optional[str] = None,
        translation_service: Optional[str] = None,
        confidence: Optional[float] = None,
        quality: Optional[float] = None
    ) -> Optional[int]:
        """
        Record a cache lookup statistic.

        Args:
            session_id: Bot session ID
            text: Source text
            source_language: Source language code
            target_language: Target language code
            was_cache_hit: Whether cache hit occurred
            cache_latency_ms: Latency for cache lookup
            translation_latency_ms: Latency for translation (if miss)
            model_used: Model name
            translation_service: Service used
            confidence: Translation confidence
            quality: Translation quality score

        Returns:
            cache_stat_id or None if failed
        """
        try:
            cache_key = self.generate_cache_key(text, source_language, target_language)

            async with self.db.get_connection() as conn:
                stat_id = await conn.fetchval(
                    """
                    SELECT bot_sessions.record_cache_stat(
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                    )
                    """,
                    session_id,
                    cache_key,
                    text,  # Will be truncated to 200 chars in function
                    source_language,
                    target_language,
                    was_cache_hit,
                    cache_latency_ms,
                    translation_latency_ms,
                    model_used,
                    translation_service,
                    confidence,
                    quality
                )

                logger.debug(
                    f"Recorded cache stat {stat_id}: "
                    f"hit={was_cache_hit}, lang={source_language}→{target_language}"
                )

                return stat_id

        except Exception as e:
            logger.error(f"Failed to record cache stat: {e}")
            return None

    async def get_cache_performance(
        self,
        session_id: Optional[str] = None,
        model_used: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get cache performance statistics.

        Args:
            session_id: Filter by session ID
            model_used: Filter by model

        Returns:
            List of cache performance records
        """
        try:
            async with self.db.get_connection() as conn:
                query = "SELECT * FROM bot_sessions.cache_performance WHERE 1=1"
                params = []

                if session_id:
                    query += f" AND session_id = ${len(params) + 1}"
                    params.append(session_id)

                if model_used:
                    query += f" AND model_used = ${len(params) + 1}"
                    params.append(model_used)

                rows = await conn.fetch(query, *params)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get cache performance: {e}")
            return []

    # =========================================================================
    # TRANSLATION BATCH METHODS
    # =========================================================================

    async def record_translation_batch(
        self,
        session_id: str,
        source_text: str,
        source_language: str,
        target_languages: List[str],
        total_time_ms: float,
        cache_hits: int,
        cache_misses: int,
        model_requested: Optional[str] = None,
        success_count: int = 0,
        error_count: int = 0,
        results: Optional[Dict] = None,
        batch_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Record a multi-language translation batch.

        Args:
            session_id: Bot session ID
            source_text: Source text
            source_language: Source language code
            target_languages: List of target language codes
            total_time_ms: Total processing time
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            model_requested: Requested model name
            success_count: Number of successful translations
            error_count: Number of errors
            results: Complete results JSON
            batch_id: Optional batch ID (generated if not provided)

        Returns:
            batch_id or None if failed
        """
        try:
            if batch_id is None:
                batch_id = f"batch_{uuid.uuid4().hex[:16]}"

            async with self.db.get_connection() as conn:
                returned_batch_id = await conn.fetchval(
                    """
                    SELECT bot_sessions.record_translation_batch(
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                    )
                    """,
                    batch_id,
                    session_id,
                    source_text,
                    source_language,
                    target_languages,
                    total_time_ms,
                    cache_hits,
                    cache_misses,
                    model_requested,
                    success_count,
                    error_count,
                    json.dumps(results) if results else '{}'
                )

                logger.info(
                    f"Recorded translation batch {returned_batch_id}: "
                    f"{len(target_languages)} languages, "
                    f"hits={cache_hits}, misses={cache_misses}"
                )

                return returned_batch_id

        except Exception as e:
            logger.error(f"Failed to record translation batch: {e}")
            return None

    async def get_batch_efficiency(
        self,
        session_id: Optional[str] = None,
        model_requested: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get batch translation efficiency metrics.

        Args:
            session_id: Filter by session ID
            model_requested: Filter by model

        Returns:
            List of batch efficiency records
        """
        try:
            async with self.db.get_connection() as conn:
                query = "SELECT * FROM bot_sessions.batch_efficiency WHERE 1=1"
                params = []

                if session_id:
                    query += f" AND session_id = ${len(params) + 1}"
                    params.append(session_id)

                if model_requested:
                    query += f" AND model_requested = ${len(params) + 1}"
                    params.append(model_requested)

                rows = await conn.fetch(query, *params)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get batch efficiency: {e}")
            return []

    # =========================================================================
    # MODEL PERFORMANCE METHODS
    # =========================================================================

    async def update_model_performance(
        self,
        model_name: str,
        model_backend: str,
        source_language: str,
        target_language: str,
        latency_ms: float,
        success: bool,
        confidence: Optional[float] = None,
        was_cached: bool = False
    ) -> bool:
        """
        Update model performance metrics.

        Args:
            model_name: Model name (llama, nllb, etc.)
            model_backend: Backend (llama_transformers, etc.)
            source_language: Source language code
            target_language: Target language code
            latency_ms: Translation latency
            success: Whether translation succeeded
            confidence: Translation confidence
            was_cached: Whether result was cached

        Returns:
            True if successful
        """
        try:
            async with self.db.get_connection() as conn:
                result = await conn.fetchval(
                    """
                    SELECT bot_sessions.update_model_performance(
                        $1, $2, $3, $4, $5, $6, $7, $8
                    )
                    """,
                    model_name,
                    model_backend,
                    source_language,
                    target_language,
                    latency_ms,
                    success,
                    confidence,
                    was_cached
                )

                return result

        except Exception as e:
            logger.error(f"Failed to update model performance: {e}")
            return False

    async def get_model_comparison(
        self,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get model performance comparison.

        Args:
            source_language: Filter by source language
            target_language: Filter by target language

        Returns:
            List of model comparison records
        """
        try:
            async with self.db.get_connection() as conn:
                query = "SELECT * FROM bot_sessions.model_comparison WHERE 1=1"
                params = []

                if source_language:
                    query += f" AND source_language = ${len(params) + 1}"
                    params.append(source_language)

                if target_language:
                    query += f" AND target_language = ${len(params) + 1}"
                    params.append(target_language)

                query += " ORDER BY avg_latency_ms ASC"

                rows = await conn.fetch(query, *params)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get model comparison: {e}")
            return []

    # =========================================================================
    # TRANSLATION CONTEXT METHODS
    # =========================================================================

    async def get_or_create_translation_context(
        self,
        session_id: str,
        target_language: str,
        context_window_size: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Get or create translation context for a session and language.

        Args:
            session_id: Bot session ID
            target_language: Target language code
            context_window_size: Number of recent translations to keep

        Returns:
            Context record or None
        """
        try:
            async with self.db.get_connection() as conn:
                # Try to get existing context
                context = await conn.fetchrow(
                    """
                    SELECT * FROM bot_sessions.translation_context
                    WHERE session_id = $1 AND target_language = $2
                    """,
                    session_id,
                    target_language
                )

                if context:
                    return dict(context)

                # Create new context
                new_context = await conn.fetchrow(
                    """
                    INSERT INTO bot_sessions.translation_context (
                        session_id, target_language, context_window_size
                    ) VALUES ($1, $2, $3)
                    RETURNING *
                    """,
                    session_id,
                    target_language,
                    context_window_size
                )

                logger.info(
                    f"Created translation context for {session_id}, lang={target_language}"
                )

                return dict(new_context) if new_context else None

        except Exception as e:
            logger.error(f"Failed to get/create translation context: {e}")
            return None

    async def add_to_translation_context(
        self,
        session_id: str,
        target_language: str,
        source_text: str,
        translated_text: str
    ) -> bool:
        """
        Add a translation to the context history.

        Args:
            session_id: Bot session ID
            target_language: Target language code
            source_text: Source text
            translated_text: Translated text

        Returns:
            True if successful
        """
        try:
            async with self.db.get_connection() as conn:
                # Get current context
                context = await conn.fetchrow(
                    """
                    SELECT context_window_size, recent_translations
                    FROM bot_sessions.translation_context
                    WHERE session_id = $1 AND target_language = $2
                    """,
                    session_id,
                    target_language
                )

                if not context:
                    logger.warning(f"No context found for {session_id}, {target_language}")
                    return False

                window_size = context['context_window_size']
                recent_translations = context['recent_translations'] or []

                # Add new translation
                new_entry = {
                    "source": source_text,
                    "translation": translated_text,
                    "timestamp": datetime.utcnow().isoformat()
                }

                recent_translations.append(new_entry)

                # Keep only last N translations
                if len(recent_translations) > window_size:
                    recent_translations = recent_translations[-window_size:]

                # Update context
                await conn.execute(
                    """
                    UPDATE bot_sessions.translation_context
                    SET recent_translations = $1, last_activity = NOW()
                    WHERE session_id = $2 AND target_language = $3
                    """,
                    json.dumps(recent_translations),
                    session_id,
                    target_language
                )

                return True

        except Exception as e:
            logger.error(f"Failed to add to translation context: {e}")
            return False

    # =========================================================================
    # SESSION SUMMARY METHODS
    # =========================================================================

    async def get_session_translation_summary(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get translation summary for a session with optimization metrics.

        Args:
            session_id: Bot session ID

        Returns:
            Session summary or None
        """
        try:
            async with self.db.get_connection() as conn:
                summary = await conn.fetchrow(
                    """
                    SELECT * FROM bot_sessions.session_translation_summary
                    WHERE session_id = $1
                    """,
                    session_id
                )

                return dict(summary) if summary else None

        except Exception as e:
            logger.error(f"Failed to get session translation summary: {e}")
            return None

    # =========================================================================
    # BULK UPDATE METHODS
    # =========================================================================

    async def update_translation_optimization_metadata(
        self,
        translation_id: str,
        model_name: Optional[str] = None,
        model_backend: Optional[str] = None,
        batch_id: Optional[str] = None,
        was_cached: Optional[bool] = None,
        cache_latency_ms: Optional[float] = None,
        translation_latency_ms: Optional[float] = None,
        optimization_metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update optimization metadata for an existing translation record.

        Args:
            translation_id: Translation ID
            model_name: Model name
            model_backend: Model backend
            batch_id: Batch ID
            was_cached: Whether cached
            cache_latency_ms: Cache latency
            translation_latency_ms: Translation latency
            optimization_metadata: Additional metadata

        Returns:
            True if successful
        """
        try:
            async with self.db.get_connection() as conn:
                # Build dynamic update query
                updates = []
                params = [translation_id]
                param_idx = 2

                if model_name is not None:
                    updates.append(f"model_name = ${param_idx}")
                    params.append(model_name)
                    param_idx += 1

                if model_backend is not None:
                    updates.append(f"model_backend = ${param_idx}")
                    params.append(model_backend)
                    param_idx += 1

                if batch_id is not None:
                    updates.append(f"batch_id = ${param_idx}")
                    params.append(batch_id)
                    param_idx += 1

                if was_cached is not None:
                    updates.append(f"was_cached = ${param_idx}")
                    params.append(was_cached)
                    param_idx += 1

                if cache_latency_ms is not None:
                    updates.append(f"cache_latency_ms = ${param_idx}")
                    params.append(cache_latency_ms)
                    param_idx += 1

                if translation_latency_ms is not None:
                    updates.append(f"translation_latency_ms = ${param_idx}")
                    params.append(translation_latency_ms)
                    param_idx += 1

                if optimization_metadata is not None:
                    updates.append(f"optimization_metadata = ${param_idx}")
                    params.append(json.dumps(optimization_metadata))
                    param_idx += 1

                if not updates:
                    return True  # Nothing to update

                query = f"""
                    UPDATE bot_sessions.translations
                    SET {', '.join(updates)}
                    WHERE translation_id = $1
                """

                await conn.execute(query, *params)

                return True

        except Exception as e:
            logger.error(f"Failed to update translation optimization metadata: {e}")
            return False
