"""
Audio Coordinator Cache Integration

This file contains the updated methods for integrating translation caching
into the AudioCoordinator class.

INTEGRATION INSTRUCTIONS:
1. Add imports at top of audio_coordinator.py
2. Add cache initialization in __init__
3. Replace _request_translations method
4. Update _process_single_translation method (or remove if using new flow)
5. Add cleanup method
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# STEP 1: ADD THESE IMPORTS TO audio_coordinator.py
# =============================================================================
"""
from .translation_cache import TranslationResultCache
from database.translation_optimization_adapter import TranslationOptimizationAdapter
"""

# =============================================================================
# STEP 2: ADD TO AudioCoordinator.__init__()
# =============================================================================
"""
def __init__(self, ...):
    # ... existing initialization ...

    # Translation optimization components
    self.translation_opt_adapter = None
    self.translation_cache = None

    # Initialize translation optimization if database available
    if self.database_adapter:
        try:
            from database.translation_optimization_adapter import TranslationOptimizationAdapter
            self.translation_opt_adapter = TranslationOptimizationAdapter(
                self.database_adapter.db_manager
            )
        except ImportError:
            logger.warning("Translation optimization adapter not available")

    # Initialize translation cache
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/1")
    cache_enabled = os.getenv("TRANSLATION_CACHE_ENABLED", "true").lower() == "true"
    cache_ttl = int(os.getenv("TRANSLATION_CACHE_TTL", "3600"))  # 1 hour default

    if cache_enabled:
        try:
            from .translation_cache import TranslationResultCache
            self.translation_cache = TranslationResultCache(
                redis_url=redis_url,
                ttl=cache_ttl,
                db_adapter=self.translation_opt_adapter,
                session_id=None  # Will be set per-session
            )
            logger.info(f"Translation cache enabled: TTL={cache_ttl}s")
        except Exception as e:
            logger.warning(f"Failed to initialize translation cache: {e}")
            self.translation_cache = None
    else:
        logger.info("Translation cache disabled by configuration")
"""

# =============================================================================
# STEP 3: REPLACE _request_translations METHOD
# =============================================================================

async def _request_translations_optimized(
    self,
    session_id: str,
    transcript_id: str,
    transcript_result: Dict[str, Any],
    target_languages: List[str]
):
    """
    OPTIMIZED: Request translations with caching and multi-language batching.

    This replaces the old _request_translations method with:
    - Cache checking first
    - Multi-language translation endpoint
    - Database tracking
    - Proper error handling
    """
    source_language = transcript_result.get("language", "auto")
    text = transcript_result["text"]

    # Filter out source language
    target_langs = [lang for lang in target_languages if lang != source_language]

    if not target_langs:
        logger.debug(f"No target languages to translate to (source={source_language})")
        return

    logger.info(
        f"Requesting translations: {len(target_langs)} languages "
        f"for '{text[:50]}...' ({source_language}→{target_langs})"
    )

    # Start timing for performance tracking
    start_time = time.time()

    # STEP 1: Check cache for all target languages (if cache enabled)
    cached_results = {}
    needs_translation = []

    if self.translation_cache:
        try:
            # Set session ID for database tracking
            self.translation_cache.session_id = session_id

            # Multi-get from cache
            cache_results = await self.translation_cache.get_multi(
                text=text,
                source_lang=source_language,
                target_langs=target_langs
            )

            # Separate cached vs needs translation
            for lang in target_langs:
                if cache_results.get(lang):
                    cached_results[lang] = cache_results[lang]
                    logger.debug(f"Cache HIT: {source_language}→{lang}")
                else:
                    needs_translation.append(lang)
                    logger.debug(f"Cache MISS: {source_language}→{lang}")

        except Exception as cache_error:
            logger.error(f"Cache lookup failed: {cache_error}, proceeding without cache")
            needs_translation = target_langs
    else:
        # No cache, translate all
        needs_translation = target_langs

    # STEP 2: Translate only what's not cached (using multi-language endpoint)
    new_translations = {}

    if needs_translation:
        logger.info(
            f"Translating {len(needs_translation)} uncached languages: {needs_translation}"
        )

        try:
            # Use optimized multi-language endpoint via translation client
            if self.translation_client:
                translation_results = await self.translation_client.translate_to_multiple_languages(
                    text=text,
                    source_language=source_language,
                    target_languages=needs_translation,
                    session_id=session_id
                )

                # Convert TranslationResponse objects to dict format
                for lang, response in translation_results.items():
                    if response and hasattr(response, 'translated_text'):
                        new_translations[lang] = {
                            "translated_text": response.translated_text,
                            "confidence": response.confidence,
                            "metadata": {
                                "backend_used": response.backend_used,
                                "model_used": response.model_used,
                                "processing_time": response.processing_time
                            }
                        }
            else:
                # Fallback to service client pool
                logger.warning("Translation client not available, using service pool")
                for lang in needs_translation:
                    translation_result = await self.service_client.send_to_translation_service(
                        session_id, transcript_result, lang
                    )
                    if translation_result:
                        new_translations[lang] = translation_result

            # STEP 3: Store new translations in cache
            if self.translation_cache and new_translations:
                try:
                    await self.translation_cache.set_multi(
                        text=text,
                        source_lang=source_language,
                        translations=new_translations
                    )
                    logger.debug(f"Cached {len(new_translations)} new translations")
                except Exception as cache_error:
                    logger.error(f"Failed to cache translations: {cache_error}")

        except Exception as translation_error:
            logger.error(f"Translation failed: {translation_error}")
            self.processing_errors += 1
            return

    # STEP 4: Combine cached + new translations
    all_translations = {**cached_results, **new_translations}

    # Calculate performance metrics
    total_time_ms = (time.time() - start_time) * 1000
    cache_hits = len(cached_results)
    cache_misses = len(new_translations)
    cache_hit_rate = cache_hits / len(target_langs) if target_langs else 0

    logger.info(
        f"Translation complete: {len(all_translations)} languages in {total_time_ms:.2f}ms "
        f"(cache: {cache_hits} hits, {cache_misses} misses, {cache_hit_rate:.1%} hit rate)"
    )

    # STEP 5: Record batch metadata in database
    if self.translation_opt_adapter:
        try:
            # Generate batch ID
            batch_id = f"batch_{session_id}_{transcript_id}"

            # Record batch performance
            await self.translation_opt_adapter.record_translation_batch(
                session_id=session_id,
                source_text=text,
                source_language=source_language,
                target_languages=target_langs,
                total_time_ms=total_time_ms,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                success_count=len(all_translations),
                error_count=len(target_langs) - len(all_translations),
                results=all_translations,
                batch_id=batch_id
            )
        except Exception as db_error:
            logger.error(f"Failed to record batch in database: {db_error}")

    # STEP 6: Store each translation in database and emit events
    for target_lang, translation_data in all_translations.items():
        try:
            await self._store_and_emit_translation(
                session_id=session_id,
                transcript_id=transcript_id,
                transcript_result=transcript_result,
                target_language=target_lang,
                translation_data=translation_data,
                was_cached=(target_lang in cached_results)
            )
        except Exception as store_error:
            logger.error(f"Failed to store translation {target_lang}: {store_error}")


async def _store_and_emit_translation(
    self,
    session_id: str,
    transcript_id: str,
    transcript_result: Dict[str, Any],
    target_language: str,
    translation_data: Dict[str, Any],
    was_cached: bool
):
    """
    Store translation in database and emit event.

    Args:
        session_id: Bot session ID
        transcript_id: Transcript ID
        transcript_result: Original transcript data
        target_language: Target language code
        translation_data: Translation result
        was_cached: Whether result came from cache
    """
    # Store translation in database (if available)
    translation_id = None

    if self.database_adapter:
        translation_id = await self.database_adapter.store_translation(
            session_id,
            transcript_id,
            {
                "translated_text": translation_data.get("translated_text", ""),
                "source_language": transcript_result.get("language", "auto"),
                "target_language": target_language,
                "confidence": translation_data.get("confidence", 0.0),
                "translation_service": translation_data.get("metadata", {}).get("backend_used", "unknown"),
                "speaker_id": transcript_result.get("speaker_info", {}).get("speaker_id"),
                "speaker_name": transcript_result.get("speaker_info", {}).get("speaker_name"),
                "start_timestamp": transcript_result.get("start_timestamp", 0.0),
                "end_timestamp": transcript_result.get("end_timestamp", 0.0),
                "metadata": translation_data.get("metadata", {}),
            }
        )

        # Update translation with optimization metadata
        if translation_id and self.translation_opt_adapter:
            try:
                await self.translation_opt_adapter.update_translation_optimization_metadata(
                    translation_id=translation_id,
                    model_name=translation_data.get("metadata", {}).get("model_used"),
                    model_backend=translation_data.get("metadata", {}).get("backend_used"),
                    was_cached=was_cached,
                    optimization_metadata={
                        "was_cached": was_cached,
                        "processing_time_ms": translation_data.get("metadata", {}).get("processing_time"),
                    }
                )
            except Exception as opt_error:
                logger.error(f"Failed to update optimization metadata: {opt_error}")
    else:
        # Generate fake ID for non-persistent mode
        translation_id = f"translation_{transcript_id}_{target_language}"

    if translation_id:
        self.total_translations_generated += 1

        # Emit translation ready event
        if self.on_translation_ready:
            self.on_translation_ready({
                "session_id": session_id,
                "translation_id": translation_id,
                "transcript_id": transcript_id,
                "original_text": transcript_result["text"],
                "translated_text": translation_data.get("translated_text", ""),
                "source_language": transcript_result.get("language", "auto"),
                "target_language": target_language,
                "speaker_info": transcript_result.get("speaker_info", {}),
                "confidence": translation_data.get("confidence", 0.0),
                "start_timestamp": transcript_result.get("start_timestamp", 0.0),
                "end_timestamp": transcript_result.get("end_timestamp", 0.0),
                "was_cached": was_cached,
            })
    else:
        logger.warning(f"Failed to store translation for transcript {transcript_id}")


# =============================================================================
# STEP 4: ADD CLEANUP METHOD
# =============================================================================

async def cleanup_cache(self):
    """
    Cleanup translation cache on shutdown.
    Call this in AudioCoordinator.shutdown() or cleanup methods.
    """
    if self.translation_cache:
        try:
            # Get final stats
            stats = self.translation_cache.get_stats()
            logger.info(f"Translation cache final stats: {stats}")

            # Close Redis connection
            await self.translation_cache.close()

            logger.info("Translation cache cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up translation cache: {e}")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
# In audio_coordinator.py, update the __init__ and methods:

class AudioCoordinator:
    def __init__(self, ...):
        # ... existing code ...

        # ADD: Translation optimization components
        self.translation_opt_adapter = None
        self.translation_cache = None

        if self.database_adapter:
            try:
                from database.translation_optimization_adapter import TranslationOptimizationAdapter
                self.translation_opt_adapter = TranslationOptimizationAdapter(
                    self.database_adapter.db_manager
                )
            except ImportError:
                logger.warning("Translation optimization adapter not available")

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/1")
        cache_enabled = os.getenv("TRANSLATION_CACHE_ENABLED", "true").lower() == "true"
        cache_ttl = int(os.getenv("TRANSLATION_CACHE_TTL", "3600"))

        if cache_enabled:
            try:
                from .translation_cache import TranslationResultCache
                self.translation_cache = TranslationResultCache(
                    redis_url=redis_url,
                    ttl=cache_ttl,
                    db_adapter=self.translation_opt_adapter,
                    session_id=None
                )
                logger.info(f"Translation cache enabled: TTL={cache_ttl}s")
            except Exception as e:
                logger.warning(f"Failed to initialize translation cache: {e}")
                self.translation_cache = None

        # ... rest of initialization ...

    # REPLACE: _request_translations method with _request_translations_optimized
    async def _request_translations(self, session_id, transcript_id, transcript_result, target_languages):
        return await _request_translations_optimized(
            self, session_id, transcript_id, transcript_result, target_languages
        )

    # ADD: cleanup method
    async def cleanup(self):
        await cleanup_cache(self)
        # ... other cleanup ...
"""
