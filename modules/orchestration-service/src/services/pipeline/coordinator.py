"""
Transcription Pipeline Coordinator

Generic, source-agnostic coordinator that wires all orchestration components.
Works with any transcript source via adapters.

This is the main orchestration class that:
1. Receives raw chunks from any source (Fireflies, Google Meet, etc.)
2. Adapts them to unified format
3. Aggregates chunks into sentences
4. Translates with context windows and glossary
5. Stores to database
6. Emits captions for display

All business logic (aggregation, context windows, glossary, captions)
is shared and DRY across sources.

Usage:
    coordinator = TranscriptionPipelineCoordinator(
        config=PipelineConfig(...),
        adapter=FirefliesChunkAdapter(),
        glossary_service=glossary_service,
        translation_client=translation_client,
        caption_buffer=caption_buffer,
    )

    await coordinator.initialize()

    # Process chunks from any source
    async def on_chunk(raw_chunk):
        await coordinator.process_raw_chunk(raw_chunk)
"""

import asyncio
import contextlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from livetranslate_common.logging import get_logger

# Import existing models (using Fireflies models as canonical internal format)
from models.fireflies import (
    FirefliesChunk,
    FirefliesSessionConfig,
    TranslationResult,
    TranslationUnit,
)
from services.caption_buffer import Caption, CaptionBuffer
from services.rolling_window_translator import RollingWindowTranslator

# Import existing services
from services.sentence_aggregator import SentenceAggregator

from .adapters.base import ChunkAdapter, TranscriptChunk
from .config import PipelineConfig, PipelineStats

logger = get_logger()


class TranscriptionPipelineCoordinator:
    """
    Generic coordinator for transcription -> translation -> caption pipeline.

    Works with ANY transcript source by using adapters:
    - FirefliesChunkAdapter for Fireflies WebSocket
    - GoogleMeetChunkAdapter for Google Meet browser capture
    - Future adapters for other sources

    All business logic (aggregation, context windows, glossary, captions)
    is shared and DRY across sources.
    """

    def __init__(
        self,
        config: PipelineConfig,
        adapter: ChunkAdapter,
        # Injected dependencies (shared services)
        glossary_service: Any = None,
        translation_client: Any = None,  # TranslationServiceClient
        llm_client: Any = None,  # LLMClientProtocol for prompt-based translation
        caption_buffer: CaptionBuffer | None = None,
        # Optional
        db_manager: Any = None,
        obs_output: Any = None,
        meeting_intelligence: Any = None,  # MeetingIntelligenceService
        event_publisher: Any = None,  # EventPublisher for decoupled events
    ):
        """
        Initialize the pipeline coordinator.

        Args:
            config: Pipeline configuration
            adapter: Source-specific chunk adapter
            glossary_service: Optional GlossaryService for term injection
            translation_client: Optional TranslationServiceClient (legacy)
            llm_client: Optional LLMClientProtocol for prompt-based translation
            caption_buffer: CaptionBuffer for display output
            db_manager: Optional database manager for persistence
            obs_output: Optional OBS output for streaming
            meeting_intelligence: Optional MeetingIntelligenceService
            event_publisher: Optional EventPublisher for intelligence events
        """
        self.config = config
        self.adapter = adapter

        # Shared services (DRY - same for all sources)
        self.glossary_service = glossary_service
        self.translation_client = translation_client
        self.llm_client = llm_client
        self.caption_buffer = caption_buffer
        self.db_manager = db_manager
        self.obs_output = obs_output
        self.meeting_intelligence = meeting_intelligence
        self.event_publisher = event_publisher

        # Auto-notes buffer
        self._auto_note_buffer: list[dict[str, Any]] = []

        # Internal components (created in initialize)
        self._sentence_aggregator: SentenceAggregator | None = None
        self._rolling_translator: RollingWindowTranslator | None = None
        self._glossary_terms: dict[str, str] = {}

        # Callbacks (user-settable)
        self._on_sentence_ready: Callable | None = None
        self._on_translation_ready: Callable | None = None
        self._on_caption_event: Callable | None = None
        self._on_error: Callable | None = None

        # State
        self._stats = PipelineStats()
        self._initialized = False

        # Background task tracking (prevents fire-and-forget)
        self._background_tasks: set = set()

    async def initialize(self) -> bool:
        """
        Initialize all pipeline components.

        Must be called before processing chunks.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # 1. Create FirefliesSessionConfig from PipelineConfig
            #    (SentenceAggregator uses this internally)
            fireflies_config = FirefliesSessionConfig(
                api_key="",  # Not used for aggregation
                transcript_id=self.config.transcript_id,
                target_languages=self.config.target_languages,
                pause_threshold_ms=self.config.pause_threshold_ms,
                max_buffer_words=self.config.max_words_per_sentence,
                max_buffer_seconds=self.config.max_time_per_sentence_ms / 1000.0,
                min_words_for_translation=self.config.min_words_for_translation,
                use_nlp_boundary_detection=self.config.use_nlp_boundary_detection,
                context_window_size=self.config.speaker_context_window,
                include_cross_speaker_context=self.config.include_cross_speaker_context,
                glossary_id=self.config.glossary_id,
                domain=self.config.domain,
            )

            # 2. Create sentence aggregator with callback
            self._sentence_aggregator = SentenceAggregator(
                session_id=self.config.session_id,
                transcript_id=self.config.transcript_id,
                config=fireflies_config,
                on_sentence_ready=self._handle_sentence_ready_sync,
            )

            # 3. Load glossary terms if configured
            if self.config.glossary_id and self.glossary_service:
                self._glossary_terms = await self._load_glossary()

            # 4. Create rolling window translator
            if self.translation_client or self.llm_client:
                self._rolling_translator = RollingWindowTranslator(
                    translation_client=self.translation_client,
                    glossary_service=self.glossary_service,
                    window_size=self.config.speaker_context_window,
                    include_cross_speaker_context=self.config.include_cross_speaker_context,
                    max_cross_speaker_sentences=self.config.global_context_window,
                    llm_client=self.llm_client,
                )

            # 5. Register caption callbacks if caption buffer provided
            if self.caption_buffer is not None:
                self.caption_buffer.on_caption_added = self._handle_caption_added_async
                self.caption_buffer.on_caption_updated = self._handle_caption_updated_async
                self.caption_buffer.on_caption_expired = self._handle_caption_expired_async

            self._initialized = True
            logger.info(
                f"Pipeline initialized for {self.adapter.source_type} "
                f"session {self.config.session_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            self._stats.errors += 1
            if self._on_error:
                await self._on_error(str(e))
            return False

    async def process_raw_chunk(self, raw_chunk: Any) -> None:
        """
        Process a raw chunk from ANY source.

        The adapter converts it to unified TranscriptChunk format,
        then we convert to FirefliesChunk for internal processing.

        Args:
            raw_chunk: Raw chunk data from the source API
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        self._stats.chunks_received += 1
        self._stats.last_chunk_at = datetime.now(UTC)

        try:
            # Step 1: Adapt source-specific format to unified format
            unified_chunk = self.adapter.adapt(raw_chunk)

            # Track speaker
            if unified_chunk.speaker_name:
                self._stats.record_speaker(unified_chunk.speaker_name)

            # Step 2: Convert to FirefliesChunk (internal format for SentenceAggregator)
            fireflies_chunk = self._to_fireflies_chunk(unified_chunk)

            # Step 3: Process through sentence aggregator
            #         (This triggers _handle_sentence_ready_sync callback for each sentence)
            self._sentence_aggregator.process_chunk(fireflies_chunk)

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Error processing chunk: {e}")
            if self._on_error:
                await self._on_error(str(e))

    def _to_fireflies_chunk(self, chunk: TranscriptChunk) -> FirefliesChunk:
        """
        Convert unified TranscriptChunk to FirefliesChunk.

        This allows us to reuse the existing SentenceAggregator which
        is built around FirefliesChunk.
        """
        return FirefliesChunk(
            transcript_id=chunk.transcript_id or self.config.transcript_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            speaker_name=chunk.speaker_name or "Unknown",
            start_time=chunk.start_time_seconds,
            end_time=chunk.end_time_seconds,
        )

    def _handle_sentence_ready_sync(self, unit: TranslationUnit) -> None:
        """
        Synchronous callback for SentenceAggregator.

        Wraps the async handler for compatibility with sync callback API.
        """
        try:
            # Get or create event loop
            try:
                asyncio.get_running_loop()
                # Already in async context - schedule coroutine with proper tracking
                task = asyncio.create_task(self._handle_sentence_ready(unit))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except RuntimeError:
                # No running loop - run synchronously
                asyncio.run(self._handle_sentence_ready(unit))
        except Exception as e:
            logger.error(f"Error in sentence callback: {e}")
            self._stats.errors += 1

    async def _handle_sentence_ready(self, unit: TranslationUnit) -> None:
        """
        Called when SentenceAggregator produces a complete sentence.

        This is the main pipeline processing for a sentence.
        """
        self._stats.sentences_produced += 1

        # Notify external callback
        if self._on_sentence_ready:
            try:
                await self._on_sentence_ready(unit)
            except Exception as e:
                logger.error(f"Error in on_sentence_ready callback: {e}")

        try:
            # Step 1: Store transcript to database (if configured)
            transcript_id = None
            if self.db_manager:
                transcript_id = await self._store_transcript(unit)

            # Step 2: Translate with context + glossary
            translation_result = None
            if self._rolling_translator:
                translation_result = await self._translate_sentence(unit)

            if translation_result:
                # Step 3: Store translation to database (if configured)
                if self.db_manager and transcript_id:
                    await self._store_translation(transcript_id, translation_result)

                # Step 4: Add to caption buffer (triggers display callbacks)
                if self.caption_buffer is not None:
                    self._add_caption(unit, translation_result)

                # Notify external callback
                if self._on_translation_ready:
                    try:
                        await self._on_translation_ready(unit, translation_result)
                    except Exception as e:
                        logger.error(f"Error in on_translation_ready callback: {e}")

            # Step 5: Auto-notes accumulation (non-blocking)
            if self.config.enable_auto_notes and self.meeting_intelligence:
                self._auto_note_buffer.append(
                    {
                        "text": unit.text,
                        "speaker_name": unit.speaker_name,
                        "start_time": unit.start_time,
                        "end_time": unit.end_time,
                    }
                )
                if len(self._auto_note_buffer) >= self.config.auto_notes_interval:
                    # Snapshot buffer and clear immediately so pipeline is not blocked
                    buffer_snapshot = list(self._auto_note_buffer)
                    self._auto_note_buffer.clear()
                    # Fire-and-forget: LLM call runs in background
                    task = asyncio.create_task(self._generate_auto_note(buffer_snapshot))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

        except Exception as e:
            self._stats.errors += 1
            self._stats.translations_failed += 1
            logger.error(f"Pipeline error processing sentence: {e}")
            if self._on_error:
                await self._on_error(str(e))

    async def _load_glossary(self) -> dict[str, str]:
        """Load glossary terms for the session domain."""
        if not self.glossary_service or not self.config.glossary_id:
            return {}

        try:
            # Try to parse glossary_id as UUID
            glossary_uuid = None
            with contextlib.suppress(ValueError):
                glossary_uuid = UUID(self.config.glossary_id)

            target_lang = self.config.target_languages[0] if self.config.target_languages else "es"

            terms = await self.glossary_service.get_terms(
                glossary_id=glossary_uuid,
                source_language=self.config.source_language,
                target_language=target_lang,
            )
            logger.info(f"Loaded {len(terms)} glossary terms for domain '{self.config.domain}'")
            return terms

        except Exception as e:
            logger.warning(f"Failed to load glossary: {e}")
            return {}

    async def _translate_sentence(self, unit: TranslationUnit) -> TranslationResult | None:
        """Translate using RollingWindowTranslator."""
        if not self._rolling_translator:
            return None

        try:
            target_lang = self.config.target_languages[0] if self.config.target_languages else "es"

            # Try to parse glossary_id as UUID
            glossary_uuid = None
            if self.config.glossary_id:
                with contextlib.suppress(ValueError):
                    glossary_uuid = UUID(self.config.glossary_id)

            result = await self._rolling_translator.translate(
                unit=unit,
                target_language=target_lang,
                glossary_id=glossary_uuid,
                domain=self.config.domain,
                source_language=self.config.source_language,
            )

            # Update stats
            if result:
                self._stats.record_translation(result.translation_time_ms)

            return result

        except Exception as e:
            logger.warning(f"Translation unavailable: {e}")
            self._stats.translations_failed += 1
            return None

    async def _store_transcript(self, unit: TranslationUnit) -> str | None:
        """Store transcript to database."""
        if not self.db_manager:
            return None

        try:
            return await self.db_manager.transcript_manager.store_transcript(
                session_id=self.config.session_id,
                source_type=self.adapter.source_type,  # DRY: adapter provides source
                transcript_text=unit.text,
                language_code=self.config.source_language,
                start_timestamp=unit.start_time,
                end_timestamp=unit.end_time,
                speaker_info={"speaker_name": unit.speaker_name},
                processing_metadata={
                    "boundary_type": unit.boundary_type,
                    "chunk_ids": unit.chunk_ids,
                    "source_metadata": self.config.source_metadata,
                },
            )
        except Exception as e:
            logger.error(f"Failed to store transcript: {e}")
            return None

    async def _store_translation(self, transcript_id: str, result: TranslationResult) -> str | None:
        """Store translation to database."""
        if not self.db_manager:
            return None

        try:
            return await self.db_manager.translation_manager.store_translation(
                session_id=self.config.session_id,
                source_transcript_id=transcript_id,
                translated_text=result.translated,
                source_language=result.source_language,
                target_language=result.target_language,
                translation_confidence=result.confidence,
                translation_service="translation_v3",
                speaker_info={"speaker_name": result.speaker_name},
                processing_metadata={
                    "context_sentences_used": result.context_sentences_used,
                    "glossary_terms_applied": result.glossary_terms_applied,
                    "translation_time_ms": result.translation_time_ms,
                },
            )
        except Exception as e:
            logger.error(f"Failed to store translation: {e}")
            return None

    def _add_caption(self, unit: TranslationUnit, result: TranslationResult) -> None:
        """Add translation to caption buffer."""
        if self.caption_buffer is None:
            return

        self.caption_buffer.add_caption(
            translated_text=result.translated,
            speaker_name=unit.speaker_name,
            original_text=unit.text,
            target_language=result.target_language,
            confidence=result.confidence,
        )
        self._stats.captions_displayed += 1

    # Caption callbacks - async wrappers for sync caption buffer callbacks
    async def _handle_caption_added_async(self, caption: Caption) -> None:
        """Handle caption added event."""
        # Update OBS if connected
        if self.obs_output and hasattr(self.obs_output, "is_connected"):
            if self.obs_output.is_connected:
                try:
                    await self.obs_output.update_caption(caption)
                except Exception as e:
                    logger.error(f"OBS update failed: {e}")

        # Notify external callback
        if self._on_caption_event:
            try:
                await self._on_caption_event("caption_added", caption)
            except Exception as e:
                logger.error(f"Error in caption callback: {e}")

    async def _handle_caption_updated_async(self, caption: Caption) -> None:
        """Handle caption updated event."""
        if self.obs_output and hasattr(self.obs_output, "is_connected"):
            if self.obs_output.is_connected:
                try:
                    await self.obs_output.update_caption(caption)
                except Exception as e:
                    logger.error(f"OBS update failed: {e}")

        if self._on_caption_event:
            try:
                await self._on_caption_event("caption_updated", caption)
            except Exception as e:
                logger.error(f"Error in caption callback: {e}")

    async def _handle_caption_expired_async(self, caption: Caption) -> None:
        """Handle caption expired event."""
        if self.obs_output and hasattr(self.obs_output, "is_connected"):
            if self.obs_output.is_connected:
                try:
                    await self.obs_output.clear_caption()
                except Exception as e:
                    logger.error(f"OBS clear failed: {e}")

        if self._on_caption_event:
            try:
                await self._on_caption_event("caption_expired", caption)
            except Exception as e:
                logger.error(f"Error in caption callback: {e}")

    async def _generate_auto_note(self, sentences: list[dict[str, Any]] | None = None) -> None:
        """
        Generate auto-note from sentences. Runs as a background task.

        Publishes an 'auto_note_requested' event via EventPublisher for
        observability and future external consumers. The actual LLM call
        is still handled in-process for reliability.

        Args:
            sentences: Snapshot of sentence buffer. If None, uses and clears
                       self._auto_note_buffer (for flush() calls).
        """
        if not self.meeting_intelligence:
            return

        # If no snapshot provided (called from flush), take current buffer
        if sentences is None:
            if not self._auto_note_buffer:
                return
            sentences = list(self._auto_note_buffer)
            self._auto_note_buffer.clear()

        if not sentences:
            return

        # Publish event for decoupled observability / external consumers
        if self.event_publisher:
            await self.event_publisher.publish(
                "intelligence",
                "auto_note_requested",
                {
                    "session_id": self.config.session_id,
                    "sentence_count": len(sentences),
                    "speakers": list({s.get("speaker_name", "Unknown") for s in sentences}),
                    "template": self.config.auto_notes_template,
                },
                source="pipeline-coordinator",
            )

        try:
            note = await self.meeting_intelligence.generate_auto_note(
                session_id=self.config.session_id,
                sentences=sentences,
            )
            self._stats.auto_notes_generated += 1
            logger.info(
                f"Auto-note generated for session {self.config.session_id}: "
                f"{note.get('content', '')[:80]}..."
            )

            # Publish completion event
            if self.event_publisher:
                await self.event_publisher.publish(
                    "intelligence",
                    "auto_note_generated",
                    {
                        "session_id": self.config.session_id,
                        "note_id": note.get("note_id"),
                        "sentence_count": len(sentences),
                    },
                    source="pipeline-coordinator",
                )

        except Exception as e:
            # Don't re-queue — if LLM is unavailable, retrying just creates noise.
            # Sentences are dropped; they'll be captured in future auto-notes
            # once the backend becomes available.
            if not hasattr(self, "_auto_note_warned"):
                logger.warning(
                    f"Auto-note generation unavailable ({len(sentences)} sentences dropped): {e}"
                )
                self._auto_note_warned = True
            else:
                logger.debug(
                    f"Auto-note generation still unavailable ({len(sentences)} sentences dropped)"
                )
            self._stats.errors += 1

    async def flush(self) -> None:
        """
        Flush any remaining buffered content.

        Call this at session end to ensure all text is processed.
        """
        if self._sentence_aggregator:
            remaining = self._sentence_aggregator.flush_all()
            for unit in remaining:
                await self._handle_sentence_ready(unit)

        # Flush any remaining auto-notes
        if self._auto_note_buffer and self.meeting_intelligence:
            await self._generate_auto_note()

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        stats = self._stats.to_dict()
        stats["source_type"] = self.adapter.source_type
        stats["session_id"] = self.config.session_id
        stats["initialized"] = self._initialized

        # Add aggregator stats if available
        if self._sentence_aggregator:
            stats["aggregator"] = {
                "chunks_processed": self._sentence_aggregator.chunks_processed,
                "sentences_produced": self._sentence_aggregator.sentences_produced,
                "buffers_active": len(self._sentence_aggregator.buffers),
            }

        return stats

    # Callback setters
    def on_sentence_ready(self, callback: Callable) -> "TranscriptionPipelineCoordinator":
        """Set callback for when a sentence is ready."""
        self._on_sentence_ready = callback
        return self

    def on_translation_ready(self, callback: Callable) -> "TranscriptionPipelineCoordinator":
        """Set callback for when translation is complete."""
        self._on_translation_ready = callback
        return self

    def on_caption_event(self, callback: Callable) -> "TranscriptionPipelineCoordinator":
        """Set callback for caption events (added, updated, expired)."""
        self._on_caption_event = callback
        return self

    def on_error(self, callback: Callable) -> "TranscriptionPipelineCoordinator":
        """Set callback for errors."""
        self._on_error = callback
        return self
