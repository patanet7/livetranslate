"""
Fireflies Router

FastAPI router for Fireflies.ai integration.
Provides endpoints for:
- Connecting to Fireflies realtime transcription
- Managing active Fireflies sessions
- Querying active meetings
- Session status and statistics

All transcripts are stored in the existing bot_sessions database
with source_type='fireflies'.
"""

import asyncio
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from clients.fireflies_client import (
    FirefliesAPIError,
    FirefliesClient,
    FirefliesGraphQLClient,
    FirefliesRateLimitError,
)
from dependencies import (
    get_data_pipeline,
    get_event_publisher,
    get_fireflies_llm_client,
    get_glossary_pipeline_adapter,
    get_meeting_intelligence_service,
    get_translation_service_client,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from livetranslate_common.logging import get_logger
from models.fireflies import (
    ActiveMeetingsResponse,
    FirefliesChunk,
    FirefliesConnectionStatus,
    FirefliesConnectResponse,
    FirefliesSession,
    FirefliesSessionConfig,
)
from pydantic import BaseModel, Field
from routers.captions import get_connection_manager as get_ws_manager
from services.caption_buffer import CaptionBuffer
from services.meeting_store import MeetingStore

# Pipeline imports (DRY coordinator)
from services.pipeline import (
    FirefliesChunkAdapter,
    ImportChunkAdapter,
    PipelineConfig,
    TranscriptionPipelineCoordinator,
)
from services.pipeline.command_interceptor import CommandInterceptor
from services.pipeline.live_caption_manager import LiveCaptionManager

from config import FirefliesSettings, get_settings

logger = get_logger()

# Create router
router = APIRouter(prefix="/fireflies", tags=["fireflies"])

# Auto-connect polling state
_auto_connect_task: asyncio.Task[None] | None = None
_known_meeting_ids: set[str] = set()


# =============================================================================
# Session Manager (In-Memory State)
# =============================================================================


class FirefliesSessionManager:
    """
    Manages active Fireflies sessions.

    Coordinates between:
    - FirefliesClient (API communication)
    - Database (persistence)
    - Translation pipeline (processing)
    """

    def __init__(self):
        self._sessions: dict[str, FirefliesSession] = {}
        self._clients: dict[str, FirefliesClient] = {}
        self._transcript_handlers: dict[str, Any] = {}
        # Pipeline coordinators (DRY - same for all sources)
        self._coordinators: dict[str, TranscriptionPipelineCoordinator] = {}
        self._caption_buffers: dict[str, CaptionBuffer] = {}
        # Meeting persistence (lazily initialized)
        self._meeting_store: MeetingStore | None = None
        # Runtime translation backend configuration (hot-swappable)
        self.translation_config: dict[str, Any] | None = None

    async def _get_meeting_store(self) -> MeetingStore | None:
        """Get or create MeetingStore instance (lazy initialization).

        Returns None if DATABASE_URL is not configured, allowing the system
        to run without persistence.
        """
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            return None
        if self._meeting_store is None:
            self._meeting_store = MeetingStore(db_url)
            await self._meeting_store.initialize()
        return self._meeting_store

    def get_session(self, session_id: str) -> FirefliesSession | None:
        """Get session by ID"""
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> list[FirefliesSession]:
        """Get all active sessions"""
        return list(self._sessions.values())

    def get_client(self, session_id: str) -> FirefliesClient | None:
        """Get Fireflies client for session"""
        return self._clients.get(session_id)

    async def create_session(
        self,
        config: FirefliesSessionConfig,
        on_transcript=None,
        on_status_change=None,
        on_error=None,
        # Optional service injections for pipeline
        glossary_service=None,
        translation_client=None,
        llm_client=None,
        db_manager=None,
        obs_output=None,
        meeting_intelligence=None,
        event_publisher=None,
        meeting_title: str | None = None,
    ) -> FirefliesSession:
        """Create and connect a new Fireflies session with full pipeline"""

        session_id = f"ff_session_{uuid.uuid4().hex[:12]}"

        # Create session record
        session = FirefliesSession(
            session_id=session_id,
            fireflies_transcript_id=config.transcript_id,
            config=config,
            connection_status=FirefliesConnectionStatus.CONNECTING,
        )

        # Create Fireflies client (use api_base_url override for demo mode)
        if config.api_base_url:
            client = FirefliesClient(
                api_key=config.api_key,
                graphql_endpoint=f"{config.api_base_url}/graphql",
                websocket_endpoint=config.api_base_url,
                socketio_path="/ws/realtime",
            )
        else:
            client = FirefliesClient(api_key=config.api_key)

        # Store session and client
        self._sessions[session_id] = session
        self._clients[session_id] = client

        # Create meeting record in DB if persistence is enabled
        if os.environ.get("MEETING_AUTO_SAVE", "true").lower() == "true":
            meeting_store = await self._get_meeting_store()
            if meeting_store:
                try:
                    meeting_id = await meeting_store.create_meeting(
                        fireflies_transcript_id=config.transcript_id,
                        title=meeting_title,
                        target_languages=config.target_languages,
                        meeting_metadata={
                            "fireflies_transcript_id": config.transcript_id,
                            "ingest_mode": "realtime",
                        },
                        source="fireflies",
                    )
                    session.meeting_db_id = meeting_id
                    logger.info(
                        "meeting_record_created",
                        meeting_id=meeting_id,
                        transcript_id=config.transcript_id,
                    )
                except Exception as e:
                    logger.error(
                        "meeting_record_creation_failed",
                        transcript_id=config.transcript_id,
                        error=str(e),
                    )
                    # Fail hard: if we can't persist, don't start a session that silently loses data
                    raise RuntimeError(
                        f"Cannot start session: DB persistence is enabled but meeting "
                        f"record creation failed: {e}"
                    ) from e

        # =====================================================================
        # Pipeline Setup (DRY - shared coordinator for all sources)
        # =====================================================================

        # Create caption buffer for this session
        caption_buffer = CaptionBuffer(
            max_captions=5,
            default_duration=4.0,
        )
        self._caption_buffers[session_id] = caption_buffer

        # Create pipeline config from Fireflies config
        pipeline_config = PipelineConfig(
            session_id=session_id,
            source_type="fireflies",
            transcript_id=config.transcript_id,
            target_languages=config.target_languages,
            pause_threshold_ms=config.pause_threshold_ms,
            max_words_per_sentence=config.max_buffer_words,
            max_time_per_sentence_ms=config.max_buffer_seconds * 1000,
            min_words_for_translation=config.min_words_for_translation,
            use_nlp_boundary_detection=config.use_nlp_boundary_detection,
            speaker_context_window=config.context_window_size,
            include_cross_speaker_context=config.include_cross_speaker_context,
            glossary_id=config.glossary_id,
            domain=config.domain or "general",
            source_metadata={"fireflies_transcript_id": config.transcript_id},
        )

        # Configure auto-notes if meeting_intelligence is provided
        if meeting_intelligence:
            from config import get_settings

            intel_settings = get_settings().intelligence
            pipeline_config.enable_auto_notes = intel_settings.auto_notes_enabled
            pipeline_config.auto_notes_interval = intel_settings.auto_notes_interval_sentences
            pipeline_config.auto_notes_template = intel_settings.auto_notes_template

        # Create coordinator with Fireflies adapter
        coordinator = TranscriptionPipelineCoordinator(
            config=pipeline_config,
            adapter=FirefliesChunkAdapter(),
            glossary_service=glossary_service,
            translation_client=translation_client,
            llm_client=llm_client,
            caption_buffer=caption_buffer,
            db_manager=db_manager,
            obs_output=obs_output,
            meeting_intelligence=meeting_intelligence,
            event_publisher=event_publisher,
        )

        # Initialize the pipeline
        await coordinator.initialize()
        self._coordinators[session_id] = coordinator

        # Bridge captions to WebSocket clients (captions.html)
        ws_manager = get_ws_manager()

        # LiveCaptionManager: config-driven display filtering
        live_caption_mgr = LiveCaptionManager(
            config=pipeline_config,
            broadcast=ws_manager.broadcast_to_session,
            session_id=session_id,
        )
        coordinator.on_caption_event(live_caption_mgr.handle_caption_event)

        # Persist completed sentences to DB during live sessions
        if session.meeting_db_id:
            _meeting_db_id = session.meeting_db_id  # capture for closure

            async def _persist_sentence(unit) -> None:
                """Store completed sentence from pipeline to meeting DB."""
                meeting_store = await self._get_meeting_store()
                if meeting_store:
                    try:
                        sentence_id = await meeting_store.store_sentence(
                            meeting_id=_meeting_db_id,
                            text=unit.text,
                            speaker_name=unit.speaker_name,
                            start_time=unit.start_time,
                            end_time=unit.end_time,
                            boundary_type=unit.boundary_type,
                            chunk_ids=unit.chunk_ids,
                        )
                        setattr(unit, "_meeting_sentence_id", sentence_id)
                    except Exception as e:
                        session.persistence_failures += 1
                        session.persistence_healthy = False
                        session.error_count += 1
                        session.last_error = f"sentence_storage_failed: {e}"
                        logger.error(
                            "sentence_storage_failed",
                            meeting_id=_meeting_db_id,
                            session_id=session_id,
                            persistence_failures=session.persistence_failures,
                            error=str(e),
                        )

            coordinator.on_sentence_ready(_persist_sentence)

            async def _persist_translation(unit, result) -> None:
                """Store pipeline translations in canonical meeting_translations."""
                meeting_store = await self._get_meeting_store()
                sentence_id = getattr(unit, "_meeting_sentence_id", None)
                if meeting_store is None or sentence_id is None:
                    return
                try:
                    await meeting_store.store_translation(
                        sentence_id=sentence_id,
                        translated_text=result.translated,
                        target_language=result.target_language,
                        source_language=result.source_language,
                        confidence=result.confidence,
                        translation_time_ms=result.translation_time_ms,
                        model_used=getattr(result, "model_used", None),
                    )
                except Exception as e:
                    session.persistence_failures += 1
                    session.persistence_healthy = False
                    session.error_count += 1
                    session.last_error = f"translation_storage_failed: {e}"
                    logger.error(
                        "translation_storage_failed",
                        meeting_id=_meeting_db_id,
                        session_id=session_id,
                        persistence_failures=session.persistence_failures,
                        error=str(e),
                    )

            coordinator.on_translation_ready(_persist_translation)

        # CommandInterceptor: voice command detection (config-driven)
        command_interceptor = CommandInterceptor(
            config=pipeline_config,
            coordinator=coordinator,
            ws_broadcast=ws_manager.broadcast_to_session,
            session_id=session_id,
        )

        logger.info("pipeline_coordinator_initialized", session_id=session_id)

        # =====================================================================
        # Connect to realtime with callbacks
        # =====================================================================

        async def handle_transcript(chunk: FirefliesChunk):
            session.chunks_received += 1
            session.last_chunk_time = datetime.now(UTC)
            session.last_chunk_id = chunk.chunk_id

            # Track unique speakers
            if chunk.speaker_name not in session.speakers_detected:
                session.speakers_detected.append(chunk.speaker_name)

            # CommandInterceptor: check for voice commands before pipeline processing
            if command_interceptor.check(chunk.text):
                await command_interceptor.execute(chunk.text)
                return  # Don't process commands through the pipeline

            # Process through pipeline coordinator (DRY - handles all orchestration)
            try:
                await coordinator.process_raw_chunk(chunk)
                # Update session stats from coordinator
                stats = coordinator.get_stats()
                session.sentences_produced = stats.get("sentences_produced", 0)
                session.translations_completed = stats.get("translations_completed", 0)
            except Exception as e:
                logger.error("pipeline_processing_error", session_id=session_id, error=str(e))
                session.error_count += 1

            # Store chunk to DB if persistence is enabled (non-blocking)
            if session.meeting_db_id:
                meeting_store = await self._get_meeting_store()
                if meeting_store:
                    try:
                        await meeting_store.store_chunk(
                            meeting_id=session.meeting_db_id,
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            speaker_name=chunk.speaker_name,
                            start_time=chunk.start_time,
                            end_time=chunk.end_time,
                        )
                    except Exception as e:
                        session.persistence_failures += 1
                        session.persistence_healthy = False
                        session.error_count += 1
                        session.last_error = f"chunk_storage_failed: {e}"
                        logger.error(
                            "chunk_storage_failed",
                            meeting_id=session.meeting_db_id,
                            session_id=session_id,
                            chunk_id=chunk.chunk_id,
                            persistence_failures=session.persistence_failures,
                            error=str(e),
                        )

            # Call user callback (after pipeline processing)
            if on_transcript:
                await on_transcript(session_id, chunk)

        async def handle_status(new_status: FirefliesConnectionStatus, message: str | None):
            session.connection_status = new_status
            if new_status == FirefliesConnectionStatus.CONNECTED:
                session.connected_at = datetime.now(UTC)
            elif new_status == FirefliesConnectionStatus.ERROR:
                session.error_count += 1
                session.last_error = message

            if on_status_change:
                await on_status_change(session_id, new_status, message)

        async def handle_error(message: str, exception: Exception | None):
            session.error_count += 1
            session.last_error = message

            if on_error:
                await on_error(session_id, message, exception)

        try:
            await client.connect_realtime(
                transcript_id=config.transcript_id,
                on_transcript=handle_transcript,
                on_status_change=handle_status,
                on_error=handle_error,
                on_live_update=live_caption_mgr.handle_interim_update,
                auto_reconnect=True,
            )
        except Exception as e:
            session.connection_status = FirefliesConnectionStatus.ERROR
            session.last_error = str(e)
            logger.error("fireflies_session_connect_failed", session_id=session_id, error=str(e))

        return session

    async def disconnect_session(self, session_id: str) -> bool:
        """Disconnect and remove a session"""
        if session_id not in self._sessions:
            return False

        # Flush pipeline coordinator before disconnecting
        if session_id in self._coordinators:
            coordinator = self._coordinators.pop(session_id)
            try:
                await coordinator.flush()
                logger.info("pipeline_flushed", session_id=session_id)
            except Exception as e:
                logger.error("pipeline_flush_failed", session_id=session_id, error=str(e))

        # Clean up caption buffer
        if session_id in self._caption_buffers:
            self._caption_buffers.pop(session_id)

        # Disconnect client
        if session_id in self._clients:
            client = self._clients.pop(session_id)
            await client.close()

        # Update session status
        session = self._sessions.pop(session_id)
        session.connection_status = FirefliesConnectionStatus.DISCONNECTED

        # Mark meeting as complete in DB if persistence was active
        if session.meeting_db_id:
            meeting_store = await self._get_meeting_store()
            if meeting_store:
                try:
                    await meeting_store.complete_meeting(session.meeting_db_id)
                    logger.info(
                        "meeting_completed",
                        meeting_id=session.meeting_db_id,
                        session_id=session_id,
                    )
                except Exception as e:
                    session.persistence_failures += 1
                    session.persistence_healthy = False
                    logger.error(
                        "meeting_completion_failed",
                        meeting_id=session.meeting_db_id,
                        session_id=session_id,
                        error=str(e),
                    )

        if not session.persistence_healthy:
            logger.warning(
                "session_ended_with_persistence_failures",
                session_id=session_id,
                persistence_failures=session.persistence_failures,
            )

        logger.info("fireflies_session_disconnected", session_id=session_id)
        return True

    def get_coordinator(self, session_id: str) -> TranscriptionPipelineCoordinator | None:
        """Get pipeline coordinator for session"""
        return self._coordinators.get(session_id)

    def get_caption_buffer(self, session_id: str) -> CaptionBuffer | None:
        """Get caption buffer for session"""
        return self._caption_buffers.get(session_id)

    async def create_import_session(
        self,
        transcript_id: str,
        transcript_title: str,
        target_languages: list[str],
        meeting_db_id: str | None = None,
        glossary_id: str | None = None,
        domain: str | None = None,
        db_manager=None,
        translation_client=None,
        llm_client=None,
    ) -> tuple[str, TranscriptionPipelineCoordinator]:
        """
        Create a session for importing historical transcripts.

        Uses the same pipeline as live sessions (DRY) but without real-time connection.
        Sentences are fed through the coordinator manually.

        Args:
            transcript_id: Original transcript ID (e.g., Fireflies ID)
            transcript_title: Title for the session
            target_languages: Languages to translate to
            glossary_id: Optional glossary for translation
            domain: Optional domain for glossary filtering
            db_manager: Database manager for persistence
            translation_client: Translation service client
            llm_client: LLMClientProtocol for prompt-based translation

        Returns:
            Tuple of (session_id, coordinator) for manual chunk processing
        """
        session_id = f"import_{uuid.uuid4().hex[:12]}"

        # Create caption buffer (even for import, for consistency)
        caption_buffer = CaptionBuffer(
            max_captions=5,
            default_duration=4.0,
        )
        self._caption_buffers[session_id] = caption_buffer

        # Create pipeline config
        pipeline_config = PipelineConfig(
            session_id=session_id,
            source_type="fireflies_import",
            transcript_id=transcript_id,
            target_languages=target_languages,
            # Import uses relaxed aggregation since sentences are already formed
            pause_threshold_ms=100,  # Low threshold for import
            max_words_per_sentence=1000,  # High limit - sentences already segmented
            max_time_per_sentence_ms=60000,  # 60s - already segmented
            min_words_for_translation=1,  # Translate everything
            use_nlp_boundary_detection=False,  # Not needed for import
            speaker_context_window=3,  # Use context for better translation
            include_cross_speaker_context=True,
            glossary_id=glossary_id,
            domain=domain or "general",
            source_metadata={
                "fireflies_transcript_id": transcript_id,
                "import_type": "fireflies",
                "title": transcript_title,
            },
        )

        # Create coordinator with import adapter
        coordinator = TranscriptionPipelineCoordinator(
            config=pipeline_config,
            adapter=ImportChunkAdapter(source_name="fireflies_import"),
            translation_client=translation_client,
            llm_client=llm_client,
            caption_buffer=caption_buffer,
            db_manager=db_manager,
        )

        # Initialize the pipeline
        await coordinator.initialize()
        self._coordinators[session_id] = coordinator

        # Bridge captions to WebSocket clients (captions.html)
        ws_manager = get_ws_manager()

        async def _broadcast_import_caption_to_ws(event_type: str, caption) -> None:
            if event_type == "caption_expired":
                await ws_manager.broadcast_to_session(
                    session_id,
                    {"event": "caption_expired", "caption_id": caption.id},
                )
            else:
                await ws_manager.broadcast_to_session(
                    session_id,
                    {"event": event_type, "caption": caption.to_dict()},
                    target_language=caption.target_language,
                )

        coordinator.on_caption_event(_broadcast_import_caption_to_ws)

        if meeting_db_id:

            async def _persist_sentence(unit) -> None:
                meeting_store = await self._get_meeting_store()
                if meeting_store is None:
                    return
                sentence_id = await meeting_store.store_sentence(
                    meeting_id=meeting_db_id,
                    text=unit.text,
                    speaker_name=unit.speaker_name,
                    start_time=unit.start_time,
                    end_time=unit.end_time,
                    boundary_type=unit.boundary_type,
                    chunk_ids=unit.chunk_ids,
                )
                setattr(unit, "_meeting_sentence_id", sentence_id)

            async def _persist_translation(unit, result) -> None:
                meeting_store = await self._get_meeting_store()
                sentence_id = getattr(unit, "_meeting_sentence_id", None)
                if meeting_store is None or sentence_id is None:
                    return
                await meeting_store.store_translation(
                    sentence_id=sentence_id,
                    translated_text=result.translated,
                    target_language=result.target_language,
                    source_language=result.source_language,
                    confidence=result.confidence,
                    translation_time_ms=result.translation_time_ms,
                    model_used=getattr(result, "model_used", None),
                )

            coordinator.on_sentence_ready(_persist_sentence)
            coordinator.on_translation_ready(_persist_translation)

        logger.info("import_session_created", session_id=session_id, transcript_id=transcript_id)

        return session_id, coordinator

    async def finalize_import_session(self, session_id: str) -> dict[str, Any]:
        """
        Finalize an import session after all sentences have been processed.

        Flushes the coordinator and returns final stats.

        Args:
            session_id: The import session ID

        Returns:
            Final statistics from the pipeline
        """
        coordinator = self._coordinators.get(session_id)
        if not coordinator:
            return {"error": f"Session {session_id} not found"}

        # Flush any remaining buffered content
        await coordinator.flush()

        # Get final stats
        stats = coordinator.get_stats()

        # Clean up
        self._coordinators.pop(session_id, None)
        self._caption_buffers.pop(session_id, None)

        logger.info("import_session_finalized", session_id=session_id, stats=stats)

        return stats


# Global session manager instance
_session_manager: FirefliesSessionManager | None = None


def get_session_manager() -> FirefliesSessionManager:
    """Get or create the session manager singleton"""
    global _session_manager
    if _session_manager is None:
        _session_manager = FirefliesSessionManager()
    return _session_manager


def get_fireflies_config() -> FirefliesSettings:
    """Get Fireflies configuration from settings"""
    return get_settings().fireflies


def get_api_key_from_config() -> str | None:
    """Get API key from config if available"""
    config = get_fireflies_config()
    return config.api_key if config.has_api_key() else None


# =============================================================================
# Auto-Connect Polling Loop
# =============================================================================


async def _auto_connect_loop(manager: FirefliesSessionManager) -> None:
    """Poll for active Fireflies meetings and auto-connect new ones.

    Runs continuously until cancelled.  On each poll cycle:
    - Fetches the active meeting list from Fireflies GraphQL.
    - Connects to any meeting not yet tracked in ``_known_meeting_ids``.
    - Triggers ``_download_meeting_data`` for meetings that disappeared
      (i.e. ended) since the last poll.

    Environment variables:
        FIREFLIES_API_KEY         - Required. Fireflies API key.
        FIREFLIES_POLL_INTERVAL   - Seconds between polls (default 30).
        DEFAULT_TARGET_LANGUAGE   - Target language for auto-sessions (default "zh").
        MEETING_DOWNLOAD_ON_COMPLETE - "true" to download full data on end (default "true").
    """
    global _known_meeting_ids

    api_key = os.environ.get("FIREFLIES_API_KEY", "")
    poll_interval = int(os.environ.get("FIREFLIES_POLL_INTERVAL", "30"))
    target_language = os.environ.get("DEFAULT_TARGET_LANGUAGE", "zh")

    logger.info("auto_connect_loop_started", poll_interval=poll_interval)

    while True:
        try:
            client = FirefliesGraphQLClient(api_key=api_key)
            try:
                meetings = await client.get_active_meetings()
            finally:
                await client.close()

            current_ids = {m.id for m in meetings} if meetings else set()

            # New meetings: auto-connect
            for meeting in meetings or []:
                if meeting.id not in _known_meeting_ids:
                    logger.info(
                        "auto_connect_new_meeting",
                        meeting_id=meeting.id,
                        title=meeting.title,
                    )
                    try:
                        config = FirefliesSessionConfig(
                            api_key=api_key,
                            transcript_id=meeting.id,
                            target_languages=[target_language],
                        )

                        # Get optional services for the pipeline
                        data_pipeline = get_data_pipeline()
                        db_manager = data_pipeline.db_manager if data_pipeline else None

                        llm_client = None
                        try:
                            llm_client = get_fireflies_llm_client()
                            await llm_client.connect()
                        except Exception as e:
                            logger.warning(
                                "auto_connect_llm_client_unavailable",
                                meeting_id=meeting.id,
                                error=str(e),
                            )

                        try:
                            meeting_intelligence = get_meeting_intelligence_service()
                        except Exception as e:
                            meeting_intelligence = None
                            logger.warning(
                                "auto_connect_meeting_intelligence_unavailable",
                                meeting_id=meeting.id,
                                error=str(e),
                            )

                        try:
                            event_publisher = get_event_publisher()
                        except Exception as e:
                            event_publisher = None
                            logger.warning(
                                "auto_connect_event_publisher_unavailable",
                                meeting_id=meeting.id,
                                error=str(e),
                            )

                        glossary_service = None
                        try:
                            glossary_service = get_glossary_pipeline_adapter()
                        except Exception as e:
                            logger.warning(
                                "auto_connect_glossary_unavailable",
                                meeting_id=meeting.id,
                                error=str(e),
                            )

                        await manager.create_session(
                            config,
                            db_manager=db_manager,
                            llm_client=llm_client,
                            meeting_intelligence=meeting_intelligence,
                            event_publisher=event_publisher,
                            glossary_service=glossary_service,
                            meeting_title=meeting.title,
                        )
                    except Exception as exc:
                        logger.error(
                            "auto_connect_failed",
                            meeting_id=meeting.id,
                            error=str(exc),
                        )

            # Ended meetings: trigger full download
            ended_ids = _known_meeting_ids - current_ids
            for ended_id in ended_ids:
                logger.info("auto_connect_meeting_ended", meeting_id=ended_id)
                if os.environ.get("MEETING_DOWNLOAD_ON_COMPLETE", "true").lower() == "true":
                    asyncio.create_task(_download_meeting_data(ended_id))

            _known_meeting_ids = current_ids

        except Exception as exc:
            logger.error("auto_connect_poll_error", error=str(exc))

        await asyncio.sleep(poll_interval)


async def start_auto_connect() -> None:
    """Start auto-connect polling if FIREFLIES_AUTO_CONNECT=true.

    Called from the app lifespan in main_fastapi.py.
    """
    global _auto_connect_task

    if os.environ.get("FIREFLIES_AUTO_CONNECT", "false").lower() != "true":
        return

    api_key = os.environ.get("FIREFLIES_API_KEY")
    if not api_key:
        logger.warning("fireflies_auto_connect_no_api_key")
        return

    manager = get_session_manager()
    _auto_connect_task = asyncio.create_task(_auto_connect_loop(manager))
    logger.info("fireflies_auto_connect_enabled")


async def stop_auto_connect() -> None:
    """Cancel the auto-connect polling task on shutdown.

    Called from the app lifespan in main_fastapi.py.
    """
    global _auto_connect_task

    if _auto_connect_task is not None:
        _auto_connect_task.cancel()
        try:
            await _auto_connect_task
        except asyncio.CancelledError:
            pass
        _auto_connect_task = None
        logger.info("fireflies_auto_connect_stopped")


# =============================================================================
# Boot-time Sync
# =============================================================================


async def boot_sync_fireflies() -> None:
    """Sync all Fireflies transcripts on first boot of the day.

    Called from the app lifespan in main_fastapi.py.
    Checks FIREFLIES_BOOT_SYNC env var (default: true).
    Runs once per day — subsequent restarts (including --reload) skip it
    by checking a DB timestamp. Uses the bulk query to avoid N+1.
    Non-fatal — if anything fails, the service still starts.
    """
    if os.environ.get("FIREFLIES_BOOT_SYNC", "true").lower() != "true":
        logger.info("fireflies_boot_sync_disabled")
        return

    api_key = os.environ.get("FIREFLIES_API_KEY")
    db_url = os.environ.get("DATABASE_URL")
    if not api_key or not db_url:
        logger.info(
            "fireflies_boot_sync_skipped",
            has_api_key=bool(api_key),
            has_db_url=bool(db_url),
        )
        return

    # Check if we already synced today (once-per-day guard)
    store = MeetingStore(db_url)
    await store.initialize()
    try:
        last_sync = await store._pool.fetchval(
            """SELECT value FROM system_config WHERE key = 'last_boot_sync_at'"""
        )
        if last_sync:
            from datetime import timedelta

            try:
                last_sync_time = datetime.fromisoformat(last_sync)
                if datetime.now(UTC) - last_sync_time < timedelta(hours=24):
                    logger.info(
                        "fireflies_boot_sync_skipped_recent",
                        last_sync=last_sync,
                    )
                    return
            except (ValueError, TypeError):
                pass  # Corrupted value — proceed with sync
    except Exception:
        # system_config table may not exist yet — create it
        await store._pool.execute(
            """CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )"""
        )

    logger.info("fireflies_boot_sync_starting")

    # Use bulk query — 1 API call per 50 transcripts (with full data)
    client = FirefliesGraphQLClient(api_key=api_key)
    all_transcripts: list[dict[str, Any]] = []
    try:
        skip = 0
        page_size = 50
        while True:
            page = await client.get_full_transcripts(limit=page_size, skip=skip)
            if not page:
                break
            all_transcripts.extend(page)
            if len(page) < page_size:
                break
            skip += page_size
    except FirefliesRateLimitError as e:
        logger.warning(
            "fireflies_boot_sync_rate_limited",
            retry_after=e.retry_after,
            fetched_so_far=len(all_transcripts),
        )
        # Process whatever we fetched before hitting the limit
    finally:
        await client.close()

    if not all_transcripts:
        logger.info("fireflies_boot_sync_no_transcripts")
        # Still record the sync attempt to avoid retrying on every restart
        try:
            await store._pool.execute(
                """INSERT INTO system_config (key, value, updated_at)
                   VALUES ('last_boot_sync_at', $1, NOW())
                   ON CONFLICT (key) DO UPDATE SET value = $1, updated_at = NOW()""",
                datetime.now(UTC).isoformat(),
            )
        except Exception:
            pass
        await store.close()
        return

    # Process transcripts inline (no individual API calls!)
    new_count = 0
    updated_count = 0
    already_synced = 0
    errors = 0
    try:
        for transcript in all_transcripts:
            ff_id = transcript.get("id")
            if not ff_id:
                continue
            meeting = await store.get_meeting_by_ff_id(ff_id)
            if meeting and meeting.get("insight_count", 0) > 0:
                already_synced += 1
                continue

            # Store directly from bulk response — no extra API call needed
            try:
                await _store_transcript_to_db(transcript, store)
                if meeting:
                    updated_count += 1
                else:
                    new_count += 1

                # Auto-trigger diarization if rules match
                try:
                    from services.diarization.auto_trigger import maybe_trigger_diarization
                    from services.diarization.db import get_diarization_rules
                    from models.diarization import DiarizationRules
                    from database.database import get_database_manager
                    _synced_meeting = await store.get_meeting_by_ff_id(ff_id)
                    if _synced_meeting:
                        _meeting_meta = {
                            "id": _synced_meeting["id"],
                            "title": transcript.get("title", ""),
                            "participants": transcript.get("participants") or [],
                            "duration": int((transcript.get("duration") or 0)),
                            "sentence_count": len(transcript.get("sentences") or []),
                        }
                        async with get_database_manager().get_session() as _db:
                            _rules_dict = await get_diarization_rules(_db)
                        _rules = DiarizationRules(**_rules_dict)
                        await maybe_trigger_diarization(
                            _meeting_meta, _rules, None
                        )
                except Exception:
                    pass  # Diarization not available — skip silently
            except Exception as e:
                errors += 1
                # Mark failed with retry tracking (exponential backoff)
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                try:
                    stuck = meeting or await store.get_meeting_by_ff_id(ff_id)
                    if stuck:
                        await store.mark_sync_failed(
                            str(stuck["id"]), str(e), is_rate_limit=is_rate_limit,
                        )
                except Exception:
                    pass  # best-effort status update
                logger.warning(
                    "boot_sync_transcript_failed",
                    ff_id=ff_id,
                    exc_info=True,
                )

        # Record successful sync timestamp
        await store._pool.execute(
            """INSERT INTO system_config (key, value, updated_at)
               VALUES ('last_boot_sync_at', $1, NOW())
               ON CONFLICT (key) DO UPDATE SET value = $1, updated_at = NOW()""",
            datetime.now(UTC).isoformat(),
        )

        # Clean up stuck syncing meetings and retry eligible failures
        cleaned = await store.cleanup_stuck_syncing(older_than_minutes=30)
        retryable = await store.get_retryable_meetings(limit=10)
        retried_ok = 0
        for meeting in retryable:
            ff_id_r = meeting.get("fireflies_transcript_id")
            if not ff_id_r:
                continue
            try:
                await _download_meeting_data(ff_id_r, known_meeting_id=str(meeting["id"]))
                retried_ok += 1
            except Exception:
                logger.warning("boot_retry_failed", ff_id=ff_id_r, exc_info=True)
    finally:
        await store.close()

    logger.info(
        "fireflies_boot_sync_complete",
        total=len(all_transcripts),
        new=new_count,
        updated=updated_count,
        already_synced=already_synced,
        errors=errors,
        cleaned_stuck=cleaned,
        retried=len(retryable),
        retried_ok=retried_ok,
        api_calls=len(all_transcripts) // 50 + 1,
    )


# =============================================================================
# API Request/Response Models
# =============================================================================


class TranslationConfigRequest(BaseModel):
    """Runtime translation backend configuration."""

    backend: str = Field(
        default="ollama",
        description="Translation backend: ollama, vllm, openai, groq",
    )
    model: str = Field(default="qwen2.5:3b", description="Model name/identifier")
    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Backend API base URL",
    )
    target_language: str = Field(default="zh", description="Target language code")
    temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=2048, ge=1, le=8192, description="Maximum tokens per translation"
    )


class ConnectRequest(BaseModel):
    """Request to connect to Fireflies realtime"""

    api_key: str | None = Field(
        default=None,
        description="Fireflies API key (optional, uses .env if not provided)",
    )
    transcript_id: str = Field(..., description="Transcript ID from active_meetings")
    target_languages: list[str] | None = Field(
        default=None,
        description="Target languages for translation (optional, uses .env default)",
    )
    glossary_id: str | None = Field(default=None, description="Optional glossary ID")
    domain: str | None = Field(default=None, description="Domain for glossary filtering")
    translation_model: str | None = Field(
        default=None,
        description="Translation model/service to use (ollama, groq, etc.)",
    )

    # API base URL override (for testing/demo mode — points at mock server)
    api_base_url: str | None = Field(
        default=None,
        description="Override Fireflies API base URL (for testing/demo mode)",
    )

    # Sentence aggregation config (all optional, uses .env defaults)
    pause_threshold_ms: float | None = Field(default=None)
    max_buffer_words: int | None = Field(default=None)
    context_window_size: int | None = Field(default=None)

    api_base_url: str | None = Field(
        default=None,
        description="Override Fireflies API base URL (for testing/demo mode)",
    )


class SessionResponse(BaseModel):
    """Response with session information"""

    session_id: str
    transcript_id: str
    connection_status: str
    chunks_received: int
    sentences_produced: int
    translations_completed: int
    speakers_detected: list[str]
    connected_at: datetime | None
    error_count: int
    last_error: str | None
    persistence_failures: int = 0
    persistence_healthy: bool = True


class DisconnectRequest(BaseModel):
    """Request to disconnect a session"""

    session_id: str = Field(..., description="Session ID to disconnect")


class InviteBotRequest(BaseModel):
    """Request to invite Fireflies bot to a meeting."""

    meeting_link: str = Field(description="Google Meet, Zoom, or Teams URL")
    title: str | None = Field(default=None, description="Optional meeting title (max 256 chars)")
    duration: int = Field(default=60, ge=15, le=120, description="Expected duration in minutes")


class GetMeetingsRequest(BaseModel):
    """Request to get active meetings"""

    api_key: str | None = Field(
        default=None,
        description="Fireflies API key (optional, uses .env if not provided)",
    )
    email: str | None = Field(default=None, description="Filter by email")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/connect",
    response_model=FirefliesConnectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Connect to Fireflies realtime transcription",
    description="Start receiving live transcripts from a Fireflies meeting",
)
async def connect_to_fireflies(
    request: ConnectRequest,
    background_tasks: BackgroundTasks,
    manager: FirefliesSessionManager = Depends(get_session_manager),
    ff_config: FirefliesSettings = Depends(get_fireflies_config),
    translation_client=Depends(get_translation_service_client),
):
    """
    Connect to a Fireflies realtime transcript stream.

    This creates a new session that:
    1. Connects to Fireflies WebSocket API
    2. Receives transcript chunks in real-time
    3. Aggregates chunks into sentences
    4. Translates to target languages
    5. Stores everything in the database

    API key and other settings are optional - will use .env configuration if not provided.
    """
    # Get API key from request or config
    api_key = request.api_key or ff_config.api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fireflies API key required. Provide in request or set FIREFLIES_API_KEY in .env",
        )

    try:
        # Create session config with request values or config defaults
        config = FirefliesSessionConfig(
            api_key=api_key,
            transcript_id=request.transcript_id,
            target_languages=request.target_languages or ff_config.default_target_languages,
            glossary_id=request.glossary_id,
            domain=request.domain,
            translation_model=request.translation_model,  # Pass through model selection
            pause_threshold_ms=request.pause_threshold_ms or ff_config.pause_threshold_ms,
            max_buffer_words=request.max_buffer_words or ff_config.max_buffer_words,
            context_window_size=request.context_window_size or ff_config.context_window_size,
            api_base_url=request.api_base_url,
        )

        logger.info(
            "session_config",
            languages=config.target_languages,
            model=config.translation_model or "default",
        )

        # Get database manager for transcript/translation storage
        data_pipeline = get_data_pipeline()
        db_manager = data_pipeline.db_manager if data_pipeline else None

        if db_manager:
            logger.info("db_manager_connected")
        else:
            logger.warning("db_manager_unavailable", consequence="transcripts_not_persisted")

        # Get optional intelligence and event publisher services
        try:
            meeting_intelligence = get_meeting_intelligence_service()
        except Exception as e:
            meeting_intelligence = None
            logger.warning(
                "meeting_intelligence_unavailable",
                error=str(e),
            )

        try:
            event_publisher = get_event_publisher()
        except Exception as e:
            event_publisher = None
            logger.warning(
                "event_publisher_unavailable",
                error=str(e),
            )

        # Create LLM client for translation pipeline
        llm_client = None
        try:
            llm_client = get_fireflies_llm_client()
            await llm_client.connect()
            logger.info("llm_client_connected")
        except Exception as e:
            logger.warning(
                "llm_client_unavailable",
                error=str(e),
                consequence="translations_skipped",
            )

        # Get glossary adapter for translation pipeline
        glossary_service = None
        try:
            glossary_service = get_glossary_pipeline_adapter()
        except Exception as e:
            logger.warning("glossary_adapter_unavailable", error=str(e))

        # Create session with transcript handling and database persistence
        session = await manager.create_session(
            config,
            db_manager=db_manager,
            meeting_intelligence=meeting_intelligence,
            event_publisher=event_publisher,
            translation_client=translation_client,
            llm_client=llm_client,
            glossary_service=glossary_service,
        )

        logger.info(
            "fireflies_session_created",
            session_id=session.session_id,
            transcript_id=request.transcript_id,
        )

        # Fetch transcript metadata and update meeting title in background
        if session.meeting_db_id:
            _bg_meeting_id = session.meeting_db_id
            _bg_transcript_id = request.transcript_id

            async def _update_meeting_title():
                client = FirefliesClient(api_key=api_key)
                try:
                    detail = await client.get_transcript_detail(_bg_transcript_id)
                    title = detail.get("title") if detail else None
                    if title:
                        meeting_store = await manager._get_meeting_store()
                        if meeting_store:
                            await meeting_store.update_meeting(
                                _bg_meeting_id, title=title
                            )
                            logger.info(
                                "meeting_title_updated",
                                meeting_id=_bg_meeting_id,
                                title=title,
                            )
                except Exception as e:
                    logger.warning("meeting_title_fetch_failed", error=str(e))
                finally:
                    await client.close()

            background_tasks.add_task(_update_meeting_title)

        return FirefliesConnectResponse(
            success=True,
            message="Connected to Fireflies realtime API",
            session_id=session.session_id,
            connection_status=session.connection_status,
            transcript_id=request.transcript_id,
            meeting_id=session.meeting_db_id,
        )

    except FirefliesAPIError as e:
        logger.error("fireflies_api_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fireflies API error: {e!s}",
        ) from e
    except Exception as e:
        logger.exception("fireflies_connect_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect: {e!s}",
        ) from e


@router.post(
    "/disconnect",
    status_code=status.HTTP_200_OK,
    summary="Disconnect from Fireflies",
    description="Stop receiving transcripts and disconnect the session",
)
async def disconnect_from_fireflies(
    request: DisconnectRequest,
    manager: FirefliesSessionManager = Depends(get_session_manager),
):
    """Disconnect from a Fireflies session"""
    success = await manager.disconnect_session(request.session_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {request.session_id}",
        )

    return {
        "success": True,
        "message": f"Disconnected session: {request.session_id}",
    }


@router.get(
    "/sessions",
    response_model=list[SessionResponse],
    summary="Get all Fireflies sessions (live + persisted)",
)
async def get_sessions(
    manager: FirefliesSessionManager = Depends(get_session_manager),
):
    """Get all Fireflies sessions — live in-memory sessions plus persisted meetings from DB."""
    # 1. Live in-memory sessions (currently connected)
    live_sessions = manager.get_all_sessions()
    live_ids: set[str] = set()

    results: list[SessionResponse] = []
    for s in live_sessions:
        live_ids.add(s.session_id)
        results.append(
            SessionResponse(
                session_id=s.session_id,
                transcript_id=s.fireflies_transcript_id,
                connection_status=s.connection_status.value
                if hasattr(s.connection_status, "value")
                else str(s.connection_status),
                chunks_received=s.chunks_received,
                sentences_produced=s.sentences_produced,
                translations_completed=s.translations_completed,
                speakers_detected=s.speakers_detected,
                connected_at=s.connected_at,
                error_count=s.error_count,
                last_error=s.last_error,
            )
        )

    # 2. Persisted meetings from database (supplements live sessions)
    try:
        from database import get_db_session
        from database.models import Meeting, MeetingChunk, MeetingSentence, MeetingSpeaker, MeetingTranslation
        from sqlalchemy import func, select

        from database.database import get_database_manager
        db_manager = get_database_manager()

        async with db_manager.get_session() as db:
            # Correlated subqueries for counts
            chunk_count = (
                select(func.count(MeetingChunk.id))
                .where(MeetingChunk.meeting_id == Meeting.id)
                .correlate(Meeting)
                .scalar_subquery()
                .label("chunk_count")
            )
            sentence_count = (
                select(func.count(MeetingSentence.id))
                .where(MeetingSentence.meeting_id == Meeting.id)
                .correlate(Meeting)
                .scalar_subquery()
                .label("sentence_count")
            )
            # Translation count via sentence join
            translation_count = (
                select(func.count(MeetingTranslation.id))
                .where(
                    MeetingTranslation.sentence_id.in_(
                        select(MeetingSentence.id).where(
                            MeetingSentence.meeting_id == Meeting.id
                        )
                    )
                )
                .correlate(Meeting)
                .scalar_subquery()
                .label("translation_count")
            )

            rows = await db.execute(
                select(Meeting, chunk_count, sentence_count, translation_count)
                .order_by(Meeting.created_at.desc())
                .limit(50)
            )

            for meeting, chunks, sentences, translations in rows.all():
                meeting_id_str = str(meeting.id)
                # Skip meetings that are already tracked as live sessions
                if meeting_id_str in live_ids:
                    continue
                # Also check by fireflies transcript ID
                if meeting.fireflies_transcript_id and meeting.fireflies_transcript_id in live_ids:
                    continue

                # Determine connection status from meeting status
                status_str = meeting.status or "completed"
                if status_str == "live":
                    conn_status = "CONNECTED"
                elif status_str in ("completed", "synced"):
                    conn_status = "COMPLETED"
                else:
                    conn_status = "DISCONNECTED"

                # Get speaker names from the speakers relationship
                speaker_names: list[str] = []
                try:
                    speaker_rows = await db.execute(
                        select(MeetingSpeaker.speaker_name)
                        .where(MeetingSpeaker.meeting_id == meeting.id)
                    )
                    speaker_names = [r[0] for r in speaker_rows.all() if r[0]]
                except Exception:
                    pass

                results.append(
                    SessionResponse(
                        session_id=meeting_id_str,
                        transcript_id=meeting.fireflies_transcript_id or "",
                        connection_status=conn_status,
                        chunks_received=chunks or 0,
                        sentences_produced=sentences or 0,
                        translations_completed=translations or 0,
                        speakers_detected=speaker_names,
                        connected_at=meeting.start_time or meeting.created_at,
                        error_count=0,
                        last_error=None,
                    )
                )
    except Exception as e:
        # If DB is unavailable, still return live sessions
        logger.warning("sessions_db_fallback_failed", error=str(e))

    return results


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get a specific Fireflies session",
)
async def get_session(
    session_id: str,
    manager: FirefliesSessionManager = Depends(get_session_manager),
):
    """Get details of a specific Fireflies session (live or persisted)."""
    # 1. Check in-memory live sessions first
    session = manager.get_session(session_id)

    if session:
        return SessionResponse(
            session_id=session.session_id,
            transcript_id=session.fireflies_transcript_id,
            connection_status=session.connection_status.value
            if hasattr(session.connection_status, "value")
            else str(session.connection_status),
            chunks_received=session.chunks_received,
            sentences_produced=session.sentences_produced,
            translations_completed=session.translations_completed,
            speakers_detected=session.speakers_detected,
            connected_at=session.connected_at,
            error_count=session.error_count,
            last_error=session.last_error,
            persistence_failures=session.persistence_failures,
            persistence_healthy=session.persistence_healthy,
        )

    # 2. Fall back to database for persisted meetings
    from database.database import get_database_manager
    from database.models import Meeting, MeetingChunk, MeetingSentence, MeetingSpeaker, MeetingTranslation
    from sqlalchemy import func, select
    import uuid as _uuid

    try:
        db_manager = get_database_manager()
    except RuntimeError:
        from errors import NotFoundError
        raise NotFoundError("Session", session_id)

    async with db_manager.get_session() as db:
        # Try by UUID first, then by fireflies transcript ID
        meeting = None
        try:
            parsed_uuid = _uuid.UUID(session_id)
            result = await db.execute(
                select(Meeting).where(Meeting.id == parsed_uuid)
            )
            meeting = result.scalar_one_or_none()
        except ValueError:
            pass

        if not meeting:
            result = await db.execute(
                select(Meeting).where(Meeting.fireflies_transcript_id == session_id)
            )
            meeting = result.scalar_one_or_none()

        if not meeting:
            from errors import NotFoundError
            raise NotFoundError("Session", session_id)

        # Get counts
        chunks = await db.scalar(
            select(func.count(MeetingChunk.id)).where(MeetingChunk.meeting_id == meeting.id)
        ) or 0
        sentences = await db.scalar(
            select(func.count(MeetingSentence.id)).where(MeetingSentence.meeting_id == meeting.id)
        ) or 0
        translations = await db.scalar(
            select(func.count(MeetingTranslation.id)).where(
                MeetingTranslation.sentence_id.in_(
                    select(MeetingSentence.id).where(MeetingSentence.meeting_id == meeting.id)
                )
            )
        ) or 0
        speaker_rows = await db.execute(
            select(MeetingSpeaker.speaker_name).where(MeetingSpeaker.meeting_id == meeting.id)
        )
        speaker_names = [r[0] for r in speaker_rows.all() if r[0]]

        status_str = meeting.status or "completed"
        conn_status = "CONNECTED" if status_str == "live" else (
            "COMPLETED" if status_str in ("completed", "synced") else "DISCONNECTED"
        )

        return SessionResponse(
            session_id=str(meeting.id),
            transcript_id=meeting.fireflies_transcript_id or "",
            connection_status=conn_status,
            chunks_received=chunks,
            sentences_produced=sentences,
            translations_completed=translations,
            speakers_detected=speaker_names,
            connected_at=meeting.start_time or meeting.created_at,
            error_count=0,
            last_error=None,
        )


class DisplayModeRequest(BaseModel):
    """Request to change caption display mode"""

    mode: str = Field(default="both", description="Display mode: english, translated, both")


@router.put(
    "/sessions/{session_id}/display-mode",
    summary="Set caption display mode",
    description="Broadcast display mode change to all caption clients in a session",
)
async def set_display_mode(session_id: str, body: DisplayModeRequest) -> dict[str, Any]:
    """Broadcast display mode change to all caption clients in a session."""
    ws_manager = get_ws_manager()
    await ws_manager.broadcast_to_session(
        session_id,
        {"event": "set_display_mode", "mode": body.mode},
    )
    logger.info("display_mode_changed", session_id=session_id, mode=body.mode)
    return {"success": True, "mode": body.mode}


@router.post(
    "/sessions/{session_id}/pause",
    summary="Pause live pipeline",
    description="Pause chunk processing for a session. Chunks received while paused are dropped.",
)
async def pause_session(
    session_id: str,
    manager: FirefliesSessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Pause the pipeline for a session."""
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    coordinator = manager.get_coordinator(session_id)
    if not coordinator:
        raise HTTPException(status_code=400, detail="No pipeline coordinator for session")

    coordinator.pause()
    session.is_paused = True

    # Notify caption clients
    ws_manager = get_ws_manager()
    await ws_manager.broadcast_to_session(
        session_id,
        {"event": "pipeline_paused", "session_id": session_id},
    )

    logger.info("session_paused", session_id=session_id)
    return {"success": True, "paused": True}


@router.post(
    "/sessions/{session_id}/resume",
    summary="Resume live pipeline",
    description="Resume chunk processing for a paused session.",
)
async def resume_session(
    session_id: str,
    manager: FirefliesSessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Resume the pipeline for a session."""
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    coordinator = manager.get_coordinator(session_id)
    if not coordinator:
        raise HTTPException(status_code=400, detail="No pipeline coordinator for session")

    coordinator.resume()
    session.is_paused = False

    # Notify caption clients
    ws_manager = get_ws_manager()
    await ws_manager.broadcast_to_session(
        session_id,
        {"event": "pipeline_resumed", "session_id": session_id},
    )

    logger.info("session_resumed", session_id=session_id)
    return {"success": True, "paused": False}


class TargetLanguagesRequest(BaseModel):
    """Request to change target languages for a live session"""

    target_languages: list[str] = Field(
        ..., description="New target languages (e.g. ['es', 'zh'])", min_length=1
    )


@router.put(
    "/sessions/{session_id}/target-languages",
    summary="Change target languages for a live session",
    description="Update target languages without disconnecting. Takes effect on the next chunk.",
)
async def set_target_languages(
    session_id: str,
    body: TargetLanguagesRequest,
    manager: FirefliesSessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Change the target languages for translation on a live session."""
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    coordinator = manager.get_coordinator(session_id)
    if not coordinator:
        raise HTTPException(status_code=400, detail="No pipeline coordinator for session")

    old_languages = list(coordinator.config.target_languages)
    coordinator.config.target_languages = body.target_languages

    if session.meeting_db_id:
        meeting_store = await manager._get_meeting_store()
        if meeting_store is not None:
            await meeting_store.update_meeting(
                session.meeting_db_id,
                target_languages=body.target_languages,
            )

    # Reload glossary if languages changed (different language may need different terms)
    if coordinator.glossary_service and coordinator.config.glossary_id:
        try:
            coordinator._glossary_terms = await coordinator._load_glossary()
        except Exception as e:
            logger.warning("glossary_reload_failed", error=str(e))

    # Notify caption clients
    ws_manager = get_ws_manager()
    await ws_manager.broadcast_to_session(
        session_id,
        {
            "event": "target_languages_changed",
            "session_id": session_id,
            "target_languages": body.target_languages,
        },
    )

    logger.info(
        "target_languages_changed",
        session_id=session_id,
        old=old_languages,
        new=body.target_languages,
    )
    return {
        "success": True,
        "target_languages": body.target_languages,
        "previous": old_languages,
    }


@router.post(
    "/meetings",
    response_model=ActiveMeetingsResponse,
    summary="Get active meetings from Fireflies",
    description="Query Fireflies GraphQL API for active meetings",
)
async def get_active_meetings(
    request: GetMeetingsRequest,
    ff_config: FirefliesSettings = Depends(get_fireflies_config),
):
    """
    Get active meetings from Fireflies.

    Returns a list of meetings that can be connected to.
    Use the meeting ID as the transcript_id when connecting.

    API key is optional - will use .env configuration if not provided.
    """
    # Get API key from request or config
    api_key = request.api_key or ff_config.api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fireflies API key required. Provide in request or set FIREFLIES_API_KEY in .env",
        )

    try:
        client = FirefliesClient(api_key=api_key)

        meetings = await client.get_active_meetings(email=request.email)

        await client.close()

        return ActiveMeetingsResponse(
            success=True,
            meetings=meetings,
            count=len(meetings),
        )

    except FirefliesAPIError as e:
        logger.error("fireflies_api_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fireflies API error: {e!s}",
        ) from e
    except Exception as e:
        logger.exception("get_active_meetings_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get meetings: {e!s}",
        ) from e


@router.get(
    "/health",
    summary="Check Fireflies integration health",
)
async def health_check(
    manager: FirefliesSessionManager = Depends(get_session_manager),
):
    """Check the health of Fireflies integration"""
    sessions = manager.get_all_sessions()
    connected = [s for s in sessions if s.connection_status == FirefliesConnectionStatus.CONNECTED]

    return {
        "status": "healthy",
        "total_sessions": len(sessions),
        "connected_sessions": len(connected),
        "sessions": [
            {
                "session_id": s.session_id,
                "status": s.connection_status.value
                if hasattr(s.connection_status, "value")
                else str(s.connection_status),
                "chunks_received": s.chunks_received,
            }
            for s in sessions
        ],
    }


# =============================================================================
# Runtime Translation Config (Hot-Swap Backend)
# =============================================================================


@router.put(
    "/config/translation",
    summary="Update translation backend configuration at runtime",
    description="Hot-swap between Ollama, vLLM, OpenAI, Groq without restart",
)
async def update_translation_config(config: TranslationConfigRequest) -> dict[str, Any]:
    """Update translation backend configuration at runtime.

    Changes take effect immediately. Optionally forwards config to
    the translation service for hot-reload.
    """
    manager = get_session_manager()
    config_dict = config.model_dump()

    # Store config on session manager for future sessions
    manager.translation_config = config_dict

    # Attempt to forward to translation service for hot-reload
    translation_url = os.environ.get("TRANSLATION_SERVICE_URL", "http://localhost:5003")
    forwarded = False
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as http_client:
            resp = await http_client.post(
                f"{translation_url}/api/config/update",
                json=config_dict,
            )
            if resp.status_code == 200:
                forwarded = True
                logger.info(
                    "translation_config_forwarded",
                    backend=config.backend,
                    model=config.model,
                )
    except Exception as e:
        logger.warning("translation_config_forward_failed", error=str(e))

    logger.info(
        "translation_config_updated",
        backend=config.backend,
        model=config.model,
        forwarded=forwarded,
    )

    result: dict[str, Any] = {"success": True, "config": config_dict}
    if not forwarded:
        result["warning"] = (
            "Translation service not updated (may not be running or endpoint not available)"
        )
    return result


@router.get(
    "/config/translation",
    summary="Get current translation configuration",
    description="Returns the active translation backend configuration",
)
async def get_translation_config() -> dict[str, Any]:
    """Get current translation configuration."""
    manager = get_session_manager()

    # Return stored config or derive from environment
    stored_config = getattr(manager, "translation_config", None)
    if stored_config:
        return {"config": stored_config}

    # Default from environment
    return {
        "config": {
            "backend": "ollama"
            if os.environ.get("OLLAMA_ENABLE", "true").lower() == "true"
            else "none",
            "model": os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "target_language": os.environ.get("DEFAULT_TARGET_LANGUAGE", "zh"),
            "temperature": float(os.environ.get("TRANSLATION_TEMPERATURE", "0.3")),
            "max_tokens": int(os.environ.get("TRANSLATION_MAX_TOKENS", "2048")),
        }
    }


# =============================================================================
# Shared: Store Transcript Data to DB
# =============================================================================


async def _store_transcript_to_db(
    transcript: dict[str, Any],
    store: MeetingStore,
) -> str:
    """Store a fully-populated Fireflies transcript into the database.

    Handles find-or-create for the meeting record, then stores sentences,
    insights, speaker analytics, and media URLs. Used by both the bulk
    sync path (inline from paginated query) and the single-transcript
    download path (_download_meeting_data).

    Args:
        transcript: Raw transcript dict from Fireflies GraphQL (either the
            plural `transcripts` query or `download_full_transcript` structured result).
        store: Initialized MeetingStore instance.

    Returns:
        The database meeting ID (UUID string).
    """
    ff_id = transcript.get("id", "")

    # --- Build structured result matching download_full_transcript() output ---
    insights: list[dict[str, Any]] = []
    if transcript.get("summary"):
        insights.append({"type": "summary", "content": transcript["summary"]})
    if transcript.get("analytics"):
        analytics = transcript["analytics"]
        if analytics.get("sentiments"):
            insights.append({"type": "sentiment", "content": analytics["sentiments"]})
        if analytics.get("speakers"):
            insights.append({"type": "speaker_analytics", "content": analytics["speakers"]})
        if analytics.get("categories"):
            insights.append({"type": "ai_filters", "content": analytics["categories"]})
    if transcript.get("meeting_attendees") or transcript.get("meeting_attendance"):
        insights.append({
            "type": "attendance",
            "content": {
                "attendees": transcript.get("meeting_attendees", []),
                "attendance": transcript.get("meeting_attendance", []),
            },
        })

    sentences = transcript.get("sentences", [])
    summary = transcript.get("summary", {})
    target_languages = [os.environ.get("DEFAULT_TARGET_LANGUAGE", "zh")]

    # --- Find or create meeting record ---
    meeting = await store.get_meeting_by_ff_id(ff_id)
    if meeting:
        meeting_db_id = str(meeting["id"])
        await store.update_meeting(meeting_db_id, target_languages=target_languages)
        await store.complete_meeting(meeting_db_id)
    else:
        meeting_db_id = await store.create_meeting(
            fireflies_transcript_id=ff_id,
            title=transcript.get("title"),
            meeting_link=transcript.get("meeting_link"),
            organizer_email=transcript.get("organizer_email"),
            participants=transcript.get("participants"),
            target_languages=target_languages,
            meeting_metadata={"ingest_mode": "fireflies_sync"},
            source="fireflies",
            status="completed",
        )

    # Mark sync in progress (outside transaction — visible to other queries immediately)
    await store.update_sync_status(meeting_db_id, "syncing")

    # All data inserts happen inside a single transaction for atomicity.
    # If any insert fails, the entire batch rolls back — no orphaned data.
    import json as _json

    async with store.transaction() as conn:
        # Store all structured insights
        for insight in insights:
            content_json = _json.dumps(insight["content"])
            await conn.execute(
                """INSERT INTO meeting_data_insights (id, meeting_id, insight_type, content, source)
                   VALUES ($1::uuid, $2::uuid, $3, $4::jsonb, 'fireflies')""",
                str(uuid.uuid4()), meeting_db_id, insight["type"], content_json,
            )

        # Store sentences with ai_filters
        for sentence in sentences:
            sentence_id = str(uuid.uuid4())
            await conn.execute(
                """INSERT INTO meeting_sentences (id, meeting_id, text, speaker_name,
                                                   start_time, end_time, boundary_type, chunk_ids)
                   VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, 'fireflies_download', '[]'::jsonb)""",
                sentence_id, meeting_db_id,
                sentence.get("text", ""),
                sentence.get("speaker_name", "Unknown"),
                float(sentence.get("start_time", 0)),
                float(sentence.get("end_time", 0)),
            )
            if sentence.get("ai_filters"):
                ai_content = _json.dumps({
                    "sentence_id": sentence_id,
                    "filters": sentence["ai_filters"],
                })
                await conn.execute(
                    """INSERT INTO meeting_data_insights (id, meeting_id, insight_type, content, source)
                       VALUES ($1::uuid, $2::uuid, 'sentence_ai_filter', $3::jsonb, 'fireflies')""",
                    str(uuid.uuid4()), meeting_db_id, ai_content,
                )

        # Store speaker analytics from insights blob
        for insight in insights:
            if insight.get("type") == "speaker_analytics":
                for speaker_data in insight.get("content") or []:
                    if isinstance(speaker_data, dict):
                        analytics_json = _json.dumps(speaker_data)
                        await conn.execute(
                            """INSERT INTO meeting_speakers (id, meeting_id, speaker_name, email,
                                                             talk_time_seconds, word_count, sentiment_score, analytics)
                               VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8::jsonb)
                               ON CONFLICT (meeting_id, speaker_name) DO UPDATE SET
                                   talk_time_seconds = EXCLUDED.talk_time_seconds,
                                   word_count = EXCLUDED.word_count,
                                   analytics = EXCLUDED.analytics""",
                            str(uuid.uuid4()), meeting_db_id,
                            speaker_data.get("name", "Unknown"),
                            speaker_data.get("email"),
                            float(speaker_data.get("duration", 0)),
                            int(speaker_data.get("word_count", 0)),
                            None,
                            analytics_json,
                        )

        # Store summary sub-fields as individual insights
        if summary and isinstance(summary, dict):
            for insight_type in [
                "overview", "action_items", "outline", "keywords", "shorthand_bullet",
            ]:
                sub_content = summary.get(insight_type)
                if sub_content:
                    content_val = (
                        {"text": sub_content} if isinstance(sub_content, str) else sub_content
                    )
                    await conn.execute(
                        """INSERT INTO meeting_data_insights (id, meeting_id, insight_type, content, source)
                           VALUES ($1::uuid, $2::uuid, $3, $4::jsonb, 'fireflies')""",
                        str(uuid.uuid4()), meeting_db_id, insight_type, _json.dumps(content_val),
                    )

        # Store speakers from analytics blob
        raw_analytics = transcript.get("analytics", {})
        if raw_analytics and isinstance(raw_analytics, dict):
            for speaker_data in raw_analytics.get("speakers", []):
                if isinstance(speaker_data, dict):
                    sentiment = speaker_data.get("sentiment")
                    sentiment_score = (
                        sentiment.get("score") if isinstance(sentiment, dict) else None
                    )
                    analytics_json = _json.dumps(speaker_data)
                    await conn.execute(
                        """INSERT INTO meeting_speakers (id, meeting_id, speaker_name, email,
                                                         talk_time_seconds, word_count, sentiment_score, analytics)
                           VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8::jsonb)
                           ON CONFLICT (meeting_id, speaker_name) DO UPDATE SET
                               email = COALESCE(EXCLUDED.email, meeting_speakers.email),
                               talk_time_seconds = EXCLUDED.talk_time_seconds,
                               word_count = EXCLUDED.word_count,
                               sentiment_score = EXCLUDED.sentiment_score,
                               analytics = COALESCE(EXCLUDED.analytics, meeting_speakers.analytics)""",
                        str(uuid.uuid4()), meeting_db_id,
                        speaker_data.get("name", "Unknown"),
                        speaker_data.get("email"),
                        float(speaker_data.get("talk_time", 0)),
                        int(speaker_data.get("word_count", 0)),
                        sentiment_score,
                        analytics_json,
                    )

    # Mark sync complete with media URLs (outside transaction — only after success)
    await store.update_sync_status(
        meeting_db_id,
        "synced",
        audio_url=transcript.get("audio_url"),
        video_url=transcript.get("video_url"),
        transcript_url=transcript.get("transcript_url"),
    )

    logger.info(
        "transcript_stored_to_db",
        fireflies_id=ff_id,
        meeting_db_id=meeting_db_id,
        insight_count=len(insights),
        sentence_count=len(sentences),
    )

    return meeting_db_id


# =============================================================================
# Post-Meeting Webhook & Full Data Download (single-transcript path)
# =============================================================================


async def _download_meeting_data(
    fireflies_transcript_id: str,
    known_meeting_id: str | None = None,
) -> None:
    """Download a single transcript from Fireflies and store to DB.

    Used by per-meeting sync and webhook — makes 1 API call to fetch the
    individual transcript via TRANSCRIPT_FULL_QUERY. For bulk sync, prefer
    the inline path via get_full_transcripts() + _store_transcript_to_db().

    Args:
        fireflies_transcript_id: Fireflies transcript ID to download.
        known_meeting_id: If provided, used for error-status updates instead
            of looking up by fireflies_transcript_id (avoids duplicate-row bugs).
    """
    api_key = os.environ.get("FIREFLIES_API_KEY", "")
    db_url = os.environ.get("DATABASE_URL", "")

    if not api_key or not db_url:
        logger.error(
            "download_meeting_data_missing_config",
            has_api_key=bool(api_key),
            has_db_url=bool(db_url),
        )
        return

    try:
        # Download full transcript from Fireflies (1 API call)
        client = FirefliesGraphQLClient(api_key=api_key)
        try:
            result = await client.download_full_transcript(fireflies_transcript_id)
        finally:
            await client.close()

        if not result:
            logger.error(
                "download_meeting_data_no_result",
                transcript_id=fireflies_transcript_id,
            )
            return

        # Delegate to shared storage function
        store = MeetingStore(db_url)
        await store.initialize()
        try:
            _transcript_data = result["transcript"]
            await _store_transcript_to_db(_transcript_data, store)

            # Auto-trigger diarization if rules match
            try:
                from services.diarization.auto_trigger import maybe_trigger_diarization
                from services.diarization.db import get_diarization_rules
                from models.diarization import DiarizationRules
                from database.database import get_database_manager
                _ff_id = _transcript_data.get("id", fireflies_transcript_id)
                _synced_meeting = await store.get_meeting_by_ff_id(_ff_id)
                if _synced_meeting:
                    _meeting_meta = {
                        "id": _synced_meeting["id"],
                        "title": _transcript_data.get("title", ""),
                        "participants": _transcript_data.get("participants") or [],
                        "duration": int((_transcript_data.get("duration") or 0)),
                        "sentence_count": len(_transcript_data.get("sentences") or []),
                    }
                    async with get_database_manager().get_session() as _db:
                        _rules_dict = await get_diarization_rules(_db)
                    _rules = DiarizationRules(**_rules_dict)
                    await maybe_trigger_diarization(
                        _meeting_meta, _rules, None
                    )
            except Exception:
                pass  # Diarization not available — skip silently
        finally:
            await store.close()

    except Exception as e:
        # Attempt to mark sync as failed with retry tracking
        is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
        try:
            _db_url = os.environ.get("DATABASE_URL", "")
            if _db_url:
                _err_store = MeetingStore(_db_url)
                await _err_store.initialize()
                try:
                    _mid = known_meeting_id
                    if not _mid:
                        _meeting = await _err_store.get_meeting_by_ff_id(
                            fireflies_transcript_id
                        )
                        _mid = str(_meeting["id"]) if _meeting else None
                    if _mid:
                        await _err_store.mark_sync_failed(
                            _mid, str(e), is_rate_limit=is_rate_limit,
                        )
                finally:
                    await _err_store.close()
        except Exception:
            logger.warning(
                "sync_status_update_on_error_failed",
                transcript_id=fireflies_transcript_id,
                exc_info=True,
            )

        logger.error(
            "download_meeting_data_failed",
            transcript_id=fireflies_transcript_id,
            error=str(e),
        )


@router.post(
    "/webhook",
    summary="Handle Fireflies post-meeting webhook",
    description="Receives Fireflies webhook notifications when transcription completes and triggers background data download",
)
async def fireflies_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Handle Fireflies post-meeting webhook.

    Fireflies POSTs here when transcription completes. Payload:
    {"meetingId": "xxx", "eventType": "Transcription completed", "clientReferenceId": "..."}

    The endpoint returns immediately (200) and processes the download in a
    background task so Fireflies does not time out waiting for our response.
    """
    body = await request.json()
    event_type = body.get("eventType")
    meeting_id = body.get("meetingId")

    if event_type == "Transcription completed" and meeting_id:
        logger.info(
            "fireflies_webhook_received",
            event_type=event_type,
            meeting_id=meeting_id,
        )
        background_tasks.add_task(_download_meeting_data, meeting_id)
        return {"status": "accepted"}

    logger.debug("fireflies_webhook_ignored", event_type=event_type)
    return {"status": "ignored"}


# =============================================================================
# Invite Bot (Paste Meeting Link)
# =============================================================================


@router.post(
    "/invite-bot",
    summary="Invite Fireflies bot to a meeting",
    description="Paste a meeting link to invite Fireflies bot and auto-connect when ready",
)
async def invite_fireflies_bot(
    request: InviteBotRequest,
    background_tasks: BackgroundTasks,
    ff_config: FirefliesSettings = Depends(get_fireflies_config),
) -> dict[str, Any]:
    """Invite Fireflies bot to a meeting and auto-connect when ready.

    Rate limit: 3 requests per 20 minutes (Fireflies API limit).
    """
    api_key = ff_config.api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FIREFLIES_API_KEY not configured",
        )

    client = FirefliesGraphQLClient(api_key=api_key)
    try:
        result = await client.add_to_live_meeting(
            meeting_link=request.meeting_link,
            title=request.title,
            duration=request.duration,
        )
    finally:
        await client.close()

    if result.get("success"):
        logger.info(
            "fireflies_bot_invited",
            meeting_link=request.meeting_link,
            title=request.title,
        )
        # Poll for the meeting to appear in active_meetings, then auto-connect
        background_tasks.add_task(
            _wait_and_connect_meeting,
            api_key,
            request.meeting_link,
            request.title,
        )
        return {
            "success": True,
            "message": "Fireflies bot invited. Will auto-connect when ready.",
        }

    logger.warning("fireflies_bot_invite_failed", result=result)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=result.get("message", "Failed to invite bot"),
    )


async def _wait_and_connect_meeting(
    api_key: str,
    meeting_link: str,
    title: str | None,
) -> None:
    """Wait for Fireflies to join the meeting, then auto-connect our session."""
    target_language = os.environ.get("DEFAULT_TARGET_LANGUAGE", "zh")
    max_attempts = 20  # 20 * 15s = 5 minutes max wait

    for attempt in range(max_attempts):
        await asyncio.sleep(15)  # Poll every 15 seconds

        try:
            client = FirefliesGraphQLClient(api_key=api_key)
            try:
                meetings = await client.get_active_meetings()
            finally:
                await client.close()

            if not meetings:
                continue

            # Look for a meeting matching our link
            for meeting in meetings:
                if meeting.meeting_link and meeting_link in meeting.meeting_link:
                    logger.info(
                        "wait_and_connect_found",
                        meeting_id=meeting.id,
                        attempt=attempt + 1,
                    )

                    # Build session config matching the auto-connect pattern
                    config = FirefliesSessionConfig(
                        api_key=api_key,
                        transcript_id=meeting.id,
                        target_languages=[target_language],
                    )

                    # Get optional services for the pipeline
                    data_pipeline = get_data_pipeline()
                    db_manager = data_pipeline.db_manager if data_pipeline else None

                    llm_client = None
                    try:
                        llm_client = get_fireflies_llm_client()
                        await llm_client.connect()
                    except Exception as e:
                        logger.warning(
                            "wait_connect_llm_client_unavailable",
                            meeting_id=meeting.id,
                            error=str(e),
                        )

                    try:
                        meeting_intelligence = get_meeting_intelligence_service()
                    except Exception as e:
                        meeting_intelligence = None
                        logger.warning(
                            "wait_connect_meeting_intelligence_unavailable",
                            meeting_id=meeting.id,
                            error=str(e),
                        )

                    try:
                        event_publisher = get_event_publisher()
                    except Exception as e:
                        event_publisher = None
                        logger.warning(
                            "wait_connect_event_publisher_unavailable",
                            meeting_id=meeting.id,
                            error=str(e),
                        )

                    glossary_service = None
                    try:
                        glossary_service = get_glossary_pipeline_adapter()
                    except Exception as e:
                        logger.warning(
                            "wait_connect_glossary_unavailable",
                            meeting_id=meeting.id,
                            error=str(e),
                        )

                    manager = get_session_manager()
                    await manager.create_session(
                        config,
                        db_manager=db_manager,
                        llm_client=llm_client,
                        meeting_intelligence=meeting_intelligence,
                        event_publisher=event_publisher,
                        glossary_service=glossary_service,
                    )
                    return

        except Exception as e:
            logger.error(
                "wait_and_connect_poll_error",
                attempt=attempt + 1,
                error=str(e),
            )

    logger.warning("wait_and_connect_timeout", meeting_link=meeting_link)


# =============================================================================
# Past Transcripts Endpoints
# =============================================================================


class GetTranscriptsRequest(BaseModel):
    """Request to get past transcripts from Fireflies"""

    api_key: str | None = None
    limit: int = Field(default=20, ge=1, le=100, description="Max transcripts to return")
    skip: int = Field(default=0, ge=0, description="Pagination offset")


class TranscriptsResponse(BaseModel):
    """Response containing past transcripts"""

    success: bool
    transcripts: list[dict[str, Any]]
    count: int


class SyncAllRequest(BaseModel):
    """Request to sync all Fireflies transcripts to local DB."""

    api_key: str | None = Field(
        default=None,
        description="Fireflies API key (optional, uses .env if not provided)",
    )


@router.post(
    "/sync-all",
    summary="Sync all Fireflies transcripts to local database",
    description=(
        "Fetches all past transcripts from Fireflies using the bulk query "
        "(up to 50 per API call with full data) and stores them inline. "
        "No individual per-transcript API calls needed."
    ),
)
async def sync_all_transcripts(
    request: SyncAllRequest,
    ff_config: FirefliesSettings = Depends(get_fireflies_config),
) -> dict[str, Any]:
    """Sync all Fireflies transcripts to the local database.

    Uses the bulk PAST_TRANSCRIPTS_FULL_QUERY to fetch complete transcript
    data (sentences, analytics, summary, media) in pages of 50 — then
    stores each directly without spawning per-transcript background tasks.

    API calls: ceil(N / 50) instead of 1 + N.
    """
    api_key = request.api_key or ff_config.api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fireflies API key required. Provide in request or set FIREFLIES_API_KEY in .env",
        )

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DATABASE_URL not configured",
        )

    # Paginate through all Fireflies transcripts with FULL data
    client = FirefliesGraphQLClient(api_key=api_key)
    all_transcripts: list[dict[str, Any]] = []
    api_calls = 0
    try:
        skip = 0
        page_size = 50
        while True:
            try:
                page = await client.get_full_transcripts(limit=page_size, skip=skip)
            except FirefliesRateLimitError as e:
                if not all_transcripts:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "message": "Fireflies API rate limit exceeded",
                            "retry_after": e.retry_after,
                        },
                    ) from e
                # Process whatever we fetched before the limit
                logger.warning(
                    "sync_all_rate_limited_partial",
                    fetched=len(all_transcripts),
                    retry_after=e.retry_after,
                )
                break
            api_calls += 1
            if not page:
                break
            all_transcripts.extend(page)
            if len(page) < page_size:
                break
            skip += page_size
    finally:
        await client.close()

    # Store transcripts inline (no extra API calls!)
    store = MeetingStore(db_url)
    await store.initialize()
    synced = 0
    skipped = 0
    errors = 0
    try:
        for transcript in all_transcripts:
            ff_id = transcript.get("id")
            if not ff_id:
                continue
            meeting = await store.get_meeting_by_ff_id(ff_id)
            if meeting and meeting.get("insight_count", 0) > 0:
                skipped += 1
            else:
                try:
                    await _store_transcript_to_db(transcript, store)
                    synced += 1
                except Exception as e:
                    errors += 1
                    # Mark failed with retry tracking (exponential backoff)
                    is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                    try:
                        stuck = meeting or await store.get_meeting_by_ff_id(ff_id)
                        if stuck:
                            await store.mark_sync_failed(
                                str(stuck["id"]), str(e), is_rate_limit=is_rate_limit,
                            )
                    except Exception:
                        pass  # best-effort status update
                    logger.warning(
                        "sync_all_transcript_failed",
                        ff_id=ff_id,
                        exc_info=True,
                    )
    finally:
        await store.close()

    logger.info(
        "sync_all_complete",
        total=len(all_transcripts),
        synced=synced,
        skipped=skipped,
        errors=errors,
        api_calls=api_calls,
    )

    return {
        "synced": synced,
        "skipped": skipped,
        "errors": errors,
        "total": len(all_transcripts),
        "api_calls_used": api_calls,
    }


@router.get(
    "/sync-status",
    summary="Get last sync timestamp and stats",
    description="Returns when the last boot sync or manual sync-all ran",
)
async def get_sync_status() -> dict[str, Any]:
    """Get the last Fireflies sync timestamp."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return {"last_sync_at": None, "message": "DATABASE_URL not configured"}

    store = MeetingStore(db_url)
    await store.initialize()
    try:
        row = await store._pool.fetchrow(
            "SELECT value, updated_at FROM system_config WHERE key = 'last_boot_sync_at'"
        )
        if row:
            return {
                "last_sync_at": row["value"],
                "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
            }
        return {"last_sync_at": None, "message": "No sync has run yet"}
    finally:
        await store.close()


@router.post(
    "/retry-failed",
    summary="Retry failed meeting syncs",
    description="Retries all meetings in 'failed' state that are under the retry limit "
    "and past their backoff window. Uses exponential backoff.",
)
async def retry_failed_meetings() -> dict[str, Any]:
    """Retry eligible failed meetings with exponential backoff."""
    db_url = os.environ.get("DATABASE_URL", "")
    api_key = os.environ.get("FIREFLIES_API_KEY", "")
    if not db_url or not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="DATABASE_URL and FIREFLIES_API_KEY required",
        )

    store = MeetingStore(db_url)
    await store.initialize()
    try:
        # Clean up any stuck syncing meetings first
        cleaned = await store.cleanup_stuck_syncing(older_than_minutes=30)

        retryable = await store.get_retryable_meetings(limit=20)
        if not retryable:
            return {
                "retried": 0,
                "succeeded": 0,
                "failed_again": 0,
                "cleaned_stuck": cleaned,
                "message": "No meetings eligible for retry",
            }

        succeeded = 0
        failed_again = 0
        for meeting in retryable:
            ff_id = meeting.get("fireflies_transcript_id")
            mid = str(meeting["id"])
            if not ff_id:
                continue
            try:
                await _download_meeting_data(ff_id, known_meeting_id=mid)
                succeeded += 1
            except Exception:
                failed_again += 1
                logger.warning(
                    "retry_failed_again", meeting_id=mid, ff_id=ff_id, exc_info=True,
                )

        return {
            "retried": len(retryable),
            "succeeded": succeeded,
            "failed_again": failed_again,
            "cleaned_stuck": cleaned,
        }
    finally:
        await store.close()


@router.post(
    "/reset/{meeting_id}",
    summary="Reset a meeting for full re-sync",
    description="Clears all sentences, insights, and speakers for a meeting "
    "then resets its sync status so it will be re-synced on the next sync-all.",
)
async def reset_meeting(meeting_id: str) -> dict[str, Any]:
    """Reset a meeting's synced data so it can be re-synced from scratch."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")

    store = MeetingStore(db_url)
    await store.initialize()
    try:
        found = await store.reset_meeting_for_resync(meeting_id)
        if not found:
            raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")
        return {"meeting_id": meeting_id, "status": "reset", "message": "Ready for re-sync"}
    finally:
        await store.close()


@router.get(
    "/failed",
    summary="List failed and retryable meetings",
    description="Returns meetings that failed to sync, including retry count and next retry time.",
)
async def list_failed_meetings() -> dict[str, Any]:
    """List meetings with failed sync status."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")

    store = MeetingStore(db_url)
    await store.initialize()
    try:
        rows = await store._pool.fetch(
            """SELECT id, title, fireflies_transcript_id, sync_status, sync_error,
                      retry_count, last_retry_at, next_retry_at
               FROM meetings
               WHERE sync_status = 'failed'
               ORDER BY next_retry_at ASC NULLS FIRST
               LIMIT 100"""
        )
        meetings = []
        max_retries = MeetingStore._MAX_RETRIES
        for r in rows:
            m = dict(r)
            m["id"] = str(m["id"])
            m["retryable"] = (
                m["retry_count"] < max_retries
                and (m["next_retry_at"] is None or m["next_retry_at"] <= datetime.now(UTC))
            )
            m["last_retry_at"] = str(m["last_retry_at"]) if m["last_retry_at"] else None
            m["next_retry_at"] = str(m["next_retry_at"]) if m["next_retry_at"] else None
            meetings.append(m)
        return {
            "total": len(meetings),
            "retryable": sum(1 for m in meetings if m["retryable"]),
            "exhausted": sum(1 for m in meetings if not m["retryable"]),
            "meetings": meetings,
        }
    finally:
        await store.close()


@router.post(
    "/transcripts",
    response_model=TranscriptsResponse,
    summary="Get past transcripts from Fireflies",
    description="Query Fireflies GraphQL API for past meeting transcripts",
)
async def get_past_transcripts(
    request: GetTranscriptsRequest,
    ff_config: FirefliesSettings = Depends(get_fireflies_config),
):
    """
    Get past transcripts from Fireflies.

    Returns a list of past meeting transcripts with title, date, duration, etc.

    API key is optional - will use .env configuration if not provided.
    """
    api_key = request.api_key or ff_config.api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fireflies API key required. Provide in request or set FIREFLIES_API_KEY in .env",
        )

    try:
        client = FirefliesClient(api_key=api_key)

        transcripts = await client.get_transcripts(
            limit=request.limit,
            skip=request.skip,
        )

        await client.close()

        return TranscriptsResponse(
            success=True,
            transcripts=transcripts,
            count=len(transcripts),
        )

    except FirefliesAPIError as e:
        logger.error("fireflies_api_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fireflies API error: {e!s}",
        ) from e
    except Exception as e:
        logger.exception("get_transcripts_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transcripts: {e!s}",
        ) from e


class GetTranscriptDetailRequest(BaseModel):
    """Request to get transcript details"""

    api_key: str | None = None


@router.post(
    "/transcript/{transcript_id}",
    summary="Get transcript detail with sentences",
    description="Get full transcript including individual sentences",
)
async def get_transcript_detail(
    transcript_id: str,
    request: GetTranscriptDetailRequest,
    ff_config: FirefliesSettings = Depends(get_fireflies_config),
):
    """
    Get detailed transcript including sentences.

    Returns the full transcript with all sentences, speakers, and timestamps.
    """
    api_key = request.api_key or ff_config.api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fireflies API key required. Provide in request or set FIREFLIES_API_KEY in .env",
        )

    try:
        client = FirefliesClient(api_key=api_key)

        transcript = await client.get_transcript_detail(transcript_id)

        await client.close()

        if not transcript:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcript {transcript_id} not found",
            )

        return {
            "success": True,
            "transcript": transcript,
        }

    except HTTPException:
        raise
    except FirefliesAPIError as e:
        logger.error("fireflies_api_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fireflies API error: {e!s}",
        ) from e
    except Exception as e:
        logger.exception("get_transcript_detail_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transcript: {e!s}",
        ) from e


# =============================================================================
# Import Transcript to Local Database (Session-Based Pipeline)
# =============================================================================


class ImportTranscriptRequest(BaseModel):
    """Request to import a Fireflies transcript to local database"""

    api_key: str | None = None
    include_translations: bool = True  # Default to including translations
    target_language: str | None = Field(default="en", description="Target language for translation")
    glossary_id: str | None = None
    domain: str | None = None


class ImportProgress(BaseModel):
    """Progress update for import"""

    session_id: str
    total_sentences: int
    processed: int
    translations_completed: int
    errors: int
    status: str


@router.post(
    "/import/{transcript_id}",
    summary="Import Fireflies transcript to local database",
    description="Fetch transcript from Fireflies and process through the same pipeline as live data (DRY)",
)
async def import_transcript_to_db(
    transcript_id: str,
    request: ImportTranscriptRequest,
    manager: FirefliesSessionManager = Depends(get_session_manager),
    ff_config: FirefliesSettings = Depends(get_fireflies_config),
):
    """
    Import a Fireflies transcript to local database using the same pipeline as live data.

    This ensures:
    - Same database storage logic (BotSession, Transcript, Translation)
    - Same translation flow with context windows
    - Same glossary application
    - Consistent data format across live and imported data

    The import processes each sentence through the TranscriptionPipelineCoordinator,
    which handles all the orchestration DRY.
    """
    api_key = request.api_key or ff_config.api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fireflies API key required",
        )

    try:
        # Step 1: Fetch FULL transcript from Fireflies (with ai_filters, analytics, etc.)
        client = FirefliesClient(api_key=api_key)
        full_result = await client.download_full_transcript(transcript_id)
        await client.close()

        if not full_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcript {transcript_id} not found in Fireflies",
            )

        transcript_data = full_result["transcript"]
        sentences = full_result.get("sentences", [])
        insights = full_result.get("insights", [])

        if not sentences:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transcript has no sentences to import",
            )

        transcript_title = transcript_data.get("title", f"Fireflies Import: {transcript_id}")
        target_lang = request.target_language or "en"

        logger.info("import_starting", sentence_count=len(sentences), transcript_id=transcript_id)

        # Step 2: Get database manager and translation client
        data_pipeline = get_data_pipeline()
        db_manager = data_pipeline.db_manager if data_pipeline else None

        # Get translation clients if translations requested
        translation_client = None
        llm_client = None

        if request.include_translations:
            try:
                llm_client = get_fireflies_llm_client()
                await llm_client.connect()
                logger.info("import_llm_client_connected")
            except Exception as e:
                logger.warning("import_llm_client_unavailable", error=str(e))
                # Falling back to new TranslationService singleton when LLM client unavailable.
                try:
                    from dependencies import get_translation_service_client
                    translation_client = get_translation_service_client()
                except Exception as e2:
                    logger.warning("import_translation_client_unavailable", error=str(e2))

        meeting_db_id: str | None = None
        store: MeetingStore | None = None

        if os.environ.get("DATABASE_URL", ""):
            store = await manager._get_meeting_store()
        if store:
            meeting = await store.get_meeting_by_ff_id(transcript_id)
            if meeting:
                meeting_db_id = str(meeting["id"])
                await store.update_meeting(
                    meeting_db_id,
                    target_languages=[target_lang],
                    meeting_metadata={"ingest_mode": "fireflies_import"},
                )
            else:
                meeting_db_id = await store.create_meeting(
                    fireflies_transcript_id=transcript_id,
                    title=transcript_title,
                    meeting_link=transcript_data.get("meeting_link"),
                    organizer_email=transcript_data.get("organizer_email"),
                    participants=transcript_data.get("participants"),
                    target_languages=[target_lang],
                    meeting_metadata={"ingest_mode": "fireflies_import"},
                    source="fireflies",
                    status="completed",
                )

        # Step 3: Create import session (uses same pipeline as live sessions)
        session_id, coordinator = await manager.create_import_session(
            transcript_id=transcript_id,
            transcript_title=transcript_title,
            target_languages=[target_lang],
            meeting_db_id=meeting_db_id,
            glossary_id=request.glossary_id,
            domain=request.domain,
            db_manager=db_manager,
            translation_client=translation_client,
            llm_client=llm_client,
        )

        # Step 3b: Store Fireflies insights (summary, analytics, attendance, etc.)
        insights_stored = 0
        if insights and db_manager and store and meeting_db_id:
            try:
                    for insight in insights:
                        await store.store_insight(
                            meeting_id=meeting_db_id,
                            insight_type=insight["type"],
                            content=insight["content"],
                            source="fireflies",
                        )
                        insights_stored += 1

                    logger.info(
                        "import_insights_stored",
                        transcript_id=transcript_id,
                        insights_stored=insights_stored,
                    )
            except Exception as e:
                logger.error(
                    "import_insights_storage_failed",
                    transcript_id=transcript_id,
                    error=str(e),
                )

        # Step 4: Process each sentence through the pipeline
        # This is the same flow as live data - DRY!
        processed = 0
        errors = 0

        for i, sentence in enumerate(sentences):
            try:
                # Add index for chunk_id generation
                sentence["index"] = i
                sentence["transcript_id"] = transcript_id

                # Process through coordinator (same as live chunks)
                await coordinator.process_raw_chunk(sentence)
                processed += 1

                # Log progress every 10 sentences
                if (i + 1) % 10 == 0:
                    logger.info("import_progress", current=i + 1, total=len(sentences))

            except Exception as e:
                logger.error("import_sentence_processing_error", index=i, error=str(e))
                errors += 1

        # Step 5: Finalize the session
        final_stats = await manager.finalize_import_session(session_id)

        logger.info(
            "import_complete",
            transcript_id=transcript_id,
            processed=processed,
            translated=final_stats.get("translations_completed", 0),
            errors=errors,
        )

        return {
            "success": True,
            "session_id": session_id,
            "transcript_id": transcript_id,
            "title": transcript_title,
            "total_sentences": len(sentences),
            "processed": processed,
            "translations_completed": final_stats.get("translations_completed", 0),
            "sentences_stored": final_stats.get("sentences_produced", 0),
            "errors": errors,
            "insights_stored": insights_stored,
            "target_language": target_lang,
            "glossary_id": request.glossary_id,
            "pipeline_stats": final_stats,
            "message": f"Successfully imported {processed} sentences and {insights_stored} insights",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("import_transcript_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import transcript: {e!s}",
        ) from e


# =============================================================================
# Dashboard Configuration
# =============================================================================
# NOTE: This endpoint proxies to the centralized /api/system/ui-config endpoint.
# All configuration is centralized in config/system_constants.py.
# DO NOT add duplicate definitions here.


@router.get(
    "/dashboard/config",
    summary="Get dashboard configuration",
    description="Returns centralized configuration for dashboards. Proxies to /api/system/ui-config.",
)
async def get_dashboard_config():
    """
    Get centralized dashboard configuration.

    This endpoint proxies to the system-wide UI config endpoint.
    All dashboards should use this for consistent configuration.

    Returns:
    - Supported languages with codes and display names
    - Glossary domain options
    - Default values
    - Available translation models (if service is available)
    - Prompt templates (if available)
    """
    # Import from centralized system constants
    from system_constants import (
        DEFAULT_CONFIG,
        GLOSSARY_DOMAINS,
        PROMPT_TEMPLATE_VARIABLES,
        SUPPORTED_LANGUAGES,
    )

    config = {
        "languages": SUPPORTED_LANGUAGES,
        "language_codes": [lang["code"] for lang in SUPPORTED_LANGUAGES],
        "domains": GLOSSARY_DOMAINS,
        "defaults": DEFAULT_CONFIG,
        "prompt_variables": PROMPT_TEMPLATE_VARIABLES,
    }

    # get_models() not yet on new TranslationService; returning config model for now.
    try:
        from dependencies import get_translation_service_client
        svc = get_translation_service_client()
        models = [{"name": svc.config.model, "description": "Configured Ollama model"}]
        config["translation_models"] = models
        config["translation_service_available"] = True
    except Exception as e:
        logger.warning("translation_service_unavailable", error=str(e))
        config["translation_models"] = []
        config["translation_service_available"] = False
        config["translation_service_error"] = str(e)

    # Fetch prompts from translation service (NO FALLBACK)
    try:
        import aiohttp

        from config import get_settings

        settings = get_settings()
        translation_url = getattr(settings, "translation_service_url", "http://localhost:5003")

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{translation_url}/prompts", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp,
        ):
            if resp.status == 200:
                prompts_data = await resp.json()
                config["prompt_templates"] = prompts_data.get("prompts", [])
                config["prompts_available"] = True
            else:
                config["prompt_templates"] = []
                config["prompts_available"] = False
                config["prompts_error"] = f"HTTP {resp.status}"
    except Exception as e:
        logger.warning("prompt_fetch_failed", error=str(e))
        config["prompt_templates"] = []
        config["prompts_available"] = False
        config["prompts_error"] = str(e)

    return config


# =============================================================================
# Demo Mode Endpoints
# =============================================================================


@router.post(
    "/demo/start",
    status_code=status.HTTP_200_OK,
    summary="Start Fireflies demo mode",
    description="Launch a mock Fireflies server and create a live session with simulated conversation",
)
async def start_demo(
    manager: FirefliesSessionManager = Depends(get_session_manager),
    ff_config: FirefliesSettings = Depends(get_fireflies_config),
    mode: str = "passthrough",
    speed: float = 1.0,
):
    """
    Start demo mode:
    1. Launches a mock Fireflies server on port 8090
    2. Creates a real Fireflies session pointing to the local mock
    3. The full pipeline runs: mock → Socket.IO → orchestration → captions WebSocket

    Args:
        mode: "passthrough", "pretranslated", or "replay" (real captured meeting data)
        speed: For replay mode, playback speed multiplier (2.0 = 2x faster)
    """
    from services.demo_manager import DEMO_API_KEY, get_demo_manager

    demo = get_demo_manager()

    if demo.active:
        return {
            "success": True,
            "message": "Demo already running",
            "session_id": demo.session_id,
            "transcript_id": demo.transcript_id,
            "speakers": demo.get_status()["speakers"],
            "mode": demo.mode,
        }

    try:
        # Start mock server with conversation scenario
        demo_info = await demo.start(
            speakers=["Alice Chen", "Bob Martinez", "Charlie Kim"],
            num_exchanges=30,
            chunk_delay_ms=2000.0,
            mode=mode,
            speed_multiplier=speed,
        )

        # Create a real Fireflies session pointing to the local mock
        config = FirefliesSessionConfig(
            api_key=DEMO_API_KEY,
            transcript_id=demo_info["transcript_id"],
            target_languages=ff_config.default_target_languages or ["es"],
            api_base_url=demo_info["base_url"],
        )

        # Get optional services for the pipeline
        data_pipeline = get_data_pipeline()
        db_manager = data_pipeline.db_manager if data_pipeline else None

        llm_client = None
        try:
            llm_client = get_fireflies_llm_client()
            await llm_client.connect()
        except Exception as e:
            logger.warning("demo_llm_client_unavailable", error=str(e))

        try:
            meeting_intelligence = get_meeting_intelligence_service()
        except Exception as e:
            meeting_intelligence = None
            logger.warning("demo_meeting_intelligence_unavailable", error=str(e))

        try:
            event_publisher = get_event_publisher()
        except Exception as e:
            event_publisher = None
            logger.warning("demo_event_publisher_unavailable", error=str(e))

        glossary_service = None
        try:
            glossary_service = get_glossary_pipeline_adapter()
        except Exception as e:
            logger.warning("demo_glossary_unavailable", error=str(e))

        session = await manager.create_session(
            config,
            db_manager=db_manager,
            llm_client=llm_client,
            meeting_intelligence=meeting_intelligence,
            event_publisher=event_publisher,
            glossary_service=glossary_service,
        )

        demo.session_id = session.session_id

        # For pretranslated mode, start background injection task
        if mode == "pretranslated":
            from routers.captions import get_caption_buffer, get_connection_manager

            caption_buffer = get_caption_buffer(session.session_id)
            ws_manager = get_connection_manager()
            demo.start_pretranslated_injection(caption_buffer, ws_manager)
            logger.info(
                "demo_pretranslated_injection_started",
                session_id=session.session_id,
            )

        logger.info("demo_session_created", session_id=session.session_id, mode=mode)

        return {
            "success": True,
            "message": "Demo started successfully",
            "session_id": session.session_id,
            "transcript_id": demo_info["transcript_id"],
            "speakers": demo_info["speakers"],
            "num_exchanges": demo_info["num_exchanges"],
            "chunk_delay_ms": demo_info["chunk_delay_ms"],
            "mode": mode,
        }

    except Exception as e:
        # Clean up on failure
        await demo.stop()
        logger.exception("demo_start_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start demo: {e!s}",
        ) from e


@router.post(
    "/demo/stop",
    status_code=status.HTTP_200_OK,
    summary="Stop Fireflies demo mode",
    description="Disconnect the demo session and stop the mock server",
)
async def stop_demo(
    manager: FirefliesSessionManager = Depends(get_session_manager),
):
    """Stop demo mode: disconnect session and shut down mock server."""
    from services.demo_manager import get_demo_manager

    demo = get_demo_manager()

    if not demo.active:
        return {"success": True, "message": "Demo is not running"}

    # Disconnect the Fireflies session
    if demo.session_id:
        await manager.disconnect_session(demo.session_id)

    # Stop the mock server
    await demo.stop()

    return {"success": True, "message": "Demo stopped"}


@router.get(
    "/demo/status",
    status_code=status.HTTP_200_OK,
    summary="Get demo mode status",
    description="Check if demo mode is active and get session info",
)
async def get_demo_status():
    """Get current demo status."""
    from services.demo_manager import get_demo_manager

    demo = get_demo_manager()
    return demo.get_status()
