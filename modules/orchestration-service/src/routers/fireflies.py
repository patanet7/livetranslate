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
)
from dependencies import (
    get_data_pipeline,
    get_event_publisher,
    get_fireflies_llm_client,
    get_meeting_intelligence_service,
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
                        title=None,  # Updated later from Fireflies data
                        source="fireflies",
                    )
                    session.meeting_db_id = meeting_id
                    logger.info(
                        "meeting_record_created",
                        meeting_id=meeting_id,
                        transcript_id=config.transcript_id,
                    )
                except Exception as e:
                    logger.error("meeting_record_creation_failed", error=str(e))

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

        async def _broadcast_caption_to_ws(event_type: str, caption) -> None:
            """Forward caption events from pipeline to WebSocket clients."""
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

        coordinator.on_caption_event(_broadcast_caption_to_ws)

        # Bridge interim (word-by-word) caption updates to WebSocket clients
        async def handle_live_update(chunk: FirefliesChunk, is_final: bool) -> None:
            """Broadcast interim caption updates to WebSocket clients."""
            await ws_manager.broadcast_to_session(
                session_id,
                {
                    "event": "interim_caption",
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "speaker_name": chunk.speaker_name,
                    "speaker_color": None,  # CaptionBuffer assigns colors for final captions
                    "is_final": is_final,
                },
            )

        logger.info(f"Pipeline coordinator initialized for session {session_id}")

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

            # Process through pipeline coordinator (DRY - handles all orchestration)
            try:
                await coordinator.process_raw_chunk(chunk)
                # Update session stats from coordinator
                stats = coordinator.get_stats()
                session.sentences_produced = stats.get("sentences_produced", 0)
                session.translations_completed = stats.get("translations_completed", 0)
            except Exception as e:
                logger.error(f"Pipeline error for session {session_id}: {e}")
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
                        logger.error(
                            "chunk_storage_failed",
                            error=str(e),
                            chunk_id=chunk.chunk_id,
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
                on_live_update=handle_live_update,
                auto_reconnect=True,
            )
        except Exception as e:
            session.connection_status = FirefliesConnectionStatus.ERROR
            session.last_error = str(e)
            logger.error(f"Failed to connect Fireflies session {session_id}: {e}")

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
                logger.info(f"Flushed pipeline for session {session_id}")
            except Exception as e:
                logger.error(f"Error flushing pipeline: {e}")

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
                    logger.error("meeting_completion_failed", error=str(e))

        logger.info(f"Disconnected Fireflies session: {session_id}")
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

        logger.info(f"Import session created: {session_id} for transcript {transcript_id}")

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

        logger.info(f"Import session finalized: {session_id}, stats: {stats}")

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
                        except Exception:
                            pass  # Translation will be skipped

                        try:
                            meeting_intelligence = get_meeting_intelligence_service()
                        except Exception:
                            meeting_intelligence = None

                        try:
                            event_publisher = get_event_publisher()
                        except Exception:
                            event_publisher = None

                        await manager.create_session(
                            config,
                            db_manager=db_manager,
                            llm_client=llm_client,
                            meeting_intelligence=meeting_intelligence,
                            event_publisher=event_publisher,
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


@router.on_event("startup")
async def _start_auto_connect() -> None:
    """Start auto-connect polling if FIREFLIES_AUTO_CONNECT=true."""
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


@router.on_event("shutdown")
async def _stop_auto_connect() -> None:
    """Cancel the auto-connect polling task on shutdown."""
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

    # Sentence aggregation config (all optional, uses .env defaults)
    pause_threshold_ms: float | None = Field(default=None)
    max_buffer_words: int | None = Field(default=None)
    context_window_size: int | None = Field(default=None)


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
        )

        logger.info(
            f"Session config: languages={config.target_languages}, "
            f"model={config.translation_model or 'default'}"
        )

        # Get database manager for transcript/translation storage
        data_pipeline = get_data_pipeline()
        db_manager = data_pipeline.db_manager if data_pipeline else None

        if db_manager:
            logger.info("Database manager connected - transcripts and translations will be stored")
        else:
            logger.warning(
                "No database manager - transcripts and translations will NOT be persisted"
            )

        # Get optional intelligence and event publisher services
        try:
            meeting_intelligence = get_meeting_intelligence_service()
        except Exception:
            meeting_intelligence = None

        try:
            event_publisher = get_event_publisher()
        except Exception:
            event_publisher = None

        # Create LLM client for translation pipeline
        llm_client = None
        try:
            llm_client = get_fireflies_llm_client()
            await llm_client.connect()
            logger.info("LLM client connected for Fireflies translation pipeline")
        except Exception as e:
            logger.warning(
                f"LLM client unavailable for Fireflies pipeline: {e}. "
                "Translations will be skipped but transcripts will still be stored."
            )

        # Create session with transcript handling and database persistence
        session = await manager.create_session(
            config,
            db_manager=db_manager,
            meeting_intelligence=meeting_intelligence,
            event_publisher=event_publisher,
            llm_client=llm_client,
        )

        logger.info(
            f"Created Fireflies session: {session.session_id} "
            f"for transcript: {request.transcript_id}"
        )

        return FirefliesConnectResponse(
            success=True,
            message="Connected to Fireflies realtime API",
            session_id=session.session_id,
            connection_status=session.connection_status,
            transcript_id=request.transcript_id,
        )

    except FirefliesAPIError as e:
        logger.error(f"Fireflies API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fireflies API error: {e!s}",
        ) from e
    except Exception as e:
        logger.exception(f"Failed to connect to Fireflies: {e}")
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
    summary="Get all active Fireflies sessions",
)
async def get_sessions(
    manager: FirefliesSessionManager = Depends(get_session_manager),
):
    """Get all active Fireflies sessions"""
    sessions = manager.get_all_sessions()

    return [
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
        for s in sessions
    ]


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get a specific Fireflies session",
)
async def get_session(
    session_id: str,
    manager: FirefliesSessionManager = Depends(get_session_manager),
):
    """Get details of a specific Fireflies session"""
    session = manager.get_session(session_id)

    if not session:
        from errors import NotFoundError

        raise NotFoundError("Session", session_id)

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
        logger.error(f"Fireflies API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fireflies API error: {e!s}",
        ) from e
    except Exception as e:
        logger.exception(f"Failed to get active meetings: {e}")
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
# Post-Meeting Webhook & Full Data Download
# =============================================================================


async def _download_meeting_data(fireflies_transcript_id: str) -> None:
    """Download full transcript and insights from Fireflies after meeting completes.

    Called as a background task when the Fireflies webhook fires or when
    auto-connect polling detects a meeting has ended.  Downloads all data
    via the expanded GraphQL query and stores everything in PostgreSQL
    via MeetingStore.
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
        # Download full transcript from Fireflies
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

        # Initialize meeting store
        store = MeetingStore(db_url)
        await store.initialize()

        try:
            # Find or create meeting record
            meeting = await store.get_meeting_by_ff_id(fireflies_transcript_id)
            if meeting:
                meeting_db_id = str(meeting["id"])
                await store.complete_meeting(meeting_db_id)
            else:
                transcript_data = result["transcript"]
                meeting_db_id = await store.create_meeting(
                    fireflies_transcript_id=fireflies_transcript_id,
                    title=transcript_data.get("title"),
                    meeting_link=transcript_data.get("meeting_link"),
                    organizer_email=transcript_data.get("organizer_email"),
                    participants=transcript_data.get("participants"),
                    source="fireflies",
                    status="completed",
                )

            # Store all insights
            for insight in result.get("insights", []):
                await store.store_insight(
                    meeting_id=meeting_db_id,
                    insight_type=insight["type"],
                    content=insight["content"],
                    source="fireflies",
                )

            # Store sentences with ai_filters
            for sentence in result.get("sentences", []):
                sentence_id = await store.store_sentence(
                    meeting_id=meeting_db_id,
                    text=sentence.get("text", ""),
                    speaker_name=sentence.get("speaker_name", "Unknown"),
                    start_time=float(sentence.get("start_time", 0)),
                    end_time=float(sentence.get("end_time", 0)),
                    boundary_type="fireflies_download",
                )
                # Store ai_filters as per-sentence insight
                if sentence.get("ai_filters"):
                    await store.store_insight(
                        meeting_id=meeting_db_id,
                        insight_type="sentence_ai_filter",
                        content={
                            "sentence_id": sentence_id,
                            "filters": sentence["ai_filters"],
                        },
                        source="fireflies",
                    )

            # Store speaker analytics
            for insight in result.get("insights", []):
                if insight.get("type") == "speaker_analytics":
                    for speaker_data in insight.get("content") or []:
                        if isinstance(speaker_data, dict):
                            await store.store_speaker(
                                meeting_id=meeting_db_id,
                                speaker_name=speaker_data.get("name", "Unknown"),
                                talk_time_seconds=float(
                                    speaker_data.get("duration", 0)
                                ),
                                word_count=int(speaker_data.get("word_count", 0)),
                                analytics=speaker_data,
                            )

            logger.info(
                "meeting_data_downloaded",
                transcript_id=fireflies_transcript_id,
                meeting_db_id=meeting_db_id,
                insight_count=len(result.get("insights", [])),
                sentence_count=len(result.get("sentences", [])),
            )
        finally:
            await store.close()

    except Exception as e:
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
                    except Exception:
                        pass  # Translation will be skipped

                    try:
                        meeting_intelligence = get_meeting_intelligence_service()
                    except Exception:
                        meeting_intelligence = None

                    try:
                        event_publisher = get_event_publisher()
                    except Exception:
                        event_publisher = None

                    manager = get_session_manager()
                    await manager.create_session(
                        config,
                        db_manager=db_manager,
                        llm_client=llm_client,
                        meeting_intelligence=meeting_intelligence,
                        event_publisher=event_publisher,
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
        logger.error(f"Fireflies API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fireflies API error: {e!s}",
        ) from e
    except Exception as e:
        logger.exception(f"Failed to get transcripts: {e}")
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
        logger.error(f"Fireflies API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fireflies API error: {e!s}",
        ) from e
    except Exception as e:
        logger.exception(f"Failed to get transcript: {e}")
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
        # Step 1: Fetch transcript from Fireflies
        client = FirefliesClient(api_key=api_key)
        transcript_data = await client.get_transcript_detail(transcript_id)
        await client.close()

        if not transcript_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcript {transcript_id} not found in Fireflies",
            )

        sentences = transcript_data.get("sentences", [])
        if not sentences:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transcript has no sentences to import",
            )

        transcript_title = transcript_data.get("title", f"Fireflies Import: {transcript_id}")
        target_lang = request.target_language or "en"

        logger.info(f"Starting import of {len(sentences)} sentences from {transcript_id}")

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
                logger.info("LLM client connected for Fireflies import pipeline")
            except Exception as e:
                logger.warning(f"LLM client unavailable for import: {e}")
                # Fall back to legacy TranslationServiceClient
                try:
                    from clients.translation_service_client import TranslationServiceClient

                    translation_client = TranslationServiceClient()
                except Exception as e2:
                    logger.warning(f"Translation client also unavailable: {e2}")

        # Step 3: Create import session (uses same pipeline as live sessions)
        session_id, coordinator = await manager.create_import_session(
            transcript_id=transcript_id,
            transcript_title=transcript_title,
            target_languages=[target_lang],
            glossary_id=request.glossary_id,
            domain=request.domain,
            db_manager=db_manager,
            translation_client=translation_client,
            llm_client=llm_client,
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
                    logger.info(f"Import progress: {i + 1}/{len(sentences)} sentences")

            except Exception as e:
                logger.error(f"Error processing sentence {i}: {e}")
                errors += 1

        # Step 5: Finalize the session
        final_stats = await manager.finalize_import_session(session_id)

        logger.info(
            f"Import complete for {transcript_id}: "
            f"{processed} processed, {final_stats.get('translations_completed', 0)} translated, "
            f"{errors} errors"
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
            "target_language": target_lang,
            "glossary_id": request.glossary_id,
            "pipeline_stats": final_stats,
            "message": f"Successfully imported {processed} sentences to local database",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to import transcript: {e}")
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

    # Fetch available models from translation service (NO FALLBACK)
    try:
        from clients.translation_service_client import TranslationServiceClient

        client = TranslationServiceClient()
        models = await client.get_models()
        config["translation_models"] = models
        config["translation_service_available"] = True
    except Exception as e:
        logger.warning(f"Translation service unavailable: {e}")
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
        logger.warning(f"Could not fetch prompts: {e}")
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
):
    """
    Start demo mode:
    1. Launches a mock Fireflies server on port 8090
    2. Creates a real Fireflies session pointing to the local mock
    3. The full pipeline runs: mock → Socket.IO → orchestration → captions WebSocket
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
        }

    try:
        # Start mock server with conversation scenario
        demo_info = await demo.start(
            speakers=["Alice Chen", "Bob Martinez", "Charlie Kim"],
            num_exchanges=30,
            chunk_delay_ms=2000.0,
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
            logger.warning(f"LLM client unavailable for demo: {e}")

        try:
            meeting_intelligence = get_meeting_intelligence_service()
        except Exception:
            meeting_intelligence = None

        try:
            event_publisher = get_event_publisher()
        except Exception:
            event_publisher = None

        session = await manager.create_session(
            config,
            db_manager=db_manager,
            llm_client=llm_client,
            meeting_intelligence=meeting_intelligence,
            event_publisher=event_publisher,
        )

        demo.session_id = session.session_id

        logger.info(f"Demo session created: {session.session_id}")

        return {
            "success": True,
            "message": "Demo started successfully",
            "session_id": session.session_id,
            "transcript_id": demo_info["transcript_id"],
            "speakers": demo_info["speakers"],
            "num_exchanges": demo_info["num_exchanges"],
            "chunk_delay_ms": demo_info["chunk_delay_ms"],
        }

    except Exception as e:
        # Clean up on failure
        await demo.stop()
        logger.exception(f"Failed to start demo: {e}")
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
