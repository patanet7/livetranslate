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

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from models.fireflies import (
    FirefliesChunk,
    FirefliesSession,
    FirefliesSessionConfig,
    FirefliesConnectionStatus,
    FirefliesConnectResponse,
    ActiveMeetingsResponse,
)
from clients.fireflies_client import (
    FirefliesClient,
    FirefliesAPIError,
)
from config import get_settings, FirefliesSettings
from dependencies import get_data_pipeline

# Pipeline imports (DRY coordinator)
from services.pipeline import (
    TranscriptionPipelineCoordinator,
    PipelineConfig,
    FirefliesChunkAdapter,
)
from services.caption_buffer import CaptionBuffer

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/fireflies", tags=["fireflies"])


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
        self._sessions: Dict[str, FirefliesSession] = {}
        self._clients: Dict[str, FirefliesClient] = {}
        self._transcript_handlers: Dict[str, Any] = {}
        # Pipeline coordinators (DRY - same for all sources)
        self._coordinators: Dict[str, TranscriptionPipelineCoordinator] = {}
        self._caption_buffers: Dict[str, CaptionBuffer] = {}

    def get_session(self, session_id: str) -> Optional[FirefliesSession]:
        """Get session by ID"""
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> List[FirefliesSession]:
        """Get all active sessions"""
        return list(self._sessions.values())

    def get_client(self, session_id: str) -> Optional[FirefliesClient]:
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
        simple_translation_client=None,
        db_manager=None,
        obs_output=None,
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

        # Create Fireflies client
        client = FirefliesClient(api_key=config.api_key)

        # Store session and client
        self._sessions[session_id] = session
        self._clients[session_id] = client

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

        # Create coordinator with Fireflies adapter
        coordinator = TranscriptionPipelineCoordinator(
            config=pipeline_config,
            adapter=FirefliesChunkAdapter(),
            glossary_service=glossary_service,
            translation_client=translation_client,
            simple_translation_client=simple_translation_client,
            caption_buffer=caption_buffer,
            db_manager=db_manager,
            obs_output=obs_output,
        )

        # Initialize the pipeline
        await coordinator.initialize()
        self._coordinators[session_id] = coordinator

        logger.info(f"Pipeline coordinator initialized for session {session_id}")

        # =====================================================================
        # Connect to realtime with callbacks
        # =====================================================================

        async def handle_transcript(chunk: FirefliesChunk):
            session.chunks_received += 1
            session.last_chunk_time = datetime.now(timezone.utc)
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

            # Call user callback (after pipeline processing)
            if on_transcript:
                await on_transcript(session_id, chunk)

        async def handle_status(
            new_status: FirefliesConnectionStatus, message: Optional[str]
        ):
            session.connection_status = new_status
            if new_status == FirefliesConnectionStatus.CONNECTED:
                session.connected_at = datetime.now(timezone.utc)
            elif new_status == FirefliesConnectionStatus.ERROR:
                session.error_count += 1
                session.last_error = message

            if on_status_change:
                await on_status_change(session_id, new_status, message)

        async def handle_error(message: str, exception: Optional[Exception]):
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

        logger.info(f"Disconnected Fireflies session: {session_id}")
        return True

    def get_coordinator(self, session_id: str) -> Optional[TranscriptionPipelineCoordinator]:
        """Get pipeline coordinator for session"""
        return self._coordinators.get(session_id)

    def get_caption_buffer(self, session_id: str) -> Optional[CaptionBuffer]:
        """Get caption buffer for session"""
        return self._caption_buffers.get(session_id)


# Global session manager instance
_session_manager: Optional[FirefliesSessionManager] = None


def get_session_manager() -> FirefliesSessionManager:
    """Get or create the session manager singleton"""
    global _session_manager
    if _session_manager is None:
        _session_manager = FirefliesSessionManager()
    return _session_manager


def get_fireflies_config() -> FirefliesSettings:
    """Get Fireflies configuration from settings"""
    return get_settings().fireflies


def get_api_key_from_config() -> Optional[str]:
    """Get API key from config if available"""
    config = get_fireflies_config()
    return config.api_key if config.has_api_key() else None


# =============================================================================
# API Request/Response Models
# =============================================================================


class ConnectRequest(BaseModel):
    """Request to connect to Fireflies realtime"""

    api_key: Optional[str] = Field(
        default=None,
        description="Fireflies API key (optional, uses .env if not provided)",
    )
    transcript_id: str = Field(..., description="Transcript ID from active_meetings")
    target_languages: Optional[List[str]] = Field(
        default=None,
        description="Target languages for translation (optional, uses .env default)",
    )
    glossary_id: Optional[str] = Field(default=None, description="Optional glossary ID")
    domain: Optional[str] = Field(
        default=None, description="Domain for glossary filtering"
    )
    translation_model: Optional[str] = Field(
        default=None,
        description="Translation model/service to use (ollama, groq, etc.)",
    )

    # Sentence aggregation config (all optional, uses .env defaults)
    pause_threshold_ms: Optional[float] = Field(default=None)
    max_buffer_words: Optional[int] = Field(default=None)
    context_window_size: Optional[int] = Field(default=None)


class SessionResponse(BaseModel):
    """Response with session information"""

    session_id: str
    transcript_id: str
    connection_status: str
    chunks_received: int
    sentences_produced: int
    translations_completed: int
    speakers_detected: List[str]
    connected_at: Optional[datetime]
    error_count: int
    last_error: Optional[str]


class DisconnectRequest(BaseModel):
    """Request to disconnect a session"""

    session_id: str = Field(..., description="Session ID to disconnect")


class GetMeetingsRequest(BaseModel):
    """Request to get active meetings"""

    api_key: Optional[str] = Field(
        default=None,
        description="Fireflies API key (optional, uses .env if not provided)",
    )
    email: Optional[str] = Field(default=None, description="Filter by email")


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
            target_languages=request.target_languages
            or ff_config.default_target_languages,
            glossary_id=request.glossary_id,
            domain=request.domain,
            translation_model=request.translation_model,  # Pass through model selection
            pause_threshold_ms=request.pause_threshold_ms
            or ff_config.pause_threshold_ms,
            max_buffer_words=request.max_buffer_words or ff_config.max_buffer_words,
            context_window_size=request.context_window_size
            or ff_config.context_window_size,
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
            logger.warning("No database manager - transcripts and translations will NOT be persisted")

        # Create session with transcript handling and database persistence
        session = await manager.create_session(
            config,
            db_manager=db_manager,
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
            detail=f"Fireflies API error: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Failed to connect to Fireflies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect: {str(e)}",
        )


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
    response_model=List[SessionResponse],
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
            connection_status=s.connection_status.value,
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
        connection_status=session.connection_status.value,
        chunks_received=session.chunks_received,
        sentences_produced=session.sentences_produced,
        translations_completed=session.translations_completed,
        speakers_detected=session.speakers_detected,
        connected_at=session.connected_at,
        error_count=session.error_count,
        last_error=session.last_error,
    )


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
            detail=f"Fireflies API error: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Failed to get active meetings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get meetings: {str(e)}",
        )


@router.get(
    "/health",
    summary="Check Fireflies integration health",
)
async def health_check(
    manager: FirefliesSessionManager = Depends(get_session_manager),
):
    """Check the health of Fireflies integration"""
    sessions = manager.get_all_sessions()
    connected = [
        s
        for s in sessions
        if s.connection_status == FirefliesConnectionStatus.CONNECTED
    ]

    return {
        "status": "healthy",
        "total_sessions": len(sessions),
        "connected_sessions": len(connected),
        "sessions": [
            {
                "session_id": s.session_id,
                "status": s.connection_status.value if hasattr(s.connection_status, 'value') else str(s.connection_status),
                "chunks_received": s.chunks_received,
            }
            for s in sessions
        ],
    }


# =============================================================================
# Past Transcripts Endpoints
# =============================================================================


class GetTranscriptsRequest(BaseModel):
    """Request to get past transcripts from Fireflies"""

    api_key: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100, description="Max transcripts to return")
    skip: int = Field(default=0, ge=0, description="Pagination offset")


class TranscriptsResponse(BaseModel):
    """Response containing past transcripts"""

    success: bool
    transcripts: List[Dict[str, Any]]
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
            detail=f"Fireflies API error: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Failed to get transcripts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transcripts: {str(e)}",
        )


class GetTranscriptDetailRequest(BaseModel):
    """Request to get transcript details"""

    api_key: Optional[str] = None


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
            detail=f"Fireflies API error: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Failed to get transcript: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transcript: {str(e)}",
        )
