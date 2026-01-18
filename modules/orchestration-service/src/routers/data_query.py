#!/usr/bin/env python3
"""
Data Query API Router

FastAPI router providing comprehensive query endpoints for transcription
and translation data with advanced filtering, timeline reconstruction,
speaker statistics, and full-text search capabilities.

Endpoints:
- GET /api/data/sessions/{session_id}/transcripts - Query transcripts with filters
- GET /api/data/sessions/{session_id}/translations - Query translations
- GET /api/data/sessions/{session_id}/timeline - Get complete timeline
- GET /api/data/sessions/{session_id}/speakers - Get speaker statistics
- GET /api/data/sessions/{session_id}/speakers/{speaker_id} - Get speaker detail
- GET /api/data/sessions/{session_id}/search - Full-text search

Author: LiveTranslate Team
Version: 1.0
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/data", tags=["data-query"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class TranscriptResponse(BaseModel):
    """Transcript response model."""

    transcript_id: str
    session_id: str
    source_type: str
    transcript_text: str
    language_code: str
    start_timestamp: float
    end_timestamp: float
    duration: float
    speaker_id: str | None = None
    speaker_name: str | None = None
    confidence_score: float | None = None
    segment_index: int
    audio_file_id: str | None = None
    created_at: datetime
    metadata: dict[str, Any] = {}


class TranslationResponse(BaseModel):
    """Translation response model."""

    translation_id: str
    session_id: str
    source_transcript_id: str
    translated_text: str
    source_language: str
    target_language: str
    translation_confidence: float | None = None
    translation_service: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    start_timestamp: float
    end_timestamp: float
    duration: float
    created_at: datetime
    metadata: dict[str, Any] = {}


class TimelineEntryResponse(BaseModel):
    """Timeline entry response model."""

    timestamp: float
    duration: float
    entry_type: str  # 'transcript', 'translation'
    content: str
    language: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = {}


class TimelineResponse(BaseModel):
    """Complete timeline response."""

    session_id: str
    total_entries: int
    start_time: float | None = None
    end_time: float | None = None
    entries: list[TimelineEntryResponse]


class SpeakerStatisticsResponse(BaseModel):
    """Speaker statistics response."""

    session_id: str
    speaker_id: str
    speaker_name: str | None = None
    identification_method: str | None = None
    identification_confidence: float | None = None
    total_segments: int
    total_speaking_time: float
    average_confidence: float
    languages_translated_to: int
    total_translations: int


class SpeakersResponse(BaseModel):
    """Multiple speakers response."""

    session_id: str
    total_speakers: int
    speakers: list[SpeakerStatisticsResponse]


class SpeakerDetailResponse(BaseModel):
    """Detailed speaker information."""

    statistics: SpeakerStatisticsResponse
    timeline: list[TimelineEntryResponse]
    recent_transcripts: list[TranscriptResponse]
    recent_translations: list[TranslationResponse]


class SearchResponse(BaseModel):
    """Search results response."""

    session_id: str
    query: str
    total_results: int
    results: list[TranscriptResponse]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str | None = None


# ============================================================================
# DEPENDENCY: DATA PIPELINE
# ============================================================================


_pipeline_instance = None

# Background task tracking (prevents fire-and-forget)
_background_tasks: set = set()


def get_pipeline():
    """Get or create data pipeline instance."""
    global _pipeline_instance

    if _pipeline_instance is None:
        import os

        from pipeline.data_pipeline import create_data_pipeline

        # Load configuration from environment
        db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "livetranslate"),
            "username": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "livetranslate"),
        }

        audio_storage_path = os.getenv("AUDIO_STORAGE_PATH", "/tmp/livetranslate/audio")

        # Create pipeline (synchronous initialization)
        _pipeline_instance = create_data_pipeline(
            db_config=db_config,
            audio_storage_path=audio_storage_path,
            enable_speaker_tracking=True,
            enable_segment_continuity=True,
        )

        # Initialize database manager (needs to be done asynchronously)
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule initialization with proper task tracking
                task = asyncio.create_task(
                    _pipeline_instance.db_manager.initialize()
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
            else:
                # Initialize synchronously
                loop.run_until_complete(_pipeline_instance.db_manager.initialize())
        except Exception as e:
            logger.warning(f"Could not initialize database on startup: {e}")

    return _pipeline_instance


# ============================================================================
# API ENDPOINTS
# ============================================================================


class SessionSummary(BaseModel):
    """Summary information for a transcript session."""

    session_id: str
    source_type: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    transcript_count: int = 0
    translation_count: int = 0
    speaker_count: int = 0
    total_duration: float = 0.0
    languages: list[str] = []
    metadata: dict[str, Any] = {}


class SessionsListResponse(BaseModel):
    """Response containing list of sessions."""

    total: int
    sessions: list[SessionSummary]


@router.get(
    "/sessions",
    response_model=SessionsListResponse,
    summary="List all transcript sessions",
    description="Retrieve a list of all transcript sessions from the database with optional filtering",
)
async def list_sessions(
    source_type: str | None = Query(
        None, description="Filter by source type (fireflies, audio_upload, google_meet)"
    ),
    limit: int = Query(50, description="Maximum sessions to return", ge=1, le=500),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    pipeline=Depends(get_pipeline),
):
    """
    List all transcript sessions from database.

    Returns a list of session summaries with:
    - Session ID and source type
    - Transcript and translation counts
    - Speaker counts
    - Total duration
    - Languages used

    Supports filtering by source type and pagination.
    """
    try:
        # Ensure database is initialized
        if not pipeline.db_manager.db_pool:
            await pipeline.db_manager.initialize()

        # Query for unique sessions with aggregated stats
        # This uses the transcript table to get distinct sessions
        query = """
            SELECT
                t.session_id,
                t.source_type,
                MIN(t.created_at) as created_at,
                MAX(t.created_at) as updated_at,
                COUNT(DISTINCT t.transcript_id) as transcript_count,
                COUNT(DISTINCT tr.translation_id) as translation_count,
                COUNT(DISTINCT t.speaker_id) FILTER (WHERE t.speaker_id IS NOT NULL) as speaker_count,
                COALESCE(SUM(t.end_timestamp - t.start_timestamp), 0) as total_duration,
                ARRAY_AGG(DISTINCT t.language_code) FILTER (WHERE t.language_code IS NOT NULL) as languages,
                MAX(t.processing_metadata) as metadata
            FROM transcripts t
            LEFT JOIN translations tr ON t.session_id = tr.session_id
            WHERE ($1::text IS NULL OR t.source_type = $1)
            GROUP BY t.session_id, t.source_type
            ORDER BY MAX(t.created_at) DESC
            LIMIT $2 OFFSET $3
        """

        async with pipeline.db_manager.db_pool.acquire() as conn:
            rows = await conn.fetch(query, source_type, limit, offset)

            # Get total count for pagination
            count_query = """
                SELECT COUNT(DISTINCT session_id) as total
                FROM transcripts
                WHERE ($1::text IS NULL OR source_type = $1)
            """
            count_row = await conn.fetchrow(count_query, source_type)
            total = count_row["total"] if count_row else 0

        # Convert to response models
        sessions = []
        for row in rows:
            # Extract title from metadata if available
            metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
            title = metadata.get("title") or metadata.get("source_metadata", {}).get("title")

            sessions.append(
                SessionSummary(
                    session_id=row["session_id"],
                    source_type=row["source_type"],
                    title=title,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    transcript_count=row["transcript_count"],
                    translation_count=row["translation_count"],
                    speaker_count=row["speaker_count"],
                    total_duration=float(row["total_duration"]) if row["total_duration"] else 0.0,
                    languages=row["languages"] or [],
                    metadata=metadata,
                )
            )

        logger.info(f"Retrieved {len(sessions)} sessions (total: {total})")
        return SessionsListResponse(total=total, sessions=sessions)

    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/sessions/{session_id}/transcripts",
    response_model=list[TranscriptResponse],
    summary="Get session transcripts",
    description="Retrieve all transcripts for a session with optional filters",
)
async def get_transcripts(
    session_id: str = Path(..., description="Session identifier"),
    source_type: str | None = Query(
        None, description="Filter by source type (whisper_service, google_meet)"
    ),
    language: str | None = Query(None, description="Filter by language code"),
    speaker_id: str | None = Query(None, description="Filter by speaker ID"),
    start_time: float | None = Query(None, description="Filter by start timestamp (seconds)"),
    end_time: float | None = Query(None, description="Filter by end timestamp (seconds)"),
    limit: int = Query(100, description="Maximum results to return", ge=1, le=1000),
    pipeline=Depends(get_pipeline),
):
    """
    Get transcripts for a session with optional filtering.

    Supports filtering by:
    - Source type (whisper_service, google_meet, manual)
    - Language code
    - Speaker ID
    - Time range

    Returns transcripts in chronological order.
    """
    try:
        # Ensure database is initialized
        if not pipeline.db_manager.db_pool:
            await pipeline.db_manager.initialize()

        # Get transcripts based on filters
        if start_time is not None and end_time is not None:
            transcripts = await pipeline.db_manager.transcript_manager.get_transcript_by_timerange(
                session_id, start_time, end_time
            )
        else:
            transcripts = await pipeline.db_manager.transcript_manager.get_session_transcripts(
                session_id, source_type=source_type
            )

        # Apply additional filters
        filtered = []
        for t in transcripts:
            if language and t.language_code != language:
                continue
            if speaker_id and t.speaker_id != speaker_id:
                continue
            filtered.append(t)

        # Limit results
        filtered = filtered[:limit]

        # Convert to response models
        responses = []
        for t in filtered:
            responses.append(
                TranscriptResponse(
                    transcript_id=t.transcript_id,
                    session_id=t.session_id,
                    source_type=t.source_type,
                    transcript_text=t.transcript_text,
                    language_code=t.language_code,
                    start_timestamp=t.start_timestamp,
                    end_timestamp=t.end_timestamp,
                    duration=t.end_timestamp - t.start_timestamp,
                    speaker_id=t.speaker_id,
                    speaker_name=t.speaker_name,
                    confidence_score=t.confidence_score,
                    segment_index=t.segment_index,
                    audio_file_id=t.audio_file_id,
                    created_at=t.created_at,
                    metadata=t.processing_metadata or {},
                )
            )

        logger.info(f"Retrieved {len(responses)} transcripts for session {session_id}")
        return responses

    except Exception as e:
        logger.error(f"Error getting transcripts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/sessions/{session_id}/translations",
    response_model=list[TranslationResponse],
    summary="Get session translations",
    description="Retrieve all translations for a session with optional filters",
)
async def get_translations(
    session_id: str = Path(..., description="Session identifier"),
    target_language: str | None = Query(None, description="Filter by target language"),
    speaker_id: str | None = Query(None, description="Filter by speaker ID"),
    limit: int = Query(100, description="Maximum results to return", ge=1, le=1000),
    pipeline=Depends(get_pipeline),
):
    """
    Get translations for a session with optional filtering.

    Supports filtering by:
    - Target language
    - Speaker ID

    Returns translations in chronological order.
    """
    try:
        # Ensure database is initialized
        if not pipeline.db_manager.db_pool:
            await pipeline.db_manager.initialize()

        # Get translations
        translations = await pipeline.db_manager.translation_manager.get_session_translations(
            session_id, target_language=target_language
        )

        # Apply speaker filter
        if speaker_id:
            translations = [t for t in translations if t.speaker_id == speaker_id]

        # Limit results
        translations = translations[:limit]

        # Convert to response models
        responses = []
        for t in translations:
            responses.append(
                TranslationResponse(
                    translation_id=t.translation_id,
                    session_id=t.session_id,
                    source_transcript_id=t.source_transcript_id,
                    translated_text=t.translated_text,
                    source_language=t.source_language,
                    target_language=t.target_language,
                    translation_confidence=t.translation_confidence,
                    translation_service=t.translation_service,
                    speaker_id=t.speaker_id,
                    speaker_name=t.speaker_name,
                    start_timestamp=t.start_timestamp,
                    end_timestamp=t.end_timestamp,
                    duration=t.end_timestamp - t.start_timestamp,
                    created_at=t.created_at,
                    metadata=t.processing_metadata or {},
                )
            )

        logger.info(f"Retrieved {len(responses)} translations for session {session_id}")
        return responses

    except Exception as e:
        logger.error(f"Error getting translations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/sessions/{session_id}/timeline",
    response_model=TimelineResponse,
    summary="Get session timeline",
    description="Retrieve complete timeline with transcripts and translations",
)
async def get_timeline(
    session_id: str = Path(..., description="Session identifier"),
    start_time: float | None = Query(None, description="Filter by start timestamp (seconds)"),
    end_time: float | None = Query(None, description="Filter by end timestamp (seconds)"),
    include_translations: bool = Query(True, description="Include translations in timeline"),
    language: str | None = Query(None, description="Filter by language code"),
    speaker_id: str | None = Query(None, description="Filter by speaker ID"),
    limit: int = Query(500, description="Maximum entries to return", ge=1, le=2000),
    pipeline=Depends(get_pipeline),
):
    """
    Get complete session timeline.

    Returns a chronologically ordered list of all transcription and
    translation events with timestamps and speaker information.

    Supports filtering by time range, language, and speaker.
    """
    try:
        # Ensure database is initialized
        if not pipeline.db_manager.db_pool:
            await pipeline.db_manager.initialize()

        # Get timeline
        timeline = await pipeline.get_session_timeline(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            include_translations=include_translations,
            language_filter=language,
            speaker_filter=speaker_id,
        )

        # Limit results
        timeline = timeline[:limit]

        # Convert to response models
        entries = []
        for entry in timeline:
            entries.append(
                TimelineEntryResponse(
                    timestamp=entry.timestamp,
                    duration=entry.duration,
                    entry_type=entry.entry_type,
                    content=entry.content,
                    language=entry.language,
                    speaker_id=entry.speaker_id,
                    speaker_name=entry.speaker_name,
                    confidence=entry.confidence,
                    metadata=entry.metadata or {},
                )
            )

        # Calculate time range
        timeline_start = entries[0].timestamp if entries else None
        timeline_end = entries[-1].timestamp if entries else None

        response = TimelineResponse(
            session_id=session_id,
            total_entries=len(entries),
            start_time=timeline_start,
            end_time=timeline_end,
            entries=entries,
        )

        logger.info(f"Retrieved timeline for session {session_id}: {len(entries)} entries")
        return response

    except Exception as e:
        logger.error(f"Error getting timeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/sessions/{session_id}/speakers",
    response_model=SpeakersResponse,
    summary="Get speaker statistics",
    description="Retrieve statistics for all speakers in a session",
)
async def get_speakers(
    session_id: str = Path(..., description="Session identifier"),
    pipeline=Depends(get_pipeline),
):
    """
    Get statistics for all speakers in a session.

    Returns comprehensive statistics including:
    - Total speaking time
    - Number of segments
    - Average confidence
    - Translation counts
    - Speaker identification information
    """
    try:
        # Ensure database is initialized
        if not pipeline.db_manager.db_pool:
            await pipeline.db_manager.initialize()

        # Get speaker statistics
        stats = await pipeline.get_speaker_statistics(session_id)

        # Convert to response models
        speakers = []
        for stat in stats:
            speakers.append(
                SpeakerStatisticsResponse(
                    session_id=stat.session_id,
                    speaker_id=stat.speaker_id,
                    speaker_name=stat.speaker_name,
                    identification_method=stat.identification_method,
                    identification_confidence=stat.identification_confidence,
                    total_segments=stat.total_segments,
                    total_speaking_time=stat.total_speaking_time,
                    average_confidence=stat.average_confidence,
                    languages_translated_to=stat.languages_translated_to,
                    total_translations=stat.total_translations,
                )
            )

        response = SpeakersResponse(
            session_id=session_id, total_speakers=len(speakers), speakers=speakers
        )

        logger.info(f"Retrieved statistics for {len(speakers)} speakers in session {session_id}")
        return response

    except Exception as e:
        logger.error(f"Error getting speaker statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/sessions/{session_id}/speakers/{speaker_id}",
    response_model=SpeakerDetailResponse,
    summary="Get speaker details",
    description="Retrieve detailed information for a specific speaker",
)
async def get_speaker_detail(
    session_id: str = Path(..., description="Session identifier"),
    speaker_id: str = Path(..., description="Speaker identifier"),
    include_timeline: bool = Query(True, description="Include speaker timeline"),
    limit_transcripts: int = Query(10, description="Number of recent transcripts", ge=0, le=100),
    limit_translations: int = Query(10, description="Number of recent translations", ge=0, le=100),
    pipeline=Depends(get_pipeline),
):
    """
    Get detailed information for a specific speaker.

    Returns:
    - Speaker statistics
    - Timeline of speaker's contributions
    - Recent transcripts
    - Recent translations
    """
    try:
        # Ensure database is initialized
        if not pipeline.db_manager.db_pool:
            await pipeline.db_manager.initialize()

        # Get speaker statistics
        all_stats = await pipeline.get_speaker_statistics(session_id)
        speaker_stats = next((s for s in all_stats if s.speaker_id == speaker_id), None)

        if not speaker_stats:
            raise HTTPException(
                status_code=404,
                detail=f"Speaker {speaker_id} not found in session {session_id}",
            )

        # Get speaker timeline
        timeline_entries = []
        if include_timeline:
            timeline = await pipeline.get_speaker_timeline(
                session_id, speaker_id, include_translations=True
            )
            for entry in timeline:
                timeline_entries.append(
                    TimelineEntryResponse(
                        timestamp=entry.timestamp,
                        duration=entry.duration,
                        entry_type=entry.entry_type,
                        content=entry.content,
                        language=entry.language,
                        speaker_id=entry.speaker_id,
                        speaker_name=entry.speaker_name,
                        confidence=entry.confidence,
                        metadata=entry.metadata or {},
                    )
                )

        # Get recent transcripts
        transcripts = await pipeline.db_manager.transcript_manager.get_session_transcripts(
            session_id
        )
        speaker_transcripts = [t for t in transcripts if t.speaker_id == speaker_id]
        speaker_transcripts = sorted(speaker_transcripts, key=lambda x: x.created_at, reverse=True)[
            :limit_transcripts
        ]

        recent_transcripts = []
        for t in speaker_transcripts:
            recent_transcripts.append(
                TranscriptResponse(
                    transcript_id=t.transcript_id,
                    session_id=t.session_id,
                    source_type=t.source_type,
                    transcript_text=t.transcript_text,
                    language_code=t.language_code,
                    start_timestamp=t.start_timestamp,
                    end_timestamp=t.end_timestamp,
                    duration=t.end_timestamp - t.start_timestamp,
                    speaker_id=t.speaker_id,
                    speaker_name=t.speaker_name,
                    confidence_score=t.confidence_score,
                    segment_index=t.segment_index,
                    audio_file_id=t.audio_file_id,
                    created_at=t.created_at,
                    metadata=t.processing_metadata or {},
                )
            )

        # Get recent translations
        translations = await pipeline.db_manager.translation_manager.get_session_translations(
            session_id
        )
        speaker_translations = [t for t in translations if t.speaker_id == speaker_id]
        speaker_translations = sorted(
            speaker_translations, key=lambda x: x.created_at, reverse=True
        )[:limit_translations]

        recent_translations = []
        for t in speaker_translations:
            recent_translations.append(
                TranslationResponse(
                    translation_id=t.translation_id,
                    session_id=t.session_id,
                    source_transcript_id=t.source_transcript_id,
                    translated_text=t.translated_text,
                    source_language=t.source_language,
                    target_language=t.target_language,
                    translation_confidence=t.translation_confidence,
                    translation_service=t.translation_service,
                    speaker_id=t.speaker_id,
                    speaker_name=t.speaker_name,
                    start_timestamp=t.start_timestamp,
                    end_timestamp=t.end_timestamp,
                    duration=t.end_timestamp - t.start_timestamp,
                    created_at=t.created_at,
                    metadata=t.processing_metadata or {},
                )
            )

        # Build response
        response = SpeakerDetailResponse(
            statistics=SpeakerStatisticsResponse(
                session_id=speaker_stats.session_id,
                speaker_id=speaker_stats.speaker_id,
                speaker_name=speaker_stats.speaker_name,
                identification_method=speaker_stats.identification_method,
                identification_confidence=speaker_stats.identification_confidence,
                total_segments=speaker_stats.total_segments,
                total_speaking_time=speaker_stats.total_speaking_time,
                average_confidence=speaker_stats.average_confidence,
                languages_translated_to=speaker_stats.languages_translated_to,
                total_translations=speaker_stats.total_translations,
            ),
            timeline=timeline_entries,
            recent_transcripts=recent_transcripts,
            recent_translations=recent_translations,
        )

        logger.info(f"Retrieved details for speaker {speaker_id} in session {session_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting speaker detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/sessions/{session_id}/search",
    response_model=SearchResponse,
    summary="Search transcripts",
    description="Full-text search across session transcripts",
)
async def search_transcripts(
    session_id: str = Path(..., description="Session identifier"),
    query: str = Query(..., description="Search query", min_length=1),
    language: str | None = Query(None, description="Filter by language code"),
    use_fuzzy: bool = Query(True, description="Use fuzzy matching (similarity search)"),
    limit: int = Query(50, description="Maximum results to return", ge=1, le=200),
    pipeline=Depends(get_pipeline),
):
    """
    Full-text search across session transcripts.

    Supports:
    - Full-text search using PostgreSQL tsvector
    - Fuzzy matching using trigram similarity
    - Language filtering
    - Relevance ranking

    Returns matching transcripts ordered by relevance.
    """
    try:
        # Ensure database is initialized
        if not pipeline.db_manager.db_pool:
            await pipeline.db_manager.initialize()

        # Search transcripts
        results = await pipeline.search_transcripts(
            session_id=session_id, query=query, language=language, use_fuzzy=use_fuzzy
        )

        # Limit results
        results = results[:limit]

        # Convert to response models
        transcript_responses = []
        for t in results:
            transcript_responses.append(
                TranscriptResponse(
                    transcript_id=t.transcript_id,
                    session_id=t.session_id,
                    source_type=t.source_type,
                    transcript_text=t.transcript_text,
                    language_code=t.language_code,
                    start_timestamp=t.start_timestamp,
                    end_timestamp=t.end_timestamp,
                    duration=t.end_timestamp - t.start_timestamp,
                    speaker_id=t.speaker_id,
                    speaker_name=t.speaker_name,
                    confidence_score=t.confidence_score,
                    segment_index=t.segment_index,
                    audio_file_id=t.audio_file_id,
                    created_at=t.created_at,
                    metadata=t.processing_metadata or {},
                )
            )

        response = SearchResponse(
            session_id=session_id,
            query=query,
            total_results=len(transcript_responses),
            results=transcript_responses,
        )

        logger.info(
            f"Search for '{query}' in session {session_id}: {len(transcript_responses)} results"
        )
        return response

    except Exception as e:
        logger.error(f"Error searching transcripts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# HEALTH CHECK
# ============================================================================


@router.get(
    "/health",
    summary="Health check",
    description="Check data query service health",
)
async def health_check(pipeline=Depends(get_pipeline)):
    """Check service health and database connectivity."""
    try:
        # Check database connection
        if not pipeline.db_manager.db_pool:
            await pipeline.db_manager.initialize()

        # Get database statistics
        stats = await pipeline.db_manager.get_database_statistics()

        return {
            "status": "healthy",
            "database": "connected",
            "statistics": stats,
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
        }
