"""
Translation Router

FastAPI router for handling translation requests and proxying them to the translation service.
Provides comprehensive translation endpoints with proper error handling and validation.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from clients.translation_service_client import (
    LanguageDetectionResponse,
    TranslationRequest,
    TranslationResponse,
    TranslationServiceClient,
)
from database import BotSession, SessionEvent, Translation, get_db_session
from dependencies import (
    get_current_user,
    get_translation_service_client,
    rate_limit_api,
)
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# ============================================================================
# Database Persistence Helper
# ============================================================================


async def persist_translation_to_db(
    db: AsyncSession,
    session_id: str,
    original_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
    confidence: float,
    model_used: str,
    backend_used: str,
) -> uuid.UUID | None:
    """
    Persist translation to database if session_id is provided.

    Returns the translation_id if successful, None otherwise.
    """
    try:
        # Verify session exists
        session_uuid = uuid.UUID(session_id)
        bot_session = await db.get(BotSession, session_uuid)

        if not bot_session:
            logger.warning(f"Session {session_id} not found, skipping database persistence")
            return None

        # Create translation record
        translation = Translation(
            session_id=session_uuid,
            original_text=original_text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            confidence=confidence,
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            word_count=len(original_text.split()),
            character_count=len(original_text),
            session_metadata={
                "model_used": model_used,
                "backend_used": backend_used,
            },
        )

        db.add(translation)
        await db.commit()
        await db.refresh(translation)

        logger.info(f"Translation persisted to database: {translation.translation_id}")
        return translation.translation_id

    except ValueError as e:
        logger.error(f"Invalid session_id format: {session_id} - {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to persist translation to database: {e}")
        await db.rollback()
        return None


# ============================================================================
# Request/Response Models
# ============================================================================


class TranslateTextRequest(BaseModel):
    """Request model for text translation"""

    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language code")
    source_language: str | None = Field(
        None, description="Source language code (auto-detect if None)"
    )
    service: str = Field("ollama", description="Translation service backend (ollama, groq, etc.)")
    quality: str = Field("balanced", description="Translation quality (fast, balanced, quality)")
    prompt_id: str | None = Field(None, description="Custom prompt template ID")
    session_id: str | None = Field(None, description="Session ID for context")
    context: str | None = Field(
        None, description="Previous sentences for context-aware translation"
    )

    # Legacy field support - accept 'model' but map it to 'service'
    model: str | None = Field(None, description="[DEPRECATED] Use 'service' instead")

    @property
    def backend_service(self) -> str:
        """Get the backend service to use, with legacy 'model' field support"""
        # If model field is provided and not default, use it (backward compatibility)
        if self.model and self.model != "default":
            return self.model
        # Otherwise use service field
        return self.service

    model_config = ConfigDict(extra="ignore")  # Ignore extra fields from frontend


class BatchTranslateRequest(BaseModel):
    """Request model for batch translation"""

    requests: list[TranslateTextRequest] = Field(..., min_length=1, max_length=100)


class LanguageDetectRequest(BaseModel):
    """Request model for language detection"""

    text: str = Field(..., min_length=1, max_length=10000)


class StreamTranslateRequest(BaseModel):
    """Request model for streaming translation"""

    session_id: str = Field(..., description="Session ID for streaming")
    text: str = Field(..., description="Text chunk to translate")
    target_language: str = Field(..., description="Target language")
    is_final: bool = Field(False, description="Whether this is the final chunk")


class TranslationApiResponse(BaseModel):
    """Standardized translation response"""

    translated_text: str
    source_language: str | None = "auto"  # Allow None, default to "auto"
    target_language: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    model_used: str
    backend_used: str
    session_id: str | None = None
    timestamp: str  # Use string instead of datetime for JSON serialization

    model_config = ConfigDict(ser_json_timedelta="iso8601")


# ============================================================================
# Translation Endpoints
# ============================================================================


@router.post("/", response_model=TranslationApiResponse)
async def translate_text_root(
    request: TranslateTextRequest,
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    db: AsyncSession = Depends(get_db_session),
    _: None = Depends(rate_limit_api),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> TranslationApiResponse:
    """
    Translate text using the translation service (root endpoint)

    **Features:**
    - Auto language detection
    - Multiple translation models
    - Quality settings (fast, balanced, quality)
    - Custom prompt templates
    - Session context support
    - Database persistence (when session_id provided)
    """
    try:
        logger.info(f"ROOT Translation request received: {request.model_dump()}")
        logger.info(
            f"Translation request: {request.source_language or 'auto'} -> {request.target_language}"
        )

        # Create translation request for the service
        translation_request = TranslationRequest(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            model=request.backend_service,  # Use backend_service property for proper mapping
            quality=request.quality,
        )
        logger.info(f"Created translation service request: {translation_request.model_dump()}")

        # Call translation service
        try:
            result = await translation_client.translate(translation_request)
        except Exception as service_error:
            logger.warning(
                f"Translation service unavailable, creating mock result: {service_error}"
            )
            # Create a mock response when translation service is unavailable
            result = TranslationResponse(
                translated_text=f"[Translation service unavailable] Original text: {request.text}",
                source_language=request.source_language or "auto",
                target_language=request.target_language,
                confidence=0.0,
                processing_time=0.0,
                model_used=request.backend_service,
                backend_used="fallback",
            )

        # Persist to database if session_id is provided
        if request.session_id:
            await persist_translation_to_db(
                db=db,
                session_id=request.session_id,
                original_text=request.text,
                translated_text=result.translated_text,
                source_language=result.source_language or "auto",
                target_language=result.target_language,
                confidence=result.confidence,
                model_used=result.model_used,
                backend_used=getattr(result, "backend_used", "unknown"),
            )

        # Convert to API response format
        return TranslationApiResponse(
            translated_text=result.translated_text,
            source_language=result.source_language,
            target_language=result.target_language,
            confidence=result.confidence,
            processing_time=result.processing_time,
            model_used=result.model_used,
            backend_used=getattr(result, "backend_used", "unknown"),  # Default if not available
            session_id=request.session_id,
            timestamp=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {e!s}",
        ) from e


@router.post("/translate", response_model=TranslationApiResponse)
async def translate_text(
    request: TranslateTextRequest,
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    db: AsyncSession = Depends(get_db_session),
    _: None = Depends(rate_limit_api),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> TranslationApiResponse:
    """
    Translate text using the translation service

    **Features:**
    - Auto language detection
    - Multiple translation models
    - Quality settings (fast, balanced, quality)
    - Custom prompt templates
    - Session context support
    - Database persistence (when session_id provided)
    """
    try:
        logger.info(
            f"Translation request: {request.source_language or 'auto'} -> {request.target_language}"
        )

        # Create translation request for the service
        translation_request = TranslationRequest(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            model=request.backend_service,  # Use backend_service property for proper mapping
            quality=request.quality,
            context=request.context,  # Pass context for better translation quality
        )

        # Call translation service
        try:
            result = await translation_client.translate(translation_request)
        except Exception as service_error:
            logger.warning(
                f"Translation service unavailable, creating mock result: {service_error}"
            )
            # Create a mock response when translation service is unavailable
            result = TranslationResponse(
                translated_text=f"[Translation service unavailable] Original text: {request.text}",
                source_language=request.source_language or "auto",
                target_language=request.target_language,
                confidence=0.0,
                processing_time=0.0,
                model_used=request.backend_service,
                backend_used="fallback",
            )

        # Persist to database if session_id is provided
        if request.session_id:
            await persist_translation_to_db(
                db=db,
                session_id=request.session_id,
                original_text=request.text,
                translated_text=result.translated_text,
                source_language=result.source_language or "auto",
                target_language=result.target_language,
                confidence=result.confidence,
                model_used=result.model_used,
                backend_used=getattr(result, "backend_used", "unknown"),
            )

        # Convert to API response format
        return TranslationApiResponse(
            translated_text=result.translated_text,
            source_language=result.source_language,
            target_language=result.target_language,
            confidence=result.confidence,
            processing_time=result.processing_time,
            model_used=result.model_used,
            backend_used=getattr(result, "backend_used", "unknown"),  # Default if not available
            session_id=request.session_id,
            timestamp=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {e!s}",
        ) from e


@router.post("/batch", response_model=list[TranslationApiResponse])
async def translate_batch(
    request: BatchTranslateRequest,
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    _: None = Depends(rate_limit_api),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> list[TranslationApiResponse]:
    """
    Batch translate multiple texts

    **Features:**
    - Process up to 100 texts in one request
    - Parallel processing for better performance
    - Individual error handling per text
    """
    try:
        logger.info(f"Batch translation request: {len(request.requests)} texts")

        # Convert to service request format
        service_requests = [
            TranslationRequest(
                text=req.text,
                source_language=req.source_language,
                target_language=req.target_language,
                model=req.model,
                quality=req.quality,
            )
            for req in request.requests
        ]

        # Call batch translation service
        results = await translation_client.translate_batch(service_requests)

        # Convert to API response format
        api_responses = []
        for i, result in enumerate(results):
            api_responses.append(
                TranslationApiResponse(
                    translated_text=result.translated_text,
                    source_language=result.source_language,
                    target_language=result.target_language,
                    confidence=result.confidence,
                    processing_time=result.processing_time,
                    model_used=result.model_used,
                    backend_used=result.backend_used,
                    session_id=request.requests[i].session_id,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )

        return api_responses

    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch translation failed: {e!s}",
        ) from e


# ============================================================================
# Job Tracking Endpoints
# ============================================================================


class CreateJobRequest(BaseModel):
    """Request to create a translation job"""

    job_type: str = Field(..., description="Type of job: translation, transcription")
    source_id: str = Field(..., description="Source identifier (transcript_id, file_id, etc.)")
    target_language: str | None = None
    total_items: int = Field(..., ge=1, description="Total items to process")
    metadata: dict[str, Any] | None = None


class UpdateJobRequest(BaseModel):
    """Request to update job progress"""

    completed_items: int = Field(..., ge=0)
    status: str = Field(
        "in_progress", description="Job status: pending, in_progress, completed, failed"
    )
    error_message: str | None = None
    partial_results: list[dict[str, Any]] | None = None


class JobResponse(BaseModel):
    """Job status response"""

    job_id: str
    job_type: str
    source_id: str
    status: str
    completed_items: int
    total_items: int
    progress_percent: float
    created_at: str
    updated_at: str
    metadata: dict[str, Any] | None = None


@router.post("/jobs", response_model=JobResponse)
async def create_job(
    request: CreateJobRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a new translation/transcription job for tracking.

    Jobs are tracked using SessionEvent table for flexibility.
    """
    try:
        job_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        # Store job as a SessionEvent
        event = SessionEvent(
            event_id=uuid.UUID(job_id),
            session_id=None,  # Jobs don't require a session
            event_type="job",
            event_name=f"{request.job_type}_job_created",
            event_data={
                "job_type": request.job_type,
                "source_id": request.source_id,
                "target_language": request.target_language,
                "total_items": request.total_items,
                "completed_items": 0,
                "status": "pending",
                "metadata": request.metadata or {},
            },
            severity="info",
            source="job_tracker",
            timestamp=now,
        )
        db.add(event)
        await db.commit()

        logger.info(f"Created job {job_id} for {request.job_type}")

        return JobResponse(
            job_id=job_id,
            job_type=request.job_type,
            source_id=request.source_id,
            status="pending",
            completed_items=0,
            total_items=request.total_items,
            progress_percent=0.0,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            metadata=request.metadata,
        )

    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {e!s}",
        ) from e


@router.patch("/jobs/{job_id}", response_model=JobResponse)
async def update_job(
    job_id: str,
    request: UpdateJobRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Update job progress."""
    try:
        from sqlalchemy import select

        # Find the job event
        result = await db.execute(
            select(SessionEvent).where(SessionEvent.event_id == uuid.UUID(job_id))
        )
        event = result.scalar_one_or_none()

        if not event:
            raise HTTPException(status_code=404, detail="Job not found")

        # Update the event data
        now = datetime.now(UTC)
        event_data = event.event_data or {}
        event_data["completed_items"] = request.completed_items
        event_data["status"] = request.status
        if request.error_message:
            event_data["error_message"] = request.error_message
        if request.partial_results:
            event_data["partial_results"] = request.partial_results

        event.event_data = event_data
        event.timestamp = now
        event.event_name = f"{event_data.get('job_type', 'unknown')}_job_{request.status}"

        await db.commit()

        total = event_data.get("total_items", 1)
        completed = request.completed_items
        progress = (completed / total * 100) if total > 0 else 0

        return JobResponse(
            job_id=job_id,
            job_type=event_data.get("job_type", "unknown"),
            source_id=event_data.get("source_id", ""),
            status=request.status,
            completed_items=completed,
            total_items=total,
            progress_percent=round(progress, 1),
            created_at=event_data.get("created_at", now.isoformat()),
            updated_at=now.isoformat(),
            metadata=event_data.get("metadata"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update job: {e!s}",
        ) from e


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get job status."""
    try:
        from sqlalchemy import select

        result = await db.execute(
            select(SessionEvent).where(SessionEvent.event_id == uuid.UUID(job_id))
        )
        event = result.scalar_one_or_none()

        if not event:
            raise HTTPException(status_code=404, detail="Job not found")

        event_data = event.event_data or {}
        total = event_data.get("total_items", 1)
        completed = event_data.get("completed_items", 0)
        progress = (completed / total * 100) if total > 0 else 0

        return JobResponse(
            job_id=job_id,
            job_type=event_data.get("job_type", "unknown"),
            source_id=event_data.get("source_id", ""),
            status=event_data.get("status", "unknown"),
            completed_items=completed,
            total_items=total,
            progress_percent=round(progress, 1),
            created_at=event_data.get("created_at", event.timestamp.isoformat()),
            updated_at=event.timestamp.isoformat(),
            metadata=event_data.get("metadata"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job: {e!s}",
        ) from e


@router.get("/jobs")
async def list_jobs(
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db_session),
):
    """List recent jobs."""
    try:
        from sqlalchemy import desc, select

        query = (
            select(SessionEvent)
            .where(SessionEvent.event_type == "job")
            .order_by(desc(SessionEvent.timestamp))
            .limit(limit)
        )

        result = await db.execute(query)
        events = result.scalars().all()

        jobs = []
        for event in events:
            event_data = event.event_data or {}

            # Filter by status/type if specified
            if status and event_data.get("status") != status:
                continue
            if job_type and event_data.get("job_type") != job_type:
                continue

            total = event_data.get("total_items", 1)
            completed = event_data.get("completed_items", 0)
            progress = (completed / total * 100) if total > 0 else 0

            jobs.append(
                {
                    "job_id": str(event.event_id),
                    "job_type": event_data.get("job_type", "unknown"),
                    "source_id": event_data.get("source_id", ""),
                    "status": event_data.get("status", "unknown"),
                    "completed_items": completed,
                    "total_items": total,
                    "progress_percent": round(progress, 1),
                    "created_at": event_data.get("created_at", event.timestamp.isoformat()),
                    "updated_at": event.timestamp.isoformat(),
                }
            )

        return {"jobs": jobs, "count": len(jobs)}

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {e!s}",
        ) from e


@router.post("/detect", response_model=LanguageDetectionResponse)
async def detect_language(
    request: LanguageDetectRequest,
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    _: None = Depends(rate_limit_api),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> LanguageDetectionResponse:
    """
    Detect the language of input text

    **Features:**
    - High accuracy language detection
    - Confidence scoring
    - Alternative language suggestions
    """
    try:
        logger.info(f"Language detection request for text: {request.text[:50]}...")

        # Call language detection service
        result = await translation_client.detect_language(request.text)

        return result

    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection failed: {e!s}",
        ) from e


@router.post("/stream")
async def translate_stream(
    request: StreamTranslateRequest,
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    _: None = Depends(rate_limit_api),
    current_user: dict[str, Any] | None = Depends(get_current_user),
):
    """
    Stream translation for real-time processing

    **Features:**
    - Real-time translation streaming
    - Session-based context management
    - Progressive translation updates
    """
    try:
        logger.info(f"Stream translation request for session: {request.session_id}")

        # Call streaming translation service
        result = await translation_client.translate_realtime(
            session_id=request.session_id,
            text=request.text,
            target_language=request.target_language,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Streaming translation failed",
            )

        return result

    except Exception as e:
        logger.error(f"Stream translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stream translation failed: {e!s}",
        ) from e


# ============================================================================
# Service Information Endpoints
# ============================================================================


@router.get("/languages")
async def get_supported_languages(
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get list of supported languages

    **Returns:**
    - Complete list of supported language codes and names
    - Language capabilities and features
    """
    try:
        languages = await translation_client.get_supported_languages()

        return {
            "languages": languages,
            "total_count": len(languages),
            "auto_detect": True,
            "bidirectional": True,
        }

    except Exception as e:
        logger.error(f"Failed to get supported languages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get supported languages: {e!s}",
        ) from e


@router.get("/models")
async def get_available_models(
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get list of available translation models

    **Returns:**
    - Available translation models
    - Model capabilities and performance characteristics
    """
    try:
        models = await translation_client.get_models()

        return {
            "models": models,
            "total_count": len(models),
            "default_model": "default",
        }

    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available models: {e!s}",
        ) from e


@router.get("/health")
async def translation_service_health(
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Check translation service health

    **Returns:**
    - Service health status
    - Performance metrics
    - Backend information
    """
    try:
        health_data = await translation_client.health_check()
        device_info = await translation_client.get_device_info()

        # Extract backend from nested structure (translation service returns details.backend)
        backend = (
            health_data.get("backend") or health_data.get("details", {}).get("backend") or "unknown"
        )

        # Extract available backends for more info
        available_backends = health_data.get("details", {}).get("available_backends", [])
        if available_backends and backend == "unknown":
            backend = available_backends[0] if available_backends else "unknown"

        return {
            "status": health_data.get("status", "unknown"),
            "service": "translation",
            "backend": backend,
            "available_backends": available_backends,
            "device": device_info.get("device", "unknown"),
            "timestamp": datetime.now(UTC).isoformat(),
            "details": {
                "health": health_data,
                "device": device_info,
            },
        }

    except Exception as e:
        logger.error(f"Translation service health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "translation",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@router.get("/stats")
async def get_translation_statistics(
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get translation service statistics

    **Returns:**
    - Usage statistics
    - Performance metrics
    - Service analytics
    """
    try:
        stats = await translation_client.get_statistics()

        return {
            "statistics": stats,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get translation statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get translation statistics: {e!s}",
        ) from e


# ============================================================================
# Session Management Endpoints
# ============================================================================


@router.post("/session/start")
async def start_translation_session(
    config: dict[str, Any],
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    _: None = Depends(rate_limit_api),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Start a real-time translation session

    **Features:**
    - Session-based translation with context
    - Configuration for streaming parameters
    - Real-time session management
    """
    try:
        session_id = await translation_client.start_realtime_session(config)

        return {
            "session_id": session_id,
            "status": "started",
            "config": config,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to start translation session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start translation session: {e!s}",
        ) from e


@router.post("/session/{session_id}/stop")
async def stop_translation_session(
    session_id: str,
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    _: None = Depends(rate_limit_api),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Stop a real-time translation session

    **Features:**
    - Clean session termination
    - Session statistics and summary
    - Resource cleanup
    """
    try:
        result = await translation_client.stop_realtime_session(session_id)

        return {
            "session_id": session_id,
            "status": "stopped",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to stop translation session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop translation session: {e!s}",
        ) from e


# ============================================================================
# Quality Assessment Endpoints
# ============================================================================


@router.post("/quality")
async def assess_translation_quality(
    original: str,
    translated: str,
    source_language: str,
    target_language: str,
    translation_client: TranslationServiceClient = Depends(get_translation_service_client),
    _: None = Depends(rate_limit_api),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Assess translation quality metrics

    **Features:**
    - Quality scoring algorithms
    - Detailed quality metrics
    - Translation accuracy assessment
    """
    try:
        quality_data = await translation_client.get_translation_quality(
            original=original,
            translated=translated,
            source_lang=source_language,
            target_lang=target_language,
        )

        return {
            "quality_assessment": quality_data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Translation quality assessment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation quality assessment failed: {e!s}",
        ) from e


# ============================================================================
# Dashboard Settings Endpoints (Database-backed)
# ============================================================================

# Constants for dashboard settings
DASHBOARD_SETTINGS_KEY = "fireflies-dashboard-settings"


class ModelSettingRequest(BaseModel):
    """Request model for setting translation model"""

    model: str = Field(..., description="Translation model to use")


class PromptSettingRequest(BaseModel):
    """Request model for setting prompt template"""

    template: str = Field(..., description="Prompt template text")


async def get_dashboard_setting(db: AsyncSession, setting_name: str) -> str | None:
    """Get a dashboard setting from the database using session_events table."""
    from sqlalchemy import desc, select

    try:
        # Query for the most recent setting event
        stmt = (
            select(SessionEvent)
            .where(SessionEvent.event_type == "dashboard_setting")
            .where(SessionEvent.event_name == setting_name)
            .order_by(desc(SessionEvent.timestamp))
            .limit(1)
        )
        result = await db.execute(stmt)
        event = result.scalar_one_or_none()

        if event and event.event_data:
            return event.event_data.get("value")
        return None
    except Exception as e:
        logger.error(f"Failed to get dashboard setting '{setting_name}': {e}")
        return None


async def save_dashboard_setting(db: AsyncSession, setting_name: str, value: str) -> bool:
    """Save a dashboard setting to the database using session_events table."""
    try:
        # Create a new session event for the setting
        event = SessionEvent(
            event_id=uuid.uuid4(),
            session_id=None,  # Dashboard settings are not session-specific
            event_type="dashboard_setting",
            event_name=setting_name,
            event_data={"value": value, "saved_at": datetime.now(UTC).isoformat()},
            timestamp=datetime.now(UTC),
            severity="info",
            source="fireflies-dashboard",
        )
        db.add(event)
        await db.commit()
        logger.info(f"Saved dashboard setting '{setting_name}' to database")
        return True
    except Exception as e:
        logger.error(f"Failed to save dashboard setting '{setting_name}': {e}")
        await db.rollback()
        return False


@router.post("/model")
async def set_translation_model(
    request: ModelSettingRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Set the default translation model for the dashboard.

    **Features:**
    - Persists model selection to database
    - Used by Fireflies dashboard for default model setting
    """
    try:
        logger.info(f"Setting translation model to: {request.model}")

        success = await save_dashboard_setting(db, "translation_model", request.model)

        if success:
            return {
                "success": True,
                "model": request.model,
                "message": "Translation model saved to database",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save model setting to database",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set translation model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set translation model: {e!s}",
        ) from e


@router.get("/model")
async def get_translation_model(
    db: AsyncSession = Depends(get_db_session),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get the default translation model from the database.

    **Features:**
    - Retrieves persisted model selection
    - Returns default if not set
    """
    try:
        model = await get_dashboard_setting(db, "translation_model")

        return {
            "model": model or "default",
            "source": "database" if model else "default",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get translation model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get translation model: {e!s}",
        ) from e


@router.post("/prompt")
async def set_translation_prompt(
    request: PromptSettingRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Set the translation prompt template for the dashboard.

    **Features:**
    - Persists prompt template to database
    - Used by Fireflies dashboard for custom prompts
    """
    try:
        logger.info(f"Saving translation prompt template ({len(request.template)} chars)")

        success = await save_dashboard_setting(db, "translation_prompt", request.template)

        if success:
            return {
                "success": True,
                "template_length": len(request.template),
                "message": "Prompt template saved to database",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save prompt template to database",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set translation prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set translation prompt: {e!s}",
        ) from e


@router.get("/prompt")
async def get_translation_prompt(
    db: AsyncSession = Depends(get_db_session),
    current_user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get the translation prompt template from the database.

    **Features:**
    - Retrieves persisted prompt template
    - Returns default template if not set
    """
    try:
        template = await get_dashboard_setting(db, "translation_prompt")

        default_template = """Translate to {target_language}:
{current_sentence}

Translation:"""

        return {
            "template": template or default_template,
            "source": "database" if template else "default",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get translation prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get translation prompt: {e!s}",
        ) from e
