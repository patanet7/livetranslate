"""
Translation Router

FastAPI router for handling translation requests and proxying them to the translation service.
Provides comprehensive translation endpoints with proper error handling and validation.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import (
    get_translation_service_client,
    rate_limit_api,
    get_current_user,
)
from clients.translation_service_client import (
    TranslationServiceClient,
    TranslationRequest,
    TranslationResponse,
    LanguageDetectionResponse,
)
from database import get_db_session, Translation, BotSession

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
) -> Optional[uuid.UUID]:
    """
    Persist translation to database if session_id is provided.

    Returns the translation_id if successful, None otherwise.
    """
    try:
        # Verify session exists
        session_uuid = uuid.UUID(session_id)
        bot_session = await db.get(BotSession, session_uuid)

        if not bot_session:
            logger.warning(
                f"Session {session_id} not found, skipping database persistence"
            )
            return None

        # Create translation record
        translation = Translation(
            session_id=session_uuid,
            original_text=original_text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            confidence=confidence,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
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
    source_language: Optional[str] = Field(
        None, description="Source language code (auto-detect if None)"
    )
    service: str = Field(
        "ollama", description="Translation service backend (ollama, groq, etc.)"
    )
    quality: str = Field(
        "balanced", description="Translation quality (fast, balanced, quality)"
    )
    prompt_id: Optional[str] = Field(None, description="Custom prompt template ID")
    session_id: Optional[str] = Field(None, description="Session ID for context")

    # Legacy field support - accept 'model' but map it to 'service'
    model: Optional[str] = Field(None, description="[DEPRECATED] Use 'service' instead")

    @property
    def backend_service(self) -> str:
        """Get the backend service to use, with legacy 'model' field support"""
        # If model field is provided and not default, use it (backward compatibility)
        if self.model and self.model != "default":
            return self.model
        # Otherwise use service field
        return self.service

    class Config:
        extra = "ignore"  # Ignore extra fields from frontend


class BatchTranslateRequest(BaseModel):
    """Request model for batch translation"""

    requests: List[TranslateTextRequest] = Field(..., min_items=1, max_items=100)


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
    source_language: Optional[str] = "auto"  # Allow None, default to "auto"
    target_language: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    model_used: str
    backend_used: str
    session_id: Optional[str] = None
    timestamp: str  # Use string instead of datetime for JSON serialization

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# Translation Endpoints
# ============================================================================


@router.post("/", response_model=TranslationApiResponse)
async def translate_text_root(
    request: TranslateTextRequest,
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    db: AsyncSession = Depends(get_db_session),
    _: None = Depends(rate_limit_api),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
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
        logger.info(
            f"Created translation service request: {translation_request.model_dump()}"
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
            backend_used=getattr(
                result, "backend_used", "unknown"
            ),  # Default if not available
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}",
        )


@router.post("/translate", response_model=TranslationApiResponse)
async def translate_text(
    request: TranslateTextRequest,
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    db: AsyncSession = Depends(get_db_session),
    _: None = Depends(rate_limit_api),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
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
            backend_used=getattr(
                result, "backend_used", "unknown"
            ),  # Default if not available
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}",
        )


@router.post("/batch", response_model=List[TranslationApiResponse])
async def translate_batch(
    request: BatchTranslateRequest,
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    _: None = Depends(rate_limit_api),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> List[TranslationApiResponse]:
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
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

        return api_responses

    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch translation failed: {str(e)}",
        )


@router.post("/detect", response_model=LanguageDetectionResponse)
async def detect_language(
    request: LanguageDetectRequest,
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    _: None = Depends(rate_limit_api),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
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
            detail=f"Language detection failed: {str(e)}",
        )


@router.post("/stream")
async def translate_stream(
    request: StreamTranslateRequest,
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    _: None = Depends(rate_limit_api),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
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
            detail=f"Stream translation failed: {str(e)}",
        )


# ============================================================================
# Service Information Endpoints
# ============================================================================


@router.get("/languages")
async def get_supported_languages(
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
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
            detail=f"Failed to get supported languages: {str(e)}",
        )


@router.get("/models")
async def get_available_models(
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
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
            detail=f"Failed to get available models: {str(e)}",
        )


@router.get("/health")
async def translation_service_health(
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
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

        return {
            "status": health_data.get("status", "unknown"),
            "service": "translation",
            "backend": health_data.get("backend", "unknown"),
            "device": device_info.get("device", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
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
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/stats")
async def get_translation_statistics(
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
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
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get translation statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get translation statistics: {str(e)}",
        )


# ============================================================================
# Session Management Endpoints
# ============================================================================


@router.post("/session/start")
async def start_translation_session(
    config: Dict[str, Any],
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    _: None = Depends(rate_limit_api),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
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
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to start translation session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start translation session: {str(e)}",
        )


@router.post("/session/{session_id}/stop")
async def stop_translation_session(
    session_id: str,
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    _: None = Depends(rate_limit_api),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
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
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to stop translation session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop translation session: {str(e)}",
        )


# ============================================================================
# Quality Assessment Endpoints
# ============================================================================


@router.post("/quality")
async def assess_translation_quality(
    original: str,
    translated: str,
    source_language: str,
    target_language: str,
    translation_client: TranslationServiceClient = Depends(
        get_translation_service_client
    ),
    _: None = Depends(rate_limit_api),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
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
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Translation quality assessment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation quality assessment failed: {str(e)}",
        )
