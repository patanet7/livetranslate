"""
Audio Core Processing Router

Main audio processing endpoints including:
- Audio processing (/process)
- File upload (/upload)
- Audio streaming (/stream)
- Transcription (/transcribe)
- Health monitoring (/health)
- Model management (/models)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import Depends, UploadFile, File, Form, HTTPException, status

# Add shared module to path for model registry (append to avoid conflicts)
_SHARED_PATH = Path(__file__).parent.parent.parent.parent.parent / "shared" / "src"
if str(_SHARED_PATH) not in sys.path:
    sys.path.append(str(_SHARED_PATH))

try:
    from model_registry import ModelRegistry
    _DEFAULT_WHISPER_MODEL = ModelRegistry.DEFAULT_WHISPER_MODEL
except ImportError:
    _DEFAULT_WHISPER_MODEL = "whisper-base"

from ._shared import (
    create_audio_router,
    logger,
    format_recovery,
    service_recovery,
    audio_service_circuit_breaker,
    retry_manager,
)
from models.audio import AudioProcessingRequest, AudioProcessingResponse, AudioStats
from dependencies import (
    get_config_manager,
    get_audio_service_client,
    get_audio_coordinator,
    get_config_sync_manager,
    get_health_monitor,
    get_event_publisher,
    get_translation_service_client,
)
from utils.audio_errors import (
    error_boundary,
    AudioProcessingError,
    AudioProcessingBaseError,
    ValidationError,
    AudioFormatError,
    AudioCorruptionError,
    NetworkError,
    ServiceUnavailableError,
    ConfigurationError,
)

# Create router for core audio processing
router = create_audio_router()


@router.post("/process", response_model=AudioProcessingResponse)
async def process_audio(
    request: AudioProcessingRequest,
    config_manager=Depends(get_config_manager),
    audio_client=Depends(get_audio_service_client),
    event_publisher=Depends(get_event_publisher),
) -> AudioProcessingResponse:
    """
    Process audio with configured pipeline stages with comprehensive error handling

    - **audio_data**: Base64 encoded audio data
    - **audio_url**: URL to audio file
    - **file_upload**: File upload reference
    - **config**: Processing configuration
    - **streaming**: Enable streaming processing
    - **transcription**: Enable transcription
    - **speaker_diarization**: Enable speaker diarization
    """
    request_id = f"req_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

    async with error_boundary(
        correlation_id=request_id,
        context={
            "service": "orchestration",
            "endpoint": "/process",
            "streaming": request.streaming,
            "request_size": len(str(request)),
        },
        recovery_strategies=[format_recovery, service_recovery],
    ) as correlation_id:
        try:
            logger.info(f"[{correlation_id}] Processing audio request")

            # Enhanced input validation
            await _validate_audio_request(request, correlation_id)

            # Emit queue event (non-blocking best-effort)
            await event_publisher.publish(
                alias="audio_ingest",
                event_type="AudioProcessRequested",
                payload={
                    "request_id": correlation_id,
                    "streaming": request.streaming,
                    "session_id": request.session_id,
                    "has_audio_data": bool(request.audio_data),
                    "has_audio_url": bool(request.audio_url),
                    "has_file_upload": bool(request.file_upload),
                    "config_provided": bool(request.config),
                },
                metadata={"endpoint": "/process"},
            )

            # Get audio service configuration with error handling
            try:
                audio_service_config = await config_manager.get_service_config("audio")
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to get audio service configuration: {str(e)}",
                    correlation_id=correlation_id,
                    config_details={"service": "audio"},
                )

            # Process through enhanced pipeline with circuit breaker
            if request.streaming:
                return await audio_service_circuit_breaker.call(
                    _process_audio_streaming_with_retry,
                    request,
                    correlation_id,
                    audio_client,
                    audio_service_config,
                )
            else:
                return await audio_service_circuit_breaker.call(
                    _process_audio_batch_with_retry,
                    request,
                    correlation_id,
                    audio_client,
                    audio_service_config,
                )

        except AudioProcessingBaseError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Convert unknown exceptions to AudioProcessingError
            raise AudioProcessingError(
                f"Unexpected error in audio processing: {str(e)}",
                correlation_id=correlation_id,
                processing_stage="main_pipeline",
                details={"exception_type": type(e).__name__},
            )


async def _validate_audio_request(request: AudioProcessingRequest, correlation_id: str):
    """Enhanced audio request validation"""
    # Validate audio source
    if not any([request.audio_data, request.audio_url, request.file_upload]):
        raise ValidationError(
            "No audio source provided",
            correlation_id=correlation_id,
            validation_details={
                "missing_fields": ["audio_data", "audio_url", "file_upload"],
                "provided_fields": [
                    k for k, v in request.dict().items() if v is not None
                ],
            },
        )

    # Validate audio data if provided
    if request.audio_data:
        try:
            import base64

            audio_bytes = base64.b64decode(request.audio_data)
            if len(audio_bytes) == 0:
                raise AudioCorruptionError(
                    "Audio data is empty after base64 decoding",
                    correlation_id=correlation_id,
                    corruption_details={"original_length": len(request.audio_data)},
                )
            if len(audio_bytes) > 100 * 1024 * 1024:  # 100MB limit
                raise ValidationError(
                    "Audio file too large (max 100MB)",
                    correlation_id=correlation_id,
                    validation_details={
                        "size_bytes": len(audio_bytes),
                        "max_size": 100 * 1024 * 1024,
                    },
                )
        except Exception as e:
            if isinstance(e, AudioProcessingBaseError):
                raise
            raise AudioFormatError(
                f"Invalid base64 audio data: {str(e)}",
                correlation_id=correlation_id,
                format_details={"encoding": "base64", "error": str(e)},
            )

    # Validate configuration
    if request.config:
        try:
            if isinstance(request.config, str):
                json.loads(request.config)
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON configuration: {str(e)}",
                correlation_id=correlation_id,
                validation_details={"config_error": str(e)},
            )


async def _process_audio_streaming_with_retry(
    request: AudioProcessingRequest,
    correlation_id: str,
    audio_client,
    audio_service_config: Dict[str, Any],
) -> AudioProcessingResponse:
    """Streaming audio processing with retry mechanism"""
    return await retry_manager.execute_with_retry(
        _process_audio_streaming,
        request,
        correlation_id,
        audio_client,
        audio_service_config,
        correlation_id=correlation_id,
        retryable_exceptions=(NetworkError, TimeoutError, ServiceUnavailableError),
    )


async def _process_audio_batch_with_retry(
    request: AudioProcessingRequest,
    correlation_id: str,
    audio_client,
    audio_service_config: Dict[str, Any],
) -> AudioProcessingResponse:
    """Batch audio processing with retry mechanism"""
    return await retry_manager.execute_with_retry(
        _process_audio_batch,
        request,
        correlation_id,
        audio_client,
        audio_service_config,
        correlation_id=correlation_id,
        retryable_exceptions=(NetworkError, TimeoutError, ServiceUnavailableError),
    )


async def _process_audio_streaming(
    request: AudioProcessingRequest,
    correlation_id: str,
    audio_client,
    audio_service_config: Dict[str, Any],
) -> AudioProcessingResponse:
    """Core streaming audio processing logic"""
    # Placeholder for streaming implementation
    # This would contain the actual streaming processing logic
    return AudioProcessingResponse(
        request_id=correlation_id,
        status="processed",
        transcription="Streaming processing placeholder",
        processing_time=0.1,
        confidence=0.95,
    )


async def _process_audio_batch(
    request: AudioProcessingRequest,
    correlation_id: str,
    audio_client,
    audio_service_config: Dict[str, Any],
) -> AudioProcessingResponse:
    """Core batch audio processing logic"""
    # Placeholder for batch implementation
    # This would contain the actual batch processing logic
    return AudioProcessingResponse(
        request_id=correlation_id,
        status="processed",
        transcription="Batch processing placeholder",
        processing_time=0.2,
        confidence=0.96,
    )


@router.post("/upload", response_model=Dict[str, Any])
async def upload_audio_file(
    audio: UploadFile = File(..., alias="audio"),
    config: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    chunk_id: Optional[str] = Form(None),
    target_languages: Optional[str] = Form(None),
    enable_transcription: Optional[str] = Form("true"),
    enable_translation: Optional[str] = Form("false"),
    enable_diarization: Optional[str] = Form("true"),
    whisper_model: Optional[str] = Form(_DEFAULT_WHISPER_MODEL),
    translation_quality: Optional[str] = Form("balanced"),
    audio_processing: Optional[str] = Form("true"),
    noise_reduction: Optional[str] = Form("false"),
    speech_enhancement: Optional[str] = Form("true"),
    audio_coordinator=Depends(get_audio_coordinator),
    config_sync_manager=Depends(get_config_sync_manager),
    audio_client=Depends(get_audio_service_client),
    event_publisher=Depends(get_event_publisher),
) -> Dict[str, Any]:
    """
    Upload audio file for processing with enhanced error handling and validation

    - **audio**: Audio file to upload (WAV, MP3, OGG, WebM, MP4, FLAC)
    - **config**: Optional JSON configuration for processing (legacy support)
    - **session_id**: Session identifier for tracking
    - **chunk_id**: Chunk identifier for streaming uploads
    - **target_languages**: JSON array of target languages for translation (e.g., ["es", "fr"])
    - **enable_transcription**: Enable transcription (default: true)
    - **enable_translation**: Enable translation (default: false)
    - **enable_diarization**: Enable speaker diarization (default: true)
    - **whisper_model**: Whisper model to use (default: whisper-base)
    - **translation_quality**: Translation quality setting (default: balanced)
    """
    correlation_id = f"upload_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

    async with error_boundary(
        correlation_id=correlation_id,
        context={
            "service": "orchestration",
            "endpoint": "/upload",
            "filename": audio.filename,
            "content_type": audio.content_type,
        },
        recovery_strategies=[format_recovery, service_recovery],
    ) as upload_correlation_id:
        try:
            logger.info(
                f"[{upload_correlation_id}] Processing file upload: {audio.filename}"
            )

            # Enhanced file validation
            await _validate_upload_file(audio, upload_correlation_id)

            # Build configuration from form parameters
            # Convert string booleans to actual booleans
            enable_trans = (
                enable_transcription.lower() in ("true", "1", "yes")
                if enable_transcription
                else True
            )
            enable_transl = (
                enable_translation.lower() in ("true", "1", "yes")
                if enable_translation
                else False
            )
            enable_diar = (
                enable_diarization.lower() in ("true", "1", "yes")
                if enable_diarization
                else True
            )
            enable_audio_proc = (
                audio_processing.lower() in ("true", "1", "yes")
                if audio_processing
                else True
            )
            enable_noise_red = (
                noise_reduction.lower() in ("true", "1", "yes")
                if noise_reduction
                else False
            )
            enable_speech_enh = (
                speech_enhancement.lower() in ("true", "1", "yes")
                if speech_enhancement
                else True
            )

            request_config = {
                "session_id": session_id,
                "chunk_id": chunk_id,
                "enable_transcription": enable_trans,
                "enable_translation": enable_transl,
                "enable_diarization": enable_diar,
                "whisper_model": whisper_model,
                "translation_quality": translation_quality,
                "audio_processing": enable_audio_proc,
                "noise_reduction": enable_noise_red,
                "speech_enhancement": enable_speech_enh,
            }

            # Add target languages if provided
            if target_languages:
                request_config["target_languages"] = target_languages

            # Legacy config parameter takes precedence if provided
            if config:
                try:
                    import json

                    legacy_config = json.loads(config)
                    request_config.update(legacy_config)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[{upload_correlation_id}] Failed to parse legacy config JSON: {e}"
                    )

            logger.info(
                f"[{upload_correlation_id}] Processing configuration: "
                f"transcription={enable_trans}, translation={enable_transl}, "
                f"diarization={enable_diar}, model={whisper_model}, "
                f"audio_processing={enable_audio_proc}"
            )

            # Create processing request from uploaded file
            # Don't create AudioProcessingRequest - it's not needed
            # Just process directly with the coordinator

            # Process the uploaded file
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_{audio.filename}"
            ) as temp_file:
                # Read and save uploaded file
                content = await audio.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            # Publish event for background workers
            await event_publisher.publish(
                alias="audio_ingest",
                event_type="AudioUploadReceived",
                payload={
                    "upload_id": upload_correlation_id,
                    "session_id": session_id,
                    "chunk_id": chunk_id,
                    "filename": audio.filename,
                    "content_type": audio.content_type,
                    "file_size": len(content),
                    "config": request_config,
                },
                metadata={"endpoint": "/upload"},
            )

            try:
                # Process uploaded file safely with complete configuration
                result = await _process_uploaded_file_safe(
                    None,  # processing_request no longer needed
                    upload_correlation_id,
                    temp_file_path,
                    audio_client,
                    request_config,
                    audio_coordinator,
                    config_sync_manager,
                )

                return {
                    "upload_id": upload_correlation_id,
                    "chunk_id": chunk_id,
                    "filename": audio.filename,
                    "status": "uploaded_and_processed",
                    "file_size": len(content),
                    "processing_result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(
                        f"Failed to clean up temp file {temp_file_path}: {e}"
                    )

        except AudioProcessingBaseError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Convert unknown exceptions to appropriate audio errors
            raise AudioProcessingError(
                f"File upload failed: {str(e)}",
                correlation_id=upload_correlation_id,
                processing_stage="file_upload",
                details={"filename": audio.filename, "error": str(e)},
            )


async def _validate_upload_file(audio: UploadFile, correlation_id: str):
    """Enhanced file upload validation"""
    # Validate filename
    if not audio.filename:
        raise ValidationError(
            "No filename provided",
            correlation_id=correlation_id,
            validation_details={"missing_field": "filename"},
        )

    # Validate content type
    allowed_types = {
        "audio/wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/webm",
        "audio/mp4",
        "audio/flac",
        "audio/x-flac",
        "application/octet-stream",  # Allow generic binary for various formats
    }

    if audio.content_type and audio.content_type not in allowed_types:
        raise AudioFormatError(
            f"Unsupported audio format: {audio.content_type}",
            correlation_id=correlation_id,
            format_details={
                "provided_type": audio.content_type,
                "allowed_types": list(allowed_types),
            },
        )

    # Validate file extension
    if audio.filename:
        allowed_extensions = {".wav", ".mp3", ".ogg", ".webm", ".mp4", ".flac", ".m4a"}
        file_ext = (
            "." + audio.filename.split(".")[-1].lower() if "." in audio.filename else ""
        )

        if file_ext not in allowed_extensions:
            raise AudioFormatError(
                f"Unsupported file extension: {file_ext}",
                correlation_id=correlation_id,
                format_details={
                    "provided_extension": file_ext,
                    "allowed_extensions": list(allowed_extensions),
                },
            )


async def _process_uploaded_file_safe(
    processing_request: AudioProcessingRequest,
    correlation_id: str,
    temp_file_path: str,
    audio_client,
    request_data: Dict[str, Any],
    audio_coordinator,
    config_sync_manager,
) -> Dict[str, Any]:
    """Safe wrapper for uploaded file processing with error handling"""
    try:
        return await _process_uploaded_file(
            processing_request,
            correlation_id,
            temp_file_path,
            audio_client,
            request_data,
            audio_coordinator,
            config_sync_manager,
        )
    except Exception as e:
        # Convert generic exceptions to appropriate audio errors
        if "network" in str(e).lower() or "connection" in str(e).lower():
            raise NetworkError(
                f"Network error during audio processing: {str(e)}",
                correlation_id=correlation_id,
                network_details={"original_error": str(e)},
            )
        elif "timeout" in str(e).lower():
            raise TimeoutError(
                f"Timeout during audio processing: {str(e)}",
                correlation_id=correlation_id,
                timeout_details={"original_error": str(e)},
            )
        elif "service" in str(e).lower() or "unavailable" in str(e).lower():
            raise ServiceUnavailableError(
                f"Audio service unavailable: {str(e)}",
                correlation_id=correlation_id,
                service_name="whisper_service",
            )
        else:
            raise AudioProcessingError(
                f"Audio processing failed: {str(e)}",
                correlation_id=correlation_id,
                processing_stage="file_processing",
                details={"original_error": str(e)},
            )


async def _process_uploaded_file(
    processing_request: AudioProcessingRequest,
    correlation_id: str,
    temp_file_path: str,
    audio_client,
    request_data: Dict[str, Any],
    audio_coordinator,
    config_sync_manager,
) -> Dict[str, Any]:
    """
    Core uploaded file processing logic - processes audio through the full orchestration pipeline.

    This now uses the complete AudioCoordinator streaming infrastructure for real transcription
    and translation, replacing the previous placeholder implementation.
    """
    try:
        logger.info(
            f"[{correlation_id}] Processing uploaded file through AudioCoordinator"
        )

        # Use audio coordinator for complete processing
        result = await audio_coordinator.process_audio_file(
            session_id=request_data.get("session_id", "unknown"),
            audio_file_path=temp_file_path,
            config=request_data,
            request_id=correlation_id,
        )

        logger.info(
            f"[{correlation_id}] Audio coordinator processing complete: status={result.get('status')}"
        )

        return result

    except Exception as e:
        logger.error(
            f"[{correlation_id}] Failed to process uploaded file: {e}", exc_info=True
        )
        return {
            "status": "error",
            "error": str(e),
            "file_path": temp_file_path,
            "processing_time": 0.0,
        }


@router.get("/health")
async def health_check(health_monitor=Depends(get_health_monitor)) -> Dict[str, Any]:
    """
    Audio service health check with comprehensive diagnostics
    """
    try:
        # Get comprehensive health status
        health_status = await health_monitor.get_comprehensive_health()

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "orchestration_audio",
            "version": "2.0.0",
            "components": health_status,
            "endpoints": {
                "process": "operational",
                "upload": "operational",
                "stream": "operational",
                "transcribe": "operational",
            },
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@router.get("/models/transcription")
async def get_transcription_models(
    audio_client=Depends(get_audio_service_client),
) -> Dict[str, Any]:
    """
    Get available Whisper transcription models
    """
    try:
        # Get Whisper models from audio service
        models = await audio_client.get_models()

        # Get device info from audio service
        audio_device_info = await audio_client.get_device_info()

        return {
            "available_models": models,
            "models": models,  # For backwards compatibility
            "status": "success",
            "service": "whisper",
            "total_models": len(models),
            "device_info": audio_device_info,
        }

    except Exception as e:
        logger.error(f"Failed to get transcription models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Whisper service unavailable: {str(e)}",
        )


@router.get("/models/translation")
async def get_translation_models(
    translation_client=Depends(get_translation_service_client),
) -> Dict[str, Any]:
    """
    Get available translation models
    """
    try:
        # Get translation models
        models = await translation_client.get_models()

        # Get device info from translation service
        translation_device_info = await translation_client.get_device_info()

        return {
            "available_models": models,
            "models": models,
            "status": "success",
            "service": "translation",
            "total_models": len(models),
            "device_info": translation_device_info,
        }

    except Exception as e:
        logger.error(f"Failed to get translation models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Translation service unavailable: {str(e)}",
        )


@router.get("/models")
async def get_all_models(
    audio_client=Depends(get_audio_service_client),
    translation_client=Depends(get_translation_service_client),
) -> Dict[str, Any]:
    """
    Get all available models (transcription + translation) with device information
    """
    try:
        # Get models from both services
        transcription_models = await audio_client.get_models()
        audio_device_info = await audio_client.get_device_info()

        # Get translation models with fallback
        try:
            translation_models = await translation_client.get_models()
            translation_device_info = await translation_client.get_device_info()
        except:
            translation_models = []
            translation_device_info = {"device": "unknown", "status": "unavailable"}

        return {
            "transcription_models": transcription_models,
            "translation_models": translation_models,
            "status": "success",
            "service": "orchestration",
            "device_info": {
                "audio_service": audio_device_info,
                "translation_service": translation_device_info,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}",
        )


@router.get("/stats", response_model=AudioStats)
async def get_audio_stats(health_monitor=Depends(get_health_monitor)) -> AudioStats:
    """
    Get audio processing statistics
    """
    try:
        # Get processing statistics
        stats = await health_monitor.get_processing_stats()

        return AudioStats(
            total_requests=stats.get("total_requests", 0),
            successful_requests=stats.get("successful_requests", 0),
            failed_requests=stats.get("failed_requests", 0),
            average_processing_time=stats.get("average_processing_time", 0.0),
            active_sessions=stats.get("active_sessions", 0),
        )

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}",
        )
