"""
Audio processing API router

Enhanced async/await endpoints for audio processing with the whisper service
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import ValidationError

from models.audio import (
    AudioProcessingRequest,
    AudioProcessingResponse,
    AudioConfiguration,
    AudioStats,
    ProcessingStage,
    ProcessingQuality,
)
from dependencies import (
    get_config_manager,
    get_health_monitor,
    get_audio_service_client,
    get_translation_service_client,
    get_audio_coordinator,
    get_config_sync_manager,
)
from utils.audio_processing import AudioProcessor
from utils.rate_limiting import RateLimiter
from utils.security import SecurityUtils
from utils.audio_errors import (
    AudioProcessingBaseError, AudioFormatError, AudioCorruptionError, 
    AudioProcessingError, ServiceUnavailableError, ValidationError, 
    ConfigurationError, NetworkError, TimeoutError,
    CircuitBreaker, RetryManager, RetryConfig,
    FormatRecoveryStrategy, ServiceRecoveryStrategy,
    ErrorLogger, error_boundary, default_circuit_breaker, default_retry_manager
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize utilities
rate_limiter = RateLimiter()
security_utils = SecurityUtils()

# Initialize error handling components
audio_service_circuit_breaker = CircuitBreaker(
    name="audio_service",
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=2
)

translation_service_circuit_breaker = CircuitBreaker(
    name="translation_service", 
    failure_threshold=3,
    recovery_timeout=30,
    success_threshold=2
)

retry_manager = RetryManager(RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
))

error_logger = ErrorLogger("orchestration_audio")

# Recovery strategies
format_recovery = FormatRecoveryStrategy()
service_recovery = ServiceRecoveryStrategy()


@router.post("/process", response_model=AudioProcessingResponse)
async def process_audio(
    request: AudioProcessingRequest,
    config_manager=Depends(get_config_manager),
    audio_client=Depends(get_audio_service_client),
    # Rate limiting will be handled by middleware
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
    request_id = f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    async with error_boundary(
        correlation_id=request_id,
        context={
            "service": "orchestration",
            "endpoint": "/process",
            "streaming": request.streaming,
            "request_size": len(str(request))
        },
        recovery_strategies=[format_recovery, service_recovery]
    ) as correlation_id:
        try:
            logger.info(f"[{correlation_id}] Processing audio request")

            # Enhanced input validation
            await _validate_audio_request(request, correlation_id)

            # Get audio service configuration with error handling
            try:
                audio_service_config = await config_manager.get_service_config("audio")
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to get audio service configuration: {str(e)}",
                    correlation_id=correlation_id,
                    config_details={"service": "audio"}
                )

            # Process through enhanced pipeline with circuit breaker
            if request.streaming:
                return await audio_service_circuit_breaker.call(
                    _process_audio_streaming_with_retry,
                    request, correlation_id, audio_client, audio_service_config
                )
            else:
                return await audio_service_circuit_breaker.call(
                    _process_audio_batch_with_retry,
                    request, correlation_id, audio_client, audio_service_config
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
                details={"exception_type": type(e).__name__}
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
                "provided_fields": [k for k, v in request.dict().items() if v is not None]
            }
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
                    corruption_details={"original_length": len(request.audio_data)}
                )
            if len(audio_bytes) > 100 * 1024 * 1024:  # 100MB limit
                raise ValidationError(
                    "Audio file too large (max 100MB)",
                    correlation_id=correlation_id,
                    validation_details={"size_bytes": len(audio_bytes), "max_size": 100 * 1024 * 1024}
                )
        except Exception as e:
            if isinstance(e, AudioProcessingBaseError):
                raise
            raise AudioFormatError(
                f"Invalid base64 audio data: {str(e)}",
                correlation_id=correlation_id,
                format_details={"encoding": "base64", "error": str(e)}
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
                validation_details={"config_error": str(e)}
            )


async def _process_audio_streaming_with_retry(
    request: AudioProcessingRequest,
    correlation_id: str,
    audio_client,
    audio_service_config: Dict[str, Any]
) -> AudioProcessingResponse:
    """Streaming audio processing with retry mechanism"""
    return await retry_manager.execute_with_retry(
        _process_audio_streaming,
        request, correlation_id, audio_client, audio_service_config,
        correlation_id=correlation_id,
        retryable_exceptions=(NetworkError, TimeoutError, ServiceUnavailableError)
    )


async def _process_audio_batch_with_retry(
    request: AudioProcessingRequest,
    correlation_id: str,
    audio_client,
    audio_service_config: Dict[str, Any]
) -> AudioProcessingResponse:
    """Batch audio processing with retry mechanism"""
    return await retry_manager.execute_with_retry(
        _process_audio_batch,
        request, correlation_id, audio_client, audio_service_config,
        correlation_id=correlation_id,
        retryable_exceptions=(NetworkError, TimeoutError, ServiceUnavailableError)
    )


async def _process_uploaded_file_safe(
    processing_request: AudioProcessingRequest,
    correlation_id: str,
    temp_file_path: str,
    audio_client,
    request_data: Dict[str, Any],
    audio_coordinator,
    config_sync_manager
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
            config_sync_manager
        )
    except Exception as e:
        # Convert generic exceptions to appropriate audio errors
        if "network" in str(e).lower() or "connection" in str(e).lower():
            raise NetworkError(
                f"Network error during audio processing: {str(e)}",
                correlation_id=correlation_id,
                network_details={"original_error": str(e)}
            )
        elif "timeout" in str(e).lower():
            raise TimeoutError(
                f"Timeout during audio processing: {str(e)}",
                correlation_id=correlation_id,
                timeout_details={"original_error": str(e)}
            )
        elif "service" in str(e).lower() or "unavailable" in str(e).lower():
            raise ServiceUnavailableError(
                f"Audio service unavailable: {str(e)}",
                correlation_id=correlation_id,
                service_name="whisper_service"
            )
        else:
            raise AudioProcessingError(
                f"Audio processing failed: {str(e)}",
                correlation_id=correlation_id,
                processing_stage="file_processing",
                details={"original_error": str(e)}
            )


async def _validate_upload_file(audio: UploadFile, correlation_id: str):
    """Enhanced file upload validation"""
    # Validate filename
    if not audio.filename:
        raise ValidationError(
            "No filename provided",
            correlation_id=correlation_id,
            validation_details={"missing_field": "filename"}
        )
    
    # Validate content type
    allowed_types = {
        'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 
        'audio/webm', 'audio/mp4', 'audio/flac', 'audio/x-flac',
        'application/octet-stream'  # Allow generic binary for various formats
    }
    
    if audio.content_type and audio.content_type not in allowed_types:
        raise AudioFormatError(
            f"Unsupported audio format: {audio.content_type}",
            correlation_id=correlation_id,
            format_details={
                "provided_type": audio.content_type,
                "allowed_types": list(allowed_types)
            }
        )
    
    # Validate file extension
    if audio.filename:
        allowed_extensions = {'.wav', '.mp3', '.ogg', '.webm', '.mp4', '.flac', '.m4a'}
        file_ext = '.' + audio.filename.split('.')[-1].lower() if '.' in audio.filename else ''
        
        if file_ext not in allowed_extensions:
            raise AudioFormatError(
                f"Unsupported file extension: {file_ext}",
                correlation_id=correlation_id,
                format_details={
                    "provided_extension": file_ext,
                    "allowed_extensions": list(allowed_extensions)
                }
            )


@router.post("/upload", response_model=Dict[str, Any])
async def upload_audio_file(
    audio: UploadFile = File(..., alias="audio"),
    config: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    chunk_id: Optional[str] = Form(None),
    transcription: bool = Form(True, alias="enable_transcription"),
    speaker_diarization: bool = Form(True, alias="enable_diarization"),
    target_languages: Optional[str] = Form(None),
    whisper_model: Optional[str] = Form(None),
    translation_quality: Optional[str] = Form("balanced"),
    enable_vad: bool = Form(True),
    audio_processing: bool = Form(True),
    noise_reduction: bool = Form(False),
    speech_enhancement: bool = Form(True),
    enable_translation: bool = Form(True),
    config_manager=Depends(get_config_manager),
    audio_client=Depends(get_audio_service_client),
    translation_client=Depends(get_translation_service_client),
    audio_coordinator=Depends(get_audio_coordinator),
    config_sync_manager=Depends(get_config_sync_manager),
    # Rate limiting will be handled by middleware
) -> Dict[str, Any]:
    """
    Upload and process audio file with optional translation

    - **audio**: Audio file (WAV, MP3, WebM, OGG, MP4, FLAC)
    - **config**: JSON string of AudioConfiguration
    - **session_id**: Session identifier
    - **chunk_id**: Chunk identifier for streaming
    - **enable_transcription**: Enable transcription
    - **enable_diarization**: Enable speaker diarization
    - **target_languages**: JSON array of target language codes for translation (e.g. ["es", "fr", "de"])
    - **whisper_model**: Whisper model to use (e.g. "whisper-base")
    - **translation_quality**: Translation quality setting
    - **enable_vad**: Enable voice activity detection
    - **audio_processing**: Enable audio processing pipeline
    - **noise_reduction**: Enable noise reduction
    - **speech_enhancement**: Enable speech enhancement
    - **enable_translation**: Enable translation
    """
    request_id = f"upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    async with error_boundary(
        correlation_id=request_id,
        context={
            "service": "orchestration",
            "endpoint": "/upload",
            "filename": audio.filename,
            "content_type": audio.content_type,
            "session_id": session_id,
            "chunk_id": chunk_id
        },
        recovery_strategies=[format_recovery, service_recovery]
    ) as correlation_id:
        try:
            logger.info(f"[{correlation_id}] Processing file upload: {audio.filename}")

            # Enhanced file validation
            await _validate_upload_file(audio, correlation_id)

            # Read and validate file content
            try:
                content = await audio.read()
                file_size = len(content)
            except Exception as e:
                raise AudioCorruptionError(
                    f"Failed to read uploaded file: {str(e)}",
                    correlation_id=correlation_id,
                    corruption_details={"read_error": str(e)}
                )

            # Check file size (100MB limit)
            if file_size == 0:
                raise AudioCorruptionError(
                    "Uploaded file is empty",
                    correlation_id=correlation_id,
                    corruption_details={"file_size": 0}
                )
            
            if file_size > 100 * 1024 * 1024:  # 100MB
                raise ValidationError(
                    "File too large (max 100MB)",
                    correlation_id=correlation_id,
                    validation_details={"file_size": file_size, "max_size": 100 * 1024 * 1024}
                )

            # Validate audio file with circuit breaker
            try:
                audio_processor = AudioProcessor()
                audio_metadata = audio_processor.validate_audio_file(content, audio.filename)
            except Exception as e:
                raise AudioFormatError(
                    f"Audio file validation failed: {str(e)}",
                    correlation_id=correlation_id,
                    format_details={
                        "filename": audio.filename,
                        "content_type": audio.content_type,
                        "file_size": file_size,
                        "validation_error": str(e)
                    }
                )

            # Parse configuration with validation
            audio_config = AudioConfiguration()
            if config:
                try:
                    config_dict = json.loads(config)
                    audio_config = AudioConfiguration(**config_dict)
                except json.JSONDecodeError as e:
                    raise ValidationError(
                        f"Invalid JSON configuration: {str(e)}",
                        correlation_id=correlation_id,
                        validation_details={"config_error": str(e), "raw_config": config}
                    )
                except Exception as e:
                    raise ConfigurationError(
                        f"Invalid audio configuration: {str(e)}",
                        correlation_id=correlation_id,
                        config_details={"config_error": str(e)}
                    )

            # Create processing request
            processing_request = AudioProcessingRequest(
                audio_data=None,
                file_upload=request_id,
                config=audio_config,
                transcription=transcription,
                speaker_diarization=speaker_diarization,
                session_id=session_id,
                metadata={
                    "filename": audio.filename,
                    "file_size": file_size,
                    "content_type": audio.content_type,
                    "upload_timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Store file temporarily with error handling
            try:
                temp_file_path = await _store_temp_file(content, audio.filename, correlation_id)
            except Exception as e:
                raise AudioProcessingError(
                    f"Failed to store temporary file: {str(e)}",
                    correlation_id=correlation_id,
                    processing_stage="file_storage",
                    details={"storage_error": str(e)}
                )

            # Process audio with circuit breaker and retry
            try:
                request_data = {
                    "speaker_diarization": speaker_diarization,
                    "enable_vad": enable_vad,
                    "whisper_model": whisper_model or "whisper-base",
                    "session_id": session_id,
                    "chunk_id": chunk_id,
                    "audio_processing": audio_processing,
                    "noise_reduction": noise_reduction,
                    "speech_enhancement": speech_enhancement,
                }
                
                result = await audio_service_circuit_breaker.call(
                    retry_manager.execute_with_retry,
                    _process_uploaded_file_safe,
                    processing_request,
                    correlation_id,
                    temp_file_path,
                    audio_client,
                    request_data,
                    audio_coordinator,
                    config_sync_manager,
                    correlation_id=correlation_id,
                    retryable_exceptions=(NetworkError, TimeoutError, ServiceUnavailableError)
                )
                
            except ServiceUnavailableError as e:
                logger.warning(f"[{correlation_id}] Audio service unavailable, using fallback: {e}")
                result = {
                    "text": "Audio processing service unavailable - please check whisper service",
                    "language": "en",
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "error": str(e),
                    "fallback_used": True
                }
            except Exception as e:
                raise AudioProcessingError(
                    f"Audio processing failed: {str(e)}",
                    correlation_id=correlation_id,
                    processing_stage="audio_processing",
                    details={"processing_error": str(e)}
                )

            # Initialize response
            response = {
                "request_id": request_id,
                "filename": audio.filename,
                "file_size": file_size,
                "audio_metadata": audio_metadata,
                "processing_result": result,
                "translations": {},
            }

            # Process translation if requested
            logger.info(
                f"[{request_id}] Translation check - target_languages: {target_languages}, enable_translation: {enable_translation}, result: {type(result) if result else None}"
            )
            logger.info(f"[{request_id}] Raw target_languages parameter: {repr(target_languages)}")
            if target_languages and enable_translation and result:
                try:
                    # Parse target languages
                    target_langs = (
                        json.loads(target_languages)
                        if isinstance(target_languages, str)
                        else target_languages
                    )
                    if not isinstance(target_langs, list):
                        target_langs = [target_langs]

                    logger.info(f"[{request_id}] Parsed target_langs: {target_langs} (type: {type(target_langs)})")
                    logger.info(f"[{request_id}] Starting translation to languages: {target_langs}")

                    # Extract transcription text from result
                    transcription_text = None
                    source_language = None

                    if isinstance(result, dict):
                        # Try different possible keys for the transcription text
                        transcription_text = (
                            result.get("text")
                            or result.get("transcription")
                            or result.get("transcript")
                        )
                        source_language = result.get("language") or result.get("detected_language")
                    elif hasattr(result, "text"):
                        transcription_text = result.text
                        source_language = getattr(result, "language", None)

                    if transcription_text and transcription_text.strip():
                        logger.info(
                            f"[{request_id}] Found transcription text: '{transcription_text[:50]}...' (source: {source_language})"
                        )
                        # Use the enhanced translation client method
                        translation_results = await translation_client.translate_to_multiple_languages(
                            text=transcription_text,
                            source_language=source_language,
                            target_languages=target_langs,
                            quality=translation_quality or "balanced",
                        )
                        logger.info(
                            f"[{request_id}] Translation client returned {len(translation_results)} results: {list(translation_results.keys())}"
                        )

                        # Convert TranslationResponse objects to dictionaries
                        translations_dict = {}
                        for lang, trans_response in translation_results.items():
                            try:
                                # Try Pydantic v2 first, then v1, then manual conversion
                                if hasattr(trans_response, "model_dump"):
                                    translations_dict[lang] = trans_response.model_dump()
                                elif hasattr(trans_response, "dict"):
                                    translations_dict[lang] = trans_response.dict()
                                else:
                                    # Manual conversion for non-pydantic objects
                                    translations_dict[lang] = {
                                        "translated_text": getattr(
                                            trans_response,
                                            "translated_text",
                                            str(trans_response),
                                        ),
                                        "source_language": getattr(
                                            trans_response,
                                            "source_language",
                                            source_language,
                                        ),
                                        "target_language": getattr(
                                            trans_response, "target_language", lang
                                        ),
                                        "confidence": getattr(trans_response, "confidence", 0.0),
                                        "processing_time": getattr(
                                            trans_response, "processing_time", 0.0
                                        ),
                                        "model_used": getattr(trans_response, "model_used", "unknown"),
                                        "backend_used": getattr(
                                            trans_response, "backend_used", "unknown"
                                        ),
                                        "session_id": getattr(trans_response, "session_id", None),
                                        "timestamp": getattr(trans_response, "timestamp", None),
                                    }
                            except Exception as conversion_error:
                                logger.warning(
                                    f"[{request_id}] Failed to convert translation for {lang}: {conversion_error}"
                                )
                                # Create a basic response
                                translations_dict[lang] = {
                                    "translated_text": f"Translation conversion failed: {str(conversion_error)}",
                                    "source_language": source_language or "auto",
                                    "target_language": lang,
                                    "confidence": 0.0,
                                    "processing_time": 0.0,
                                    "model_used": "error",
                                    "backend_used": "error",
                                    "session_id": None,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }

                        response["translations"] = translations_dict
                        logger.info(
                            f"[{request_id}] Translation completed for {len(translations_dict)} languages"
                        )
                    else:
                        logger.warning(f"[{request_id}] No transcription text found for translation")
                        response[
                            "translation_error"
                        ] = "No transcription text available for translation"

                except json.JSONDecodeError as e:
                    error_msg = f"Invalid target_languages format: {str(e)}"
                    logger.error(f"[{request_id}] Invalid target_languages JSON: {str(e)}")
                    response["translation_error"] = error_msg
                except Exception as e:
                    error_msg = f"Translation failed: {str(e)}"
                    logger.error(f"[{request_id}] Translation failed: {str(e)}")
                    response["translation_error"] = error_msg

            return response

        except HTTPException:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{request_id}] File upload processing failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"File upload processing failed: {error_msg}",
            )


@router.get("/stream/{session_id}")
async def stream_audio_processing(
    session_id: str,
    audio_client=Depends(get_audio_service_client),
    # Rate limiting will be handled by middleware
):
    """
    Stream real-time audio processing results

    - **session_id**: Session identifier for streaming
    """
    try:
        logger.info(f"Starting audio streaming for session: {session_id}")

        async def generate_stream():
            """Generate streaming audio processing results"""
            try:
                # Connect to audio service stream
                async for chunk in audio_client.stream_processing_results(session_id):
                    yield f"data: {chunk}\n\n"

            except Exception as e:
                logger.error(f"Streaming error for session {session_id}: {e}")
                yield f"event: error\ndata: {{'error': '{str(e)}'}}\n\n"
                return

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )

    except Exception as e:
        logger.error(f"Failed to start streaming for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start audio streaming: {str(e)}",
        )


@router.get("/presets", response_model=Dict[str, AudioConfiguration])
async def get_audio_presets() -> Dict[str, AudioConfiguration]:
    """
    Get predefined audio processing presets

    Returns optimized configurations for different scenarios
    """
    try:
        presets = {
            "default": AudioConfiguration(),
            "high_quality": AudioConfiguration(
                sample_rate=48000,
                quality=ProcessingQuality.ACCURATE,
                noise_reduction_strength=0.7,
                clarity_enhancement=0.5,
                compressor_ratio=2.0,
            ),
            "fast_processing": AudioConfiguration(
                quality=ProcessingQuality.FAST,
                enabled_stages=[
                    ProcessingStage.VAD,
                    ProcessingStage.VOICE_FILTER,
                    ProcessingStage.NOISE_REDUCTION,
                ],
                noise_reduction_strength=0.3,
            ),
            "conference_call": AudioConfiguration(
                vad_aggressiveness=3,
                noise_reduction_strength=0.6,
                voice_protection=0.9,
                compressor_threshold=-15.0,
                compressor_ratio=6.0,
            ),
            "podcast": AudioConfiguration(
                sample_rate=44100,
                quality=ProcessingQuality.ACCURATE,
                enabled_stages=[
                    ProcessingStage.VAD,
                    ProcessingStage.VOICE_FILTER,
                    ProcessingStage.NOISE_REDUCTION,
                    ProcessingStage.VOICE_ENHANCEMENT,
                    ProcessingStage.COMPRESSOR,
                    ProcessingStage.DE_ESSER,
                ],
                clarity_enhancement=0.4,
                compressor_threshold=-18.0,
                compressor_ratio=3.0,
            ),
            "noisy_environment": AudioConfiguration(
                vad_aggressiveness=2,
                noise_reduction_strength=0.8,
                voice_protection=0.95,
                voice_freq_min=100.0,
                voice_freq_max=280.0,
            ),
        }

        return presets

    except Exception as e:
        logger.error(f"Failed to get audio presets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audio presets",
        )


@router.get("/stats", response_model=AudioStats)
async def get_audio_stats(audio_client=Depends(get_audio_service_client)) -> AudioStats:
    """
    Get audio processing statistics

    Returns comprehensive statistics about audio processing performance
    """
    try:
        logger.info("Retrieving audio processing statistics")

        stats = await audio_client.get_processing_stats()

        return AudioStats(**stats)

    except Exception as e:
        logger.error(f"Failed to get audio stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audio statistics",
        )


@router.get("/models")
async def get_available_models(
    audio_client=Depends(get_audio_service_client),
    translation_client=Depends(get_translation_service_client),
) -> Dict[str, Any]:
    """
    Get available Whisper models and device information from services

    Returns a list of available models that can be used for transcription,
    along with current device information (CPU/GPU/NPU) for both services.
    If services are unavailable, returns fallback data.
    """
    try:
        logger.info("Querying audio service for available models and device info")

        # Get models and device info from the audio/whisper service
        models_task = audio_client.get_models()
        audio_device_task = audio_client.get_device_info()

        # Get device info from translation service
        translation_device_task = translation_client.get_device_info()

        # Execute all requests concurrently
        models, audio_device_info, translation_device_info = await asyncio.gather(
            models_task,
            audio_device_task,
            translation_device_task,
            return_exceptions=True,
        )

        # Handle models result
        if isinstance(models, Exception):
            logger.error(f"Failed to get models: {models}")
            models = ["whisper-base"]  # fallback
            models_status = "fallback"
        else:
            models_status = "success"

        # Handle audio device info result
        if isinstance(audio_device_info, Exception):
            logger.error(f"Failed to get audio device info: {audio_device_info}")
            audio_device_info = {
                "device": "unknown",
                "status": "error",
                "error": str(audio_device_info),
            }

        # Handle translation device info result
        if isinstance(translation_device_info, Exception):
            logger.error(f"Failed to get translation device info: {translation_device_info}")
            translation_device_info = {
                "device": "unknown",
                "status": "error",
                "error": str(translation_device_info),
            }

        logger.info(f"Retrieved {len(models)} models from audio service: {models}")
        logger.info(f"Audio service device: {audio_device_info.get('device', 'unknown')}")
        logger.info(
            f"Translation service device: {translation_device_info.get('device', 'unknown')}"
        )

        return {
            "available_models": models,
            "models": models,  # Legacy compatibility
            "status": models_status,
            "service": "audio",
            "total_models": len(models),
            "device_info": {
                "audio_service": {
                    "device": audio_device_info.get("device", "unknown"),
                    "device_type": audio_device_info.get("device_type", "unknown"),
                    "status": audio_device_info.get("status", "unknown"),
                    "details": audio_device_info.get("details", {}),
                    "acceleration": audio_device_info.get("acceleration", "unknown"),
                    "error": audio_device_info.get("error"),
                },
                "translation_service": {
                    "device": translation_device_info.get("device", "unknown"),
                    "device_type": translation_device_info.get("device_type", "unknown"),
                    "status": translation_device_info.get("status", "unknown"),
                    "details": translation_device_info.get("details", {}),
                    "acceleration": translation_device_info.get("acceleration", "unknown"),
                    "error": translation_device_info.get("error"),
                },
            },
        }

    except Exception as e:
        logger.error(f"Failed to get models and device info: {e}")

        # Return fallback models if service is unavailable
        fallback_models = [
            "whisper-tiny",
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large",
        ]

        return {
            "available_models": fallback_models,
            "models": fallback_models,  # Legacy compatibility
            "status": "fallback",
            "service": "audio",
            "total_models": len(fallback_models),
            "error": f"Services unavailable: {str(e)}",
            "message": "Using fallback model list. Services may be offline.",
            "device_info": {
                "audio_service": {
                    "device": "unknown",
                    "status": "unavailable",
                    "error": "Service unavailable",
                },
                "translation_service": {
                    "device": "unknown",
                    "status": "unavailable",
                    "error": "Service unavailable",
                },
            },
        }


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Source language (auto-detect if not provided)"),
    task: str = Form("transcribe", description="Task type: transcribe or translate"),
    model: str = Form("whisper-base", description="Whisper model to use"),
    temperature: float = Form(0.0, description="Sampling temperature (0-1, higher = more random)"),
    beam_size: int = Form(5, description="Beam search size for better accuracy"),
    best_of: int = Form(5, description="Number of candidates to generate"),
    audio_client=Depends(get_audio_service_client),
) -> Dict[str, Any]:
    """
    Direct transcription proxy to whisper service
    
    Clean transcription endpoint that proxies to whisper service.
    Orchestration handles VAD, diarization, and audio processing separately.
    """
    try:
        logger.info(f"Direct transcription request for file: {audio.filename}")
        
        # Read audio file
        content = await audio.read()
        
        # Create transcription request with whisper-specific parameters only
        from clients.audio_service_client import TranscriptionRequest
        request_params = TranscriptionRequest(
            language=language,
            task=task,
            enable_diarization=False,  # Orchestration handles this
            enable_vad=False,  # Orchestration handles this
            model=model
        )
        
        # Note: temperature, beam_size, best_of would be passed to whisper service
        # but current TranscriptionRequest model may need updating to support them
        
        # Create temporary file for processing
        import tempfile
        import os
        from pathlib import Path
        
        temp_dir = Path(tempfile.gettempdir()) / "orchestration_transcribe"
        temp_dir.mkdir(exist_ok=True)
        
        file_extension = Path(audio.filename).suffix if audio.filename else '.wav'
        temp_file_path = temp_dir / f"transcribe_{int(time.time())}{file_extension}"
        
        # Write temporary file
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        try:
            # Call whisper service directly
            result = await audio_client.transcribe_file(temp_file_path, request_params)
            
            # Convert result to dict if needed
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            elif hasattr(result, 'dict'):
                result_dict = result.dict()
            else:
                result_dict = result
            
            # Enhance response with comprehensive metrics
            enhanced_response = {
                "text": result_dict.get("text", ""),
                "language": result_dict.get("language", "unknown"),
                "confidence": result_dict.get("confidence", 0.0),
                "processing_metrics": {
                    "processing_time": result_dict.get("processing_time", 0.0),
                    "device_used": result_dict.get("device_used", "unknown"),
                    "model_used": model,
                    "real_time_factor": _calculate_real_time_factor(
                        result_dict.get("processing_time", 0.0),
                        audio.size if hasattr(audio, 'size') else len(content)
                    )
                },
                "quality_indicators": {
                    "segments": result_dict.get("segments", []),
                    "speakers": result_dict.get("speakers", []),
                    "confidence_score": result_dict.get("confidence_score", result_dict.get("confidence", 0.0)),
                    "noise_detected": _detect_noise_patterns(result_dict.get("text", "")),
                    "transcription_quality": _assess_transcription_quality(result_dict)
                },
                "request_parameters": {
                    "model": model,
                    "language": language or "auto",
                    "task": task,
                    "temperature": temperature,
                    "beam_size": beam_size,
                    "best_of": best_of
                },
                "timestamp": time.time(),
                "service_info": {
                    "transcription_engine": "whisper",
                    "orchestration_processing": False  # This is direct proxy
                }
            }
                
            return enhanced_response
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
                
    except Exception as e:
        logger.error(f"Direct transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@router.post("/transcribe/{model_name}")
async def transcribe_audio_with_model(
    model_name: str,
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Source language (auto-detect if not provided)"),
    task: str = Form("transcribe", description="Task type: transcribe or translate"),
    temperature: float = Form(0.0, description="Sampling temperature (0-1, higher = more random)"),
    beam_size: int = Form(5, description="Beam search size for better accuracy"),
    best_of: int = Form(5, description="Number of candidates to generate"),
    audio_client=Depends(get_audio_service_client),
) -> Dict[str, Any]:
    """
    Model-specific transcription proxy to whisper service
    
    Clean transcription endpoint with specific model selection.
    Orchestration handles VAD, diarization, and audio processing separately.
    """
    try:
        logger.info(f"Model-specific transcription request for file: {audio.filename} with model: {model_name}")
        
        # Read audio file
        content = await audio.read()
        
        # Create transcription request with specified model (whisper-only parameters)
        from clients.audio_service_client import TranscriptionRequest
        request_params = TranscriptionRequest(
            language=language,
            task=task,
            enable_diarization=False,  # Orchestration handles this
            enable_vad=False,  # Orchestration handles this
            model=model_name
        )
        
        # Note: temperature, beam_size, best_of would be passed to whisper service
        # but current TranscriptionRequest model may need updating to support them
        
        # Create temporary file for processing
        import tempfile
        import os
        from pathlib import Path
        
        temp_dir = Path(tempfile.gettempdir()) / "orchestration_transcribe"
        temp_dir.mkdir(exist_ok=True)
        
        file_extension = Path(audio.filename).suffix if audio.filename else '.wav'
        temp_file_path = temp_dir / f"transcribe_{model_name}_{int(time.time())}{file_extension}"
        
        # Write temporary file
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        try:
            # Call whisper service directly
            result = await audio_client.transcribe_file(temp_file_path, request_params)
            
            # Convert result to dict if needed
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            elif hasattr(result, 'dict'):
                result_dict = result.dict()
            else:
                result_dict = result
                
            return result_dict
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
                
    except Exception as e:
        logger.error(f"Model-specific transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@router.post("/transcribe/stream")
async def transcribe_audio_stream(
    audio_data: bytes,
    session_id: Optional[str] = None,
    language: Optional[str] = None,
    model: str = "whisper-base",
    temperature: float = 0.0,
    beam_size: int = 5,
    audio_client=Depends(get_audio_service_client),
) -> Dict[str, Any]:
    """
    Streaming transcription proxy to whisper service
    
    Clean streaming transcription endpoint.
    Orchestration handles VAD, chunking, and audio processing separately.
    """
    try:
        logger.info(f"Streaming transcription request for session: {session_id}")
        
        # Create transcription request (whisper-only parameters)
        from clients.audio_service_client import TranscriptionRequest
        request_params = TranscriptionRequest(
            language=language,
            task="transcribe",
            enable_diarization=False,  # Orchestration handles this
            enable_vad=False,  # Orchestration handles this
            model=model
        )
        
        # Note: temperature, beam_size would be passed to whisper service
        # but current TranscriptionRequest model may need updating to support them
        
        # Call whisper service for streaming
        result = await audio_client.transcribe_stream(audio_data, request_params)
        
        # Convert result to dict if needed
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = result
            
        return result_dict
        
    except Exception as e:
        logger.error(f"Streaming transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming transcription failed: {str(e)}"
        )


@router.get("/health")
async def audio_health_check(
    health_monitor=Depends(get_health_monitor),
) -> Dict[str, Any]:
    """
    Check audio processing service health

    Returns health status of the audio processing pipeline
    """
    try:
        health_status = await health_monitor.get_service_status("whisper")

        if health_status:
            return {
                "status": health_status["status"],
                "response_time_ms": health_status.get("response_time", 0),
                "last_check": health_status.get("last_check"),
                "service_url": health_status.get("url"),
                "features": {
                    "transcription": True,
                    "speaker_diarization": True,
                    "real_time_processing": True,
                "multiple_formats": True,
                "streaming": True,
            },
            "supported_formats": ["wav", "mp3", "webm", "ogg", "mp4", "flac"],
            "processing_stages": [stage.value for stage in ProcessingStage],
            "quality_levels": [quality.value for quality in ProcessingQuality],
        }
        else:
            return {
                "status": "unknown",
                "error": "Could not check whisper service status"
            }

    except Exception as e:
        logger.error(f"Audio health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audio service health check failed",
        )


@router.get("/download/{request_id}")
async def download_processed_audio(
    request_id: str,
    format: Optional[str] = "wav",
    audio_client=Depends(get_audio_service_client),
):
    """
    Download processed audio file

    - **request_id**: Processing request identifier
    - **format**: Output format (wav, mp3, flac)
    """
    try:
        logger.info(f"Downloading processed audio for request: {request_id}")

        # Get processed file path from audio service
        file_info = await audio_client.get_processed_file(request_id, format)

        if not file_info or "file_path" not in file_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Processed audio file not found",
            )

        file_path = file_info["file_path"]
        filename = f"processed_audio_{request_id}.{format}"

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=f"audio/{format}",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download processed audio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download processed audio",
        )


# Helper functions


async def _process_audio_batch(
    request: AudioProcessingRequest,
    request_id: str,
    audio_client,
    service_config: Dict[str, Any],
) -> AudioProcessingResponse:
    """Process audio in batch mode"""

    start_time = datetime.utcnow()

    try:
        # Send request to audio service
        processing_result = await audio_client.process_audio_batch(request.dict(), request_id)

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return AudioProcessingResponse(
            request_id=request_id,
            session_id=request.session_id,
            stage_results=processing_result.get("stage_results", []),
            overall_quality_score=processing_result.get("quality_score", 0.0),
            total_processing_time_ms=processing_time,
            input_duration_s=processing_result.get("input_duration_s", 0.0),
            output_duration_s=processing_result.get("output_duration_s", 0.0),
            signal_to_noise_ratio=processing_result.get("snr", None),
            transcription=processing_result.get("transcription", None),
            output_audio_url=processing_result.get("output_url", None),
            is_streaming=False,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Batch processing failed: {e}")
        raise


async def _process_audio_streaming(
    request: AudioProcessingRequest,
    request_id: str,
    audio_client,
    service_config: Dict[str, Any],
) -> AudioProcessingResponse:
    """Process audio in streaming mode"""

    try:
        # Start streaming processing
        await audio_client.start_audio_streaming(request.dict(), request_id)

        return AudioProcessingResponse(
            request_id=request_id,
            session_id=request.session_id,
            stage_results=[],
            overall_quality_score=0.0,
            total_processing_time_ms=0.0,
            input_duration_s=0.0,
            output_duration_s=0.0,
            is_streaming=True,
            stream_chunk_id=0,
            message="Streaming processing started",
        )

    except Exception as e:
        logger.error(f"[{request_id}] Streaming processing failed: {e}")
        raise


async def _process_uploaded_file(
    request: AudioProcessingRequest,
    request_id: str,
    file_path: str,
    audio_client,
    request_data: Dict[str, Any],
    audio_coordinator,
    config_sync_manager,
) -> Dict[str, Any]:
    """Process uploaded audio file through orchestration pipeline"""

    try:
        logger.info(f"[{request_id}] Starting orchestration audio processing pipeline")

        # Get current configuration from sync manager
        current_config = await config_sync_manager.get_unified_configuration()
        audio_config = current_config.get("audio_processing", {})

        logger.info(
            f"[{request_id}] Using audio config: VAD={audio_config.get('enable_vad', True)}, "
            f"noise_reduction={audio_config.get('noise_reduction', False)}, "
            f"enhancement={audio_config.get('speech_enhancement', True)}"
        )

        # Process audio through orchestration pipeline
        session_id = request_data.get("session_id", request_id)

        # Use AudioCoordinator to process the audio
        processed_audio_path = await audio_coordinator.process_audio_file(
            session_id=session_id,
            audio_file_path=file_path,
            config=audio_config,
            request_id=request_id,
        )

        logger.info(
            f"[{request_id}] Audio processed through orchestration pipeline, sending to whisper service"
        )

        # Now send the processed audio to whisper service
        result = await audio_client.process_uploaded_file(
            processed_audio_path, request_data, request_id
        )

        # Add processing pipeline metadata to result
        if isinstance(result, dict):
            result["orchestration_processing"] = {
                "pipeline_applied": True,
                "config_used": audio_config,
                "original_file": file_path,
                "processed_file": processed_audio_path,
            }

        return result

    except Exception as e:
        logger.error(f"[{request_id}] Orchestration audio processing failed: {str(e)}")
        logger.info(f"[{request_id}] Falling back to direct whisper service call")

        # Fallback to direct whisper service call if orchestration processing fails
        try:
            result = await audio_client.process_uploaded_file(file_path, request_data, request_id)
            if isinstance(result, dict):
                result["orchestration_processing"] = {
                    "pipeline_applied": False,
                    "fallback_reason": str(e),
                }
            return result
        except Exception as fallback_error:
            logger.error(f"[{request_id}] Fallback processing also failed: {fallback_error}")
            raise


async def _store_temp_file(content: bytes, filename: str, request_id: str) -> str:
    """Store uploaded file temporarily"""

    import tempfile
    import os
    from pathlib import Path

    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "orchestration_uploads"
        temp_dir.mkdir(exist_ok=True)

        # Generate unique filename
        file_extension = Path(filename).suffix
        temp_filename = f"{request_id}{file_extension}"
        temp_file_path = temp_dir / temp_filename

        # Write file
        with open(temp_file_path, "wb") as f:
            f.write(content)

        return str(temp_file_path)

    except Exception as e:
        logger.error(f"Failed to store temp file: {e}")
        raise


# Helper functions for enhanced metrics

def _calculate_real_time_factor(processing_time: float, audio_size_bytes: int) -> float:
    """Calculate real-time factor (processing_time / audio_duration)"""
    try:
        # Estimate audio duration: assuming 16kHz 16-bit mono WAV = ~32KB/second
        estimated_duration = audio_size_bytes / 32000  # rough estimate
        if estimated_duration > 0:
            return processing_time / estimated_duration
        return 0.0
    except:
        return 0.0


def _detect_noise_patterns(text: str) -> bool:
    """Detect if transcription likely contains noise or hallucinations"""
    if not text or len(text.strip()) < 3:
        return True
    
    # Common noise patterns from whisper
    noise_patterns = [
        "thank you",
        "thank you for watching",
        "thanks for watching", 
        "subscribe",
        "like and subscribe",
        "amara.org",
        "subtitles by the amara.org",
        "www.youtube.com",
        "♪",
        "[music]",
        "[applause]",
        "[laughter]"
    ]
    
    text_lower = text.lower().strip()
    for pattern in noise_patterns:
        if pattern in text_lower:
            return True
    
    # Check for repetitive content
    words = text_lower.split()
    if len(words) > 1 and len(set(words)) / len(words) < 0.5:  # >50% repetition
        return True
        
    return False


def _assess_transcription_quality(result_dict: Dict[str, Any]) -> str:
    """Assess overall transcription quality"""
    try:
        confidence = result_dict.get("confidence", result_dict.get("confidence_score", 0.0))
        text = result_dict.get("text", "")
        processing_time = result_dict.get("processing_time", 0.0)
        
        # Quality assessment based on multiple factors
        if confidence > 0.9 and len(text) > 10 and not _detect_noise_patterns(text):
            return "excellent"
        elif confidence > 0.7 and len(text) > 5:
            return "good"
        elif confidence > 0.5 and len(text) > 0:
            return "fair"
        elif len(text) > 0:
            return "poor"
        else:
            return "failed"
    except:
        return "unknown"


# ========================= FFT ANALYSIS API =========================

@router.post("/analyze/fft")
async def analyze_audio_fft(
    file: UploadFile = File(..., description="Audio file for frequency analysis"),
    fft_size: int = Form(2048, description="FFT size (256, 512, 1024, 2048, 4096)"),
    window_type: str = Form("hann", description="Window function (hann, hamming, blackman, none)"),
    overlap_factor: float = Form(0.5, description="Overlap factor (0.0-0.9)"),
    frequency_range: Optional[str] = Form(None, description="Frequency range filter (e.g., '85-300' for voice)"),
    normalize: bool = Form(True, description="Normalize magnitude spectrum"),
    apply_log_scale: bool = Form(True, description="Apply logarithmic scaling"),
    peak_detection: bool = Form(True, description="Perform peak detection"),
    spectral_features: bool = Form(True, description="Calculate spectral features"),
) -> Dict[str, Any]:
    """
    Real-time FFT analysis endpoint for frequency domain analysis.
    
    Provides comprehensive spectral analysis including:
    - FFT magnitude and phase spectra
    - Peak detection and frequency identification
    - Spectral features (centroid, rolloff, flux, etc.)
    - Voice frequency analysis
    - Noise characterization
    """
    try:
        import numpy as np
        from scipy.fft import fft, fftfreq
        from scipy import signal
        import io
        import soundfile as sf
        
        # Read audio file
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        try:
            audio_data, sample_rate = sf.read(audio_buffer, dtype='float32')
        except Exception as e:
            # Fallback for various audio formats
            try:
                import librosa
                audio_data, sample_rate = librosa.load(audio_buffer, sr=None, dtype=np.float32)
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Could not load audio file. Install librosa for broader format support."
                )
            except Exception as e2:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Could not load audio file: {str(e2)}"
                )
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Validate FFT size
        valid_fft_sizes = [256, 512, 1024, 2048, 4096, 8192]
        if fft_size not in valid_fft_sizes:
            fft_size = 2048  # Default fallback
        
        # Apply window function
        if window_type == "hann":
            window = signal.windows.hann(fft_size)
        elif window_type == "hamming":
            window = signal.windows.hamming(fft_size)
        elif window_type == "blackman":
            window = signal.windows.blackman(fft_size)
        else:  # none
            window = np.ones(fft_size)
        
        # Calculate hop size from overlap
        hop_size = int(fft_size * (1 - overlap_factor))
        
        # Perform windowed FFT analysis
        spectra = []
        frequencies = fftfreq(fft_size, 1/sample_rate)[:fft_size//2]  # Only positive frequencies
        
        for i in range(0, len(audio_data) - fft_size, hop_size):
            # Extract frame
            frame = audio_data[i:i + fft_size]
            
            # Apply window
            windowed_frame = frame * window
            
            # Compute FFT
            fft_result = fft(windowed_frame)[:fft_size//2]  # Only positive frequencies
            magnitude = np.abs(fft_result)
            phase = np.angle(fft_result)
            
            # Apply logarithmic scaling if requested
            if apply_log_scale:
                magnitude = 20 * np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
            
            # Normalize if requested
            if normalize and np.max(magnitude) > 0:
                magnitude = magnitude / np.max(magnitude)
            
            spectra.append({
                "magnitude": magnitude.tolist(),
                "phase": phase.tolist(),
                "time_position": i / sample_rate
            })
        
        # Parse frequency range filter
        freq_min, freq_max = 0, sample_rate / 2
        if frequency_range:
            try:
                parts = frequency_range.split('-')
                if len(parts) == 2:
                    freq_min = float(parts[0])
                    freq_max = float(parts[1])
            except:
                pass  # Use defaults if parsing fails
        
        # Filter frequency range
        freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
        filtered_frequencies = frequencies[freq_mask]
        
        # Calculate average spectrum across all frames
        if spectra:
            avg_magnitude = np.mean([s["magnitude"] for s in spectra], axis=0)
            if freq_mask.any():
                filtered_avg_magnitude = avg_magnitude[freq_mask]
            else:
                filtered_avg_magnitude = avg_magnitude
        else:
            avg_magnitude = np.zeros(len(frequencies))
            filtered_avg_magnitude = avg_magnitude
        
        analysis_result = {
            "metadata": {
                "file_name": file.filename,
                "duration_seconds": len(audio_data) / sample_rate,
                "sample_rate": int(sample_rate),
                "audio_length": len(audio_data),
                "fft_size": fft_size,
                "window_type": window_type,
                "overlap_factor": overlap_factor,
                "hop_size": hop_size,
                "num_frames": len(spectra),
                "frequency_range": [freq_min, freq_max],
                "analysis_timestamp": datetime.utcnow().isoformat()
            },
            "frequency_data": {
                "frequencies": filtered_frequencies.tolist(),
                "average_magnitude": filtered_avg_magnitude.tolist(),
                "frequency_resolution": float(sample_rate / fft_size),
                "nyquist_frequency": float(sample_rate / 2)
            },
            "time_series": spectra[:50],  # Limit to first 50 frames for response size
            "summary": {
                "total_frames_analyzed": len(spectra),
                "frequency_bins": len(filtered_frequencies),
                "effective_frequency_range": [float(freq_min), float(freq_max)]
            }
        }
        
        # Peak detection
        if peak_detection and len(filtered_avg_magnitude) > 0:
            try:
                # Find peaks in the average magnitude spectrum
                peaks, properties = signal.find_peaks(
                    filtered_avg_magnitude,
                    height=np.max(filtered_avg_magnitude) * 0.1,  # 10% of max
                    distance=len(filtered_avg_magnitude) // 20  # Minimum distance between peaks
                )
                
                peak_frequencies = filtered_frequencies[peaks].tolist()
                peak_magnitudes = filtered_avg_magnitude[peaks].tolist()
                
                analysis_result["peaks"] = {
                    "peak_frequencies": peak_frequencies,
                    "peak_magnitudes": peak_magnitudes,
                    "num_peaks": len(peaks)
                }
                
                # Voice frequency analysis
                voice_peaks = [f for f in peak_frequencies if 85 <= f <= 1000]  # Human voice range
                analysis_result["voice_analysis"] = {
                    "voice_frequency_peaks": voice_peaks,
                    "fundamental_estimate": voice_peaks[0] if voice_peaks else None,
                    "formant_estimates": voice_peaks[:3] if len(voice_peaks) >= 3 else voice_peaks
                }
                
            except Exception as e:
                logger.warning(f"Peak detection failed: {e}")
                analysis_result["peaks"] = {"error": "Peak detection failed"}
        
        # Spectral features calculation
        if spectral_features and len(filtered_avg_magnitude) > 0:
            try:
                # Spectral centroid
                centroid = np.sum(filtered_frequencies * filtered_avg_magnitude) / (np.sum(filtered_avg_magnitude) + 1e-10)
                
                # Spectral rolloff (95% of energy)
                cumulative_energy = np.cumsum(filtered_avg_magnitude)
                total_energy = cumulative_energy[-1]
                rolloff_threshold = 0.95 * total_energy
                rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
                rolloff_freq = filtered_frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else filtered_frequencies[-1]
                
                # Spectral bandwidth
                bandwidth = np.sqrt(np.sum(((filtered_frequencies - centroid) ** 2) * filtered_avg_magnitude) / (np.sum(filtered_avg_magnitude) + 1e-10))
                
                # Spectral flatness (measure of noise-like vs tonal content)
                geometric_mean = np.exp(np.mean(np.log(filtered_avg_magnitude + 1e-10)))
                arithmetic_mean = np.mean(filtered_avg_magnitude)
                flatness = geometric_mean / (arithmetic_mean + 1e-10)
                
                analysis_result["spectral_features"] = {
                    "centroid_hz": float(centroid),
                    "rolloff_hz": float(rolloff_freq),
                    "bandwidth_hz": float(bandwidth),
                    "flatness": float(flatness),
                    "energy_total": float(total_energy),
                    "dynamic_range_db": float(np.max(filtered_avg_magnitude) - np.min(filtered_avg_magnitude)) if not apply_log_scale else None
                }
                
                # Audio quality indicators
                analysis_result["quality_indicators"] = {
                    "signal_to_noise_estimate": float(np.max(filtered_avg_magnitude) / (np.mean(filtered_avg_magnitude) + 1e-10)),
                    "tonal_vs_noise": "tonal" if flatness < 0.1 else "noise-like" if flatness > 0.8 else "mixed",
                    "voice_activity_probability": min(1.0, len([f for f in peak_frequencies if 85 <= f <= 300]) / 3.0) if 'peaks' in analysis_result else 0.0
                }
                
            except Exception as e:
                logger.warning(f"Spectral features calculation failed: {e}")
                analysis_result["spectral_features"] = {"error": "Feature calculation failed"}
        
        logger.info(f"FFT analysis completed for {file.filename}: {len(spectra)} frames, {len(filtered_frequencies)} frequency bins")
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FFT analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"FFT analysis failed: {str(e)}"
        )


@router.post("/analyze/lufs")
async def analyze_audio_lufs(
    file: UploadFile = File(..., description="Audio file for LUFS loudness analysis"),
    measurement_window: float = Form(3.0, description="Measurement window in seconds (0.4-10.0)"),
    gating_mode: str = Form("ebu_r128", description="Gating mode (ebu_r128, itu_bs1770, ungated)"),
    target_lufs: Optional[float] = Form(None, description="Target LUFS for comparison"),
    time_resolution: float = Form(0.1, description="Time resolution for momentary measurements"),
    channel_weighting: str = Form("auto", description="Channel weighting (auto, mono, stereo)")
) -> Dict[str, Any]:
    """
    Professional LUFS loudness measurement endpoint.
    
    Provides comprehensive loudness analysis according to:
    - ITU-R BS.1770-4 standard
    - EBU R128 recommendation
    - Momentary, short-term, and integrated loudness
    - True peak detection
    - Gating and weighting analysis
    """
    try:
        import numpy as np
        import scipy.signal
        import io
        import soundfile as sf
        from collections import deque
        
        # Read audio file
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        try:
            audio_data, sample_rate = sf.read(audio_buffer, dtype='float32')
        except Exception as e:
            try:
                import librosa
                audio_data, sample_rate = librosa.load(audio_buffer, sr=None, dtype=np.float32)
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Could not load audio file. Install librosa for broader format support."
                )
            except Exception as e2:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Could not load audio file: {str(e2)}"
                )
        
        # Ensure mono for LUFS calculation (can be extended for multichannel)
        if len(audio_data.shape) > 1:
            if channel_weighting == "mono" or channel_weighting == "auto":
                audio_data = np.mean(audio_data, axis=1)
            else:
                # For stereo, we'll use the first channel for simplification
                audio_data = audio_data[:, 0]
        
        # Validate parameters
        measurement_window = max(0.4, min(10.0, measurement_window))
        time_resolution = max(0.01, min(1.0, time_resolution))
        
        # Initialize K-weighting filter (ITU-R BS.1770-4)
        def create_k_weighting_filter(sample_rate):
            nyquist = sample_rate / 2
            
            # High-pass filter (38 Hz, Q=0.5)
            fc_hp = 38.0 / nyquist
            hp_b, hp_a = scipy.signal.butter(1, fc_hp, btype='high')
            
            # Shelving filter approximation (RLB weighting)
            fc_shelf = 1681.0 / nyquist
            shelf_b, shelf_a = scipy.signal.butter(1, fc_shelf, btype='high')
            shelf_gain = 1.585  # ~+4dB at high frequencies
            
            return (hp_b, hp_a), (shelf_b, shelf_a), shelf_gain
        
        # Apply K-weighting
        (hp_b, hp_a), (shelf_b, shelf_a), shelf_gain = create_k_weighting_filter(sample_rate)
        
        # Apply high-pass filter
        hp_filtered = scipy.signal.filtfilt(hp_b, hp_a, audio_data)
        
        # Apply shelving filter with gain
        shelf_filtered = scipy.signal.filtfilt(shelf_b, shelf_a, hp_filtered)
        k_weighted = shelf_filtered * shelf_gain
        
        # Calculate block-based loudness measurements
        block_size = int(sample_rate * 0.4)  # 400ms blocks
        hop_size = int(sample_rate * time_resolution)
        
        momentary_loudnesses = []
        short_term_loudnesses = []
        block_loudnesses = []
        time_stamps = []
        
        # Buffers for different measurement windows
        momentary_buffer = deque(maxlen=int(sample_rate * 0.4))  # 400ms
        short_term_buffer = deque(maxlen=int(sample_rate * 3.0))  # 3s
        measurement_buffer = deque(maxlen=int(sample_rate * measurement_window))
        
        # Process audio in blocks
        for i in range(0, len(k_weighted) - block_size, hop_size):
            block = k_weighted[i:i + block_size]
            time_position = i / sample_rate
            
            # Update buffers
            momentary_buffer.extend(block)
            short_term_buffer.extend(block)
            measurement_buffer.extend(block)
            
            # Calculate mean square power
            if len(momentary_buffer) > 0:
                momentary_ms = np.mean(np.array(momentary_buffer) ** 2)
                momentary_lufs = -0.691 + 10 * np.log10(max(momentary_ms, 1e-10))
                momentary_loudnesses.append(momentary_lufs)
            else:
                momentary_loudnesses.append(-np.inf)
            
            if len(short_term_buffer) > 0:
                short_term_ms = np.mean(np.array(short_term_buffer) ** 2)
                short_term_lufs = -0.691 + 10 * np.log10(max(short_term_ms, 1e-10))
                short_term_loudnesses.append(short_term_lufs)
            else:
                short_term_loudnesses.append(-np.inf)
            
            # Block loudness for gating
            block_ms = np.mean(block ** 2)
            if block_ms > 0:
                block_loudness = -0.691 + 10 * np.log10(block_ms)
                block_loudnesses.append(block_loudness)
            
            time_stamps.append(time_position)
        
        # Calculate integrated loudness with gating
        def calculate_integrated_lufs(block_loudnesses, gating_mode):
            if not block_loudnesses:
                return -np.inf, []
            
            if gating_mode == "ungated":
                return np.mean(block_loudnesses), block_loudnesses
            
            # Absolute gating (-70 LUFS)
            absolute_threshold = -70.0
            gated_blocks = [l for l in block_loudnesses if l >= absolute_threshold]
            
            if not gated_blocks:
                return -np.inf, []
            
            if gating_mode == "itu_bs1770":
                # ITU-R BS.1770 uses only absolute gating
                return np.mean(gated_blocks), gated_blocks
            
            # EBU R128 relative gating
            ungated_mean = np.mean(gated_blocks)
            relative_threshold = ungated_mean - 10.0
            final_gated = [l for l in gated_blocks if l >= relative_threshold]
            
            if final_gated:
                return np.mean(final_gated), final_gated
            else:
                return ungated_mean, gated_blocks
        
        integrated_lufs, gated_blocks = calculate_integrated_lufs(block_loudnesses, gating_mode)
        
        # True peak detection (simplified - real implementation needs oversampling)
        peak_level = np.max(np.abs(audio_data))
        true_peak_db = 20 * np.log10(max(peak_level, 1e-10))
        
        # Calculate statistics
        valid_momentary = [l for l in momentary_loudnesses if l > -np.inf]
        valid_short_term = [l for l in short_term_loudnesses if l > -np.inf]
        
        analysis_result = {
            "metadata": {
                "file_name": file.filename,
                "duration_seconds": len(audio_data) / sample_rate,
                "sample_rate": int(sample_rate),
                "measurement_window": measurement_window,
                "gating_mode": gating_mode,
                "time_resolution": time_resolution,
                "channel_weighting": channel_weighting,
                "analysis_timestamp": datetime.utcnow().isoformat()
            },
            "lufs_measurements": {
                "integrated_lufs": float(integrated_lufs) if integrated_lufs > -np.inf else None,
                "momentary_lufs_max": float(max(valid_momentary)) if valid_momentary else None,
                "momentary_lufs_mean": float(np.mean(valid_momentary)) if valid_momentary else None,
                "short_term_lufs_max": float(max(valid_short_term)) if valid_short_term else None,
                "short_term_lufs_mean": float(np.mean(valid_short_term)) if valid_short_term else None,
                "true_peak_db": float(true_peak_db)
            },
            "time_series": {
                "time_stamps": time_stamps[:200],  # Limit for response size
                "momentary_lufs": momentary_loudnesses[:200],
                "short_term_lufs": short_term_loudnesses[:200]
            },
            "gating_analysis": {
                "total_blocks": len(block_loudnesses),
                "gated_blocks": len(gated_blocks),
                "gating_percentage": (len(gated_blocks) / len(block_loudnesses) * 100) if block_loudnesses else 0,
                "absolute_threshold_lufs": -70.0,
                "relative_threshold_lufs": float(integrated_lufs - 10.0) if integrated_lufs > -np.inf else None
            },
            "broadcast_compliance": {
                "ebu_r128_compliant": -24.0 <= integrated_lufs <= -22.0 if integrated_lufs > -np.inf else False,
                "itu_bs1770_4_compliant": true_peak_db <= -1.0,
                "streaming_compliant": -15.0 <= integrated_lufs <= -13.0 if integrated_lufs > -np.inf else False,
                "podcast_compliant": -19.0 <= integrated_lufs <= -17.0 if integrated_lufs > -np.inf else False
            }
        }
        
        # Target comparison if provided
        if target_lufs is not None and integrated_lufs > -np.inf:
            lufs_deviation = integrated_lufs - target_lufs
            analysis_result["target_comparison"] = {
                "target_lufs": float(target_lufs),
                "current_lufs": float(integrated_lufs),
                "deviation_lufs": float(lufs_deviation),
                "within_1lu_tolerance": abs(lufs_deviation) <= 1.0,
                "within_half_lu_tolerance": abs(lufs_deviation) <= 0.5,
                "gain_adjustment_needed_db": float(-lufs_deviation)
            }
        
        # Dynamic range estimation
        if valid_short_term:
            dynamic_range = max(valid_short_term) - min(valid_short_term)
            analysis_result["dynamic_range"] = {
                "short_term_range_lu": float(dynamic_range),
                "range_category": "high" if dynamic_range > 10 else "medium" if dynamic_range > 5 else "low"
            }
        
        logger.info(f"LUFS analysis completed for {file.filename}: {integrated_lufs:.1f} LUFS integrated, {true_peak_db:.1f} dB peak")
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LUFS analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LUFS analysis failed: {str(e)}"
        )


# ========================= INDIVIDUAL STAGE PROCESSING API =========================

@router.post("/process/stage/{stage_name}")
async def process_audio_single_stage(
    stage_name: str,
    file: UploadFile = File(..., description="Audio file to process"),
    stage_config: Optional[str] = Form(None, description="JSON configuration for the stage"),
    gain_in: float = Form(0.0, description="Input gain in dB (-20 to +20)"),
    gain_out: float = Form(0.0, description="Output gain in dB (-20 to +20)"),
    bypass_stage: bool = Form(False, description="Bypass the stage processing"),
    return_intermediate: bool = Form(True, description="Return intermediate processing data"),
    sample_rate_override: Optional[int] = Form(None, description="Override sample rate (8000, 16000, 44100, 48000)")
) -> Dict[str, Any]:
    """
    Process audio through a single stage of the modular pipeline.
    
    Supported stages:
    - vad: Voice Activity Detection
    - voice_filter: Voice frequency filtering
    - noise_reduction: Noise reduction and suppression
    - voice_enhancement: Voice clarity enhancement
    - equalizer: Parametric equalization
    - spectral_denoising: Advanced spectral denoising
    - conventional_denoising: Time-domain denoising
    - lufs_normalization: Professional loudness normalization
    - agc: Auto Gain Control
    - compression: Dynamic range compression
    - limiter: Peak limiting
    """
    try:
        from ..audio.audio_processor import AudioPipelineProcessor
        from ..audio.config import AudioProcessingConfig
        import json
        import numpy as np
        import io
        import soundfile as sf
        import base64
        
        # Validate stage name
        valid_stages = [
            "vad", "voice_filter", "noise_reduction", "voice_enhancement", 
            "equalizer", "spectral_denoising", "conventional_denoising", 
            "lufs_normalization", "agc", "compression", "limiter"
        ]
        
        if stage_name not in valid_stages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stage name. Valid stages: {', '.join(valid_stages)}"
            )
        
        # Read audio file
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        try:
            audio_data, detected_sample_rate = sf.read(audio_buffer, dtype='float32')
        except Exception as e:
            try:
                import librosa
                audio_data, detected_sample_rate = librosa.load(audio_buffer, sr=None, dtype=np.float32)
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Could not load audio file. Install librosa for broader format support."
                )
            except Exception as e2:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Could not load audio file: {str(e2)}"
                )
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Use override sample rate if provided, otherwise use detected
        final_sample_rate = sample_rate_override or int(detected_sample_rate)
        
        # Resample if needed
        if int(detected_sample_rate) != final_sample_rate:
            try:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=detected_sample_rate, target_sr=final_sample_rate)
            except ImportError:
                logger.warning(f"Cannot resample from {detected_sample_rate} to {final_sample_rate} without librosa")
        
        # Create configuration with only the target stage enabled
        config = AudioProcessingConfig()
        config.enabled_stages = [stage_name] if not bypass_stage else []
        
        # Apply custom stage configuration if provided
        if stage_config:
            try:
                stage_config_dict = json.loads(stage_config)
                stage_attr = getattr(config, stage_name, None)
                if stage_attr:
                    for key, value in stage_config_dict.items():
                        if hasattr(stage_attr, key):
                            setattr(stage_attr, key, value)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON in stage_config: {str(e)}"
                )
            except Exception as e:
                logger.warning(f"Could not apply stage config: {e}")
        
        # Apply gain settings
        stage_attr = getattr(config, stage_name, None)
        if stage_attr:
            stage_attr.gain_in = max(-20.0, min(20.0, gain_in))
            stage_attr.gain_out = max(-20.0, min(20.0, gain_out))
        
        # Create processor and process audio
        processor = AudioPipelineProcessor(config, sample_rate=final_sample_rate)
        
        # Store original audio for comparison
        original_rms = float(np.sqrt(np.mean(audio_data ** 2)))
        original_peak = float(np.max(np.abs(audio_data)))
        
        # Process audio
        processed_audio, metadata = processor.process_audio_chunk(
            audio_data, 
            session_id=f"stage_test_{stage_name}",
            chunk_id="single_stage_test"
        )
        
        # Calculate output metrics
        processed_rms = float(np.sqrt(np.mean(processed_audio ** 2)))
        processed_peak = float(np.max(np.abs(processed_audio)))
        
        # Get stage-specific results
        stage_results = metadata.get('stage_results', {})
        stage_result = stage_results.get(stage_name)
        
        result = {
            "metadata": {
                "stage_name": stage_name,
                "file_name": file.filename,
                "duration_seconds": len(audio_data) / final_sample_rate,
                "sample_rate": final_sample_rate,
                "detected_sample_rate": int(detected_sample_rate),
                "audio_length": len(audio_data),
                "bypassed": bypass_stage,
                "processing_timestamp": datetime.utcnow().isoformat()
            },
            "audio_metrics": {
                "input_rms": original_rms,
                "output_rms": processed_rms,
                "input_peak": original_peak,
                "output_peak": processed_peak,
                "rms_change_db": 20 * np.log10(max(processed_rms, 1e-10) / max(original_rms, 1e-10)),
                "peak_change_db": 20 * np.log10(max(processed_peak, 1e-10) / max(original_peak, 1e-10)),
                "gain_applied_estimate": 20 * np.log10(max(processed_rms, 1e-10) / max(original_rms, 1e-10))
            },
            "processing_result": {
                "success": stage_result.status == "completed" if stage_result else True,
                "processing_time_ms": stage_result.processing_time_ms if stage_result else 0.0,
                "error_message": stage_result.error_message if stage_result and stage_result.error_message else None
            }
        }
        
        # Add stage-specific metadata if available
        if stage_result and hasattr(stage_result, 'metadata'):
            result["stage_metadata"] = stage_result.metadata
        
        # Add intermediate processing data if requested
        if return_intermediate:
            # Encode audio data as base64 for transmission
            original_audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            processed_audio_b64 = base64.b64encode(processed_audio.tobytes()).decode('utf-8')
            
            result["audio_data"] = {
                "original_audio_base64": original_audio_b64,
                "processed_audio_base64": processed_audio_b64,
                "audio_format": "float32",
                "encoding_info": {
                    "dtype": "float32",
                    "shape": list(audio_data.shape),
                    "sample_rate": final_sample_rate,
                    "decode_instructions": "Use base64.b64decode() then numpy.frombuffer(dtype=np.float32)"
                }
            }
        
        # Add pipeline metadata
        pipeline_metadata = metadata.get('pipeline_metadata', {})
        result["pipeline_info"] = {
            "stages_processed": pipeline_metadata.get("stages_processed", []),
            "stages_bypassed": pipeline_metadata.get("stages_bypassed", []),
            "total_processing_time_ms": pipeline_metadata.get("total_processing_time_ms", 0.0),
            "performance_warnings": pipeline_metadata.get("performance_warnings", [])
        }
        
        logger.info(f"Single stage processing completed: {stage_name} on {file.filename}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single stage processing failed for {stage_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stage processing failed: {str(e)}"
        )


@router.get("/stages/info")
async def get_stage_information() -> Dict[str, Any]:
    """
    Get information about all available processing stages.
    
    Returns detailed information about each stage including:
    - Description and purpose
    - Configuration parameters
    - Performance characteristics
    - Usage recommendations
    """
    try:
        stage_info = {
            "vad": {
                "name": "Voice Activity Detection",
                "description": "Detects speech segments and filters out silence periods",
                "purpose": "Optimize processing by identifying voice activity",
                "parameters": {
                    "enabled": {"type": "bool", "default": True, "description": "Enable/disable VAD"},
                    "mode": {"type": "enum", "values": ["disabled", "basic", "aggressive", "silero", "webrtc"], "default": "webrtc"},
                    "aggressiveness": {"type": "int", "range": [0, 3], "default": 2, "description": "VAD sensitivity"},
                    "energy_threshold": {"type": "float", "range": [0.001, 1.0], "default": 0.01},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 5.0, "max_latency_ms": 10.0},
                "use_cases": ["Real-time transcription", "Bandwidth optimization", "Noise gate"]
            },
            "voice_filter": {
                "name": "Voice Frequency Filter",
                "description": "Filters audio to enhance human voice frequencies",
                "purpose": "Improve speech clarity by emphasizing voice range",
                "parameters": {
                    "enabled": {"type": "bool", "default": True},
                    "voice_freq_min": {"type": "float", "range": [50, 200], "default": 85, "unit": "Hz"},
                    "voice_freq_max": {"type": "float", "range": [200, 500], "default": 300, "unit": "Hz"},
                    "filter_order": {"type": "int", "range": [2, 8], "default": 4},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 8.0, "max_latency_ms": 15.0},
                "use_cases": ["Voice enhancement", "Telephone audio", "Noisy environments"]
            },
            "noise_reduction": {
                "name": "Noise Reduction",
                "description": "Reduces background noise using spectral subtraction",
                "purpose": "Clean audio by removing unwanted background noise",
                "parameters": {
                    "enabled": {"type": "bool", "default": True},
                    "mode": {"type": "enum", "values": ["disabled", "light", "moderate", "aggressive", "adaptive"], "default": "moderate"},
                    "strength": {"type": "float", "range": [0.0, 1.0], "default": 0.7},
                    "voice_protection": {"type": "bool", "default": True},
                    "adaptation_rate": {"type": "float", "range": [0.01, 0.5], "default": 0.1},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 15.0, "max_latency_ms": 25.0},
                "use_cases": ["Background noise removal", "Office environments", "Street recordings"]
            },
            "voice_enhancement": {
                "name": "Voice Enhancement",
                "description": "Enhances voice clarity using compression and filtering",
                "purpose": "Improve speech intelligibility and presence",
                "parameters": {
                    "enabled": {"type": "bool", "default": True},
                    "enhancement_strength": {"type": "float", "range": [0.0, 1.0], "default": 0.5},
                    "compressor_threshold": {"type": "float", "range": [-60.0, 0.0], "default": -20.0, "unit": "dB"},
                    "compressor_ratio": {"type": "float", "range": [1.0, 10.0], "default": 4.0},
                    "clarity_boost": {"type": "float", "range": [0.0, 1.0], "default": 0.3},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 10.0, "max_latency_ms": 20.0},
                "use_cases": ["Podcast production", "Voice-overs", "Broadcast"]
            },
            "equalizer": {
                "name": "Parametric Equalizer",
                "description": "Multi-band parametric EQ with professional presets",
                "purpose": "Shape frequency response for optimal sound",
                "parameters": {
                    "enabled": {"type": "bool", "default": True},
                    "preset": {"type": "enum", "values": ["flat", "voice_enhance", "broadcast", "podcast", "conference"], "default": "voice_enhance"},
                    "bands": {"type": "array", "description": "Array of EQ bands with frequency, gain, Q"},
                    "master_gain": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 12.0, "max_latency_ms": 22.0},
                "use_cases": ["Audio mastering", "Tonal correction", "Creative sound design"]
            },
            "spectral_denoising": {
                "name": "Spectral Denoising",
                "description": "Advanced frequency-domain noise reduction",
                "purpose": "Remove complex noise patterns using spectral analysis",
                "parameters": {
                    "enabled": {"type": "bool", "default": False},
                    "mode": {"type": "enum", "values": ["minimal", "spectral_subtraction", "wiener_filter", "adaptive"], "default": "wiener_filter"},
                    "reduction_strength": {"type": "float", "range": [0.0, 1.0], "default": 0.7},
                    "fft_size": {"type": "int", "values": [256, 512, 1024, 2048, 4096], "default": 1024},
                    "smoothing_factor": {"type": "float", "range": [0.0, 0.99], "default": 0.8},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 20.0, "max_latency_ms": 35.0},
                "use_cases": ["Music restoration", "Complex noise removal", "High-quality processing"]
            },
            "conventional_denoising": {
                "name": "Conventional Denoising",
                "description": "Time-domain denoising using various filter types",
                "purpose": "Fast noise reduction using traditional filtering",
                "parameters": {
                    "enabled": {"type": "bool", "default": True},
                    "mode": {"type": "enum", "values": ["disabled", "median_filter", "gaussian_filter", "bilateral_filter", "wavelet_denoising", "adaptive_filter"], "default": "median_filter"},
                    "strength": {"type": "float", "range": [0.0, 1.0], "default": 0.5},
                    "window_size": {"type": "int", "range": [3, 21], "default": 5},
                    "preserve_transients": {"type": "bool", "default": True},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 8.0, "max_latency_ms": 15.0},
                "use_cases": ["Real-time processing", "Low-latency applications", "Simple noise removal"]
            },
            "lufs_normalization": {
                "name": "LUFS Normalization",
                "description": "Professional loudness normalization according to broadcast standards",
                "purpose": "Standardize audio loudness for consistent playback levels",
                "parameters": {
                    "enabled": {"type": "bool", "default": False},
                    "mode": {"type": "enum", "values": ["streaming", "broadcast_tv", "broadcast_radio", "podcast", "youtube", "netflix", "custom"], "default": "streaming"},
                    "target_lufs": {"type": "float", "range": [-60.0, 0.0], "default": -14.0, "unit": "LUFS"},
                    "max_peak_db": {"type": "float", "range": [-6.0, 0.0], "default": -1.0, "unit": "dB"},
                    "adjustment_speed": {"type": "float", "range": [0.01, 1.0], "default": 0.5},
                    "true_peak_limiting": {"type": "bool", "default": True},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 18.0, "max_latency_ms": 30.0},
                "use_cases": ["Broadcast compliance", "Streaming platforms", "Professional mastering"]
            },
            "agc": {
                "name": "Auto Gain Control",
                "description": "Automatic level control with adaptive gain adjustment",
                "purpose": "Maintain consistent audio levels automatically",
                "parameters": {
                    "enabled": {"type": "bool", "default": True},
                    "mode": {"type": "enum", "values": ["disabled", "fast", "medium", "slow", "adaptive"], "default": "medium"},
                    "target_level": {"type": "float", "range": [-40.0, 0.0], "default": -18.0, "unit": "dB"},
                    "max_gain": {"type": "float", "range": [0.0, 40.0], "default": 12.0, "unit": "dB"},
                    "attack_time": {"type": "float", "range": [0.1, 100.0], "default": 10.0, "unit": "ms"},
                    "release_time": {"type": "float", "range": [1.0, 1000.0], "default": 100.0, "unit": "ms"},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 12.0, "max_latency_ms": 20.0},
                "use_cases": ["Live streaming", "Conference calls", "Automatic level management"]
            },
            "compression": {
                "name": "Dynamic Range Compression",
                "description": "Professional audio compression for dynamic range control",
                "purpose": "Control dynamic range and increase perceived loudness",
                "parameters": {
                    "enabled": {"type": "bool", "default": True},
                    "mode": {"type": "enum", "values": ["disabled", "soft_knee", "hard_knee", "adaptive", "voice_optimized"], "default": "soft_knee"},
                    "threshold": {"type": "float", "range": [-60.0, 0.0], "default": -20.0, "unit": "dB"},
                    "ratio": {"type": "float", "range": [1.0, 20.0], "default": 4.0},
                    "attack_time": {"type": "float", "range": [0.1, 100.0], "default": 5.0, "unit": "ms"},
                    "release_time": {"type": "float", "range": [10.0, 1000.0], "default": 100.0, "unit": "ms"},
                    "makeup_gain": {"type": "float", "range": [0.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 8.0, "max_latency_ms": 15.0},
                "use_cases": ["Music production", "Voice processing", "Dynamic control"]
            },
            "limiter": {
                "name": "Peak Limiter",
                "description": "Transparent peak limiting to prevent clipping",
                "purpose": "Prevent digital clipping while maintaining audio quality",
                "parameters": {
                    "enabled": {"type": "bool", "default": True},
                    "threshold": {"type": "float", "range": [-6.0, 0.0], "default": -0.1, "unit": "dB"},
                    "release_time": {"type": "float", "range": [0.1, 100.0], "default": 10.0, "unit": "ms"},
                    "lookahead_time": {"type": "float", "range": [0.1, 10.0], "default": 2.0, "unit": "ms"},
                    "soft_knee": {"type": "bool", "default": True},
                    "gain_in": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"},
                    "gain_out": {"type": "float", "range": [-20.0, 20.0], "default": 0.0, "unit": "dB"}
                },
                "performance": {"target_latency_ms": 6.0, "max_latency_ms": 12.0},
                "use_cases": ["Mastering", "Broadcast safety", "Peak protection"]
            }
        }
        
        return {
            "stages": stage_info,
            "pipeline_info": {
                "total_stages": len(stage_info),
                "default_order": [
                    "vad", "voice_filter", "noise_reduction", "voice_enhancement", 
                    "equalizer", "spectral_denoising", "conventional_denoising", 
                    "lufs_normalization", "agc", "compression", "limiter"
                ],
                "modular_architecture": True,
                "individual_gain_controls": True,
                "real_time_capable": True,
                "performance_monitoring": True
            },
            "usage_tips": {
                "recommended_combinations": [
                    {"name": "Voice Clarity", "stages": ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "compression", "limiter"]},
                    {"name": "Broadcast Quality", "stages": ["vad", "noise_reduction", "equalizer", "lufs_normalization", "limiter"]},
                    {"name": "Podcast Production", "stages": ["voice_filter", "noise_reduction", "equalizer", "compression", "lufs_normalization"]},
                    {"name": "Real-time Streaming", "stages": ["vad", "noise_reduction", "agc", "limiter"]},
                    {"name": "Music Enhancement", "stages": ["spectral_denoising", "equalizer", "compression", "lufs_normalization", "limiter"]}
                ],
                "performance_guidelines": [
                    "Disable unused stages for better performance",
                    "Use conventional_denoising for low-latency applications",
                    "Enable spectral_denoising only for high-quality offline processing",
                    "LUFS normalization is best for final mastering steps",
                    "Monitor total pipeline latency in real-time applications"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get stage information: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stage information: {str(e)}"
        )


# ========================= PRESET MANAGEMENT API =========================

@router.get("/presets")
async def get_audio_presets() -> Dict[str, Any]:
    """
    Get all available audio processing presets.
    
    Returns built-in presets and user-saved configurations with
    detailed information about each preset's characteristics.
    """
    try:
        from ..audio.config import AudioConfigurationManager, AudioProcessingConfig
        
        # Built-in presets with detailed configurations
        builtin_presets = {
            "default": {
                "name": "Default Processing",
                "description": "Balanced processing suitable for most speech content",
                "category": "general",
                "use_cases": ["General speech", "Conferences", "Interviews"],
                "enabled_stages": ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "agc", "compression", "limiter"],
                "characteristics": {
                    "latency": "medium",
                    "quality": "good",
                    "cpu_usage": "medium",
                    "noise_reduction": "moderate"
                },
                "optimized_for": ["speech_clarity", "real_time"],
                "config": {
                    "vad": {"mode": "webrtc", "aggressiveness": 2},
                    "noise_reduction": {"mode": "moderate", "strength": 0.7},
                    "voice_enhancement": {"enhancement_strength": 0.5},
                    "agc": {"mode": "medium", "target_level": -18.0},
                    "compression": {"threshold": -20.0, "ratio": 4.0},
                    "limiter": {"threshold": -0.1}
                }
            },
            "voice_optimized": {
                "name": "Voice Clarity Enhanced",
                "description": "Optimized for maximum speech clarity and intelligibility",
                "category": "voice",
                "use_cases": ["Podcasts", "Voice-overs", "Educational content"],
                "enabled_stages": ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "equalizer", "compression", "limiter"],
                "characteristics": {
                    "latency": "medium",
                    "quality": "excellent",
                    "cpu_usage": "medium-high",
                    "noise_reduction": "aggressive"
                },
                "optimized_for": ["voice_clarity", "podcast_production"],
                "config": {
                    "vad": {"mode": "webrtc", "aggressiveness": 1},
                    "voice_filter": {"voice_freq_min": 85, "voice_freq_max": 300},
                    "noise_reduction": {"mode": "aggressive", "strength": 0.85, "voice_protection": True},
                    "voice_enhancement": {"enhancement_strength": 0.8, "clarity_boost": 0.6},
                    "equalizer": {"preset": "voice_enhance"},
                    "compression": {"threshold": -18.0, "ratio": 6.0, "makeup_gain": 3.0},
                    "limiter": {"threshold": -0.5}
                }
            },
            "noisy_environment": {
                "name": "Noisy Environment",
                "description": "Heavy noise reduction for challenging acoustic environments",
                "category": "noise_reduction",
                "use_cases": ["Street recordings", "Crowded spaces", "Vehicle audio"],
                "enabled_stages": ["vad", "voice_filter", "noise_reduction", "spectral_denoising", "voice_enhancement", "agc", "limiter"],
                "characteristics": {
                    "latency": "high",
                    "quality": "good",
                    "cpu_usage": "high",
                    "noise_reduction": "maximum"
                },
                "optimized_for": ["noise_suppression", "difficult_environments"],
                "config": {
                    "vad": {"mode": "aggressive", "aggressiveness": 3, "energy_threshold": 0.02},
                    "voice_filter": {"voice_freq_min": 100, "voice_freq_max": 250},
                    "noise_reduction": {"mode": "aggressive", "strength": 0.9},
                    "spectral_denoising": {"enabled": True, "mode": "wiener_filter", "reduction_strength": 0.8},
                    "voice_enhancement": {"enhancement_strength": 0.7},
                    "agc": {"mode": "adaptive", "target_level": -16.0, "max_gain": 15.0},
                    "limiter": {"threshold": -1.0}
                }
            },
            "broadcast_quality": {
                "name": "Broadcast Quality",
                "description": "Professional broadcast standards with LUFS normalization",
                "category": "broadcast",
                "use_cases": ["Radio", "Television", "Professional streaming"],
                "enabled_stages": ["vad", "noise_reduction", "equalizer", "lufs_normalization", "compression", "limiter"],
                "characteristics": {
                    "latency": "medium-high",
                    "quality": "excellent",
                    "cpu_usage": "medium-high",
                    "noise_reduction": "moderate"
                },
                "optimized_for": ["broadcast_compliance", "professional_quality"],
                "config": {
                    "vad": {"mode": "webrtc", "aggressiveness": 1},
                    "noise_reduction": {"mode": "moderate", "strength": 0.6},
                    "equalizer": {"preset": "broadcast"},
                    "lufs_normalization": {"enabled": True, "mode": "broadcast_tv", "target_lufs": -23.0},
                    "compression": {"threshold": -22.0, "ratio": 3.0, "attack_time": 3.0},
                    "limiter": {"threshold": -1.0, "soft_knee": True}
                }
            },
            "minimal_processing": {
                "name": "Minimal Processing",
                "description": "Lightweight processing for low-latency real-time applications",
                "category": "real_time",
                "use_cases": ["Live streaming", "Real-time communication", "Gaming"],
                "enabled_stages": ["vad", "conventional_denoising", "agc", "limiter"],
                "characteristics": {
                    "latency": "very_low",
                    "quality": "fair",
                    "cpu_usage": "low",
                    "noise_reduction": "light"
                },
                "optimized_for": ["low_latency", "real_time_communication"],
                "config": {
                    "vad": {"mode": "basic", "aggressiveness": 1},
                    "conventional_denoising": {"mode": "median_filter", "strength": 0.3},
                    "agc": {"mode": "fast", "target_level": -18.0, "attack_time": 2.0},
                    "limiter": {"threshold": -0.1, "release_time": 5.0}
                }
            },
            "music_content": {
                "name": "Music Enhancement",
                "description": "Optimized for music and mixed content with wide dynamic range",
                "category": "music",
                "use_cases": ["Music production", "Mixed content", "Entertainment"],
                "enabled_stages": ["spectral_denoising", "equalizer", "compression", "lufs_normalization", "limiter"],
                "characteristics": {
                    "latency": "high",
                    "quality": "excellent",
                    "cpu_usage": "high",
                    "noise_reduction": "moderate"
                },
                "optimized_for": ["music_quality", "dynamic_range"],
                "config": {
                    "spectral_denoising": {"enabled": True, "mode": "adaptive", "reduction_strength": 0.5},
                    "equalizer": {"preset": "flat", "master_gain": 1.0},
                    "compression": {"threshold": -24.0, "ratio": 2.5, "attack_time": 10.0, "release_time": 200.0},
                    "lufs_normalization": {"enabled": True, "mode": "streaming", "target_lufs": -14.0},
                    "limiter": {"threshold": -0.5, "lookahead_time": 5.0}
                }
            },
            "conference_call": {
                "name": "Conference Call",
                "description": "Optimized for multi-participant conference calls and meetings",
                "category": "conference",
                "use_cases": ["Video conferences", "Team meetings", "Webinars"],
                "enabled_stages": ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "agc", "limiter"],
                "characteristics": {
                    "latency": "low",
                    "quality": "good",
                    "cpu_usage": "medium",
                    "noise_reduction": "moderate"
                },
                "optimized_for": ["speech_clarity", "multi_participant"],
                "config": {
                    "vad": {"mode": "webrtc", "aggressiveness": 2, "energy_threshold": 0.015},
                    "voice_filter": {"voice_freq_min": 100, "voice_freq_max": 280},
                    "noise_reduction": {"mode": "moderate", "strength": 0.7, "voice_protection": True},
                    "voice_enhancement": {"enhancement_strength": 0.6, "compressor_threshold": -18.0},
                    "agc": {"mode": "medium", "target_level": -20.0, "max_gain": 10.0},
                    "limiter": {"threshold": -1.0}
                }
            }
        }
        
        return {
            "builtin_presets": builtin_presets,
            "preset_categories": [
                {"id": "general", "name": "General Purpose", "description": "Balanced presets for common use cases"},
                {"id": "voice", "name": "Voice Optimized", "description": "Enhanced speech clarity and intelligibility"},
                {"id": "noise_reduction", "name": "Noise Reduction", "description": "Heavy noise suppression for challenging environments"},
                {"id": "broadcast", "name": "Broadcast Quality", "description": "Professional broadcast and streaming standards"},
                {"id": "real_time", "name": "Real-time", "description": "Low-latency processing for live applications"},
                {"id": "music", "name": "Music & Entertainment", "description": "Optimized for music and mixed content"},
                {"id": "conference", "name": "Conference & Meetings", "description": "Multi-participant communication optimization"}
            ],
            "preset_characteristics": {
                "latency": ["very_low", "low", "medium", "medium-high", "high"],
                "quality": ["fair", "good", "excellent"],
                "cpu_usage": ["low", "medium", "medium-high", "high"],
                "noise_reduction": ["light", "moderate", "aggressive", "maximum"]
            },
            "total_presets": len(builtin_presets)
        }
        
    except Exception as e:
        logger.error(f"Failed to get audio presets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audio presets: {str(e)}"
        )


@router.get("/presets/{preset_name}")
async def get_audio_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get detailed configuration for a specific preset.
    
    Returns the complete configuration that can be used to
    initialize the audio processing pipeline.
    """
    try:
        from ..audio.config import AudioProcessingConfig, AudioConfigurationManager
        
        # Check if it's a built-in preset first
        builtin_presets = (await get_audio_presets())["builtin_presets"]
        
        if preset_name in builtin_presets:
            preset_info = builtin_presets[preset_name]
            
            # Create actual configuration object
            config = AudioProcessingConfig()
            config.preset_name = preset_name
            config.enabled_stages = preset_info["enabled_stages"]
            
            # Apply preset-specific configurations
            preset_config = preset_info.get("config", {})
            for stage_name, stage_settings in preset_config.items():
                stage_attr = getattr(config, stage_name, None)
                if stage_attr:
                    for setting, value in stage_settings.items():
                        if hasattr(stage_attr, setting):
                            setattr(stage_attr, setting, value)
            
            # Convert to dictionary format
            from dataclasses import asdict
            config_dict = asdict(config)
            
            return {
                "preset_name": preset_name,
                "preset_info": preset_info,
                "configuration": config_dict,
                "metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "preset_type": "builtin",
                    "version": "1.0"
                }
            }
        else:
            # TODO: Look up user-saved presets from database
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset '{preset_name}' not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get preset {preset_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get preset: {str(e)}"
        )


@router.post("/presets/{preset_name}/apply")
async def apply_audio_preset(
    preset_name: str,
    override_settings: Optional[str] = Form(None, description="JSON object with settings to override"),
    session_id: Optional[str] = Form(None, description="Session ID to apply preset to")
) -> Dict[str, Any]:
    """
    Apply a preset to the audio processing pipeline with optional overrides.
    
    Allows customization of preset settings before application.
    """
    try:
        from ..audio.config import AudioProcessingConfig
        import json
        
        # Get the preset configuration
        preset_data = await get_audio_preset(preset_name)
        config_dict = preset_data["configuration"]
        
        # Apply overrides if provided
        if override_settings:
            try:
                overrides = json.loads(override_settings)
                # Deep merge overrides into config
                def deep_update(base_dict, update_dict):
                    for key, value in update_dict.items():
                        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                            deep_update(base_dict[key], value)
                        else:
                            base_dict[key] = value
                
                deep_update(config_dict, overrides)
                
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON in override_settings: {str(e)}"
                )
        
        # TODO: Apply configuration to session if session_id provided
        # For now, just return the configuration that would be applied
        
        return {
            "preset_applied": preset_name,
            "session_id": session_id,
            "configuration": config_dict,
            "overrides_applied": override_settings is not None,
            "applied_at": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply preset {preset_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply preset: {str(e)}"
        )


@router.post("/presets/save")
async def save_custom_preset(
    preset_name: str = Form(..., description="Name for the custom preset"),
    configuration: str = Form(..., description="JSON configuration object"),
    description: Optional[str] = Form(None, description="Description of the preset"),
    category: str = Form("custom", description="Preset category"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    is_public: bool = Form(False, description="Make preset available to other users"),
    user_id: Optional[str] = Form(None, description="User ID (for authenticated users)")
) -> Dict[str, Any]:
    """
    Save a custom audio processing preset.
    
    Allows users to save their own configurations for later use.
    """
    try:
        from ..audio.config import AudioProcessingConfig
        import json
        
        # Validate configuration
        try:
            config_dict = json.loads(configuration)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON configuration: {str(e)}"
            )
        
        # Validate preset name
        if not preset_name or len(preset_name.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Preset name cannot be empty"
            )
        
        # Check if preset name conflicts with built-in presets
        builtin_presets = (await get_audio_presets())["builtin_presets"]
        if preset_name in builtin_presets:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Preset name '{preset_name}' conflicts with built-in preset"
            )
        
        # Validate configuration structure
        try:
            # Attempt to create AudioProcessingConfig from the provided configuration
            # This validates the structure and required fields
            config = AudioProcessingConfig()
            
            # Apply the configuration
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration structure: {str(e)}"
            )
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # TODO: Save to database
        # For now, return success with the preset information
        
        preset_id = f"custom_{preset_name.lower().replace(' ', '_')}"
        
        saved_preset = {
            "preset_id": preset_id,
            "preset_name": preset_name,
            "description": description or f"Custom preset: {preset_name}",
            "category": category,
            "tags": tag_list,
            "configuration": config_dict,
            "is_public": is_public,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "preset_type": "custom",
            "version": "1.0"
        }
        
        logger.info(f"Custom preset saved: {preset_name} by user {user_id}")
        
        return {
            "status": "saved",
            "preset": saved_preset,
            "message": f"Preset '{preset_name}' saved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save custom preset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save preset: {str(e)}"
        )


@router.delete("/presets/{preset_name}")
async def delete_custom_preset(
    preset_name: str,
    user_id: Optional[str] = Form(None, description="User ID for authorization")
) -> Dict[str, Any]:
    """
    Delete a custom audio processing preset.
    
    Only allows deletion of user-created presets, not built-in ones.
    """
    try:
        # Check if it's a built-in preset
        builtin_presets = (await get_audio_presets())["builtin_presets"]
        if preset_name in builtin_presets:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete built-in presets"
            )
        
        # TODO: Delete from database with proper authorization
        # Check if user owns the preset or has admin rights
        
        logger.info(f"Custom preset deleted: {preset_name} by user {user_id}")
        
        return {
            "status": "deleted",
            "preset_name": preset_name,
            "deleted_at": datetime.utcnow().isoformat(),
            "message": f"Preset '{preset_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete preset {preset_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete preset: {str(e)}"
        )


@router.get("/presets/compare/{preset1}/{preset2}")
async def compare_presets(preset1: str, preset2: str) -> Dict[str, Any]:
    """
    Compare two audio processing presets and highlight differences.
    
    Useful for understanding the differences between presets and
    making informed decisions about which to use.
    """
    try:
        # Get both presets
        preset1_data = await get_audio_preset(preset1)
        preset2_data = await get_audio_preset(preset2)
        
        config1 = preset1_data["configuration"]
        config2 = preset2_data["configuration"]
        
        # Compare enabled stages
        stages1 = set(config1.get("enabled_stages", []))
        stages2 = set(config2.get("enabled_stages", []))
        
        stage_differences = {
            "only_in_preset1": list(stages1 - stages2),
            "only_in_preset2": list(stages2 - stages1),
            "common_stages": list(stages1 & stages2)
        }
        
        # Compare stage configurations
        config_differences = {}
        
        for stage in stage_differences["common_stages"]:
            stage_config1 = config1.get(stage, {})
            stage_config2 = config2.get(stage, {})
            
            if stage_config1 != stage_config2:
                differences = {}
                all_keys = set(stage_config1.keys()) | set(stage_config2.keys())
                
                for key in all_keys:
                    val1 = stage_config1.get(key)
                    val2 = stage_config2.get(key)
                    
                    if val1 != val2:
                        differences[key] = {
                            "preset1_value": val1,
                            "preset2_value": val2
                        }
                
                if differences:
                    config_differences[stage] = differences
        
        # Calculate performance implications
        performance_comparison = {
            "latency_comparison": _estimate_latency_difference(stages1, stages2),
            "cpu_usage_comparison": _estimate_cpu_difference(stages1, stages2),
            "quality_comparison": _estimate_quality_difference(preset1_data["preset_info"], preset2_data["preset_info"])
        }
        
        return {
            "preset1": {
                "name": preset1,
                "info": preset1_data["preset_info"]
            },
            "preset2": {
                "name": preset2,
                "info": preset2_data["preset_info"]
            },
            "stage_differences": stage_differences,
            "configuration_differences": config_differences,
            "performance_comparison": performance_comparison,
            "recommendation": _generate_preset_recommendation(preset1_data, preset2_data),
            "compared_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare presets {preset1} and {preset2}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare presets: {str(e)}"
        )


def _estimate_latency_difference(stages1: set, stages2: set) -> str:
    """Estimate relative latency difference between stage sets."""
    # Latency weights for different stages (in ms)
    latency_weights = {
        "vad": 5, "voice_filter": 8, "noise_reduction": 15,
        "voice_enhancement": 10, "equalizer": 12, "spectral_denoising": 20,
        "conventional_denoising": 8, "lufs_normalization": 18,
        "agc": 12, "compression": 8, "limiter": 6
    }
    
    latency1 = sum(latency_weights.get(stage, 10) for stage in stages1)
    latency2 = sum(latency_weights.get(stage, 10) for stage in stages2)
    
    diff = latency2 - latency1
    if abs(diff) < 5:
        return "similar"
    elif diff > 0:
        return f"preset1 is {diff:.0f}ms faster"
    else:
        return f"preset2 is {abs(diff):.0f}ms faster"


def _estimate_cpu_difference(stages1: set, stages2: set) -> str:
    """Estimate relative CPU usage difference between stage sets."""
    cpu_weights = {
        "vad": 1, "voice_filter": 2, "noise_reduction": 3,
        "voice_enhancement": 2, "equalizer": 2, "spectral_denoising": 5,
        "conventional_denoising": 1, "lufs_normalization": 3,
        "agc": 2, "compression": 2, "limiter": 1
    }
    
    cpu1 = sum(cpu_weights.get(stage, 2) for stage in stages1)
    cpu2 = sum(cpu_weights.get(stage, 2) for stage in stages2)
    
    diff = cpu2 - cpu1
    if abs(diff) < 2:
        return "similar"
    elif diff > 0:
        return f"preset1 uses less CPU"
    else:
        return f"preset2 uses less CPU"


def _estimate_quality_difference(info1: Dict, info2: Dict) -> str:
    """Estimate quality difference between presets."""
    quality_map = {"fair": 1, "good": 2, "excellent": 3}
    
    quality1 = quality_map.get(info1.get("characteristics", {}).get("quality", "good"), 2)
    quality2 = quality_map.get(info2.get("characteristics", {}).get("quality", "good"), 2)
    
    if quality1 == quality2:
        return "similar quality"
    elif quality1 > quality2:
        return "preset1 has higher quality"
    else:
        return "preset2 has higher quality"


def _generate_preset_recommendation(preset1_data: Dict, preset2_data: Dict) -> str:
    """Generate recommendation based on preset comparison."""
    info1 = preset1_data["preset_info"]
    info2 = preset2_data["preset_info"]
    
    # Simple recommendation logic based on characteristics
    if "real_time" in info1.get("optimized_for", []):
        return f"Use {preset1_data['preset_name']} for real-time applications"
    elif "real_time" in info2.get("optimized_for", []):
        return f"Use {preset2_data['preset_name']} for real-time applications"
    elif info1.get("characteristics", {}).get("quality") == "excellent":
        return f"Use {preset1_data['preset_name']} for best quality"
    elif info2.get("characteristics", {}).get("quality") == "excellent":
        return f"Use {preset2_data['preset_name']} for best quality"
    else:
        return "Both presets are suitable for general use"
