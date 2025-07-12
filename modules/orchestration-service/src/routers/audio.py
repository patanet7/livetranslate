"""
Audio processing API router

Enhanced async/await endpoints for audio processing with the whisper service
"""

import asyncio
import json
import logging
from typing import List, Optional, Dict, Any
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
from models.system import ErrorResponse
from dependencies import (
    get_config_manager,
    get_health_monitor,
    get_audio_service_client,
    get_translation_service_client,
)
from utils.audio_processing import AudioProcessor
from utils.rate_limiting import RateLimiter
from utils.security import SecurityUtils

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize utilities
rate_limiter = RateLimiter()
security_utils = SecurityUtils()


@router.post("/process", response_model=AudioProcessingResponse)
async def process_audio(
    request: AudioProcessingRequest,
    config_manager=Depends(get_config_manager),
    audio_client=Depends(get_audio_service_client),
    # Rate limiting will be handled by middleware
) -> AudioProcessingResponse:
    """
    Process audio with configured pipeline stages

    - **audio_data**: Base64 encoded audio data
    - **audio_url**: URL to audio file
    - **file_upload**: File upload reference
    - **config**: Processing configuration
    - **streaming**: Enable streaming processing
    - **transcription**: Enable transcription
    - **speaker_diarization**: Enable speaker diarization
    """
    request_id = f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    try:
        logger.info(f"[{request_id}] Processing audio request")

        # Validate audio source
        if not any([request.audio_data, request.audio_url, request.file_upload]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No audio source provided",
            )

        # Get audio service configuration
        audio_service_config = await config_manager.get_service_config("audio")

        # Process through enhanced pipeline
        if request.streaming:
            return await _process_audio_streaming(
                request, request_id, audio_client, audio_service_config
            )
        else:
            return await _process_audio_batch(
                request, request_id, audio_client, audio_service_config
            )

    except ValidationError as e:
        logger.error(f"[{request_id}] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid audio processing request: {str(e)}",
        )
    except Exception as e:
        logger.error(f"[{request_id}] Audio processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(e)}",
        )


@router.post("/upload", response_model=Dict[str, Any])
async def upload_audio_file(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    transcription: bool = Form(True),
    speaker_diarization: bool = Form(True),
    target_languages: Optional[str] = Form(None),
    config_manager=Depends(get_config_manager),
    translation_client=Depends(get_translation_service_client),
    # Rate limiting will be handled by middleware
) -> Dict[str, Any]:
    """
    Upload and process audio file with optional translation

    - **file**: Audio file (WAV, MP3, WebM, OGG, MP4, FLAC)
    - **config**: JSON string of AudioConfiguration
    - **session_id**: Session identifier
    - **transcription**: Enable transcription
    - **speaker_diarization**: Enable speaker diarization
    - **target_languages**: JSON array of target language codes for translation (e.g. ["es", "fr", "de"])
    """
    request_id = f"upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    try:
        logger.info(f"[{request_id}] Processing file upload: {file.filename}")

        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided"
            )

        # Check file size (100MB limit)
        file_size = 0
        content = await file.read()
        file_size = len(content)

        if file_size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large (max 100MB)",
            )

        # Validate audio file
        audio_processor = AudioProcessor()
        audio_metadata = audio_processor.validate_audio_file(content, file.filename)

        # Parse configuration
        audio_config = AudioConfiguration()
        if config:
            try:
                import json

                config_dict = json.loads(config)
                audio_config = AudioConfiguration(**config_dict)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"[{request_id}] Invalid config, using defaults: {e}")

        # Create processing request
        processing_request = AudioProcessingRequest(
            audio_data=None,
            file_upload=request_id,
            config=audio_config,
            transcription=transcription,
            speaker_diarization=speaker_diarization,
            session_id=session_id,
            metadata={
                "filename": file.filename,
                "file_size": file_size,
                "content_type": file.content_type,
                "upload_timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Store file temporarily
        temp_file_path = await _store_temp_file(content, file.filename, request_id)

        # Process audio
        audio_client = await get_audio_service_client()
        result = await _process_uploaded_file(
            processing_request, request_id, temp_file_path, audio_client
        )

        # Initialize response
        response = {
            "request_id": request_id,
            "filename": file.filename,
            "file_size": file_size,
            "audio_metadata": audio_metadata,
            "processing_result": result,
            "translations": {}
        }

        # Process translation if requested
        if target_languages and transcription and result:
            try:
                # Parse target languages
                target_langs = json.loads(target_languages) if isinstance(target_languages, str) else target_languages
                if not isinstance(target_langs, list):
                    target_langs = [target_langs]
                
                logger.info(f"[{request_id}] Starting translation to languages: {target_langs}")
                
                # Extract transcription text from result
                transcription_text = None
                source_language = None
                
                if isinstance(result, dict):
                    # Try different possible keys for the transcription text
                    transcription_text = result.get("text") or result.get("transcription") or result.get("transcript")
                    source_language = result.get("language") or result.get("detected_language")
                elif hasattr(result, 'text'):
                    transcription_text = result.text
                    source_language = getattr(result, 'language', None)
                
                if transcription_text and transcription_text.strip():
                    # Use the enhanced translation client method
                    translation_results = await translation_client.translate_to_multiple_languages(
                        text=transcription_text,
                        source_language=source_language,
                        target_languages=target_langs,
                        quality="balanced"
                    )
                    
                    # Convert TranslationResponse objects to dictionaries
                    translations_dict = {}
                    for lang, trans_response in translation_results.items():
                        if hasattr(trans_response, 'dict'):
                            translations_dict[lang] = trans_response.dict()
                        else:
                            # Fallback if not a pydantic model
                            translations_dict[lang] = {
                                "translated_text": getattr(trans_response, 'translated_text', str(trans_response)),
                                "source_language": getattr(trans_response, 'source_language', source_language),
                                "target_language": getattr(trans_response, 'target_language', lang),
                                "confidence": getattr(trans_response, 'confidence', 0.0),
                                "processing_time": getattr(trans_response, 'processing_time', 0.0),
                                "model_used": getattr(trans_response, 'model_used', 'unknown')
                            }
                    
                    response["translations"] = translations_dict
                    logger.info(f"[{request_id}] Translation completed for {len(translations_dict)} languages")
                else:
                    logger.warning(f"[{request_id}] No transcription text found for translation")
                    response["translation_error"] = "No transcription text available for translation"
                    
            except json.JSONDecodeError as e:
                logger.error(f"[{request_id}] Invalid target_languages JSON: {e}")
                response["translation_error"] = f"Invalid target_languages format: {str(e)}"
            except Exception as e:
                logger.error(f"[{request_id}] Translation failed: {e}")
                response["translation_error"] = f"Translation failed: {str(e)}"

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] File upload processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload processing failed: {str(e)}",
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
            models_task, audio_device_task, translation_device_task,
            return_exceptions=True
        )
        
        # Handle models result
        if isinstance(models, Exception):
            logger.error(f"Failed to get models: {models}")
            models = ["base"]  # fallback
            models_status = "fallback"
        else:
            models_status = "success"
        
        # Handle audio device info result
        if isinstance(audio_device_info, Exception):
            logger.error(f"Failed to get audio device info: {audio_device_info}")
            audio_device_info = {"device": "unknown", "status": "error", "error": str(audio_device_info)}
        
        # Handle translation device info result
        if isinstance(translation_device_info, Exception):
            logger.error(f"Failed to get translation device info: {translation_device_info}")
            translation_device_info = {"device": "unknown", "status": "error", "error": str(translation_device_info)}
        
        logger.info(f"Retrieved {len(models)} models from audio service: {models}")
        logger.info(f"Audio service device: {audio_device_info.get('device', 'unknown')}")
        logger.info(f"Translation service device: {translation_device_info.get('device', 'unknown')}")
        
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
                    "error": audio_device_info.get("error")
                },
                "translation_service": {
                    "device": translation_device_info.get("device", "unknown"),
                    "device_type": translation_device_info.get("device_type", "unknown"), 
                    "status": translation_device_info.get("status", "unknown"),
                    "details": translation_device_info.get("details", {}),
                    "acceleration": translation_device_info.get("acceleration", "unknown"),
                    "error": translation_device_info.get("error")
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get models and device info: {e}")
        
        # Return fallback models if service is unavailable
        fallback_models = ["tiny", "base", "small", "medium", "large"]
        
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
                    "error": "Service unavailable"
                },
                "translation_service": {
                    "device": "unknown", 
                    "status": "unavailable",
                    "error": "Service unavailable"
                }
            }
        }


@router.get("/health")
async def audio_health_check(
    health_monitor=Depends(get_health_monitor),
) -> Dict[str, Any]:
    """
    Check audio processing service health

    Returns health status of the audio processing pipeline
    """
    try:
        health_status = await health_monitor.check_service_health("audio")

        return {
            "status": health_status["status"],
            "response_time_ms": health_status.get("response_time_ms"),
            "last_check": health_status.get("last_check"),
            "version": health_status.get("version"),
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
        processing_result = await audio_client.process_audio_batch(
            request.dict(), request_id
        )

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
        stream_result = await audio_client.start_audio_streaming(
            request.dict(), request_id
        )

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
    request: AudioProcessingRequest, request_id: str, file_path: str, audio_client
) -> Dict[str, Any]:
    """Process uploaded audio file"""

    try:
        # Send file to audio service for processing
        result = await audio_client.process_uploaded_file(
            file_path, request.dict(), request_id
        )

        return result

    except Exception as e:
        logger.error(f"[{request_id}] Uploaded file processing failed: {e}")
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
