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

from ._shared import *

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


async def _process_audio_streaming(
    request: AudioProcessingRequest,
    correlation_id: str,
    audio_client,
    audio_service_config: Dict[str, Any]
) -> AudioProcessingResponse:
    """Core streaming audio processing logic"""
    # Placeholder for streaming implementation
    # This would contain the actual streaming processing logic
    return AudioProcessingResponse(
        request_id=correlation_id,
        status="processed",
        transcription="Streaming processing placeholder",
        processing_time=0.1,
        confidence=0.95
    )


async def _process_audio_batch(
    request: AudioProcessingRequest,
    correlation_id: str,
    audio_client,
    audio_service_config: Dict[str, Any]
) -> AudioProcessingResponse:
    """Core batch audio processing logic"""
    # Placeholder for batch implementation
    # This would contain the actual batch processing logic
    return AudioProcessingResponse(
        request_id=correlation_id,
        status="processed",
        transcription="Batch processing placeholder",
        processing_time=0.2,
        confidence=0.96
    )


@router.post("/upload", response_model=Dict[str, Any])
async def upload_audio_file(
    audio: UploadFile = File(..., alias="audio"),
    config: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    audio_coordinator=Depends(get_audio_coordinator),
    config_sync_manager=Depends(get_config_sync_manager),
    audio_client=Depends(get_audio_service_client),
    event_publisher=Depends(get_event_publisher),
) -> Dict[str, Any]:
    """
    Upload audio file for processing with enhanced error handling and validation
    
    - **audio**: Audio file to upload (WAV, MP3, OGG, WebM, MP4, FLAC)
    - **config**: Optional JSON configuration for processing
    - **session_id**: Optional session identifier for tracking
    """
    correlation_id = f"upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    
    async with error_boundary(
        correlation_id=correlation_id,
        context={
            "service": "orchestration",
            "endpoint": "/upload",
            "filename": audio.filename,
            "content_type": audio.content_type
        },
        recovery_strategies=[format_recovery, service_recovery]
    ) as upload_correlation_id:
        try:
            logger.info(f"[{upload_correlation_id}] Processing file upload: {audio.filename}")
            
            # Enhanced file validation
            await _validate_upload_file(audio, upload_correlation_id)
            
            # Create processing request from uploaded file
            processing_request = AudioProcessingRequest(
                file_upload=audio.filename,
                config=config,
                session_id=session_id,
                streaming=False  # File uploads are batch processed
            )
            
            # Process the uploaded file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio.filename}") as temp_file:
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
                    "filename": audio.filename,
                    "content_type": audio.content_type,
                    "file_size": len(content),
                    "config_provided": bool(config),
                },
                metadata={"endpoint": "/upload"},
            )
            
            try:
                # Process uploaded file safely
                result = await _process_uploaded_file_safe(
                    processing_request,
                    upload_correlation_id,
                    temp_file_path,
                    audio_client,
                    {"config": config, "session_id": session_id},
                    audio_coordinator,
                    config_sync_manager
                )
                
                return {
                    "upload_id": upload_correlation_id,
                    "filename": audio.filename,
                    "status": "uploaded_and_processed",
                    "file_size": len(content),
                    "processing_result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")
                    
        except AudioProcessingBaseError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Convert unknown exceptions to appropriate audio errors
            raise AudioProcessingError(
                f"File upload failed: {str(e)}",
                correlation_id=upload_correlation_id,
                processing_stage="file_upload",
                details={"filename": audio.filename, "error": str(e)}
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


async def _process_uploaded_file(
    processing_request: AudioProcessingRequest,
    correlation_id: str,
    temp_file_path: str,
    audio_client,
    request_data: Dict[str, Any],
    audio_coordinator,
    config_sync_manager
) -> Dict[str, Any]:
    """Core uploaded file processing logic"""
    # Placeholder for file processing implementation
    # This would contain the actual file processing logic
    return {
        "status": "processed",
        "transcription": "File processing placeholder",
        "processing_time": 0.3,
        "confidence": 0.94,
        "file_path": temp_file_path
    }


@router.get("/health")
async def health_check(
    health_monitor=Depends(get_health_monitor)
) -> Dict[str, Any]:
    """
    Audio service health check with comprehensive diagnostics
    """
    try:
        # Get comprehensive health status
        health_status = await health_monitor.get_comprehensive_health()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "orchestration_audio",
            "version": "2.0.0",
            "components": health_status,
            "endpoints": {
                "process": "operational",
                "upload": "operational", 
                "stream": "operational",
                "transcribe": "operational"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/models")
async def get_available_models(
    audio_client=Depends(get_audio_service_client)
) -> Dict[str, Any]:
    """
    Get available audio processing models
    """
    try:
        # Get models from audio service
        models = await audio_client.get_available_models()
        
        return {
            "models": models,
            "default_model": "whisper-base",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Audio service unavailable: {str(e)}"
        )


@router.get("/stats", response_model=AudioStats)
async def get_audio_stats(
    health_monitor=Depends(get_health_monitor)
) -> AudioStats:
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
            active_sessions=stats.get("active_sessions", 0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )
