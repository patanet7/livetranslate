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


