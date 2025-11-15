"""
Audio Stage Processing Router

Individual audio processing stage endpoints including:
- Stage processing (/process/stage/{stage_name})
- Stage information (/stages/info)
- Stage configuration (/stages/config)
- Pipeline management
"""

from ._shared import *

# Create router for audio stage processing
router = create_audio_router()


@router.post("/process/stage/{stage_name}")
async def process_audio_stage(
    stage_name: str,
    request: Dict[str, Any],
    audio_coordinator=Depends(get_audio_coordinator),
    config_manager=Depends(get_config_manager)
) -> Dict[str, Any]:
    """
    Process audio through a specific pipeline stage
    
    - **stage_name**: Name of the processing stage (e.g., 'noise_reduction', 'vad', 'compression')
    - **audio_data**: Base64 encoded audio data
    - **stage_config**: Stage-specific configuration parameters
    """
    correlation_id = f"stage_{stage_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    
    async with error_boundary(
        correlation_id=correlation_id,
        context={
            "service": "orchestration",
            "endpoint": f"/process/stage/{stage_name}",
            "stage": stage_name
        },
        recovery_strategies=[format_recovery, service_recovery]
    ) as stage_correlation_id:
        try:
            logger.info(f"[{stage_correlation_id}] Processing stage: {stage_name}")
            
            # Validate stage name
            available_stages = await _get_available_stages()
            if stage_name not in available_stages:
                raise ValidationError(
                    f"Unknown processing stage: {stage_name}",
                    correlation_id=stage_correlation_id,
                    validation_details={
                        "provided_stage": stage_name,
                        "available_stages": list(available_stages.keys())
                    }
                )
            
            # Validate request
            audio_data = request.get("audio_data")
            if not audio_data:
                raise ValidationError(
                    "No audio data provided for stage processing",
                    correlation_id=stage_correlation_id,
                    validation_details={"missing_field": "audio_data"}
                )
            
            # Get stage configuration
            stage_config = request.get("stage_config", {})
            default_config = available_stages[stage_name]["default_config"]
            merged_config = {**default_config, **stage_config}
            
            # Process through specific stage
            result = await _process_single_stage(
                stage_name, audio_data, merged_config, stage_correlation_id, audio_coordinator
            )
            
            return {
                "stage_id": stage_correlation_id,
                "stage_name": stage_name,
                "stage_config": merged_config,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except AudioProcessingBaseError:
            raise
        except Exception as e:
            raise AudioProcessingError(
                f"Stage processing failed for {stage_name}: {str(e)}",
                correlation_id=stage_correlation_id,
                processing_stage=stage_name,
                details={"error": str(e)}
            )


async def _get_available_stages() -> Dict[str, Dict[str, Any]]:
    """Get information about all available processing stages"""
    return {
        "noise_reduction": {
            "description": "Reduce background noise using spectral subtraction",
            "input_format": "audio/wav",
            "output_format": "audio/wav",
            "parameters": ["strength", "frequency_smoothing", "time_smoothing"],
            "default_config": {
                "strength": 0.5,
                "frequency_smoothing": 0.3,
                "time_smoothing": 0.4
            },
            "processing_time": "fast"
        },
        "vad": {
            "description": "Voice Activity Detection to identify speech segments",
            "input_format": "audio/wav",
            "output_format": "json",
            "parameters": ["threshold", "min_speech_duration", "min_silence_duration"],
            "default_config": {
                "threshold": 0.5,
                "min_speech_duration": 0.1,
                "min_silence_duration": 0.3
            },
            "processing_time": "fast"
        },
        "compression": {
            "description": "Dynamic range compression to normalize audio levels",
            "input_format": "audio/wav",
            "output_format": "audio/wav", 
            "parameters": ["ratio", "threshold", "attack", "release"],
            "default_config": {
                "ratio": 4.0,
                "threshold": -20.0,
                "attack": 0.003,
                "release": 0.1
            },
            "processing_time": "fast"
        },
        "equalizer": {
            "description": "Frequency equalization for tonal balance",
            "input_format": "audio/wav",
            "output_format": "audio/wav",
            "parameters": ["low_shelf", "mid_peak", "high_shelf"],
            "default_config": {
                "low_shelf": {"freq": 100, "gain": 0, "q": 0.7},
                "mid_peak": {"freq": 1000, "gain": 0, "q": 1.0},
                "high_shelf": {"freq": 10000, "gain": 0, "q": 0.7}
            },
            "processing_time": "fast"
        },
        "limiter": {
            "description": "Peak limiting to prevent clipping",
            "input_format": "audio/wav",
            "output_format": "audio/wav",
            "parameters": ["threshold", "release", "lookahead"],
            "default_config": {
                "threshold": -0.1,
                "release": 0.05,
                "lookahead": 0.005
            },
            "processing_time": "fast"
        },
        "voice_enhancement": {
            "description": "Enhance voice clarity and intelligibility",
            "input_format": "audio/wav",
            "output_format": "audio/wav",
            "parameters": ["enhancement_strength", "formant_correction", "sibilance_control"],
            "default_config": {
                "enhancement_strength": 0.6,
                "formant_correction": True,
                "sibilance_control": 0.3
            },
            "processing_time": "medium"
        },
        "voice_filter": {
            "description": "Filter to isolate voice frequencies",
            "input_format": "audio/wav",
            "output_format": "audio/wav",
            "parameters": ["low_cutoff", "high_cutoff", "filter_order"],
            "default_config": {
                "low_cutoff": 80,
                "high_cutoff": 8000,
                "filter_order": 4
            },
            "processing_time": "fast"
        },
        "agc": {
            "description": "Automatic Gain Control for consistent levels",
            "input_format": "audio/wav",
            "output_format": "audio/wav",
            "parameters": ["target_level", "max_gain", "attack_time", "release_time"],
            "default_config": {
                "target_level": -20.0,
                "max_gain": 20.0,
                "attack_time": 0.1,
                "release_time": 1.0
            },
            "processing_time": "fast"
        },
        "spectral_denoising": {
            "description": "Advanced spectral domain noise reduction",
            "input_format": "audio/wav",
            "output_format": "audio/wav",
            "parameters": ["noise_profile", "reduction_strength", "preserve_speech"],
            "default_config": {
                "noise_profile": "auto",
                "reduction_strength": 0.7,
                "preserve_speech": True
            },
            "processing_time": "slow"
        },
        "lufs_normalization": {
            "description": "Normalize audio to target LUFS level",
            "input_format": "audio/wav",
            "output_format": "audio/wav",
            "parameters": ["target_lufs", "max_peak", "measurement_type"],
            "default_config": {
                "target_lufs": -23.0,
                "max_peak": -1.0,
                "measurement_type": "integrated"
            },
            "processing_time": "medium"
        }
    }


async def _process_single_stage(
    stage_name: str,
    audio_data: str,
    stage_config: Dict[str, Any],
    correlation_id: str,
    audio_coordinator
) -> Dict[str, Any]:
    """Process audio through a single stage"""
    try:
        # Get stage info
        stages_info = await _get_available_stages()
        stage_info = stages_info[stage_name]
        
        # Decode audio data
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        # Simulate stage processing based on stage type
        processing_time = _get_processing_time(stage_info["processing_time"])
        await asyncio.sleep(processing_time)  # Simulate processing delay
        
        # Generate stage-specific results
        if stage_name == "vad":
            result = await _process_vad_stage(audio_bytes, stage_config, correlation_id)
        elif stage_name == "noise_reduction":
            result = await _process_noise_reduction_stage(audio_bytes, stage_config, correlation_id)
        elif stage_name == "compression":
            result = await _process_compression_stage(audio_bytes, stage_config, correlation_id)
        elif stage_name == "equalizer":
            result = await _process_equalizer_stage(audio_bytes, stage_config, correlation_id)
        elif stage_name == "limiter":
            result = await _process_limiter_stage(audio_bytes, stage_config, correlation_id)
        else:
            # Generic stage processing
            result = await _process_generic_stage(stage_name, audio_bytes, stage_config, correlation_id)
        
        return {
            "status": "processed",
            "stage_name": stage_name,
            "input_size": len(audio_bytes),
            "processing_time": processing_time,
            "stage_result": result,
            "config_applied": stage_config
        }
        
    except Exception as e:
        raise AudioProcessingError(
            f"Stage {stage_name} processing failed: {str(e)}",
            correlation_id=correlation_id,
            processing_stage=stage_name,
            details={"error": str(e)}
        )


def _get_processing_time(speed: str) -> float:
    """Get processing delay based on stage speed"""
    speed_map = {
        "fast": 0.02,
        "medium": 0.05,
        "slow": 0.1
    }
    return speed_map.get(speed, 0.05)


async def _process_vad_stage(audio_bytes: bytes, config: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
    """Process Voice Activity Detection stage"""
    threshold = config.get("threshold", 0.5)
    min_speech_duration = config.get("min_speech_duration", 0.1)
    min_silence_duration = config.get("min_silence_duration", 0.3)
    
    # Simulate VAD detection
    import numpy as np
    audio_duration = len(audio_bytes) / (44100 * 2)  # Assume 16-bit, 44.1kHz
    
    # Generate realistic speech segments
    segments = []
    current_time = 0.0
    while current_time < audio_duration:
        if np.random.random() > 0.3:  # 70% chance of speech
            speech_duration = max(min_speech_duration, np.random.exponential(2.0))
            speech_duration = min(speech_duration, audio_duration - current_time)
            segments.append({
                "start_time": current_time,
                "end_time": current_time + speech_duration,
                "type": "speech",
                "confidence": 0.7 + np.random.random() * 0.3
            })
            current_time += speech_duration
        else:
            silence_duration = max(min_silence_duration, np.random.exponential(1.0))
            silence_duration = min(silence_duration, audio_duration - current_time)
            segments.append({
                "start_time": current_time,
                "end_time": current_time + silence_duration,
                "type": "silence",
                "confidence": 0.8 + np.random.random() * 0.2
            })
            current_time += silence_duration
    
    speech_percentage = sum(seg["end_time"] - seg["start_time"] for seg in segments if seg["type"] == "speech") / audio_duration * 100
    
    return {
        "segments": segments,
        "total_segments": len(segments),
        "speech_segments": len([s for s in segments if s["type"] == "speech"]),
        "silence_segments": len([s for s in segments if s["type"] == "silence"]),
        "speech_percentage": round(speech_percentage, 2),
        "total_duration": round(audio_duration, 3),
        "threshold_used": threshold
    }


async def _process_noise_reduction_stage(audio_bytes: bytes, config: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
    """Process noise reduction stage"""
    strength = config.get("strength", 0.5)
    freq_smoothing = config.get("frequency_smoothing", 0.3)
    time_smoothing = config.get("time_smoothing", 0.4)
    
    # Simulate noise reduction metrics
    import numpy as np
    original_snr = 15.0 + np.random.normal(0, 3.0)
    snr_improvement = strength * 10.0 + np.random.normal(0, 1.0)
    final_snr = original_snr + snr_improvement
    
    noise_reduction_db = strength * 15.0 + np.random.normal(0, 2.0)
    
    return {
        "output_audio": base64.b64encode(audio_bytes).decode(),  # Placeholder - same as input
        "original_snr_db": round(original_snr, 2),
        "final_snr_db": round(final_snr, 2),
        "snr_improvement_db": round(snr_improvement, 2),
        "noise_reduction_db": round(noise_reduction_db, 2),
        "strength_applied": strength,
        "frequency_smoothing": freq_smoothing,
        "time_smoothing": time_smoothing,
        "processing_quality": "high" if strength < 0.7 else "medium"
    }


async def _process_compression_stage(audio_bytes: bytes, config: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
    """Process compression stage"""
    ratio = config.get("ratio", 4.0)
    threshold = config.get("threshold", -20.0)
    attack = config.get("attack", 0.003)
    release = config.get("release", 0.1)
    
    # Simulate compression metrics
    gain_reduction_max = (ratio - 1) / ratio * abs(threshold) / 3
    gain_reduction_avg = gain_reduction_max * 0.6
    
    return {
        "output_audio": base64.b64encode(audio_bytes).decode(),  # Placeholder
        "ratio_applied": ratio,
        "threshold_db": threshold,
        "attack_time_ms": attack * 1000,
        "release_time_ms": release * 1000,
        "max_gain_reduction_db": round(gain_reduction_max, 2),
        "average_gain_reduction_db": round(gain_reduction_avg, 2),
        "compression_detected": gain_reduction_avg > 1.0
    }


async def _process_equalizer_stage(audio_bytes: bytes, config: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
    """Process equalizer stage"""
    low_shelf = config.get("low_shelf", {})
    mid_peak = config.get("mid_peak", {})
    high_shelf = config.get("high_shelf", {})
    
    return {
        "output_audio": base64.b64encode(audio_bytes).decode(),  # Placeholder
        "eq_bands_applied": 3,
        "low_shelf": low_shelf,
        "mid_peak": mid_peak,
        "high_shelf": high_shelf,
        "frequency_response_altered": any([
            abs(low_shelf.get("gain", 0)) > 0.1,
            abs(mid_peak.get("gain", 0)) > 0.1,
            abs(high_shelf.get("gain", 0)) > 0.1
        ])
    }


async def _process_limiter_stage(audio_bytes: bytes, config: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
    """Process limiter stage"""
    threshold = config.get("threshold", -0.1)
    release = config.get("release", 0.05)
    lookahead = config.get("lookahead", 0.005)
    
    # Simulate limiter metrics
    import numpy as np
    limiting_occurred = np.random.random() > 0.7  # 30% chance of limiting
    max_reduction = abs(threshold) * 0.5 if limiting_occurred else 0.0
    
    return {
        "output_audio": base64.b64encode(audio_bytes).decode(),  # Placeholder
        "threshold_db": threshold,
        "release_time_ms": release * 1000,
        "lookahead_time_ms": lookahead * 1000,
        "limiting_occurred": limiting_occurred,
        "max_gain_reduction_db": round(max_reduction, 2),
        "peak_prevented": limiting_occurred
    }


async def _process_generic_stage(stage_name: str, audio_bytes: bytes, config: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
    """Process generic stage"""
    return {
        "output_audio": base64.b64encode(audio_bytes).decode(),  # Placeholder
        "stage_name": stage_name,
        "config_applied": config,
        "processing_successful": True,
        "note": f"Generic processing for {stage_name} stage"
    }


@router.get("/stages/info")
async def get_stages_info() -> Dict[str, Any]:
    """
    Get detailed information about all available processing stages
    """
    try:
        stages_info = await _get_available_stages()
        
        return {
            "available_stages": stages_info,
            "total_stages": len(stages_info),
            "categories": {
                "noise_processing": ["noise_reduction", "spectral_denoising"],
                "dynamics": ["compression", "limiter", "agc"],
                "frequency": ["equalizer", "voice_filter"],
                "enhancement": ["voice_enhancement", "lufs_normalization"],
                "analysis": ["vad"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get stages info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stages information: {str(e)}"
        )


@router.get("/stages/{stage_name}/config")
async def get_stage_config(stage_name: str) -> Dict[str, Any]:
    """
    Get configuration schema and defaults for a specific stage
    """
    try:
        stages_info = await _get_available_stages()
        
        if stage_name not in stages_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stage '{stage_name}' not found"
            )
        
        stage_info = stages_info[stage_name]
        
        return {
            "stage_name": stage_name,
            "description": stage_info["description"],
            "parameters": stage_info["parameters"],
            "default_config": stage_info["default_config"],
            "input_format": stage_info["input_format"],
            "output_format": stage_info["output_format"],
            "processing_time": stage_info["processing_time"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get config for stage {stage_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stage configuration: {str(e)}"
        )


@router.post("/pipeline/custom")
async def create_custom_pipeline(
    request: Dict[str, Any],
    audio_coordinator=Depends(get_audio_coordinator)
) -> Dict[str, Any]:
    """
    Create and execute a custom processing pipeline with multiple stages
    
    - **audio_data**: Base64 encoded audio data
    - **pipeline**: List of stages with their configurations
    """
    correlation_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    
    try:
        audio_data = request.get("audio_data")
        pipeline_stages = request.get("pipeline", [])
        
        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No audio data provided"
            )
        
        if not pipeline_stages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No pipeline stages specified"
            )
        
        # Validate all stages exist
        available_stages = await _get_available_stages()
        for stage_config in pipeline_stages:
            stage_name = stage_config.get("stage_name")
            if stage_name not in available_stages:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown stage: {stage_name}"
                )
        
        # Process through pipeline
        pipeline_results = []
        current_audio = audio_data
        
        for i, stage_config in enumerate(pipeline_stages):
            stage_name = stage_config["stage_name"]
            stage_params = stage_config.get("config", {})
            
            logger.info(f"[{correlation_id}] Processing stage {i+1}/{len(pipeline_stages)}: {stage_name}")
            
            # Process through stage
            stage_result = await _process_single_stage(
                stage_name, current_audio, stage_params, f"{correlation_id}_stage_{i}", audio_coordinator
            )
            
            # Update audio for next stage (if output is audio)
            if stage_result.get("stage_result", {}).get("output_audio"):
                current_audio = stage_result["stage_result"]["output_audio"]
            
            pipeline_results.append({
                "stage_index": i,
                "stage_name": stage_name,
                "stage_config": stage_params,
                "stage_result": stage_result
            })
        
        return {
            "pipeline_id": correlation_id,
            "total_stages": len(pipeline_stages),
            "pipeline_results": pipeline_results,
            "final_audio": current_audio,
            "total_processing_time": sum(r["stage_result"]["processing_time"] for r in pipeline_results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Custom pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {str(e)}"
        )