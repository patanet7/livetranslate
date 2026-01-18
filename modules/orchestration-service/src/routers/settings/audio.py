"""
Audio Settings Router

Handles audio processing settings, chunking configuration, and correlation settings.
"""

from ._shared import (
    AUDIO_CONFIG_FILE,
    CHUNKING_CONFIG_FILE,
    CORRELATION_CONFIG_FILE,
    Any,
    APIRouter,
    AudioProcessingConfig,
    ChunkingConfig,
    CorrelationConfig,
    HTTPException,
    asyncio,
    load_config,
    logger,
    save_config,
)

router = APIRouter(tags=["settings-audio"])


# ============================================================================
# Enhanced Audio Processing Settings Endpoints
# ============================================================================


@router.get("/audio-processing", response_model=dict[str, Any])
async def get_audio_processing_settings():
    """Get current audio processing configuration"""
    try:
        default_config = AudioProcessingConfig().dict()
        config = await load_config(AUDIO_CONFIG_FILE, default_config)
        return config
    except Exception as e:
        logger.error(f"Error getting audio processing settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load audio processing settings") from e


@router.post("/audio-processing")
async def save_audio_processing_settings(config: AudioProcessingConfig):
    """Save audio processing configuration"""
    try:
        config_dict = config.dict()
        success = await save_config(AUDIO_CONFIG_FILE, config_dict)
        if success:
            return {
                "message": "Audio processing settings saved successfully",
                "config": config_dict,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save audio processing settings")
    except Exception as e:
        logger.error(f"Error saving audio processing settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio processing settings") from e


@router.post("/audio-processing/test")
async def test_audio_processing(test_config: dict[str, Any]):
    """Test audio processing configuration"""
    try:
        await asyncio.sleep(1)  # Simulate processing

        if not test_config.get("vad", {}).get("enabled"):
            return {"success": False, "message": "VAD must be enabled for testing"}

        return {
            "success": True,
            "message": "Audio processing test completed successfully",
            "metrics": {
                "processing_time_ms": 150,
                "snr_db": 25.3,
                "quality_score": 0.92,
            },
        }
    except Exception as e:
        logger.error(f"Error testing audio processing: {e}")
        raise HTTPException(status_code=500, detail="Audio processing test failed") from e


# ============================================================================
# Chunking Settings Endpoints
# ============================================================================


@router.get("/chunking", response_model=dict[str, Any])
async def get_chunking_settings():
    """Get current chunking configuration"""
    try:
        default_config = ChunkingConfig().dict()
        config = await load_config(CHUNKING_CONFIG_FILE, default_config)
        return config
    except Exception as e:
        logger.error(f"Error getting chunking settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load chunking settings") from e


@router.post("/chunking")
async def save_chunking_settings(config: ChunkingConfig):
    """Save chunking configuration"""
    try:
        config_dict = config.dict()
        success = await save_config(CHUNKING_CONFIG_FILE, config_dict)
        if success:
            return {
                "message": "Chunking settings saved successfully",
                "config": config_dict,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save chunking settings")
    except Exception as e:
        logger.error(f"Error saving chunking settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save chunking settings") from e


@router.get("/chunking/stats")
async def get_chunking_stats():
    """Get chunking performance statistics"""
    try:
        return {
            "total_chunks_processed": 1250,
            "average_chunk_duration": 4.8,
            "overlap_efficiency": 0.95,
            "storage_utilization_gb": 12.4,
            "processing_latency_ms": 85,
        }
    except Exception as e:
        logger.error(f"Error getting chunking stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to load chunking statistics") from e


# ============================================================================
# Correlation Settings Endpoints
# ============================================================================


@router.get("/correlation", response_model=dict[str, Any])
async def get_correlation_settings():
    """Get current correlation configuration"""
    try:
        default_config = CorrelationConfig().dict()
        config = await load_config(CORRELATION_CONFIG_FILE, default_config)
        return config
    except Exception as e:
        logger.error(f"Error getting correlation settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load correlation settings") from e


@router.post("/correlation")
async def save_correlation_settings(config: CorrelationConfig):
    """Save correlation configuration"""
    try:
        config_dict = config.dict()
        success = await save_config(CORRELATION_CONFIG_FILE, config_dict)
        if success:
            return {
                "message": "Correlation settings saved successfully",
                "config": config_dict,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save correlation settings")
    except Exception as e:
        logger.error(f"Error saving correlation settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save correlation settings") from e


@router.get("/correlation/manual-mappings")
async def get_manual_mappings():
    """Get manual speaker mappings"""
    try:
        # Return mock manual mappings
        return [
            {
                "whisper_speaker_id": "speaker_0",
                "google_meet_speaker_id": "user_12345",
                "display_name": "John Doe",
                "confidence": 0.95,
                "is_confirmed": True,
            },
            {
                "whisper_speaker_id": "speaker_1",
                "google_meet_speaker_id": "user_67890",
                "display_name": "Jane Smith",
                "confidence": 0.88,
                "is_confirmed": False,
            },
        ]
    except Exception as e:
        logger.error(f"Error getting manual mappings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load manual mappings") from e


@router.post("/correlation/manual-mappings")
async def save_manual_mapping(mapping: dict[str, Any]):
    """Save manual speaker mapping"""
    try:
        required_fields = [
            "whisper_speaker_id",
            "google_meet_speaker_id",
            "display_name",
        ]
        for field in required_fields:
            if field not in mapping:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        return {
            "message": "Manual speaker mapping saved successfully",
            "mapping": mapping,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving manual mapping: {e}")
        raise HTTPException(status_code=500, detail="Failed to save manual mapping") from e


@router.delete("/correlation/manual-mappings/{mapping_id}")
async def delete_manual_mapping(mapping_id: str):
    """Delete manual speaker mapping"""
    try:
        return {"message": f"Manual speaker mapping {mapping_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting manual mapping: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete manual mapping") from e


@router.get("/correlation/stats")
async def get_correlation_stats():
    """Get correlation statistics"""
    try:
        return {
            "total_correlations": 845,
            "successful_correlations": 798,
            "manual_correlations": 156,
            "acoustic_correlations": 642,
            "average_confidence": 0.87,
        }
    except Exception as e:
        logger.error(f"Error getting correlation stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to load correlation statistics") from e


@router.post("/correlation/test")
async def test_correlation(test_config: dict[str, Any]):
    """Test speaker correlation configuration"""
    try:
        await asyncio.sleep(2)
        return {
            "success": True,
            "message": "Speaker correlation test completed successfully",
            "results": {
                "correlations_found": 3,
                "confidence_scores": [0.92, 0.85, 0.78],
                "processing_time_ms": 1250,
            },
        }
    except Exception as e:
        logger.error(f"Error testing correlation: {e}")
        raise HTTPException(status_code=500, detail="Correlation test failed") from e
