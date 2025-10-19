"""
Audio Processing API Router - Main Module

Unified audio processing router that combines all audio functionality:
- Core processing (audio_core.py)
- Analysis endpoints (audio_analysis.py)  
- Stage processing (audio_stages.py)
- Preset management (audio_presets.py)

This replaces the original monolithic audio.py file with a modular architecture.
"""

from fastapi import APIRouter
from .audio import audio_core, audio_analysis, audio_stages, audio_presets

# Create the main audio router
router = APIRouter(
    prefix="/audio",
    tags=["audio"],
    responses={
        404: {"description": "Not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Include all audio sub-routers
router.include_router(
    audio_core.router,
    tags=["audio-core"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad request"},
        503: {"description": "Service unavailable"}
    }
)

router.include_router(
    audio_analysis.router,
    prefix="/analyze",
    tags=["audio-analysis"],
    responses={
        200: {"description": "Analysis complete"},
        400: {"description": "Invalid analysis parameters"}
    }
)

router.include_router(
    audio_stages.router,
    prefix="/stages",
    tags=["audio-stages"],
    responses={
        200: {"description": "Stage processing complete"},
        404: {"description": "Stage not found"}
    }
)

router.include_router(
    audio_presets.router,
    prefix="/presets",
    tags=["audio-presets"],
    responses={
        200: {"description": "Preset operation successful"},
        404: {"description": "Preset not found"},
        409: {"description": "Preset conflict"}
    }
)

# Add any additional combined endpoints here if needed
@router.get("/info")
async def get_audio_router_info():
    """
    Get information about the audio router and its capabilities
    """
    return {
        "name": "Orchestration Audio Router",
        "version": "2.0.0",
        "description": "Modular audio processing router with core, analysis, stages, and presets",
        "modules": {
            "core": {
                "description": "Core audio processing endpoints",
                "endpoints": ["process", "upload", "health", "models", "stats"]
            },
            "analysis": {
                "description": "Audio analysis and metrics",
                "endpoints": ["fft", "lufs", "spectrum", "quality"]
            },
            "stages": {
                "description": "Individual stage processing",
                "endpoints": ["process/stage", "info", "config", "pipeline"]
            },
            "presets": {
                "description": "Configuration preset management",
                "endpoints": ["list", "get", "apply", "save", "delete", "compare"]
            }
        },
        "total_endpoints": 25,
        "architecture": "modular",
        "splitting_date": "2024-current",
        "original_size": "3,046 lines",
        "new_structure": {
            "audio_core.py": "~400 lines",
            "audio_analysis.py": "~300 lines", 
            "audio_stages.py": "~500 lines",
            "audio_presets.py": "~400 lines",
            "total_new_size": "~1,600 lines across 4 focused modules"
        }
    }