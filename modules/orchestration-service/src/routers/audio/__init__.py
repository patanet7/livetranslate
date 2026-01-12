"""
Audio Router Package

Modular audio processing router split from the original monolithic audio.py file.

Modules:
- _shared: Common imports, utilities, and shared components
- audio_core: Core processing endpoints (process, upload, health, models)
- audio_analysis: Analysis endpoints (FFT, LUFS, spectrum, quality)
- audio_stages: Stage processing endpoints (individual stages, pipeline)
- audio_presets: Preset management endpoints (CRUD operations, comparison)
- audio_coordination: Session management and audio coordinator endpoints
- websocket_audio: Real-time WebSocket streaming endpoint
"""

# Export individual modules for importing
from . import (
    audio_core,
    audio_analysis,
    audio_stages,
    audio_presets,
    audio_coordination,
    websocket_audio,
)
from fastapi import APIRouter

# Aggregate router so main app can mount /audio with all sub-routes
# Note: No prefix here since main_fastapi.py already adds /api/audio
router = APIRouter(tags=["audio"])
router.include_router(audio_core.router)
router.include_router(audio_analysis.router)
router.include_router(audio_stages.router)
router.include_router(audio_presets.router)

# Audio coordination router (session management)
# Mounted at /api/audio-coordination by main_fastapi.py
coordination_router = audio_coordination.router

# WebSocket audio streaming router
# Mounted at /api/audio (shares prefix with core) for WebSocket endpoint
websocket_router = websocket_audio.router

__all__ = [
    "audio_core",
    "audio_analysis",
    "audio_stages",
    "audio_presets",
    "audio_coordination",
    "websocket_audio",
    "router",
    "coordination_router",
    "websocket_router",
]
