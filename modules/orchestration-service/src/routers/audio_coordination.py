"""Audio coordination router providing access to the in-process AudioCoordinator."""

from __future__ import annotations

import asyncio
import base64
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from audio.models import AudioStreamingSession, SourceType
from dependencies import get_audio_coordinator


router = APIRouter()


class CreateSessionRequest(BaseModel):
    bot_session_id: str = Field(..., description="Parent bot session identifier")
    source_type: SourceType = Field(SourceType.BOT_AUDIO, description="Audio source type")
    target_languages: Optional[List[str]] = Field(
        default=None, description="Languages to request translations for"
    )
    custom_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Chunking configuration overrides"
    )
    auto_start: bool = Field(
        default=True, description="Automatically start the session after creation"
    )


class StartStopResponse(BaseModel):
    session_id: str
    success: bool
    details: Optional[Dict[str, Any]] = None


class AudioChunkRequest(BaseModel):
    session_id: str
    audio_base64: str = Field(..., description="Base64 encoded float32 audio samples")


async def _ensure_coordinator_ready(coordinator) -> None:
    if hasattr(coordinator, "is_running") and not coordinator.is_running:
        initialized = await coordinator.initialize()
        if not initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Audio coordinator failed to initialize",
            )


def _session_to_dict(session: AudioStreamingSession) -> Dict[str, Any]:
    data = session.dict()
    for key, value in list(data.items()):
        if hasattr(value, "isoformat"):
            data[key] = value.isoformat()
    return data


@router.get("/health")
async def audio_coordination_health(coordinator=Depends(get_audio_coordinator)) -> Dict[str, Any]:
    """Return current status of the audio coordination subsystem."""
    running = getattr(coordinator, "is_running", False)
    if running:
        stats = coordinator.session_manager.get_session_statistics()
    else:
        stats = {}
    return {
        "running": running,
        "statistics": stats,
    }


@router.get("/sessions")
async def list_audio_sessions(coordinator=Depends(get_audio_coordinator)) -> Dict[str, Any]:
    await _ensure_coordinator_ready(coordinator)
    sessions = coordinator.session_manager.get_all_sessions()
    return {"sessions": [_session_to_dict(session) for session in sessions]}


@router.post("/sessions", response_model=Dict[str, Any])
async def create_audio_session(
    request: CreateSessionRequest,
    coordinator=Depends(get_audio_coordinator),
) -> Dict[str, Any]:
    await _ensure_coordinator_ready(coordinator)
    session_id = await coordinator.create_audio_session(
        bot_session_id=request.bot_session_id,
        source_type=request.source_type,
        target_languages=request.target_languages,
        custom_config=request.custom_config,
    )
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create audio session",
        )

    started = False
    if request.auto_start:
        started = await coordinator.start_audio_session(session_id)

    return {"session_id": session_id, "started": started}


@router.post("/sessions/{session_id}/start", response_model=StartStopResponse)
async def start_audio_session(session_id: str, coordinator=Depends(get_audio_coordinator)) -> StartStopResponse:
    await _ensure_coordinator_ready(coordinator)
    success = await coordinator.start_audio_session(session_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return StartStopResponse(session_id=session_id, success=True)


@router.post("/sessions/{session_id}/stop", response_model=StartStopResponse)
async def stop_audio_session(session_id: str, coordinator=Depends(get_audio_coordinator)) -> StartStopResponse:
    await _ensure_coordinator_ready(coordinator)
    details = await coordinator.stop_audio_session(session_id)
    return StartStopResponse(session_id=session_id, success=True, details=details)


@router.get("/sessions/{session_id}")
async def get_audio_session(session_id: str, coordinator=Depends(get_audio_coordinator)) -> Dict[str, Any]:
    await _ensure_coordinator_ready(coordinator)
    session = coordinator.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return _session_to_dict(session)


@router.post("/sessions/{session_id}/chunks")
async def add_audio_chunk(
    session_id: str,
    chunk: AudioChunkRequest,
    coordinator=Depends(get_audio_coordinator),
) -> Dict[str, Any]:
    if session_id != chunk.session_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session ID mismatch")

    await _ensure_coordinator_ready(coordinator)

    try:
        audio_bytes = base64.b64decode(chunk.audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid audio payload: {exc}") from exc

    if audio_array.size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    success = await coordinator.add_audio_data(session_id, audio_array)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or inactive")

    return {"session_id": session_id, "samples_processed": int(audio_array.size)}


@router.get("/statistics")
async def get_audio_statistics(coordinator=Depends(get_audio_coordinator)) -> Dict[str, Any]:
    await _ensure_coordinator_ready(coordinator)
    return coordinator.session_manager.get_session_statistics()


@router.get("/config/schema")
async def get_audio_config_schema(coordinator=Depends(get_audio_coordinator)) -> Dict[str, Any]:
    await _ensure_coordinator_ready(coordinator)
    return coordinator.get_audio_config_schema()


@router.post("/config/presets/{preset_name}")
async def apply_audio_preset(preset_name: str, coordinator=Depends(get_audio_coordinator)) -> Dict[str, Any]:
    await _ensure_coordinator_ready(coordinator)
    success = await coordinator.apply_audio_preset(preset_name)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found")
    return {"preset": preset_name, "applied": True}

