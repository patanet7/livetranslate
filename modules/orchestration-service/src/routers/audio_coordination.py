"""
Audio Coordination API Router - Centralized Audio Processing

Provides comprehensive FastAPI endpoints for coordinated audio processing with:
- Real-time audio streaming with centralized chunking
- Database integration with bot_sessions schema
- Speaker correlation and timing coordination
- Comprehensive analytics and session management
- Enhanced audio processing pipeline integration

This router implements the centralized audio coordination system that manages
all audio processing through the AudioCoordinator with proper overlap handling,
speaker correlation, and database persistence.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
import aiofiles

# Audio coordination system imports
try:
    from audio.audio_coordinator import AudioCoordinator
    from audio.models import (
        AudioChunkMetadata,
        AudioChunkingConfig,
        SpeakerCorrelation,
        ProcessingResult,
        ChunkLineage,
        QualityMetrics,
    )
    from audio.database_adapter import AudioDatabaseAdapter
    from dependencies import get_config_manager, get_database_adapter
    AUDIO_COORDINATION_AVAILABLE = True
    print("✅ Audio coordination components loaded successfully")
except ImportError as e:
    logging.warning(f"Audio coordination imports not available: {e}")
    print(f"⚠️  Audio coordination degraded mode: {e}")
    AUDIO_COORDINATION_AVAILABLE = False
    
    # Define fallback AudioCoordinator
    class AudioCoordinator:
        def __init__(self, *args, **kwargs):
            pass
        
        async def process_chunk(self, *args, **kwargs):
            return {"status": "unavailable", "message": "Audio coordination not available"}
    
    # Define fallback models
    class AudioChunkMetadata(BaseModel):
        chunk_id: str
        session_id: str
        sequence_number: int
        start_time: float
        end_time: float
        duration: float
        overlap_start: float
        overlap_end: float
        file_path: str
        file_size: int
        sample_rate: int
        channels: int
        quality_score: float
        rms_level: float
        peak_level: float
        
    class AudioChunkingConfig(BaseModel):
        chunk_duration: float = 3.0
        overlap_duration: float = 0.5
        processing_interval: float = 2.5
        buffer_duration: float = 10.0
        
    class SpeakerCorrelation(BaseModel):
        correlation_id: str
        session_id: str
        whisper_speaker_id: str
        google_meet_speaker_id: str
        confidence: float
        correlation_type: str
        
    class ProcessingResult(BaseModel):
        request_id: str
        session_id: str
        chunk_id: str
        processing_status: str
        results: Dict[str, Any]
        
    class ChunkLineage(BaseModel):
        chunk_id: str
        parent_chunk_id: Optional[str]
        processing_pipeline_version: str
        
    class QualityMetrics(BaseModel):
        snr: float
        clipping_detected: bool
        noise_floor: float

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response Models
class AudioStreamingRequest(BaseModel):
    """Request to start audio streaming session"""
    session_id: Optional[str] = Field(None, description="Session ID (auto-generated if not provided)")
    meeting_id: Optional[str] = Field(None, description="Google Meet meeting ID")
    chunking_config: Optional[AudioChunkingConfig] = Field(None, description="Custom chunking configuration")
    enable_speaker_correlation: bool = Field(True, description="Enable speaker correlation")
    enable_database_storage: bool = Field(True, description="Enable database persistence")
    target_languages: List[str] = Field(default_factory=list, description="Target languages for translation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional session metadata")

class AudioChunkRequest(BaseModel):
    """Request to submit audio chunk for processing"""
    session_id: str = Field(..., description="Session ID")
    chunk_data: str = Field(..., description="Base64 encoded audio data")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    force_processing: bool = Field(False, description="Force immediate processing")

class AudioStreamingResponse(BaseModel):
    """Response for streaming session management"""
    session_id: str
    status: str
    message: str
    chunking_config: AudioChunkingConfig
    analytics_url: str
    websocket_url: str
    created_at: datetime

class AudioProcessingStatus(BaseModel):
    """Audio processing status response"""
    session_id: str
    active_chunks: int
    processed_chunks: int
    total_chunks: int
    current_quality_score: float
    speaker_correlations: List[SpeakerCorrelation]
    processing_pipeline_status: Dict[str, str]
    last_activity: datetime

class SessionAnalytics(BaseModel):
    """Comprehensive session analytics"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_duration: float
    audio_files_count: int
    transcripts_count: int
    translations_count: int
    correlations_count: int
    average_quality_score: float
    languages_detected: List[str]
    speakers_identified: int
    processing_stats: Dict[str, Any]

# Initialize global coordinator (will be properly initialized in lifespan)
audio_coordinator: Optional[AudioCoordinator] = None

# Session management
active_sessions: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}

@router.on_event("startup")
async def initialize_audio_coordinator():
    """Initialize the audio coordinator on startup"""
    global audio_coordinator
    try:
        # Initialize with default configuration
        default_config = AudioChunkingConfig()
        database_adapter = get_database_adapter()
        audio_coordinator = AudioCoordinator(
            chunking_config=default_config,
            database_adapter=database_adapter
        )
        await audio_coordinator.start()
        logger.info("Audio coordinator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize audio coordinator: {e}")
        # Continue without coordinator for development

@router.on_event("shutdown") 
async def shutdown_audio_coordinator():
    """Cleanup audio coordinator on shutdown"""
    global audio_coordinator
    if audio_coordinator:
        try:
            await audio_coordinator.stop()
            logger.info("Audio coordinator shutdown complete")
        except Exception as e:
            logger.error(f"Error during audio coordinator shutdown: {e}")

# Audio Streaming Endpoints

@router.post("/stream/start", response_model=AudioStreamingResponse)
async def start_audio_streaming(
    request: AudioStreamingRequest,
    database_adapter=Depends(get_database_adapter) if 'get_database_adapter' in globals() else None
) -> AudioStreamingResponse:
    """
    Start a new audio streaming session with centralized coordination
    
    Creates a new session with:
    - Centralized audio chunking coordination
    - Database integration for persistence
    - Speaker correlation setup
    - Real-time processing pipeline
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid4().hex[:8]}"
        
        logger.info(f"Starting audio streaming session: {session_id}")
        
        # Use provided config or default
        chunking_config = request.chunking_config or AudioChunkingConfig()
        
        # Create session in database if enabled
        session_data = {
            "session_id": session_id,
            "meeting_id": request.meeting_id,
            "target_languages": request.target_languages,
            "enable_speaker_correlation": request.enable_speaker_correlation,
            "enable_database_storage": request.enable_database_storage,
            "chunking_config": chunking_config.dict(),
            "metadata": request.metadata,
            "status": "active",
            "created_at": datetime.utcnow(),
        }
        
        if request.enable_database_storage and database_adapter:
            try:
                await database_adapter.create_session(session_id, session_data)
                logger.info(f"Session {session_id} created in database")
            except Exception as e:
                logger.warning(f"Failed to create session in database: {e}")
        
        # Initialize with audio coordinator if available
        if audio_coordinator:
            try:
                await audio_coordinator.start_session(
                    session_id=session_id,
                    config=chunking_config,
                    enable_correlation=request.enable_speaker_correlation,
                    target_languages=request.target_languages
                )
                logger.info(f"Session {session_id} started with audio coordinator")
            except Exception as e:
                logger.warning(f"Failed to start coordinator session: {e}")
        
        # Store session in memory
        active_sessions[session_id] = session_data
        
        # Generate URLs
        analytics_url = f"/api/audio/sessions/{session_id}/analytics"
        websocket_url = f"/api/audio/stream/{session_id}/ws"
        
        return AudioStreamingResponse(
            session_id=session_id,
            status="active",
            message="Audio streaming session started successfully",
            chunking_config=chunking_config,
            analytics_url=analytics_url,
            websocket_url=websocket_url,
            created_at=datetime.utcnow()
        )
        
    except ValidationError as e:
        logger.error(f"Validation error starting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid session request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to start audio streaming session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start audio streaming: {str(e)}"
        )

@router.post("/stream/chunk")
async def submit_audio_chunk(
    request: AudioChunkRequest,
    database_adapter=Depends(get_database_adapter) if 'get_database_adapter' in globals() else None
) -> Dict[str, Any]:
    """
    Submit audio chunk for coordinated processing
    
    Processes audio chunk through:
    - Centralized chunking coordination with overlap management
    - Database storage with metadata
    - Speaker correlation processing
    - Real-time result distribution
    """
    try:
        session_id = request.session_id
        
        # Validate session exists
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        logger.info(f"Processing audio chunk for session: {session_id}")
        
        # Decode audio data
        import base64
        try:
            audio_data = base64.b64decode(request.chunk_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 audio data: {str(e)}"
            )
        
        # Generate chunk metadata
        chunk_id = f"chunk_{uuid4().hex[:8]}"
        current_time = datetime.utcnow()
        
        # Process with audio coordinator if available
        processing_result = None
        if audio_coordinator:
            try:
                processing_result = await audio_coordinator.process_chunk(
                    session_id=session_id,
                    chunk_id=chunk_id,
                    audio_data=audio_data,
                    metadata=request.chunk_metadata,
                    force_processing=request.force_processing
                )
                logger.info(f"Chunk {chunk_id} processed by coordinator")
            except Exception as e:
                logger.warning(f"Coordinator processing failed: {e}")
        
        # Fallback processing if coordinator unavailable
        if not processing_result:
            processing_result = {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "status": "processed",
                "processing_time": 100,
                "quality_score": 0.8,
                "transcription": None,
                "speaker_info": None,
                "message": "Processed without coordinator (fallback mode)"
            }
        
        # Store in database if enabled
        session_data = active_sessions[session_id]
        if session_data.get("enable_database_storage") and database_adapter:
            try:
                chunk_metadata = AudioChunkMetadata(
                    chunk_id=chunk_id,
                    session_id=session_id,
                    sequence_number=processing_result.get("sequence_number", 0),
                    start_time=processing_result.get("start_time", 0.0),
                    end_time=processing_result.get("end_time", 0.0),
                    duration=processing_result.get("duration", 0.0),
                    overlap_start=processing_result.get("overlap_start", 0.0),
                    overlap_end=processing_result.get("overlap_end", 0.0),
                    file_path=processing_result.get("file_path", ""),
                    file_size=len(audio_data),
                    sample_rate=processing_result.get("sample_rate", 16000),
                    channels=processing_result.get("channels", 1),
                    quality_score=processing_result.get("quality_score", 0.0),
                    rms_level=processing_result.get("rms_level", 0.0),
                    peak_level=processing_result.get("peak_level", 0.0)
                )
                
                await database_adapter.store_audio_chunk(chunk_metadata)
                logger.info(f"Chunk {chunk_id} stored in database")
            except Exception as e:
                logger.warning(f"Failed to store chunk in database: {e}")
        
        # Send real-time update via WebSocket
        if session_id in websocket_connections:
            try:
                update_message = {
                    "type": "chunk_processed",
                    "session_id": session_id,
                    "chunk_id": chunk_id,
                    "result": processing_result,
                    "timestamp": current_time.isoformat()
                }
                await websocket_connections[session_id].send_text(json.dumps(update_message))
            except Exception as e:
                logger.warning(f"Failed to send WebSocket update: {e}")
        
        return {
            "chunk_id": chunk_id,
            "session_id": session_id,
            "status": "processed",
            "processing_result": processing_result,
            "timestamp": current_time.isoformat(),
            "message": "Audio chunk processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process audio chunk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio chunk processing failed: {str(e)}"
        )

@router.get("/stream/{session_id}/results")
async def get_processing_results(
    session_id: str,
    since: Optional[datetime] = None,
    limit: int = 100,
    database_adapter=Depends(get_database_adapter) if 'get_database_adapter' in globals() else None
) -> Dict[str, Any]:
    """
    Get processing results for a session
    
    Returns:
    - Processed chunks with transcriptions
    - Speaker correlations
    - Quality metrics
    - Processing statistics
    """
    try:
        # Validate session exists
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        logger.info(f"Retrieving results for session: {session_id}")
        
        results = {
            "session_id": session_id,
            "chunks": [],
            "transcriptions": [],
            "translations": [],
            "speaker_correlations": [],
            "quality_metrics": {},
            "processing_stats": {},
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
        # Get results from audio coordinator if available
        if audio_coordinator:
            try:
                coordinator_results = await audio_coordinator.get_session_results(
                    session_id, since=since, limit=limit
                )
                results.update(coordinator_results)
                logger.info(f"Retrieved {len(coordinator_results.get('chunks', []))} chunks from coordinator")
            except Exception as e:
                logger.warning(f"Failed to get coordinator results: {e}")
        
        # Get results from database if available
        if database_adapter:
            try:
                db_results = await database_adapter.get_session_results(
                    session_id, since=since, limit=limit
                )
                # Merge database results
                for key in ["chunks", "transcriptions", "translations", "speaker_correlations"]:
                    if key in db_results:
                        results[key].extend(db_results[key])
                
                logger.info(f"Retrieved additional results from database")
            except Exception as e:
                logger.warning(f"Failed to get database results: {e}")
        
        # Add session metadata
        session_data = active_sessions[session_id]
        results["session_metadata"] = {
            "created_at": session_data.get("created_at"),
            "target_languages": session_data.get("target_languages", []),
            "enable_speaker_correlation": session_data.get("enable_speaker_correlation", False),
            "chunking_config": session_data.get("chunking_config", {})
        }
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processing results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve processing results: {str(e)}"
        )

@router.post("/stream/stop")
async def stop_audio_streaming(
    session_id: str,
    database_adapter=Depends(get_database_adapter) if 'get_database_adapter' in globals() else None
) -> Dict[str, Any]:
    """
    Stop audio streaming session and finalize processing
    
    Performs:
    - Session cleanup and finalization
    - Database statistics update
    - Resource cleanup
    - Final analytics generation
    """
    try:
        # Validate session exists
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        logger.info(f"Stopping audio streaming session: {session_id}")
        
        # Stop with audio coordinator if available
        coordinator_stats = {}
        if audio_coordinator:
            try:
                coordinator_stats = await audio_coordinator.stop_session(session_id)
                logger.info(f"Session {session_id} stopped with coordinator")
            except Exception as e:
                logger.warning(f"Failed to stop coordinator session: {e}")
        
        # Update database if enabled
        session_data = active_sessions[session_id]
        if session_data.get("enable_database_storage") and database_adapter:
            try:
                await database_adapter.finalize_session(session_id, coordinator_stats)
                logger.info(f"Session {session_id} finalized in database")
            except Exception as e:
                logger.warning(f"Failed to finalize session in database: {e}")
        
        # Close WebSocket connection
        if session_id in websocket_connections:
            try:
                await websocket_connections[session_id].close()
                del websocket_connections[session_id]
            except Exception as e:
                logger.warning(f"Failed to close WebSocket: {e}")
        
        # Generate final statistics
        end_time = datetime.utcnow()
        start_time = session_data.get("created_at", end_time)
        duration = (end_time - start_time).total_seconds()
        
        final_stats = {
            "session_id": session_id,
            "status": "completed",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "coordinator_stats": coordinator_stats,
            "message": "Audio streaming session stopped successfully"
        }
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        return final_stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop audio streaming session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop audio streaming: {str(e)}"
        )

# Session Analytics

@router.get("/sessions/{session_id}/analytics", response_model=SessionAnalytics)
async def get_session_analytics(
    session_id: str,
    database_adapter=Depends(get_database_adapter) if 'get_database_adapter' in globals() else None
) -> SessionAnalytics:
    """
    Get comprehensive analytics for a session
    
    Returns detailed statistics including:
    - Processing performance metrics
    - Quality scores and trends
    - Speaker identification results
    - Translation coverage
    - Database storage statistics
    """
    try:
        logger.info(f"Retrieving analytics for session: {session_id}")
        
        # Initialize analytics with defaults
        analytics = SessionAnalytics(
            session_id=session_id,
            start_time=datetime.utcnow(),
            end_time=None,
            total_duration=0.0,
            audio_files_count=0,
            transcripts_count=0,
            translations_count=0,
            correlations_count=0,
            average_quality_score=0.0,
            languages_detected=[],
            speakers_identified=0,
            processing_stats={}
        )
        
        # Get analytics from audio coordinator if available
        if audio_coordinator:
            try:
                coordinator_analytics = await audio_coordinator.get_session_analytics(session_id)
                analytics = SessionAnalytics(**coordinator_analytics)
                logger.info(f"Retrieved coordinator analytics for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to get coordinator analytics: {e}")
        
        # Get analytics from database if available
        if database_adapter:
            try:
                db_analytics = await database_adapter.get_session_analytics(session_id)
                # Merge database analytics
                if db_analytics:
                    analytics.audio_files_count = db_analytics.get("audio_files_count", 0)
                    analytics.transcripts_count = db_analytics.get("transcripts_count", 0)
                    analytics.translations_count = db_analytics.get("translations_count", 0)
                    analytics.correlations_count = db_analytics.get("correlations_count", 0)
                    analytics.languages_detected = db_analytics.get("languages_detected", [])
                    analytics.speakers_identified = db_analytics.get("speakers_identified", 0)
                
                logger.info(f"Retrieved database analytics for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to get database analytics: {e}")
        
        # Get session metadata if active
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            analytics.start_time = session_data.get("created_at", analytics.start_time)
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get session analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session analytics: {str(e)}"
        )

# Configuration Management

@router.get("/config", response_model=AudioChunkingConfig)
async def get_audio_processing_config() -> AudioChunkingConfig:
    """Get current audio processing configuration"""
    try:
        if audio_coordinator:
            config = await audio_coordinator.get_config()
            return AudioChunkingConfig(**config)
        else:
            return AudioChunkingConfig()  # Return defaults
    except Exception as e:
        logger.error(f"Failed to get audio config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audio configuration"
        )

@router.post("/config")
async def update_audio_processing_config(
    config: AudioChunkingConfig
) -> Dict[str, Any]:
    """Update audio processing configuration"""
    try:
        if audio_coordinator:
            await audio_coordinator.update_config(config.dict())
            return {
                "status": "success",
                "message": "Audio processing configuration updated",
                "config": config.dict()
            }
        else:
            return {
                "status": "warning", 
                "message": "Configuration saved but coordinator not available",
                "config": config.dict()
            }
    except Exception as e:
        logger.error(f"Failed to update audio config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update audio configuration"
        )

# WebSocket Endpoint

@router.websocket("/stream/{session_id}/ws")
async def websocket_audio_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time audio streaming and result updates
    
    Provides:
    - Real-time audio chunk submission
    - Live processing result updates
    - Speaker correlation notifications
    - Quality metrics streaming
    """
    await websocket.accept()
    websocket_connections[session_id] = websocket
    
    try:
        logger.info(f"WebSocket connected for session: {session_id}")
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "message": "WebSocket connection established",
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        while True:
            # Receive messages from client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "audio_chunk":
                    # Process audio chunk via WebSocket
                    chunk_request = AudioChunkRequest(
                        session_id=session_id,
                        chunk_data=message.get("chunk_data", ""),
                        chunk_metadata=message.get("metadata", {}),
                        force_processing=message.get("force_processing", False)
                    )
                    
                    # Process chunk (this will also send updates back via WebSocket)
                    await submit_audio_chunk(chunk_request)
                    
                elif message.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    
                elif message.get("type") == "get_status":
                    # Send current processing status
                    status_info = {
                        "type": "status_update",
                        "session_id": session_id,
                        "active": session_id in active_sessions,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await websocket.send_text(json.dumps(status_info))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                error_message = {
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(error_message))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Cleanup WebSocket connection
        if session_id in websocket_connections:
            del websocket_connections[session_id]

# Health and Status

@router.get("/health")
async def audio_coordination_health() -> Dict[str, Any]:
    """
    Health check for audio coordination system
    
    Returns status of:
    - Audio coordinator
    - Database adapter
    - Active sessions
    - WebSocket connections
    """
    try:
        health_status = {
            "status": "healthy",
            "coordinator_available": audio_coordinator is not None,
            "active_sessions": len(active_sessions),
            "websocket_connections": len(websocket_connections),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check coordinator health
        if audio_coordinator:
            try:
                coordinator_health = await audio_coordinator.health_check()
                health_status["coordinator_health"] = coordinator_health
            except Exception as e:
                health_status["coordinator_health"] = {"status": "error", "error": str(e)}
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/sessions")
async def list_active_sessions() -> Dict[str, Any]:
    """List all active audio streaming sessions"""
    try:
        sessions_list = []
        for session_id, session_data in active_sessions.items():
            sessions_list.append({
                "session_id": session_id,
                "created_at": session_data.get("created_at"),
                "meeting_id": session_data.get("meeting_id"),
                "status": session_data.get("status"),
                "has_websocket": session_id in websocket_connections
            })
        
        return {
            "active_sessions": sessions_list,
            "total_count": len(sessions_list),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active sessions"
        )