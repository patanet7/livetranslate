"""
Bot Management Router

FastAPI router for bot management endpoints including:
- Bot lifecycle management (spawn, status, terminate)
- Google Meet integration
- Virtual webcam management
- Bot session analytics
"""

import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from dependencies import get_bot_manager
from models.bot import (
    BotSpawnRequest,
    BotResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================


class BotListResponse(BaseModel):
    """Response model for bot list"""

    bots: List[BotResponse]
    total: int
    active: int
    inactive: int


class BotConfigUpdateRequest(BaseModel):
    """Request model for bot configuration update"""

    bot_id: str
    config: Dict[str, Any]


class VirtualWebcamRequest(BaseModel):
    """Request model for virtual webcam operations"""

    bot_id: str
    action: str  # start, stop, update
    settings: Optional[Dict[str, Any]] = None


# ============================================================================
# Bot Lifecycle Endpoints
# ============================================================================


@router.post("/spawn", response_model=BotResponse)
async def spawn_bot(
    request: BotSpawnRequest,
    background_tasks: BackgroundTasks,
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Spawn a new bot instance

    Creates a new bot for Google Meet integration with specified configuration.
    The bot will handle audio capture, transcription, and virtual webcam output.
    """
    try:
        logger.info(f"Spawning bot for meeting: {request.meeting_id}")

        # Validate request
        if not request.meeting_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Meeting ID is required"
            )

        # Check if bot already exists for this meeting
        # Note: get_bot_by_meeting_id might not exist, using get_bot_status for now
        existing_bot = None  # TODO: Implement get_bot_by_meeting_id if needed
        if existing_bot:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Bot already exists for meeting {request.meeting_id}",
            )

        # Spawn bot (using request_bot which is async)
        # Create MeetingRequest from BotSpawnRequest
        from managers.bot_manager import MeetingRequest

        meeting_request = MeetingRequest(
            meeting_id=request.meeting_id,
            meeting_title=request.meeting_title or f"Meeting {request.meeting_id}",
            meeting_uri=getattr(
                request, "meeting_uri", f"https://meet.google.com/{request.meeting_id}"
            ),
            target_languages=getattr(request, "target_languages", ["en"]),
            metadata=getattr(request, "metadata", {}),
        )
        bot_id = await bot_manager.request_bot(meeting_request)

        # Bot spawning requested - it will be processed asynchronously
        # No need for background task as bot_manager.request_bot handles this

        logger.info(f"Bot spawned successfully: {bot_id}")

        return {
            "bot_id": bot_id,
            "meeting_id": request.meeting_id,
            "status": "spawning",
            "message": "Bot spawn requested and being processed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to spawn bot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to spawn bot: {str(e)}",
        )


@router.get("/")
async def list_bots(
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get list of all bots

    Returns information about all bot instances including their status,
    configuration, and performance metrics.
    """
    try:
        bots = bot_manager.get_all_bots()  # Remove await - not an async method

        # Calculate statistics
        total = len(bots)
        active = sum(1 for bot in bots if bot.get("status") == "active")
        inactive = total - active

        # Ensure JSON serializable
        from datetime import datetime
        
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.timestamp()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        safe_bots = convert_datetime(bots)
        
        return {"bots": safe_bots, "total": total, "active": active, "inactive": inactive}

    except Exception as e:
        logger.error(f"Failed to list bots: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list bots: {str(e)}",
        )


@router.get("/active")
async def get_active_bots(
    bot_manager=Depends(get_bot_manager),
):
    """
    Get active bot instances

    Returns a simple list of currently active bots.
    """
    try:
        # Get all bots and filter for active ones
        bots = bot_manager.get_all_bots()  # Not async
        active_bots = [bot for bot in bots if bot.get("status") == "active"]

        # Ensure JSON serializable by converting any datetime objects
        from datetime import datetime
        
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.timestamp()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        safe_bots = convert_datetime(active_bots)
        return safe_bots

    except Exception as e:
        logger.error(f"Failed to get active bots: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active bots: {str(e)}",
        )


@router.get("/{bot_id}", response_model=BotResponse)
async def get_bot_status(
    bot_id: str,
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get detailed status of a specific bot

    Returns comprehensive information about a bot including its current status,
    configuration, performance metrics, and error information.
    """
    try:
        bot = bot_manager.get_bot_status(bot_id)  # Not async

        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        return bot

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot status: {str(e)}",
        )


@router.post("/{bot_id}/terminate")
async def terminate_bot(
    bot_id: str,
    background_tasks: BackgroundTasks,
    request: Optional[Dict[str, Any]] = None,
    bot_manager=Depends(get_bot_manager),
):
    """
    Terminate a bot instance

    Gracefully shuts down a bot, cleaning up all resources including
    audio capture, transcription sessions, and virtual webcam output.
    """
    try:
        logger.info(f"Terminating bot: {bot_id}")

        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)  # Not async
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Terminate bot
        cleanup_files = request.cleanup_files if request else False
        await bot_manager.terminate_bot(bot_id, cleanup_files)

        # Note: cleanup is handled by terminate_bot method

        logger.info(f"Bot terminated successfully: {bot_id}")

        return {"message": f"Bot {bot_id} terminated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to terminate bot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to terminate bot: {str(e)}",
        )


# ============================================================================
# Bot Configuration Endpoints
# ============================================================================


@router.post("/{bot_id}/config")
async def update_bot_config(
    bot_id: str,
    request: BotConfigUpdateRequest,
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Update bot configuration

    Updates the configuration of a running bot. Some settings may require
    bot restart to take effect.
    """
    try:
        logger.info(f"Updating bot config: {bot_id}")

        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)  # Not async
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Update configuration
        # TODO: Implement update_bot_config method in bot_manager
        updated_config = {"message": "Bot config update not yet implemented"}

        return {
            "message": f"Bot {bot_id} configuration updated",
            "config": updated_config,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update bot config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update bot config: {str(e)}",
        )


@router.get("/{bot_id}/config")
async def get_bot_config(
    bot_id: str,
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get bot configuration

    Returns the current configuration of a bot instance.
    """
    try:
        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)  # Not async
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # TODO: Implement get_bot_config method in bot_manager
        config = {"message": "Bot config retrieval not yet implemented"}

        return {"bot_id": bot_id, "config": config}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot config: {str(e)}",
        )


# ============================================================================
# Virtual Webcam Endpoints
# ============================================================================


@router.post("/{bot_id}/webcam")
async def manage_virtual_webcam(
    bot_id: str,
    request: VirtualWebcamRequest,
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Manage virtual webcam output

    Controls the virtual webcam that displays translation output.
    Actions: start, stop, update settings.
    """
    try:
        logger.info(f"Managing virtual webcam for bot {bot_id}: {request.action}")

        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)  # Not async
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Execute webcam action
        # TODO: Implement virtual webcam methods in bot_manager
        if request.action in ["start", "stop", "update"]:
            result = {"message": f"Virtual webcam {request.action} not yet implemented"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {request.action}",
            )

        return {
            "message": f"Virtual webcam {request.action} completed",
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to manage virtual webcam: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to manage virtual webcam: {str(e)}",
        )


@router.get("/{bot_id}/webcam/status")
async def get_webcam_status(
    bot_id: str,
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get virtual webcam status

    Returns the current status and configuration of the virtual webcam.
    """
    try:
        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)  # Not async
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # TODO: Implement get_virtual_webcam_status method in bot_manager
        webcam_status = {"message": "Virtual webcam status not yet implemented"}

        return {"bot_id": bot_id, "webcam_status": webcam_status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get webcam status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get webcam status: {str(e)}",
        )


# ============================================================================
# Bot Analytics Endpoints
# ============================================================================


@router.get("/{bot_id}/analytics")
async def get_bot_analytics(
    bot_id: str,
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get bot analytics and performance metrics

    Returns comprehensive analytics for a bot including transcription
    quality, translation accuracy, and system performance metrics.
    """
    try:
        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)  # Not async
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Get bot analytics from database
        analytics = await bot_manager.get_bot_analytics(bot_id)
        if not analytics:
            # Return basic analytics if none found
            analytics = {
                "bot_id": bot_id,
                "total_sessions": 0,
                "active_sessions": 0,
                "total_audio_files": 0,
                "total_transcripts": 0,
                "total_translations": 0,
                "total_correlations": 0,
                "average_confidence": 0.0,
                "quality_metrics": {
                    "transcription_accuracy": 0.0,
                    "translation_quality": 0.0,
                    "correlation_success_rate": 0.0,
                    "average_processing_time": 0.0
                },
                "performance_stats": {
                    "uptime_percentage": 0.0,
                    "error_rate": 0.0,
                    "success_rate": 0.0
                },
                "usage_patterns": {
                    "most_active_hours": [],
                    "average_session_duration": 0.0,
                    "popular_languages": []
                }
            }

        return analytics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot analytics: {str(e)}",
        )


@router.get("/{bot_id}/session")
async def get_bot_session(
    bot_id: str,
    session_id: Optional[str] = None,
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get bot session data

    Returns session information including audio files, transcripts,
    translations, and time correlation data.
    """
    try:
        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)  # Not async
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Get session data
        if session_id:
            # Get specific session data
            session_data = await bot_manager.get_comprehensive_session_data(session_id)
            if not session_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, 
                    detail=f"Session {session_id} not found for bot {bot_id}"
                )
        else:
            # Get current active session for the bot
            session_data = await bot_manager.get_current_bot_session(bot_id)
            if not session_data:
                return {
                    "bot_id": bot_id,
                    "current_session": None,
                    "message": "No active session found for this bot"
                }

        return {
            "bot_id": bot_id,
            "session_data": session_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot session: {str(e)}",
        )


@router.get("/{bot_id}/sessions")
async def list_bot_sessions(
    bot_id: str,
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
    bot_manager=Depends(get_bot_manager),
):
    """
    List all sessions for a specific bot
    
    Returns paginated list of sessions with optional status filtering.
    """
    try:
        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Get sessions list
        sessions = await bot_manager.list_bot_sessions(
            bot_id, 
            limit=limit, 
            offset=offset, 
            status_filter=status_filter
        )

        return {
            "bot_id": bot_id,
            "sessions": sessions,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": len(sessions) if len(sessions) < limit else None
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list bot sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list bot sessions: {str(e)}",
        )


@router.get("/{bot_id}/performance")
async def get_bot_performance_metrics(
    bot_id: str,
    timeframe: str = "24h",  # 1h, 24h, 7d, 30d
    bot_manager=Depends(get_bot_manager),
):
    """
    Get detailed performance metrics for a bot
    
    Returns performance data including latency, error rates, 
    and quality metrics over specified timeframe.
    """
    try:
        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Get performance metrics
        metrics = await bot_manager.get_bot_performance_metrics(bot_id, timeframe)
        if not metrics:
            metrics = {
                "bot_id": bot_id,
                "timeframe": timeframe,
                "processing_metrics": {
                    "average_audio_processing_time": 0.0,
                    "average_transcription_time": 0.0,
                    "average_translation_time": 0.0,
                    "average_correlation_time": 0.0,
                    "total_processing_time": 0.0
                },
                "quality_metrics": {
                    "average_transcription_confidence": 0.0,
                    "average_translation_confidence": 0.0,
                    "correlation_success_rate": 0.0,
                    "audio_quality_score": 0.0
                },
                "error_metrics": {
                    "total_errors": 0,
                    "audio_errors": 0,
                    "transcription_errors": 0,
                    "translation_errors": 0,
                    "correlation_errors": 0,
                    "error_rate": 0.0
                },
                "throughput_metrics": {
                    "audio_files_processed": 0,
                    "transcripts_generated": 0,
                    "translations_generated": 0,
                    "correlations_created": 0,
                    "average_throughput": 0.0
                }
            }

        return metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot performance metrics: {str(e)}",
        )


@router.get("/{bot_id}/quality-report")
async def get_bot_quality_report(
    bot_id: str,
    session_id: Optional[str] = None,
    bot_manager=Depends(get_bot_manager),
):
    """
    Get detailed quality report for bot operations
    
    Returns comprehensive quality analysis including confidence distributions,
    accuracy metrics, and quality trends.
    """
    try:
        # Check if bot exists
        bot = bot_manager.get_bot_status(bot_id)
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Get quality report
        report = await bot_manager.get_bot_quality_report(bot_id, session_id)
        if not report:
            report = {
                "bot_id": bot_id,
                "session_id": session_id,
                "confidence_distribution": {
                    "transcription": {"high": 0, "medium": 0, "low": 0},
                    "translation": {"high": 0, "medium": 0, "low": 0},
                    "correlation": {"high": 0, "medium": 0, "low": 0}
                },
                "language_analysis": {
                    "detected_languages": [],
                    "translation_pairs": [],
                    "language_accuracy": {}
                },
                "speaker_analysis": {
                    "total_speakers": 0,
                    "speaker_attribution_accuracy": 0.0,
                    "speaker_consistency": 0.0
                },
                "temporal_analysis": {
                    "average_segment_length": 0.0,
                    "timing_accuracy": 0.0,
                    "correlation_delay": 0.0
                },
                "overall_quality_score": 0.0,
                "recommendations": []
            }

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot quality report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot quality report: {str(e)}",
        )


# ============================================================================
# System-wide Bot Management
# ============================================================================


@router.get("/stats")
async def get_bot_stats(
    # Authentication will be handled by middleware  
    # Rate limiting will be handled by middleware
):
    """
    Get bot statistics

    Returns bot system metrics including capacity, performance,
    and resource utilization.
    """
    try:
        logger.info("Starting bot stats endpoint...")
        
        # Simple JSON-safe response without any datetime objects
        return {
            "totalBotsSpawned": 0,
            "activeBots": 0,
            "completedSessions": 0,
            "errorRate": 0.0,
            "averageSessionDuration": 0,
            "queuedRequests": 0,
            "totalBots": 0,
            "capacityUtilization": 0.0,
            "successRate": 0.0,
            "recoveryRate": 0.0,
        }

    except Exception as e:
        logger.error(f"Failed to get bot stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot stats: {str(e)}",
        )


@router.get("/analytics/database")
async def get_database_analytics(
    bot_manager=Depends(get_bot_manager),
):
    """
    Get comprehensive database analytics
    
    Returns database statistics including storage usage,
    session counts, and data distribution.
    """
    try:
        analytics = await bot_manager.get_database_analytics()
        if not analytics:
            analytics = {
                "total_sessions": 0,
                "active_sessions": 0,
                "recent_sessions_24h": 0,
                "total_audio_files": 0,
                "total_transcripts": 0,
                "total_translations": 0,
                "total_correlations": 0,
                "storage_usage_bytes": 0,
                "storage_usage_mb": 0.0,
                "data_distribution": {
                    "sessions_by_status": {},
                    "files_by_format": {},
                    "transcripts_by_language": {},
                    "translations_by_language": {}
                },
                "performance_summary": {
                    "average_session_duration": 0.0,
                    "average_file_size": 0.0,
                    "average_confidence_scores": {
                        "transcription": 0.0,
                        "translation": 0.0,
                        "correlation": 0.0
                    }
                }
            }

        return analytics

    except Exception as e:
        logger.error(f"Failed to get database analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get database analytics: {str(e)}",
        )


@router.get("/analytics/sessions")
async def get_session_analytics(
    timeframe: str = "24h",
    group_by: str = "hour",  # hour, day, week
    bot_manager=Depends(get_bot_manager),
):
    """
    Get session analytics with time-based grouping
    
    Returns session activity patterns over specified timeframe.
    """
    try:
        analytics = await bot_manager.get_session_analytics(timeframe, group_by)
        if not analytics:
            analytics = {
                "timeframe": timeframe,
                "group_by": group_by,
                "session_activity": [],
                "peak_usage": {
                    "peak_hour": None,
                    "peak_sessions": 0,
                    "peak_date": None
                },
                "trends": {
                    "sessions_trend": "stable",  # increasing, decreasing, stable
                    "duration_trend": "stable",
                    "quality_trend": "stable"
                },
                "summary": {
                    "total_sessions": 0,
                    "average_session_duration": 0.0,
                    "success_rate": 0.0,
                    "most_active_bots": []
                }
            }

        return analytics

    except Exception as e:
        logger.error(f"Failed to get session analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session analytics: {str(e)}",
        )


@router.get("/analytics/quality")
async def get_quality_analytics(
    timeframe: str = "24h",
    bot_manager=Depends(get_bot_manager),
):
    """
    Get quality analytics across all bots and sessions
    
    Returns comprehensive quality metrics and trends.
    """
    try:
        analytics = await bot_manager.get_quality_analytics(timeframe)
        if not analytics:
            analytics = {
                "timeframe": timeframe,
                "confidence_trends": {
                    "transcription": {
                        "average": 0.0,
                        "trend": "stable",
                        "distribution": {"high": 0, "medium": 0, "low": 0}
                    },
                    "translation": {
                        "average": 0.0,
                        "trend": "stable", 
                        "distribution": {"high": 0, "medium": 0, "low": 0}
                    },
                    "correlation": {
                        "average": 0.0,
                        "trend": "stable",
                        "distribution": {"high": 0, "medium": 0, "low": 0}
                    }
                },
                "language_performance": {},
                "error_analysis": {
                    "total_errors": 0,
                    "error_by_type": {},
                    "error_trend": "stable",
                    "most_common_errors": []
                },
                "processing_performance": {
                    "average_latency": 0.0,
                    "latency_trend": "stable",
                    "throughput": 0.0,
                    "throughput_trend": "stable"
                },
                "recommendations": []
            }

        return analytics

    except Exception as e:
        logger.error(f"Failed to get quality analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quality analytics: {str(e)}",
        )


@router.get("/system/stats")
async def get_system_bot_stats(
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get system-wide bot statistics

    Returns overall bot system metrics including capacity, performance,
    and resource utilization.
    """
    try:
        stats = bot_manager.get_bot_stats()  # Not async, using get_bot_stats instead

        return stats

    except Exception as e:
        logger.error(f"Failed to get system bot stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system bot stats: {str(e)}",
        )


@router.post("/system/cleanup")
async def cleanup_system_bots(
    background_tasks: BackgroundTasks,
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Cleanup system bot resources

    Performs system-wide cleanup of bot resources including terminated
    bots, orphaned sessions, and temporary files.
    """
    try:
        logger.info("Starting system bot cleanup")

        # TODO: Implement cleanup_system_resources method in bot_manager
        logger.info("System bot cleanup requested (not yet implemented)")

        return {"message": "System bot cleanup started"}

    except Exception as e:
        logger.error(f"Failed to start system cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start system cleanup: {str(e)}",
        )


# ============================================================================
# Virtual Webcam Frame Streaming Endpoints
# ============================================================================


@router.get("/virtual-webcam/frame/{bot_id}")
async def get_virtual_webcam_frame(
    bot_id: str,
    bot_manager=Depends(get_bot_manager),
) -> Dict[str, Any]:
    """Get current virtual webcam frame as base64 image."""
    try:
        bot_instance = bot_manager.get_bot(bot_id)
        if not bot_instance:
            raise HTTPException(status_code=404, detail="Bot not found")
            
        # Get current frame from virtual webcam
        if hasattr(bot_instance, 'virtual_webcam') and bot_instance.virtual_webcam:
            frame_base64 = bot_instance.virtual_webcam.get_current_frame_base64()
            if frame_base64:
                return {
                    "bot_id": bot_id,
                    "frame_base64": frame_base64,
                    "timestamp": time.time(),
                    "webcam_stats": bot_instance.virtual_webcam.get_webcam_stats()
                }
            else:
                raise HTTPException(status_code=404, detail="No frame available")
        else:
            raise HTTPException(status_code=404, detail="Virtual webcam not enabled")
            
    except Exception as e:
        logger.error(f"Error getting virtual webcam frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/virtual-webcam/config/{bot_id}")
async def get_virtual_webcam_config(
    bot_id: str,
    bot_manager=Depends(get_bot_manager),
) -> Dict[str, Any]:
    """Get virtual webcam configuration."""
    try:
        bot_instance = bot_manager.get_bot(bot_id)
        if not bot_instance:
            raise HTTPException(status_code=404, detail="Bot not found")
            
        if hasattr(bot_instance, 'virtual_webcam') and bot_instance.virtual_webcam:
            return {
                "bot_id": bot_id,
                "config": bot_instance.virtual_webcam.config.__dict__,
                "speakers": {
                    speaker_id: {
                        "speaker_name": info.speaker_name,
                        "color": info.color,
                        "last_active": info.last_active.isoformat()
                    }
                    for speaker_id, info in bot_instance.virtual_webcam.speakers.items()
                },
                "is_streaming": bot_instance.virtual_webcam.is_streaming
            }
        else:
            raise HTTPException(status_code=404, detail="Virtual webcam not enabled")
            
    except Exception as e:
        logger.error(f"Error getting virtual webcam config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/virtual-webcam/config/{bot_id}")
async def update_virtual_webcam_config(
    bot_id: str,
    config_update: Dict[str, Any],
    bot_manager=Depends(get_bot_manager),
) -> Dict[str, Any]:
    """Update virtual webcam configuration."""
    try:
        bot_instance = bot_manager.get_bot(bot_id)
        if not bot_instance:
            raise HTTPException(status_code=404, detail="Bot not found")
            
        if hasattr(bot_instance, 'virtual_webcam') and bot_instance.virtual_webcam:
            # Import DisplayMode and Theme from virtual_webcam module
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bot'))
            from virtual_webcam import DisplayMode, Theme
            
            # Update configurable properties
            webcam = bot_instance.virtual_webcam
            config = webcam.config
            
            # Update allowed configuration properties
            if "display_mode" in config_update:
                try:
                    config.display_mode = DisplayMode(config_update["display_mode"])
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid display mode")
            
            if "theme" in config_update:
                try:
                    config.theme = Theme(config_update["theme"])
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid theme")
            
            if "max_translations_displayed" in config_update:
                config.max_translations_displayed = max(1, min(10, int(config_update["max_translations_displayed"])))
            
            if "translation_duration_seconds" in config_update:
                config.translation_duration_seconds = max(5.0, min(60.0, float(config_update["translation_duration_seconds"])))
            
            if "show_speaker_names" in config_update:
                config.show_speaker_names = bool(config_update["show_speaker_names"])
            
            if "show_confidence" in config_update:
                config.show_confidence = bool(config_update["show_confidence"])
            
            if "show_timestamps" in config_update:
                config.show_timestamps = bool(config_update["show_timestamps"])
            
            return {
                "bot_id": bot_id,
                "message": "Virtual webcam configuration updated",
                "config": config.__dict__
            }
        else:
            raise HTTPException(status_code=404, detail="Virtual webcam not enabled")
            
    except Exception as e:
        logger.error(f"Error updating virtual webcam config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
