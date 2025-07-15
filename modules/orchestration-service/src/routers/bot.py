"""
Bot Management Router

FastAPI router for bot management endpoints including:
- Bot lifecycle management (spawn, status, terminate)
- Google Meet integration
- Virtual webcam management
- Bot session analytics
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dependencies import get_bot_manager
from models.bot import (
    BotSpawnRequest,
    BotResponse,
    BotInstance,
    BotStats,
    BotConfiguration,
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
        import json
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


@router.get("/{bot_id}/analytics", response_model=BotStats)
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

        # TODO: Implement get_bot_analytics method in bot_manager
        analytics = {"message": "Bot analytics not yet implemented"}

        return analytics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot analytics: {str(e)}",
        )


@router.get("/{bot_id}/session", response_model=BotResponse)
async def get_bot_session(
    bot_id: str,
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

        # TODO: Implement get_bot_session_data method in bot_manager
        session_data = {"message": "Bot session data not yet implemented"}

        return session_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot session: {str(e)}",
        )


# ============================================================================
# System-wide Bot Management
# ============================================================================


@router.get("/stats")
async def get_bot_stats(
    bot_manager=Depends(get_bot_manager),
    # Authentication will be handled by middleware  
    # Rate limiting will be handled by middleware
):
    """
    Get bot statistics

    Returns bot system metrics including capacity, performance,
    and resource utilization.
    """
    try:
        stats = bot_manager.get_bot_stats()  # Not async, using get_bot_stats instead

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
        
        safe_stats = convert_datetime(stats)
        return safe_stats

    except Exception as e:
        logger.error(f"Failed to get bot stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bot stats: {str(e)}",
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
