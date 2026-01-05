"""
Bot Lifecycle Management Router

Bot lifecycle endpoints including:
- Bot spawning (/spawn)
- Bot listing (/)
- Active bots (/active)
- Bot details (/{bot_id})
- Bot termination (/{bot_id}/terminate)
"""

from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, status

from ._shared import (
    create_bot_router,
    BotResponse,
    BotListResponse,
    logger,
    get_error_response,
    validate_bot_exists,
)
from models.bot import BotSpawnRequest
from dependencies import get_bot_manager

# Create router for bot lifecycle management
router = create_bot_router()


@router.post("/spawn", response_model=BotResponse)
async def spawn_bot(
    request: BotSpawnRequest,
    bot_manager=Depends(get_bot_manager),
    event_publisher=Depends(get_event_publisher),
) -> BotResponse:
    """
    Spawn a new bot instance for a meeting

    - **meeting_id**: Unique identifier for the meeting
    - **meeting_url**: URL of the meeting to join
    - **bot_type**: Type of bot to spawn (default: google_meet)
    - **config**: Bot configuration parameters
    """
    try:
        logger.info(f"Spawning bot for meeting: {request.meeting_id}")

        # Validate request
        if not request.meeting_id:
            raise get_error_response(
                status.HTTP_400_BAD_REQUEST,
                "Meeting ID is required",
                {"field": "meeting_id"},
            )

        if not request.meeting_url:
            raise get_error_response(
                status.HTTP_400_BAD_REQUEST,
                "Meeting URL is required",
                {"field": "meeting_url"},
            )

        # Check if bot already exists for this meeting
        existing_bot = None  # TODO: Implement get_bot_by_meeting_id if needed
        if existing_bot:
            logger.warning(f"Bot already exists for meeting {request.meeting_id}")
            return BotResponse(
                bot_id=existing_bot.bot_id,
                status="already_exists",
                message=f"Bot already exists for meeting {request.meeting_id}",
                meeting_id=request.meeting_id,
                created_at=existing_bot.created_at,
            )

        # Spawn new bot
        bot_instance = await bot_manager.spawn_bot(
            meeting_id=request.meeting_id,
            meeting_url=request.meeting_url,
            bot_type=request.bot_type or "google_meet",
            config=request.config or {},
        )

        logger.info(f"Bot spawned successfully: {bot_instance.bot_id}")

        await event_publisher.publish(
            alias="bot_control",
            event_type="BotRequested",
            payload={
                "bot_id": bot_instance.bot_id,
                "meeting_id": request.meeting_id,
                "meeting_url": request.meeting_url,
                "bot_type": bot_instance.bot_type,
                "config": request.config or {},
            },
            metadata={"endpoint": "/bots/spawn"},
        )

        return BotResponse(
            bot_id=bot_instance.bot_id,
            status="spawned",
            message="Bot spawned successfully",
            meeting_id=request.meeting_id,
            meeting_url=request.meeting_url,
            bot_type=bot_instance.bot_type,
            created_at=bot_instance.created_at,
            config=bot_instance.config,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to spawn bot: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to spawn bot: {str(e)}",
            {"meeting_id": request.meeting_id, "error": str(e)},
        )


@router.get("/")
async def list_bots(
    status_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    bot_manager=Depends(get_bot_manager),
) -> BotListResponse:
    """
    List all bots with optional filtering

    - **status_filter**: Filter by bot status (active, inactive, terminated)
    - **limit**: Maximum number of bots to return (default: 100)
    - **offset**: Number of bots to skip (default: 0)
    """
    try:
        logger.info(f"Listing bots with filter: {status_filter}")

        # Get all bots
        all_bots = await bot_manager.list_bots()

        # Apply status filter if provided
        if status_filter:
            filtered_bots = [
                bot for bot in all_bots if bot.status.lower() == status_filter.lower()
            ]
        else:
            filtered_bots = all_bots

        # Apply pagination
        paginated_bots = filtered_bots[offset : offset + limit]

        # Convert to response format
        bot_responses = []
        for bot in paginated_bots:
            bot_responses.append(
                BotResponse(
                    bot_id=bot.bot_id,
                    status=bot.status,
                    message=f"Bot {bot.status}",
                    meeting_id=bot.meeting_id,
                    meeting_url=bot.meeting_url,
                    bot_type=bot.bot_type,
                    created_at=bot.created_at,
                    config=bot.config,
                )
            )

        # Calculate statistics
        active_count = len([b for b in all_bots if b.status == "active"])
        inactive_count = len(all_bots) - active_count

        return BotListResponse(
            bots=bot_responses,
            total=len(filtered_bots),
            active=active_count,
            inactive=inactive_count,
        )

    except Exception as e:
        logger.error(f"Failed to list bots: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to list bots: {str(e)}"
        )


@router.get("/active")
async def list_active_bots(bot_manager=Depends(get_bot_manager)) -> BotListResponse:
    """
    List only active bots
    """
    try:
        logger.info("Listing active bots")

        # Get active bots
        active_bots = await bot_manager.get_active_bots()

        # Convert to response format
        bot_responses = []
        for bot in active_bots:
            bot_responses.append(
                BotResponse(
                    bot_id=bot.bot_id,
                    status=bot.status,
                    message="Bot is active",
                    meeting_id=bot.meeting_id,
                    meeting_url=bot.meeting_url,
                    bot_type=bot.bot_type,
                    created_at=bot.created_at,
                    config=bot.config,
                )
            )

        return BotListResponse(
            bots=bot_responses,
            total=len(active_bots),
            active=len(active_bots),
            inactive=0,
        )

    except Exception as e:
        logger.error(f"Failed to list active bots: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to list active bots: {str(e)}",
        )


@router.get("/{bot_id}", response_model=BotResponse)
async def get_bot_details(
    bot_id: str, bot_manager=Depends(get_bot_manager)
) -> BotResponse:
    """
    Get detailed information about a specific bot
    """
    try:
        logger.info(f"Getting details for bot: {bot_id}")

        # Validate and get bot
        bot_instance = await validate_bot_exists(bot_id, bot_manager)

        # Get additional bot details
        bot_stats = await bot_manager.get_bot_stats(bot_id)

        return BotResponse(
            bot_id=bot_instance.bot_id,
            status=bot_instance.status,
            message="Bot details retrieved",
            meeting_id=bot_instance.meeting_id,
            meeting_url=bot_instance.meeting_url,
            bot_type=bot_instance.bot_type,
            created_at=bot_instance.created_at,
            updated_at=bot_instance.updated_at,
            config=bot_instance.config,
            stats=bot_stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot details for {bot_id}: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get bot details: {str(e)}",
            {"bot_id": bot_id},
        )


@router.post("/{bot_id}/terminate")
async def terminate_bot(
    bot_id: str,
    reason: Optional[str] = None,
    bot_manager=Depends(get_bot_manager),
    event_publisher=Depends(get_event_publisher),
) -> Dict[str, Any]:
    """
    Terminate a bot instance

    - **reason**: Optional reason for termination
    """
    try:
        logger.info(f"Terminating bot: {bot_id}")

        # Validate bot exists
        await validate_bot_exists(bot_id, bot_manager)

        # Terminate bot
        termination_result = await bot_manager.terminate_bot(
            bot_id=bot_id, reason=reason or "Manual termination"
        )

        logger.info(f"Bot {bot_id} terminated successfully")

        await event_publisher.publish(
            alias="bot_control",
            event_type="BotStopRequested",
            payload={
                "bot_id": bot_id,
                "reason": reason or "Manual termination",
                "termination_result": termination_result,
            },
            metadata={"endpoint": f"/bots/{bot_id}/terminate"},
        )

        return {
            "bot_id": bot_id,
            "status": "terminated",
            "message": "Bot terminated successfully",
            "reason": reason or "Manual termination",
            "terminated_at": datetime.utcnow().isoformat(),
            "termination_result": termination_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to terminate bot {bot_id}: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to terminate bot: {str(e)}",
            {"bot_id": bot_id},
        )


@router.get("/{bot_id}/status")
async def get_bot_status(
    bot_id: str, bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """
    Get current status of a bot
    """
    try:
        logger.info(f"Getting status for bot: {bot_id}")

        # Validate and get bot
        bot_instance = await validate_bot_exists(bot_id, bot_manager)

        # Get detailed status
        detailed_status = await bot_manager.get_bot_detailed_status(bot_id)

        return {
            "bot_id": bot_id,
            "status": bot_instance.status,
            "health": detailed_status.get("health", "unknown"),
            "uptime": detailed_status.get("uptime", 0),
            "last_activity": detailed_status.get("last_activity"),
            "error_count": detailed_status.get("error_count", 0),
            "meeting_connected": detailed_status.get("meeting_connected", False),
            "audio_active": detailed_status.get("audio_active", False),
            "processing_active": detailed_status.get("processing_active", False),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bot status for {bot_id}: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get bot status: {str(e)}",
            {"bot_id": bot_id},
        )


@router.post("/{bot_id}/restart")
async def restart_bot(
    bot_id: str, bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
    """
    Restart a bot instance
    """
    try:
        logger.info(f"Restarting bot: {bot_id}")

        # Validate bot exists
        await validate_bot_exists(bot_id, bot_manager)

        # Restart bot
        restart_result = await bot_manager.restart_bot(bot_id)

        logger.info(f"Bot {bot_id} restarted successfully")

        return {
            "bot_id": bot_id,
            "status": "restarted",
            "message": "Bot restarted successfully",
            "restarted_at": datetime.utcnow().isoformat(),
            "restart_result": restart_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart bot {bot_id}: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to restart bot: {str(e)}",
            {"bot_id": bot_id},
        )
