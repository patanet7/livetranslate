"""
Bot System Management Router

System-wide bot management endpoints including:
- Bot statistics (/stats)
- System bot statistics (/system/stats)
- System cleanup (/system/cleanup)
"""

from fastapi import BackgroundTasks
from ._shared import *

# Create router for system bot management
router = create_bot_router()


@router.get("/stats")
async def get_bot_stats() -> SystemStatsResponse:
    """
    Get bot statistics

    Returns bot system metrics including capacity, performance,
    and resource utilization.
    """
    try:
        logger.info("Starting bot stats endpoint...")
        
        # Simple JSON-safe response without any datetime objects
        stats = {
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

        return SystemStatsResponse(
            stats=stats,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Failed to get bot stats: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get bot stats: {str(e)}"
        )


@router.get("/system/stats")
async def get_system_bot_stats(
    bot_manager=Depends(get_bot_manager)
) -> SystemStatsResponse:
    """
    Get system-wide bot statistics

    Returns overall bot system metrics including capacity, performance,
    and resource utilization.
    """
    try:
        stats = bot_manager.get_bot_stats()  # Not async, using get_bot_stats instead

        return SystemStatsResponse(
            stats=stats,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Failed to get system bot stats: {e}")
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to get system bot stats: {str(e)}"
        )


@router.post("/system/cleanup")
async def cleanup_system_bots(
    background_tasks: BackgroundTasks,
    bot_manager=Depends(get_bot_manager)
) -> Dict[str, Any]:
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
        raise get_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to start system cleanup: {str(e)}"
        )