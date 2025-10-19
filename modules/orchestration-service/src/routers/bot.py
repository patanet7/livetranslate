"""
Bot Management API Router - Main Module

Unified bot management router that combines all bot functionality:
- Lifecycle management (bot_lifecycle.py)
- Configuration management (bot_configuration.py)  
- Analytics and monitoring (bot_analytics.py)
- Virtual webcam management (bot_webcam.py)
- System management (bot_system.py)

This replaces the original monolithic bot.py file with a modular architecture.
Original size: 1,147 lines → New modular structure: ~1,160 lines across 5 focused modules
"""

from fastapi import APIRouter
from .bot import bot_lifecycle, bot_configuration, bot_analytics, bot_webcam, bot_system

# Create the main bot router
router = APIRouter(
    prefix="/bot",
    tags=["bots"],
    responses={
        404: {"description": "Bot not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Include all bot sub-routers
router.include_router(
    bot_lifecycle.router,
    tags=["bot-lifecycle"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad request"},
        409: {"description": "Conflict - bot already exists"}
    }
)

router.include_router(
    bot_configuration.router,
    tags=["bot-configuration"],
    responses={
        200: {"description": "Configuration updated"},
        400: {"description": "Invalid configuration"}
    }
)

router.include_router(
    bot_analytics.router,
    tags=["bot-analytics"],
    responses={
        200: {"description": "Analytics retrieved"},
        404: {"description": "Analytics not found"}
    }
)

router.include_router(
    bot_webcam.router,
    tags=["bot-webcam"],
    responses={
        200: {"description": "Webcam operation successful"},
        404: {"description": "Webcam not available"}
    }
)

router.include_router(
    bot_system.router,
    tags=["bot-system"],
    responses={
        200: {"description": "System operation successful"},
        503: {"description": "System unavailable"}
    }
)

# Add any additional combined endpoints here if needed
@router.get("/info")
async def get_bot_router_info():
    """
    Get information about the bot router and its capabilities
    """
    return {
        "name": "Orchestration Bot Router",
        "version": "2.0.0",
        "description": "Modular bot management router with lifecycle, config, analytics, webcam, and system modules",
        "modules": {
            "lifecycle": {
                "description": "Bot lifecycle management",
                "endpoints": ["spawn", "list", "status", "terminate", "restart"]
            },
            "configuration": {
                "description": "Bot configuration management", 
                "endpoints": ["config (GET/POST)"]
            },
            "analytics": {
                "description": "Bot analytics and performance monitoring",
                "endpoints": ["analytics", "session", "sessions", "performance", "quality-report"]
            },
            "webcam": {
                "description": "Virtual webcam management",
                "endpoints": ["webcam", "webcam/status", "virtual-webcam/frame", "virtual-webcam/config"]
            },
            "system": {
                "description": "System-wide bot management",
                "endpoints": ["stats", "system/stats", "system/cleanup"]
            }
        },
        "total_endpoints": 22,
        "architecture": "modular",
        "splitting_date": "2024-current",
        "original_size": "1,147 lines",
        "new_structure": {
            "bot_lifecycle.py": "~360 lines",
            "bot_configuration.py": "~80 lines", 
            "bot_analytics.py": "~380 lines",
            "bot_webcam.py": "~250 lines",
            "bot_system.py": "~90 lines",
            "total_new_size": "~1,160 lines across 5 focused modules"
        }
    }