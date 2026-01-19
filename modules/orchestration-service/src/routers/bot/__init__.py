"""
Bot Management API Router - package entry point.

Combines lifecycle, configuration, analytics, webcam, system, and Docker management sub-routers.

Modules:
- bot_lifecycle: Bot spawning and termination (uses BotManager)
- bot_configuration: Bot configuration management
- bot_analytics: Bot analytics and metrics
- bot_webcam: Virtual webcam control
- bot_system: System-level bot operations
- bot_docker_management: Docker container management (uses DockerBotManager)
- bot_docker_callbacks: Callbacks from Docker containers
"""

from fastapi import APIRouter

from . import (
    bot_analytics,
    bot_configuration,
    bot_docker_callbacks,
    bot_docker_management,
    bot_lifecycle,
    bot_system,
    bot_webcam,
)

router = APIRouter(
    prefix="/bot",
    tags=["bots"],
    responses={
        404: {"description": "Bot not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)

router.include_router(
    bot_lifecycle.router,
    tags=["bot-lifecycle"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad request"},
        409: {"description": "Conflict - bot already exists"},
    },
)

router.include_router(
    bot_configuration.router,
    tags=["bot-configuration"],
    responses={
        200: {"description": "Configuration updated"},
        400: {"description": "Invalid configuration"},
    },
)

router.include_router(
    bot_analytics.router,
    tags=["bot-analytics"],
    responses={
        200: {"description": "Analytics retrieved"},
        404: {"description": "Analytics not found"},
    },
)

router.include_router(
    bot_webcam.router,
    tags=["bot-webcam"],
    responses={
        200: {"description": "Webcam operation successful"},
        404: {"description": "Webcam not available"},
    },
)

router.include_router(
    bot_system.router,
    tags=["bot-system"],
    responses={
        200: {"description": "System operation successful"},
        503: {"description": "System unavailable"},
    },
)

# Docker-based bot management router (separate from main router)
# Mounted at /api/bots by main_fastapi.py
docker_management_router = bot_docker_management.router

# Docker callbacks router (for container status updates)
# Mounted at /api/bots/internal/callback by main_fastapi.py
docker_callbacks_router = bot_docker_callbacks.router

__all__ = [
    "bot_analytics",
    "bot_configuration",
    "bot_docker_callbacks",
    "bot_docker_management",
    "bot_lifecycle",
    "bot_system",
    "bot_webcam",
    "docker_callbacks_router",
    "docker_management_router",
    "router",
]
