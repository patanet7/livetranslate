"""
Bot Management API Router - package entry point.

Combines lifecycle, configuration, analytics, webcam, and system sub-routers.
"""

from fastapi import APIRouter

from . import (
    bot_analytics,
    bot_configuration,
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

__all__ = [
    "router",
    "bot_analytics",
    "bot_configuration",
    "bot_lifecycle",
    "bot_system",
    "bot_webcam",
]
