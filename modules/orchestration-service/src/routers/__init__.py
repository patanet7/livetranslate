"""
FastAPI routers for different API endpoints

Modern async/await API endpoints with comprehensive validation and documentation
"""

from .audio import (
    coordination_router as audio_coordination_router,
    router as audio_router,
    websocket_router as websocket_audio_router,
)
from .bot import (
    docker_callbacks_router as bot_callbacks_router,
    docker_management_router as bot_management_router,
    router as bot_router,
)
from .fireflies import router as fireflies_router
from .settings import router as settings_router
from .system import router as system_router
from .translation import router as translation_router
from .websocket import router as websocket_router

__all__ = [
    "audio_coordination_router",
    "audio_router",
    "bot_callbacks_router",
    "bot_management_router",
    "bot_router",
    "fireflies_router",
    "settings_router",
    "system_router",
    "translation_router",
    "websocket_audio_router",
    "websocket_router",
]
