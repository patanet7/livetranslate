"""
FastAPI routers for different API endpoints

Modern async/await API endpoints with comprehensive validation and documentation
"""

from .audio import router as audio_router
from .audio import coordination_router as audio_coordination_router
from .audio import websocket_router as websocket_audio_router
from .bot import router as bot_router
from .bot import docker_management_router as bot_management_router
from .bot import docker_callbacks_router as bot_callbacks_router
from .websocket import router as websocket_router
from .system import router as system_router
from .settings import router as settings_router
from .translation import router as translation_router
from .fireflies import router as fireflies_router

__all__ = [
    "audio_router",
    "audio_coordination_router",
    "websocket_audio_router",
    "bot_router",
    "bot_management_router",
    "bot_callbacks_router",
    "websocket_router",
    "system_router",
    "settings_router",
    "translation_router",
    "fireflies_router",
]
