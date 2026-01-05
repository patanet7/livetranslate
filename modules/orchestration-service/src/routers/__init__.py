"""
FastAPI routers for different API endpoints

Modern async/await API endpoints with comprehensive validation and documentation
"""

from .audio import router as audio_router
from .audio_coordination import router as audio_coordination_router
from .bot import router as bot_router
from .websocket import router as websocket_router
from .system import router as system_router
from .settings import router as settings_router
from .translation import router as translation_router

__all__ = [
    "audio_router",
    "audio_coordination_router",
    "bot_router",
    "websocket_router",
    "system_router",
    "settings_router",
    "translation_router",
]
