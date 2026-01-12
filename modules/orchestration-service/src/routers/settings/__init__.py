"""
Settings Router Package

Modular settings management router split from the original monolithic settings.py file.

Modules:
- _shared: Common imports, utilities, models, and shared components
- general: User settings, system settings, service settings, backup/restore, validation
- audio: Audio processing, chunking, and correlation settings
- translation: Translation service configuration and testing
- bot: Bot management settings and templates
- sync: Configuration synchronization with whisper service and translation model management
- prompts: Prompt template CRUD and testing
"""

from fastapi import APIRouter

from . import general, audio, translation, bot, sync, prompts

# Aggregate router that combines all sub-routers
# Note: No prefix here since main_fastapi.py adds /api/settings
router = APIRouter(tags=["settings"])

# Include all sub-routers
# General settings (user, system, services, backup/restore, validation, export/import)
router.include_router(general.router)

# Audio-related settings (audio-processing, chunking, correlation)
router.include_router(audio.router)

# Translation settings
router.include_router(translation.router)

# Bot settings
router.include_router(bot.router)

# Configuration sync endpoints (all /sync/* paths)
router.include_router(sync.router)

# Prompt management endpoints
router.include_router(prompts.router)

__all__ = [
    "router",
    "general",
    "audio",
    "translation",
    "bot",
    "sync",
    "prompts",
]
