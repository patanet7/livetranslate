"""
Internal service facades used to embed formerly external microservices directly
into the orchestration stack.

These helpers let the FastAPI app call translation and audio capabilities
without going through HTTP hops. The modules are intentionally lightweight
wrappers so we can swap in local or remote implementations transparently.
"""

from .audio import (
    UnifiedAudioError,
    UnifiedAudioService,
    get_unified_audio_service,
)

# NOTE: internal_services.translation removed — superseded by translation.service.TranslationService

__all__ = [
    "UnifiedAudioError",
    "UnifiedAudioService",
    "get_unified_audio_service",
]
