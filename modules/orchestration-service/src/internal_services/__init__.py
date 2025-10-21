"""
Internal service facades used to embed formerly external microservices directly
into the orchestration stack.

These helpers let the FastAPI app call translation and audio capabilities
without going through HTTP hops. The modules are intentionally lightweight
wrappers so we can swap in local or remote implementations transparently.
"""

from .translation import (
    get_unified_translation_service,
    UnifiedTranslationService,
    UnifiedTranslationError,
)
from .audio import (
    get_unified_audio_service,
    UnifiedAudioService,
    UnifiedAudioError,
)

__all__ = [
    "UnifiedTranslationService",
    "UnifiedTranslationError",
    "get_unified_translation_service",
    "UnifiedAudioService",
    "UnifiedAudioError",
    "get_unified_audio_service",
]
