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
from .translation import (
    UnifiedTranslationError,
    UnifiedTranslationService,
    get_unified_translation_service,
)

__all__ = [
    "UnifiedAudioError",
    "UnifiedAudioService",
    "UnifiedTranslationError",
    "UnifiedTranslationService",
    "get_unified_audio_service",
    "get_unified_translation_service",
]
