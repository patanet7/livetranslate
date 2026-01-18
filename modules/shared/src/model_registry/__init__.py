"""
Model Registry package for LiveTranslate.

Provides centralized model naming, aliases, and validation for all services.

Usage:
    from modules.shared.src.model_registry import (
        ModelRegistry,
        normalize_whisper_model,
        is_valid_whisper_model,
        WhisperModelSize,
        TranslationBackend,
    )

    # Normalize model names (e.g., "base" -> "whisper-base")
    canonical = normalize_whisper_model("base")

    # Validate model names
    if is_valid_whisper_model("whisper-large-v3"):
        ...

    # Get model information
    info = ModelRegistry.get_whisper_model_info("whisper-base")
"""

from .model_registry import (
    TRANSLATION_MODELS,
    WHISPER_MODELS,
    # Main registry class
    ModelRegistry,
    # Translation types
    TranslationBackend,
    TranslationModelInfo,
    WhisperModelInfo,
    # Whisper types
    WhisperModelSize,
    get_translation_fallback_chain,
    get_whisper_fallback_chain,
    is_valid_whisper_model,
    # Convenience functions
    normalize_whisper_model,
)

__all__ = [
    "TRANSLATION_MODELS",
    "WHISPER_MODELS",
    # Main registry
    "ModelRegistry",
    # Translation
    "TranslationBackend",
    "TranslationModelInfo",
    "WhisperModelInfo",
    # Whisper
    "WhisperModelSize",
    "get_translation_fallback_chain",
    "get_whisper_fallback_chain",
    "is_valid_whisper_model",
    # Convenience functions
    "normalize_whisper_model",
]
