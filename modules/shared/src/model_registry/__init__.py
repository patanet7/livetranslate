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
    # Main registry class
    ModelRegistry,
    # Whisper types
    WhisperModelSize,
    WhisperModelInfo,
    WHISPER_MODELS,
    # Translation types
    TranslationBackend,
    TranslationModelInfo,
    TRANSLATION_MODELS,
    # Convenience functions
    normalize_whisper_model,
    is_valid_whisper_model,
    get_whisper_fallback_chain,
    get_translation_fallback_chain,
)

__all__ = [
    # Main registry
    "ModelRegistry",
    # Whisper
    "WhisperModelSize",
    "WhisperModelInfo",
    "WHISPER_MODELS",
    # Translation
    "TranslationBackend",
    "TranslationModelInfo",
    "TRANSLATION_MODELS",
    # Convenience functions
    "normalize_whisper_model",
    "is_valid_whisper_model",
    "get_whisper_fallback_chain",
    "get_translation_fallback_chain",
]
