"""
Translation Service Module

This module provides translation capabilities using local inference (vLLM, Ollama)
with fallback to external APIs. It integrates with the shared inference infrastructure
for efficient model management.

Key Components:
- TranslationService: Main service class with multiple backend support
- LocalTranslationService: Local inference using vLLM/Ollama
- LegacyTranslationWrapper: Integration with existing external APIs
- API Server: Flask-based REST API with WebSocket streaming

Usage:
    from translation_service import create_translation_service, TranslationRequest

    # Create service
    service = await create_translation_service()

    # Create request
    request = TranslationRequest(
        text="Hello world",
        target_language="es"
    )

    # Translate
    result = await service.translate(request)
    print(result.translated_text)
"""

from .api_server import create_app
from .legacy_wrapper import LegacyTranslationWrapper
from .local_translation import LocalTranslationService
from .translation_service import (
    TranslationRequest,
    TranslationResult,
    TranslationService,
    create_translation_service,
)

__version__ = "1.0.0"
__author__ = "LiveTranslate Team"

# Public API
__all__ = [
    "LegacyTranslationWrapper",
    "LocalTranslationService",
    "TranslationRequest",
    "TranslationResult",
    "TranslationService",
    "create_app",
    "create_translation_service",
]


# Factory function for easy module usage
async def create_service(config=None):
    """
    Convenience factory function to create a translation service

    Args:
        config: Optional configuration dictionary

    Returns:
        Initialized TranslationService instance
    """
    return await create_translation_service(config)


# Module-level configuration
DEFAULT_CONFIG = {
    "use_local_inference": True,
    "use_legacy_apis": True,
    "translation_model": "meta-llama/Llama-3.1-8B-Instruct",
    "max_tokens": 1024,
    "temperature": 0.1,
    "timeout": 30,
    "retry_attempts": 3,
    "confidence_threshold": 0.8,
}
