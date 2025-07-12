"""
Service Clients Package

Provides client classes for communicating with external services
in the LiveTranslate ecosystem.
"""

from .audio_service_client import AudioServiceClient
from .translation_service_client import TranslationServiceClient

__all__ = ["AudioServiceClient", "TranslationServiceClient"]
