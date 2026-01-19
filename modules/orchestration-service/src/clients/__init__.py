"""
Service Clients Package

Provides client classes for communicating with external services
in the LiveTranslate ecosystem.
"""

from .audio_service_client import AudioServiceClient
from .fireflies_client import (
    FirefliesAPIError,
    FirefliesAuthError,
    FirefliesClient,
    FirefliesConnectionError,
    FirefliesGraphQLClient,
    FirefliesRealtimeClient,
)
from .translation_service_client import TranslationServiceClient

__all__ = [
    "AudioServiceClient",
    "FirefliesAPIError",
    "FirefliesAuthError",
    # Fireflies clients
    "FirefliesClient",
    "FirefliesConnectionError",
    "FirefliesGraphQLClient",
    "FirefliesRealtimeClient",
    "TranslationServiceClient",
]
