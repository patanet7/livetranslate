"""
Service Clients Package

Provides client classes for communicating with external services
in the LiveTranslate ecosystem.
"""

from .audio_service_client import AudioServiceClient
from .translation_service_client import TranslationServiceClient
from .fireflies_client import (
    FirefliesClient,
    FirefliesGraphQLClient,
    FirefliesRealtimeClient,
    FirefliesAPIError,
    FirefliesConnectionError,
    FirefliesAuthError,
)

__all__ = [
    "AudioServiceClient",
    "TranslationServiceClient",
    # Fireflies clients
    "FirefliesClient",
    "FirefliesGraphQLClient",
    "FirefliesRealtimeClient",
    "FirefliesAPIError",
    "FirefliesConnectionError",
    "FirefliesAuthError",
]
