"""
Shared components for audio router modules

Common imports, utilities, and configurations used across all audio router components.
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import ValidationError

# Model imports
from models.audio import (
    AudioProcessingRequest,
    AudioProcessingResponse,
    AudioConfiguration,
    AudioStats,
    ProcessingStage,
    ProcessingQuality,
)

# Dependency imports
from dependencies import (
    get_config_manager,
    get_health_monitor,
    get_audio_service_client,
    get_translation_service_client,
    get_audio_coordinator,
    get_config_sync_manager,
)

# Utility imports
from utils.audio_processing import AudioProcessor
from utils.rate_limiting import RateLimiter
from utils.security import SecurityUtils
from utils.audio_errors import (
    AudioProcessingBaseError, AudioFormatError, AudioCorruptionError, 
    AudioProcessingError, ServiceUnavailableError, ValidationError, 
    ConfigurationError, NetworkError, TimeoutError,
    CircuitBreaker, RetryManager, RetryConfig,
    FormatRecoveryStrategy, ServiceRecoveryStrategy,
    ErrorLogger, error_boundary, default_circuit_breaker, default_retry_manager
)

# Shared logger
logger = logging.getLogger(__name__)

# Shared utilities (initialized once)
rate_limiter = RateLimiter()
security_utils = SecurityUtils()

# Shared error handling components
audio_service_circuit_breaker = CircuitBreaker(
    name="audio_service",
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=2
)

translation_service_circuit_breaker = CircuitBreaker(
    name="translation_service", 
    failure_threshold=3,
    recovery_timeout=30,
    success_threshold=2
)

retry_manager = RetryManager(RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
))

error_logger = ErrorLogger("orchestration_audio")

# Recovery strategies
format_recovery = FormatRecoveryStrategy()
service_recovery = ServiceRecoveryStrategy()

# Shared utility functions
def get_common_dependencies():
    """Get common dependencies used across audio endpoints."""
    return {
        "rate_limiter": rate_limiter,
        "security_utils": security_utils,
        "audio_circuit_breaker": audio_service_circuit_breaker,
        "translation_circuit_breaker": translation_service_circuit_breaker,
        "retry_manager": retry_manager,
        "error_logger": error_logger,
        "format_recovery": format_recovery,
        "service_recovery": service_recovery,
    }

def create_audio_router(prefix: str = "") -> APIRouter:
    """Create a standardized audio router with common configuration."""
    return APIRouter(
        prefix=prefix,
        tags=["audio"],
        responses={
            404: {"description": "Not found"},
            422: {"description": "Validation error"},
            500: {"description": "Internal server error"}
        }
    )