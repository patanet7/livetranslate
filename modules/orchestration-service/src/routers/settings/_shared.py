"""
Shared components for settings router modules

Common imports, utilities, models, and configurations used across all settings router components.
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiohttp
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Initialize logger
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Sync Imports with Fallbacks
# ============================================================================

# Try to import dependencies, fallback to basic functionality if not available
try:
    from audio.config_sync import (
        ConfigSyncModes,
        apply_configuration_preset,
        get_config_sync_manager,
        get_config_sync_mode,
        get_unified_configuration,
        sync_all_configurations,
        update_configuration,
    )

    CONFIG_SYNC_AVAILABLE = True
    logger.info(" Configuration sync manager available")
except ImportError as e:
    CONFIG_SYNC_AVAILABLE = False
    logger.warning(f" Configuration sync manager not available: {e}")

    class ConfigSyncModes(str, Enum):
        API_ONLY = "api"
        WORKER = "worker"

    def get_config_sync_mode() -> "ConfigSyncModes":
        return ConfigSyncModes.API_ONLY

    async def get_config_sync_manager():
        return None

    async def get_unified_configuration():
        return {}

    async def update_configuration(
        component: str, config_updates: dict[str, Any], propagate: bool = True
    ):
        return {"success": False, "errors": ["Config sync not available"]}

    async def apply_configuration_preset(preset_name: str):
        return {"success": False, "error": "Config sync not available"}

    async def sync_all_configurations():
        return {"success": False, "errors": ["Config sync not available"]}


try:
    from dependencies import get_config_manager, get_event_publisher
except ImportError:

    def get_config_manager():
        return None

    async def get_event_publisher():
        return None


try:
    from models.config import (
        ConfigResponse,
        ConfigUpdate,
        ConfigUpdateResponse,
        ConfigValidation,
    )
except ImportError:
    # Define basic models if not available
    class ConfigResponse(BaseModel):
        data: dict[str, Any]
        updated_at: datetime

    class ConfigUpdate(BaseModel):
        data: dict[str, Any]

    class ConfigValidation(BaseModel):
        valid: bool
        errors: list[str] = []

    class ConfigUpdateResponse(BaseModel):
        message: str
        updated_keys: list[str] = []


# ============================================================================
# Configuration File Paths
# ============================================================================

CONFIG_DIR = Path("./config")
AUDIO_CONFIG_FILE = CONFIG_DIR / "audio_processing.json"
CHUNKING_CONFIG_FILE = CONFIG_DIR / "chunking.json"
CORRELATION_CONFIG_FILE = CONFIG_DIR / "correlation.json"
TRANSLATION_CONFIG_FILE = CONFIG_DIR / "translation.json"
BOT_CONFIG_FILE = CONFIG_DIR / "bot_management.json"
SYSTEM_CONFIG_FILE = CONFIG_DIR / "system.json"

# Ensure config directory exists
CONFIG_DIR.mkdir(exist_ok=True)


# ============================================================================
# Request/Response Models
# ============================================================================


class UserSettingsRequest(BaseModel):
    """Request model for user settings"""

    theme: str | None = Field(None, description="UI theme (light/dark)")
    language: str | None = Field(None, description="Interface language")
    notifications: bool | None = Field(None, description="Enable notifications")
    audio_auto_start: bool | None = Field(None, description="Auto-start audio capture")
    default_translation_language: str | None = Field(None, description="Default target language")
    transcription_model: str | None = Field(None, description="Preferred transcription model")
    custom_settings: dict[str, Any] | None = Field(None, description="Custom user settings")


class UserConfigResponse(BaseModel):
    """Response model for user settings"""

    user_id: str
    theme: str
    language: str
    notifications: bool
    audio_auto_start: bool
    default_translation_language: str
    transcription_model: str
    custom_settings: dict[str, Any]
    updated_at: datetime


class SystemSettingsRequest(BaseModel):
    """Request model for system settings"""

    websocket_max_connections: int | None = Field(None, ge=1, le=100000)
    websocket_timeout: int | None = Field(None, ge=30, le=3600)
    health_check_interval: int | None = Field(None, ge=5, le=300)
    api_rate_limit: int | None = Field(None, ge=1, le=10000)
    log_level: str | None = Field(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    enable_metrics: bool | None = None
    maintenance_mode: bool | None = None


class ServiceSettingsRequest(BaseModel):
    """Request model for service settings"""

    service_name: str = Field(..., description="Name of the service")
    url: str | None = Field(None, description="Service URL")
    timeout: int | None = Field(None, ge=1, le=300)
    retries: int | None = Field(None, ge=0, le=10)
    health_check_path: str | None = Field(None, description="Health check endpoint")
    enabled: bool | None = Field(None, description="Enable/disable service")
    custom_config: dict[str, Any] | None = Field(None, description="Custom service configuration")


class SettingsBackupResponse(BaseModel):
    """Response model for settings backup"""

    backup_id: str
    created_at: datetime
    settings_count: int
    size_bytes: int


# ============================================================================
# Configuration Helpers (use centralized constants)
# ============================================================================


def _get_language_config() -> dict[str, Any]:
    """Get language configuration from centralized system constants."""
    try:
        from system_constants import (
            DEFAULT_CONFIG,
            VALID_LANGUAGE_CODES,
        )

        return {
            "auto_detect": DEFAULT_CONFIG.get("auto_detect_language", True),
            "default_source_language": DEFAULT_CONFIG.get("default_source_language", "en"),
            "target_languages": DEFAULT_CONFIG.get("default_target_languages", ["en"]),
            "supported_languages": VALID_LANGUAGE_CODES,
            "confidence_threshold": DEFAULT_CONFIG.get("confidence_threshold", 0.8),
        }
    except ImportError:
        logger.warning("Could not import system_constants for language config")
        return {
            "auto_detect": True,
            "default_source_language": "en",
            "target_languages": ["en"],
            "supported_languages": ["en", "es", "fr", "de"],
            "confidence_threshold": 0.8,
        }


# ============================================================================
# Enhanced Configuration Models
# ============================================================================


class AudioProcessingConfig(BaseModel):
    """Audio processing configuration schema"""

    vad: dict[str, Any] = {
        "enabled": True,
        "mode": "webrtc",
        "aggressiveness": 2,
        "energy_threshold": 0.01,
        "voice_freq_min": 85,
        "voice_freq_max": 300,
    }
    voice_filter: dict[str, Any] = {
        "enabled": True,
        "fundamental_min": 85,
        "fundamental_max": 300,
        "formant1_min": 200,
        "formant1_max": 1000,
        "preserve_formants": True,
    }
    noise_reduction: dict[str, Any] = {
        "enabled": True,
        "mode": "moderate",
        "strength": 0.7,
        "voice_protection": True,
    }
    voice_enhancement: dict[str, Any] = {
        "enabled": True,
        "normalize": False,
        "compressor": {"threshold": -20, "ratio": 3, "knee": 2.0},
    }
    limiting: dict[str, Any] = {"enabled": True, "threshold": -3, "release_time": 10}
    quality_control: dict[str, Any] = {
        "min_snr_db": 10,
        "max_clipping_percent": 1.0,
        "silence_threshold": 0.0001,
        "enable_quality_gates": True,
    }


class ChunkingConfig(BaseModel):
    """Audio chunking configuration schema"""

    chunking: dict[str, Any] = {
        "chunk_duration": 5.0,
        "overlap_duration": 0.5,
        "overlap_mode": "adaptive",
        "min_chunk_duration": 1.0,
        "max_chunk_duration": 30.0,
        "voice_activity_chunking": True,
    }
    storage: dict[str, Any] = {
        "audio_storage_path": "/data/audio",
        "file_format": "wav",
        "compression": False,
        "cleanup_old_chunks": True,
        "retention_hours": 24,
    }
    coordination: dict[str, Any] = {
        "coordinate_with_services": True,
        "sync_chunk_boundaries": True,
        "chunk_metadata_storage": True,
        "enable_chunk_correlation": True,
    }
    database: dict[str, Any] = {
        "store_chunk_metadata": True,
        "store_audio_hashes": True,
        "correlation_tracking": True,
        "performance_metrics": True,
    }


class CorrelationConfig(BaseModel):
    """Speaker correlation configuration schema"""

    general: dict[str, Any] = {
        "enabled": True,
        "correlation_mode": "hybrid",
        "fallback_to_acoustic": True,
        "confidence_threshold": 0.7,
        "auto_correlation_timeout": 30000,
    }
    manual: dict[str, Any] = {
        "enabled": True,
        "allow_manual_override": True,
        "manual_mapping_priority": True,
        "require_confirmation": False,
        "default_speaker_names": ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"],
    }
    acoustic: dict[str, Any] = {
        "enabled": True,
        "algorithm": "cosine_similarity",
        "similarity_threshold": 0.8,
        "voice_embedding_model": "resemblyzer",
        "speaker_identification_confidence": 0.75,
        "adaptive_threshold": True,
    }
    google_meet: dict[str, Any] = {
        "enabled": True,
        "api_correlation": True,
        "caption_correlation": True,
        "participant_matching": True,
        "use_display_names": True,
        "fallback_on_api_failure": True,
    }
    timing: dict[str, Any] = {
        "time_drift_correction": True,
        "max_time_drift_ms": 1000,
        "correlation_window_ms": 5000,
        "timestamp_alignment": "adaptive",
        "sync_quality_threshold": 0.8,
    }


class TranslationConfig(BaseModel):
    """Translation service configuration schema"""

    service: dict[str, Any] = {
        "enabled": True,
        "service_url": "http://localhost:5003",
        "inference_engine": "vllm",
        "model_name": "llama2-7b-chat",
        "fallback_model": "orca-mini-3b",
        "timeout_ms": 30000,
        "max_retries": 3,
    }
    # Languages configuration - imports from centralized system constants
    # See: config/system_constants.py for the single source of truth
    languages: dict[str, Any] = Field(default_factory=lambda: _get_language_config())
    quality: dict[str, Any] = {
        "quality_threshold": 0.7,
        "confidence_scoring": True,
        "translation_validation": True,
        "context_preservation": True,
        "speaker_attribution": True,
    }
    model: dict[str, Any] = {
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "context_window": 2048,
        "batch_size": 4,
    }


class BotConfig(BaseModel):
    """Bot management configuration schema"""

    manager: dict[str, Any] = {
        "enabled": True,
        "max_concurrent_bots": 10,
        "bot_spawn_timeout": 30000,
        "health_check_interval": 5000,
        "auto_recovery": True,
        "max_recovery_attempts": 3,
        "cleanup_on_shutdown": True,
    }
    google_meet: dict[str, Any] = {
        "enabled": True,
        "credentials_path": "/config/google-meet-credentials.json",
        "oauth_scopes": [
            "https://www.googleapis.com/auth/meetings.space.created",
            "https://www.googleapis.com/auth/meetings.space.readonly",
        ],
        "api_timeout_ms": 10000,
        "fallback_mode": True,
        "meeting_detection": True,
    }
    audio_capture: dict[str, Any] = {
        "enabled": True,
        "capture_method": "loopback",
        "audio_device": "default",
        "sample_rate": 16000,
        "channels": 1,
        "buffer_duration_ms": 100,
        "quality_threshold": 0.3,
    }
    performance: dict[str, Any] = {
        "cpu_limit_percent": 50,
        "memory_limit_mb": 1024,
        "disk_space_limit_gb": 5,
        "priority_level": "normal",
        "process_isolation": True,
    }


class SystemConfig(BaseModel):
    """System configuration schema"""

    general: dict[str, Any] = {
        "system_name": "LiveTranslate System",
        "environment": "development",
        "debug_mode": True,
        "log_level": "info",
        "timezone": "UTC",
        "language": "en",
    }
    security: dict[str, Any] = {
        "enable_authentication": False,
        "session_timeout_minutes": 60,
        "rate_limiting": True,
        "rate_limit_requests_per_minute": 100,
        "cors_enabled": True,
        "allowed_origins": ["http://localhost:5173", "http://localhost:3000"],
        "api_key_required": False,
    }
    performance: dict[str, Any] = {
        "max_concurrent_sessions": 100,
        "request_timeout_ms": 30000,
        "connection_pool_size": 20,
        "cache_enabled": True,
        "cache_ttl_minutes": 15,
        "compression_enabled": True,
    }
    monitoring: dict[str, Any] = {
        "health_checks": True,
        "metrics_collection": True,
        "error_tracking": True,
        "performance_monitoring": True,
        "alert_email_enabled": False,
        "alert_email_addresses": [],
        "alert_thresholds": {
            "cpu_usage_percent": 80,
            "memory_usage_percent": 85,
            "disk_usage_percent": 90,
            "error_rate_percent": 5,
        },
    }


# ============================================================================
# Prompt Management Models
# ============================================================================


class PromptTemplateRequest(BaseModel):
    """Request model for prompt template creation/update"""

    id: str = Field(..., description="Unique prompt ID")
    name: str = Field(..., description="Prompt name")
    description: str = Field(..., description="Prompt description")
    template: str = Field(..., description="Prompt template with variables")
    system_message: str | None = Field(None, description="System message for AI model")
    language_pairs: list[str] | None = Field(default=["*"], description="Supported language pairs")
    category: str = Field(default="general", description="Prompt category")
    version: str = Field(default="1.0", description="Prompt version")
    is_active: bool = Field(default=True, description="Whether prompt is active")
    metadata: dict[str, Any] | None = Field(default={}, description="Additional metadata")


class PromptTestRequest(BaseModel):
    """Request model for prompt testing"""

    text: str = Field(..., description="Text to translate")
    source_language: str = Field(default="auto", description="Source language")
    target_language: str = Field(default="en", description="Target language")
    context: str | None = Field(default="", description="Additional context")
    style: str | None = Field(default="", description="Translation style")
    domain: str | None = Field(default="", description="Domain/field")
    session_id: str | None = Field(None, description="Session ID")
    confidence_threshold: float = Field(default=0.8, description="Confidence threshold")
    preserve_formatting: bool = Field(default=True, description="Preserve formatting")


class ModelSwitchRequest(BaseModel):
    """Request to switch translation model at runtime"""

    model: str = Field(..., description="Model name (e.g., 'llama2:7b', 'mistral:latest')")
    backend: str = Field("ollama", description="Backend to use: ollama, groq, vllm, openai")


# ============================================================================
# Configuration Management Functions
# ============================================================================


async def load_config(file_path: Path, default_config: dict[str, Any]) -> dict[str, Any]:
    """Load configuration from file with fallback to defaults"""
    try:
        if file_path.exists():
            async with aiofiles.open(file_path) as f:
                content = await f.read()
                config = json.loads(content)
                logger.info(f"Loaded configuration from {file_path}")
                return {**default_config, **config}
        else:
            logger.info(f"Configuration file {file_path} not found, using defaults")
            return default_config
    except Exception as e:
        logger.error(f"Error loading configuration from {file_path}: {e}")
        return default_config


async def save_config(file_path: Path, config: dict[str, Any]) -> bool:
    """Save configuration to file"""
    try:
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(config, indent=2))
        logger.info(f"Saved configuration to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {file_path}: {e}")
        return False


# ============================================================================
# Translation Service Client
# ============================================================================

TRANSLATION_SERVICE_URL = os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:5003")


async def get_translation_service_client():
    """Get HTTP client for translation service"""
    timeout = aiohttp.ClientTimeout(total=30)
    return aiohttp.ClientSession(timeout=timeout)


# ============================================================================
# Utility Functions
# ============================================================================


def create_settings_router(prefix: str = "", tags: list[str] | None = None) -> APIRouter:
    """Create a standardized settings router with common configuration."""
    return APIRouter(
        prefix=prefix,
        tags=tags or ["settings"],
        responses={
            404: {"description": "Not found"},
            422: {"description": "Validation error"},
            500: {"description": "Internal server error"},
        },
    )


def get_error_response(
    status_code: int, message: str, details: dict[str, Any] | None = None
) -> HTTPException:
    """Create standardized error response"""
    error_detail = {"message": message}
    if details:
        error_detail["details"] = details

    return HTTPException(status_code=status_code, detail=error_detail)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "AUDIO_CONFIG_FILE",
    "BOT_CONFIG_FILE",
    "CHUNKING_CONFIG_FILE",
    # Config file paths
    "CONFIG_DIR",
    # Config sync
    "CONFIG_SYNC_AVAILABLE",
    "CORRELATION_CONFIG_FILE",
    "SYSTEM_CONFIG_FILE",
    "TRANSLATION_CONFIG_FILE",
    "TRANSLATION_SERVICE_URL",
    "UTC",
    # FastAPI/Pydantic imports for re-export
    "APIRouter",
    "Any",
    # Models - Enhanced Config
    "AudioProcessingConfig",
    "BaseModel",
    "BotConfig",
    "ChunkingConfig",
    # Models - Config
    "ConfigResponse",
    "ConfigSyncModes",
    "ConfigUpdate",
    "ConfigUpdateResponse",
    "ConfigValidation",
    "CorrelationConfig",
    "Depends",
    "Field",
    "HTTPException",
    "JSONResponse",
    "ModelSwitchRequest",
    "Optional",
    # Models - Prompts
    "PromptTemplateRequest",
    "PromptTestRequest",
    "ServiceSettingsRequest",
    "SettingsBackupResponse",
    "SystemConfig",
    "SystemSettingsRequest",
    "TranslationConfig",
    "UserConfigResponse",
    # Models - User/System/Service
    "UserSettingsRequest",
    "aiohttp",
    "apply_configuration_preset",
    # Other re-exports
    "asyncio",
    "create_settings_router",
    "datetime",
    # Dependencies
    "get_config_manager",
    "get_config_sync_manager",
    "get_config_sync_mode",
    "get_error_response",
    "get_event_publisher",
    "get_translation_service_client",
    "get_unified_configuration",
    # Functions
    "load_config",
    # Logger
    "logger",
    "save_config",
    "status",
    "sync_all_configurations",
    "timezone",
    "update_configuration",
]
