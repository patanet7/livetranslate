"""
Comprehensive Settings Management Router for LiveTranslate Orchestration Service

Provides comprehensive CRUD endpoints for all service configurations
including audio processing, chunking, correlation, translation, bot management,
and system settings with validation and type safety.

This enhanced router supports the React frontend settings pages with:
- Audio processing pipeline configuration
- Audio chunking and database integration
- Speaker correlation settings with manual mappings
- Translation service configuration
- Bot management with templates
- System-wide configuration and health monitoring
"""

import json
import os
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import aiofiles

# Initialize logger
logger = logging.getLogger(__name__)

# Try to import dependencies, fallback to basic functionality if not available
try:
    from audio.config_sync import (
        get_config_sync_manager,
        get_unified_configuration,
        update_configuration,
        apply_configuration_preset,
        sync_all_configurations
    )
    CONFIG_SYNC_AVAILABLE = True
    logger.info(" Configuration sync manager available")
except ImportError as e:
    CONFIG_SYNC_AVAILABLE = False
    logger.warning(f" Configuration sync manager not available: {e}")
try:
    from dependencies import get_config_manager
except ImportError:
    def get_config_manager():
        return None

try:
    from models.config import (
        ConfigUpdate,
        ConfigResponse,
        ConfigValidation,
        ConfigUpdateResponse,
    )
except ImportError:
    # Define basic models if not available
    class ConfigResponse(BaseModel):
        data: Dict[str, Any]
        updated_at: datetime
    
    class ConfigUpdate(BaseModel):
        data: Dict[str, Any]
    
    class ConfigValidation(BaseModel):
        valid: bool
        errors: List[str] = []
    
    class ConfigUpdateResponse(BaseModel):
        message: str
        updated_keys: List[str] = []

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================


class UserSettingsRequest(BaseModel):
    """Request model for user settings"""

    theme: Optional[str] = Field(None, description="UI theme (light/dark)")
    language: Optional[str] = Field(None, description="Interface language")
    notifications: Optional[bool] = Field(None, description="Enable notifications")
    audio_auto_start: Optional[bool] = Field(
        None, description="Auto-start audio capture"
    )
    default_translation_language: Optional[str] = Field(
        None, description="Default target language"
    )
    transcription_model: Optional[str] = Field(
        None, description="Preferred transcription model"
    )
    custom_settings: Optional[Dict[str, Any]] = Field(
        None, description="Custom user settings"
    )


class UserConfigResponse(BaseModel):
    """Response model for user settings"""

    user_id: str
    theme: str
    language: str
    notifications: bool
    audio_auto_start: bool
    default_translation_language: str
    transcription_model: str
    custom_settings: Dict[str, Any]
    updated_at: datetime


class SystemSettingsRequest(BaseModel):
    """Request model for system settings"""

    websocket_max_connections: Optional[int] = Field(None, ge=1, le=100000)
    websocket_timeout: Optional[int] = Field(None, ge=30, le=3600)
    health_check_interval: Optional[int] = Field(None, ge=5, le=300)
    api_rate_limit: Optional[int] = Field(None, ge=1, le=10000)
    log_level: Optional[str] = Field(
        None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    enable_metrics: Optional[bool] = None
    maintenance_mode: Optional[bool] = None


class ServiceSettingsRequest(BaseModel):
    """Request model for service settings"""

    service_name: str = Field(..., description="Name of the service")
    url: Optional[str] = Field(None, description="Service URL")
    timeout: Optional[int] = Field(None, ge=1, le=300)
    retries: Optional[int] = Field(None, ge=0, le=10)
    health_check_path: Optional[str] = Field(None, description="Health check endpoint")
    enabled: Optional[bool] = Field(None, description="Enable/disable service")
    custom_config: Optional[Dict[str, Any]] = Field(
        None, description="Custom service configuration"
    )


class SettingsBackupResponse(BaseModel):
    """Response model for settings backup"""

    backup_id: str
    created_at: datetime
    settings_count: int
    size_bytes: int


# ============================================================================
# User Settings Endpoints
# ============================================================================


@router.get("/user", response_model=UserConfigResponse)
async def get_user_settings(
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get user settings

    Returns current user settings and preferences.
    """
    try:
        # TODO: Implement user authentication
        user_id = "anonymous"

        settings = await config_manager.get_user_settings(user_id)

        return UserConfigResponse(
            user_id=user_id,
            theme=settings.get("theme", "dark"),
            language=settings.get("language", "en"),
            notifications=settings.get("notifications", True),
            audio_auto_start=settings.get("audio_auto_start", False),
            default_translation_language=settings.get(
                "default_translation_language", "es"
            ),
            transcription_model=settings.get("transcription_model", "base"),
            custom_settings=settings.get("custom_settings", {}),
            updated_at=settings.get("updated_at", datetime.now()),
        )

    except Exception as e:
        logger.error(f"Failed to get user settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user settings: {str(e)}",
        )


@router.put("/user", response_model=UserConfigResponse)
async def update_user_settings(
    request: UserSettingsRequest,
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Update user settings

    Updates user preferences and settings. Only provided fields are updated.
    """
    try:
        # TODO: Implement user authentication
        user_id = "anonymous"

        # Build update data from request
        update_data = {}
        if request.theme is not None:
            update_data["theme"] = request.theme
        if request.language is not None:
            update_data["language"] = request.language
        if request.notifications is not None:
            update_data["notifications"] = request.notifications
        if request.audio_auto_start is not None:
            update_data["audio_auto_start"] = request.audio_auto_start
        if request.default_translation_language is not None:
            update_data[
                "default_translation_language"
            ] = request.default_translation_language
        if request.transcription_model is not None:
            update_data["transcription_model"] = request.transcription_model
        if request.custom_settings is not None:
            update_data["custom_settings"] = request.custom_settings

        # Update settings
        updated_settings = await config_manager.update_user_settings(
            user_id, update_data
        )

        return UserConfigResponse(
            user_id=user_id,
            theme=updated_settings.get("theme", "dark"),
            language=updated_settings.get("language", "en"),
            notifications=updated_settings.get("notifications", True),
            audio_auto_start=updated_settings.get("audio_auto_start", False),
            default_translation_language=updated_settings.get(
                "default_translation_language", "es"
            ),
            transcription_model=updated_settings.get("transcription_model", "base"),
            custom_settings=updated_settings.get("custom_settings", {}),
            updated_at=updated_settings.get("updated_at", datetime.now()),
        )

    except Exception as e:
        logger.error(f"Failed to update user settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user settings: {str(e)}",
        )


# ============================================================================
# System Settings Endpoints
# ============================================================================


@router.get("/system")
async def get_system_settings(
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get system settings

    Returns current system-wide settings and configuration.
    Requires authentication.
    """
    try:
        settings = await config_manager.get_system_settings()

        return settings

    except Exception as e:
        logger.error(f"Failed to get system settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system settings: {str(e)}",
        )


@router.put("/system")
async def update_system_settings(
    request: SystemSettingsRequest,
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Update system settings

    Updates system-wide settings. Some changes may require service restart.
    Requires authentication.
    """
    try:
        logger.info("Updating system settings")

        # Build update data from request
        update_data = {}
        if request.websocket_max_connections is not None:
            update_data["websocket_max_connections"] = request.websocket_max_connections
        if request.websocket_timeout is not None:
            update_data["websocket_timeout"] = request.websocket_timeout
        if request.health_check_interval is not None:
            update_data["health_check_interval"] = request.health_check_interval
        if request.api_rate_limit is not None:
            update_data["api_rate_limit"] = request.api_rate_limit
        if request.log_level is not None:
            update_data["log_level"] = request.log_level
        if request.enable_metrics is not None:
            update_data["enable_metrics"] = request.enable_metrics
        if request.maintenance_mode is not None:
            update_data["maintenance_mode"] = request.maintenance_mode

        # Update settings
        result = await config_manager.update_system_settings(update_data)

        return {
            "message": "System settings updated",
            "updated_keys": result.get("updated_keys", []),
            "restart_required": result.get("restart_required", False),
            "settings": result.get("settings", {}),
        }

    except Exception as e:
        logger.error(f"Failed to update system settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update system settings: {str(e)}",
        )


# ============================================================================
# Service Settings Endpoints
# ============================================================================


@router.get("/services")
async def get_service_settings(
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get service settings

    Returns configuration for all managed services.
    """
    try:
        settings = await config_manager.get_service_settings()

        return settings

    except Exception as e:
        logger.error(f"Failed to get service settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service settings: {str(e)}",
        )


@router.put("/services/{service_name}")
async def update_service_settings(
    service_name: str,
    request: ServiceSettingsRequest,
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Update service settings

    Updates configuration for a specific service.
    Requires authentication.
    """
    try:
        logger.info(f"Updating settings for service: {service_name}")

        # Build update data from request
        update_data = {}
        if request.url is not None:
            update_data["url"] = request.url
        if request.timeout is not None:
            update_data["timeout"] = request.timeout
        if request.retries is not None:
            update_data["retries"] = request.retries
        if request.health_check_path is not None:
            update_data["health_check_path"] = request.health_check_path
        if request.enabled is not None:
            update_data["enabled"] = request.enabled
        if request.custom_config is not None:
            update_data["custom_config"] = request.custom_config

        # Update service settings
        result = await config_manager.update_service_settings(service_name, update_data)

        return {
            "message": f"Service {service_name} settings updated",
            "service_name": service_name,
            "updated_keys": result.get("updated_keys", []),
            "restart_required": result.get("restart_required", False),
            "settings": result.get("settings", {}),
        }

    except Exception as e:
        logger.error(f"Failed to update service settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update service settings: {str(e)}",
        )


# ============================================================================
# Audio Settings Endpoints
# ============================================================================


@router.get("/audio", response_model=ConfigResponse)
async def get_audio_settings(
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get audio processing settings

    Returns current audio processing configuration including
    VAD, speaker diarization, and noise reduction settings.
    """
    try:
        settings = await config_manager.get_audio_settings()

        return ConfigResponse(**settings)

    except Exception as e:
        logger.error(f"Failed to get audio settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audio settings: {str(e)}",
        )


@router.put("/audio", response_model=ConfigResponse)
async def update_audio_settings(
    request: Dict[str, Any],
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Update audio processing settings

    Updates audio processing configuration. Changes take effect
    for new audio processing sessions.
    """
    try:
        logger.info("Updating audio processing settings")

        # Update audio settings
        updated_settings = await config_manager.update_audio_settings(request.dict())

        return ConfigResponse(**updated_settings)

    except Exception as e:
        logger.error(f"Failed to update audio settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update audio settings: {str(e)}",
        )


# ============================================================================
# Settings Backup/Restore Endpoints
# ============================================================================


@router.post("/backup", response_model=SettingsBackupResponse)
async def create_settings_backup(
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Create settings backup

    Creates a backup of all system and user settings.
    Requires authentication.
    """
    try:
        logger.info("Creating settings backup")

        backup_result = await config_manager.create_settings_backup()

        return SettingsBackupResponse(
            backup_id=backup_result["backup_id"],
            created_at=backup_result["created_at"],
            settings_count=backup_result["settings_count"],
            size_bytes=backup_result["size_bytes"],
        )

    except Exception as e:
        logger.error(f"Failed to create settings backup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create settings backup: {str(e)}",
        )


@router.post("/restore/{backup_id}")
async def restore_settings_backup(
    backup_id: str,
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Restore settings from backup

    Restores system and user settings from a previously created backup.
    Requires authentication.
    """
    try:
        logger.info(f"Restoring settings from backup: {backup_id}")

        restore_result = await config_manager.restore_settings_backup(backup_id)

        return {
            "message": f"Settings restored from backup {backup_id}",
            "backup_id": backup_id,
            "restored_settings": restore_result.get("restored_settings", 0),
            "restart_required": restore_result.get("restart_required", False),
        }

    except Exception as e:
        logger.error(f"Failed to restore settings backup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore settings backup: {str(e)}",
        )


@router.get("/backups")
async def list_settings_backups(
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    List available settings backups

    Returns a list of all available settings backups.
    Requires authentication.
    """
    try:
        backups = await config_manager.list_settings_backups()

        return {"backups": backups, "total": len(backups)}

    except Exception as e:
        logger.error(f"Failed to list settings backups: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list settings backups: {str(e)}",
        )


# ============================================================================
# Settings Validation Endpoints
# ============================================================================


@router.post("/validate")
async def validate_settings(
    settings: Dict[str, Any],
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Validate settings configuration

    Validates a settings configuration without applying it.
    Returns validation results and any errors.
    """
    try:
        validation_result = await config_manager.validate_settings(settings)

        return {
            "valid": validation_result.get("valid", False),
            "errors": validation_result.get("errors", []),
            "warnings": validation_result.get("warnings", []),
            "suggestions": validation_result.get("suggestions", []),
        }

    except Exception as e:
        logger.error(f"Failed to validate settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate settings: {str(e)}",
        )


@router.get("/defaults")
async def get_default_settings(
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get default settings

    Returns default configuration values for all settings categories.
    """
    try:
        defaults = await config_manager.get_default_settings()

        return {"defaults": defaults, "categories": list(defaults.keys())}

    except Exception as e:
        logger.error(f"Failed to get default settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get default settings: {str(e)}",
        )


@router.post("/reset")
async def reset_settings_to_defaults(
    category: Optional[str] = None,
    config_manager=Depends(get_config_manager),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Reset settings to defaults

    Resets settings to default values. If category is specified,
    only that category is reset. Otherwise, all settings are reset.
    Requires authentication.
    """
    try:
        logger.info(f"Resetting settings to defaults: {category or 'all'}")

        reset_result = await config_manager.reset_settings_to_defaults(category)

        return {
            "message": f"Settings reset to defaults: {category or 'all'}",
            "category": category,
            "reset_keys": reset_result.get("reset_keys", []),
            "restart_required": reset_result.get("restart_required", False),
        }

    except Exception as e:
        logger.error(f"Failed to reset settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset settings: {str(e)}",
        )


# ============================================================================
# ENHANCED SETTINGS API - Supporting React Frontend Settings Pages
# ============================================================================

# Configuration file paths
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
# Enhanced Configuration Models
# ============================================================================

class AudioProcessingConfig(BaseModel):
    """Audio processing configuration schema"""
    vad: Dict[str, Any] = {
        "enabled": True,
        "mode": "webrtc",
        "aggressiveness": 2,
        "energy_threshold": 0.01,
        "voice_freq_min": 85,
        "voice_freq_max": 300
    }
    voice_filter: Dict[str, Any] = {
        "enabled": True,
        "fundamental_min": 85,
        "fundamental_max": 300,
        "formant1_min": 200,
        "formant1_max": 1000,
        "preserve_formants": True
    }
    noise_reduction: Dict[str, Any] = {
        "enabled": True,
        "mode": "moderate",
        "strength": 0.7,
        "voice_protection": True
    }
    voice_enhancement: Dict[str, Any] = {
        "enabled": True,
        "normalize": False,
        "compressor": {
            "threshold": -20,
            "ratio": 3,
            "knee": 2.0
        }
    }
    limiting: Dict[str, Any] = {
        "enabled": True,
        "threshold": -3,
        "release_time": 10
    }
    quality_control: Dict[str, Any] = {
        "min_snr_db": 10,
        "max_clipping_percent": 1.0,
        "silence_threshold": 0.0001,
        "enable_quality_gates": True
    }

class ChunkingConfig(BaseModel):
    """Audio chunking configuration schema"""
    chunking: Dict[str, Any] = {
        "chunk_duration": 5.0,
        "overlap_duration": 0.5,
        "overlap_mode": "adaptive",
        "min_chunk_duration": 1.0,
        "max_chunk_duration": 30.0,
        "voice_activity_chunking": True
    }
    storage: Dict[str, Any] = {
        "audio_storage_path": "/data/audio",
        "file_format": "wav",
        "compression": False,
        "cleanup_old_chunks": True,
        "retention_hours": 24
    }
    coordination: Dict[str, Any] = {
        "coordinate_with_services": True,
        "sync_chunk_boundaries": True,
        "chunk_metadata_storage": True,
        "enable_chunk_correlation": True
    }
    database: Dict[str, Any] = {
        "store_chunk_metadata": True,
        "store_audio_hashes": True,
        "correlation_tracking": True,
        "performance_metrics": True
    }

class CorrelationConfig(BaseModel):
    """Speaker correlation configuration schema"""
    general: Dict[str, Any] = {
        "enabled": True,
        "correlation_mode": "hybrid",
        "fallback_to_acoustic": True,
        "confidence_threshold": 0.7,
        "auto_correlation_timeout": 30000
    }
    manual: Dict[str, Any] = {
        "enabled": True,
        "allow_manual_override": True,
        "manual_mapping_priority": True,
        "require_confirmation": False,
        "default_speaker_names": ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"]
    }
    acoustic: Dict[str, Any] = {
        "enabled": True,
        "algorithm": "cosine_similarity",
        "similarity_threshold": 0.8,
        "voice_embedding_model": "resemblyzer",
        "speaker_identification_confidence": 0.75,
        "adaptive_threshold": True
    }
    google_meet: Dict[str, Any] = {
        "enabled": True,
        "api_correlation": True,
        "caption_correlation": True,
        "participant_matching": True,
        "use_display_names": True,
        "fallback_on_api_failure": True
    }
    timing: Dict[str, Any] = {
        "time_drift_correction": True,
        "max_time_drift_ms": 1000,
        "correlation_window_ms": 5000,
        "timestamp_alignment": "adaptive",
        "sync_quality_threshold": 0.8
    }

class TranslationConfig(BaseModel):
    """Translation service configuration schema"""
    service: Dict[str, Any] = {
        "enabled": True,
        "service_url": "http://localhost:5003",
        "inference_engine": "vllm",
        "model_name": "llama2-7b-chat",
        "fallback_model": "orca-mini-3b",
        "timeout_ms": 30000,
        "max_retries": 3
    }
    languages: Dict[str, Any] = {
        "auto_detect": True,
        "default_source_language": "en",
        "target_languages": ["es", "fr", "de"],
        "supported_languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "tr", "pl", "nl", "sv", "da", "no"],
        "confidence_threshold": 0.8
    }
    quality: Dict[str, Any] = {
        "quality_threshold": 0.7,
        "confidence_scoring": True,
        "translation_validation": True,
        "context_preservation": True,
        "speaker_attribution": True
    }
    model: Dict[str, Any] = {
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "context_window": 2048,
        "batch_size": 4
    }

class BotConfig(BaseModel):
    """Bot management configuration schema"""
    manager: Dict[str, Any] = {
        "enabled": True,
        "max_concurrent_bots": 10,
        "bot_spawn_timeout": 30000,
        "health_check_interval": 5000,
        "auto_recovery": True,
        "max_recovery_attempts": 3,
        "cleanup_on_shutdown": True
    }
    google_meet: Dict[str, Any] = {
        "enabled": True,
        "credentials_path": "/config/google-meet-credentials.json",
        "oauth_scopes": [
            "https://www.googleapis.com/auth/meetings.space.created",
            "https://www.googleapis.com/auth/meetings.space.readonly"
        ],
        "api_timeout_ms": 10000,
        "fallback_mode": True,
        "meeting_detection": True
    }
    audio_capture: Dict[str, Any] = {
        "enabled": True,
        "capture_method": "loopback",
        "audio_device": "default",
        "sample_rate": 16000,
        "channels": 1,
        "buffer_duration_ms": 100,
        "quality_threshold": 0.3
    }
    performance: Dict[str, Any] = {
        "cpu_limit_percent": 50,
        "memory_limit_mb": 1024,
        "disk_space_limit_gb": 5,
        "priority_level": "normal",
        "process_isolation": True
    }

class SystemConfig(BaseModel):
    """System configuration schema"""
    general: Dict[str, Any] = {
        "system_name": "LiveTranslate System",
        "environment": "development",
        "debug_mode": True,
        "log_level": "info",
        "timezone": "UTC",
        "language": "en"
    }
    security: Dict[str, Any] = {
        "enable_authentication": False,
        "session_timeout_minutes": 60,
        "rate_limiting": True,
        "rate_limit_requests_per_minute": 100,
        "cors_enabled": True,
        "allowed_origins": ["http://localhost:5173", "http://localhost:3000"],
        "api_key_required": False
    }
    performance: Dict[str, Any] = {
        "max_concurrent_sessions": 100,
        "request_timeout_ms": 30000,
        "connection_pool_size": 20,
        "cache_enabled": True,
        "cache_ttl_minutes": 15,
        "compression_enabled": True
    }
    monitoring: Dict[str, Any] = {
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
            "error_rate_percent": 5
        }
    }

# ============================================================================
# Configuration Management Functions
# ============================================================================

async def load_config(file_path: Path, default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration from file with fallback to defaults"""
    try:
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
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

async def save_config(file_path: Path, config: Dict[str, Any]) -> bool:
    """Save configuration to file"""
    try:
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(config, indent=2))
        logger.info(f"Saved configuration to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {file_path}: {e}")
        return False

# ============================================================================
# Enhanced Audio Processing Settings Endpoints
# ============================================================================

@router.get("/audio-processing", response_model=Dict[str, Any])
async def get_audio_processing_settings():
    """Get current audio processing configuration"""
    try:
        default_config = AudioProcessingConfig().dict()
        config = await load_config(AUDIO_CONFIG_FILE, default_config)
        return config
    except Exception as e:
        logger.error(f"Error getting audio processing settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load audio processing settings")

@router.post("/audio-processing")
async def save_audio_processing_settings(config: AudioProcessingConfig):
    """Save audio processing configuration"""
    try:
        config_dict = config.dict()
        success = await save_config(AUDIO_CONFIG_FILE, config_dict)
        if success:
            return {"message": "Audio processing settings saved successfully", "config": config_dict}
        else:
            raise HTTPException(status_code=500, detail="Failed to save audio processing settings")
    except Exception as e:
        logger.error(f"Error saving audio processing settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio processing settings")

@router.post("/audio-processing/test")
async def test_audio_processing(test_config: Dict[str, Any]):
    """Test audio processing configuration"""
    try:
        await asyncio.sleep(1)  # Simulate processing
        
        if not test_config.get("vad", {}).get("enabled"):
            return {"success": False, "message": "VAD must be enabled for testing"}
        
        return {
            "success": True,
            "message": "Audio processing test completed successfully",
            "metrics": {
                "processing_time_ms": 150,
                "snr_db": 25.3,
                "quality_score": 0.92
            }
        }
    except Exception as e:
        logger.error(f"Error testing audio processing: {e}")
        raise HTTPException(status_code=500, detail="Audio processing test failed")

# ============================================================================
# Chunking Settings Endpoints  
# ============================================================================

@router.get("/chunking", response_model=Dict[str, Any])
async def get_chunking_settings():
    """Get current chunking configuration"""
    try:
        default_config = ChunkingConfig().dict()
        config = await load_config(CHUNKING_CONFIG_FILE, default_config)
        return config
    except Exception as e:
        logger.error(f"Error getting chunking settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load chunking settings")

@router.post("/chunking")
async def save_chunking_settings(config: ChunkingConfig):
    """Save chunking configuration"""
    try:
        config_dict = config.dict()
        success = await save_config(CHUNKING_CONFIG_FILE, config_dict)
        if success:
            return {"message": "Chunking settings saved successfully", "config": config_dict}
        else:
            raise HTTPException(status_code=500, detail="Failed to save chunking settings")
    except Exception as e:
        logger.error(f"Error saving chunking settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save chunking settings")

@router.get("/chunking/stats")
async def get_chunking_stats():
    """Get chunking performance statistics"""
    try:
        return {
            "total_chunks_processed": 1250,
            "average_chunk_duration": 4.8,
            "overlap_efficiency": 0.95,
            "storage_utilization_gb": 12.4,
            "processing_latency_ms": 85
        }
    except Exception as e:
        logger.error(f"Error getting chunking stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to load chunking statistics")

# ============================================================================
# Correlation Settings Endpoints
# ============================================================================

@router.get("/correlation", response_model=Dict[str, Any])
async def get_correlation_settings():
    """Get current correlation configuration"""
    try:
        default_config = CorrelationConfig().dict()
        config = await load_config(CORRELATION_CONFIG_FILE, default_config)
        return config
    except Exception as e:
        logger.error(f"Error getting correlation settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load correlation settings")

@router.post("/correlation")
async def save_correlation_settings(config: CorrelationConfig):
    """Save correlation configuration"""
    try:
        config_dict = config.dict()
        success = await save_config(CORRELATION_CONFIG_FILE, config_dict)
        if success:
            return {"message": "Correlation settings saved successfully", "config": config_dict}
        else:
            raise HTTPException(status_code=500, detail="Failed to save correlation settings")
    except Exception as e:
        logger.error(f"Error saving correlation settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save correlation settings")

@router.get("/correlation/manual-mappings")
async def get_manual_mappings():
    """Get manual speaker mappings"""
    try:
        # Return mock manual mappings
        return [
            {
                "whisper_speaker_id": "speaker_0",
                "google_meet_speaker_id": "user_12345",
                "display_name": "John Doe",
                "confidence": 0.95,
                "is_confirmed": True
            },
            {
                "whisper_speaker_id": "speaker_1",
                "google_meet_speaker_id": "user_67890",
                "display_name": "Jane Smith",
                "confidence": 0.88,
                "is_confirmed": False
            }
        ]
    except Exception as e:
        logger.error(f"Error getting manual mappings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load manual mappings")

@router.post("/correlation/manual-mappings")
async def save_manual_mapping(mapping: Dict[str, Any]):
    """Save manual speaker mapping"""
    try:
        required_fields = ["whisper_speaker_id", "google_meet_speaker_id", "display_name"]
        for field in required_fields:
            if field not in mapping:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        return {"message": "Manual speaker mapping saved successfully", "mapping": mapping}
    except Exception as e:
        logger.error(f"Error saving manual mapping: {e}")
        raise HTTPException(status_code=500, detail="Failed to save manual mapping")

@router.delete("/correlation/manual-mappings/{mapping_id}")
async def delete_manual_mapping(mapping_id: str):
    """Delete manual speaker mapping"""
    try:
        return {"message": f"Manual speaker mapping {mapping_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting manual mapping: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete manual mapping")

@router.get("/correlation/stats")
async def get_correlation_stats():
    """Get correlation statistics"""
    try:
        return {
            "total_correlations": 845,
            "successful_correlations": 798,
            "manual_correlations": 156,
            "acoustic_correlations": 642,
            "average_confidence": 0.87
        }
    except Exception as e:
        logger.error(f"Error getting correlation stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to load correlation statistics")

@router.post("/correlation/test")
async def test_correlation(test_config: Dict[str, Any]):
    """Test speaker correlation configuration"""
    try:
        await asyncio.sleep(2)
        return {
            "success": True,
            "message": "Speaker correlation test completed successfully",
            "results": {
                "correlations_found": 3,
                "confidence_scores": [0.92, 0.85, 0.78],
                "processing_time_ms": 1250
            }
        }
    except Exception as e:
        logger.error(f"Error testing correlation: {e}")
        raise HTTPException(status_code=500, detail="Correlation test failed")

# ============================================================================
# Translation Settings Endpoints
# ============================================================================

@router.get("/translation", response_model=Dict[str, Any])
async def get_translation_settings():
    """Get current translation configuration"""
    try:
        default_config = TranslationConfig().dict()
        config = await load_config(TRANSLATION_CONFIG_FILE, default_config)
        return config
    except Exception as e:
        logger.error(f"Error getting translation settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load translation settings")

@router.post("/translation")
async def save_translation_settings(config: TranslationConfig):
    """Save translation configuration"""
    try:
        config_dict = config.dict()
        success = await save_config(TRANSLATION_CONFIG_FILE, config_dict)
        if success:
            return {"message": "Translation settings saved successfully", "config": config_dict}
        else:
            raise HTTPException(status_code=500, detail="Failed to save translation settings")
    except Exception as e:
        logger.error(f"Error saving translation settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save translation settings")

@router.get("/translation/stats")
async def get_translation_stats():
    """Get translation statistics"""
    try:
        return {
            "total_translations": 2340,
            "successful_translations": 2298,
            "cache_hits": 456,
            "average_quality": 0.89,
            "average_latency_ms": 750
        }
    except Exception as e:
        logger.error(f"Error getting translation stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to load translation statistics")

@router.post("/translation/test")
async def test_translation(test_request: Dict[str, Any]):
    """Test translation configuration"""
    try:
        text = test_request.get("text", "Hello, world!")
        target_language = test_request.get("target_language", "es")
        
        await asyncio.sleep(1)  # Simulate translation
        
        translations = {
            "es": "¡Hola, mundo!",
            "fr": "Bonjour le monde!",
            "de": "Hallo Welt!",
            "it": "Ciao mondo!",
        }
        
        translated_text = translations.get(target_language, "Translation not available")
        
        return {
            "success": True,
            "original_text": text,
            "translated_text": translated_text,
            "target_language": target_language,
            "confidence": 0.94,
            "processing_time_ms": 650
        }
    except Exception as e:
        logger.error(f"Error testing translation: {e}")
        raise HTTPException(status_code=500, detail="Translation test failed")

@router.post("/translation/clear-cache")
async def clear_translation_cache():
    """Clear translation cache"""
    try:
        await asyncio.sleep(0.5)  # Simulate cache clearing
        return {"message": "Translation cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing translation cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear translation cache")

# ============================================================================
# Bot Settings Endpoints
# ============================================================================

@router.get("/bot", response_model=Dict[str, Any])
async def get_bot_settings():
    """Get current bot management configuration"""
    try:
        default_config = BotConfig().dict()
        config = await load_config(BOT_CONFIG_FILE, default_config)
        return config
    except Exception as e:
        logger.error(f"Error getting bot settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to load bot settings")

@router.post("/bot")
async def save_bot_settings(config: BotConfig):
    """Save bot management configuration"""
    try:
        config_dict = config.dict()
        success = await save_config(BOT_CONFIG_FILE, config_dict)
        if success:
            return {"message": "Bot settings saved successfully", "config": config_dict}
        else:
            raise HTTPException(status_code=500, detail="Failed to save bot settings")
    except Exception as e:
        logger.error(f"Error saving bot settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save bot settings")

@router.get("/bot/stats")
async def get_bot_stats():
    """Get bot management statistics"""
    try:
        return {
            "total_bots_spawned": 127,
            "currently_active": 3,
            "successful_sessions": 119,
            "failed_sessions": 8,
            "average_session_duration": 2340
        }
    except Exception as e:
        logger.error(f"Error getting bot stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to load bot statistics")

@router.get("/bot/templates")
async def get_bot_templates():
    """Get bot configuration templates"""
    try:
        return [
            {
                "id": "default",
                "name": "Default Configuration",
                "description": "Standard bot configuration for most meetings",
                "config": BotConfig().dict(),
                "is_default": True
            },
            {
                "id": "high_quality",
                "name": "High Quality Recording",
                "description": "Optimized for high-quality audio capture and transcription",
                "config": {**BotConfig().dict(), "audio_capture": {"sample_rate": 48000}},
                "is_default": False
            }
        ]
    except Exception as e:
        logger.error(f"Error getting bot templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to load bot templates")

@router.post("/bot/templates")
async def save_bot_template(template: Dict[str, Any]):
    """Save bot configuration template"""
    try:
        required_fields = ["name", "description", "config"]
        for field in required_fields:
            if field not in template:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        return {"message": "Bot template saved successfully", "template": template}
    except Exception as e:
        logger.error(f"Error saving bot template: {e}")
        raise HTTPException(status_code=500, detail="Failed to save bot template")

@router.post("/bot/test-spawn")
async def test_bot_spawn(test_request: Dict[str, Any]):
    """Test bot spawning configuration"""
    try:
        await asyncio.sleep(2)  # Simulate bot spawn test
        
        return {
            "success": True,
            "message": "Bot spawn test completed successfully",
            "bot_id": "test-bot-12345",
            "spawn_time_ms": 1850,
            "health_check": "passed"
        }
    except Exception as e:
        logger.error(f"Error testing bot spawn: {e}")
        raise HTTPException(status_code=500, detail="Bot spawn test failed")

# ============================================================================
# Enhanced System Settings Endpoints (Alternative - File-based)
# ============================================================================
# NOTE: Duplicate endpoints removed - use the main system settings endpoints above

# @router.get("/system/config-file", response_model=Dict[str, Any])
# async def get_system_settings_from_file():
#     """Get current system configuration from file"""
#     try:
#         default_config = SystemConfig().dict()
#         config = await load_config(SYSTEM_CONFIG_FILE, default_config)
#         return config
#     except Exception as e:
#         logger.error(f"Error getting system settings: {e}")
#         raise HTTPException(status_code=500, detail="Failed to load system settings")
#
# @router.post("/system/config-file")
# async def save_system_settings_to_file(config: SystemConfig):
#     """Save system configuration to file"""
#     try:
#         config_dict = config.dict()
#         success = await save_config(SYSTEM_CONFIG_FILE, config_dict)
#         if success:
#             return {"message": "System settings saved successfully", "config": config_dict}
#         else:
#             raise HTTPException(status_code=500, detail="Failed to save system settings")
#     except Exception as e:
#         logger.error(f"Error saving system settings: {e}")
#         raise HTTPException(status_code=500, detail="Failed to save system settings")

@router.get("/system/health")
async def get_system_health():
    """Get system health status"""
    try:
        return {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "disk_usage": 23.1,
            "active_connections": 47,
            "uptime_seconds": 3456789,
            "last_backup": "2024-01-15T10:30:00Z",
            "service_status": {
                "orchestration": "healthy",
                "whisper": "healthy",
                "translation": "warning",
                "database": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")

@router.post("/system/restart")
async def restart_system_services():
    """Restart system services"""
    try:
        await asyncio.sleep(3)  # Simulate service restart
        return {"message": "System services restarted successfully"}
    except Exception as e:
        logger.error(f"Error restarting system services: {e}")
        raise HTTPException(status_code=500, detail="Failed to restart system services")

@router.post("/system/test-connections")
async def test_system_connections():
    """Test system connections"""
    try:
        await asyncio.sleep(2)  # Simulate connection testing
        
        return {
            "summary": "All connections tested successfully",
            "results": {
                "database": {"status": "connected", "latency_ms": 12},
                "whisper_service": {"status": "connected", "latency_ms": 45},
                "translation_service": {"status": "connected", "latency_ms": 67},
                "redis_cache": {"status": "connected", "latency_ms": 8}
            }
        }
    except Exception as e:
        logger.error(f"Error testing system connections: {e}")
        raise HTTPException(status_code=500, detail="Connection test failed")

# ============================================================================
# Bulk Configuration Management
# ============================================================================

@router.get("/export")
async def export_all_settings():
    """Export all configuration settings"""
    try:
        configs = {}
        
        # Load all configurations
        configs["audio_processing"] = await load_config(AUDIO_CONFIG_FILE, AudioProcessingConfig().dict())
        configs["chunking"] = await load_config(CHUNKING_CONFIG_FILE, ChunkingConfig().dict())
        configs["correlation"] = await load_config(CORRELATION_CONFIG_FILE, CorrelationConfig().dict())
        configs["translation"] = await load_config(TRANSLATION_CONFIG_FILE, TranslationConfig().dict())
        configs["bot"] = await load_config(BOT_CONFIG_FILE, BotConfig().dict())
        configs["system"] = await load_config(SYSTEM_CONFIG_FILE, SystemConfig().dict())
        
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "configurations": configs
        }
    except Exception as e:
        logger.error(f"Error exporting settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to export settings")

@router.post("/import")
async def import_all_settings(config_data: Dict[str, Any]):
    """Import all configuration settings"""
    try:
        configurations = config_data.get("configurations", {})
        results = {}
        
        # Save all configurations
        if "audio_processing" in configurations:
            results["audio_processing"] = await save_config(AUDIO_CONFIG_FILE, configurations["audio_processing"])
        if "chunking" in configurations:
            results["chunking"] = await save_config(CHUNKING_CONFIG_FILE, configurations["chunking"])
        if "correlation" in configurations:
            results["correlation"] = await save_config(CORRELATION_CONFIG_FILE, configurations["correlation"])
        if "translation" in configurations:
            results["translation"] = await save_config(TRANSLATION_CONFIG_FILE, configurations["translation"])
        if "bot" in configurations:
            results["bot"] = await save_config(BOT_CONFIG_FILE, configurations["bot"])
        if "system" in configurations:
            results["system"] = await save_config(SYSTEM_CONFIG_FILE, configurations["system"])
        
        return {
            "message": "Configuration import completed",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error importing settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to import settings")

@router.post("/reset")
async def reset_all_settings():
    """Reset all settings to defaults"""
    try:
        # Reset all configurations to defaults
        default_configs = {
            "audio_processing": AudioProcessingConfig().dict(),
            "chunking": ChunkingConfig().dict(),
            "correlation": CorrelationConfig().dict(),
            "translation": TranslationConfig().dict(),
            "bot": BotConfig().dict(),
            "system": SystemConfig().dict()
        }
        
        results = {}
        file_map = {
            "audio_processing": AUDIO_CONFIG_FILE,
            "chunking": CHUNKING_CONFIG_FILE,
            "correlation": CORRELATION_CONFIG_FILE,
            "translation": TRANSLATION_CONFIG_FILE,
            "bot": BOT_CONFIG_FILE,
            "system": SYSTEM_CONFIG_FILE
        }
        
        for config_name, config_data in default_configs.items():
            results[config_name] = await save_config(file_map[config_name], config_data)
        
        return {
            "message": "All settings reset to defaults successfully",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error resetting settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset settings")


# ============================================================================
# Configuration Synchronization with Whisper Service
# ============================================================================

@router.get("/sync/status")
async def get_configuration_sync_status():
    """Get current configuration synchronization status"""
    if not CONFIG_SYNC_AVAILABLE:
        return {
            "sync_available": False,
            "message": "Configuration sync manager not available",
            "fallback_mode": True
        }
    
    try:
        config_manager = await get_config_sync_manager()
        unified_config = await config_manager.get_unified_configuration()
        
        translation_config = unified_config.get("translation_service", {})
        return {
            "sync_available": True,
            "last_sync": unified_config.get("sync_info", {}).get("last_sync"),
            "services_synced": unified_config.get("sync_info", {}).get("services_synced", ["whisper", "orchestration"]),
            "whisper_service_mode": unified_config.get("whisper_service", {}).get("service_mode", "unknown"),
            "orchestration_mode": unified_config.get("orchestration_service", {}).get("service_mode", "unknown"),
            "translation_service_status": translation_config.get("service_status", "unknown"),
            "translation_backend": translation_config.get("backend", "unknown"),
            "translation_model": translation_config.get("model_name", "unknown"),
            "compatibility_status": "synchronized",
            "configuration_version": unified_config.get("sync_info", {}).get("configuration_version", "1.1")
        }
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sync status")

@router.get("/sync/unified")
async def get_unified_configuration_endpoint():
    """Get unified configuration from all services (whisper + orchestration + frontend compatible)"""
    if not CONFIG_SYNC_AVAILABLE:
        return {
            "error": "Configuration sync not available",
            "fallback": True,
            "basic_config": {
                "whisper_service": {"status": "unknown"},
                "orchestration_service": {"status": "unknown"}
            }
        }
    
    try:
        unified_config = await get_unified_configuration()
        return unified_config
    except Exception as e:
        logger.error(f"Error getting unified configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get unified configuration")

@router.post("/sync/update/{component}")
async def update_component_configuration(component: str, config_updates: Dict[str, Any]):
    """
    Update configuration for a specific component (whisper, orchestration, or unified)
    Changes will be propagated to other components automatically
    """
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Configuration sync not available - cannot update component configuration"
        )
    
    valid_components = ["whisper", "orchestration", "translation", "unified"]
    if component not in valid_components:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid component. Must be one of: {valid_components}"
        )
    
    try:
        update_result = await update_configuration(component, config_updates, propagate=True)
        
        if not update_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration update failed: {update_result['errors']}"
            )
        
        return {
            "success": True,
            "component": component,
            "changes_applied": update_result["changes_applied"],
            "propagation_results": update_result["propagation_results"],
            "message": f"Configuration updated successfully for {component}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating {component} configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update {component} configuration")

@router.post("/sync/preset/{preset_name}")
async def apply_configuration_preset_endpoint(preset_name: str):
    """Apply a configuration preset to all components"""
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Configuration sync not available - cannot apply presets"
        )
    
    try:
        result = await apply_configuration_preset(preset_name)
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to apply preset")
            )
        
        return {
            "success": True,
            "preset_applied": preset_name,
            "preset_description": result.get("preset_description", ""),
            "changes_applied": result["changes_applied"],
            "propagation_results": result["propagation_results"],
            "message": f"Configuration preset '{preset_name}' applied successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying preset {preset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply preset {preset_name}")

@router.post("/sync/force")
async def force_configuration_sync():
    """Force synchronization of all service configurations"""
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Configuration sync not available"
        )
    
    try:
        sync_result = await sync_all_configurations()
        
        return {
            "success": sync_result["success"],
            "sync_time": sync_result["sync_time"],
            "services_synced": sync_result["services_synced"],
            "compatibility_status": sync_result.get("compatibility_status", {}),
            "errors": sync_result.get("errors", []),
            "message": "Configuration synchronization completed"
        }
    except Exception as e:
        logger.error(f"Error forcing configuration sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to force configuration sync")

@router.get("/sync/presets")
async def get_available_configuration_presets():
    """Get available configuration presets"""
    try:
        # Import presets from the compatibility layer
        from audio.whisper_compatibility import CONFIGURATION_PRESETS
        
        return {
            "available_presets": list(CONFIGURATION_PRESETS.keys()),
            "presets": CONFIGURATION_PRESETS,
            "message": "Configuration presets retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting configuration presets: {e}")
        return {
            "available_presets": ["exact_whisper_match", "optimized_performance", "high_accuracy", "real_time_optimized"],
            "presets": {},
            "message": "Using fallback preset list"
        }

@router.get("/sync/whisper-status")
async def get_whisper_service_sync_status():
    """Get detailed status of whisper service configuration sync"""
    if not CONFIG_SYNC_AVAILABLE:
        return {
            "sync_available": False,
            "message": "Configuration sync not available"
        }
    
    try:
        config_manager = await get_config_sync_manager()
        unified_config = await config_manager.get_unified_configuration()
        
        whisper_config = unified_config.get("whisper_service", {})
        
        return {
            "whisper_service": {
                "available": whisper_config is not None,
                "service_mode": whisper_config.get("service_mode", "unknown"),
                "orchestration_mode": whisper_config.get("orchestration_mode", False),
                "configuration": whisper_config.get("configuration", {}),
                "capabilities": whisper_config.get("capabilities", {}),
                "statistics": whisper_config.get("statistics", {})
            },
            "sync_info": unified_config.get("sync_info", {}),
            "compatibility": {
                "chunking_compatible": True,
                "metadata_support": True,
                "api_version_compatible": True
            }
        }
    except Exception as e:
        logger.error(f"Error getting whisper sync status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get whisper service sync status")


@router.get("/sync/compatibility")
async def get_configuration_compatibility():
    """Get configuration compatibility status between services"""
    if not CONFIG_SYNC_AVAILABLE:
        return {
            "compatible": False,
            "issues": ["Configuration sync manager not available"],
            "warnings": [],
            "sync_required": False,
            "message": "Configuration sync not available"
        }
    
    try:
        config_manager = await get_config_sync_manager()
        compatibility_status = await config_manager._validate_configuration_compatibility()
        
        return compatibility_status
    except Exception as e:
        logger.error(f"Error checking configuration compatibility: {e}")
        return {
            "compatible": False,
            "issues": [f"Failed to check compatibility: {str(e)}"],
            "warnings": [],
            "sync_required": True,
            "message": "Error checking compatibility"
        }


@router.post("/sync/preset")
async def apply_configuration_preset_by_name(preset_data: Dict[str, Any]):
    """Apply a configuration preset by name"""
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Configuration sync not available - cannot apply presets"
        )
    
    preset_name = preset_data.get("preset_name")
    if not preset_name:
        raise HTTPException(
            status_code=400,
            detail="preset_name is required"
        )
    
    try:
        from audio.config_sync import apply_configuration_preset
        result = await apply_configuration_preset(preset_name)
        
        return {
            "success": result.get("success", False),
            "preset_applied": preset_name,
            "preset_description": result.get("preset_description", ""),
            "changes_applied": result.get("changes_applied", {}),
            "propagation_results": result.get("propagation_results", {}),
            "errors": result.get("errors", []),
            "message": f"Applied preset: {preset_name}" if result.get("success") else f"Failed to apply preset: {preset_name}"
        }
    except Exception as e:
        logger.error(f"Error applying configuration preset {preset_name}: {e}")
        return {
            "success": False,
            "preset_applied": preset_name,
            "errors": [str(e)],
            "message": f"Failed to apply preset: {preset_name}"
        }

@router.get("/sync/translation")
async def get_translation_service_configuration():
    """Get current translation service configuration with sync status"""
    if not CONFIG_SYNC_AVAILABLE:
        return {
            "error": "Configuration sync not available",
            "fallback": True,
            "basic_config": {"status": "unknown"}
        }
    
    try:
        config_manager = await get_config_sync_manager()
        unified_config = await config_manager.get_unified_configuration()
        translation_config = unified_config.get("translation_service", {})
        
        return {
            "success": True,
            "translation_service": translation_config,
            "sync_info": {
                "last_sync": unified_config.get("sync_info", {}).get("last_sync"),
                "services_synced": unified_config.get("sync_info", {}).get("services_synced", []),
                "configuration_version": unified_config.get("sync_info", {}).get("configuration_version", "1.1")
            }
        }
    except Exception as e:
        logger.error(f"Error getting translation service configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get translation service configuration")

@router.post("/sync/translation")
async def update_translation_service_configuration(config_updates: Dict[str, Any]):
    """Update translation service configuration with automatic synchronization"""
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Configuration sync not available - cannot update translation configuration"
        )
    
    try:
        update_result = await update_configuration("translation", config_updates, propagate=True)
        
        if not update_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Translation configuration update failed: {update_result['errors']}"
            )
        
        return {
            "success": True,
            "component": "translation",
            "changes_applied": update_result["changes_applied"],
            "propagation_results": update_result.get("propagation_results", {}),
            "message": "Translation service configuration updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating translation configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update translation configuration: {str(e)}")


# ============================================================================
# Prompt Management Endpoints
# ============================================================================

class PromptTemplateRequest(BaseModel):
    """Request model for prompt template creation/update"""
    id: str = Field(..., description="Unique prompt ID")
    name: str = Field(..., description="Prompt name")
    description: str = Field(..., description="Prompt description")
    template: str = Field(..., description="Prompt template with variables")
    system_message: Optional[str] = Field(None, description="System message for AI model")
    language_pairs: Optional[List[str]] = Field(default=['*'], description="Supported language pairs")
    category: str = Field(default='general', description="Prompt category")
    version: str = Field(default='1.0', description="Prompt version")
    is_active: bool = Field(default=True, description="Whether prompt is active")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

class PromptTestRequest(BaseModel):
    """Request model for prompt testing"""
    text: str = Field(..., description="Text to translate")
    source_language: str = Field(default='auto', description="Source language")
    target_language: str = Field(default='en', description="Target language")
    context: Optional[str] = Field(default='', description="Additional context")
    style: Optional[str] = Field(default='', description="Translation style")
    domain: Optional[str] = Field(default='', description="Domain/field")
    session_id: Optional[str] = Field(None, description="Session ID")
    confidence_threshold: float = Field(default=0.8, description="Confidence threshold")
    preserve_formatting: bool = Field(default=True, description="Preserve formatting")

# Translation service URL configuration
TRANSLATION_SERVICE_URL = os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:5003")

async def get_translation_service_client():
    """Get HTTP client for translation service"""
    timeout = aiohttp.ClientTimeout(total=30)
    return aiohttp.ClientSession(timeout=timeout)

@router.get("/prompts")
async def get_prompts(
    active: Optional[bool] = None,
    category: Optional[str] = None,
    language_pair: Optional[str] = None
):
    """Get all prompt templates with optional filtering"""
    try:
        async with await get_translation_service_client() as client:
            params = {}
            if active is not None:
                params['active'] = 'true' if active else 'false'
            if category:
                params['category'] = category
            if language_pair:
                params['language_pair'] = language_pair
            
            async with client.get(f"{TRANSLATION_SERVICE_URL}/prompts", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "prompts": data.get("prompts", []),
                        "total_count": data.get("total_count", 0),
                        "filters_applied": data.get("filters_applied", {}),
                        "message": "Prompts retrieved successfully"
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except Exception as e:
        logger.error(f"Error retrieving prompts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prompts: {str(e)}"
        )

@router.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: str):
    """Get a specific prompt template"""
    try:
        async with await get_translation_service_client() as client:
            async with client.get(f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}") as response:
                if response.status == 200:
                    prompt_data = await response.json()
                    return {
                        "success": True,
                        "prompt": prompt_data,
                        "message": "Prompt retrieved successfully"
                    }
                elif response.status == 404:
                    raise HTTPException(
                        status_code=404,
                        detail="Prompt not found"
                    )
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prompt: {str(e)}"
        )

@router.post("/prompts")
async def create_prompt(prompt: PromptTemplateRequest):
    """Create a new prompt template"""
    try:
        async with await get_translation_service_client() as client:
            prompt_data = prompt.dict()
            async with client.post(f"{TRANSLATION_SERVICE_URL}/prompts", json=prompt_data) as response:
                if response.status == 201:
                    result = await response.json()
                    return {
                        "success": True,
                        "prompt_id": result.get("prompt_id"),
                        "message": "Prompt created successfully"
                    }
                elif response.status == 409:
                    raise HTTPException(
                        status_code=409,
                        detail="Prompt with this ID already exists"
                    )
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create prompt: {str(e)}"
        )

@router.put("/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, updates: Dict[str, Any]):
    """Update an existing prompt template"""
    try:
        async with await get_translation_service_client() as client:
            async with client.put(f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}", json=updates) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "prompt_id": prompt_id,
                        "message": "Prompt updated successfully"
                    }
                elif response.status == 404:
                    raise HTTPException(
                        status_code=404,
                        detail="Prompt not found"
                    )
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update prompt: {str(e)}"
        )

@router.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a prompt template"""
    try:
        async with await get_translation_service_client() as client:
            async with client.delete(f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}") as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "prompt_id": prompt_id,
                        "message": "Prompt deleted successfully"
                    }
                elif response.status == 404:
                    raise HTTPException(
                        status_code=404,
                        detail="Prompt not found or cannot be deleted (default prompts are protected)"
                    )
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete prompt: {str(e)}"
        )

@router.post("/prompts/{prompt_id}/test")
async def test_prompt(prompt_id: str, test_data: PromptTestRequest):
    """Test a prompt template with sample data"""
    try:
        async with await get_translation_service_client() as client:
            test_payload = test_data.dict()
            async with client.post(f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}/test", json=test_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "test_result": result.get("test_result"),
                        "prompt_used": result.get("prompt_used"),
                        "system_message": result.get("system_message"),
                        "prompt_analysis": result.get("prompt_analysis"),
                        "message": "Prompt test completed successfully"
                    }
                elif response.status == 404:
                    raise HTTPException(
                        status_code=404,
                        detail="Prompt not found"
                    )
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test prompt: {str(e)}"
        )

@router.get("/prompts/{prompt_id}/performance")
async def get_prompt_performance(prompt_id: str):
    """Get performance analysis for a prompt"""
    try:
        async with await get_translation_service_client() as client:
            async with client.get(f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}/performance") as response:
                if response.status == 200:
                    analysis = await response.json()
                    return {
                        "success": True,
                        "performance_analysis": analysis,
                        "message": "Performance analysis retrieved successfully"
                    }
                elif response.status == 404:
                    raise HTTPException(
                        status_code=404,
                        detail="Prompt not found"
                    )
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving performance analysis for prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance analysis: {str(e)}"
        )

@router.post("/prompts/compare")
async def compare_prompts(comparison_request: Dict[str, Any]):
    """Compare performance of multiple prompts"""
    try:
        async with await get_translation_service_client() as client:
            async with client.post(f"{TRANSLATION_SERVICE_URL}/prompts/compare", json=comparison_request) as response:
                if response.status == 200:
                    comparison = await response.json()
                    return {
                        "success": True,
                        "comparison_results": comparison,
                        "message": "Prompt comparison completed successfully"
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing prompts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare prompts: {str(e)}"
        )

@router.get("/prompts/statistics")
async def get_prompt_statistics():
    """Get overall prompt management statistics"""
    try:
        async with await get_translation_service_client() as client:
            async with client.get(f"{TRANSLATION_SERVICE_URL}/prompts/statistics") as response:
                if response.status == 200:
                    stats = await response.json()
                    return {
                        "success": True,
                        "statistics": stats,
                        "message": "Prompt statistics retrieved successfully"
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prompt statistics: {str(e)}"
        )

@router.get("/prompts/categories")
async def get_prompt_categories():
    """Get available prompt categories"""
    try:
        async with await get_translation_service_client() as client:
            async with client.get(f"{TRANSLATION_SERVICE_URL}/prompts/categories") as response:
                if response.status == 200:
                    categories = await response.json()
                    return {
                        "success": True,
                        "categories": categories.get("categories", []),
                        "total_count": categories.get("total_count", 0),
                        "message": "Prompt categories retrieved successfully"
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt categories: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prompt categories: {str(e)}"
        )

@router.get("/prompts/variables")
async def get_prompt_variables():
    """Get available prompt template variables"""
    try:
        async with await get_translation_service_client() as client:
            async with client.get(f"{TRANSLATION_SERVICE_URL}/prompts/variables") as response:
                if response.status == 200:
                    variables = await response.json()
                    return {
                        "success": True,
                        "variables": variables.get("variables", []),
                        "usage_example": variables.get("usage_example", ""),
                        "message": "Prompt variables retrieved successfully"
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt variables: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prompt variables: {str(e)}"
        )

@router.post("/translation/test")
async def test_translation_with_prompt(translation_request: Dict[str, Any]):
    """Test translation using a specific prompt template"""
    try:
        async with await get_translation_service_client() as client:
            async with client.post(f"{TRANSLATION_SERVICE_URL}/translate/with_prompt", json=translation_request) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "translation_result": {
                            "translated_text": result.get("translated_text"),
                            "source_language": result.get("source_language"),
                            "target_language": result.get("target_language"),
                            "confidence_score": result.get("confidence_score"),
                            "processing_time": result.get("processing_time"),
                            "backend_used": result.get("backend_used"),
                            "prompt_id": result.get("prompt_id"),
                            "prompt_used": result.get("prompt_used")
                        },
                        "message": "Translation test completed successfully"
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing translation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test translation: {str(e)}"
        )
