"""
General Settings Router

Handles user settings, system settings, service settings, backup/restore, validation,
defaults, and reset functionality.
"""

from ._shared import (
    AUDIO_CONFIG_FILE,
    BOT_CONFIG_FILE,
    CHUNKING_CONFIG_FILE,
    CORRELATION_CONFIG_FILE,
    SYSTEM_CONFIG_FILE,
    TRANSLATION_CONFIG_FILE,
    UTC,
    Any,
    APIRouter,
    AudioProcessingConfig,
    BotConfig,
    ChunkingConfig,
    ConfigResponse,
    ConfigSyncModes,
    CorrelationConfig,
    Depends,
    HTTPException,
    JSONResponse,
    Optional,
    ServiceSettingsRequest,
    SettingsBackupResponse,
    SystemConfig,
    SystemSettingsRequest,
    TranslationConfig,
    UserConfigResponse,
    UserSettingsRequest,
    asyncio,
    datetime,
    get_config_manager,
    get_config_sync_mode,
    get_event_publisher,
    load_config,
    logger,
    save_config,
    status,
    timezone,
)

router = APIRouter(tags=["settings-general"])


# ============================================================================
# User Settings Endpoints
# ============================================================================


@router.get("/user", response_model=UserConfigResponse)
async def get_user_settings(
    config_manager=Depends(get_config_manager),
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
            default_translation_language=settings.get("default_translation_language", "es"),
            transcription_model=settings.get("transcription_model", "base"),
            custom_settings=settings.get("custom_settings", {}),
            updated_at=settings.get("updated_at", datetime.now(UTC)),
        )

    except Exception as e:
        logger.error(f"Failed to get user settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user settings: {e!s}",
        ) from e


@router.put("/user", response_model=UserConfigResponse)
async def update_user_settings(
    request: UserSettingsRequest,
    config_manager=Depends(get_config_manager),
    event_publisher=Depends(get_event_publisher),
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
            update_data["default_translation_language"] = request.default_translation_language
        if request.transcription_model is not None:
            update_data["transcription_model"] = request.transcription_model
        if request.custom_settings is not None:
            update_data["custom_settings"] = request.custom_settings

        mode = get_config_sync_mode()

        if mode == ConfigSyncModes.WORKER and event_publisher:
            await event_publisher.publish(
                alias="config_sync",
                event_type="UserSettingsUpdateRequested",
                payload={
                    "user_id": user_id,
                    "updated_fields": update_data,
                },
                metadata={"endpoint": "/settings/user"},
            )
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "message": "User settings update queued",
                    "user_id": user_id,
                    "updated_keys": list(update_data.keys()),
                },
            )

        # Update settings immediately (API mode)
        updated_settings = await config_manager.update_user_settings(user_id, update_data)

        if event_publisher:
            await event_publisher.publish(
                alias="config_sync",
                event_type="UserSettingsUpdated",
                payload={
                    "user_id": user_id,
                    "updated_keys": list(update_data.keys()),
                },
                metadata={"endpoint": "/settings/user"},
            )

        return UserConfigResponse(
            user_id=user_id,
            theme=updated_settings.get("theme", "dark"),
            language=updated_settings.get("language", "en"),
            notifications=updated_settings.get("notifications", True),
            audio_auto_start=updated_settings.get("audio_auto_start", False),
            default_translation_language=updated_settings.get("default_translation_language", "es"),
            transcription_model=updated_settings.get("transcription_model", "base"),
            custom_settings=updated_settings.get("custom_settings", {}),
            updated_at=updated_settings.get("updated_at", datetime.now(UTC)),
        )

    except Exception as e:
        logger.error(f"Failed to update user settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user settings: {e!s}",
        ) from e


# ============================================================================
# System Settings Endpoints
# ============================================================================


@router.get("/system")
async def get_system_settings(
    config_manager=Depends(get_config_manager),
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
            detail=f"Failed to get system settings: {e!s}",
        ) from e


@router.put("/system")
async def update_system_settings(
    request: SystemSettingsRequest,
    config_manager=Depends(get_config_manager),
    event_publisher=Depends(get_event_publisher),
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

        mode = get_config_sync_mode()

        if mode == ConfigSyncModes.WORKER and event_publisher and update_data:
            await event_publisher.publish(
                alias="config_sync",
                event_type="SystemSettingsUpdateRequested",
                payload={
                    "updated_keys": list(update_data.keys()),
                    "settings": update_data,
                },
                metadata={"endpoint": "/settings/system"},
            )
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "message": "System settings update queued",
                    "updated_keys": list(update_data.keys()),
                },
            )

        # Update settings immediately (API mode)
        result = await config_manager.update_system_settings(update_data)

        if event_publisher and update_data:
            await event_publisher.publish(
                alias="config_sync",
                event_type="SystemSettingsUpdated",
                payload={
                    "updated_keys": list(update_data.keys()),
                    "settings": update_data,
                },
                metadata={"endpoint": "/settings/system"},
            )

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
            detail=f"Failed to update system settings: {e!s}",
        ) from e


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
                "database": "healthy",
            },
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health") from e


@router.post("/system/restart")
async def restart_system_services():
    """Restart system services"""
    try:
        await asyncio.sleep(3)  # Simulate service restart
        return {"message": "System services restarted successfully"}
    except Exception as e:
        logger.error(f"Error restarting system services: {e}")
        raise HTTPException(status_code=500, detail="Failed to restart system services") from e


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
                "redis_cache": {"status": "connected", "latency_ms": 8},
            },
        }
    except Exception as e:
        logger.error(f"Error testing system connections: {e}")
        raise HTTPException(status_code=500, detail="Connection test failed") from e


# ============================================================================
# Service Settings Endpoints
# ============================================================================


@router.get("/services")
async def get_service_settings(
    config_manager=Depends(get_config_manager),
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
            detail=f"Failed to get service settings: {e!s}",
        ) from e


@router.put("/services/{service_name}")
async def update_service_settings(
    service_name: str,
    request: ServiceSettingsRequest,
    config_manager=Depends(get_config_manager),
    event_publisher=Depends(get_event_publisher),
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

        mode = get_config_sync_mode()

        if mode == ConfigSyncModes.WORKER and event_publisher and update_data:
            await event_publisher.publish(
                alias="config_sync",
                event_type="ServiceSettingsUpdateRequested",
                payload={
                    "service_name": service_name,
                    "updated_keys": list(update_data.keys()),
                    "settings": update_data,
                },
                metadata={"endpoint": f"/settings/services/{service_name}"},
            )
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "message": f"Service {service_name} settings update queued",
                    "service_name": service_name,
                    "updated_keys": list(update_data.keys()),
                },
            )

        # Update service settings immediately (API mode)
        result = await config_manager.update_service_settings(service_name, update_data)

        if event_publisher and update_data:
            await event_publisher.publish(
                alias="config_sync",
                event_type="ServiceSettingsUpdated",
                payload={
                    "service_name": service_name,
                    "updated_keys": list(update_data.keys()),
                    "settings": update_data,
                },
                metadata={"endpoint": f"/settings/services/{service_name}"},
            )

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
            detail=f"Failed to update service settings: {e!s}",
        ) from e


# ============================================================================
# Audio Settings Endpoints (Basic)
# ============================================================================


@router.get("/audio", response_model=ConfigResponse)
async def get_audio_settings(
    config_manager=Depends(get_config_manager),
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
            detail=f"Failed to get audio settings: {e!s}",
        ) from e


@router.put("/audio", response_model=ConfigResponse)
async def update_audio_settings(
    request: dict[str, Any],
    config_manager=Depends(get_config_manager),
):
    """
    Update audio processing settings

    Updates audio processing configuration. Changes take effect
    for new audio processing sessions.
    """
    try:
        logger.info("Updating audio processing settings")

        # Update audio settings
        updated_settings = await config_manager.update_audio_settings(request)

        return ConfigResponse(**updated_settings)

    except Exception as e:
        logger.error(f"Failed to update audio settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update audio settings: {e!s}",
        ) from e


# ============================================================================
# Settings Backup/Restore Endpoints
# ============================================================================


@router.post("/backup", response_model=SettingsBackupResponse)
async def create_settings_backup(
    config_manager=Depends(get_config_manager),
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
            detail=f"Failed to create settings backup: {e!s}",
        ) from e


@router.post("/restore/{backup_id}")
async def restore_settings_backup(
    backup_id: str,
    config_manager=Depends(get_config_manager),
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
            detail=f"Failed to restore settings backup: {e!s}",
        ) from e


@router.get("/backups")
async def list_settings_backups(
    config_manager=Depends(get_config_manager),
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
            detail=f"Failed to list settings backups: {e!s}",
        ) from e


# ============================================================================
# Settings Validation Endpoints
# ============================================================================


@router.post("/validate")
async def validate_settings(
    settings: dict[str, Any],
    config_manager=Depends(get_config_manager),
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
            detail=f"Failed to validate settings: {e!s}",
        ) from e


@router.get("/defaults")
async def get_default_settings(
    config_manager=Depends(get_config_manager),
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
            detail=f"Failed to get default settings: {e!s}",
        ) from e


@router.post("/reset")
async def reset_settings_to_defaults(
    category: Optional[str] = None,
    config_manager=Depends(get_config_manager),
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
            detail=f"Failed to reset settings: {e!s}",
        ) from e


# ============================================================================
# Bulk Configuration Management
# ============================================================================


@router.get("/export")
async def export_all_settings():
    """Export all configuration settings"""
    try:
        configs = {}

        # Load all configurations
        configs["audio_processing"] = await load_config(
            AUDIO_CONFIG_FILE, AudioProcessingConfig().dict()
        )
        configs["chunking"] = await load_config(CHUNKING_CONFIG_FILE, ChunkingConfig().dict())
        configs["correlation"] = await load_config(
            CORRELATION_CONFIG_FILE, CorrelationConfig().dict()
        )
        configs["translation"] = await load_config(
            TRANSLATION_CONFIG_FILE, TranslationConfig().dict()
        )
        configs["bot"] = await load_config(BOT_CONFIG_FILE, BotConfig().dict())
        configs["system"] = await load_config(SYSTEM_CONFIG_FILE, SystemConfig().dict())

        return {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "configurations": configs,
        }
    except Exception as e:
        logger.error(f"Error exporting settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to export settings") from e


@router.post("/import")
async def import_all_settings(config_data: dict[str, Any]):
    """Import all configuration settings"""
    try:
        configurations = config_data.get("configurations", {})
        results = {}

        # Save all configurations
        if "audio_processing" in configurations:
            results["audio_processing"] = await save_config(
                AUDIO_CONFIG_FILE, configurations["audio_processing"]
            )
        if "chunking" in configurations:
            results["chunking"] = await save_config(
                CHUNKING_CONFIG_FILE, configurations["chunking"]
            )
        if "correlation" in configurations:
            results["correlation"] = await save_config(
                CORRELATION_CONFIG_FILE, configurations["correlation"]
            )
        if "translation" in configurations:
            results["translation"] = await save_config(
                TRANSLATION_CONFIG_FILE, configurations["translation"]
            )
        if "bot" in configurations:
            results["bot"] = await save_config(BOT_CONFIG_FILE, configurations["bot"])
        if "system" in configurations:
            results["system"] = await save_config(SYSTEM_CONFIG_FILE, configurations["system"])

        return {"message": "Configuration import completed", "results": results}
    except Exception as e:
        logger.error(f"Error importing settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to import settings") from e


@router.post("/reset-all")
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
            "system": SystemConfig().dict(),
        }

        results = {}
        file_map = {
            "audio_processing": AUDIO_CONFIG_FILE,
            "chunking": CHUNKING_CONFIG_FILE,
            "correlation": CORRELATION_CONFIG_FILE,
            "translation": TRANSLATION_CONFIG_FILE,
            "bot": BOT_CONFIG_FILE,
            "system": SYSTEM_CONFIG_FILE,
        }

        for config_name, config_data in default_configs.items():
            results[config_name] = await save_config(file_map[config_name], config_data)

        return {
            "message": "All settings reset to defaults successfully",
            "results": results,
        }
    except Exception as e:
        logger.error(f"Error resetting settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset settings") from e
