"""
Configuration Sync Router

Handles all sync/* endpoints including presets, whisper-status, compatibility,
translation service configuration sync, and runtime model management.
"""

from ._shared import (
    CONFIG_SYNC_AVAILABLE,
    TRANSLATION_SERVICE_URL,
    Any,
    APIRouter,
    HTTPException,
    ModelSwitchRequest,
    aiohttp,
    apply_configuration_preset,
    get_config_sync_manager,
    get_translation_service_client,
    get_unified_configuration,
    logger,
    sync_all_configurations,
    update_configuration,
)

router = APIRouter(prefix="/sync", tags=["settings-sync"])


# ============================================================================
# Configuration Synchronization with Whisper Service
# ============================================================================


@router.get("/status")
async def get_configuration_sync_status():
    """Get current configuration synchronization status"""
    if not CONFIG_SYNC_AVAILABLE:
        return {
            "sync_available": False,
            "message": "Configuration sync manager not available",
            "fallback_mode": True,
        }

    try:
        config_manager = await get_config_sync_manager()
        unified_config = await config_manager.get_unified_configuration()

        translation_config = unified_config.get("translation_service", {})
        return {
            "sync_available": True,
            "last_sync": unified_config.get("sync_info", {}).get("last_sync"),
            "services_synced": unified_config.get("sync_info", {}).get(
                "services_synced", ["whisper", "orchestration"]
            ),
            "whisper_service_mode": unified_config.get("whisper_service", {}).get(
                "service_mode", "unknown"
            ),
            "orchestration_mode": unified_config.get("orchestration_service", {}).get(
                "service_mode", "unknown"
            ),
            "translation_service_status": translation_config.get("service_status", "unknown"),
            "translation_backend": translation_config.get("backend", "unknown"),
            "translation_model": translation_config.get("model_name", "unknown"),
            "compatibility_status": "synchronized",
            "configuration_version": unified_config.get("sync_info", {}).get(
                "configuration_version", "1.1"
            ),
        }
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sync status") from e


@router.get("/unified")
async def get_unified_configuration_endpoint():
    """Get unified configuration from all services (whisper + orchestration + frontend compatible)"""
    if not CONFIG_SYNC_AVAILABLE:
        return {
            "error": "Configuration sync not available",
            "fallback": True,
            "basic_config": {
                "whisper_service": {"status": "unknown"},
                "orchestration_service": {"status": "unknown"},
            },
        }

    try:
        unified_config = await get_unified_configuration()
        return unified_config
    except Exception as e:
        logger.error(f"Error getting unified configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get unified configuration") from e


@router.post("/update/{component}")
async def update_component_configuration(component: str, config_updates: dict[str, Any]):
    """
    Update configuration for a specific component (whisper, orchestration, or unified)
    Changes will be propagated to other components automatically
    """
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Configuration sync not available - cannot update component configuration",
        )

    valid_components = ["whisper", "orchestration", "translation", "unified"]
    if component not in valid_components:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid component. Must be one of: {valid_components}",
        )

    try:
        update_result = await update_configuration(component, config_updates, propagate=True)

        if not update_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration update failed: {update_result['errors']}",
            )

        return {
            "success": True,
            "component": component,
            "changes_applied": update_result["changes_applied"],
            "propagation_results": update_result["propagation_results"],
            "message": f"Configuration updated successfully for {component}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating {component} configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update {component} configuration") from e


@router.post("/preset/{preset_name}")
async def apply_configuration_preset_endpoint(preset_name: str):
    """Apply a configuration preset to all components"""
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Configuration sync not available - cannot apply presets",
        )

    try:
        result = await apply_configuration_preset(preset_name)

        if not result["success"]:
            raise HTTPException(
                status_code=400, detail=result.get("error", "Failed to apply preset")
            )

        return {
            "success": True,
            "preset_applied": preset_name,
            "preset_description": result.get("preset_description", ""),
            "changes_applied": result["changes_applied"],
            "propagation_results": result["propagation_results"],
            "message": f"Configuration preset '{preset_name}' applied successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying preset {preset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply preset {preset_name}") from e


@router.post("/force")
async def force_configuration_sync():
    """Force synchronization of all service configurations"""
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration sync not available")

    try:
        sync_result = await sync_all_configurations()

        return {
            "success": sync_result["success"],
            "sync_time": sync_result["sync_time"],
            "services_synced": sync_result["services_synced"],
            "compatibility_status": sync_result.get("compatibility_status", {}),
            "errors": sync_result.get("errors", []),
            "message": "Configuration synchronization completed",
        }
    except Exception as e:
        logger.error(f"Error forcing configuration sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to force configuration sync") from e


@router.get("/presets")
async def get_available_configuration_presets():
    """Get available configuration presets"""
    try:
        # Import presets from the compatibility layer
        from audio.whisper_compatibility import CONFIGURATION_PRESETS

        return {
            "available_presets": list(CONFIGURATION_PRESETS.keys()),
            "presets": CONFIGURATION_PRESETS,
            "message": "Configuration presets retrieved successfully",
        }
    except Exception as e:
        logger.error(f"Error getting configuration presets: {e}")
        return {
            "available_presets": [
                "exact_whisper_match",
                "optimized_performance",
                "high_accuracy",
                "real_time_optimized",
            ],
            "presets": {},
            "message": "Using fallback preset list",
        }


@router.get("/whisper-status")
async def get_whisper_service_sync_status():
    """Get detailed status of whisper service configuration sync"""
    if not CONFIG_SYNC_AVAILABLE:
        return {"sync_available": False, "message": "Configuration sync not available"}

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
                "statistics": whisper_config.get("statistics", {}),
            },
            "sync_info": unified_config.get("sync_info", {}),
            "compatibility": {
                "chunking_compatible": True,
                "metadata_support": True,
                "api_version_compatible": True,
            },
        }
    except Exception as e:
        logger.error(f"Error getting whisper sync status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get whisper service sync status") from e


@router.get("/compatibility")
async def get_configuration_compatibility():
    """Get configuration compatibility status between services"""
    if not CONFIG_SYNC_AVAILABLE:
        return {
            "compatible": False,
            "issues": ["Configuration sync manager not available"],
            "warnings": [],
            "sync_required": False,
            "message": "Configuration sync not available",
        }

    try:
        config_manager = await get_config_sync_manager()
        compatibility_status = await config_manager._validate_configuration_compatibility()

        return compatibility_status
    except Exception as e:
        logger.error(f"Error checking configuration compatibility: {e}")
        return {
            "compatible": False,
            "issues": [f"Failed to check compatibility: {e!s}"],
            "warnings": [],
            "sync_required": True,
            "message": "Error checking compatibility",
        }


@router.post("/preset")
async def apply_configuration_preset_by_name(preset_data: dict[str, Any]):
    """Apply a configuration preset by name"""
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Configuration sync not available - cannot apply presets",
        )

    preset_name = preset_data.get("preset_name")
    if not preset_name:
        raise HTTPException(status_code=400, detail="preset_name is required")

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
            "message": f"Applied preset: {preset_name}"
            if result.get("success")
            else f"Failed to apply preset: {preset_name}",
        }
    except Exception as e:
        logger.error(f"Error applying configuration preset {preset_name}: {e}")
        return {
            "success": False,
            "preset_applied": preset_name,
            "errors": [str(e)],
            "message": f"Failed to apply preset: {preset_name}",
        }


@router.get("/translation")
async def get_translation_service_configuration():
    """Get current translation service configuration with sync status"""
    if not CONFIG_SYNC_AVAILABLE:
        return {
            "error": "Configuration sync not available",
            "fallback": True,
            "basic_config": {"status": "unknown"},
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
                "configuration_version": unified_config.get("sync_info", {}).get(
                    "configuration_version", "1.1"
                ),
            },
        }
    except Exception as e:
        logger.error(f"Error getting translation service configuration: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get translation service configuration"
        ) from e


@router.post("/translation")
async def update_translation_service_configuration(config_updates: dict[str, Any]):
    """Update translation service configuration with automatic synchronization"""
    if not CONFIG_SYNC_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Configuration sync not available - cannot update translation configuration",
        )

    try:
        update_result = await update_configuration("translation", config_updates, propagate=True)

        if not update_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Translation configuration update failed: {update_result['errors']}",
            )

        return {
            "success": True,
            "component": "translation",
            "changes_applied": update_result["changes_applied"],
            "propagation_results": update_result.get("propagation_results", {}),
            "message": "Translation service configuration updated successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating translation configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update translation configuration: {e!s}",
        ) from e


# ============================================================================
# Translation Model Management - Dynamic Model Switching
# ============================================================================


@router.post("/translation/switch-model")
async def switch_translation_model(request: ModelSwitchRequest):
    """
    Switch translation model at runtime via orchestration service.

    This endpoint proxies the model switch request to the translation service's
    RuntimeModelManager, allowing the orchestration service to control which
    model the translation service uses without restarting either service.

    **Usage:**
    ```json
    {
      "model": "llama2:7b",
      "backend": "ollama"
    }
    ```

    **Supported Backends:**
    - `ollama` - Local Ollama instance
    - `groq` - Groq cloud API
    - `vllm` - vLLM server
    - `openai` - OpenAI API
    """
    try:
        async with await get_translation_service_client() as client:
            payload = {
                "model": request.model,
                "backend": request.backend,
            }
            async with client.post(
                f"{TRANSLATION_SERVICE_URL}/api/models/switch",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": result.get("success", False),
                        "model": result.get("model"),
                        "backend": result.get("backend"),
                        "message": result.get("message"),
                        "cached_models": result.get("cached_models", 0),
                        "source": "orchestration_proxy",
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}",
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service for model switch: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable - cannot switch model",
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching translation model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch translation model: {e!s}") from e


@router.post("/translation/preload-model")
async def preload_translation_model(request: ModelSwitchRequest):
    """
    Preload a translation model for faster switching later.

    Preloads the model in the translation service's cache without switching to it.
    """
    try:
        async with await get_translation_service_client() as client:
            payload = {
                "model": request.model,
                "backend": request.backend,
            }
            async with client.post(
                f"{TRANSLATION_SERVICE_URL}/api/models/preload",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": result.get("success", False),
                        "model": result.get("model"),
                        "backend": result.get("backend"),
                        "message": result.get("message"),
                        "cached_models": result.get("cached_models", 0),
                        "source": "orchestration_proxy",
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}",
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service for model preload: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable - cannot preload model",
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error preloading translation model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preload translation model: {e!s}") from e


@router.get("/translation/model-status")
async def get_translation_model_status():
    """
    Get current translation model manager status.

    Returns information about the currently active model, cached models,
    and supported backends.
    """
    try:
        async with (
            await get_translation_service_client() as client,
            client.get(f"{TRANSLATION_SERVICE_URL}/api/models/status") as response,
        ):
            if response.status == 200:
                result = await response.json()
                return {
                    "success": True,
                    "current_model": result.get("current_model"),
                    "current_backend": result.get("current_backend"),
                    "is_ready": result.get("is_ready", False),
                    "cached_models": result.get("cached_models", []),
                    "cache_size": result.get("cache_size", 0),
                    "supported_backends": result.get("supported_backends", []),
                    "source": "orchestration_proxy",
                }
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service for model status: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable - cannot get model status",
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting translation model status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get translation model status: {e!s}"
        ) from e


@router.get("/translation/available-models/{backend}")
async def get_available_translation_models(backend: str = "ollama"):
    """
    Get list of available models from a specific backend.

    **Example:**
    ```
    GET /api/settings/sync/translation/available-models/ollama
    ```
    """
    try:
        async with (
            await get_translation_service_client() as client,
            client.get(f"{TRANSLATION_SERVICE_URL}/api/models/list/{backend}") as response,
        ):
            if response.status == 200:
                result = await response.json()
                return {
                    "success": True,
                    "backend": result.get("backend"),
                    "models": result.get("models", []),
                    "count": result.get("count", 0),
                    "timestamp": result.get("timestamp"),
                    "source": "orchestration_proxy",
                }
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service for available models: {e}")
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable - cannot list models",
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting available translation models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {e!s}") from e
