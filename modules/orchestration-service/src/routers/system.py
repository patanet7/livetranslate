"""
System Management Router

FastAPI router for system-wide management endpoints including:
- System health monitoring
- Service status and coordination
- Performance metrics
- System configuration
"""

import re
import time
from datetime import UTC, datetime
from typing import Any

import psutil
from dependencies import (
    get_health_monitor,
    get_websocket_manager,
)
from fastapi import APIRouter, Depends, HTTPException, status
from livetranslate_common.logging import get_logger
from models.system import (
    ServiceHealth,
    SystemMetrics,
)
from pydantic import BaseModel, Field, field_validator
from routers.settings._shared import SYSTEM_CONFIG_FILE, load_config, save_config

router = APIRouter()
logger = get_logger()

# ============================================================================
# Request/Response Models
# ============================================================================


class SystemStatsResponse(BaseModel):
    """System statistics response"""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: dict[str, int]
    process_count: int
    uptime: float
    timestamp: datetime


class ServiceHealthSummary(BaseModel):
    """Service health summary"""

    total_services: int
    healthy_services: int
    unhealthy_services: int
    degraded_services: int
    services: list[list[ServiceHealth]]


class SystemMetricsResponse(BaseModel):
    """System metrics response"""

    performance: SystemMetrics
    services: ServiceHealthSummary
    websocket_stats: dict[str, Any]
    system_stats: SystemStatsResponse


class DomainItem(BaseModel):
    value: str = Field(..., max_length=64)
    label: str = Field(..., max_length=128)
    description: str = Field(default="", max_length=512)

    @field_validator("value")
    @classmethod
    def value_must_be_slug(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9_-]+$", v):
            raise ValueError("value must be a lowercase alphanumeric slug (a-z, 0-9, _, -)")
        return v


class SystemConfigUpdate(BaseModel):
    enabled_languages: list[str] | None = None
    custom_domains: list[DomainItem] | None = None
    disabled_domains: list[str] | None = None
    defaults: dict[str, Any] | None = None


# ============================================================================
# System Health Endpoints
# ============================================================================


@router.get("/health")
async def get_system_health(
    health_monitor=Depends(get_health_monitor),
    # Rate limiting will be handled by middleware
):
    """
    Get comprehensive system health status

    Returns overall system health including all service statuses,
    performance metrics, and system resources.
    """
    try:
        # Get system health from health monitor
        health_data = await health_monitor.get_system_health()

        # Ensure JSON serializable by converting datetime objects
        from datetime import datetime

        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.timestamp()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        safe_health_data = convert_datetime(health_data)
        return safe_health_data

    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {e!s}",
        ) from e


@router.get("/health/detailed")
async def get_detailed_health(
    health_monitor=Depends(get_health_monitor),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get detailed system health information

    Returns comprehensive health information including service details,
    error logs, and diagnostic information.
    """
    try:
        detailed_health = await health_monitor.get_detailed_health()

        return detailed_health

    except Exception as e:
        logger.error(f"Failed to get detailed health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get detailed health: {e!s}",
        ) from e


# ============================================================================
# Service Status Endpoints
# ============================================================================


@router.get("/services", response_model=ServiceHealthSummary)
async def get_services_status(
    health_monitor=Depends(get_health_monitor),
    # Rate limiting will be handled by middleware
):
    """
    Get status of all managed services

    Returns health status and basic information for all services
    in the LiveTranslate ecosystem.
    """
    try:
        services_status = await health_monitor.get_all_services_status()

        # Calculate summary statistics
        total = len(services_status)
        healthy = sum(1 for service in services_status if service.status == "healthy")
        unhealthy = sum(1 for service in services_status if service.status == "unhealthy")
        degraded = sum(1 for service in services_status if service.status == "degraded")

        return ServiceHealthSummary(
            total_services=total,
            healthy_services=healthy,
            unhealthy_services=unhealthy,
            degraded_services=degraded,
            services=services_status,
        )

    except Exception as e:
        logger.error(f"Failed to get services status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get services status: {e!s}",
        ) from e


@router.get("/services/{service_name}")
async def get_service_status(
    service_name: str,
    health_monitor=Depends(get_health_monitor),
    # Rate limiting will be handled by middleware
):
    """
    Get detailed status of a specific service

    Returns comprehensive information about a specific service including
    health metrics, performance data, and configuration.
    """
    try:
        service_status = await health_monitor.get_service_status(service_name)

        if not service_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service {service_name} not found",
            )

        return service_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service status: {e!s}",
        ) from e


@router.post("/services/{service_name}/restart")
async def restart_service(
    service_name: str,
    health_monitor=Depends(get_health_monitor),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Restart a specific service

    Attempts to restart a service by sending restart signals
    and monitoring recovery status.
    """
    try:
        logger.info(f"Restarting service: {service_name}")

        # Check if service exists
        service_status = await health_monitor.get_service_status(service_name)
        if not service_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service {service_name} not found",
            )

        # Attempt restart
        restart_result = await health_monitor.restart_service(service_name)

        return {
            "message": f"Service {service_name} restart initiated",
            "result": restart_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart service: {e!s}",
        ) from e


# ============================================================================
# Performance Metrics Endpoints
# ============================================================================


@router.get("/metrics")
async def get_system_metrics(
    health_monitor=Depends(get_health_monitor),
    websocket_manager=Depends(get_websocket_manager),
    # Rate limiting will be handled by middleware
):
    """
    Get comprehensive system metrics

    Returns performance metrics, service health, WebSocket statistics,
    and system resource usage.
    """
    try:
        # Get performance metrics
        performance = await health_monitor.get_performance_metrics()

        # Get service health summary
        services_status = await health_monitor.get_all_services_status()
        total = len(services_status)
        healthy = sum(1 for s in services_status if s.get("status") == "healthy")
        unhealthy = sum(1 for s in services_status if s.get("status") == "unhealthy")
        degraded = sum(1 for s in services_status if s.get("status") == "degraded")

        # Get WebSocket statistics
        websocket_stats = await websocket_manager.get_connection_stats()

        # Get system statistics
        system_stats = await _get_system_stats()

        metrics_data = {
            "performance": performance,
            "services": {
                "total_services": total,
                "healthy_services": healthy,
                "unhealthy_services": unhealthy,
                "degraded_services": degraded,
                "services": services_status,
            },
            "websocket_stats": websocket_stats,
            "system_stats": system_stats,
        }

        # Ensure JSON serializable by converting datetime objects
        from datetime import datetime

        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.timestamp()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        safe_metrics = convert_datetime(metrics_data)
        return safe_metrics

    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {e!s}",
        ) from e


@router.get("/metrics/performance")
async def get_performance_metrics(
    health_monitor=Depends(get_health_monitor),
    # Rate limiting will be handled by middleware
):
    """
    Get system performance metrics

    Returns detailed performance metrics including response times,
    throughput, and resource utilization.
    """
    try:
        metrics = await health_monitor.get_performance_metrics()

        return metrics

    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {e!s}",
        ) from e


# ============================================================================
# System Configuration Endpoints
# ============================================================================


@router.get("/ui-config")
async def get_ui_config():
    """
    Get UI configuration constants.

    Returns centralized configuration for all UI components including:
    - Supported languages (with codes, names, native names)
    - Glossary domains
    - Default settings
    - Prompt template variables

    This is the SINGLE SOURCE OF TRUTH for UI configuration.
    All dashboards and UIs should fetch from this endpoint.
    """
    from system_constants import (
        DEFAULT_CONFIG,
        GLOSSARY_DOMAINS,
        PROMPT_TEMPLATE_VARIABLES,
        SUPPORTED_LANGUAGES,
    )

    # Fetch translation models from translation service (if available)
    translation_models = []
    translation_service_available = False
    # TODO: wire up to new TranslationService — TranslationServiceClient removed.
    # Model listing not yet available on translation.service.TranslationService.
    try:
        from dependencies import get_translation_service_client
        svc = get_translation_service_client()
        translation_models = [{"name": svc.config.model, "description": "Configured Ollama model"}]
        translation_service_available = True
    except Exception as e:
        logger.warning(f"Translation service unavailable for model list: {e}")

    # Fetch prompts from translation service (if available)
    prompt_templates = []
    prompts_available = False
    try:
        import aiohttp
        from config import get_settings

        settings = get_settings()
        translation_url = getattr(settings, "translation_service_url", "http://localhost:5003")

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{translation_url}/prompts", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp,
        ):
            if resp.status == 200:
                prompts_data = await resp.json()
                prompt_templates = prompts_data.get("prompts", [])
                prompts_available = True
    except Exception as e:
        logger.debug(f"Could not fetch prompts: {e}")

    # Load user overrides
    overrides = await load_config(SYSTEM_CONFIG_FILE, {})

    # Filter languages if enabled_languages is set
    enabled_langs = overrides.get("enabled_languages")
    if enabled_langs:
        languages = [lang for lang in SUPPORTED_LANGUAGES if lang["code"] in enabled_langs]
    else:
        languages = SUPPORTED_LANGUAGES

    # Merge domains: start with built-in, remove disabled, add custom
    domains = list(GLOSSARY_DOMAINS)
    disabled_domains = overrides.get("disabled_domains", [])
    domains = [d for d in domains if d["value"] not in disabled_domains]
    custom_domains = overrides.get("custom_domains", [])
    domains.extend(custom_domains)

    # Merge defaults
    defaults = {**DEFAULT_CONFIG, **overrides.get("defaults", {})}

    return {
        # Core configuration (merged)
        "languages": languages,
        "language_codes": [lang["code"] for lang in languages],
        "domains": domains,
        "defaults": defaults,
        "prompt_variables": PROMPT_TEMPLATE_VARIABLES,
        # Dynamic configuration (from services)
        "translation_models": translation_models,
        "translation_service_available": translation_service_available,
        "prompt_templates": prompt_templates,
        "prompts_available": prompts_available,
        # Override metadata
        "has_overrides": bool(overrides),
        "enabled_language_count": len(languages),
        "total_language_count": len(SUPPORTED_LANGUAGES),
        # Metadata
        "config_version": "1.0",
        "source": "orchestration-service",
    }


@router.put("/ui-config")
async def update_ui_config(config: SystemConfigUpdate):
    """
    Update system configuration overrides.

    Saves user customizations to ./config/system.json.
    These overrides merge on top of system_constants.py defaults.
    """
    from system_constants import VALID_DOMAINS, VALID_LANGUAGE_CODES

    if config.enabled_languages is not None and len(config.enabled_languages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="enabled_languages must contain at least one language code, or be omitted to enable all",
        )

    # Validate language codes
    if config.enabled_languages is not None:
        invalid = [c for c in config.enabled_languages if c not in VALID_LANGUAGE_CODES]
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid language codes: {invalid}",
            )

    # Validate disabled_domains reference existing built-in domains
    if config.disabled_domains is not None:
        invalid = [d for d in config.disabled_domains if d not in VALID_DOMAINS]
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid domain values: {invalid}",
            )

    # Load existing overrides and merge
    existing = await load_config(SYSTEM_CONFIG_FILE, {})
    update_data = config.model_dump(exclude_none=True)

    # Deep-merge defaults so partial updates don't erase sibling keys
    merged = {**existing, **update_data}
    if "defaults" in update_data and "defaults" in existing:
        merged["defaults"] = {**existing["defaults"], **update_data["defaults"]}

    success = await save_config(SYSTEM_CONFIG_FILE, merged)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save configuration",
        )

    return {"status": "ok", "message": "Configuration saved"}


@router.post("/ui-config/reset")
async def reset_ui_config():
    """
    Reset system configuration to factory defaults.

    Deletes the override file, restoring system_constants.py values.
    """
    SYSTEM_CONFIG_FILE.unlink(missing_ok=True)
    logger.info("System configuration reset to factory defaults")
    return {"status": "ok", "message": "Reset to factory defaults"}


@router.get("/config", response_model=dict[str, Any])
async def get_system_config(
    health_monitor=Depends(get_health_monitor),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Get system configuration

    Returns current system configuration including service URLs,
    timeouts, and operational parameters.
    """
    try:
        config = await health_monitor.get_system_config()

        return dict[str, Any](**config)

    except Exception as e:
        logger.error(f"Failed to get system config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system config: {e!s}",
        ) from e


@router.post("/config")
async def update_system_config(
    config_update: dict[str, Any],
    health_monitor=Depends(get_health_monitor),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Update system configuration

    Updates system configuration. Some changes may require service
    restart to take effect.
    """
    try:
        logger.info("Updating system configuration")

        # Update configuration
        result = await health_monitor.update_system_config(config_update)

        return {
            "message": "System configuration updated",
            "updated_keys": result.get("updated_keys", []),
            "restart_required": result.get("restart_required", False),
        }

    except Exception as e:
        logger.error(f"Failed to update system config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update system config: {e!s}",
        ) from e


# ============================================================================
# System Maintenance Endpoints
# ============================================================================


@router.post("/maintenance/start")
async def start_maintenance_mode(
    health_monitor=Depends(get_health_monitor),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Start system maintenance mode

    Puts the system into maintenance mode, gracefully handling
    existing connections and preventing new ones.
    """
    try:
        logger.info("Starting maintenance mode")

        result = await health_monitor.start_maintenance_mode()

        return {"message": "Maintenance mode started", "details": result}

    except Exception as e:
        logger.error(f"Failed to start maintenance mode: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start maintenance mode: {e!s}",
        ) from e


@router.post("/maintenance/stop")
async def stop_maintenance_mode(
    health_monitor=Depends(get_health_monitor),
    # Authentication will be handled by middleware
    # Rate limiting will be handled by middleware
):
    """
    Stop system maintenance mode

    Exits maintenance mode and resumes normal operation.
    """
    try:
        logger.info("Stopping maintenance mode")

        result = await health_monitor.stop_maintenance_mode()

        return {"message": "Maintenance mode stopped", "details": result}

    except Exception as e:
        logger.error(f"Failed to stop maintenance mode: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop maintenance mode: {e!s}",
        ) from e


@router.get("/maintenance/status")
async def get_maintenance_status(
    health_monitor=Depends(get_health_monitor),
    # Rate limiting will be handled by middleware
):
    """
    Get maintenance mode status

    Returns current maintenance mode status and details.
    """
    try:
        status = await health_monitor.get_maintenance_status()

        return status

    except Exception as e:
        logger.error(f"Failed to get maintenance status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get maintenance status: {e!s}",
        ) from e


# ============================================================================
# System Utilities
# ============================================================================


async def _get_system_stats() -> dict:
    """Get system resource statistics"""
    try:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_usage = (disk.used / disk.total) * 100

        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        }

        # Process count
        process_count = len(psutil.pids())

        # Uptime (approximate)
        uptime = time.time() - psutil.boot_time()

        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "network_io": network_io,
            "process_count": process_count,
            "uptime": uptime,
            "timestamp": time.time(),  # Use timestamp instead of datetime
        }

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        # Return default values on error
        return {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_io": {},
            "process_count": 0,
            "uptime": 0.0,
            "timestamp": time.time(),
        }


@router.get("/status")
async def get_system_status(
    # Rate limiting will be handled by middleware
):
    """
    Get basic system status

    Returns a simple status check for load balancers and monitoring.
    """
    try:
        return {
            "status": "ok",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "2.0.0",
            "service": "orchestration-service",
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System status check failed",
        ) from e
