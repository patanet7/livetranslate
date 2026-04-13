"""
System Management Router

FastAPI router for system-wide management endpoints including:
- System health monitoring
- Service status and coordination
- Performance metrics
- System configuration
- Capability detection (ScreenCaptureKit availability)
"""

import os
import platform
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

import psutil
from dependencies import (
    get_database_manager,
    get_health_monitor,
    get_websocket_manager,
)
from fastapi import APIRouter, Depends, HTTPException, status
from livetranslate_common.logging import get_logger
from meeting.translation_recovery import get_translation_recovery_metrics
from models.system import (
    ServiceHealth,
    SystemMetrics,
)
from pydantic import BaseModel, Field, field_validator
from routers.settings._shared import SYSTEM_CONFIG_FILE, load_config, save_config
from services.meeting_store import MeetingStore

router = APIRouter()
logger = get_logger()
_meeting_metrics_store: MeetingStore | None = None


async def _get_meeting_metrics_store() -> MeetingStore | None:
    global _meeting_metrics_store
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return None
    if _meeting_metrics_store is None:
        _meeting_metrics_store = MeetingStore(db_url)
        await _meeting_metrics_store.initialize()
    return _meeting_metrics_store

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
        meeting_store = await _get_meeting_metrics_store()
        translation_backlog = (
            await meeting_store.list_translation_backlog(limit=25, offset=0, only_pending=True)
            if meeting_store is not None
            else {"meetings": [], "summary": {"pending_translation_count": 0}, "total": 0}
        )
        recovery_counters = await get_translation_recovery_metrics()

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
            "translation_backlog": {
                "summary": translation_backlog["summary"],
                "top_meetings": translation_backlog["meetings"],
                "total_meetings": translation_backlog["total"],
                "recovery_counters": recovery_counters,
            },
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
    # Model listing not yet available on translation.service.TranslationService.
    try:
        from dependencies import get_translation_service_client
        svc = get_translation_service_client()
        translation_models = [{"name": svc.config.model, "description": "Configured Ollama model"}]
        translation_service_available = True
    except Exception as e:
        logger.warning(f"Translation service unavailable for model list: {e}")

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
        "prompt_templates": [],
        "prompts_available": False,
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


# ============================================================================
# Capability Detection Endpoints
# ============================================================================


class ScreenCaptureAvailability(BaseModel):
    available: bool
    reason: str | None = None
    platform: str
    version: str | None = None


@router.get("/screencapture-available", response_model=ScreenCaptureAvailability)
async def check_screencapture_available() -> ScreenCaptureAvailability:
    """Check if ScreenCaptureKit audio capture is available on this host."""
    system = platform.system()

    if system != "Darwin":
        return ScreenCaptureAvailability(
            available=False,
            reason="ScreenCaptureKit requires macOS",
            platform=system,
        )

    # Check macOS version (need 13.0+)
    version = platform.mac_ver()[0]
    major = int(version.split(".")[0]) if version else 0

    if major < 13:
        return ScreenCaptureAvailability(
            available=False,
            reason=f"ScreenCaptureKit requires macOS 13+, found {version}",
            platform=system,
            version=version,
        )

    # Check if capture binary is installed
    try:
        from audio.screencapture_source import ScreenCaptureAudioSource

        if not ScreenCaptureAudioSource.is_available():
            return ScreenCaptureAvailability(
                available=False,
                reason="livetranslate-capture binary not installed",
                platform=system,
                version=version,
            )
    except ImportError:
        return ScreenCaptureAvailability(
            available=False,
            reason="ScreenCaptureKit audio source module not installed",
            platform=system,
            version=version,
        )

    return ScreenCaptureAvailability(
        available=True,
        platform=system,
        version=version,
    )


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


# =============================================================================
# Dashboard Statistics (real database metrics)
# =============================================================================


class MeetingsBySource(BaseModel):
    """Meeting counts broken down by source."""
    fireflies: int = 0
    loopback: int = 0
    gmeet: int = 0
    other: int = 0


class MeetingsByStatus(BaseModel):
    """Meeting counts broken down by status."""
    ephemeral: int = 0
    active: int = 0
    completed: int = 0
    interrupted: int = 0


class ActiveMeeting(BaseModel):
    """Info about an active/in-progress meeting."""
    id: str
    source: str
    status: str
    title: str | None = None
    started_at: str | None = None
    duration_seconds: int | None = None
    chunks_count: int = 0
    translations_count: int = 0


class DailyActivity(BaseModel):
    """Activity counts for a single day."""
    date: str
    meetings: int = 0
    chunks: int = 0
    translations: int = 0
    audio_minutes: float = 0.0


class ServiceStatus(BaseModel):
    """Status of a backend service."""
    name: str
    healthy: bool
    latency_ms: float | None = None
    last_check: str | None = None
    error: str | None = None


class DashboardStats(BaseModel):
    """Comprehensive dashboard statistics from real database."""
    # Summary counts
    total_meetings: int = 0
    active_meetings: int = 0
    total_chunks: int = 0
    total_translations: int = 0
    total_audio_minutes: float = 0.0

    # Breakdowns
    by_source: MeetingsBySource = Field(default_factory=MeetingsBySource)
    by_status: MeetingsByStatus = Field(default_factory=MeetingsByStatus)

    # Active/in-progress meetings
    active_meeting_list: list[ActiveMeeting] = Field(default_factory=list)

    # Time series for charts (last 7 days)
    daily_activity: list[DailyActivity] = Field(default_factory=list)

    # Service health
    services: list[ServiceStatus] = Field(default_factory=list)

    # Metadata
    generated_at: str = ""
    database_connected: bool = False


@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    health_monitor=Depends(get_health_monitor),
) -> DashboardStats:
    """
    Get real dashboard statistics from the database.

    Returns comprehensive metrics including:
    - Meeting counts by source and status
    - Total transcription chunks and translations
    - Active/in-progress meetings
    - 7-day activity chart data
    - Service health status
    """
    stats = DashboardStats(generated_at=datetime.now(UTC).isoformat())

    try:
        db_manager = get_database_manager()
        stats.database_connected = True
    except Exception as e:
        logger.warning(f"Database not available for dashboard stats: {e}")
        # Return partial stats with service health only
        stats.services = await _get_service_health(health_monitor)
        return stats

    try:
        async with db_manager.get_session() as session:
            from database.models import Meeting, MeetingChunk, MeetingTranslation

            # Total meetings count
            result = await session.execute(select(func.count(Meeting.id)))
            stats.total_meetings = result.scalar() or 0

            # Meetings by status
            status_result = await session.execute(
                select(Meeting.status, func.count(Meeting.id)).group_by(Meeting.status)
            )
            for status_name, count in status_result:
                if status_name == "ephemeral":
                    stats.by_status.ephemeral = count
                elif status_name == "active":
                    stats.by_status.active = count
                elif status_name == "completed":
                    stats.by_status.completed = count
                elif status_name == "interrupted":
                    stats.by_status.interrupted = count

            stats.active_meetings = stats.by_status.ephemeral + stats.by_status.active

            # Meetings by source
            source_result = await session.execute(
                select(Meeting.source, func.count(Meeting.id)).group_by(Meeting.source)
            )
            for source_name, count in source_result:
                if source_name == "fireflies":
                    stats.by_source.fireflies = count
                elif source_name == "loopback":
                    stats.by_source.loopback = count
                elif source_name in ("gmeet", "google_meet"):
                    stats.by_source.gmeet = count
                else:
                    stats.by_source.other += count

            # Total chunks
            chunk_result = await session.execute(select(func.count(MeetingChunk.id)))
            stats.total_chunks = chunk_result.scalar() or 0

            # Total translations
            translation_result = await session.execute(
                select(func.count(MeetingTranslation.id))
            )
            stats.total_translations = translation_result.scalar() or 0

            # Total audio minutes (from meeting durations)
            duration_result = await session.execute(
                select(func.sum(Meeting.duration)).where(Meeting.duration.isnot(None))
            )
            total_seconds = duration_result.scalar() or 0
            stats.total_audio_minutes = round(total_seconds / 60.0, 1)

            # Active meetings list (ephemeral + active status)
            active_result = await session.execute(
                select(Meeting)
                .where(Meeting.status.in_(["ephemeral", "active"]))
                .order_by(Meeting.started_at.desc())
                .limit(10)
            )
            active_meetings = active_result.scalars().all()

            for m in active_meetings:
                # Get chunk count for this meeting
                chunk_count_result = await session.execute(
                    select(func.count(MeetingChunk.id)).where(
                        MeetingChunk.meeting_id == m.id
                    )
                )
                chunk_count = chunk_count_result.scalar() or 0

                # Get translation count
                trans_count_result = await session.execute(
                    select(func.count(MeetingTranslation.id)).where(
                        MeetingTranslation.meeting_id == m.id
                    )
                )
                trans_count = trans_count_result.scalar() or 0

                # Calculate duration if started
                duration = None
                if m.started_at:
                    duration = int((datetime.now(UTC) - m.started_at).total_seconds())

                stats.active_meeting_list.append(
                    ActiveMeeting(
                        id=str(m.id),
                        source=m.source or "unknown",
                        status=m.status or "unknown",
                        title=m.title,
                        started_at=m.started_at.isoformat() if m.started_at else None,
                        duration_seconds=duration,
                        chunks_count=chunk_count,
                        translations_count=trans_count,
                    )
                )

            # Daily activity for last 7 days
            today = datetime.now(UTC).date()
            for i in range(7):
                day = today - timedelta(days=i)
                day_start = datetime.combine(day, datetime.min.time()).replace(tzinfo=UTC)
                day_end = day_start + timedelta(days=1)

                # Meetings created this day
                day_meetings = await session.execute(
                    select(func.count(Meeting.id)).where(
                        Meeting.created_at >= day_start,
                        Meeting.created_at < day_end,
                    )
                )
                meetings_count = day_meetings.scalar() or 0

                # Chunks created this day
                day_chunks = await session.execute(
                    select(func.count(MeetingChunk.id)).where(
                        MeetingChunk.created_at >= day_start,
                        MeetingChunk.created_at < day_end,
                    )
                )
                chunks_count = day_chunks.scalar() or 0

                # Translations created this day
                day_trans = await session.execute(
                    select(func.count(MeetingTranslation.id)).where(
                        MeetingTranslation.created_at >= day_start,
                        MeetingTranslation.created_at < day_end,
                    )
                )
                trans_count = day_trans.scalar() or 0

                # Audio minutes from meetings ending this day
                day_duration = await session.execute(
                    select(func.sum(Meeting.duration)).where(
                        Meeting.ended_at >= day_start,
                        Meeting.ended_at < day_end,
                        Meeting.duration.isnot(None),
                    )
                )
                day_seconds = day_duration.scalar() or 0

                stats.daily_activity.append(
                    DailyActivity(
                        date=day.isoformat(),
                        meetings=meetings_count,
                        chunks=chunks_count,
                        translations=trans_count,
                        audio_minutes=round(day_seconds / 60.0, 1),
                    )
                )

            # Reverse to show oldest first (for charts)
            stats.daily_activity.reverse()

    except Exception as e:
        logger.error(f"Failed to query dashboard stats: {e}")
        # Continue with partial data

    # Service health (always try to get this)
    stats.services = await _get_service_health(health_monitor)

    return stats


async def _get_service_health(health_monitor) -> list[ServiceStatus]:
    """Check health of all backend services."""
    services = []
    now = datetime.now(UTC).isoformat()

    # Orchestration (self)
    services.append(
        ServiceStatus(name="orchestration", healthy=True, last_check=now)
    )

    # Transcription service
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.monotonic()
            resp = await client.get("http://localhost:5001/health")
            latency = (time.monotonic() - start) * 1000
            services.append(
                ServiceStatus(
                    name="transcription",
                    healthy=resp.status_code == 200,
                    latency_ms=round(latency, 1),
                    last_check=now,
                )
            )
    except Exception as e:
        services.append(
            ServiceStatus(name="transcription", healthy=False, error=str(e), last_check=now)
        )

    # vLLM-MLX STT (Whisper)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.monotonic()
            resp = await client.get("http://localhost:8005/health")
            latency = (time.monotonic() - start) * 1000
            services.append(
                ServiceStatus(
                    name="vllm-stt",
                    healthy=resp.status_code == 200,
                    latency_ms=round(latency, 1),
                    last_check=now,
                )
            )
    except Exception as e:
        services.append(
            ServiceStatus(name="vllm-stt", healthy=False, error=str(e), last_check=now)
        )

    # vLLM-MLX LLM (Translation)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.monotonic()
            resp = await client.get("http://localhost:8006/health")
            latency = (time.monotonic() - start) * 1000
            services.append(
                ServiceStatus(
                    name="vllm-llm",
                    healthy=resp.status_code == 200,
                    latency_ms=round(latency, 1),
                    last_check=now,
                )
            )
    except Exception as e:
        services.append(
            ServiceStatus(name="vllm-llm", healthy=False, error=str(e), last_check=now)
        )

    # Database
    try:
        db_manager = get_database_manager()
        async with db_manager.get_session() as session:
            start = time.monotonic()
            await session.execute(select(1))
            latency = (time.monotonic() - start) * 1000
            services.append(
                ServiceStatus(
                    name="database",
                    healthy=True,
                    latency_ms=round(latency, 1),
                    last_check=now,
                )
            )
    except Exception as e:
        services.append(
            ServiceStatus(name="database", healthy=False, error=str(e), last_check=now)
        )

    return services
