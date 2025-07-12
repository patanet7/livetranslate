"""
System Management Router

FastAPI router for system-wide management endpoints including:
- System health monitoring
- Service status and coordination
- Performance metrics
- System configuration
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import psutil
import time
from datetime import datetime

from dependencies import (
    get_health_monitor,
    get_websocket_manager,
    get_audio_service_client,
    get_translation_service_client,
)
from models.system import (
    SystemStatus,
    ServiceHealth,
    SystemResources,
    SystemMetrics,
    ErrorResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================


class SystemStatsResponse(BaseModel):
    """System statistics response"""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    uptime: float
    timestamp: datetime


class ServiceHealthSummary(BaseModel):
    """Service health summary"""

    total_services: int
    healthy_services: int
    unhealthy_services: int
    degraded_services: int
    services: List[List[ServiceHealth]]


class SystemMetricsResponse(BaseModel):
    """System metrics response"""

    performance: SystemMetrics
    services: ServiceHealthSummary
    websocket_stats: Dict[str, Any]
    system_stats: SystemStatsResponse


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
            detail=f"Failed to get system health: {str(e)}",
        )


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
            detail=f"Failed to get detailed health: {str(e)}",
        )


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
        unhealthy = sum(
            1 for service in services_status if service.status == "unhealthy"
        )
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
            detail=f"Failed to get services status: {str(e)}",
        )


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
            detail=f"Failed to get service status: {str(e)}",
        )


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
            detail=f"Failed to restart service: {str(e)}",
        )


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
            detail=f"Failed to get system metrics: {str(e)}",
        )


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
            detail=f"Failed to get performance metrics: {str(e)}",
        )


# ============================================================================
# System Configuration Endpoints
# ============================================================================


@router.get("/config", response_model=Dict[str, Any])
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

        return Dict[str, Any](**config)

    except Exception as e:
        logger.error(f"Failed to get system config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system config: {str(e)}",
        )


@router.post("/config")
async def update_system_config(
    config_update: Dict[str, Any],
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
            detail=f"Failed to update system config: {str(e)}",
        )


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
            detail=f"Failed to start maintenance mode: {str(e)}",
        )


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
            detail=f"Failed to stop maintenance mode: {str(e)}",
        )


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
            detail=f"Failed to get maintenance status: {str(e)}",
        )


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
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "service": "orchestration-service",
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System status check failed",
        )
