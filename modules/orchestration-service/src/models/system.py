"""
System and health-related Pydantic models
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from .base import BaseModel, ResponseMixin, TimestampMixin


class ServiceStatus(str, Enum):
    """Service status enumeration"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


class ServiceHealth(BaseModel):
    """Individual service health information"""

    service_name: str = Field(
        description="Name of the service", json_schema_extra={"example": "audio-service"}
    )
    status: ServiceStatus = Field(description="Current service status")
    url: str = Field(
        description="Service URL", json_schema_extra={"example": "http://localhost:5001"}
    )
    last_check: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last health check timestamp"
    )
    response_time_ms: float | None = Field(
        default=None,
        description="Response time in milliseconds",
        json_schema_extra={"example": 45.2},
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if unhealthy",
        json_schema_extra={"example": "Connection timeout"},
    )
    version: str | None = Field(
        default=None, description="Service version", json_schema_extra={"example": "1.2.3"}
    )
    uptime_seconds: float | None = Field(
        default=None, description="Service uptime in seconds", json_schema_extra={"example": 3600.5}
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service_name": "audio-service",
                "status": "healthy",
                "url": "http://localhost:5001",
                "last_check": "2024-01-15T10:30:00Z",
                "response_time_ms": 45.2,
                "error_message": None,
                "version": "1.2.3",
                "uptime_seconds": 3600.5,
            }
        }
    )


class SystemResources(BaseModel):
    """System resource usage information"""

    cpu_percent: float = Field(
        description="CPU usage percentage", ge=0, le=100, json_schema_extra={"example": 25.5}
    )
    memory_percent: float = Field(
        description="Memory usage percentage", ge=0, le=100, json_schema_extra={"example": 65.2}
    )
    memory_used_mb: float = Field(
        description="Memory used in MB", json_schema_extra={"example": 1024.5}
    )
    memory_total_mb: float = Field(
        description="Total memory in MB", json_schema_extra={"example": 8192.0}
    )
    disk_percent: float = Field(
        description="Disk usage percentage", ge=0, le=100, json_schema_extra={"example": 45.8}
    )
    disk_used_gb: float = Field(description="Disk used in GB", json_schema_extra={"example": 125.6})
    disk_total_gb: float = Field(
        description="Total disk in GB", json_schema_extra={"example": 256.0}
    )
    load_average: list[float] | None = Field(
        default=None,
        description="System load average (1m, 5m, 15m)",
        json_schema_extra={"example": [0.5, 0.7, 0.9]},
    )


class NetworkStats(BaseModel):
    """Network statistics"""

    bytes_sent: int = Field(description="Total bytes sent", json_schema_extra={"example": 1024000})
    bytes_received: int = Field(
        description="Total bytes received", json_schema_extra={"example": 2048000}
    )
    packets_sent: int = Field(description="Total packets sent", json_schema_extra={"example": 1500})
    packets_received: int = Field(
        description="Total packets received", json_schema_extra={"example": 3000}
    )
    connections_active: int = Field(
        description="Active network connections", json_schema_extra={"example": 25}
    )


class SystemStatus(ResponseMixin, TimestampMixin):
    """Overall system status"""

    status: ServiceStatus = Field(description="Overall system status")
    version: str = Field(description="Application version", json_schema_extra={"example": "2.0.0"})
    uptime_seconds: float = Field(
        description="System uptime in seconds", json_schema_extra={"example": 86400.5}
    )
    services: dict[str, ServiceHealth] = Field(description="Individual service health statuses")
    resources: SystemResources = Field(description="System resource usage")
    network: NetworkStats = Field(description="Network statistics")
    websocket_connections: int = Field(
        description="Active WebSocket connections", json_schema_extra={"example": 150}
    )
    active_sessions: int = Field(
        description="Active user sessions", json_schema_extra={"example": 75}
    )

    @field_validator("status")
    @classmethod
    def determine_overall_status(cls, v: ServiceStatus, info: ValidationInfo) -> ServiceStatus:
        """Determine overall status based on service statuses"""
        services = (info.data if info else {}).get("services", {})

        if not services:
            return ServiceStatus.UNKNOWN

        service_statuses = [service.status for service in services.values()]

        # If any service is unhealthy, system is unhealthy
        if ServiceStatus.UNHEALTHY in service_statuses:
            return ServiceStatus.UNHEALTHY

        # If any service is degraded, system is degraded
        if ServiceStatus.DEGRADED in service_statuses:
            return ServiceStatus.DEGRADED

        # If all services are healthy, system is healthy
        if all(status == ServiceStatus.HEALTHY for status in service_statuses):
            return ServiceStatus.HEALTHY

        # Otherwise, status is unknown
        return ServiceStatus.UNKNOWN

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "System status retrieved successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "status": "healthy",
                "version": "2.0.0",
                "uptime_seconds": 86400.5,
                "services": {
                    "audio-service": {
                        "service_name": "audio-service",
                        "status": "healthy",
                        "url": "http://localhost:5001",
                        "last_check": "2024-01-15T10:30:00Z",
                        "response_time_ms": 45.2,
                        "version": "1.2.3",
                        "uptime_seconds": 3600.5,
                    }
                },
                "resources": {
                    "cpu_percent": 25.5,
                    "memory_percent": 65.2,
                    "memory_used_mb": 1024.5,
                    "memory_total_mb": 8192.0,
                    "disk_percent": 45.8,
                    "disk_used_gb": 125.6,
                    "disk_total_gb": 256.0,
                    "load_average": [0.5, 0.7, 0.9],
                },
                "network": {
                    "bytes_sent": 1024000,
                    "bytes_received": 2048000,
                    "packets_sent": 1500,
                    "packets_received": 3000,
                    "connections_active": 25,
                },
                "websocket_connections": 150,
                "active_sessions": 75,
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response format"""

    error: str = Field(
        description="Error message", json_schema_extra={"example": "Service unavailable"}
    )
    error_code: str | None = Field(
        default=None,
        description="Specific error code",
        json_schema_extra={"example": "SERVICE_UNAVAILABLE"},
    )
    status_code: int = Field(description="HTTP status code", json_schema_extra={"example": 503})
    path: str = Field(
        description="Request path that caused the error",
        json_schema_extra={"example": "/api/health"},
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Error timestamp"
    )
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")
    request_id: str | None = Field(
        default=None,
        description="Request ID for tracking",
        json_schema_extra={"example": "req_abc123def456"},
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Service unavailable",
                "error_code": "SERVICE_UNAVAILABLE",
                "status_code": 503,
                "path": "/api/health",
                "timestamp": "2024-01-15T10:30:00Z",
                "details": {"service": "audio-service", "reason": "Connection timeout"},
                "request_id": "req_abc123def456",
            }
        }
    )


class ValidationErrorResponse(BaseModel):
    """Validation error response format"""

    error: str = Field(default="Validation error", description="Error message")
    status_code: int = Field(default=422, description="HTTP status code")
    path: str = Field(
        description="Request path", json_schema_extra={"example": "/api/config/update"}
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Error timestamp"
    )
    validation_errors: list[dict[str, Any]] = Field(description="Detailed validation errors")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Validation error",
                "status_code": 422,
                "path": "/api/config/update",
                "timestamp": "2024-01-15T10:30:00Z",
                "validation_errors": [
                    {
                        "loc": ["body", "port"],
                        "msg": "ensure this value is greater than 0",
                        "type": "value_error.number.not_gt",
                        "ctx": {"limit_value": 0},
                    }
                ],
            }
        }
    )


class ServiceMetrics(BaseModel):
    """Service-specific metrics"""

    service_name: str = Field(
        description="Service name", json_schema_extra={"example": "audio-service"}
    )
    requests_total: int = Field(
        description="Total requests processed", json_schema_extra={"example": 12500}
    )
    requests_per_second: float = Field(
        description="Current requests per second", json_schema_extra={"example": 15.5}
    )
    average_response_time_ms: float = Field(
        description="Average response time in milliseconds", json_schema_extra={"example": 45.2}
    )
    error_rate_percent: float = Field(
        description="Error rate percentage", ge=0, le=100, json_schema_extra={"example": 0.5}
    )
    uptime_percent: float = Field(
        description="Uptime percentage (last 24h)",
        ge=0,
        le=100,
        json_schema_extra={"example": 99.95},
    )
    last_error: str | None = Field(
        default=None,
        description="Last error message",
        json_schema_extra={"example": "Connection timeout after 30s"},
    )
    last_error_timestamp: datetime | None = Field(default=None, description="Last error timestamp")


class SystemMetrics(BaseModel):
    """System-wide metrics"""

    overall_status: ServiceStatus = Field(description="Overall system status")
    services: list[ServiceMetrics] = Field(description="Per-service metrics")
    total_requests: int = Field(
        description="Total system requests", json_schema_extra={"example": 50000}
    )
    total_errors: int = Field(description="Total system errors", json_schema_extra={"example": 25})
    average_response_time_ms: float = Field(
        description="System average response time", json_schema_extra={"example": 52.3}
    )
    system_uptime_percent: float = Field(
        description="System uptime percentage", ge=0, le=100, json_schema_extra={"example": 99.9}
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Metrics timestamp"
    )
