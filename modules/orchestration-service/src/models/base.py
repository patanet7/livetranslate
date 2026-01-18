"""
Base Pydantic models with common functionality
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, field_validator


class BaseModel(PydanticBaseModel):
    """Enhanced base model with common configuration"""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        populate_by_name=True,
        json_schema_extra={"example": {}},
        ser_json_timedelta="iso8601",
    )


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp fields"""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")

    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now(UTC)


class IDMixin(BaseModel):
    """Mixin for models that need ID fields"""

    id: str = Field(
        description="Unique identifier", json_schema_extra={"example": "uuid-1234-5678-9abc-def"}
    )


class ResponseMixin(BaseModel):
    """Mixin for API response models"""

    success: bool = Field(default=True, description="Indicates if the operation was successful")
    message: str | None = Field(default=None, description="Human-readable message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Response timestamp"
    )


class PaginationMixin(BaseModel):
    """Mixin for paginated responses"""

    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Number of items per page")
    total_items: int = Field(default=0, ge=0, description="Total number of items")
    total_pages: int = Field(default=0, ge=0, description="Total number of pages")

    @field_validator("total_pages")
    @classmethod
    def calculate_total_pages(cls, v, info):
        """Calculate total pages based on total items and page size"""
        data = info.data if info else {}
        total_items = data.get("total_items", 0)
        page_size = data.get("page_size", 20)
        if page_size > 0:
            return (total_items + page_size - 1) // page_size
        return 0


class ErrorDetails(BaseModel):
    """Detailed error information"""

    code: str = Field(description="Error code", json_schema_extra={"example": "VALIDATION_ERROR"})
    message: str = Field(
        description="Error message", json_schema_extra={"example": "Invalid input data"}
    )
    field: str | None = Field(
        default=None,
        description="Field that caused the error",
        json_schema_extra={"example": "email"},
    )
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")


class ValidationErrorDetail(BaseModel):
    """Pydantic validation error details"""

    loc: list = Field(
        description="Error location", json_schema_extra={"example": ["body", "email"]}
    )
    msg: str = Field(description="Error message", json_schema_extra={"example": "field required"})
    type: str = Field(
        description="Error type", json_schema_extra={"example": "value_error.missing"}
    )
    ctx: dict[str, Any] | None = Field(default=None, description="Error context")


class HealthStatus(BaseModel):
    """Health status information"""

    status: str = Field(description="Health status", json_schema_extra={"example": "healthy"})
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Health check timestamp"
    )
    uptime_seconds: float = Field(
        description="Service uptime in seconds", json_schema_extra={"example": 3600.5}
    )
    version: str = Field(description="Service version", json_schema_extra={"example": "2.0.0"})

    @field_validator("status")
    @classmethod
    def validate_status(cls, v, info=None):
        """Validate status values"""
        valid_statuses = ["healthy", "degraded", "unhealthy", "unknown"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v


class MetricsData(BaseModel):
    """Metrics data structure"""

    name: str = Field(description="Metric name", json_schema_extra={"example": "cpu_usage_percent"})
    value: float = Field(description="Metric value", json_schema_extra={"example": 45.2})
    unit: str | None = Field(
        default=None, description="Metric unit", json_schema_extra={"example": "percent"}
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Metric timestamp"
    )
    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels",
        json_schema_extra={"example": {"instance": "worker-1"}},
    )


class LogEntry(BaseModel):
    """Log entry structure"""

    level: str = Field(description="Log level", json_schema_extra={"example": "INFO"})
    message: str = Field(
        description="Log message", json_schema_extra={"example": "Service started successfully"}
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Log timestamp"
    )
    logger: str | None = Field(
        default=None,
        description="Logger name",
        json_schema_extra={"example": "orchestration.service"},
    )
    module: str | None = Field(
        default=None, description="Source module", json_schema_extra={"example": "main.py"}
    )
    function: str | None = Field(
        default=None, description="Source function", json_schema_extra={"example": "startup"}
    )
    line_number: int | None = Field(
        default=None, description="Source line number", json_schema_extra={"example": 42}
    )
    extra: dict[str, Any] | None = Field(default=None, description="Additional log data")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v, info=None):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
