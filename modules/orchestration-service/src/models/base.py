"""
Base Pydantic models with common functionality
"""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, Field, field_validator


class BaseModel(PydanticBaseModel):
    """Enhanced base model with common configuration"""

    class Config:
        # Use enum values in serialization
        use_enum_values = True

        # Validate assignment
        validate_assignment = True

        # Allow population by field name
        populate_by_name = True

        # JSON schema extra
        json_schema_extra = {"example": {}}

        # Custom JSON encoders
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp fields"""

    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Last update timestamp"
    )

    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.utcnow()


class IDMixin(BaseModel):
    """Mixin for models that need ID fields"""

    id: str = Field(description="Unique identifier", example="uuid-1234-5678-9abc-def")


class ResponseMixin(BaseModel):
    """Mixin for API response models"""

    success: bool = Field(
        default=True, description="Indicates if the operation was successful"
    )
    message: Optional[str] = Field(default=None, description="Human-readable message")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )


class PaginationMixin(BaseModel):
    """Mixin for paginated responses"""

    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(
        default=20, ge=1, le=100, description="Number of items per page"
    )
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

    code: str = Field(description="Error code", example="VALIDATION_ERROR")
    message: str = Field(description="Error message", example="Invalid input data")
    field: Optional[str] = Field(
        default=None, description="Field that caused the error", example="email"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )


class ValidationErrorDetail(BaseModel):
    """Pydantic validation error details"""

    loc: list = Field(description="Error location", example=["body", "email"])
    msg: str = Field(description="Error message", example="field required")
    type: str = Field(description="Error type", example="value_error.missing")
    ctx: Optional[Dict[str, Any]] = Field(default=None, description="Error context")


class HealthStatus(BaseModel):
    """Health status information"""

    status: str = Field(description="Health status", example="healthy")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )
    uptime_seconds: float = Field(
        description="Service uptime in seconds", example=3600.5
    )
    version: str = Field(description="Service version", example="2.0.0")

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

    name: str = Field(description="Metric name", example="cpu_usage_percent")
    value: float = Field(description="Metric value", example=45.2)
    unit: Optional[str] = Field(
        default=None, description="Metric unit", example="percent"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Metric timestamp"
    )
    labels: Optional[Dict[str, str]] = Field(
        default=None, description="Metric labels", example={"instance": "worker-1"}
    )


class LogEntry(BaseModel):
    """Log entry structure"""

    level: str = Field(description="Log level", example="INFO")
    message: str = Field(
        description="Log message", example="Service started successfully"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Log timestamp"
    )
    logger: Optional[str] = Field(
        default=None, description="Logger name", example="orchestration.service"
    )
    module: Optional[str] = Field(
        default=None, description="Source module", example="main.py"
    )
    function: Optional[str] = Field(
        default=None, description="Source function", example="startup"
    )
    line_number: Optional[int] = Field(
        default=None, description="Source line number", example=42
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional log data"
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v, info=None):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
