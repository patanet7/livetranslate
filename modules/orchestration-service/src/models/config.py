"""
Configuration-related Pydantic models
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from .base import BaseModel, ResponseMixin, TimestampMixin


class ConfigUpdate(BaseModel):
    """Configuration update request"""

    updates: dict[str, Any] = Field(
        description="Configuration updates as key-value pairs",
        json_schema_extra={
            "example": {
                "websocket.max_connections": 2000,
                "services.audio_service_timeout": 45,
                "logging.level": "DEBUG",
            }
        },
    )
    merge_strategy: str = Field(
        default="deep_merge",
        description="How to merge the updates",
        json_schema_extra={"example": "deep_merge"},
    )
    validate_changes: bool = Field(
        default=True, description="Whether to validate changes before applying"
    )
    restart_required: bool | None = Field(
        default=None, description="Whether restart is required (auto-detected if None)"
    )

    @field_validator("merge_strategy")
    @classmethod
    def validate_merge_strategy(cls, v, info=None):
        """Validate merge strategy"""
        valid_strategies = ["deep_merge", "shallow_merge", "replace"]
        if v not in valid_strategies:
            raise ValueError(f"Merge strategy must be one of: {valid_strategies}")
        return v

    @field_validator("updates")
    @classmethod
    def validate_updates_not_empty(cls, v, info=None):
        """Ensure updates dict is not empty"""
        if not v:
            raise ValueError("Updates cannot be empty")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "updates": {
                    "websocket.max_connections": 2000,
                    "services.audio_service_timeout": 45,
                    "logging.level": "DEBUG",
                    "features.enable_analytics": True,
                },
                "merge_strategy": "deep_merge",
                "validate_changes": True,
                "restart_required": False,
            }
        }
    )


class ConfigResponse(ResponseMixin, TimestampMixin):
    """Configuration response"""

    current_config: dict[str, Any] = Field(description="Current configuration")
    schema_version: str = Field(
        description="Configuration schema version", json_schema_extra={"example": "2.0.0"}
    )
    last_modified: datetime = Field(description="Last modification timestamp")
    is_valid: bool = Field(description="Whether current configuration is valid")
    validation_errors: list[str] | None = Field(
        default=None, description="Configuration validation errors"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Configuration retrieved successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "current_config": {
                    "app_name": "LiveTranslate Orchestration Service",
                    "version": "2.0.0",
                    "host": "0.0.0.0",
                    "port": 3000,
                    "websocket": {"max_connections": 1000, "heartbeat_interval": 30},
                },
                "schema_version": "2.0.0",
                "last_modified": "2024-01-15T09:00:00Z",
                "is_valid": True,
                "validation_errors": None,
            }
        }
    )


class ConfigValidation(BaseModel):
    """Configuration validation result"""

    is_valid: bool = Field(description="Whether configuration is valid")
    errors: list[str] = Field(default_factory=list, description="Validation error messages")
    warnings: list[str] = Field(default_factory=list, description="Validation warning messages")
    suggestions: list[str] = Field(
        default_factory=list, description="Configuration improvement suggestions"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_valid": False,
                "errors": [
                    "Port must be between 1 and 65535",
                    "Invalid log level: TRACE",
                ],
                "warnings": ["High worker count may impact memory usage"],
                "suggestions": [
                    "Consider enabling compression for better performance",
                    "Set up SSL certificates for production",
                ],
            }
        }
    )


class ConfigBackup(BaseModel):
    """Configuration backup"""

    backup_id: str = Field(
        description="Backup identifier", json_schema_extra={"example": "backup_20240115_103000"}
    )
    config_data: dict[str, Any] = Field(description="Backed up configuration data")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Backup creation timestamp"
    )
    description: str | None = Field(
        default=None,
        description="Backup description",
        json_schema_extra={"example": "Pre-production deployment backup"},
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Backup tags",
        json_schema_extra={"example": ["production", "pre-deploy"]},
    )
    file_size_bytes: int | None = Field(default=None, description="Backup file size in bytes")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "backup_id": "backup_20240115_103000",
                "config_data": {
                    "app_name": "LiveTranslate Orchestration Service",
                    "version": "2.0.0",
                },
                "created_at": "2024-01-15T10:30:00Z",
                "description": "Pre-production deployment backup",
                "tags": ["production", "pre-deploy"],
                "file_size_bytes": 2048,
            }
        }
    )


class ConfigDiff(BaseModel):
    """Configuration difference"""

    added: dict[str, Any] = Field(default_factory=dict, description="Added configuration keys")
    modified: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Modified configuration keys with old/new values",
    )
    removed: dict[str, Any] = Field(default_factory=dict, description="Removed configuration keys")
    unchanged_count: int = Field(
        description="Number of unchanged keys", json_schema_extra={"example": 25}
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "added": {"features.new_feature": True},
                "modified": {
                    "websocket.max_connections": {"old": 1000, "new": 2000},
                    "logging.level": {"old": "INFO", "new": "DEBUG"},
                },
                "removed": {"deprecated.old_setting": "old_value"},
                "unchanged_count": 25,
            }
        }
    )


class ConfigUpdateResponse(ResponseMixin, TimestampMixin):
    """Configuration update response"""

    updated_keys: list[str] = Field(description="List of updated configuration keys")
    validation_result: ConfigValidation = Field(
        description="Validation result for the updated configuration"
    )
    restart_required: bool = Field(
        description="Whether a restart is required for changes to take effect"
    )
    backup_id: str | None = Field(
        default=None, description="ID of the backup created before update"
    )
    diff: ConfigDiff = Field(description="Configuration differences")
    rollback_available: bool = Field(description="Whether rollback is available")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Configuration updated successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "updated_keys": ["websocket.max_connections", "logging.level"],
                "validation_result": {
                    "is_valid": True,
                    "errors": [],
                    "warnings": [],
                    "suggestions": [],
                },
                "restart_required": False,
                "backup_id": "backup_20240115_103000",
                "diff": {
                    "added": {},
                    "modified": {"websocket.max_connections": {"old": 1000, "new": 2000}},
                    "removed": {},
                    "unchanged_count": 25,
                },
                "rollback_available": True,
            }
        }
    )


class ConfigHistory(BaseModel):
    """Configuration change history"""

    change_id: str = Field(
        description="Change identifier", json_schema_extra={"example": "change_20240115_103000"}
    )
    timestamp: datetime = Field(description="Change timestamp")
    user: str | None = Field(
        default=None,
        description="User who made the change",
        json_schema_extra={"example": "admin@example.com"},
    )
    description: str = Field(
        description="Change description",
        json_schema_extra={"example": "Updated WebSocket configuration"},
    )
    changes: ConfigDiff = Field(description="Configuration changes made")
    backup_id: str | None = Field(default=None, description="Associated backup ID")
    rollback_id: str | None = Field(
        default=None, description="Rollback change ID if this was a rollback"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "change_id": "change_20240115_103000",
                "timestamp": "2024-01-15T10:30:00Z",
                "user": "admin@example.com",
                "description": "Updated WebSocket configuration",
                "changes": {
                    "added": {},
                    "modified": {"websocket.max_connections": {"old": 1000, "new": 2000}},
                    "removed": {},
                    "unchanged_count": 25,
                },
                "backup_id": "backup_20240115_103000",
                "rollback_id": None,
            }
        }
    )


class FeatureFlags(BaseModel):
    """Feature flags configuration"""

    enable_audio_processing: bool = Field(
        default=True, description="Enable audio processing features"
    )
    enable_bot_management: bool = Field(default=True, description="Enable bot management features")
    enable_translation: bool = Field(default=True, description="Enable translation features")
    enable_analytics: bool = Field(default=True, description="Enable analytics features")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_debug_endpoints: bool = Field(
        default=False, description="Enable debug/development endpoints"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enable_audio_processing": True,
                "enable_bot_management": True,
                "enable_translation": True,
                "enable_analytics": True,
                "enable_metrics": True,
                "enable_debug_endpoints": False,
            }
        }
    )
