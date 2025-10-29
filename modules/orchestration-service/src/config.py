#!/usr/bin/env python3
"""
Configuration Management for FastAPI Backend

Centralized configuration using Pydantic Settings with environment variable support
and validation.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator, ValidationInfo, ConfigDict
except ImportError:
    from pydantic import BaseSettings, Field, field_validator, ValidationInfo, ConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration"""

    url: str = Field(
        default="postgresql://localhost:5432/livetranslate",
        env="DATABASE_URL",
        description="Database connection URL",
    )
    pool_size: int = Field(
        default=10,
        env="DATABASE_POOL_SIZE",
        description="Database connection pool size",
    )
    max_overflow: int = Field(
        default=20,
        env="DATABASE_MAX_OVERFLOW",
        description="Database connection pool overflow",
    )
    pool_timeout: int = Field(
        default=30,
        env="DATABASE_POOL_TIMEOUT",
        description="Database connection pool timeout in seconds",
    )
    pool_recycle: int = Field(
        default=3600,
        env="DATABASE_POOL_RECYCLE",
        description="Database connection pool recycle time in seconds",
    )
    pool_pre_ping: bool = Field(
        default=True,
        env="DATABASE_POOL_PRE_PING",
        description="Enable connection health checks",
    )
    echo: bool = Field(
        default=False, env="DATABASE_ECHO", description="Enable SQL query logging"
    )

    model_config = ConfigDict(env_prefix="DATABASE_")


class RedisSettings(BaseSettings):
    """Redis configuration"""

    url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis connection URL",
    )
    max_connections: int = Field(
        default=10, env="REDIS_MAX_CONNECTIONS", description="Maximum Redis connections"
    )
    socket_timeout: int = Field(
        default=5,
        env="REDIS_SOCKET_TIMEOUT",
        description="Redis socket timeout in seconds",
    )

    model_config = ConfigDict(env_prefix="REDIS_")


class WebSocketSettings(BaseSettings):
    """WebSocket configuration"""

    max_connections: int = Field(
        default=1000,
        env="WEBSOCKET_MAX_CONNECTIONS",
        description="Maximum concurrent WebSocket connections",
    )
    heartbeat_interval: int = Field(
        default=30,
        env="WEBSOCKET_HEARTBEAT_INTERVAL",
        description="WebSocket heartbeat interval in seconds",
    )
    message_buffer_size: int = Field(
        default=100,
        env="WEBSOCKET_MESSAGE_BUFFER_SIZE",
        description="WebSocket message buffer size",
    )
    compression: bool = Field(
        default=True,
        env="WEBSOCKET_COMPRESSION",
        description="Enable WebSocket compression",
    )

    model_config = ConfigDict(env_prefix="WEBSOCKET_")


class ServiceSettings(BaseSettings):
    """External service configuration"""

    # Audio service
    audio_service_url: str = Field(
        default="http://localhost:5001",
        env="AUDIO_SERVICE_URL",
        description="Audio service URL",
    )
    audio_service_timeout: int = Field(
        default=30,
        env="AUDIO_SERVICE_TIMEOUT",
        description="Audio service request timeout",
    )

    # Translation service
    translation_service_url: str = Field(
        default="http://localhost:5003",
        env="TRANSLATION_SERVICE_URL",
        description="Translation service URL",
    )
    translation_service_timeout: int = Field(
        default=30,
        env="TRANSLATION_SERVICE_TIMEOUT",
        description="Translation service request timeout",
    )

    # Health check settings
    health_check_interval: int = Field(
        default=10,
        env="HEALTH_CHECK_INTERVAL",
        description="Health check interval in seconds",
    )
    health_check_timeout: int = Field(
        default=5,
        env="HEALTH_CHECK_TIMEOUT",
        description="Health check timeout in seconds",
    )

    model_config = ConfigDict(env_prefix="SERVICE_")


class SecuritySettings(BaseSettings):
    """Security configuration"""

    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY",
        description="Secret key for JWT tokens",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES",
        description="Access token expiration time in minutes",
    )
    algorithm: str = Field(
        default="HS256", env="JWT_ALGORITHM", description="JWT algorithm"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS",
        description="Allowed CORS origins",
    )

    @field_validator("cors_origins", mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    model_config = ConfigDict(env_prefix="SECURITY_")


class LoggingSettings(BaseSettings):
    """Logging configuration"""

    level: str = Field(default="INFO", env="LOG_LEVEL", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT",
        description="Log format string",
    )
    file_path: Optional[str] = Field(
        default=None, env="LOG_FILE_PATH", description="Log file path (optional)"
    )
    max_file_size_mb: int = Field(
        default=10,
        env="LOG_MAX_FILE_SIZE_MB",
        description="Maximum log file size in MB",
    )
    backup_count: int = Field(
        default=5, env="LOG_BACKUP_COUNT", description="Number of log file backups"
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    model_config = ConfigDict(env_prefix="LOG_")


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration"""

    enable_metrics: bool = Field(
        default=True, env="ENABLE_METRICS", description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=8000, env="METRICS_PORT", description="Prometheus metrics server port"
    )
    health_check_endpoint: str = Field(
        default="/health",
        env="HEALTH_CHECK_ENDPOINT",
        description="Health check endpoint path",
    )

    model_config = ConfigDict(env_prefix="MONITORING_")


class BotSettings(BaseSettings):
    """Bot management configuration"""

    # Docker settings
    docker_image: str = Field(
        default="livetranslate-bot:latest",
        env="BOT_DOCKER_IMAGE",
        description="Docker image for bot containers"
    )
    docker_network: str = Field(
        default="livetranslate_default",
        env="BOT_DOCKER_NETWORK",
        description="Docker network for bot containers"
    )

    # Database settings (optional)
    enable_database: bool = Field(
        default=False,
        env="BOT_ENABLE_DATABASE",
        description="Enable database persistence for bots"
    )
    database_host: str = Field(
        default="localhost",
        env="BOT_DATABASE_HOST",
        description="Bot database host"
    )
    database_port: int = Field(
        default=5432,
        env="BOT_DATABASE_PORT",
        description="Bot database port"
    )
    database_name: str = Field(
        default="livetranslate",
        env="BOT_DATABASE_NAME",
        description="Bot database name"
    )
    database_user: str = Field(
        default="postgres",
        env="BOT_DATABASE_USER",
        description="Bot database username"
    )
    database_password: str = Field(
        default="",
        env="BOT_DATABASE_PASSWORD",
        description="Bot database password"
    )

    # Storage settings
    audio_storage_path: str = Field(
        default="/tmp/livetranslate/audio",
        env="BOT_AUDIO_STORAGE_PATH",
        description="Path for bot audio file storage"
    )

    # Google Account Authentication (for restricted meetings)
    google_email: str = Field(
        default="",
        description="Google account email for bot authentication"
    )
    google_password: str = Field(
        default="",
        description="Google account password (use App Password if 2FA enabled)"
    )

    # Persistent Browser Profile (keeps bot logged in)
    user_data_dir: str = Field(
        default="/tmp/bot-browser-profile",
        env="BOT_USER_DATA_DIR",
        description="Path to persistent browser profile (stores login state)"
    )

    # Browser Settings
    headless: bool = Field(
        default=True,
        env="BOT_HEADLESS",
        description="Run browser in headless mode"
    )
    screenshots_enabled: bool = Field(
        default=True,
        env="BOT_SCREENSHOTS_ENABLED",
        description="Enable screenshot debugging"
    )
    screenshots_path: str = Field(
        default="/tmp/bot-screenshots",
        env="BOT_SCREENSHOTS_PATH",
        description="Path for bot screenshots"
    )

    model_config = ConfigDict(env_prefix="BOT_", env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """Main application settings"""

    # Basic server settings
    app_name: str = Field(
        default="LiveTranslate Orchestration Service",
        env="APP_NAME",
        description="Application name",
    )
    version: str = Field(
        default="2.0.0", env="APP_VERSION", description="Application version"
    )
    description: str = Field(
        default="Modern FastAPI backend for LiveTranslate orchestration",
        env="APP_DESCRIPTION",
        description="Application description",
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST", description="Server host")
    port: int = Field(default=3000, env="PORT", description="Server port")
    workers: int = Field(
        default=4, env="WORKERS", description="Number of worker processes"
    )
    debug: bool = Field(default=False, env="DEBUG", description="Enable debug mode")

    # Environment
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Environment (development, staging, production)",
    )

    # Subsystem settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    websocket: WebSocketSettings = WebSocketSettings()
    services: ServiceSettings = ServiceSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    bot: BotSettings = BotSettings()

    # Feature flags
    enable_audio_processing: bool = Field(
        default=True,
        env="ENABLE_AUDIO_PROCESSING",
        description="Enable audio processing features",
    )
    enable_bot_management: bool = Field(
        default=True,
        env="ENABLE_BOT_MANAGEMENT",
        description="Enable bot management features",
    )
    enable_translation: bool = Field(
        default=True,
        env="ENABLE_TRANSLATION",
        description="Enable translation features",
    )
    enable_analytics: bool = Field(
        default=True, env="ENABLE_ANALYTICS", description="Enable analytics features"
    )

    # File paths
    static_files_path: str = Field(
        default="frontend/dist",
        env="STATIC_FILES_PATH",
        description="Path to static files",
    )
    upload_path: str = Field(
        default="uploads", env="UPLOAD_PATH", description="Path for file uploads"
    )

    # Rate limiting
    rate_limit_requests: int = Field(
        default=100,
        env="RATE_LIMIT_REQUESTS",
        description="Rate limit requests per minute",
    )
    rate_limit_websocket_connections: int = Field(
        default=10,
        env="RATE_LIMIT_WEBSOCKET_CONNECTIONS",
        description="Rate limit WebSocket connections per IP",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError("Workers must be at least 1")
        return v

    def get_database_url(self) -> str:
        """Get database URL with proper formatting"""
        return self.database.url

    def get_redis_url(self) -> str:
        """Get Redis URL with proper formatting"""
        return self.redis.url

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins list"""
        return self.security.cors_origins

    def get_service_urls(self) -> Dict[str, str]:
        """Get all service URLs"""
        return {
            "audio": self.services.audio_service_url,
            "translation": self.services.translation_service_url,
        }

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="allow")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML/JSON file"""
    import json
    import yaml

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        if config_file.suffix.lower() in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_file.suffix.lower() == ".json":
            return json.load(f)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {config_file.suffix}"
            )


def create_settings_from_file(config_path: str) -> Settings:
    """Create settings instance from configuration file"""
    config_data = load_config_from_file(config_path)
    return Settings(**config_data)


# Development settings override
def get_development_settings() -> Settings:
    """Get development-specific settings"""
    return Settings(
        debug=True,
        environment="development",
        logging=LoggingSettings(level="DEBUG"),
        database=DatabaseSettings(echo=True),
        security=SecuritySettings(
            cors_origins=[
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
            ]
        ),
    )


# Production settings override
def get_production_settings() -> Settings:
    """Get production-specific settings"""
    return Settings(
        debug=False,
        environment="production",
        logging=LoggingSettings(
            level="INFO", file_path="/var/log/orchestration-service.log"
        ),
        database=DatabaseSettings(echo=False),
        workers=8,  # More workers for production
        websocket=WebSocketSettings(max_connections=5000),  # Higher connection limit
    )
