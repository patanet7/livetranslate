#!/usr/bin/env python3
"""
Configuration Management for FastAPI Backend

Centralized configuration using Pydantic Settings with environment variable support
and validation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator, ValidationInfo, ConfigDict
except ImportError:
    from pydantic import (
        BaseSettings,
        Field,
        field_validator,
        ConfigDict,
    )


class DatabaseSettings(BaseSettings):
    """Database configuration"""

    url: str = Field(
        default="postgresql://localhost:5432/livetranslate",
        description="Database connection URL",
    )
    pool_size: int = Field(
        default=10,
        description="Database connection pool size",
    )
    max_overflow: int = Field(
        default=20,
        description="Database connection pool overflow",
    )
    pool_timeout: int = Field(
        default=30,
        description="Database connection pool timeout in seconds",
    )
    pool_recycle: int = Field(
        default=3600,
        description="Database connection pool recycle time in seconds",
    )
    pool_pre_ping: bool = Field(
        default=True,
        description="Enable connection health checks",
    )
    echo: bool = Field(
        default=False,
        description="Enable SQL query logging",
    )

    model_config = ConfigDict(env_prefix="DATABASE_")

    @property
    def async_url(self) -> str:
        """Get async database URL (postgresql+asyncpg://) for SQLAlchemy async"""
        url = self.url
        # Convert postgresql:// to postgresql+asyncpg://
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url


class RedisSettings(BaseSettings):
    """Redis configuration"""

    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    max_connections: int = Field(
        default=10,
        description="Maximum Redis connections",
    )
    socket_timeout: int = Field(
        default=5,
        description="Redis socket timeout in seconds",
    )

    model_config = ConfigDict(env_prefix="REDIS_")


class WebSocketSettings(BaseSettings):
    """WebSocket configuration"""

    max_connections: int = Field(
        default=1000,
        description="Maximum concurrent WebSocket connections",
    )
    heartbeat_interval: int = Field(
        default=30,
        description="WebSocket heartbeat interval in seconds",
    )
    message_buffer_size: int = Field(
        default=100,
        description="WebSocket message buffer size",
    )
    compression: bool = Field(
        default=True,
        description="Enable WebSocket compression",
    )

    model_config = ConfigDict(env_prefix="WEBSOCKET_")


class ServiceSettings(BaseSettings):
    """External service configuration"""

    # Audio service
    audio_service_url: str = Field(
        default="http://localhost:5001",
        description="Audio service URL",
    )
    audio_service_timeout: int = Field(
        default=30,
        description="Audio service request timeout",
    )

    # Translation service
    translation_service_url: str = Field(
        default="http://localhost:5003",
        description="Translation service URL",
    )
    translation_service_timeout: int = Field(
        default=30,
        description="Translation service request timeout",
    )

    # Health check settings
    health_check_interval: int = Field(
        default=10,
        description="Health check interval in seconds",
    )
    health_check_timeout: int = Field(
        default=5,
        description="Health check timeout in seconds",
    )

    model_config = ConfigDict(env_prefix="SERVICE_")


class SecuritySettings(BaseSettings):
    """Security configuration"""

    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes",
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT algorithm",
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    model_config = ConfigDict(env_prefix="SECURITY_")


class LoggingSettings(BaseSettings):
    """Logging configuration"""

    level: str = Field(
        default="INFO",
        description="Logging level",
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Log file path (optional)",
    )
    max_file_size_mb: int = Field(
        default=10,
        description="Maximum log file size in MB",
    )
    backup_count: int = Field(
        default=5,
        description="Number of log file backups",
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
        default=True,
        description="Enable Prometheus metrics",
    )
    metrics_port: int = Field(
        default=8000,
        description="Prometheus metrics server port",
    )
    health_check_endpoint: str = Field(
        default="/health",
        description="Health check endpoint path",
    )

    model_config = ConfigDict(env_prefix="MONITORING_")


class BotSettings(BaseSettings):
    """Bot management configuration"""

    # Docker settings
    docker_image: str = Field(
        default="livetranslate-bot:latest",
        description="Docker image for bot containers",
    )
    docker_network: str = Field(
        default="livetranslate_default",
        description="Docker network for bot containers",
    )

    # Database settings (optional)
    enable_database: bool = Field(
        default=False,
        description="Enable database persistence for bots",
    )
    database_host: str = Field(
        default="localhost",
        description="Bot database host",
    )
    database_port: int = Field(
        default=5432,
        description="Bot database port",
    )
    database_name: str = Field(
        default="livetranslate",
        description="Bot database name",
    )
    database_user: str = Field(
        default="postgres",
        description="Bot database username",
    )
    database_password: str = Field(
        default="",
        description="Bot database password",
    )

    # Storage settings
    audio_storage_path: str = Field(
        default="/tmp/livetranslate/audio",
        description="Path for bot audio file storage",
    )

    # Google Account Authentication (for restricted meetings)
    google_email: str = Field(
        default="",
        description="Google account email for bot authentication",
    )
    google_password: str = Field(
        default="",
        description="Google account password (use App Password if 2FA enabled)",
    )

    # Persistent Browser Profile (keeps bot logged in)
    user_data_dir: str = Field(
        default="/tmp/bot-browser-profile",
        description="Path to persistent browser profile (stores login state)",
    )

    # Browser Settings
    headless: bool = Field(
        default=True,
        description="Run browser in headless mode",
    )
    screenshots_enabled: bool = Field(
        default=True,
        description="Enable screenshot debugging",
    )
    screenshots_path: str = Field(
        default="/tmp/bot-screenshots",
        description="Path for bot screenshots",
    )

    model_config = ConfigDict(env_prefix="BOT_", env_file=".env", extra="ignore")


class FirefliesSettings(BaseSettings):
    """Fireflies.ai integration configuration"""

    # API Configuration
    api_key: str = Field(
        default="",
        description="Fireflies API key for authentication",
    )
    graphql_endpoint: str = Field(
        default="https://api.fireflies.ai/graphql",
        description="Fireflies GraphQL API endpoint",
    )
    websocket_endpoint: str = Field(
        default="wss://api.fireflies.ai/realtime",
        description="Fireflies WebSocket API endpoint",
    )

    # Sentence Aggregation Settings
    pause_threshold_ms: float = Field(
        default=800.0,
        description="Pause duration (ms) indicating sentence boundary",
    )
    max_buffer_words: int = Field(
        default=30,
        description="Maximum words before forcing translation",
    )
    max_buffer_seconds: float = Field(
        default=5.0,
        description="Maximum seconds before forcing translation",
    )
    min_words_for_translation: int = Field(
        default=3,
        description="Minimum words required for translation",
    )
    use_nlp_boundary_detection: bool = Field(
        default=True,
        description="Use spaCy for sentence boundary detection",
    )

    # Translation Context Settings
    context_window_size: int = Field(
        default=3,
        description="Number of previous sentences for context",
    )
    include_cross_speaker_context: bool = Field(
        default=True,
        description="Include other speakers in context window",
    )

    # Default Target Languages
    default_target_languages: List[str] = Field(
        default=["es"],
        description="Default target languages for translation",
    )

    # Connection Settings
    auto_reconnect: bool = Field(
        default=True,
        description="Auto-reconnect on disconnect",
    )
    max_reconnect_attempts: int = Field(
        default=5,
        description="Maximum reconnection attempts",
    )

    @field_validator("default_target_languages", mode="before")
    @classmethod
    def parse_target_languages(cls, v):
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v

    def has_api_key(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key and self.api_key.strip())

    model_config = ConfigDict(env_prefix="FIREFLIES_", env_file=".env", extra="ignore")


class OBSSettings(BaseSettings):
    """OBS WebSocket output configuration"""

    # Connection Settings
    host: str = Field(
        default="localhost",
        description="OBS WebSocket server host",
    )
    port: int = Field(
        default=4455,
        description="OBS WebSocket server port",
    )
    password: Optional[str] = Field(
        default=None,
        description="OBS WebSocket server password (optional)",
    )

    # Source Configuration
    caption_source: str = Field(
        default="LiveTranslate Caption",
        description="Name of text source for captions",
    )
    speaker_source: Optional[str] = Field(
        default=None,
        description="Name of text source for speaker names",
    )

    # Behavior Settings
    auto_reconnect: bool = Field(
        default=True,
        description="Auto-reconnect on disconnect",
    )
    reconnect_interval: float = Field(
        default=5.0,
        description="Seconds between reconnection attempts",
    )
    max_reconnect_attempts: int = Field(
        default=3,
        description="Maximum reconnection attempts",
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds",
    )

    # Display Options
    show_original: bool = Field(
        default=False,
        description="Show original text alongside translation",
    )
    create_sources: bool = Field(
        default=False,
        description="Create text sources if they don't exist",
    )

    def is_configured(self) -> bool:
        """Check if OBS connection is configured"""
        return bool(self.host and self.port)

    model_config = ConfigDict(env_prefix="OBS_", env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """Main application settings"""

    # Basic server settings
    app_name: str = Field(
        default="LiveTranslate Orchestration Service",
        description="Application name",
    )
    version: str = Field(
        default="2.0.0",
        description="Application version",
    )
    description: str = Field(
        default="Modern FastAPI backend for LiveTranslate orchestration",
        description="Application description",
    )

    # Server configuration
    host: str = Field(
        default="0.0.0.0",
        description="Server host",
    )
    port: int = Field(
        default=3000,
        description="Server port",
    )
    workers: int = Field(
        default=1,
        description="Number of worker processes (use 1 for caption buffer consistency)",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Environment
    environment: str = Field(
        default="development",
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
    fireflies: FirefliesSettings = FirefliesSettings()
    obs: OBSSettings = OBSSettings()

    # Feature flags
    enable_audio_processing: bool = Field(
        default=True,
        description="Enable audio processing features",
    )
    enable_bot_management: bool = Field(
        default=True,
        description="Enable bot management features",
    )
    enable_translation: bool = Field(
        default=True,
        description="Enable translation features",
    )
    enable_analytics: bool = Field(
        default=True,
        description="Enable analytics features",
    )

    # File paths
    static_files_path: str = Field(
        default="frontend/dist",
        description="Path to static files",
    )
    upload_path: str = Field(
        default="uploads",
        description="Path for file uploads",
    )

    # Rate limiting
    rate_limit_requests: int = Field(
        default=100,
        description="Rate limit requests per minute",
    )
    rate_limit_websocket_connections: int = Field(
        default=10,
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
