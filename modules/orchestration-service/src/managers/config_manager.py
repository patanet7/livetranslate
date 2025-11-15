"""
Configuration Manager

Centralized configuration management for the orchestration service.
"""

import logging
import json
import os
from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""

    host: str = "localhost"
    port: int = 5432
    database: str = "livetranslate"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class ServiceConfig:
    """Service configuration"""

    name: str
    url: str
    health_endpoint: str = "/health"
    timeout: int = 30
    retries: int = 3
    weight: int = 1
    enabled: bool = True


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""

    max_connections: int = 1000
    heartbeat_interval: int = 30
    session_timeout: int = 1800
    buffer_size: int = 8192
    ping_interval: int = 20
    ping_timeout: int = 10
    compression: bool = True


@dataclass
class APIGatewayConfig:
    """API Gateway configuration"""

    timeout: int = 30
    retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    enable_caching: bool = True
    cache_ttl: int = 300


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""

    enabled: bool = True
    health_check_interval: int = 10
    metrics_collection_interval: int = 5
    alert_thresholds: Dict[str, Union[int, float]] = field(
        default_factory=lambda: {
            "response_time": 1000,  # ms
            "error_rate": 0.05,  # 5%
            "memory_usage": 0.8,  # 80%
            "cpu_usage": 0.8,  # 80%
        }
    )
    prometheus_endpoint: str = "/metrics"
    grafana_url: str = "http://localhost:3001"


@dataclass
class SecurityConfig:
    """Security configuration"""

    jwt_secret: str = ""
    jwt_expiration: int = 3600
    cors_origins: list = field(default_factory=lambda: ["*"])
    cors_methods: list = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: list = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    blocked_ips: list = field(default_factory=list)
    rate_limiting: bool = True


@dataclass
class BotConfig:
    """Bot configuration"""

    max_concurrent_bots: int = 10
    bot_timeout: int = 3600
    audio_storage_path: str = "/tmp/audio"
    virtual_webcam_enabled: bool = True
    virtual_webcam_device: str = "/dev/video0"
    google_meet_credentials_path: str = ""
    cleanup_on_exit: bool = True


@dataclass
class OrchestrationConfig:
    """Main orchestration configuration"""

    host: str = "0.0.0.0"
    port: int = 3000
    workers: int = 4
    debug: bool = False
    log_level: str = "INFO"

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    api_gateway: APIGatewayConfig = field(default_factory=APIGatewayConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    bot: BotConfig = field(default_factory=BotConfig)

    # Service configurations
    services: Dict[str, ServiceConfig] = field(default_factory=dict)

    # Runtime info
    loaded_at: datetime = field(default_factory=datetime.now)
    config_file: Optional[str] = None


class ConfigFileHandler(FileSystemEventHandler):
    """Handle configuration file changes"""

    def __init__(self, config_manager):
        self.config_manager = config_manager

    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.config_manager.config_file:
            logger.info(f"Configuration file changed: {event.src_path}")
            self.config_manager.reload_config()


class ConfigManager:
    """
    Configuration manager with hot-reloading support
    """

    def __init__(self, config_file: Optional[str] = None, watch_changes: bool = True):
        self.config_file = config_file or self._find_config_file()
        self.watch_changes = watch_changes
        self._config: Optional[OrchestrationConfig] = None
        self._observer: Optional[Observer] = None
        self._callbacks = []

        # Load initial configuration
        self.load_config()

        # Start file watching if enabled
        if self.watch_changes and self.config_file:
            self._start_watching()

    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in common locations"""
        possible_paths = [
            "orchestration.yaml",
            "config/orchestration.yaml",
            "config.yaml",
            "/etc/livetranslate/orchestration.yaml",
            os.path.expanduser("~/.livetranslate/orchestration.yaml"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def load_config(self):
        """Load configuration from file and environment variables"""
        # Start with default configuration
        config = OrchestrationConfig()

        # Load from file if available
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    if self.config_file.endswith((".yaml", ".yml")):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)

                config = self._merge_config(config, file_config)
                config.config_file = self.config_file
                logger.info(f"Loaded configuration from {self.config_file}")

            except Exception as e:
                logger.error(f"Failed to load config file {self.config_file}: {e}")

        # Override with environment variables
        config = self._load_from_env(config)

        # Set loaded timestamp
        config.loaded_at = datetime.now()

        self._config = config

        # Notify callbacks
        self._notify_callbacks()

    def _merge_config(
        self, base_config: OrchestrationConfig, file_config: dict
    ) -> OrchestrationConfig:
        """Merge file configuration with base configuration"""
        # Main orchestration settings
        if "orchestration" in file_config:
            orch_config = file_config["orchestration"]
            base_config.host = orch_config.get("host", base_config.host)
            base_config.port = orch_config.get("port", base_config.port)
            base_config.workers = orch_config.get("workers", base_config.workers)
            base_config.debug = orch_config.get("debug", base_config.debug)
            base_config.log_level = orch_config.get("log_level", base_config.log_level)

        # Database configuration
        if "database" in file_config:
            db_config = file_config["database"]
            base_config.database.host = db_config.get("host", base_config.database.host)
            base_config.database.port = db_config.get("port", base_config.database.port)
            base_config.database.database = db_config.get(
                "database", base_config.database.database
            )
            base_config.database.username = db_config.get(
                "username", base_config.database.username
            )
            base_config.database.password = db_config.get(
                "password", base_config.database.password
            )
            base_config.database.pool_size = db_config.get(
                "pool_size", base_config.database.pool_size
            )

        # WebSocket configuration
        if "websocket" in file_config:
            ws_config = file_config["websocket"]
            base_config.websocket.max_connections = ws_config.get(
                "max_connections", base_config.websocket.max_connections
            )
            base_config.websocket.heartbeat_interval = ws_config.get(
                "heartbeat_interval", base_config.websocket.heartbeat_interval
            )
            base_config.websocket.session_timeout = ws_config.get(
                "session_timeout", base_config.websocket.session_timeout
            )

        # API Gateway configuration
        if "api_gateway" in file_config:
            gw_config = file_config["api_gateway"]
            base_config.api_gateway.timeout = gw_config.get(
                "timeout", base_config.api_gateway.timeout
            )
            base_config.api_gateway.retries = gw_config.get(
                "retries", base_config.api_gateway.retries
            )
            base_config.api_gateway.circuit_breaker_threshold = gw_config.get(
                "circuit_breaker_threshold",
                base_config.api_gateway.circuit_breaker_threshold,
            )

        # Services configuration
        if "services" in file_config:
            for service_name, service_config in file_config["services"].items():
                base_config.services[service_name] = ServiceConfig(
                    name=service_name,
                    url=service_config["url"],
                    health_endpoint=service_config.get("health_endpoint", "/health"),
                    timeout=service_config.get("timeout", 30),
                    retries=service_config.get("retries", 3),
                    weight=service_config.get("weight", 1),
                    enabled=service_config.get("enabled", True),
                )

        # Bot configuration
        if "bot" in file_config:
            bot_config = file_config["bot"]
            base_config.bot.max_concurrent_bots = bot_config.get(
                "max_concurrent_bots", base_config.bot.max_concurrent_bots
            )
            base_config.bot.bot_timeout = bot_config.get(
                "bot_timeout", base_config.bot.bot_timeout
            )
            base_config.bot.audio_storage_path = bot_config.get(
                "audio_storage_path", base_config.bot.audio_storage_path
            )
            base_config.bot.virtual_webcam_enabled = bot_config.get(
                "virtual_webcam_enabled", base_config.bot.virtual_webcam_enabled
            )
            base_config.bot.google_meet_credentials_path = bot_config.get(
                "google_meet_credentials_path",
                base_config.bot.google_meet_credentials_path,
            )

        return base_config

    def _load_from_env(self, config: OrchestrationConfig) -> OrchestrationConfig:
        """Load configuration from environment variables"""
        # Main orchestration settings
        config.host = os.getenv("ORCHESTRATION_HOST", config.host)
        config.port = int(os.getenv("ORCHESTRATION_PORT", config.port))
        config.workers = int(os.getenv("ORCHESTRATION_WORKERS", config.workers))
        config.debug = (
            os.getenv("ORCHESTRATION_DEBUG", str(config.debug)).lower() == "true"
        )
        config.log_level = os.getenv("ORCHESTRATION_LOG_LEVEL", config.log_level)

        # Database settings
        config.database.host = os.getenv("DATABASE_HOST", config.database.host)
        config.database.port = int(os.getenv("DATABASE_PORT", config.database.port))
        config.database.database = os.getenv("DATABASE_NAME", config.database.database)
        config.database.username = os.getenv(
            "DATABASE_USERNAME", config.database.username
        )
        config.database.password = os.getenv(
            "DATABASE_PASSWORD", config.database.password
        )

        # Service URLs
        if os.getenv("AUDIO_SERVICE_URL"):
            config.services["audio-service"] = ServiceConfig(
                name="audio-service",
                url=os.getenv("AUDIO_SERVICE_URL"),
                health_endpoint=os.getenv("AUDIO_SERVICE_HEALTH_ENDPOINT", "/health"),
            )

        if os.getenv("TRANSLATION_SERVICE_URL"):
            config.services["translation-service"] = ServiceConfig(
                name="translation-service",
                url=os.getenv("TRANSLATION_SERVICE_URL"),
                health_endpoint=os.getenv(
                    "TRANSLATION_SERVICE_HEALTH_ENDPOINT", "/health"
                ),
            )

        # Bot settings
        config.bot.max_concurrent_bots = int(
            os.getenv("BOT_MAX_CONCURRENT", config.bot.max_concurrent_bots)
        )
        config.bot.audio_storage_path = os.getenv(
            "BOT_AUDIO_STORAGE", config.bot.audio_storage_path
        )
        config.bot.google_meet_credentials_path = os.getenv(
            "GOOGLE_MEET_CREDENTIALS", config.bot.google_meet_credentials_path
        )

        # Security settings
        config.security.jwt_secret = os.getenv("JWT_SECRET", config.security.jwt_secret)

        return config

    def _start_watching(self):
        """Start watching configuration file for changes"""
        if not self.config_file:
            return

        try:
            self._observer = Observer()
            handler = ConfigFileHandler(self)
            watch_path = os.path.dirname(os.path.abspath(self.config_file))
            self._observer.schedule(handler, watch_path, recursive=False)
            self._observer.start()
            logger.info(f"Started watching config file: {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to start config file watching: {e}")

    def reload_config(self):
        """Reload configuration from file"""
        logger.info("Reloading configuration...")
        old_config = self._config
        self.load_config()
        logger.info("Configuration reloaded successfully")

        # Check for significant changes
        if old_config:
            self._check_config_changes(old_config, self._config)

    def _check_config_changes(
        self, old_config: OrchestrationConfig, new_config: OrchestrationConfig
    ):
        """Check for significant configuration changes"""
        changes = []

        # Check database changes
        if old_config.database.url != new_config.database.url:
            changes.append("Database connection changed")

        # Check service changes
        old_services = set(old_config.services.keys())
        new_services = set(new_config.services.keys())

        if old_services != new_services:
            changes.append(f"Services changed: {old_services} -> {new_services}")

        # Check WebSocket changes
        if old_config.websocket.max_connections != new_config.websocket.max_connections:
            changes.append(
                f"WebSocket max connections changed: {old_config.websocket.max_connections} -> {new_config.websocket.max_connections}"
            )

        if changes:
            logger.warning(f"Significant configuration changes detected: {changes}")

    def add_change_callback(self, callback):
        """Add callback to be called when configuration changes"""
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all registered callbacks about configuration changes"""
        for callback in self._callbacks:
            try:
                callback(self._config)
            except Exception as e:
                logger.error(f"Configuration change callback failed: {e}")

    @property
    def config(self) -> OrchestrationConfig:
        """Get current configuration"""
        return self._config

    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get configuration for specific service"""
        return self._config.services.get(service_name)

    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        output_file = config_file or self.config_file

        if not output_file:
            raise ValueError("No config file specified")

        try:
            # Convert dataclass to dict
            config_dict = {
                "orchestration": {
                    "host": self._config.host,
                    "port": self._config.port,
                    "workers": self._config.workers,
                    "debug": self._config.debug,
                    "log_level": self._config.log_level,
                },
                "database": {
                    "host": self._config.database.host,
                    "port": self._config.database.port,
                    "database": self._config.database.database,
                    "username": self._config.database.username,
                    "pool_size": self._config.database.pool_size,
                },
                "websocket": {
                    "max_connections": self._config.websocket.max_connections,
                    "heartbeat_interval": self._config.websocket.heartbeat_interval,
                    "session_timeout": self._config.websocket.session_timeout,
                },
                "api_gateway": {
                    "timeout": self._config.api_gateway.timeout,
                    "retries": self._config.api_gateway.retries,
                    "circuit_breaker_threshold": self._config.api_gateway.circuit_breaker_threshold,
                },
                "services": {
                    name: {
                        "url": service.url,
                        "health_endpoint": service.health_endpoint,
                        "timeout": service.timeout,
                        "retries": service.retries,
                        "weight": service.weight,
                        "enabled": service.enabled,
                    }
                    for name, service in self._config.services.items()
                },
                "bot": {
                    "max_concurrent_bots": self._config.bot.max_concurrent_bots,
                    "bot_timeout": self._config.bot.bot_timeout,
                    "audio_storage_path": self._config.bot.audio_storage_path,
                    "virtual_webcam_enabled": self._config.bot.virtual_webcam_enabled,
                    "google_meet_credentials_path": self._config.bot.google_meet_credentials_path,
                },
            }

            with open(output_file, "w") as f:
                if output_file.endswith((".yaml", ".yml")):
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    json.dump(config_dict, f, indent=2)

            logger.info(f"Configuration saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def shutdown(self):
        """Shutdown configuration manager"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("Configuration file watching stopped")

    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown()
