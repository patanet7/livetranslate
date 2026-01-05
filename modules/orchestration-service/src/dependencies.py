#!/usr/bin/env python3
"""
Dependencies for FastAPI Dependency Injection

Provides singleton instances and dependency injection for all orchestration service components.
Manages lifecycle of managers, clients, and shared resources across the application.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any
from functools import lru_cache

# Manager imports
from managers.config_manager import ConfigManager
from managers.websocket_manager import WebSocketManager
from managers.health_monitor import HealthMonitor
from managers.bot_manager import BotManager

# Infrastructure
from infrastructure.queue import EventPublisher, DEFAULT_STREAMS

# Database imports
from database.database import DatabaseManager
from database.unified_bot_session_repository import UnifiedBotSessionRepository
from config import DatabaseSettings

# Client imports
from clients.audio_service_client import AudioServiceClient
from clients.translation_service_client import TranslationServiceClient

# Audio system imports
from audio.audio_coordinator import AudioCoordinator, create_audio_coordinator
from audio.config_sync import ConfigurationSyncManager
from audio.config import AudioConfigurationManager

# Data pipeline imports
try:
    from pipeline.data_pipeline import TranscriptionDataPipeline, create_data_pipeline
except ImportError:
    TranscriptionDataPipeline = None
    create_data_pipeline = None
    logger.warning(
        "TranscriptionDataPipeline not available - using legacy database adapter"
    )

# Utility imports
from utils.rate_limiting import RateLimiter
from utils.security import SecurityUtils
from fastapi import Depends, Request, HTTPException, status

logger = logging.getLogger(__name__)

# ============================================================================
# Singleton Instances (Global State Management)
# ============================================================================

_config_manager: Optional[ConfigManager] = None
_websocket_manager: Optional[WebSocketManager] = None
_health_monitor: Optional[HealthMonitor] = None
_bot_manager: Optional[BotManager] = None
_database_manager: Optional[DatabaseManager] = None
_unified_repository: Optional[UnifiedBotSessionRepository] = None
_audio_service_client: Optional[AudioServiceClient] = None
_translation_service_client: Optional[TranslationServiceClient] = None
_audio_coordinator: Optional[AudioCoordinator] = None
_config_sync_manager: Optional[ConfigurationSyncManager] = None
_audio_config_manager: Optional[AudioConfigurationManager] = None
_rate_limiter: Optional[RateLimiter] = None
_security_utils: Optional[SecurityUtils] = None
_event_publisher: Optional[EventPublisher] = None
_data_pipeline: Optional["TranscriptionDataPipeline"] = None


# ============================================================================
# Configuration Management
# ============================================================================


@lru_cache()
def get_config_manager() -> ConfigManager:
    """Get singleton ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        logger.info("Initializing ConfigManager singleton")
        _config_manager = ConfigManager()
        logger.info("ConfigManager initialized successfully")
    return _config_manager


@lru_cache()
def get_config_sync_manager() -> ConfigurationSyncManager:
    """Get singleton ConfigurationSyncManager instance."""
    global _config_sync_manager
    if _config_sync_manager is None:
        logger.info("Initializing ConfigurationSyncManager singleton")
        _config_sync_manager = ConfigurationSyncManager()
        logger.info("ConfigurationSyncManager initialized successfully")
    return _config_sync_manager


@lru_cache()
def get_audio_config_manager() -> AudioConfigurationManager:
    """Get singleton AudioConfigurationManager instance."""
    global _audio_config_manager
    if _audio_config_manager is None:
        logger.info("Initializing AudioConfigurationManager singleton")
        _audio_config_manager = AudioConfigurationManager()
        logger.info("AudioConfigurationManager initialized successfully")
    return _audio_config_manager


# ============================================================================
# Event Publisher
# ============================================================================


@lru_cache()
def get_event_publisher() -> EventPublisher:
    """Get singleton EventPublisher instance."""
    global _event_publisher
    if _event_publisher is None:
        redis_url = os.getenv(
            "EVENT_BUS_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )
        enabled = os.getenv("EVENT_BUS_ENABLED", "true").lower() not in {
            "0",
            "false",
            "off",
        }
        _event_publisher = EventPublisher(
            redis_url=redis_url,
            streams={**DEFAULT_STREAMS},
            enabled=enabled,
        )
        logger.info(
            "EventPublisher initialized (enabled=%s, redis_url=%s)", enabled, redis_url
        )
    return _event_publisher


# ============================================================================
# Database Management
# ============================================================================


@lru_cache()
def get_database_manager() -> DatabaseManager:
    """Get singleton DatabaseManager instance."""
    global _database_manager
    if _database_manager is None:
        logger.info("Initializing DatabaseManager singleton")
        config_manager = get_config_manager()
        settings = getattr(config_manager, "config", None)

        database_url = os.getenv("DATABASE_URL")
        if not database_url and settings and getattr(settings, "database", None):
            db_cfg = settings.database
            if hasattr(db_cfg, "url"):
                database_url = db_cfg.url
            else:
                host = getattr(db_cfg, "host", "localhost")
                port = getattr(db_cfg, "port", 5432)
                name = getattr(db_cfg, "database", "livetranslate")
                user = getattr(db_cfg, "username", "postgres")
                password = getattr(db_cfg, "password", "")
                database_url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
        db_settings = (
            DatabaseSettings(url=database_url) if database_url else DatabaseSettings()
        )
        _database_manager = DatabaseManager(db_settings)
        logger.info("DatabaseManager initialized successfully")
    return _database_manager


@lru_cache()
def get_unified_repository() -> UnifiedBotSessionRepository:
    """Get singleton UnifiedBotSessionRepository instance."""
    global _unified_repository
    if _unified_repository is None:
        logger.info("Initializing UnifiedBotSessionRepository singleton")
        db_manager = get_database_manager()
        _unified_repository = UnifiedBotSessionRepository(db_manager)
        logger.info("UnifiedBotSessionRepository initialized successfully")
    return _unified_repository


# ============================================================================
# Data Pipeline (Production-Ready Database Operations)
# ============================================================================


@lru_cache()
def get_data_pipeline() -> Optional["TranscriptionDataPipeline"]:
    """
    Get singleton TranscriptionDataPipeline instance.

    This is the production-ready database pipeline with:
    - NULL-safe timeline queries
    - LRU cache with automatic eviction
    - Connection pooling with proper configuration
    - Transaction support with automatic rollback
    - Rate limiting and backpressure protection

    Returns:
        TranscriptionDataPipeline instance if available, None otherwise
    """
    global _data_pipeline

    if not create_data_pipeline:
        logger.warning(
            "TranscriptionDataPipeline not available - pipeline module not imported"
        )
        return None

    if _data_pipeline is None:
        logger.info("Initializing TranscriptionDataPipeline singleton")

        # Get bot manager's database manager (already initialized with bot sessions)
        bot_manager = get_bot_manager()

        if hasattr(bot_manager, "database_manager") and bot_manager.database_manager:
            logger.info("Using bot_manager's database_manager for pipeline")
            _data_pipeline = create_data_pipeline(
                database_manager=bot_manager.database_manager,
                audio_storage_path=None,  # Uses database_manager's storage path
                enable_speaker_tracking=True,
                enable_segment_continuity=True,
            )
            logger.info(
                "TranscriptionDataPipeline initialized successfully with production fixes"
            )
        else:
            logger.warning(
                "Bot manager database not available - pipeline not initialized"
            )
            return None

    return _data_pipeline


# ============================================================================
# Service Clients
# ============================================================================


@lru_cache()
def get_audio_service_client() -> AudioServiceClient:
    """Get singleton AudioServiceClient instance."""
    global _audio_service_client
    if _audio_service_client is None:
        logger.info("Initializing AudioServiceClient singleton")
        config_manager = get_config_manager()
        service_config = (
            config_manager.get_service_config("audio-service")
            if hasattr(config_manager, "get_service_config")
            else None
        )
        base_url = os.getenv("AUDIO_SERVICE_URL")
        if not base_url and service_config:
            base_url = service_config.url
        if not base_url:
            base_url = "http://localhost:5001"
        timeout = int(os.getenv("AUDIO_SERVICE_TIMEOUT", 30))
        _audio_service_client = AudioServiceClient(base_url=base_url, timeout=timeout)
        logger.info("AudioServiceClient initialized successfully")
    return _audio_service_client


@lru_cache()
def get_translation_service_client() -> TranslationServiceClient:
    """Get singleton TranslationServiceClient instance."""
    global _translation_service_client
    if _translation_service_client is None:
        logger.info("Initializing TranslationServiceClient singleton")
        config_manager = get_config_manager()
        service_config = (
            config_manager.get_service_config("translation-service")
            if hasattr(config_manager, "get_service_config")
            else None
        )
        base_url = os.getenv("TRANSLATION_SERVICE_URL")
        if not base_url and service_config:
            base_url = service_config.url
        if not base_url:
            base_url = "http://localhost:5003"
        timeout = int(os.getenv("TRANSLATION_SERVICE_TIMEOUT", 30))
        _translation_service_client = TranslationServiceClient(
            base_url=base_url, timeout=timeout
        )
        logger.info("TranslationServiceClient initialized successfully")
    return _translation_service_client


# ============================================================================
# Manager Components
# ============================================================================


@lru_cache()
def get_websocket_manager() -> WebSocketManager:
    """Get singleton WebSocketManager instance."""
    global _websocket_manager
    if _websocket_manager is None:
        logger.info("Initializing WebSocketManager singleton")
        _websocket_manager = WebSocketManager()
        logger.info("WebSocketManager initialized successfully")
    return _websocket_manager


@lru_cache()
def get_health_monitor() -> HealthMonitor:
    """Get singleton HealthMonitor instance."""
    global _health_monitor
    if _health_monitor is None:
        logger.info("Initializing HealthMonitor singleton")
        # Always start with default settings; we'll override URLs below.
        _health_monitor = HealthMonitor(settings=None)

        # Override service URLs from environment or defaults
        audio_url = os.getenv(
            "AUDIO_SERVICE_URL"
        ) or _health_monitor.service_configs.get("whisper", {}).get(
            "url", "http://localhost:5001"
        )
        translation_url = os.getenv(
            "TRANSLATION_SERVICE_URL"
        ) or _health_monitor.service_configs.get("translation", {}).get(
            "url", "http://localhost:5003"
        )
        orchestration_url = os.getenv(
            "ORCHESTRATION_URL"
        ) or _health_monitor.service_configs.get("orchestration", {}).get(
            "url", "http://localhost:3000"
        )

        _health_monitor.service_configs = {
            "whisper": {"url": audio_url, "health_endpoint": "/health"},
            "translation": {"url": translation_url, "health_endpoint": "/api/health"},
            "orchestration": {
                "url": orchestration_url,
                "health_endpoint": "/api/health",
            },
        }
        logger.info("HealthMonitor initialized successfully")
    return _health_monitor


@lru_cache()
def get_bot_manager() -> BotManager:
    """Get singleton BotManager instance."""
    global _bot_manager
    if _bot_manager is None:
        logger.info("Initializing BotManager singleton")
        database = get_unified_repository()
        settings = getattr(get_config_manager(), "config", None)
        if isinstance(settings, dict):
            bot_cfg = settings.get("bot")
        else:
            bot_cfg = getattr(settings, "bot", None) if settings else None

        if bot_cfg is not None:
            from managers.config_manager import BotConfig as ConfigBotSettings

            bot_settings = ConfigBotSettings(
                max_concurrent_bots=getattr(bot_cfg, "max_concurrent_bots", 10),
                bot_timeout=getattr(bot_cfg, "bot_timeout", 3600),
                audio_storage_path=getattr(bot_cfg, "audio_storage_path", "/tmp/audio"),
                virtual_webcam_enabled=getattr(bot_cfg, "virtual_webcam_enabled", True),
                virtual_webcam_device=getattr(
                    bot_cfg, "virtual_webcam_device", "/dev/video0"
                ),
                google_meet_credentials_path=getattr(
                    bot_cfg, "google_meet_credentials_path", ""
                ),
                cleanup_on_exit=getattr(bot_cfg, "cleanup_on_exit", True),
            )
        else:
            bot_settings = None

        _bot_manager = BotManager(config=bot_settings)
        if database:
            _bot_manager.database_client = database
        logger.info("BotManager initialized successfully")
    return _bot_manager


# ============================================================================
# Audio Processing Components
# ============================================================================


@lru_cache()
def get_audio_coordinator() -> AudioCoordinator:
    """Get singleton AudioCoordinator instance."""
    global _audio_coordinator
    if _audio_coordinator is None:
        logger.info("Initializing AudioCoordinator singleton")

        config_manager = get_config_manager()
        settings = getattr(config_manager, "config", None)

        # Resolve database URL if available
        database_url = None
        if settings:
            if isinstance(settings, dict):
                database_cfg = settings.get("database")
                if isinstance(database_cfg, dict):
                    database_url = database_cfg.get("url")
                    if not database_url:
                        host = database_cfg.get("host", "localhost")
                        port = database_cfg.get("port", 5432)
                        name = database_cfg.get("database", "livetranslate")
                        user = database_cfg.get("username", "postgres")
                        password = database_cfg.get("password", "")
                        database_url = (
                            f"postgresql://{user}:{password}@{host}:{port}/{name}"
                        )
                elif database_cfg and hasattr(database_cfg, "url"):
                    database_url = database_cfg.url
            else:
                database_cfg = getattr(settings, "database", None)
                if database_cfg and hasattr(database_cfg, "url"):
                    database_url = database_cfg.url
        if not database_url:
            database_url = os.getenv("DATABASE_URL")

        # Resolve downstream service URLs (fallback to environment/defaults)
        whisper_service_cfg = (
            config_manager.get_service_config("audio-service")
            if hasattr(config_manager, "get_service_config")
            else None
        )
        translation_service_cfg = (
            config_manager.get_service_config("translation-service")
            if hasattr(config_manager, "get_service_config")
            else None
        )

        whisper_url = os.getenv("AUDIO_SERVICE_URL")
        if not whisper_url and whisper_service_cfg:
            whisper_url = whisper_service_cfg.url
        if not whisper_url:
            whisper_url = "http://localhost:5001"

        translation_url = os.getenv("TRANSLATION_SERVICE_URL")
        if not translation_url and translation_service_cfg:
            translation_url = translation_service_cfg.url
        if not translation_url:
            translation_url = "http://localhost:5003"

        service_urls = {
            "whisper_service": whisper_url,
            "translation_service": translation_url,
        }

        audio_client = get_audio_service_client()
        translation_client = get_translation_service_client()

        # Get production-ready data pipeline (preferred over legacy database_url)
        data_pipeline = get_data_pipeline()

        max_sessions = int(os.getenv("AUDIO_MAX_CONCURRENT_SESSIONS", "10"))
        audio_config_file = os.getenv("AUDIO_CONFIG_FILE")

        _audio_coordinator = create_audio_coordinator(
            database_url=database_url
            if not data_pipeline
            else None,  # Use database_url only if pipeline not available
            service_urls=service_urls,
            config=None,
            max_concurrent_sessions=max_sessions,
            audio_config_file=audio_config_file,
            audio_client=audio_client,
            translation_client=translation_client,
            data_pipeline=data_pipeline,  # Pass production-ready pipeline
        )

        if data_pipeline:
            logger.info(
                "AudioCoordinator instance created with TranscriptionDataPipeline (production-ready)"
            )
        elif database_url:
            logger.warning(
                "AudioCoordinator instance created with legacy AudioDatabaseAdapter (deprecated)"
            )
        else:
            logger.info("AudioCoordinator instance created without persistence")

    return _audio_coordinator


# ============================================================================
# Utility Components
# ============================================================================


@lru_cache()
def get_rate_limiter() -> RateLimiter:
    """Get singleton RateLimiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        logger.info("Initializing RateLimiter singleton")
        _rate_limiter = RateLimiter()
        logger.info("RateLimiter initialized successfully")
    return _rate_limiter


@lru_cache()
def get_security_utils() -> SecurityUtils:
    """Get singleton SecurityUtils instance."""
    global _security_utils
    if _security_utils is None:
        logger.info("Initializing SecurityUtils singleton")
        settings = getattr(get_config_manager(), "config", None)
        if settings and not isinstance(settings, dict):
            security_cfg = getattr(settings, "security", None)
            secret = getattr(security_cfg, "secret_key", None)
            expiry_minutes = getattr(security_cfg, "access_token_expire_minutes", 60)
        else:
            security_cfg = (
                settings.get("security") if isinstance(settings, dict) else {}
            )
            secret = None
            expiry_minutes = 60
            if isinstance(security_cfg, dict):
                secret = security_cfg.get("secret_key")
                expiry_minutes = security_cfg.get("access_token_expire_minutes", 60)

        secret = os.getenv("SECRET_KEY", secret or "change-me")
        expiry_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", expiry_minutes))

        _security_utils = SecurityUtils(secret_key=secret)
        logger.info("SecurityUtils initialized successfully")
    return _security_utils


async def rate_limit_api(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """Simple sliding-window rate limiter dependency for API endpoints."""
    client_id = request.client.host if request.client else "unknown"
    endpoint = request.url.path
    limit = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    if not await rate_limiter.is_allowed(client_id, endpoint, limit, window):
        remaining = await rate_limiter.get_remaining(client_id, endpoint, limit, window)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": "Too many requests",
                "retry_after": window,
                "remaining": remaining,
            },
        )


async def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Placeholder authentication dependency.

    Returns:
        Optional user payload. Currently unauthenticated, expands later.
    """
    return None


# ============================================================================
# Lifecycle Management
# ============================================================================


async def startup_dependencies():
    """Initialize all dependencies on application startup."""
    logger.info("Starting dependency initialization...")

    try:
        # Initialize core managers first
        _ = get_config_manager()
        logger.info(" ConfigManager initialized")

        # Initialize database components
        _ = get_database_manager()
        _ = get_unified_repository()
        logger.info(" Database components initialized")

        # Initialize service clients
        _ = get_audio_service_client()
        _ = get_translation_service_client()
        logger.info(" Service clients initialized")

        # Initialize system managers
        _ = get_websocket_manager()
        _ = get_health_monitor()
        _ = get_bot_manager()
        logger.info(" System managers initialized")

        # Initialize data pipeline (must be before audio coordinator)
        data_pipeline = get_data_pipeline()
        if data_pipeline:
            logger.info(" Data pipeline initialized (production-ready)")
        else:
            logger.warning(" Data pipeline not available (using legacy adapter)")

        # Initialize audio processing
        _ = get_audio_coordinator()
        _ = get_config_sync_manager()
        _ = get_audio_config_manager()
        logger.info(" Audio processing components initialized")

        # Initialize utilities
        _ = get_rate_limiter()
        _ = get_security_utils()
        logger.info(" Utility components initialized")

        logger.info("All dependencies initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {e}")
        raise


async def shutdown_dependencies():
    """Clean shutdown of all dependencies."""
    logger.info("Starting dependency shutdown...")

    try:
        # Shutdown in reverse order
        if _bot_manager and hasattr(_bot_manager, "stop"):
            await _bot_manager.stop()
            logger.info(" BotManager shutdown")

        if _websocket_manager and hasattr(_websocket_manager, "stop"):
            await _websocket_manager.stop()
            logger.info(" WebSocketManager shutdown")

        if _health_monitor and hasattr(_health_monitor, "stop"):
            await _health_monitor.stop()
            logger.info(" HealthMonitor shutdown")

        if _audio_coordinator:
            await _audio_coordinator.shutdown()
            logger.info(" AudioCoordinator shutdown")

        if _database_manager:
            close_fn = getattr(_database_manager, "close", None)
            if close_fn:
                maybe_coro = close_fn()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
            logger.info(" DatabaseManager shutdown")

        logger.info("All dependencies shutdown successfully")

    except Exception as e:
        logger.error(f"Error during dependency shutdown: {e}")


# ============================================================================
# Health Checks
# ============================================================================


async def health_check_dependencies() -> Dict[str, Any]:
    """Check health of all dependency components."""
    health_status = {"status": "healthy", "components": {}, "timestamp": None}

    try:
        from datetime import datetime

        health_status["timestamp"] = datetime.utcnow().isoformat()

        # Check each component
        components = [
            ("config_manager", _config_manager),
            ("database_manager", _database_manager),
            ("websocket_manager", _websocket_manager),
            ("health_monitor", _health_monitor),
            ("bot_manager", _bot_manager),
            ("audio_coordinator", _audio_coordinator),
        ]

        for name, component in components:
            if component and hasattr(component, "health_check"):
                try:
                    component_health = await component.health_check()
                    health_status["components"][name] = component_health
                except Exception as e:
                    health_status["components"][name] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }
                    health_status["status"] = "degraded"
            else:
                health_status["components"][name] = {
                    "status": "not_initialized"
                    if component is None
                    else "no_health_check"
                }

    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)

    return health_status


# ============================================================================
# Dependency Reset (for testing)
# ============================================================================


def reset_dependencies():
    """Reset all singleton instances (for testing only)."""
    global _config_manager, _websocket_manager, _health_monitor, _bot_manager
    global _database_manager, _unified_repository, _audio_service_client
    global _translation_service_client, _audio_coordinator, _config_sync_manager
    global _audio_config_manager, _rate_limiter, _security_utils, _data_pipeline

    logger.warning("Resetting all dependency singletons")

    _config_manager = None
    _websocket_manager = None
    _health_monitor = None
    _bot_manager = None
    _database_manager = None
    _unified_repository = None
    _audio_service_client = None
    _translation_service_client = None
    _audio_coordinator = None
    _config_sync_manager = None
    _audio_config_manager = None
    _rate_limiter = None
    _security_utils = None
    _data_pipeline = None

    # Clear LRU cache
    get_config_manager.cache_clear()
    get_websocket_manager.cache_clear()
    get_health_monitor.cache_clear()
    get_bot_manager.cache_clear()
    get_database_manager.cache_clear()
    get_unified_repository.cache_clear()
    get_data_pipeline.cache_clear()
    get_audio_service_client.cache_clear()
    get_translation_service_client.cache_clear()
    get_audio_coordinator.cache_clear()
    get_config_sync_manager.cache_clear()
    get_audio_config_manager.cache_clear()
    get_rate_limiter.cache_clear()
    get_security_utils.cache_clear()
