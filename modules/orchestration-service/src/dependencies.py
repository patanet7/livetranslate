#!/usr/bin/env python3
"""
Dependencies for FastAPI Dependency Injection

Provides singleton instances and dependency injection for all orchestration service components.
Manages lifecycle of managers, clients, and shared resources across the application.
"""

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

# Client imports
from clients.audio_service_client import AudioServiceClient
from clients.translation_service_client import TranslationServiceClient

# Audio system imports
from audio.audio_coordinator import AudioCoordinator
from audio.config_sync import ConfigurationSyncManager
from audio.config import AudioConfigurationManager

# Utility imports
from utils.rate_limiting import RateLimiter
from utils.security import SecurityUtils

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
        redis_url = os.getenv("EVENT_BUS_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        enabled = os.getenv("EVENT_BUS_ENABLED", "true").lower() not in {"0", "false", "off"}
        _event_publisher = EventPublisher(
            redis_url=redis_url,
            streams={**DEFAULT_STREAMS},
            enabled=enabled,
        )
        logger.info("EventPublisher initialized (enabled=%s, redis_url=%s)", enabled, redis_url)
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
        config = get_config_manager()
        _database_manager = DatabaseManager(config.database)
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
# Service Clients
# ============================================================================

@lru_cache()
def get_audio_service_client() -> AudioServiceClient:
    """Get singleton AudioServiceClient instance."""
    global _audio_service_client
    if _audio_service_client is None:
        logger.info("Initializing AudioServiceClient singleton")
        config = get_config_manager()
        _audio_service_client = AudioServiceClient(
            base_url=config.get_service_url("whisper"),
            timeout=config.get("services.whisper.timeout", 30)
        )
        logger.info("AudioServiceClient initialized successfully")
    return _audio_service_client


@lru_cache()
def get_translation_service_client() -> TranslationServiceClient:
    """Get singleton TranslationServiceClient instance."""
    global _translation_service_client
    if _translation_service_client is None:
        logger.info("Initializing TranslationServiceClient singleton")
        config = get_config_manager()
        _translation_service_client = TranslationServiceClient(
            base_url=config.get_service_url("translation"),
            timeout=config.get("services.translation.timeout", 30)
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
        config = get_config_manager()
        _websocket_manager = WebSocketManager(
            max_connections=config.get("websocket.max_connections", 1000),
            connection_timeout=config.get("websocket.connection_timeout", 1800)
        )
        logger.info("WebSocketManager initialized successfully")
    return _websocket_manager


@lru_cache()
def get_health_monitor() -> HealthMonitor:
    """Get singleton HealthMonitor instance."""
    global _health_monitor
    if _health_monitor is None:
        logger.info("Initializing HealthMonitor singleton")
        config = get_config_manager()
        _health_monitor = HealthMonitor(
            check_interval=config.get("monitoring.check_interval", 30),
            services=config.get("services", {})
        )
        logger.info("HealthMonitor initialized successfully")
    return _health_monitor


@lru_cache()
def get_bot_manager() -> BotManager:
    """Get singleton BotManager instance."""
    global _bot_manager
    if _bot_manager is None:
        logger.info("Initializing BotManager singleton")
        config = get_config_manager()
        database = get_unified_repository()
        _bot_manager = BotManager(
            config=config,
            database=database,
            max_concurrent_bots=config.get("bot.max_concurrent", 10)
        )
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
        config = get_config_manager()
        audio_client = get_audio_service_client()
        translation_client = get_translation_service_client()
        _audio_coordinator = AudioCoordinator(
            config=config,
            audio_client=audio_client,
            translation_client=translation_client
        )
        logger.info("AudioCoordinator initialized successfully")
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
        config = get_config_manager()
        _rate_limiter = RateLimiter(
            max_requests=config.get("rate_limiting.max_requests", 100),
            time_window=config.get("rate_limiting.time_window", 60)
        )
        logger.info("RateLimiter initialized successfully")
    return _rate_limiter


@lru_cache()
def get_security_utils() -> SecurityUtils:
    """Get singleton SecurityUtils instance."""
    global _security_utils
    if _security_utils is None:
        logger.info("Initializing SecurityUtils singleton")
        config = get_config_manager()
        _security_utils = SecurityUtils(
            secret_key=config.get("security.secret_key"),
            token_expiry=config.get("security.token_expiry", 3600)
        )
        logger.info("SecurityUtils initialized successfully")
    return _security_utils


# ============================================================================
# Lifecycle Management
# ============================================================================

async def startup_dependencies():
    """Initialize all dependencies on application startup."""
    logger.info("Starting dependency initialization...")
    
    try:
        # Initialize core managers first
        config = get_config_manager()
        logger.info(" ConfigManager initialized")
        
        # Initialize database components
        db_manager = get_database_manager()
        repository = get_unified_repository()
        logger.info(" Database components initialized")
        
        # Initialize service clients
        audio_client = get_audio_service_client()
        translation_client = get_translation_service_client()
        logger.info(" Service clients initialized")
        
        # Initialize system managers
        websocket_manager = get_websocket_manager()
        health_monitor = get_health_monitor()
        bot_manager = get_bot_manager()
        logger.info(" System managers initialized")
        
        # Initialize audio processing
        audio_coordinator = get_audio_coordinator()
        config_sync = get_config_sync_manager()
        audio_config = get_audio_config_manager()
        logger.info(" Audio processing components initialized")
        
        # Initialize utilities
        rate_limiter = get_rate_limiter()
        security_utils = get_security_utils()
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
        if _bot_manager:
            await _bot_manager.shutdown()
            logger.info(" BotManager shutdown")
            
        if _websocket_manager:
            await _websocket_manager.shutdown()
            logger.info(" WebSocketManager shutdown")
            
        if _health_monitor:
            await _health_monitor.shutdown()
            logger.info(" HealthMonitor shutdown")
            
        if _audio_coordinator:
            await _audio_coordinator.shutdown()
            logger.info(" AudioCoordinator shutdown")
            
        if _database_manager:
            await _database_manager.shutdown()
            logger.info(" DatabaseManager shutdown")
            
        logger.info("All dependencies shutdown successfully")
        
    except Exception as e:
        logger.error(f"Error during dependency shutdown: {e}")


# ============================================================================
# Health Checks
# ============================================================================

async def health_check_dependencies() -> Dict[str, Any]:
    """Check health of all dependency components."""
    health_status = {
        "status": "healthy",
        "components": {},
        "timestamp": None
    }
    
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
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
            else:
                health_status["components"][name] = {
                    "status": "not_initialized" if component is None else "no_health_check"
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
    global _audio_config_manager, _rate_limiter, _security_utils
    
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
    
    # Clear LRU cache
    get_config_manager.cache_clear()
    get_websocket_manager.cache_clear()
    get_health_monitor.cache_clear()
    get_bot_manager.cache_clear()
    get_database_manager.cache_clear()
    get_unified_repository.cache_clear()
    get_audio_service_client.cache_clear()
    get_translation_service_client.cache_clear()
    get_audio_coordinator.cache_clear()
    get_config_sync_manager.cache_clear()
    get_audio_config_manager.cache_clear()
    get_rate_limiter.cache_clear()
    get_security_utils.cache_clear()
