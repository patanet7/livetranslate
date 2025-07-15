"""
FastAPI Dependencies Module

Provides dependency injection for the orchestration service FastAPI application.
Includes managers, clients, and utilities needed across endpoints.
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Global instances (will be initialized in lifespan)
_config_manager = None
_websocket_manager = None
_health_monitor = None
_bot_manager = None
_audio_client = None
_translation_client = None
_audio_coordinator = None
_config_sync_manager = None

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# ============================================================================
# Manager Dependencies
# ============================================================================


def get_config_manager():
    """Get the configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        from managers.config_manager import ConfigManager

        _config_manager = ConfigManager()
    return _config_manager


def get_websocket_manager():
    """Get the WebSocket connection manager instance"""
    global _websocket_manager
    if _websocket_manager is None:
        from managers.websocket_manager import WebSocketManager

        _websocket_manager = WebSocketManager()
    return _websocket_manager


def get_health_monitor():
    """Get the health monitoring manager instance"""
    global _health_monitor
    if _health_monitor is None:
        from managers.health_monitor import HealthMonitor
        from config import get_settings

        settings = get_settings()
        _health_monitor = HealthMonitor(settings=settings)
    return _health_monitor


def get_bot_manager():
    """Get the bot management instance"""
    global _bot_manager
    if _bot_manager is None:
        from managers.bot_manager import BotManager

        _bot_manager = BotManager()
    return _bot_manager


# ============================================================================
# Service Client Dependencies
# ============================================================================


def get_audio_service_client():
    """Get the audio service client instance"""
    global _audio_client
    if _audio_client is None:
        from clients.audio_service_client import AudioServiceClient
        from config import get_settings

        config_manager = get_config_manager()
        settings = get_settings()
        _audio_client = AudioServiceClient(config_manager=config_manager, settings=settings)
    return _audio_client


def get_translation_service_client():
    """Get the translation service client instance"""
    global _translation_client
    if _translation_client is None:
        from clients.translation_service_client import TranslationServiceClient

        try:
            config_manager = get_config_manager()
            # Get translation service config, fallback to default if not configured
            translation_config = None
            if hasattr(config_manager, "config") and hasattr(config_manager.config, "services"):
                translation_config = config_manager.config.services.get("translation-service")
            _translation_client = TranslationServiceClient(translation_config)
        except Exception as e:
            logger.warning(f"Failed to get translation config, using default: {e}")
            # Create with default config
            _translation_client = TranslationServiceClient(None)
    return _translation_client


def get_audio_coordinator():
    """Get the audio coordinator instance"""
    global _audio_coordinator
    if _audio_coordinator is None:
        from audio.audio_coordinator import create_audio_coordinator
        from audio.models import get_default_chunking_config
        import os

        # Get database URL from environment or disable for development
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            # For development without PostgreSQL, disable database features
            logger.warning("No DATABASE_URL set, audio coordinator will run without database persistence")
            database_url = None

        # Configure service URLs
        service_urls = {
            "whisper_service": os.getenv("WHISPER_SERVICE_URL", "http://localhost:5001"),
            "translation_service": os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:5003"),
        }

        # Create audio coordinator with default config
        config = get_default_chunking_config()
        _audio_coordinator = create_audio_coordinator(
            database_url=database_url, service_urls=service_urls, config=config
        )
    return _audio_coordinator


def get_database_adapter():
    """Get the database adapter instance"""
    import os
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("No DATABASE_URL set, database adapter disabled")
        return None
    
    from audio.database_adapter import AudioDatabaseAdapter
    return AudioDatabaseAdapter(database_url)


def get_config_sync_manager():
    """Get the configuration sync manager instance"""
    global _config_sync_manager
    if _config_sync_manager is None:
        from audio.config_sync import ConfigurationSyncManager
        import os

        # Get service URLs for sync manager
        whisper_url = os.getenv("WHISPER_SERVICE_URL", "http://localhost:5001")
        translation_url = os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:5003")
        orchestration_url = os.getenv("ORCHESTRATION_SERVICE_URL", "http://localhost:3000")

        _config_sync_manager = ConfigurationSyncManager(
            whisper_service_url=whisper_url,
            orchestration_service_url=orchestration_url,
            translation_service_url=translation_url
        )
    return _config_sync_manager


# ============================================================================
# Database Dependencies
# ============================================================================

_database_manager = None


def get_database_manager():
    """Get the database manager instance"""
    global _database_manager
    if _database_manager is None:
        from database.database import DatabaseManager, DatabaseConfig

        # Use SQLite for testing, PostgreSQL for production
        db_config = DatabaseConfig(url="sqlite+aiosqlite:///:memory:")
        _database_manager = DatabaseManager(db_config)
        _database_manager.initialize()
    return _database_manager


async def get_database():
    """Get database session"""
    db_manager = get_database_manager()
    async with db_manager.get_session() as session:
        yield session


# ============================================================================
# Authentication Dependencies
# ============================================================================


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """
    Get current authenticated user (optional authentication)
    Returns None if no valid credentials provided
    """
    if not credentials:
        return None

    try:
        # For now, simple token validation
        # In production, implement proper JWT validation
        token = credentials.credentials

        # Basic token validation (implement proper JWT validation in production)
        if token == "dev-token":
            return {"user_id": "dev-user", "roles": ["admin"]}

        # Invalid token
        return None

    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        return None


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    Require valid authentication
    Raises HTTPException if no valid credentials
    """
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
    """
    Require admin role
    Raises HTTPException if user doesn't have admin role
    """
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


# ============================================================================
# Rate Limiting Dependencies
# ============================================================================


@lru_cache(maxsize=1)
def get_rate_limiter():
    """Get rate limiter instance"""
    from utils.rate_limiting import RateLimiter

    return RateLimiter()


async def rate_limit_general(request: Request, rate_limiter=Depends(get_rate_limiter)):
    """General rate limiting (100 requests per minute)"""
    client_ip = request.client.host
    if not await rate_limiter.is_allowed(client_ip, "general", limit=100, window=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded"
        )


async def rate_limit_websocket(request: Request, rate_limiter=Depends(get_rate_limiter)):
    """WebSocket rate limiting (10 connections per IP)"""
    client_ip = request.client.host
    if not await rate_limiter.is_allowed(client_ip, "websocket", limit=10, window=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="WebSocket connection limit exceeded",
        )


async def rate_limit_api(request: Request, rate_limiter=Depends(get_rate_limiter)):
    """API rate limiting (500 requests per minute)"""
    client_ip = request.client.host
    if not await rate_limiter.is_allowed(client_ip, "api", limit=500, window=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="API rate limit exceeded",
        )


# ============================================================================
# Validation Dependencies
# ============================================================================


async def validate_file_upload(request: Request):
    """Validate file upload requirements"""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 100 * 1024 * 1024:  # 100MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large (max 100MB)",
        )


# ============================================================================
# Utility Dependencies
# ============================================================================


@lru_cache(maxsize=1)
def get_audio_processor():
    """Get audio processing utilities"""
    from utils.audio_processing import AudioProcessor

    return AudioProcessor()


@lru_cache(maxsize=1)
def get_security_utils():
    """Get security utilities"""
    from utils.security import SecurityUtils

    return SecurityUtils()


# ============================================================================
# Lifecycle Management
# ============================================================================


async def initialize_dependencies():
    """Initialize all dependencies during application startup"""
    global _config_manager, _websocket_manager, _health_monitor, _bot_manager
    global _audio_client, _translation_client, _audio_coordinator, _config_sync_manager

    # Validate environment and dependencies first
    try:
        from utils.dependency_check import validate_startup_environment
        feature_availability = validate_startup_environment()
        logger.info(f"🎯 Available features: {feature_availability}")
    except Exception as e:
        logger.warning(f"⚠️  Dependency validation failed: {e}")
        feature_availability = {}

    logger.info("🔧 Initializing dependencies...")

    try:
        # Initialize managers
        _config_manager = get_config_manager()
        _websocket_manager = get_websocket_manager()
        _health_monitor = get_health_monitor()
        _bot_manager = get_bot_manager()

        # Initialize service clients
        _audio_client = get_audio_service_client()
        _translation_client = get_translation_service_client()

        # Initialize audio processing components
        _audio_coordinator = get_audio_coordinator()
        _config_sync_manager = get_config_sync_manager()

        # Start async managers
        if hasattr(_websocket_manager, "start"):
            await _websocket_manager.start()
        if hasattr(_health_monitor, "start"):
            await _health_monitor.start()
        if hasattr(_bot_manager, "start"):
            await _bot_manager.start()
        if hasattr(_audio_coordinator, "initialize"):
            await _audio_coordinator.initialize()
        if hasattr(_config_sync_manager, "initialize"):
            await _config_sync_manager.initialize()

        logger.info("✅ Dependencies initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize dependencies: {e}")
        raise


async def cleanup_dependencies():
    """Cleanup all dependencies during application shutdown"""
    global _config_manager, _websocket_manager, _health_monitor, _bot_manager
    global _audio_client, _translation_client, _audio_coordinator, _config_sync_manager

    logger.info("🛑 Cleaning up dependencies...")

    try:
        # Stop audio processing components
        if _audio_coordinator and hasattr(_audio_coordinator, "shutdown"):
            await _audio_coordinator.shutdown()
        if _config_sync_manager and hasattr(_config_sync_manager, "shutdown"):
            await _config_sync_manager.shutdown()

        # Stop async managers
        if _bot_manager and hasattr(_bot_manager, "stop"):
            await _bot_manager.stop()
        if _health_monitor and hasattr(_health_monitor, "stop"):
            await _health_monitor.stop()
        if _websocket_manager and hasattr(_websocket_manager, "stop"):
            await _websocket_manager.stop()

        # Close service clients
        if _audio_client and hasattr(_audio_client, "close"):
            await _audio_client.close()
        if _translation_client and hasattr(_translation_client, "close"):
            await _translation_client.close()

        logger.info("✅ Dependencies cleaned up successfully")

    except Exception as e:
        logger.error(f"❌ Failed to cleanup dependencies: {e}")


# ============================================================================
# Development Dependencies
# ============================================================================


def get_development_mode() -> bool:
    """Check if running in development mode"""
    import os

    return os.getenv("ENVIRONMENT", "development") == "development"


def development_only(dev_mode: bool = Depends(get_development_mode)):
    """Dependency that only allows access in development mode"""
    if not dev_mode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Endpoint not available in production",
        )
    return True
