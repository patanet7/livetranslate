"""
Database integration with SQLAlchemy for the FastAPI backend

Modern async database operations with comprehensive models and migrations
"""

from .base import Base, get_db_session, init_database
from .models import (
    User,
    Session,
    BotInstance,
    AudioSession,
    TranslationSession,
    SystemMetrics,
)
from .repositories import (
    UserRepository,
    SessionRepository,
    BotRepository,
    AudioRepository,
    TranslationRepository,
    MetricsRepository,
)

__all__ = [
    # Base database
    "Base",
    "get_db_session",
    "init_database",
    # Models
    "User",
    "Session",
    "BotInstance",
    "AudioSession",
    "TranslationSession",
    "SystemMetrics",
    # Repositories
    "UserRepository",
    "SessionRepository",
    "BotRepository",
    "AudioRepository",
    "TranslationRepository",
    "MetricsRepository",
]
