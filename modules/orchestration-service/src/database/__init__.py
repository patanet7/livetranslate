"""
Database Package

Database models, configuration, and utilities for the orchestration service.
"""

from .database import (
    DatabaseConfig,
    DatabaseManager,
    get_database_manager,
    initialize_database,
    get_db_session,
    DatabaseUtils,
    MigrationManager,
)
from .models import (
    Base,
    BotSession,
    AudioFile,
    Transcript,
    Translation,
    Correlation,
    Participant,
    SessionEvent,
    SessionStatistics,
    DatabaseManager as ModelDatabaseManager,
)

__all__ = [
    # Database management
    "DatabaseConfig",
    "DatabaseManager",
    "get_database_manager",
    "initialize_database",
    "get_db_session",
    "DatabaseUtils",
    "MigrationManager",
    # Models
    "Base",
    "BotSession",
    "AudioFile",
    "Transcript",
    "Translation",
    "Correlation",
    "Participant",
    "SessionEvent",
    "SessionStatistics",
    "ModelDatabaseManager",
]
