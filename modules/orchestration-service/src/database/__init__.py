"""
Database Package

Database models, configuration, and utilities for the orchestration service.
"""

from .chat_models import (
    APIToken,
    ChatMessage,
    ConversationSession,
    ConversationStatistics,
    User,
)
from .database import (
    DatabaseConfig,
    DatabaseManager,
    DatabaseUtils,
    MigrationManager,
    get_database_manager,
    get_db_session,
    initialize_database,
)
from .models import (
    AudioFile,
    Base,
    BotSession,
    Correlation,
    Glossary,
    GlossaryEntry,
    Participant,
    SessionEvent,
    SessionStatistics,
    Transcript,
    Translation,
)

__all__ = [
    "APIToken",
    "AudioFile",
    # Bot Session Models
    "Base",
    "BotSession",
    "ChatMessage",
    "ConversationSession",
    "ConversationStatistics",
    "Correlation",
    # Database management
    "DatabaseConfig",
    "DatabaseManager",
    "DatabaseUtils",
    # Glossary Models (unified for translation + Whisper prompting)
    "Glossary",
    "GlossaryEntry",
    "MigrationManager",
    "Participant",
    "SessionEvent",
    "SessionStatistics",
    "Transcript",
    "Translation",
    # Chat History Models
    "User",
    "get_database_manager",
    "get_db_session",
    "initialize_database",
]
