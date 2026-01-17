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
)
from .chat_models import (
    User,
    APIToken,
    ConversationSession,
    ChatMessage,
    ConversationStatistics,
)
from .models import (
    Glossary,
    GlossaryEntry,
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
    # Bot Session Models
    "Base",
    "BotSession",
    "AudioFile",
    "Transcript",
    "Translation",
    "Correlation",
    "Participant",
    "SessionEvent",
    "SessionStatistics",
    # Chat History Models
    "User",
    "APIToken",
    "ConversationSession",
    "ChatMessage",
    "ConversationStatistics",
    # Glossary Models (unified for translation + Whisper prompting)
    "Glossary",
    "GlossaryEntry",
]
