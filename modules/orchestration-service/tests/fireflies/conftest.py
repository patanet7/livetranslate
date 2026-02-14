#!/usr/bin/env python3
"""
Fireflies Test Configuration and Fixtures

Provides shared fixtures, configuration, and utilities for all
Fireflies integration tests.
"""

import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add src to path for imports
orchestration_root = Path(__file__).parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "fireflies: marks tests as fireflies integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# =============================================================================
# Import Models and Clients
# =============================================================================

try:
    from src.models.fireflies import (
        CaptionEntry,
        FirefliesChunk,
        FirefliesConnectionStatus,
        FirefliesMeeting,
        FirefliesSession,
        FirefliesSessionConfig,
        MeetingState,
        TranslationContext,
        TranslationResult,
        TranslationUnit,
    )
    from src.routers.fireflies import FirefliesSessionManager

    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Fireflies modules: {e}")
    MODELS_AVAILABLE = False


# =============================================================================
# Session Fixtures
# =============================================================================


@pytest.fixture
def api_key():
    """Provide test API key."""
    return "ff-test-api-key-12345"


@pytest.fixture
def transcript_id():
    """Provide test transcript ID."""
    return "test-transcript-abc123"


@pytest.fixture
def session_id():
    """Provide test session ID."""
    return "ff_session_test123"


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def sample_chunk(transcript_id):
    """Create sample FirefliesChunk."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return FirefliesChunk(
        transcript_id=transcript_id,
        chunk_id="chunk_001",
        text="Hello, this is a test transcription.",
        speaker_name="Alice",
        start_time=0.0,
        end_time=2.5,
    )


@pytest.fixture
def sample_chunks(transcript_id):
    """Create list of sample chunks for a conversation."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return [
        FirefliesChunk(
            transcript_id=transcript_id,
            chunk_id="chunk_001",
            text="Hi everyone.",
            speaker_name="Alice",
            start_time=0.0,
            end_time=1.0,
        ),
        FirefliesChunk(
            transcript_id=transcript_id,
            chunk_id="chunk_002",
            text="Hello Alice.",
            speaker_name="Bob",
            start_time=1.0,
            end_time=2.0,
        ),
        FirefliesChunk(
            transcript_id=transcript_id,
            chunk_id="chunk_003",
            text="Let's start the meeting.",
            speaker_name="Alice",
            start_time=2.0,
            end_time=4.0,
        ),
    ]


@pytest.fixture
def sample_meeting():
    """Create sample FirefliesMeeting."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return FirefliesMeeting(
        id="meeting-001",
        title="Team Standup",
        organizer_email="user@example.com",
        meeting_link="https://zoom.us/j/123456",
        start_time=datetime.now(UTC),
        state=MeetingState.ACTIVE,
    )


@pytest.fixture
def sample_session_config(api_key, transcript_id):
    """Create sample FirefliesSessionConfig."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return FirefliesSessionConfig(
        api_key=api_key,
        transcript_id=transcript_id,
        target_languages=["es", "fr"],
        pause_threshold_ms=800.0,
        max_buffer_words=30,
        context_window_size=3,
    )


@pytest.fixture
def sample_session(transcript_id, session_id):
    """Create sample FirefliesSession."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return FirefliesSession(
        session_id=session_id,
        fireflies_transcript_id=transcript_id,
        connection_status=FirefliesConnectionStatus.CONNECTED,
        connected_at=datetime.now(UTC),
        chunks_received=10,
        sentences_produced=3,
        translations_completed=9,
        speakers_detected=["Alice", "Bob"],
        error_count=0,
    )


@pytest.fixture
def sample_translation_unit(transcript_id, session_id):
    """Create sample TranslationUnit."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return TranslationUnit(
        text="Hello, how are you today?",
        speaker_name="Alice",
        start_time=0.0,
        end_time=2.5,
        session_id=session_id,
        transcript_id=transcript_id,
        chunk_ids=["chunk_001", "chunk_002"],
        boundary_type="punctuation",
    )


@pytest.fixture
def sample_translation_context():
    """Create sample TranslationContext."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return TranslationContext(
        previous_sentences=[
            "Good morning everyone.",
            "Today we will discuss the project status.",
        ],
        glossary={
            "API": "API",
            "backend": "servidor",
            "frontend": "interfaz",
        },
        target_language="es",
        source_language="en",
    )


@pytest.fixture
def sample_translation_result(session_id):
    """Create sample TranslationResult."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return TranslationResult(
        original="Hello, how are you today?",
        translated="Hola, ¿cómo estás hoy?",
        speaker_name="Alice",
        source_language="en",
        target_language="es",
        confidence=0.95,
        context_sentences_used=2,
        glossary_terms_applied=["API"],
        translation_time_ms=125.5,
        session_id=session_id,
    )


@pytest.fixture
def sample_caption_entry():
    """Create sample CaptionEntry."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return CaptionEntry(
        id="caption-001",
        original_text="Hello, how are you?",
        translated_text="Hola, ¿cómo estás?",
        speaker_name="Alice",
        speaker_color="#4CAF50",
        target_language="es",
        duration_seconds=8.0,
        confidence=0.95,
    )


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_session_manager():
    """Create mock FirefliesSessionManager."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    manager = AsyncMock(spec=FirefliesSessionManager)
    manager._sessions = {}
    manager._clients = {}
    manager.get_session.return_value = None
    manager.get_all_sessions.return_value = []
    return manager


@pytest.fixture
def mock_fireflies_config():
    """Create mock Fireflies configuration."""
    config = MagicMock()
    config.api_key = "default-test-api-key"
    config.graphql_endpoint = "https://api.fireflies.ai/graphql"
    config.websocket_endpoint = "wss://api.fireflies.ai/realtime"
    config.pause_threshold_ms = 800.0
    config.max_buffer_words = 30
    config.context_window_size = 3
    config.default_target_languages = ["es"]
    config.has_api_key.return_value = True
    return config


@pytest.fixture
def mock_graphql_client(api_key):
    """Create mock GraphQL client."""
    client = AsyncMock()
    client.api_key = api_key
    client.endpoint = "https://api.fireflies.ai/graphql"
    client._session = None
    return client


@pytest.fixture
def mock_realtime_client(api_key, transcript_id):
    """Create mock realtime client."""
    client = AsyncMock()
    client.api_key = api_key
    client.transcript_id = transcript_id
    client.endpoint = "wss://api.fireflies.ai/realtime"
    client._status = FirefliesConnectionStatus.DISCONNECTED if MODELS_AVAILABLE else "disconnected"
    client.is_connected = False
    return client


@pytest.fixture
def mock_fireflies_client(api_key, mock_graphql_client):
    """Create mock unified Fireflies client."""
    client = AsyncMock()
    client.api_key = api_key
    client._graphql = mock_graphql_client
    client._realtime_clients = {}
    return client


# =============================================================================
# WebSocket Message Fixtures
# =============================================================================


@pytest.fixture
def websocket_auth_success():
    """WebSocket auth success message."""
    return {"type": "auth.success"}


@pytest.fixture
def websocket_auth_failed():
    """WebSocket auth failed message."""
    return {"type": "auth.failed", "message": "Invalid API key"}


@pytest.fixture
def websocket_connection_established():
    """WebSocket connection established message."""
    return {"type": "connection.established"}


@pytest.fixture
def websocket_connection_error():
    """WebSocket connection error message."""
    return {"type": "connection.error", "message": "Connection refused"}


@pytest.fixture
def websocket_transcript_broadcast(transcript_id):
    """WebSocket transcript broadcast message."""
    return {
        "type": "transcription.broadcast",
        "data": {
            "transcript_id": transcript_id,
            "chunk_id": "chunk_001",
            "text": "Hello, this is a test.",
            "speaker_name": "Alice",
            "start_time": 0.0,
            "end_time": 2.0,
        },
    }


# =============================================================================
# GraphQL Response Fixtures
# =============================================================================


@pytest.fixture
def graphql_active_meetings_response(sample_meeting):
    """GraphQL active meetings response."""
    if not MODELS_AVAILABLE:
        pytest.skip("Fireflies models not available")

    return {
        "data": {
            "active_meetings": [
                {
                    "id": sample_meeting.id,
                    "title": sample_meeting.title,
                    "organizer_email": sample_meeting.organizer_email,
                    "meeting_link": sample_meeting.meeting_link,
                    "start_time": sample_meeting.start_time.isoformat() + "Z"
                    if sample_meeting.start_time
                    else None,
                    "end_time": None,
                    "privacy": "private",
                    "state": sample_meeting.state.value,
                }
            ]
        }
    }


@pytest.fixture
def graphql_empty_meetings_response():
    """GraphQL empty meetings response."""
    return {"data": {"active_meetings": []}}


@pytest.fixture
def graphql_error_response():
    """GraphQL error response."""
    return {"errors": [{"message": "Invalid API key"}]}


# =============================================================================
# Utility Functions
# =============================================================================


def create_chunk_sequence(
    transcript_id: str,
    texts: list[str],
    speakers: list[str] | None = None,
    start_time: float = 0.0,
    word_duration: float = 0.3,
) -> list:
    """Create a sequence of chunks for testing."""
    if not MODELS_AVAILABLE:
        return []

    chunks = []
    current_time = start_time

    for i, text in enumerate(texts):
        speaker = speakers[i % len(speakers)] if speakers else "Speaker"
        duration = len(text.split()) * word_duration

        chunk = FirefliesChunk(
            transcript_id=transcript_id,
            chunk_id=f"chunk_{i + 1:04d}",
            text=text,
            speaker_name=speaker,
            start_time=current_time,
            end_time=current_time + duration,
        )
        chunks.append(chunk)
        current_time += duration

    return chunks


@pytest.fixture
def chunk_factory(transcript_id):
    """Factory for creating chunks."""

    def _create_chunk(
        text: str,
        speaker_name: str = "Speaker",
        chunk_id: str | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
    ):
        if not MODELS_AVAILABLE:
            pytest.skip("Fireflies models not available")

        if chunk_id is None:
            import uuid

            chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"

        if end_time is None:
            end_time = start_time + len(text.split()) * 0.3

        return FirefliesChunk(
            transcript_id=transcript_id,
            chunk_id=chunk_id,
            text=text,
            speaker_name=speaker_name,
            start_time=start_time,
            end_time=end_time,
        )

    return _create_chunk


# =============================================================================
# Async PostgreSQL Database Fixtures (shared across intelligence tests)
# =============================================================================


def _get_async_database_url() -> str:
    """Build asyncpg database URL from environment."""
    sync_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://livetranslate:livetranslate_dev_password@localhost:5433/livetranslate_test",  # pragma: allowlist secret
    )
    # Convert postgresql:// to postgresql+asyncpg://
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if sync_url.startswith("postgresql+asyncpg://"):
        return sync_url
    return f"postgresql+asyncpg://{sync_url.split('://', 1)[-1]}"


@pytest.fixture
async def async_db_engine():
    """Create an async PostgreSQL engine for real DB tests."""
    url = _get_async_database_url()
    engine = create_async_engine(url, echo=False, pool_pre_ping=True)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session_factory(async_db_engine):
    """Create tables and return an async session factory backed by real PostgreSQL.

    Uses the real PostgreSQL database (not SQLite) so JSONB, UUID, and all
    PostgreSQL-specific column types work correctly.

    Tables are created at the start and dropped at the end to ensure isolation.
    """
    import database.models  # noqa: F401 — registers models with Base.metadata
    from database.base import Base

    async with async_db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        async_db_engine, class_=AsyncSession, expire_on_commit=False
    )

    @asynccontextmanager
    async def _factory():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    yield _factory

    async with async_db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def bot_session_id(db_session_factory):
    """Create a real BotSession in PostgreSQL and return its UUID string."""
    from database.models import BotSession

    sid = uuid.uuid4()
    async with db_session_factory() as session:
        bot_session = BotSession(
            session_id=sid,
            bot_id="test-bot-001",
            meeting_id="test-meeting-001",
            meeting_title="Intelligence Test Meeting",
            bot_type="google_meet",
            status="running",
        )
        session.add(bot_session)
        await session.commit()

    return str(sid)


@pytest.fixture
async def intelligence_service(db_session_factory):
    """Create a real MeetingIntelligenceService backed by real PostgreSQL."""
    from config import MeetingIntelligenceSettings
    from services.meeting_intelligence import MeetingIntelligenceService

    return MeetingIntelligenceService(
        db_session_factory=db_session_factory,
        translation_client=None,
        settings=MeetingIntelligenceSettings(),
    )
