#!/usr/bin/env python3
"""
Global test configuration and fixtures for comprehensive audio flow testing.

This module provides shared fixtures, configuration, and utilities for all
audio processing tests across the test suite.

IMPORTANT: All tests should use REAL services and databases, not mocks.
The environment variables below configure the real PostgreSQL database.
"""

import os
import sys
from contextlib import asynccontextmanager

# =============================================================================
# ENVIRONMENT SETUP - Must be done BEFORE any other imports
# Placeholder URL prevents import failures; overridden by postgres_container fixture.
# =============================================================================

if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = (
        "postgresql://livetranslate:livetranslate_dev_password@localhost:5433/livetranslate_test"
    )

if "TEST_DATABASE_URL" not in os.environ:
    os.environ["TEST_DATABASE_URL"] = os.environ["DATABASE_URL"]

# PostgreSQL individual settings for compatibility
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5433")
os.environ.setdefault("POSTGRES_DB", "livetranslate_test")
os.environ.setdefault("POSTGRES_USER", "livetranslate")
os.environ.setdefault("POSTGRES_PASSWORD", "livetranslate_dev_password")
os.environ.setdefault("DB_USER", "livetranslate")
os.environ.setdefault("DB_PASSWORD", "livetranslate_dev_password")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5433")
os.environ.setdefault("DB_NAME", "livetranslate_test")

# Redis (for caching tests)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")

# =============================================================================
# Standard imports
# =============================================================================

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add the src directory to Python path
current_dir = Path(__file__).parent
service_root = current_dir.parent
src_path = service_root / "src"

if src_path.exists():
    sys.path.insert(0, str(src_path))


# Create module aliases to prevent duplicate loading
# Production code uses 'src.X' imports, tests use 'X' imports
# We alias 'X' -> 'src.X' so both paths resolve to the same module
def _setup_module_aliases():
    """Setup module aliases so 'database' imports resolve to 'src.database'."""
    import importlib

    # List of top-level modules that need aliasing
    module_names = [
        "database",
        "models",
        "clients",
        "services",
        "routers",
        "managers",
        "bot",
        "utils",
        "audio",
        "pipeline",
        "middleware",
        "internal_services",
        "dependencies",
        "config",
    ]

    for name in module_names:
        try:
            # Import with src. prefix (production path)
            src_mod = importlib.import_module(f"src.{name}")
            # Alias the non-prefixed name to the src. version
            sys.modules[name] = src_mod
        except ImportError:
            pass


_setup_module_aliases()

# Import application components
if os.getenv("SKIP_MAIN_FASTAPI_IMPORT") == "1":
    logging.info("Skipping main_fastapi import due to SKIP_MAIN_FASTAPI_IMPORT=1")
    app = None
else:
    try:
        from dependencies import (
            get_audio_coordinator,
            get_audio_service_client,
            get_config_manager,
            get_config_sync_manager,
            get_health_monitor,
            get_translation_service_client,
        )
        from main_fastapi import app
    except ImportError as e:
        logging.warning(f"Could not import application components: {e}")
        app = None

# Import test data management
try:
    from fixtures.audio_test_data import AudioTestCase, AudioTestDataManager
except ImportError as e:
    logging.warning(f"Could not import audio test data: {e}")
    AudioTestDataManager = None
    AudioTestCase = None

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Testcontainers + SQLAlchemy async imports
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.postgres import PostgresContainer

# Test configuration constants
TEST_TIMEOUT = 30  # seconds
MAX_CONCURRENT_TESTS = 10
TEST_AUDIO_CACHE_SIZE = 100  # MB


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()

    yield loop

    # Clean up
    try:
        if not loop.is_closed():
            loop.close()
    except Exception as e:
        logger.warning(f"Error closing event loop: {e}")


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="audio_test_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "sample_rate": 16000,
        "test_duration": 3.0,
        "chunk_size": 2048,
        "supported_formats": ["wav", "mp3", "webm", "ogg", "mp4", "flac"],
        "timeout": TEST_TIMEOUT,
        "max_concurrent": MAX_CONCURRENT_TESTS,
        "whisper_models": ["whisper-base", "whisper-small", "whisper-medium"],
        "target_languages": ["en", "es", "fr", "de", "it"],
        "quality_levels": ["low", "medium", "high"],
    }


@pytest.fixture(scope="session")
def audio_test_manager():
    """Provide audio test data manager."""
    if AudioTestDataManager is None:
        pytest.skip("AudioTestDataManager not available")

    manager = AudioTestDataManager()
    yield manager
    manager.cleanup_cache()


@pytest.fixture
def mock_config_manager():
    """Provide a mock configuration manager."""
    mock = AsyncMock()

    # Default configuration responses
    mock.get_service_config.return_value = {
        "whisper_model": "whisper-base",
        "device": "auto",
        "enable_vad": True,
        "enable_speaker_diarization": True,
        "noise_reduction": {"enabled": False, "strength": 0.5},
        "speech_enhancement": {"enabled": True},
        "target_languages": ["en", "es"],
    }

    mock.update_service_config.return_value = True
    mock.get_all_configs.return_value = {
        "audio": {"whisper_model": "whisper-base"},
        "translation": {"quality": "balanced"},
    }

    return mock


@pytest.fixture
def mock_audio_service_client():
    """Provide a mock audio service client."""
    mock = AsyncMock()

    # Default successful response
    default_response = Mock()
    default_response.status_code = 200
    default_response.json.return_value = {
        "text": "This is a test transcription",
        "speaker_id": "speaker_0",
        "confidence": 0.95,
        "language": "en",
        "duration": 3.0,
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "This is a test transcription",
                "speaker_id": "speaker_0",
                "confidence": 0.95,
            }
        ],
        "processing_time": 1.2,
        "model_used": "whisper-base",
        "device_used": "cpu",
    }

    mock.post.return_value = default_response
    mock.get.return_value = default_response

    return mock


@pytest.fixture
def mock_translation_service_client():
    """Provide a mock translation service client."""
    mock = AsyncMock()

    # Default successful response
    default_response = Mock()
    default_response.status_code = 200
    default_response.json.return_value = {
        "translated_text": "Esta es una transcripción de prueba",
        "source_language": "en",
        "target_language": "es",
        "confidence": 0.88,
        "quality_score": 0.92,
        "processing_time": 0.8,
        "model_used": "translation-model",
    }

    mock.post.return_value = default_response
    mock.get.return_value = default_response

    return mock


@pytest.fixture
def mock_audio_coordinator():
    """Provide a mock audio coordinator."""
    mock = AsyncMock()

    # Session management
    mock.create_session.return_value = True
    mock.end_session.return_value = True
    mock.add_audio_data.return_value = True

    # Status and analytics
    mock.get_session_status.return_value = {
        "session_id": "test_session",
        "active": True,
        "chunks_processed": 1,
        "total_duration": 3.0,
        "created_at": "2024-01-01T00:00:00Z",
    }

    mock.get_session_analytics.return_value = {
        "chunks_processed": 1,
        "total_duration": 3.0,
        "average_processing_time": 1.5,
        "success_rate": 1.0,
    }

    mock.get_performance_metrics.return_value = {
        "total_chunks_processed": 1,
        "average_processing_time": 1.5,
        "average_chunk_duration": 3.0,
        "total_audio_duration": 3.0,
        "processing_throughput": 2.0,
    }

    # Configuration
    mock.get_processing_config.return_value = AsyncMock()
    mock.update_processing_config.return_value = True

    # Mock properties
    mock.active_sessions = {}
    mock.session_processors = {}
    mock.max_concurrent_sessions = 10
    mock.is_initialized = True

    return mock


@pytest.fixture
def mock_config_sync_manager():
    """Provide a mock configuration sync manager."""
    mock = AsyncMock()

    mock.get_current_config.return_value = {
        "whisper_model": "whisper-base",
        "enable_vad": True,
        "enable_speaker_diarization": True,
    }

    mock.update_processing_config.return_value = True
    mock.sync_configuration.return_value = True

    return mock


@pytest.fixture
def mock_health_monitor():
    """Provide a mock health monitor."""
    mock = AsyncMock()

    mock.get_service_health.return_value = {
        "status": "healthy",
        "services": {
            "whisper": {"status": "healthy", "response_time": 0.1},
            "translation": {"status": "healthy", "response_time": 0.2},
        },
    }

    mock.is_service_healthy.return_value = True

    return mock


@pytest.fixture
def mock_dependencies():
    """Provide all mock dependencies as a dictionary."""
    return {
        "config_manager": mock_config_manager(),
        "audio_client": mock_audio_service_client(),
        "translation_client": mock_translation_service_client(),
        "audio_coordinator": mock_audio_coordinator(),
        "config_sync_manager": mock_config_sync_manager(),
        "health_monitor": mock_health_monitor(),
    }


@pytest.fixture
def test_client_with_mocks(mock_dependencies):
    """Create FastAPI test client with all dependencies mocked."""
    if app is None:
        pytest.skip("FastAPI app not available")

    # Override dependencies
    app.dependency_overrides[get_config_manager] = lambda: mock_dependencies["config_manager"]
    app.dependency_overrides[get_audio_service_client] = lambda: mock_dependencies["audio_client"]
    app.dependency_overrides[get_translation_service_client] = lambda: mock_dependencies[
        "translation_client"
    ]
    app.dependency_overrides[get_audio_coordinator] = lambda: mock_dependencies["audio_coordinator"]
    app.dependency_overrides[get_config_sync_manager] = lambda: mock_dependencies[
        "config_sync_manager"
    ]
    app.dependency_overrides[get_health_monitor] = lambda: mock_dependencies["health_monitor"]

    client = TestClient(app)

    yield client, mock_dependencies

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    sample_rate = 16000
    duration = 3.0

    # Generate voice-like audio
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Fundamental frequency with harmonics
    signal = 0.3 * np.sin(2 * np.pi * 120 * t)  # Fundamental
    signal += 0.2 * np.sin(2 * np.pi * 240 * t)  # Second harmonic
    signal += 0.1 * np.sin(2 * np.pi * 360 * t)  # Third harmonic

    # Add formants
    signal += 0.1 * np.sin(2 * np.pi * 800 * t)  # First formant
    signal += 0.05 * np.sin(2 * np.pi * 1200 * t)  # Second formant

    # Add noise
    signal += 0.01 * np.random.randn(len(signal))

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    return {
        "voice_like": signal.astype(np.float32),
        "sample_rate": sample_rate,
        "duration": duration,
    }


@pytest.fixture
def error_simulation():
    """Provide utilities for simulating various error conditions."""

    class ErrorSimulator:
        @staticmethod
        def simulate_service_timeout():
            return httpx.TimeoutException("Service timeout")

        @staticmethod
        def simulate_service_unavailable():
            return httpx.ConnectError("Service unavailable")

        @staticmethod
        def simulate_invalid_response():
            response = Mock()
            response.status_code = 500
            response.text = "Internal Server Error"
            return response

        @staticmethod
        def simulate_network_error():
            return httpx.NetworkError("Network error")

        @staticmethod
        def simulate_rate_limit():
            response = Mock()
            response.status_code = 429
            response.text = "Rate limit exceeded"
            return response

    return ErrorSimulator()


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities."""

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.metrics = {}

        def start(self):
            self.start_time = time.time()
            return self

        def stop(self):
            if self.start_time:
                duration = time.time() - self.start_time
                return duration
            return 0.0

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = self.stop()
            self.metrics["duration"] = duration

    return PerformanceMonitor()


@pytest.fixture
def test_session_id():
    """Generate a unique test session ID."""
    import uuid

    return f"test_session_{uuid.uuid4().hex[:8]}"


# Markers for test categorization
pytestmark = [
    pytest.mark.asyncio,
]


# Hooks for test execution
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "audio_pipeline: marks tests as audio pipeline tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)

        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add markers based on test names
        if "audio_flow" in item.name:
            item.add_marker(pytest.mark.audio_pipeline)

        if "error" in item.name or "failure" in item.name:
            item.add_marker(pytest.mark.error)

        if "format" in item.name:
            item.add_marker(pytest.mark.format)

        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Log test start
    logger.info(f"Starting test: {item.name}")


def pytest_runtest_teardown(item, nextitem):
    """Teardown for each test run."""
    # Log test completion
    logger.info(f"Completed test: {item.name}")


@pytest.fixture(autouse=True)
def test_timeout():
    """Automatically apply timeout to all tests."""
    pytest.timeout = TEST_TIMEOUT


@pytest.fixture(scope="function", autouse=True)
def reset_dependency_singletons():
    """
    Reset all dependency singletons before and after each test.

    This prevents asyncio event loop binding issues where singletons
    (EventPublisher, RedisClient, DatabaseManager, etc.) created in one
    test's event loop cause "Event loop is closed" errors in subsequent tests.

    The issue occurs because:
    1. Singletons are cached via @lru_cache()
    2. Each async test function gets a fresh event loop
    3. Redis clients and asyncio.Lock() objects bind to the loop they were created in
    4. Using them from a different loop fails with "Event loop is closed"

    Solution: Reset all singletons before each test so they get recreated
    with the current test's event loop.
    """
    # Reset before test
    try:
        from src.dependencies import reset_dependencies

        reset_dependencies()
    except ImportError:
        try:
            from dependencies import reset_dependencies

            reset_dependencies()
        except ImportError:
            pass  # Skip if dependencies not available

    yield

    # Reset after test (cleanup)
    try:
        from src.dependencies import reset_dependencies

        reset_dependencies()
    except ImportError:
        try:
            from dependencies import reset_dependencies

            reset_dependencies()
        except ImportError:
            pass


# =============================================================================
# FASTAPI TEST CLIENT FIXTURES - With proper dependency initialization
# =============================================================================


@pytest.fixture(scope="function")
def initialized_app(database_url, run_migrations):
    """
    Create FastAPI app with dependencies initialized.

    The TestClient does NOT run the lifespan context manager by default.
    This fixture manually calls startup_dependencies() to initialize
    all singletons (DatabaseManager, etc.) before tests run.

    IMPORTANT: There are TWO database manager systems in this codebase:
    1. src/dependencies.py - get_database_manager() with @lru_cache
    2. src/database/database.py - global database_manager requiring initialize_database()

    Both must be initialized for the app to work properly.

    This ensures:
    - Database manager is initialized
    - All service clients are ready
    - Validation errors return 422 (not 500 from middleware)
    - Not found errors return 404 properly
    """
    if app is None:
        pytest.skip("FastAPI app not available")

    # Import startup/shutdown functions
    try:
        from src.dependencies import shutdown_dependencies, startup_dependencies
    except ImportError:
        from dependencies import shutdown_dependencies, startup_dependencies

    # Import database initialization (for the global database_manager in database/database.py)
    try:
        from src.config import DatabaseSettings
        from src.database.database import initialize_database
    except ImportError:
        from config import DatabaseSettings
        from database.database import initialize_database

    # Run startup in a new event loop
    loop = asyncio.new_event_loop()
    try:
        # Initialize the global database_manager in database/database.py
        # This is required by get_db_session() which routers use
        db_settings = DatabaseSettings(url=database_url)
        initialize_database(db_settings)
        logger.info("Database module initialized")

        # Initialize all other dependencies (config manager, service clients, etc.)
        loop.run_until_complete(startup_dependencies())
        logger.info("Dependencies initialized for test")
    except Exception as e:
        logger.warning(f"Failed to initialize dependencies: {e}")
        import traceback

        traceback.print_exc()
        # Continue anyway - some tests may still work

    yield app

    # Cleanup: shutdown dependencies
    try:
        loop.run_until_complete(shutdown_dependencies())
    except Exception as e:
        logger.warning(f"Error during dependency shutdown: {e}")
    finally:
        loop.close()


@pytest.fixture(scope="function")
def client(initialized_app):
    """
    Create TestClient with properly initialized dependencies.

    This is the main fixture for integration tests. It ensures:
    - All dependencies (DatabaseManager, etc.) are initialized
    - FastAPI validation works correctly (422 for validation errors)
    - HTTP errors return proper status codes (404 for not found)

    Usage:
        def test_something(client):
            response = client.get("/api/endpoint")
            assert response.status_code == 200
    """
    with TestClient(initialized_app) as test_client:
        yield test_client


# Session-level fixtures for resource management
@pytest.fixture(scope="session", autouse=True)
def session_setup_teardown():
    """Session-level setup and teardown."""
    logger.info("Starting comprehensive audio test session")

    yield

    logger.info("Completed comprehensive audio test session")


# Utility functions for tests
def assert_audio_quality(audio_data: np.ndarray, min_rms: float = 0.01, max_rms: float = 1.0):
    """Assert audio quality is within expected range."""
    assert not np.isnan(audio_data).any(), "Audio contains NaN values"
    assert not np.isinf(audio_data).any(), "Audio contains infinite values"
    assert len(audio_data) > 0, "Audio data is empty"

    rms = np.sqrt(np.mean(audio_data**2))
    assert min_rms <= rms <= max_rms, f"RMS level {rms} out of range [{min_rms}, {max_rms}]"


def assert_response_structure(response_data: dict[str, Any], required_fields: list[str]):
    """Assert response has required structure."""
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"


def assert_processing_time(actual_time: float, max_time: float, audio_duration: float):
    """Assert processing time is reasonable."""
    assert actual_time > 0, "Processing time must be positive"
    assert actual_time <= max_time, f"Processing time {actual_time}s exceeds maximum {max_time}s"

    real_time_factor = actual_time / audio_duration
    assert real_time_factor <= 5.0, f"Real-time factor {real_time_factor} too high"


# =============================================================================
# REAL DATABASE FIXTURES - Testcontainers + Alembic + Savepoint isolation
# =============================================================================


@pytest.fixture(scope="session")
def postgres_container():
    """Start PostgreSQL testcontainer for entire test session."""
    with PostgresContainer(
        image="postgres:16-alpine",
        username="livetranslate",
        password="test_password",  # pragma: allowlist secret
        dbname="livetranslate_test",
    ) as pg:
        # Enable extensions required by migrations
        from sqlalchemy import create_engine, text

        url = pg.get_connection_url()
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            conn.commit()
        engine.dispose()
        yield pg


@pytest.fixture(scope="session")
def database_url(postgres_container):
    """Get URL from testcontainer and set env vars globally.

    Normalises the URL to plain ``postgresql://`` so that downstream code
    (Alembic env.py, DatabaseSettings, etc.) can convert it to the async
    driver (``postgresql+asyncpg://``) as expected.

    Also sets individual POSTGRES_*/DB_* env vars for code that reads
    them directly (e.g. test_data_pipeline_integration.py).
    """
    url = postgres_container.get_connection_url()
    # testcontainers returns postgresql+psycopg2:// — strip the driver suffix
    url = url.replace("postgresql+psycopg2://", "postgresql://", 1)
    os.environ["DATABASE_URL"] = url
    os.environ["TEST_DATABASE_URL"] = url

    # Parse URL and set individual env vars for direct-connect code
    from urllib.parse import urlparse

    parsed = urlparse(url)
    os.environ["POSTGRES_HOST"] = parsed.hostname or "localhost"
    os.environ["POSTGRES_PORT"] = str(parsed.port or 5432)
    os.environ["POSTGRES_DB"] = (parsed.path or "/livetranslate_test").lstrip("/")
    os.environ["POSTGRES_USER"] = parsed.username or "livetranslate"
    os.environ["POSTGRES_PASSWORD"] = parsed.password or ""
    os.environ["DB_HOST"] = parsed.hostname or "localhost"
    os.environ["DB_PORT"] = str(parsed.port or 5432)
    os.environ["DB_NAME"] = (parsed.path or "/livetranslate_test").lstrip("/")
    os.environ["DB_USER"] = parsed.username or "livetranslate"
    os.environ["DB_PASSWORD"] = parsed.password or ""

    return url


@pytest.fixture(scope="session")
def run_migrations(database_url):
    """Run Alembic migrations once per session (sync — no event loop needed)."""
    from alembic import command
    from alembic.config import Config

    alembic_ini = Path(__file__).parent.parent / "alembic.ini"
    cfg = Config(str(alembic_ini))
    async_url = database_url
    if database_url.startswith("postgresql://"):
        async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    cfg.set_main_option("sqlalchemy.url", async_url)
    command.upgrade(cfg, "head")


@pytest.fixture(scope="session")
def bot_sessions_schema(database_url, run_migrations):
    """Create the bot_sessions PostgreSQL schema and tables.

    The BotSessionDatabaseManager and data_pipeline module use raw asyncpg
    queries against a ``bot_sessions`` schema (separate from the Alembic-managed
    ``public`` schema).  This fixture executes the base SQL schema script,
    then adds enhanced tables (speaker_identities, segment continuity columns)
    that ``database-init-complete.sql`` provides but with corrected ordering.

    Uses raw psycopg2 because the scripts contain multi-statement SQL with
    ``$$`` delimited function bodies that SQLAlchemy ``text()`` cannot handle.
    """
    import psycopg2

    scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
    base_sql_file = scripts_dir / "bot-sessions-schema.sql"

    if not base_sql_file.exists():
        pytest.skip("bot-sessions-schema.sql not found — skipping bot_sessions tests")

    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        # 1. Run base schema (sessions, audio_files, transcripts, translations,
        #    correlations, participants, events, session_statistics + indexes/views/functions)
        # Strip GRANT lines that reference the 'postgres' role (doesn't exist in testcontainer)
        base_sql = "\n".join(
            line
            for line in base_sql_file.read_text().splitlines()
            if not line.strip().upper().startswith("GRANT ")
        )
        cur.execute(base_sql)

        # 2. Add segment continuity columns to transcripts (from database-init-complete.sql)
        cur.execute("""
            ALTER TABLE bot_sessions.transcripts
                ADD COLUMN IF NOT EXISTS previous_segment_id VARCHAR(100),
                ADD COLUMN IF NOT EXISTS next_segment_id VARCHAR(100),
                ADD COLUMN IF NOT EXISTS is_segment_boundary BOOLEAN DEFAULT FALSE;

            CREATE INDEX IF NOT EXISTS idx_transcripts_previous_segment
                ON bot_sessions.transcripts (previous_segment_id);
            CREATE INDEX IF NOT EXISTS idx_transcripts_next_segment
                ON bot_sessions.transcripts (next_segment_id);
        """)

        # 3. Add speaker_identities table (participants table already exists from step 1)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_sessions.speaker_identities (
                identity_id VARCHAR(100) PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL
                    REFERENCES bot_sessions.sessions (session_id) ON DELETE CASCADE,
                speaker_label VARCHAR(100) NOT NULL,
                participant_id VARCHAR(100)
                    REFERENCES bot_sessions.participants (participant_id),
                identified_name VARCHAR(255),
                identification_method VARCHAR(50) NOT NULL
                    DEFAULT 'manual',
                identification_confidence REAL DEFAULT 0.0,
                identification_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                UNIQUE (session_id, speaker_label)
            );

            CREATE INDEX IF NOT EXISTS idx_speaker_identities_session
                ON bot_sessions.speaker_identities (session_id);
            CREATE INDEX IF NOT EXISTS idx_speaker_identities_label
                ON bot_sessions.speaker_identities (speaker_label);
        """)

        # 4. Add search_vector column + trigger for full-text search
        cur.execute("""
            ALTER TABLE bot_sessions.transcripts
                ADD COLUMN IF NOT EXISTS search_vector TSVECTOR;

            CREATE OR REPLACE FUNCTION bot_sessions.update_transcript_search_vector()
            RETURNS TRIGGER AS $fn$
            BEGIN
                NEW.search_vector := to_tsvector('english', COALESCE(NEW.transcript_text, ''));
                RETURN NEW;
            END;
            $fn$ LANGUAGE plpgsql;

            DROP TRIGGER IF EXISTS trg_transcript_search ON bot_sessions.transcripts;
            CREATE TRIGGER trg_transcript_search
                BEFORE INSERT OR UPDATE OF transcript_text ON bot_sessions.transcripts
                FOR EACH ROW EXECUTE FUNCTION bot_sessions.update_transcript_search_vector();

            ALTER TABLE bot_sessions.translations
                ADD COLUMN IF NOT EXISTS search_vector TSVECTOR;
        """)

        # 5. Replace speaker_statistics view to include identification columns
        cur.execute("""
            CREATE OR REPLACE VIEW bot_sessions.speaker_statistics AS
            SELECT
                t.session_id,
                t.speaker_id,
                COALESCE(si.identified_name, t.speaker_name) AS speaker_name,
                COALESCE(si.identification_method, 'unknown') AS identification_method,
                COALESCE(si.identification_confidence, 0.0) AS identification_confidence,
                COUNT(*) AS transcript_segments,
                SUM(t.end_timestamp - t.start_timestamp) AS total_speaking_time,
                AVG(t.confidence_score) AS avg_confidence,
                COUNT(DISTINCT tr.target_language) AS languages_translated_to,
                COUNT(DISTINCT tr.translation_id) AS total_translations
            FROM bot_sessions.transcripts AS t
            LEFT JOIN bot_sessions.translations AS tr
                ON t.transcript_id = tr.source_transcript_id
            LEFT JOIN bot_sessions.speaker_identities AS si
                ON t.session_id = si.session_id AND t.speaker_id = si.speaker_label
            WHERE t.speaker_id IS NOT NULL
            GROUP BY t.session_id, t.speaker_id, t.speaker_name,
                     si.identified_name, si.identification_method,
                     si.identification_confidence;
        """)
    conn.close()


@pytest.fixture(scope="session")
def async_db_engine(database_url, run_migrations):
    """Session-scoped async engine (created synchronously, no I/O at init).

    Uses NullPool to avoid stale-connection issues when pytest-asyncio
    creates a fresh event loop per test function.
    """
    from sqlalchemy.pool import NullPool

    async_url = database_url
    if database_url.startswith("postgresql://"):
        async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(async_url, poolclass=NullPool)
    yield engine
    # Testcontainer destruction handles cleanup


@pytest.fixture
async def db_session_factory(async_db_engine):
    """Function-scoped async session factory with table-truncation cleanup.

    Each session gets its own connection from the engine, supporting
    concurrent operations (e.g. asyncio.gather) where the savepoint
    pattern would cause conflicts. Committed data is immediately
    visible across sessions within the same test.

    At teardown, all user tables are truncated (preserving
    alembic_version) for clean per-test isolation.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

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

    # Truncate all user tables for test isolation
    async with async_db_engine.connect() as conn:
        await conn.execute(
            text(
                "DO $$ DECLARE r RECORD; BEGIN "
                "FOR r IN (SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' "
                "AND tablename != 'alembic_version') LOOP "
                "EXECUTE 'TRUNCATE TABLE public.' || quote_ident(r.tablename) || ' CASCADE'; "
                "END LOOP; END $$;"
            )
        )
        await conn.commit()


@pytest.fixture
async def db_session(async_db_engine):
    """Function-scoped single async session with table-truncation cleanup."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    session_factory = async_sessionmaker(
        async_db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with session_factory() as session:
        yield session

    # Truncate all user tables for test isolation
    async with async_db_engine.connect() as conn:
        await conn.execute(
            text(
                "DO $$ DECLARE r RECORD; BEGIN "
                "FOR r IN (SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' "
                "AND tablename != 'alembic_version') LOOP "
                "EXECUTE 'TRUNCATE TABLE public.' || quote_ident(r.tablename) || ' CASCADE'; "
                "END LOOP; END $$;"
            )
        )
        await conn.commit()


@pytest.fixture(scope="session")
def db_engine(database_url, run_migrations):
    """Sync engine for legacy tests that need psycopg2."""
    from sqlalchemy import create_engine

    engine = create_engine(database_url, echo=False, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def verify_database_connection(database_url, run_migrations):
    """Verify database is accessible at session start using SQLAlchemy."""
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(database_url, pool_pre_ping=True)

        with engine.connect() as conn:
            # Get PostgreSQL version
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"PostgreSQL: {version[:50]}...")

            # Count tables
            result = conn.execute(
                text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            """)
            )
            table_count = result.scalar()
            logger.info(f"Database tables available: {table_count}")

        engine.dispose()
        return True
    except Exception as e:
        logger.warning(f"Database verification failed: {e}")
        return False


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(verify_database_connection, database_url):
    """Setup the test environment at the start of the session."""
    logger.info("=" * 60)
    logger.info("TEST ENVIRONMENT SETUP")
    logger.info("=" * 60)
    logger.info(f"DATABASE_URL: {database_url.split('@')[1]}")  # Hide credentials
    logger.info(f"REDIS_URL: {os.environ.get('REDIS_URL', 'not set')}")
    logger.info(f"Database connected: {verify_database_connection}")
    logger.info("=" * 60)

    yield

    logger.info("Test session complete")
