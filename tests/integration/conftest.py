"""
Shared Test Fixtures for LiveTranslate Integration Tests
Provides common fixtures for database, Redis, and service mocking
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from redis import Redis
import fakeredis
import numpy as np

# Add project root and orchestration-service/src to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
ORCHESTRATION_SRC = PROJECT_ROOT / "modules" / "orchestration-service" / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ORCHESTRATION_SRC))

# Test configuration
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://test_user:test_pass@localhost:5432/livetranslate_test"
)
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")
USE_FAKE_REDIS = os.getenv("USE_FAKE_REDIS", "true").lower() == "true"


# ============================================
# Event Loop Fixture
# ============================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================
# Database Fixtures
# ============================================

@pytest.fixture(scope="session")
def test_db_engine():
    """Create test database engine (session-scoped)"""
    engine = create_engine(
        TEST_DATABASE_URL,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10
    )
    yield engine
    engine.dispose()


@pytest.fixture
async def postgres_fixture(test_db_engine) -> AsyncGenerator[Session, None]:
    """
    PostgreSQL test database with schema
    Creates all tables before test, drops after
    """
    # Import models to ensure they're registered
    from database.models import Base
    # Also import chat_models to register those tables
    import database.chat_models

    # Clean up any existing tables first (from failed tests)
    Base.metadata.drop_all(test_db_engine, checkfirst=True)

    # Create all tables
    Base.metadata.create_all(test_db_engine, checkfirst=True)

    # Create session
    SessionLocal = sessionmaker(bind=test_db_engine, expire_on_commit=False)
    session = SessionLocal()

    yield session

    # Cleanup
    session.close()
    Base.metadata.drop_all(test_db_engine, checkfirst=True)


@pytest.fixture
async def db_session(postgres_fixture) -> Session:
    """Alias for postgres_fixture for convenience"""
    return postgres_fixture


# ============================================
# Redis Fixtures
# ============================================

@pytest.fixture
async def redis_fixture() -> AsyncGenerator[Redis, None]:
    """
    Redis test cache
    Uses fakeredis in-memory if USE_FAKE_REDIS=true
    """
    if USE_FAKE_REDIS:
        # Use in-memory fake Redis for faster tests
        redis_client = fakeredis.FakeRedis(decode_responses=True)
    else:
        # Use real Redis
        redis_client = Redis.from_url(TEST_REDIS_URL, decode_responses=True)

    yield redis_client

    # Cleanup
    redis_client.flushdb()
    redis_client.close()


# ============================================
# Service Fixtures (Mocked)
# ============================================

@pytest.fixture
async def whisper_service_fixture():
    """Mocked Whisper service for testing"""
    mock_service = AsyncMock()

    # Default transcription response
    mock_service.transcribe.return_value = {
        "text": "Test transcription from Whisper",
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Test transcription from Whisper"
            }
        ],
        "language": "en",
        "confidence": 0.95,
        "speaker_id": "SPEAKER_00",
        "speaker_name": "Test Speaker"
    }

    # Health check
    mock_service.health_check.return_value = {"status": "healthy", "device": "cpu"}

    yield mock_service


@pytest.fixture
async def translation_service_fixture():
    """Mocked Translation service for testing"""
    mock_service = AsyncMock()

    # Default translation response
    mock_service.translate.return_value = {
        "translated_text": "Test translation output",
        "source_language": "en",
        "target_language": "es",
        "quality_score": 0.92,
        "confidence": 0.90
    }

    # Health check
    mock_service.health_check.return_value = {"status": "healthy", "backend": "mock"}

    yield mock_service


@pytest.fixture
async def orchestration_service_fixture(
    postgres_fixture,
    redis_fixture,
    whisper_service_fixture,
    translation_service_fixture
):
    """
    Orchestration service with test dependencies
    Provides TestClient for API testing
    """
    # Note: This will be implemented when we have the actual orchestration service
    # For now, returning a mock
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    # Create minimal test app
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    client = TestClient(app)

    yield client

    # Cleanup handled by postgres/redis fixtures


# ============================================
# Audio Test Data Generators
# ============================================

@pytest.fixture
def generate_test_audio():
    """Generate test audio data for testing"""

    def _generate(
        duration: float = 3.0,
        sample_rate: int = 16000,
        frequency: float = 440.0,
        noise_level: float = 0.0
    ) -> np.ndarray:
        """
        Generate synthetic audio for testing

        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            frequency: Tone frequency in Hz
            noise_level: Amount of noise to add (0.0-1.0)

        Returns:
            NumPy array of audio samples
        """
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, False)

        # Generate sine wave
        audio = np.sin(2 * np.pi * frequency * t)

        # Add noise if requested
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, num_samples)
            audio += noise

        # Normalize to [-1, 1]
        audio = audio / np.max(np.abs(audio))

        return audio.astype(np.float32)

    return _generate


@pytest.fixture
def generate_test_audio_chunks():
    """Generate chunked audio data for streaming tests"""

    def _generate_chunks(
        duration: float = 10.0,
        chunk_size: float = 2.0,
        sample_rate: int = 16000
    ) -> list:
        """
        Generate audio chunks for streaming tests

        Args:
            duration: Total duration in seconds
            chunk_size: Size of each chunk in seconds
            sample_rate: Sample rate in Hz

        Returns:
            List of audio chunks
        """
        chunks = []
        num_chunks = int(duration / chunk_size)

        for i in range(num_chunks):
            # Generate chunk with varying frequency
            frequency = 440.0 + (i * 50)  # Vary frequency per chunk
            chunk_duration = chunk_size
            num_samples = int(chunk_duration * sample_rate)
            t = np.linspace(0, chunk_duration, num_samples, False)
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            chunks.append(audio)

        return chunks

    return _generate_chunks


# ============================================
# Test Utilities
# ============================================

@pytest.fixture
def calculate_wer():
    """Calculate Word Error Rate for transcription quality testing"""

    def _calculate(hypothesis: str, reference: str) -> float:
        """
        Calculate Word Error Rate (WER)

        Args:
            hypothesis: Predicted transcription
            reference: Ground truth transcription

        Returns:
            WER as a float (0.0 = perfect, 1.0 = completely wrong)
        """
        # Tokenize
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        # Calculate Levenshtein distance at word level
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
        return float(wer)

    return _calculate


@pytest.fixture
def assert_latency():
    """Helper to assert latency requirements"""

    def _assert(latency_ms: float, max_latency_ms: float, context: str = ""):
        """
        Assert that latency meets requirements

        Args:
            latency_ms: Measured latency in milliseconds
            max_latency_ms: Maximum acceptable latency
            context: Context string for error message
        """
        assert latency_ms <= max_latency_ms, (
            f"{context} Latency {latency_ms}ms exceeds maximum {max_latency_ms}ms"
        )

    return _assert


# ============================================
# Baseline Metrics Storage
# ============================================

@pytest.fixture(scope="session")
def baseline_metrics():
    """
    Store baseline metrics for comparison
    Dict persists across test session
    """
    metrics = {}

    def _record(metric_name: str, value: float):
        """Record a baseline metric"""
        metrics[metric_name] = value

    def _get(metric_name: str) -> float:
        """Get a baseline metric"""
        return metrics.get(metric_name, 0.0)

    # Return both functions
    return {"record": _record, "get": _get, "data": metrics}


# ============================================
# Cleanup Fixtures
# ============================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Auto-cleanup after each test"""
    yield
    # Cleanup code runs here after test
    # (Most cleanup handled by specific fixtures)
    pass


# ============================================
# Test Markers Setup
# ============================================

def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: Tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_npu: Tests requiring Intel NPU"
    )
    config.addinivalue_line(
        "markers", "requires_db: Tests requiring database"
    )
    config.addinivalue_line(
        "markers", "requires_redis: Tests requiring Redis"
    )
    config.addinivalue_line(
        "markers", "feature_preservation: Regression tests for existing features"
    )
