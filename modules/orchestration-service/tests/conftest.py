#!/usr/bin/env python3
"""
Global test configuration and fixtures for comprehensive audio flow testing.

This module provides shared fixtures, configuration, and utilities for all
audio processing tests across the test suite.
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, Mock
import json
import os
import sys

import pytest
import pytest_asyncio
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

# Add the src directory to Python path
current_dir = Path(__file__).parent
service_root = current_dir.parent
src_path = service_root / "src"

if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import application components
try:
    from main_fastapi import app
    from dependencies import (
        get_config_manager,
        get_audio_service_client,
        get_translation_service_client,
        get_audio_coordinator,
        get_config_sync_manager,
        get_health_monitor,
    )
except ImportError as e:
    logging.warning(f"Could not import application components: {e}")
    app = None

# Import test data management
try:
    from fixtures.audio_test_data import AudioTestDataManager, AudioTestCase
except ImportError as e:
    logging.warning(f"Could not import audio test data: {e}")
    AudioTestDataManager = None
    AudioTestCase = None

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
                "confidence": 0.95
            }
        ],
        "processing_time": 1.2,
        "model_used": "whisper-base",
        "device_used": "cpu"
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
        }
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
    app.dependency_overrides[get_translation_service_client] = lambda: mock_dependencies["translation_client"]
    app.dependency_overrides[get_audio_coordinator] = lambda: mock_dependencies["audio_coordinator"]
    app.dependency_overrides[get_config_sync_manager] = lambda: mock_dependencies["config_sync_manager"]
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
    signal += 0.1 * np.sin(2 * np.pi * 800 * t)   # First formant
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
    config.addinivalue_line(
        "markers", "audio_pipeline: marks tests as audio pipeline tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


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
    
    rms = np.sqrt(np.mean(audio_data ** 2))
    assert min_rms <= rms <= max_rms, f"RMS level {rms} out of range [{min_rms}, {max_rms}]"


def assert_response_structure(response_data: Dict[str, Any], required_fields: List[str]):
    """Assert response has required structure."""
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"


def assert_processing_time(actual_time: float, max_time: float, audio_duration: float):
    """Assert processing time is reasonable."""
    assert actual_time > 0, "Processing time must be positive"
    assert actual_time <= max_time, f"Processing time {actual_time}s exceeds maximum {max_time}s"
    
    real_time_factor = actual_time / audio_duration
    assert real_time_factor <= 5.0, f"Real-time factor {real_time_factor} too high"