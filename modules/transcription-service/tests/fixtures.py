#!/usr/bin/env python3
"""
Test Fixtures

Pytest fixtures for WebSocket server testing.
"""

import asyncio
import logging
import os
import tempfile

import pytest

from . import TEST_CONFIG
from .utils import (
    HTTPTestClient,
    PerformanceMonitor,
    RedisTestHelper,
    TestAudioGenerator,
    WebSocketTestClient,
    generate_test_token,
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_config():
    """Test configuration fixture"""
    return TEST_CONFIG.copy()


@pytest.fixture
def event_loop():
    """Create an event loop for each test"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def websocket_client(test_config):
    """WebSocket test client fixture"""
    client = WebSocketTestClient(test_config["websocket_url"], test_config["test_timeout"])
    yield client
    await client.disconnect()


@pytest.fixture
async def http_client(test_config):
    """HTTP test client fixture"""
    async with HTTPTestClient(test_config["api_base_url"], test_config["test_timeout"]) as client:
        yield client


@pytest.fixture
async def authenticated_websocket_client(test_config):
    """Authenticated WebSocket test client fixture"""
    client = WebSocketTestClient(test_config["websocket_url"], test_config["test_timeout"])
    auth_token = generate_test_token("test_user")
    connected = await client.connect(auth_token)
    assert connected, "Failed to connect authenticated WebSocket client"
    yield client
    await client.disconnect()


@pytest.fixture
def audio_generator():
    """Audio generator fixture"""
    return TestAudioGenerator()


@pytest.fixture
def test_audio_sine_wave(audio_generator):
    """Generate sine wave test audio"""
    return audio_generator.generate_sine_wave(duration=2.0, frequency=440)


@pytest.fixture
def test_audio_noise(audio_generator):
    """Generate noise test audio"""
    return audio_generator.generate_noise(duration=1.0)


@pytest.fixture
def test_audio_silence(audio_generator):
    """Generate silence test audio"""
    return audio_generator.generate_silence(duration=1.0)


@pytest.fixture
def test_audio_file(test_audio_sine_wave, audio_generator):
    """Create temporary test audio file"""
    filepath = audio_generator.save_test_audio(test_audio_sine_wave, "test_audio.wav")
    yield filepath
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture"""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    yield monitor
    monitor.stop_monitoring()


@pytest.fixture
def redis_helper(test_config):
    """Redis test helper fixture"""
    helper = RedisTestHelper(test_config["redis_url"])
    helper.connect()
    # Clear any existing test data
    helper.clear_test_data()
    yield helper
    # Cleanup test data
    helper.clear_test_data()
    helper.disconnect()


@pytest.fixture
async def multiple_websocket_clients(test_config):
    """Multiple WebSocket clients for concurrent testing"""
    clients = []
    num_clients = min(10, test_config["max_concurrent_connections"] // 10)

    for _i in range(num_clients):
        client = WebSocketTestClient(test_config["websocket_url"], test_config["test_timeout"])
        connected = await client.connect()
        if connected:
            clients.append(client)

    yield clients

    # Cleanup
    for client in clients:
        await client.disconnect()


@pytest.fixture
def sample_websocket_messages():
    """Sample WebSocket messages for testing"""
    return {
        "auth": {"type": "authenticate", "token": "test_token_12345"},
        "start_audio": {
            "type": "start_audio",
            "session_id": "test_session_123",
            "audio_format": "wav",
            "sample_rate": 16000,
            "channels": 1,
        },
        "audio_data": {
            "type": "audio_data",
            "data": "deadbeef",  # hex encoded audio data
            "chunk_id": 0,
            "timestamp": 1234567890,
        },
        "end_audio": {"type": "end_audio", "session_id": "test_session_123"},
        "heartbeat": {"type": "heartbeat", "timestamp": 1234567890},
        "subscribe": {"type": "subscribe", "room": "test_room"},
        "unsubscribe": {"type": "unsubscribe", "room": "test_room"},
        "invalid_message": {"invalid": "message"},
    }


@pytest.fixture
def sample_api_endpoints():
    """Sample API endpoints for testing"""
    return {
        "health": "/health",
        "models": "/models",
        "transcribe": "/transcribe",
        "upload": "/upload",
        "sessions": "/sessions",
        "connections": "/connections",
        "performance": "/performance",
        "heartbeat": "/heartbeat",
        "reconnection": "/reconnection",
    }


@pytest.fixture
def expected_response_schemas():
    """Expected response schemas for validation"""
    return {
        "health": {
            "type": "object",
            "required": ["status", "timestamp"],
            "properties": {
                "status": {"type": "string"},
                "timestamp": {"type": "number"},
                "version": {"type": "string"},
                "uptime": {"type": "number"},
            },
        },
        "models": {
            "type": "object",
            "required": ["models"],
            "properties": {
                "models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "size": {"type": "string"},
                        },
                    },
                }
            },
        },
        "performance": {
            "type": "object",
            "required": ["connection_stats", "performance_stats"],
            "properties": {
                "connection_stats": {"type": "object"},
                "performance_stats": {"type": "object"},
                "queue_stats": {"type": "object"},
            },
        },
    }


@pytest.fixture
def stress_test_config(test_config):
    """Configuration for stress testing"""
    return {
        "duration": test_config["stress_test_duration"],
        "max_connections": test_config["max_concurrent_connections"],
        "messages_per_connection": 100,
        "audio_chunk_size": 1024,
        "ramp_up_time": 10,  # seconds
        "cool_down_time": 5,  # seconds
    }


@pytest.fixture
def error_scenarios():
    """Error scenarios for testing"""
    return {
        "invalid_auth": {
            "description": "Invalid authentication token",
            "token": "invalid_token_123",
            "expected_error": "authentication_failed",
        },
        "malformed_message": {
            "description": "Malformed JSON message",
            "message": '{"type": "invalid", "data":',  # Invalid JSON
            "expected_error": "message_parse_error",
        },
        "unsupported_audio_format": {
            "description": "Unsupported audio format",
            "message": {
                "type": "start_audio",
                "audio_format": "mp3",  # Unsupported format
                "sample_rate": 16000,
            },
            "expected_error": "unsupported_format",
        },
        "session_not_found": {
            "description": "Session not found",
            "message": {"type": "end_audio", "session_id": "nonexistent_session"},
            "expected_error": "session_not_found",
        },
        "rate_limit_exceeded": {
            "description": "Rate limit exceeded",
            "rapid_messages": 1000,  # Send 1000 messages rapidly
            "expected_error": "rate_limit_exceeded",
        },
    }


@pytest.fixture
def reconnection_scenarios():
    """Reconnection scenarios for testing"""
    return {
        "normal_reconnection": {
            "description": "Normal reconnection after disconnect",
            "disconnect_after": 5,  # seconds
            "reconnect_delay": 2,  # seconds
            "should_restore_session": True,
        },
        "session_expired": {
            "description": "Reconnection with expired session",
            "disconnect_after": 5,
            "reconnect_delay": 35 * 60,  # 35 minutes (session timeout is 30 min)
            "should_restore_session": False,
        },
        "rapid_reconnection": {
            "description": "Rapid reconnection attempts",
            "disconnect_reconnect_cycles": 10,
            "cycle_interval": 1,  # seconds
            "should_handle_gracefully": True,
        },
    }


@pytest.fixture
def benchmark_thresholds():
    """Performance benchmark thresholds"""
    return {
        "connection_time": 1.0,  # seconds
        "message_response_time": 0.1,  # seconds
        "audio_processing_time": 2.0,  # seconds
        "transcription_time": 5.0,  # seconds
        "memory_usage": 500,  # MB
        "cpu_usage": 80,  # percentage
        "concurrent_connections": 100,  # number of connections
        "throughput": 1000,  # messages per second
        "error_rate": 0.01,  # 1% error rate
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment (runs once per test session)"""
    logger.info("Setting up test environment...")

    # Create temporary directories for test files
    test_temp_dir = tempfile.mkdtemp(prefix="livetranslate_tests_")

    yield {"temp_dir": test_temp_dir}

    logger.info("Cleaning up test environment...")
    # Cleanup would go here if needed


@pytest.fixture(autouse=True)
async def test_isolation():
    """Ensure test isolation (runs before each test)"""
    # Clear any shared state before each test
    yield
    # Cleanup after each test
    await asyncio.sleep(0.1)  # Small delay to allow cleanup


# Parameterized fixtures for testing different scenarios


@pytest.fixture(params=["wav", "raw", "flac"])
def audio_format(request):
    """Different audio formats for testing"""
    return request.param


@pytest.fixture(params=[16000, 22050, 44100, 48000])
def sample_rate(request):
    """Different sample rates for testing"""
    return request.param


@pytest.fixture(params=[1, 2])
def audio_channels(request):
    """Different channel configurations for testing"""
    return request.param


@pytest.fixture(params=[256, 512, 1024, 2048, 4096])
def audio_chunk_size(request):
    """Different chunk sizes for testing"""
    return request.param


@pytest.fixture(params=[0.5, 1.0, 2.0, 5.0, 10.0])
def audio_duration(request):
    """Different audio durations for testing"""
    return request.param
