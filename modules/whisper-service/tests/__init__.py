#!/usr/bin/env python3
"""
WebSocket Server Testing Suite

Comprehensive test suite for the LiveTranslate WebSocket server implementation.
Includes unit tests, integration tests, stress tests, and performance benchmarks.
"""

__version__ = "1.0.0"
__author__ = "LiveTranslate Team"

# Test configuration
TEST_CONFIG = {
    "websocket_url": "ws://localhost:5001",
    "api_base_url": "http://localhost:5001",
    "test_timeout": 30,
    "stress_test_duration": 60,
    "max_concurrent_connections": 100,
    "test_audio_file": "test_audio.wav",
    "redis_url": "redis://localhost:6379",
}

# Test utilities - explicit imports to avoid F403/F405 errors
from .fixtures import (
    audio_channels,
    audio_chunk_size,
    audio_duration,
    audio_format,
    audio_generator,
    authenticated_websocket_client,
    benchmark_thresholds,
    error_scenarios,
    event_loop,
    expected_response_schemas,
    http_client,
    multiple_websocket_clients,
    performance_monitor,
    reconnection_scenarios,
    redis_helper,
    sample_api_endpoints,
    sample_rate,
    sample_websocket_messages,
    setup_test_environment,
    stress_test_config,
    test_audio_file,
    test_audio_noise,
    test_audio_silence,
    test_audio_sine_wave,
    test_config,
    test_isolation,
    websocket_client,
)
from .utils import (
    HTTPTestClient,
    PerformanceMonitor,
    RedisTestHelper,
    TestAudioGenerator,
    WebSocketTestClient,
    assert_response_time,
    assert_websocket_message,
    generate_test_token,
    wait_for_condition,
)
