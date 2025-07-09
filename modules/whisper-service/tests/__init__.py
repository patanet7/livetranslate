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

# Test utilities
from .utils import *
from .fixtures import * 