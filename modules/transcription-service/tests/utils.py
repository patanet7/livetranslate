#!/usr/bin/env python3
"""
Test Utilities

Common utilities and helpers for WebSocket server testing.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from typing import Any

import aiohttp
import numpy as np
import redis
import soundfile as sf
import websockets

logger = logging.getLogger(__name__)


class TestAudioGenerator:
    """Generate test audio data for testing purposes"""

    @staticmethod
    def generate_sine_wave(
        duration: float = 1.0, frequency: int = 440, sample_rate: int = 16000
    ) -> np.ndarray:
        """Generate a sine wave for testing"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.5
        return audio.astype(np.float32)

    @staticmethod
    def generate_noise(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate white noise for testing"""
        samples = int(sample_rate * duration)
        audio = np.random.uniform(-0.5, 0.5, samples)
        return audio.astype(np.float32)

    @staticmethod
    def generate_silence(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate silence for testing"""
        samples = int(sample_rate * duration)
        return np.zeros(samples, dtype=np.float32)

    @classmethod
    def save_test_audio(
        cls, audio_data: np.ndarray, filename: str, sample_rate: int = 16000
    ) -> str:
        """Save audio data to a temporary file"""
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        sf.write(filepath, audio_data, sample_rate)
        return filepath


class WebSocketTestClient:
    """Test client for WebSocket connections"""

    def __init__(self, url: str, timeout: int = 30):
        self.url = url
        self.timeout = timeout
        self.websocket = None
        self.received_messages = []
        self.connection_stats = {
            "connected_at": None,
            "disconnected_at": None,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": [],
        }

    async def connect(self, auth_token: str | None = None) -> bool:
        """Connect to WebSocket server"""
        try:
            headers = {}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"

            self.websocket = await websockets.connect(
                self.url, extra_headers=headers, timeout=self.timeout
            )
            self.connection_stats["connected_at"] = time.time()
            logger.info(f"Connected to WebSocket: {self.url}")
            return True
        except Exception as e:
            self.connection_stats["errors"].append(str(e))
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.connection_stats["disconnected_at"] = time.time()
            self.websocket = None
            logger.info("Disconnected from WebSocket")

    async def send_message(self, message: dict[str, Any]) -> bool:
        """Send a message to the WebSocket server"""
        if not self.websocket:
            return False

        try:
            await self.websocket.send(json.dumps(message))
            self.connection_stats["messages_sent"] += 1
            return True
        except Exception as e:
            self.connection_stats["errors"].append(str(e))
            logger.error(f"Failed to send message: {e}")
            return False

    async def receive_message(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Receive a message from the WebSocket server"""
        if not self.websocket:
            return None

        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=timeout or self.timeout)
            parsed_message = json.loads(message)
            self.received_messages.append(parsed_message)
            self.connection_stats["messages_received"] += 1
            return parsed_message
        except TimeoutError:
            logger.warning("Timeout waiting for message")
            return None
        except Exception as e:
            self.connection_stats["errors"].append(str(e))
            logger.error(f"Failed to receive message: {e}")
            return None

    async def send_audio_data(self, audio_data: bytes, chunk_size: int = 1024) -> bool:
        """Send audio data in chunks"""
        if not self.websocket:
            return False

        try:
            # Send start audio message
            start_message = {
                "type": "start_audio",
                "session_id": str(uuid.uuid4()),
                "audio_format": "wav",
                "sample_rate": 16000,
            }
            await self.send_message(start_message)

            # Send audio chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                chunk_message = {
                    "type": "audio_data",
                    "data": chunk.hex(),
                    "chunk_id": i // chunk_size,
                }
                await self.send_message(chunk_message)

            # Send end audio message
            end_message = {"type": "end_audio"}
            await self.send_message(end_message)

            return True
        except Exception as e:
            self.connection_stats["errors"].append(str(e))
            logger.error(f"Failed to send audio data: {e}")
            return False

    def get_connection_duration(self) -> float | None:
        """Get connection duration in seconds"""
        if self.connection_stats["connected_at"]:
            end_time = self.connection_stats["disconnected_at"] or time.time()
            return end_time - self.connection_stats["connected_at"]
        return None


class HTTPTestClient:
    """Test client for HTTP API endpoints"""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get(self, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        """Make GET request"""
        url = f"{self.base_url}{endpoint}"
        return await self.session.get(url, **kwargs)

    async def post(self, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        """Make POST request"""
        url = f"{self.base_url}{endpoint}"
        return await self.session.post(url, **kwargs)

    async def put(self, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        """Make PUT request"""
        url = f"{self.base_url}{endpoint}"
        return await self.session.put(url, **kwargs)

    async def delete(self, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        """Make DELETE request"""
        url = f"{self.base_url}{endpoint}"
        return await self.session.delete(url, **kwargs)


class PerformanceMonitor:
    """Monitor performance metrics during tests"""

    def __init__(self):
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "response_times": [],
            "throughput": 0,
            "error_count": 0,
            "success_count": 0,
            "memory_usage": [],
            "cpu_usage": [],
        }

    def start_monitoring(self):
        """Start performance monitoring"""
        self.metrics["start_time"] = time.time()

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.metrics["end_time"] = time.time()

    def record_response_time(self, response_time: float):
        """Record a response time"""
        self.metrics["response_times"].append(response_time)

    def record_success(self):
        """Record a successful operation"""
        self.metrics["success_count"] += 1

    def record_error(self):
        """Record an error"""
        self.metrics["error_count"] += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get performance statistics"""
        response_times = self.metrics["response_times"]
        duration = self.get_duration()

        stats = {
            "duration": duration,
            "total_operations": len(response_times),
            "success_count": self.metrics["success_count"],
            "error_count": self.metrics["error_count"],
            "success_rate": self.metrics["success_count"]
            / max(1, self.metrics["success_count"] + self.metrics["error_count"]),
            "throughput": len(response_times) / duration if duration > 0 else 0,
        }

        if response_times:
            stats.update(
                {
                    "avg_response_time": sum(response_times) / len(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "p95_response_time": np.percentile(response_times, 95),
                    "p99_response_time": np.percentile(response_times, 99),
                }
            )

        return stats

    def get_duration(self) -> float:
        """Get monitoring duration"""
        if self.metrics["start_time"] and self.metrics["end_time"]:
            return self.metrics["end_time"] - self.metrics["start_time"]
        return 0


class RedisTestHelper:
    """Helper for Redis-related testing"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = None

    def connect(self):
        """Connect to Redis"""
        self.client = redis.from_url(self.redis_url)

    def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            self.client.close()

    def clear_test_data(self, pattern: str = "test:*"):
        """Clear test data from Redis"""
        if self.client:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)

    def get_connection_count(self) -> int:
        """Get number of Redis connections"""
        if self.client:
            info = self.client.info()
            return info.get("connected_clients", 0)
        return 0


def assert_response_time(response_time: float, max_time: float, operation: str = "operation"):
    """Assert that response time is within acceptable limits"""
    assert (
        response_time <= max_time
    ), f"{operation} took {response_time:.3f}s, expected <= {max_time}s"


def assert_websocket_message(
    message: dict[str, Any], expected_type: str, required_fields: list[str] | None = None
):
    """Assert WebSocket message format"""
    assert "type" in message, "Message missing 'type' field"
    assert (
        message["type"] == expected_type
    ), f"Expected message type '{expected_type}', got '{message['type']}'"

    if required_fields:
        for field in required_fields:
            assert field in message, f"Message missing required field '{field}'"


def generate_test_token(user_id: str = "test_user") -> str:
    """Generate a test authentication token"""
    # Simple test token - in real implementation would use proper JWT
    return f"test_token_{user_id}_{int(time.time())}"


async def wait_for_condition(
    condition: Callable, timeout: float = 10, interval: float = 0.1
) -> bool:
    """Wait for a condition to become true"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition():
            return True
        await asyncio.sleep(interval)
    return False
