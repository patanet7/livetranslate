#!/usr/bin/env python3
"""
Fireflies Client Tests - ACTUAL TESTS (No Mocks)

Tests the real client classes, message parsing, and state management.
"""

import sys
from pathlib import Path
import json
import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))

from clients.fireflies_client import (
    FirefliesGraphQLClient,
    FirefliesRealtimeClient,
    FirefliesClient,
    FirefliesAPIError,
    FirefliesConnectionError,
    FirefliesAuthError,
    DEFAULT_GRAPHQL_ENDPOINT,
    DEFAULT_WEBSOCKET_ENDPOINT,
    DEFAULT_WEBSOCKET_PATH,
    MAX_RECONNECTION_ATTEMPTS,
    ACTIVE_MEETINGS_QUERY,
)

from models.fireflies import (
    FirefliesConnectionStatus,
)


class TestClientConstants:
    """Test client module constants."""

    def test_default_endpoints(self):
        """Test default endpoint URLs."""
        assert DEFAULT_GRAPHQL_ENDPOINT == "https://api.fireflies.ai/graphql"
        # Socket.IO endpoint (without path - path is separate)
        assert DEFAULT_WEBSOCKET_ENDPOINT == "wss://api.fireflies.ai"
        # Socket.IO path
        assert DEFAULT_WEBSOCKET_PATH == "/ws/realtime"

    def test_reconnection_settings(self):
        """Test reconnection constants."""
        assert MAX_RECONNECTION_ATTEMPTS == 5

    def test_graphql_query_structure(self):
        """Test GraphQL query contains expected fields."""
        assert "active_meetings" in ACTIVE_MEETINGS_QUERY
        assert "id" in ACTIVE_MEETINGS_QUERY
        assert "title" in ACTIVE_MEETINGS_QUERY
        assert "organizer_email" in ACTIVE_MEETINGS_QUERY
        assert "state" in ACTIVE_MEETINGS_QUERY


class TestFirefliesGraphQLClient:
    """Test GraphQL client initialization and configuration."""

    def test_client_initialization(self):
        """Test client initializes with correct values."""
        client = FirefliesGraphQLClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.endpoint == DEFAULT_GRAPHQL_ENDPOINT
        assert client._session is None

    def test_custom_endpoint(self):
        """Test client with custom endpoint."""
        client = FirefliesGraphQLClient(
            api_key="key", endpoint="https://custom.api/graphql"
        )
        assert client.endpoint == "https://custom.api/graphql"

    def test_custom_timeout(self):
        """Test client with custom timeout."""
        client = FirefliesGraphQLClient(api_key="key", timeout=60.0)
        assert client.timeout.total == 60.0


class TestFirefliesRealtimeClient:
    """Test WebSocket client initialization and message handling."""

    def test_client_initialization(self):
        """Test realtime client initializes correctly."""
        client = FirefliesRealtimeClient(
            api_key="test-key",
            transcript_id="transcript-123",
        )

        assert client.api_key == "test-key"
        assert client.transcript_id == "transcript-123"
        assert client.endpoint == DEFAULT_WEBSOCKET_ENDPOINT
        assert client.auto_reconnect is True
        assert client.max_reconnect_attempts == MAX_RECONNECTION_ATTEMPTS

    def test_initial_state(self):
        """Test initial connection state."""
        client = FirefliesRealtimeClient(
            api_key="key",
            transcript_id="t1",
        )

        assert client.status == FirefliesConnectionStatus.DISCONNECTED
        assert client.is_connected is False
        assert client._running is False

    def test_custom_settings(self):
        """Test custom client settings."""
        client = FirefliesRealtimeClient(
            api_key="key",
            transcript_id="t1",
            endpoint="wss://custom.api/ws",
            auto_reconnect=False,
            max_reconnect_attempts=10,
        )

        assert client.endpoint == "wss://custom.api/ws"
        assert client.auto_reconnect is False
        assert client.max_reconnect_attempts == 10

    @pytest.mark.asyncio
    async def test_status_transitions(self):
        """Test status transitions via _set_status method."""
        client = FirefliesRealtimeClient(api_key="key", transcript_id="t1")

        # Test initial state
        assert client._status == FirefliesConnectionStatus.DISCONNECTED

        # Test status change
        await client._set_status(FirefliesConnectionStatus.CONNECTED, "Authenticated")
        assert client._status == FirefliesConnectionStatus.CONNECTED

        # Test error status
        await client._set_status(FirefliesConnectionStatus.ERROR, "Auth failed")
        assert client._status == FirefliesConnectionStatus.ERROR

    @pytest.mark.asyncio
    async def test_status_callback_invoked(self):
        """Test that status callback is invoked on status change."""
        status_changes = []

        async def on_status(status, message):
            status_changes.append((status, message))

        client = FirefliesRealtimeClient(
            api_key="key",
            transcript_id="t1",
            on_status_change=on_status,
        )

        await client._set_status(FirefliesConnectionStatus.CONNECTING, "Starting")
        await client._set_status(FirefliesConnectionStatus.CONNECTED, "Ready")

        assert len(status_changes) == 2
        assert status_changes[0][0] == FirefliesConnectionStatus.CONNECTING
        assert status_changes[1][0] == FirefliesConnectionStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_handle_transcript_via_direct_method(self):
        """Test handling transcript data via _handle_transcript method."""
        received_chunks = []

        async def on_transcript(chunk):
            received_chunks.append(chunk)

        client = FirefliesRealtimeClient(
            api_key="key",
            transcript_id="t1",
            on_transcript=on_transcript,
        )

        # Call _handle_transcript directly (simulating Socket.IO event)
        await client._handle_transcript(
            {
                "transcript_id": "t1",
                "chunk_id": "c1",
                "text": "Hello world",
                "speaker_name": "Alice",
                "start_time": 0.0,
                "end_time": 1.5,
            }
        )

        assert len(received_chunks) == 1
        assert received_chunks[0].text == "Hello world"
        assert received_chunks[0].speaker_name == "Alice"

    @pytest.mark.asyncio
    async def test_handle_invalid_transcript_data(self):
        """Test handling invalid transcript data gracefully."""
        client = FirefliesRealtimeClient(api_key="key", transcript_id="t1")

        # Should not raise with invalid data types
        await client._handle_transcript("not a dict")
        await client._handle_transcript(None)
        await client._handle_transcript(123)

    @pytest.mark.asyncio
    async def test_handle_transcript_creates_chunk(self):
        """Test _handle_transcript creates proper FirefliesChunk."""
        chunks = []

        async def capture(chunk):
            chunks.append(chunk)

        client = FirefliesRealtimeClient(
            api_key="key",
            transcript_id="default-t",
            on_transcript=capture,
        )

        await client._handle_transcript(
            {
                "data": {
                    "chunk_id": "c1",
                    "text": "Test message",
                    "speaker_name": "Bob",
                    "start_time": 5.0,
                    "end_time": 7.5,
                }
            }
        )

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.transcript_id == "default-t"  # Uses client's default
        assert chunk.chunk_id == "c1"
        assert chunk.text == "Test message"
        assert chunk.start_time == 5.0
        assert chunk.end_time == 7.5


class TestFirefliesClient:
    """Test unified Fireflies client."""

    def test_client_initialization(self):
        """Test unified client initializes correctly."""
        client = FirefliesClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.graphql_endpoint == DEFAULT_GRAPHQL_ENDPOINT
        assert client.websocket_endpoint == DEFAULT_WEBSOCKET_ENDPOINT
        assert client._graphql is not None
        assert len(client._realtime_clients) == 0

    def test_custom_endpoints(self):
        """Test client with custom endpoints."""
        client = FirefliesClient(
            api_key="key",
            graphql_endpoint="https://custom/graphql",
            websocket_endpoint="wss://custom/ws",
        )

        assert client.graphql_endpoint == "https://custom/graphql"
        assert client.websocket_endpoint == "wss://custom/ws"

    def test_get_realtime_status_none(self):
        """Test status for nonexistent transcript."""
        client = FirefliesClient(api_key="key")
        assert client.get_realtime_status("nonexistent") is None

    def test_get_active_connections_empty(self):
        """Test empty connections list."""
        client = FirefliesClient(api_key="key")
        assert client.get_active_connections() == {}


class TestExceptions:
    """Test exception classes."""

    def test_api_error(self):
        """Test FirefliesAPIError."""
        error = FirefliesAPIError("API failed", status_code=401)
        assert str(error) == "API failed"
        assert error.status_code == 401

    def test_api_error_no_status(self):
        """Test FirefliesAPIError without status code."""
        error = FirefliesAPIError("Generic error")
        assert error.status_code is None

    def test_connection_error(self):
        """Test FirefliesConnectionError."""
        error = FirefliesConnectionError("Connection failed")
        assert str(error) == "Connection failed"

    def test_auth_error(self):
        """Test FirefliesAuthError."""
        error = FirefliesAuthError("Invalid key")
        assert str(error) == "Invalid key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
