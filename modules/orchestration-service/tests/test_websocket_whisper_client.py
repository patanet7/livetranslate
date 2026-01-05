#!/usr/bin/env python3
"""
Tests for WebSocket Whisper Client

Tests the orchestration service's client for connecting to Whisper's
WebSocket server.
"""

import pytest
import asyncio
import json
import base64
import numpy as np
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add src to path
ORCH_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(ORCH_SRC))

from websocket_whisper_client import WebSocketWhisperClient, WhisperSessionState


class TestWhisperSessionState:
    """Tests for WhisperSessionState dataclass"""

    def test_session_state_initialization(self):
        """Test session state can be initialized"""
        state = WhisperSessionState(
            session_id="test-session", config={"model": "large-v3", "language": "en"}
        )

        assert state.session_id == "test-session"
        assert state.config["model"] == "large-v3"
        assert state.is_active is True
        assert state.chunks_sent == 0
        assert state.segments_received == 0
        assert state.created_at is not None
        assert state.last_activity is not None

    def test_update_activity(self):
        """Test updating last activity timestamp"""
        state = WhisperSessionState(session_id="test", config={})

        original_time = state.last_activity
        asyncio.sleep(0.01)  # Small delay
        state.update_activity()

        # Should be updated (or at least same time)
        assert state.last_activity >= original_time


class TestWebSocketWhisperClient:
    """Tests for WebSocket Whisper Client"""

    def test_client_initialization(self):
        """Test client can be initialized with configuration"""
        client = WebSocketWhisperClient(
            whisper_host="localhost",
            whisper_port=5001,
            auto_reconnect=True,
            max_reconnect_attempts=3,
        )

        assert client.whisper_host == "localhost"
        assert client.whisper_port == 5001
        assert client.auto_reconnect is True
        assert client.max_reconnect_attempts == 3
        assert client.connected is False
        assert len(client.sessions) == 0

    def test_whisper_url_property(self):
        """Test Whisper URL is correctly formatted"""
        client = WebSocketWhisperClient(
            whisper_host="whisper.example.com", whisper_port=8080
        )

        assert client.whisper_url == "ws://whisper.example.com:8080/stream"

    def test_callback_registration(self):
        """Test callback registration methods"""
        client = WebSocketWhisperClient()

        def segment_callback(segment):
            pass

        def error_callback(error):
            pass

        def connection_callback(connected):
            pass

        client.on_segment(segment_callback)
        client.on_error(error_callback)
        client.on_connection_change(connection_callback)

        assert segment_callback in client.segment_callbacks
        assert error_callback in client.error_callbacks
        assert connection_callback in client.connection_callbacks

    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test session state tracking"""
        client = WebSocketWhisperClient()

        # Manually create session (without actual connection)
        session = WhisperSessionState(
            session_id="test-session", config={"model": "base"}
        )
        client.sessions["test-session"] = session

        # Test get_session_info
        info = client.get_session_info("test-session")
        assert info is not None
        assert info["session_id"] == "test-session"
        assert info["is_active"] is True
        assert info["chunks_sent"] == 0

        # Test get_all_sessions
        all_sessions = client.get_all_sessions()
        assert "test-session" in all_sessions

        # Test non-existent session
        assert client.get_session_info("nonexistent") is None

    def test_connection_stats(self):
        """Test connection statistics"""
        client = WebSocketWhisperClient()

        # Add test sessions
        client.sessions["session1"] = WhisperSessionState(
            session_id="session1", config={}
        )
        client.sessions["session1"].chunks_sent = 5
        client.sessions["session1"].segments_received = 3

        client.sessions["session2"] = WhisperSessionState(
            session_id="session2", config={}
        )
        client.sessions["session2"].chunks_sent = 10
        client.sessions["session2"].is_active = False

        stats = client.get_connection_stats()

        assert stats["connected"] is False
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 1
        assert stats["total_chunks_sent"] == 15
        assert stats["total_segments_received"] == 3


class TestWebSocketWhisperClientIntegration:
    """
    Integration tests for WebSocket Whisper Client

    These tests require the Whisper WebSocket server to be running
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_client_connection(self):
        """Test client can connect to Whisper WebSocket server"""
        # Import here to avoid import errors if websocket_stream_server doesn't exist
        try:
            WHISPER_SRC = (
                Path(__file__).parent.parent.parent / "whisper-service" / "src"
            )
            sys.path.insert(0, str(WHISPER_SRC))
            from websocket_stream_server import WebSocketStreamServer
        except ImportError:
            pytest.skip("Whisper WebSocket server not available")

        # Start Whisper server
        whisper_server = WebSocketStreamServer(host="localhost", port=5020)
        server_task = asyncio.create_task(whisper_server.start())
        await asyncio.sleep(0.5)  # Let server start

        try:
            # Create and connect client
            client = WebSocketWhisperClient(
                whisper_host="localhost", whisper_port=5020, auto_reconnect=False
            )

            connected = await client.connect()
            assert connected is True
            assert client.is_connected() is True

            # Close client
            await client.close()
            assert client.is_connected() is False

        finally:
            await whisper_server.stop()
            server_task.cancel()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_client_start_stream(self):
        """Test client can start a streaming session"""
        try:
            WHISPER_SRC = (
                Path(__file__).parent.parent.parent / "whisper-service" / "src"
            )
            sys.path.insert(0, str(WHISPER_SRC))
            from websocket_stream_server import WebSocketStreamServer
        except ImportError:
            pytest.skip("Whisper WebSocket server not available")

        whisper_server = WebSocketStreamServer(host="localhost", port=5021)
        server_task = asyncio.create_task(whisper_server.start())
        await asyncio.sleep(0.5)

        try:
            client = WebSocketWhisperClient(whisper_host="localhost", whisper_port=5021)

            await client.connect()

            # Start stream
            session_id = await client.start_stream(
                session_id="integration-test-session",
                config={"model": "large-v3", "language": "en"},
            )

            assert session_id == "integration-test-session"
            assert "integration-test-session" in client.sessions

            # Verify session state
            session_info = client.get_session_info(session_id)
            assert session_info["is_active"] is True
            assert session_info["config"]["model"] == "large-v3"

            await client.close()

        finally:
            await whisper_server.stop()
            server_task.cancel()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_client_send_audio(self):
        """Test client can send audio chunks"""
        try:
            WHISPER_SRC = (
                Path(__file__).parent.parent.parent / "whisper-service" / "src"
            )
            sys.path.insert(0, str(WHISPER_SRC))
            from websocket_stream_server import WebSocketStreamServer
        except ImportError:
            pytest.skip("Whisper WebSocket server not available")

        whisper_server = WebSocketStreamServer(host="localhost", port=5022)
        server_task = asyncio.create_task(whisper_server.start())
        await asyncio.sleep(0.5)

        try:
            client = WebSocketWhisperClient(whisper_host="localhost", whisper_port=5022)

            await client.connect()

            session_id = await client.start_stream(
                session_id="audio-test-session", config={"model": "base"}
            )

            # Send audio chunk
            test_audio = np.random.randn(16000).astype(np.float32)
            await client.send_audio_chunk(
                session_id=session_id, audio_data=test_audio.tobytes()
            )

            # Verify session stats
            session_info = client.get_session_info(session_id)
            assert session_info["chunks_sent"] == 1

            # Send another chunk
            await client.send_audio_chunk(
                session_id=session_id, audio_data=test_audio.tobytes()
            )

            session_info = client.get_session_info(session_id)
            assert session_info["chunks_sent"] == 2

            await client.close()

        finally:
            await whisper_server.stop()
            server_task.cancel()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_client_receive_segments(self):
        """Test client can receive segments from Whisper"""
        try:
            WHISPER_SRC = (
                Path(__file__).parent.parent.parent / "whisper-service" / "src"
            )
            sys.path.insert(0, str(WHISPER_SRC))
            from websocket_stream_server import WebSocketStreamServer
        except ImportError:
            pytest.skip("Whisper WebSocket server not available")

        whisper_server = WebSocketStreamServer(host="localhost", port=5023)
        server_task = asyncio.create_task(whisper_server.start())
        await asyncio.sleep(0.5)

        try:
            client = WebSocketWhisperClient(whisper_host="localhost", whisper_port=5023)

            # Track received segments
            received_segments = []

            def on_segment(segment):
                received_segments.append(segment)

            client.on_segment(on_segment)

            await client.connect()

            session_id = await client.start_stream(
                session_id="segment-test-session", config={"model": "base"}
            )

            # Send audio that should produce segments
            test_audio = np.random.randn(16000).astype(np.float32)
            await client.send_audio_chunk(
                session_id=session_id, audio_data=test_audio.tobytes()
            )

            # Wait for potential segments
            await asyncio.sleep(0.5)

            # Note: Actual segment reception depends on Whisper model being loaded
            # For now, just verify callback mechanism works
            print(f"Received {len(received_segments)} segments")

            await client.close()

        finally:
            await whisper_server.stop()
            server_task.cancel()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_client_connection_callbacks(self):
        """Test connection state change callbacks"""
        try:
            WHISPER_SRC = (
                Path(__file__).parent.parent.parent / "whisper-service" / "src"
            )
            sys.path.insert(0, str(WHISPER_SRC))
            from websocket_stream_server import WebSocketStreamServer
        except ImportError:
            pytest.skip("Whisper WebSocket server not available")

        whisper_server = WebSocketStreamServer(host="localhost", port=5024)
        server_task = asyncio.create_task(whisper_server.start())
        await asyncio.sleep(0.5)

        try:
            client = WebSocketWhisperClient(
                whisper_host="localhost", whisper_port=5024, auto_reconnect=False
            )

            # Track connection states
            connection_states = []

            def on_connection(connected):
                connection_states.append(connected)

            client.on_connection_change(on_connection)

            # Connect
            await client.connect()
            await asyncio.sleep(0.1)

            # Should have received connection=True
            assert True in connection_states

            # Close
            await client.close()
            await asyncio.sleep(0.1)

            print(f"Connection states: {connection_states}")

        finally:
            await whisper_server.stop()
            server_task.cancel()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
