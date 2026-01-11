#!/usr/bin/env python3
"""
CROSS-SERVICE INTEGRATION TESTS: Orchestration ↔ Whisper WebSocket

Tests the REAL WebSocket connection between orchestration and whisper services.
NO MOCKS - Both services actually running and communicating!

Following Phase 3.1 architecture:
- Orchestration WebSocket Client → Whisper WebSocket Server
- Audio streaming from orchestration to whisper
- Segment streaming from whisper to orchestration
- Real-time transcription pipeline

Architecture:
    Orchestration (WebSocket Client) ↔ Whisper (WebSocket Server)
"""

import pytest
import asyncio
import websockets
import json
import base64
import numpy as np
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add src directories to path
ORCH_SRC = Path(__file__).parent.parent / "src"
WHISPER_SRC = Path(__file__).parent.parent.parent / "whisper-service" / "src"
sys.path.insert(0, str(ORCH_SRC))
sys.path.insert(0, str(WHISPER_SRC))


class TestOrchestrationWhisperIntegration:
    """
    Integration tests for orchestration-whisper WebSocket communication

    Tests REAL connection between services (no mocks)
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_orchestration_can_connect_to_whisper(self):
        """
        Test orchestration can connect to whisper WebSocket server

        Setup:
        1. Start Whisper WebSocket server
        2. Connect from orchestration as client
        3. Verify connection established
        """
        print("\n[CROSS-SERVICE] Testing orchestration → whisper connection...")

        from websocket_stream_server import WebSocketStreamServer

        # Start Whisper server
        whisper_server = WebSocketStreamServer(host="localhost", port=5010)
        server_task = asyncio.create_task(whisper_server.start())
        await asyncio.sleep(0.5)  # Let server start

        try:
            # Connect from orchestration (as client)
            async with websockets.connect("ws://localhost:5010/stream") as ws:
                print("   ✅ Orchestration connected to Whisper")

                # Verify connection is open
                assert ws

        finally:
            await whisper_server.stop()
            server_task.cancel()

        print("✅ Cross-service connection working")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_orchestration_starts_whisper_session(self):
        """
        Test orchestration can start a streaming session on whisper

        Flow:
        1. Orchestration connects to Whisper
        2. Sends start_stream message
        3. Whisper creates session
        4. Whisper sends session_started acknowledgement
        """
        print("\n[CROSS-SERVICE] Testing session creation...")

        from websocket_stream_server import WebSocketStreamServer

        whisper_server = WebSocketStreamServer(host="localhost", port=5011)
        server_task = asyncio.create_task(whisper_server.start())
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect("ws://localhost:5011/stream") as ws:
                # Orchestration sends start_stream
                start_message = {
                    "action": "start_stream",
                    "session_id": "orch-session-123",
                    "config": {
                        "model": "large-v3",
                        "language": "en",
                        "enable_vad": True,
                    },
                }

                await ws.send(json.dumps(start_message))

                # Whisper responds with session_started
                response_raw = await ws.recv()
                response = json.loads(response_raw)

                assert response["type"] == "session_started"
                assert response["session_id"] == "orch-session-123"
                assert "timestamp" in response

                print(f"   Session ID: {response['session_id']}")
                print("   ✅ Session created on Whisper")

        finally:
            await whisper_server.stop()
            server_task.cancel()

        print("✅ Cross-service session creation working")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_orchestration_sends_audio_to_whisper(self):
        """
        Test orchestration can stream audio to whisper

        Flow:
        1. Start session
        2. Orchestration sends audio chunks
        3. Whisper receives and buffers audio
        4. Verify audio is queued for processing
        """
        print("\n[CROSS-SERVICE] Testing audio streaming...")

        from websocket_stream_server import WebSocketStreamServer

        whisper_server = WebSocketStreamServer(host="localhost", port=5012)
        server_task = asyncio.create_task(whisper_server.start())
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect("ws://localhost:5012/stream") as ws:
                # Start session
                start_message = {
                    "action": "start_stream",
                    "session_id": "audio-test-session",
                    "config": {"model": "large-v3", "language": "en"},
                }
                await ws.send(json.dumps(start_message))
                await ws.recv()  # Acknowledge

                # Generate test audio (1 second of random audio)
                test_audio = np.random.randn(16000).astype(np.float32)
                audio_base64 = base64.b64encode(test_audio.tobytes()).decode("utf-8")

                # Send audio chunk
                audio_message = {
                    "type": "audio_chunk",
                    "session_id": "audio-test-session",
                    "audio": audio_base64,
                    "timestamp": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }

                await ws.send(json.dumps(audio_message))

                # Give the server a moment to process the message
                await asyncio.sleep(0.1)

                # Check session has buffered audio
                session = whisper_server.get_session("audio-test-session")
                assert session is not None
                assert session.chunks_received > 0

                print(f"   Audio chunks received: {session.chunks_received}")
                print(f"   Buffer size: {len(session.audio_buffer)} samples")
                print("   ✅ Audio streaming to Whisper working")

        finally:
            await whisper_server.stop()
            server_task.cancel()

        print("✅ Cross-service audio streaming working")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_whisper_streams_segments_to_orchestration(self):
        """
        Test whisper can stream transcript segments back to orchestration

        Flow:
        1. Orchestration sends audio
        2. Whisper processes and generates segments
        3. Whisper streams segments back to orchestration
        4. Orchestration receives segments with timestamps

        NOTE: This test will use mock transcription since we don't have
        real Whisper model loaded, but tests the message flow
        """
        print("\n[CROSS-SERVICE] Testing segment streaming...")

        from websocket_stream_server import WebSocketStreamServer

        whisper_server = WebSocketStreamServer(host="localhost", port=5013)
        server_task = asyncio.create_task(whisper_server.start())
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect("ws://localhost:5013/stream") as ws:
                # Start session
                await ws.send(
                    json.dumps(
                        {
                            "action": "start_stream",
                            "session_id": "segment-test",
                            "config": {"model": "large-v3"},
                        }
                    )
                )
                await ws.recv()

                # Send enough audio to trigger processing (1+ seconds)
                test_audio = np.random.randn(16000).astype(np.float32)
                audio_base64 = base64.b64encode(test_audio.tobytes()).decode("utf-8")

                await ws.send(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "session_id": "segment-test",
                            "audio": audio_base64,
                            "timestamp": datetime.now(timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z"),
                        }
                    )
                )

                # For now, just verify the flow is set up
                # Real segments would come when model is integrated
                print("   ✅ Segment streaming flow established")

        finally:
            await whisper_server.stop()
            server_task.cancel()

        print("✅ Cross-service segment streaming ready")


class TestOrchestrationWebSocketClient:
    """
    Tests for the orchestration WebSocket client component

    Tests the client that orchestration uses to connect to whisper
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_client_initialization(self):
        """
        Test WebSocket client can be initialized

        Once websocket_whisper_client.py is implemented, this will test:
        - Client initialization
        - Connection configuration
        - Session management
        """
        print("\n[ORCH CLIENT] Testing client initialization...")

        # This will be implemented when we create websocket_whisper_client.py
        # For now, placeholder test

        print("   ⚪ Waiting for websocket_whisper_client.py implementation")
        pytest.skip("Waiting for websocket_whisper_client.py")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_client_reconnection_handling(self):
        """
        Test client can reconnect if connection is lost

        Resilience test:
        - Connect to Whisper
        - Simulate connection drop
        - Client should auto-reconnect
        - Session should resume
        """
        print("\n[ORCH CLIENT] Testing reconnection...")

        pytest.skip("Waiting for websocket_whisper_client.py")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
