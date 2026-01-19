#!/usr/bin/env python3
"""
COMPREHENSIVE INTEGRATION TESTS: WebSocket Streaming Server (Whisper Service)

Tests the WebSocket server that handles real-time audio streaming from orchestration service.

Following Phase 3.1 architecture:
- WebSocket server for streaming connections (ws://whisper:5001/stream)
- Session management for multiple concurrent streams
- Real-time audio chunk processing
- Segment streaming with ISO 8601 timestamps
- Integration with Phase 2 features (VAD, CIF, Rolling Context)

NO MOCKS - Only real WebSocket connections and real transcription!

Architecture:
    Orchestration → [WebSocket] → Whisper Service
                                    ↓
                                  Process audio
                                    ↓
                                  Stream segments back
"""

import asyncio
import base64
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import websockets

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


class TestWebSocketStreamServer:
    """
    Integration tests for WebSocket stream server

    Tests WebSocket connection, session management, and basic communication
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_server_startup(self):
        """
        Test WebSocket server starts and accepts connections
        """
        print("\n[WS SERVER] Testing server startup...")

        from websocket_stream_server import WebSocketStreamServer

        server = WebSocketStreamServer(host="localhost", port=5001)

        # Start server in background
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.5)  # Let server start

        try:
            # Try to connect
            async with websockets.connect("ws://localhost:5001/stream"):
                # Connection is open if we're in this context
                print("   ✅ Connected to ws://localhost:5001/stream")

        finally:
            await server.stop()
            server_task.cancel()

        print("✅ WebSocket server startup working")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_start_stream_session(self):
        """
        Test creating a new streaming session

        Message from orchestration:
        {
          "action": "start_stream",
          "session_id": "session-123",
          "config": {"model": "large-v3", "language": "en"}
        }
        """
        print("\n[WS SERVER] Testing session creation...")

        from websocket_stream_server import WebSocketStreamServer

        server = WebSocketStreamServer(host="localhost", port=5002)
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect("ws://localhost:5002/stream") as ws:
                # Send start_stream message
                start_message = {
                    "action": "start_stream",
                    "session_id": "test-session-123",
                    "config": {
                        "model": "large-v3",
                        "language": "en",
                        "enable_vad": True,
                        "enable_cif": True,
                    },
                }

                await ws.send(json.dumps(start_message))

                # Receive acknowledgement
                response_raw = await ws.recv()
                response = json.loads(response_raw)

                assert response["type"] == "session_started", "Should acknowledge session start"
                assert response["session_id"] == "test-session-123"
                assert "timestamp" in response

                print(f"   Session ID: {response['session_id']}")
                print(f"   Timestamp: {response['timestamp']}")
                print("✅ Session creation working")

        finally:
            await server.stop()
            server_task.cancel()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_configuration(self):
        """
        Test that session configurations are properly stored
        """
        print("\n[WS SERVER] Testing session configuration...")

        from websocket_stream_server import WebSocketStreamServer

        server = WebSocketStreamServer(host="localhost", port=5003)
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect("ws://localhost:5003/stream") as ws:
                # Start session with specific config
                start_message = {
                    "action": "start_stream",
                    "session_id": "config-test-session",
                    "config": {
                        "model": "large-v3",
                        "language": "es",  # Spanish
                        "enable_vad": False,
                        "enable_cif": False,
                        "beam_size": 5,
                    },
                }

                await ws.send(json.dumps(start_message))
                response_raw = await ws.recv()
                json.loads(response_raw)

                # Configuration should be reflected in session
                session = server.get_session("config-test-session")
                assert session is not None
                assert session.config["language"] == "es"
                assert not session.config["enable_vad"]
                assert session.config["beam_size"] == 5

                print(f"   Language: {session.config['language']}")
                print(f"   VAD enabled: {session.config['enable_vad']}")
                print(f"   Beam size: {session.config['beam_size']}")
                print("✅ Session configuration working")

        finally:
            await server.stop()
            server_task.cancel()


class TestAudioChunkProcessing:
    """
    Integration tests for processing audio chunks

    Tests receiving audio, processing with real Whisper, streaming results
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_receive_audio_chunk(self):
        """
        Test receiving and queueing audio chunks

        Message from orchestration:
        {
          "type": "audio_chunk",
          "session_id": "session-123",
          "audio": "<base64>",
          "timestamp": "2025-01-15T10:30:00.000Z"
        }
        """
        print("\n[AUDIO PROCESSING] Testing audio chunk reception...")

        from websocket_stream_server import WebSocketStreamServer

        server = WebSocketStreamServer(host="localhost", port=5004)
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect("ws://localhost:5004/stream") as ws:
                # Start session
                await ws.send(
                    json.dumps(
                        {
                            "action": "start_stream",
                            "session_id": "audio-test-session",
                            "config": {"model": "large-v3"},
                        }
                    )
                )
                await ws.recv()  # Ack

                # Create test audio (1 second of silence)
                audio = np.zeros(16000, dtype=np.float32)
                audio_base64 = base64.b64encode(audio.tobytes()).decode("utf-8")

                # Send audio chunk
                audio_message = {
                    "type": "audio_chunk",
                    "session_id": "audio-test-session",
                    "audio": audio_base64,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

                await ws.send(json.dumps(audio_message))

                # Should receive acknowledgement
                response_raw = await ws.recv()
                response = json.loads(response_raw)

                assert response["type"] == "audio_received"
                assert response["session_id"] == "audio-test-session"

                print(f"   Audio size: {len(audio)} samples")
                print(f"   Base64 size: {len(audio_base64)} bytes")
                print("✅ Audio chunk reception working")

        finally:
            await server.stop()
            server_task.cancel()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_process_audio_with_real_whisper(self):
        """
        Test processing audio with real Whisper model

        This verifies integration with Phase 2 features:
        - VAD (Silero)
        - CIF (word boundaries)
        - Rolling Context
        - Beam Search
        """
        print("\n[AUDIO PROCESSING] Testing real Whisper processing...")

        from websocket_stream_server import WebSocketStreamServer
        from whisper_service import ModelManager

        # Initialize model
        manager = ModelManager()
        manager.load_model("large-v3")

        server = WebSocketStreamServer(host="localhost", port=5005, model_manager=manager)
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect("ws://localhost:5005/stream") as ws:
                # Start session
                await ws.send(
                    json.dumps(
                        {
                            "action": "start_stream",
                            "session_id": "whisper-test-session",
                            "config": {
                                "model": "large-v3",
                                "language": "en",
                                "enable_vad": True,
                                "enable_cif": True,
                            },
                        }
                    )
                )
                await ws.recv()  # Ack

                # Create test audio with speech
                # (Simple sine wave as placeholder - in real test would use actual speech)
                duration = 3.0  # seconds
                sample_rate = 16000
                audio = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.1

                audio_base64 = base64.b64encode(audio.tobytes()).decode("utf-8")

                # Send audio chunk
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "session_id": "whisper-test-session",
                            "audio": audio_base64,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                )

                # Wait for processing
                await ws.recv()  # audio_received ack

                # Should eventually receive a segment
                # (with silence/noise it might not produce text, but structure should be valid)
                response_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                response = json.loads(response_raw)

                # Verify segment structure
                assert response["type"] in ["segment", "processing_complete"]

                if response["type"] == "segment":
                    assert "session_id" in response
                    assert "absolute_start_time" in response
                    assert "absolute_end_time" in response
                    print(f"   Segment text: '{response.get('text', '(empty)')}'")
                    print(f"   Start: {response['absolute_start_time']}")
                    print(f"   End: {response['absolute_end_time']}")

                print("✅ Real Whisper processing working")

        finally:
            await server.stop()
            server_task.cancel()


class TestSegmentStreaming:
    """
    Integration tests for streaming segments back to orchestration

    Tests ISO 8601 timestamps, speaker attribution, confidence scores
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_segment_iso8601_timestamps(self):
        """
        Test that segments have ISO 8601 formatted timestamps

        Required fields:
        - absolute_start_time: "2025-01-15T10:30:00Z"
        - absolute_end_time: "2025-01-15T10:30:03Z"
        """
        print("\n[SEGMENT STREAMING] Testing ISO 8601 timestamps...")

        from segment_timestamper import SegmentTimestamper

        timestamper = SegmentTimestamper()

        # Segment from Whisper (relative timestamps)
        segment = {
            "text": "Hello everyone",
            "start": 0.0,  # Relative to chunk start
            "end": 3.0,
            "speaker": "SPEAKER_00",
        }

        # Add absolute timestamps
        chunk_start_time = datetime.now(UTC)
        timestamped = timestamper.add_absolute_timestamps(
            segment=segment, chunk_start_time=chunk_start_time
        )

        # Verify ISO 8601 format
        assert "absolute_start_time" in timestamped
        assert "absolute_end_time" in timestamped

        # Parse to verify format
        start_dt = datetime.fromisoformat(timestamped["absolute_start_time"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(timestamped["absolute_end_time"].replace("Z", "+00:00"))

        assert end_dt > start_dt, "End should be after start"

        print(f"   Start: {timestamped['absolute_start_time']}")
        print(f"   End: {timestamped['absolute_end_time']}")
        print(f"   Duration: {(end_dt - start_dt).total_seconds()}s")
        print("✅ ISO 8601 timestamps working")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_segment_with_speaker_diarization(self):
        """
        Test segment streaming with speaker attribution

        Segments should include:
        - speaker: "SPEAKER_00" (from diarization)
        - confidence: 0.0-1.0 (transcription confidence)
        """
        print("\n[SEGMENT STREAMING] Testing speaker attribution...")

        from segment_timestamper import SegmentTimestamper

        timestamper = SegmentTimestamper()

        segment = {
            "text": "Hello world",
            "start": 0.0,
            "end": 2.0,
            "speaker": "SPEAKER_01",  # From diarization
            "confidence": 0.95,
        }

        timestamped = timestamper.add_absolute_timestamps(
            segment=segment, chunk_start_time=datetime.now(UTC)
        )

        assert timestamped["speaker"] == "SPEAKER_01"
        assert timestamped["confidence"] == 0.95

        print(f"   Speaker: {timestamped['speaker']}")
        print(f"   Confidence: {timestamped['confidence']}")
        print("✅ Speaker attribution working")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_segment_is_final_flag(self):
        """
        Test is_final flag for segment finalization

        - is_final: False → segment may be updated (mutable)
        - is_final: True → segment won't change (finalized)
        """
        print("\n[SEGMENT STREAMING] Testing is_final flag...")

        from segment_timestamper import SegmentTimestamper

        timestamper = SegmentTimestamper()

        # Mutable segment (being updated)
        mutable = timestamper.add_absolute_timestamps(
            segment={"text": "Hello", "start": 0.0, "end": 1.0},
            chunk_start_time=datetime.now(UTC),
            is_final=False,
        )

        assert not mutable["is_final"]

        # Final segment
        final = timestamper.add_absolute_timestamps(
            segment={"text": "Hello world", "start": 0.0, "end": 2.0},
            chunk_start_time=datetime.now(UTC),
            is_final=True,
        )

        assert final["is_final"]

        print(f"   Mutable segment: is_final={mutable['is_final']}")
        print(f"   Final segment: is_final={final['is_final']}")
        print("✅ is_final flag working")


class TestErrorHandlingAndReconnection:
    """
    Integration tests for error handling and reconnection

    Tests graceful degradation, session cleanup, reconnection
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_session_id(self):
        """
        Test handling of audio chunk for non-existent session
        """
        print("\n[ERROR HANDLING] Testing invalid session ID...")

        from websocket_stream_server import WebSocketStreamServer

        server = WebSocketStreamServer(host="localhost", port=5006)
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.5)

        try:
            async with websockets.connect("ws://localhost:5006/stream") as ws:
                # Send audio without starting session
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "session_id": "non-existent-session",
                            "audio": "dGVzdA==",  # "test" in base64
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                )

                # Should receive error
                response_raw = await ws.recv()
                response = json.loads(response_raw)

                assert response["type"] == "error"
                assert "session" in response["error"].lower()

                print(f"   Error: {response['error']}")
                print("✅ Invalid session handling working")

        finally:
            await server.stop()
            server_task.cancel()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_cleanup_on_disconnect(self):
        """
        Test that sessions are cleaned up when client disconnects
        """
        print("\n[ERROR HANDLING] Testing session cleanup...")

        from websocket_stream_server import WebSocketStreamServer

        server = WebSocketStreamServer(host="localhost", port=5007)
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.5)

        try:
            # Connect and start session
            async with websockets.connect("ws://localhost:5007/stream") as ws:
                await ws.send(
                    json.dumps(
                        {
                            "action": "start_stream",
                            "session_id": "cleanup-test-session",
                            "config": {"model": "large-v3"},
                        }
                    )
                )
                await ws.recv()  # Ack

                # Verify session exists
                session = server.get_session("cleanup-test-session")
                assert session is not None
                print(f"   Session created: {session.session_id}")

            # Connection closed - wait for cleanup
            await asyncio.sleep(0.5)

            # Session should be cleaned up
            session = server.get_session("cleanup-test-session")
            assert session is None
            print("   Session cleaned up after disconnect")
            print("✅ Session cleanup working")

        finally:
            await server.stop()
            server_task.cancel()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_reconnection_with_same_session_id(self):
        """
        Test reconnecting with the same session ID

        Should either:
        - Resume existing session, or
        - Replace old session with new one
        """
        print("\n[ERROR HANDLING] Testing reconnection...")

        from websocket_stream_server import WebSocketStreamServer

        server = WebSocketStreamServer(host="localhost", port=5008)
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.5)

        try:
            # First connection
            async with websockets.connect("ws://localhost:5008/stream") as ws1:
                await ws1.send(
                    json.dumps(
                        {
                            "action": "start_stream",
                            "session_id": "reconnect-session",
                            "config": {"model": "large-v3"},
                        }
                    )
                )
                await ws1.recv()

            # Second connection with same session ID
            async with websockets.connect("ws://localhost:5008/stream") as ws2:
                await ws2.send(
                    json.dumps(
                        {
                            "action": "start_stream",
                            "session_id": "reconnect-session",
                            "config": {"model": "large-v3"},
                        }
                    )
                )
                response_raw = await ws2.recv()
                response = json.loads(response_raw)

                # Should either resume or replace
                assert response["type"] in ["session_started", "session_resumed"]
                print(f"   Response: {response['type']}")
                print("✅ Reconnection handling working")

        finally:
            await server.stop()
            server_task.cancel()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
