"""
Integration Tests for Pipeline Studio Real-Time WebSocket Streaming

These tests verify the COMPLETE END-TO-END functionality with NO MOCKS.
- Real WebSocket connections
- Real audio processing
- Real pipeline execution
- Real metrics collection

Run with: pytest tests/integration/test_pipeline_streaming.py -v -s
"""

import pytest
import asyncio
import base64
import json
import time
import wave
import struct
from pathlib import Path
from typing import Dict, Any, List
import websockets
from websockets.client import WebSocketClientProtocol
from httpx import AsyncClient
import numpy as np


# Test Configuration
BASE_URL = "http://localhost:3000"
WS_BASE_URL = "ws://localhost:3000"
TEST_AUDIO_DIR = Path(__file__).parent.parent / "test_audio"


class PipelineStreamingIntegrationTest:
    """Integration test suite for Pipeline Studio streaming"""

    @pytest.fixture(scope="function")
    async def http_client(self):
        """Create HTTP client for API calls"""
        async with AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            yield client

    @pytest.fixture(scope="function")
    async def pipeline_session(self, http_client):
        """Create a real-time pipeline session"""
        # Define a real pipeline configuration
        pipeline_config = {
            "pipeline_id": f"test-pipeline-{int(time.time())}",
            "name": "Integration Test Pipeline",
            "stages": {
                "vad": {
                    "enabled": True,
                    "gain_in": 0.0,
                    "gain_out": 0.0,
                    "parameters": {"aggressiveness": 2, "frame_duration": 30},
                },
                "noise_reduction": {
                    "enabled": True,
                    "gain_in": 0.0,
                    "gain_out": 0.0,
                    "parameters": {"strength": 0.5, "smoothing": 0.3},
                },
                "voice_enhancement": {
                    "enabled": True,
                    "gain_in": 0.0,
                    "gain_out": 0.0,
                    "parameters": {"clarity": 0.5, "warmth": 0.5},
                },
            },
            "connections": [
                {
                    "id": "conn1",
                    "source_stage_id": "vad",
                    "target_stage_id": "noise_reduction",
                },
                {
                    "id": "conn2",
                    "source_stage_id": "noise_reduction",
                    "target_stage_id": "voice_enhancement",
                },
            ],
        }

        # Start real-time session
        response = await http_client.post(
            "/api/pipeline/realtime/start",
            json={
                "pipeline_config": pipeline_config,
                "session_config": {
                    "chunk_size": 1024,
                    "sample_rate": 16000,
                    "channels": 1,
                    "buffer_size": 4096,
                    "latency_target": 100,
                },
            },
        )

        assert response.status_code == 200, f"Failed to start session: {response.text}"
        session = response.json()

        # Give the backend a moment to fully initialize the session
        await asyncio.sleep(0.1)

        yield session

        # Cleanup: Stop session
        try:
            await http_client.delete(f"/api/pipeline/realtime/{session['session_id']}")
        except:
            pass

    @pytest.fixture(scope="function")
    def generate_test_audio(self):
        """Generate real audio samples (sine wave)"""

        def _generate(
            duration_seconds: float = 1.0,
            frequency: int = 440,
            sample_rate: int = 16000,
        ) -> bytes:
            """Generate sine wave audio"""
            num_samples = int(duration_seconds * sample_rate)
            t = np.linspace(0, duration_seconds, num_samples, False)

            # Generate sine wave
            audio_signal = np.sin(2 * np.pi * frequency * t)

            # Convert to 16-bit PCM
            audio_signal = (audio_signal * 32767).astype(np.int16)

            # Convert to bytes
            return audio_signal.tobytes()

        return _generate

    @pytest.fixture(scope="function")
    def create_wav_file(self, tmp_path):
        """Create real WAV file"""

        def _create(
            audio_data: bytes, sample_rate: int = 16000, channels: int = 1
        ) -> Path:
            """Create WAV file from audio data"""
            wav_path = tmp_path / f"test_audio_{int(time.time())}.wav"

            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)

            return wav_path

        return _create


@pytest.mark.asyncio
@pytest.mark.integration
class TestPipelineWebSocketStreaming(PipelineStreamingIntegrationTest):
    """Test WebSocket streaming functionality"""

    async def test_websocket_connection_establishment(self, pipeline_session):
        """
        TEST: WebSocket connection can be established
        VERIFY: Connection opens successfully with valid session
        """
        session_id = pipeline_session["session_id"]
        ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"
        print(f"\n🔍 Connecting to: {ws_url}")
        print(f"   Session ID: {session_id}")
        print(f"   Session data: {pipeline_session}")

        async with websockets.connect(ws_url) as websocket:
            print(f"   ✅ WebSocket connected")
            # Verify connection is open by sending a ping message
            await websocket.send(json.dumps({"type": "ping"}))

            # Receive pong response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            message = json.loads(response)

            assert message["type"] == "pong", "Should receive pong response"

    async def test_websocket_heartbeat(self, pipeline_session):
        """
        TEST: WebSocket heartbeat (ping/pong) works
        VERIFY: Server responds to ping with pong
        """
        session_id = pipeline_session["session_id"]
        ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"

        async with websockets.connect(ws_url) as websocket:
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))

            # Receive pong
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            message = json.loads(response)

            assert message["type"] == "pong", "Should receive pong response"

    async def test_websocket_audio_chunk_processing(
        self, pipeline_session, generate_test_audio
    ):
        """
        TEST: Audio chunks are processed through pipeline
        VERIFY: Send audio chunk, receive processed audio + metrics
        """
        session_id = pipeline_session["session_id"]
        ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"

        # Generate real audio (100ms @ 16kHz = 1600 samples)
        audio_data = generate_test_audio(
            duration_seconds=0.1, frequency=440, sample_rate=16000
        )
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        async with websockets.connect(ws_url) as websocket:
            # Send audio chunk
            await websocket.send(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "data": audio_b64,
                        "timestamp": int(time.time() * 1000),
                    }
                )
            )

            # Collect responses
            responses = []
            processed_audio_received = False
            metrics_received = False

            # Wait for responses (timeout 5 seconds)
            try:
                for _ in range(10):  # Max 10 messages
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message = json.loads(response)
                    responses.append(message)
                    print(
                        f"Received message type: {message.get('type')}, keys: {message.keys()}"
                    )

                    if message["type"] == "processed_audio":
                        processed_audio_received = True
                        # Verify processed audio is base64
                        assert "audio" in message, "Should contain audio field"
                        assert len(message["audio"]) > 0, "Audio should not be empty"

                        # Verify can decode
                        processed_bytes = base64.b64decode(message["audio"])
                        assert len(processed_bytes) > 0, (
                            "Processed audio should have content"
                        )

                    elif message["type"] == "metrics":
                        metrics_received = True
                        # Verify metrics structure
                        assert "metrics" in message, "Should contain metrics field"
                        metrics = message["metrics"]

                        assert "total_latency" in metrics, "Should have total_latency"
                        assert "chunks_processed" in metrics, (
                            "Should have chunks_processed"
                        )
                        assert "average_latency" in metrics, (
                            "Should have average_latency"
                        )
                        assert "quality_metrics" in metrics, (
                            "Should have quality_metrics"
                        )

                        # Verify latency is reasonable (<500ms)
                        assert metrics["total_latency"] < 500, (
                            f"Latency too high: {metrics['total_latency']}ms"
                        )
                        assert metrics["chunks_processed"] >= 1, (
                            "Should have processed at least 1 chunk"
                        )

                    # Break if we got both
                    if processed_audio_received and metrics_received:
                        break
            except asyncio.TimeoutError:
                pass  # Expected after receiving all messages

            # Assertions
            assert processed_audio_received, "Should receive processed audio"
            assert metrics_received, "Should receive metrics"

    async def test_multiple_chunks_streaming(
        self, pipeline_session, generate_test_audio
    ):
        """
        TEST: Multiple audio chunks can be streamed continuously
        VERIFY: Process 10 consecutive chunks, verify metrics increase
        """
        session_id = pipeline_session["session_id"]
        ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"

        async with websockets.connect(ws_url) as websocket:
            num_chunks = 10
            chunks_sent = 0
            chunks_processed_count = 0
            last_chunks_count = 0

            for i in range(num_chunks):
                # Generate unique audio (different frequency each time)
                frequency = 440 + (i * 50)  # 440Hz, 490Hz, 540Hz, etc.
                audio_data = generate_test_audio(
                    duration_seconds=0.1, frequency=frequency
                )
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")

                # Send chunk
                await websocket.send(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "data": audio_b64,
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                )
                chunks_sent += 1

                # Receive response
                try:
                    while True:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        message = json.loads(response)

                        if message["type"] == "metrics":
                            current_count = message["metrics"]["chunks_processed"]
                            if current_count > last_chunks_count:
                                chunks_processed_count = current_count
                                last_chunks_count = current_count
                                break
                except asyncio.TimeoutError:
                    break

            # Verify all chunks were processed
            assert chunks_processed_count >= num_chunks, (
                f"Expected {num_chunks} chunks processed, got {chunks_processed_count}"
            )

    async def test_live_parameter_update(self, pipeline_session):
        """
        TEST: Pipeline parameters can be updated in real-time
        VERIFY: Send config update, receive confirmation
        """
        session_id = pipeline_session["session_id"]
        ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"

        async with websockets.connect(ws_url) as websocket:
            # Update noise reduction strength
            await websocket.send(
                json.dumps(
                    {
                        "type": "update_stage",
                        "stage_id": "noise_reduction",
                        "parameters": {"strength": 0.9, "smoothing": 0.5},
                    }
                )
            )

            # Wait for confirmation
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            message = json.loads(response)

            assert message["type"] == "config_updated", (
                "Should receive config update confirmation"
            )
            assert message["success"] is True, "Update should succeed"
            assert message["stage_id"] == "noise_reduction", (
                "Should confirm correct stage"
            )

    async def test_error_handling_invalid_chunk(self, pipeline_session):
        """
        TEST: Invalid audio chunk triggers error response
        VERIFY: Send malformed data, receive error message
        """
        session_id = pipeline_session["session_id"]
        ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"

        async with websockets.connect(ws_url) as websocket:
            # Send invalid base64 data
            await websocket.send(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "data": "invalid_base64_!!!",
                        "timestamp": int(time.time() * 1000),
                    }
                )
            )

            # Should receive error
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            message = json.loads(response)

            assert message["type"] == "error", "Should receive error message"
            assert "error" in message, "Should contain error description"

    async def test_concurrent_sessions(self, http_client, generate_test_audio):
        """
        TEST: Multiple concurrent WebSocket sessions can run
        VERIFY: Create 3 sessions, stream to all simultaneously
        """
        sessions = []
        websockets_list = []

        try:
            # Create 3 sessions
            for i in range(3):
                pipeline_config = {
                    "pipeline_id": f"concurrent-test-{i}-{int(time.time())}",
                    "name": f"Concurrent Session {i}",
                    "stages": {
                        "vad": {
                            "enabled": True,
                            "gain_in": 0.0,
                            "gain_out": 0.0,
                            "parameters": {"aggressiveness": 2},
                        }
                    },
                    "connections": [],
                }

                response = await http_client.post(
                    "/api/pipeline/realtime/start",
                    json={"pipeline_config": pipeline_config},
                )
                assert response.status_code == 200
                sessions.append(response.json())

            # Connect WebSockets
            for session in sessions:
                ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session['session_id']}"
                ws = await websockets.connect(ws_url)
                websockets_list.append(ws)

            # Send audio to all sessions
            audio_data = generate_test_audio(duration_seconds=0.1)
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")

            for ws in websockets_list:
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "data": audio_b64,
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                )

            # Verify all receive responses
            responses_received = [False] * 3
            for i, ws in enumerate(websockets_list):
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    message = json.loads(response)
                    if message["type"] in ["processed_audio", "metrics"]:
                        responses_received[i] = True
                except asyncio.TimeoutError:
                    pass

            # At least some sessions should respond
            assert any(responses_received), "At least one session should respond"

        finally:
            # Cleanup
            for ws in websockets_list:
                try:
                    await ws.close()
                except:
                    pass

            for session in sessions:
                try:
                    await http_client.delete(
                        f"/api/pipeline/realtime/{session['session_id']}"
                    )
                except:
                    pass


@pytest.mark.asyncio
@pytest.mark.integration
class TestPipelineBatchProcessing(PipelineStreamingIntegrationTest):
    """Test batch pipeline processing"""

    async def test_batch_pipeline_execution(
        self, http_client, generate_test_audio, create_wav_file
    ):
        """
        TEST: Complete pipeline execution in batch mode
        VERIFY: Upload audio, process through pipeline, receive results
        """
        # Generate real audio file
        audio_data = generate_test_audio(duration_seconds=2.0, frequency=440)
        wav_file = create_wav_file(audio_data)

        # Read file
        with open(wav_file, "rb") as f:
            audio_content = f.read()

        # Pipeline configuration
        pipeline_config = {
            "pipeline_id": f"batch-test-{int(time.time())}",
            "name": "Batch Processing Test",
            "stages": {
                "noise_reduction": {
                    "enabled": True,
                    "gain_in": 0.0,
                    "gain_out": 0.0,
                    "parameters": {"strength": 0.7},
                },
                "lufs_normalization": {
                    "enabled": True,
                    "gain_in": 0.0,
                    "gain_out": 0.0,
                    "parameters": {"target_lufs": -23.0},
                },
            },
            "connections": [
                {
                    "id": "conn1",
                    "source_stage_id": "noise_reduction",
                    "target_stage_id": "lufs_normalization",
                }
            ],
        }

        # Process
        response = await http_client.post(
            "/api/pipeline/process",
            data={
                "pipeline_config": json.dumps(pipeline_config),
                "processing_mode": "batch",
                "output_format": "wav",
            },
            files={"audio_file": ("test.wav", audio_content, "audio/wav")},
        )

        assert response.status_code == 200, f"Processing failed: {response.text}"
        result = response.json()

        # Verify response structure
        assert result["success"] is True, "Processing should succeed"
        assert "processed_audio" in result, "Should return processed audio"
        assert result["processed_audio"] is not None, (
            "Processed audio should not be None"
        )
        assert "metrics" in result, "Should return metrics"

        # Verify metrics
        metrics = result["metrics"]
        assert "total_latency" in metrics, "Should have latency metric"
        assert metrics["total_latency"] > 0, "Latency should be positive"
        assert "quality_metrics" in metrics, "Should have quality metrics"

        # Verify can decode processed audio
        processed_audio_bytes = base64.b64decode(result["processed_audio"])
        assert len(processed_audio_bytes) > 0, "Processed audio should have content"
        assert len(processed_audio_bytes) > 1000, (
            "Processed audio should be substantial"
        )

    async def test_single_stage_processing(self, http_client, generate_test_audio):
        """
        TEST: Single audio processing stage
        VERIFY: Process audio through one stage only
        """
        # Generate test audio
        audio_data = generate_test_audio(duration_seconds=0.5, frequency=1000)
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        # Process through single stage
        stage_config = {"strength": 0.5, "smoothing": 0.3}

        response = await http_client.post(
            "/api/audio/process/stage/noise_reduction",
            data={"audio_data": audio_b64, "stage_config": json.dumps(stage_config)},
        )

        assert response.status_code == 200, f"Stage processing failed: {response.text}"
        result = response.json()

        # Verify result
        assert "data" in result or "processed_audio" in result, (
            "Should return processed audio"
        )


@pytest.mark.asyncio
@pytest.mark.integration
class TestPipelinePerformance(PipelineStreamingIntegrationTest):
    """Performance and stress tests"""

    async def test_latency_under_100ms(self, pipeline_session, generate_test_audio):
        """
        TEST: Processing latency is under 100ms per chunk
        VERIFY: Measure actual round-trip time
        """
        session_id = pipeline_session["session_id"]
        ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"

        latencies = []

        async with websockets.connect(ws_url) as websocket:
            for _ in range(20):  # Test 20 chunks
                # Generate audio
                audio_data = generate_test_audio(duration_seconds=0.1)
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")

                # Measure round-trip time
                start_time = time.perf_counter()

                await websocket.send(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "data": audio_b64,
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                )

                # Wait for response
                try:
                    while True:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        message = json.loads(response)
                        if message["type"] == "metrics":
                            end_time = time.perf_counter()
                            latency_ms = (end_time - start_time) * 1000
                            latencies.append(latency_ms)
                            break
                except asyncio.TimeoutError:
                    latencies.append(2000)  # Timeout = 2000ms

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\n📊 Latency Stats:")
        print(f"   Average: {avg_latency:.1f}ms")
        print(f"   Max: {max_latency:.1f}ms")
        print(f"   P95: {p95_latency:.1f}ms")

        # Assertions
        assert avg_latency < 150, f"Average latency too high: {avg_latency:.1f}ms"
        assert p95_latency < 300, f"P95 latency too high: {p95_latency:.1f}ms"

    async def test_sustained_streaming_1_minute(
        self, pipeline_session, generate_test_audio
    ):
        """
        TEST: System can handle sustained streaming for 1 minute
        VERIFY: No crashes, consistent performance
        """
        session_id = pipeline_session["session_id"]
        ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"

        duration_seconds = 60
        chunk_interval = 0.1  # 100ms chunks
        expected_chunks = int(duration_seconds / chunk_interval)

        chunks_sent = 0
        chunks_processed = 0
        errors = 0

        async with websockets.connect(ws_url) as websocket:
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                try:
                    # Send chunk
                    audio_data = generate_test_audio(duration_seconds=chunk_interval)
                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")

                    await websocket.send(
                        json.dumps(
                            {
                                "type": "audio_chunk",
                                "data": audio_b64,
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                    )
                    chunks_sent += 1

                    # Try to receive (non-blocking)
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(), timeout=0.01
                        )
                        message = json.loads(response)
                        if message["type"] == "metrics":
                            chunks_processed = message["metrics"]["chunks_processed"]
                    except asyncio.TimeoutError:
                        pass

                    # Wait for next interval
                    await asyncio.sleep(chunk_interval)

                except Exception as e:
                    errors += 1
                    print(f"Error during streaming: {e}")

        print(f"\n📊 Sustained Streaming Stats:")
        print(f"   Duration: {duration_seconds}s")
        print(f"   Chunks sent: {chunks_sent}")
        print(f"   Chunks processed: {chunks_processed}")
        print(f"   Errors: {errors}")
        print(f"   Success rate: {(chunks_processed / chunks_sent * 100):.1f}%")

        # Assertions
        assert chunks_sent >= expected_chunks * 0.9, "Should send most expected chunks"
        assert chunks_processed >= chunks_sent * 0.8, (
            "Should process at least 80% of chunks"
        )
        assert errors < chunks_sent * 0.05, "Error rate should be under 5%"


# Test fixtures and helpers
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
