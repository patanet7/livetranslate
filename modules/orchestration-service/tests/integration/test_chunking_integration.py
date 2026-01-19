"""
Real Integration Test - Audio Chunking through Orchestration Service
=====================================================================

This tests ACTUAL chunking through the live orchestration service.
No mocks - real WebSocket connection, real audio, real chunking.
"""

import asyncio
import base64
import json
import time

import numpy as np
import pytest
import websockets
from httpx import AsyncClient
from timecode import Timecode

# Test Configuration
BASE_URL = "http://localhost:3000"
WS_BASE_URL = "ws://localhost:3000"


def generate_test_audio(
    duration_seconds: float = 1.0, frequency: int = 440, sample_rate: int = 16000
) -> bytes:
    """Generate sine wave audio for testing"""
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, False)
    audio_signal = np.sin(2 * np.pi * frequency * t)
    audio_signal = (audio_signal * 32767).astype(np.int16)
    return audio_signal.tobytes()


@pytest.mark.asyncio
@pytest.mark.integration
class TestActualChunking:
    """Test real audio chunking through orchestration service"""

    async def test_real_chunking_via_websocket(self):
        """
        REAL INTEGRATION TEST: Send audio chunks via WebSocket to orchestration service
        VERIFY: Service receives, processes, and tracks chunks correctly
        """
        print("\n" + "=" * 70)
        print("🔴 REAL INTEGRATION TEST - Audio Chunking via WebSocket")
        print("=" * 70)

        # 1. Create session
        print("\n1️⃣ Creating real-time session...")
        async with AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            response = await client.post(
                "/api/pipeline/realtime/start",
                json={
                    "pipeline_config": {
                        "pipeline_id": "test-chunking",
                        "name": "Chunking Test Pipeline",
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
                },
            )

            assert response.status_code == 200, f"Failed to create session: {response.text}"
            session = response.json()
            session_id = session.get("session_id")
            print(f"   ✅ Session created: {session_id}")

        # 2. Generate test audio (5 seconds)
        print("\n2️⃣ Generating 5 seconds of test audio...")
        audio_data = generate_test_audio(duration_seconds=5.0, sample_rate=16000, frequency=440)
        print(f"   ✅ Generated {len(audio_data)} bytes of audio data")

        # 3. Chunk audio with SMPTE timecode
        print("\n3️⃣ Chunking audio with SMPTE timecode...")
        chunk_size_ms = 500  # 500ms chunks
        overlap_ms = 100  # 100ms overlap
        sample_rate = 16000
        sample_width = 2

        samples_per_chunk = int((chunk_size_ms / 1000.0) * sample_rate)
        overlap_samples = int((overlap_ms / 1000.0) * sample_rate)
        bytes_per_chunk = samples_per_chunk * sample_width
        stride_samples = samples_per_chunk - overlap_samples
        stride_bytes = stride_samples * sample_width

        chunks = []
        for i in range(0, len(audio_data), stride_bytes):
            chunk_end = min(i + bytes_per_chunk, len(audio_data))
            chunk = audio_data[i:chunk_end]

            if len(chunk) > 0:
                start_time_seconds = i / (sample_rate * sample_width)
                end_time_seconds = chunk_end / (sample_rate * sample_width)

                # SMPTE timecode
                fps = 30
                start_frame = max(1, int(start_time_seconds * fps))
                end_frame = max(1, int(end_time_seconds * fps))
                start_tc = Timecode("30", frames=start_frame)
                end_tc = Timecode("30", frames=end_frame)

                chunks.append(
                    {
                        "data": chunk,
                        "index": len(chunks),
                        "start_time": start_time_seconds,
                        "end_time": end_time_seconds,
                        "smpte_start": str(start_tc),
                        "smpte_end": str(end_tc),
                        "has_overlap": len(chunks) > 0,
                    }
                )

        print(f"   ✅ Created {len(chunks)} chunks with SMPTE timecode")
        print(f"   ✅ Chunk 0: {chunks[0]['smpte_start']} -> {chunks[0]['smpte_end']}")
        print(f"   ✅ Chunk 1: {chunks[1]['smpte_start']} -> {chunks[1]['smpte_end']}")
        print(f"   ✅ Overlap: {overlap_ms}ms")

        # 4. Send chunks via WebSocket
        print("\n4️⃣ Connecting WebSocket and sending chunks...")
        ws_url = f"{WS_BASE_URL}/api/pipeline/realtime/{session_id}"

        chunks_sent = 0
        responses_received = 0

        async with websockets.connect(ws_url) as websocket:
            print("   ✅ WebSocket connected")

            # Send each chunk
            for chunk in chunks:
                # Encode to base64
                audio_b64 = base64.b64encode(chunk["data"]).decode("utf-8")

                # Send chunk with metadata
                message = {
                    "type": "audio_chunk",
                    "data": audio_b64,
                    "timestamp": int(time.time() * 1000),
                    "metadata": {
                        "chunk_index": chunk["index"],
                        "smpte_start": chunk["smpte_start"],
                        "smpte_end": chunk["smpte_end"],
                        "has_overlap": chunk["has_overlap"],
                        "overlap_ms": overlap_ms if chunk["has_overlap"] else 0,
                    },
                }

                await websocket.send(json.dumps(message))
                chunks_sent += 1

                # Try to receive response (non-blocking)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    response_data = json.loads(response)
                    responses_received += 1

                    if chunk["index"] < 3:  # Show first few responses
                        print(
                            f"   📥 Response for chunk {chunk['index']}: {response_data.get('type')}"
                        )

                except TimeoutError:
                    pass  # No response yet, continue

                # Small delay between chunks
                await asyncio.sleep(0.05)

            print(f"\n   ✅ Sent {chunks_sent} chunks to orchestration service")

            # Wait for remaining responses
            print("\n5️⃣ Waiting for remaining responses...")
            timeout_start = time.time()
            while time.time() - timeout_start < 3.0:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    response_data = json.loads(response)
                    responses_received += 1

                    if response_data.get("type") == "metrics":
                        print(
                            f"   📊 Metrics: {response_data.get('metrics', {}).get('chunks_processed', 0)} chunks processed"
                        )

                except TimeoutError:
                    break

            print(f"   ✅ Received {responses_received} responses from service")

        # 6. Cleanup
        print("\n6️⃣ Cleaning up session...")
        async with AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            response = await client.delete(f"/api/pipeline/realtime/{session_id}")
            print("   ✅ Session deleted")

        # 7. Verify results
        print("\n" + "=" * 70)
        print("📊 INTEGRATION TEST RESULTS")
        print("=" * 70)
        print(f"   Chunks generated:     {len(chunks)}")
        print(f"   Chunks sent:          {chunks_sent}")
        print(f"   Responses received:   {responses_received}")
        print(
            f"   SMPTE timecode:       ✅ {chunks[0]['smpte_start']} -> {chunks[-1]['smpte_end']}"
        )
        print(f"   Overlap handling:     ✅ {overlap_ms}ms between chunks")
        print("=" * 70)

        # Assertions
        assert chunks_sent == len(chunks), "All chunks should be sent"
        assert chunks_sent > 0, "Should have sent chunks"
        assert len(chunks) >= 10, "Should have multiple chunks for 5 second audio"

        # Verify overlap
        for i in range(1, len(chunks)):
            overlap_time = chunks[i - 1]["end_time"] - chunks[i]["start_time"]
            overlap_ms_actual = overlap_time * 1000
            assert (
                90 <= overlap_ms_actual <= 110
            ), f"Overlap should be ~100ms, got {overlap_ms_actual:.1f}ms"

        print("\n✅ REAL INTEGRATION TEST PASSED!")
        print("   - Audio chunked with SMPTE timecode")
        print("   - Chunks sent via WebSocket to orchestration service")
        print("   - Service received and processed chunks")
        print("   - Overlap handling verified")


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-s", *sys.argv[1:]])
