"""
Streaming Simulation Tests - Simulates Real Frontend Streaming Behavior

These tests simulate the actual streaming use case where:
1. Frontend sends audio chunks every 2-5 seconds
2. Multiple chunks come in sequentially
3. Each chunk is processed independently
4. Results are accumulated to form complete transcription
5. System handles concurrent chunk processing
"""

import asyncio
import io
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


@pytest.fixture
def streaming_audio_chunks():
    """
    Generate multiple audio chunks simulating real streaming.

    Returns 5 chunks of 2-second audio, each with different frequency
    to simulate different speech segments.
    """
    sample_rate = 16000
    chunk_duration = 2.0  # 2 seconds per chunk
    chunks = []

    frequencies = [220, 440, 660, 880, 1100]  # Different tones for each chunk

    for freq in frequencies:
        t = np.linspace(0, chunk_duration, int(sample_rate * chunk_duration))
        audio = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        chunks.append(buffer.read())

    return chunks, sample_rate


@pytest.fixture
def mock_progressive_whisper_responses():
    """
    Mock Whisper responses that simulate progressive transcription.

    Each chunk returns part of a sentence being spoken.
    """
    return [
        {
            "text": "Hello,",
            "language": "en",
            "confidence": 0.95,
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hello,"}],
            "speaker_info": {"speakers": ["SPEAKER_00"]},
            "metadata": {"processing_time": 0.5},
        },
        {
            "text": "how are you",
            "language": "en",
            "confidence": 0.93,
            "segments": [{"start": 2.0, "end": 4.0, "text": "how are you"}],
            "speaker_info": {"speakers": ["SPEAKER_00"]},
            "metadata": {"processing_time": 0.6},
        },
        {
            "text": "doing today?",
            "language": "en",
            "confidence": 0.94,
            "segments": [{"start": 4.0, "end": 6.0, "text": "doing today?"}],
            "speaker_info": {"speakers": ["SPEAKER_00"]},
            "metadata": {"processing_time": 0.5},
        },
        {
            "text": "I'm testing",
            "language": "en",
            "confidence": 0.96,
            "segments": [{"start": 6.0, "end": 8.0, "text": "I'm testing"}],
            "speaker_info": {"speakers": ["SPEAKER_01"]},  # Different speaker!
            "metadata": {"processing_time": 0.7},
        },
        {
            "text": "the streaming system.",
            "language": "en",
            "confidence": 0.95,
            "segments": [{"start": 8.0, "end": 10.0, "text": "the streaming system."}],
            "speaker_info": {"speakers": ["SPEAKER_01"]},
            "metadata": {"processing_time": 0.5},
        },
    ]


@pytest.mark.skip(
    reason="Tests use AsyncMock/patch (violates NO MOCKING policy) and concurrent TestClient "
    "causes deadlocks. Needs rewrite to use real services with testcontainers."
)
class TestStreamingSimulation:
    """Test suite simulating real-world streaming scenarios."""

    @pytest.mark.asyncio
    async def test_sequential_chunk_streaming(
        self, streaming_audio_chunks, mock_progressive_whisper_responses
    ):
        """
        Simulate real streaming: Send multiple chunks sequentially and verify each is processed.

        This is the PRIMARY streaming use case from the frontend.
        """
        from src.main_fastapi import app

        chunks, _sample_rate = streaming_audio_chunks

        with patch("src.routers.audio.audio_core.get_audio_coordinator") as mock_get_coordinator:
            mock_coordinator = AsyncMock()

            # Create side effects for progressive responses
            chunk_results = []
            for whisper_resp in mock_progressive_whisper_responses:
                chunk_results.append(
                    {
                        "status": "processed",
                        "transcription": whisper_resp["text"],
                        "language": whisper_resp["language"],
                        "confidence": whisper_resp["confidence"],
                        "segments": whisper_resp["segments"],
                        "speakers": whisper_resp["speaker_info"]["speakers"],
                        "processing_time": whisper_resp["metadata"]["processing_time"],
                        "duration": 2.0,
                    }
                )

            mock_coordinator.process_audio_file = AsyncMock(side_effect=chunk_results)
            mock_get_coordinator.return_value = mock_coordinator

            client = TestClient(app)
            session_id = "streaming_simulation_test"

            # Simulate frontend streaming behavior
            results = []
            total_processing_time = 0.0

            for i, chunk_audio in enumerate(chunks):
                # Simulate delay between chunks (like real streaming)
                if i > 0:
                    await asyncio.sleep(0.1)  # Small delay for test speed

                response = client.post(
                    "/api/audio/upload",
                    files={"audio": (f"chunk_{i}.webm", chunk_audio, "audio/wav")},
                    data={
                        "session_id": session_id,
                        "chunk_id": f"chunk_{i}_{session_id}",
                        "enable_transcription": "true",
                        "enable_translation": "false",
                        "enable_diarization": "true",
                        "whisper_model": "whisper-base",
                    },
                )

                assert (
                    response.status_code == 200
                ), f"Chunk {i} failed with status {response.status_code}: {response.text}"

                result = response.json()
                processing_result = result["processing_result"]

                # Verify each chunk is processed correctly
                assert processing_result["status"] == "processed"
                assert (
                    processing_result["transcription"]
                    == mock_progressive_whisper_responses[i]["text"]
                )
                assert processing_result["confidence"] > 0.9

                results.append(processing_result)
                total_processing_time += processing_result["processing_time"]

                print(
                    f"✅ Chunk {i}: '{processing_result['transcription']}' "
                    f"(conf={processing_result['confidence']:.2f}, "
                    f"time={processing_result['processing_time']:.2f}s)"
                )

            # Verify all chunks processed
            assert len(results) == 5, f"Expected 5 chunks, got {len(results)}"
            assert mock_coordinator.process_audio_file.call_count == 5

            # Reconstruct full transcription
            full_text = " ".join(r["transcription"] for r in results)
            expected_text = "Hello, how are you doing today? I'm testing the streaming system."

            assert full_text == expected_text, f"Expected: '{expected_text}'\nGot: '{full_text}'"

            # Verify speaker changes were detected
            speakers = [r["speakers"][0] for r in results]
            assert speakers[0] == "SPEAKER_00"
            assert speakers[3] == "SPEAKER_01"  # Speaker changed at chunk 3

            print("\n✅ STREAMING SIMULATION PASSED!")
            print(f"   Total chunks: {len(results)}")
            print(f"   Full transcription: '{full_text}'")
            print(f"   Total processing time: {total_processing_time:.2f}s")
            print(f"   Speakers detected: {set(speakers)}")

    @pytest.mark.asyncio
    async def test_concurrent_chunk_processing(
        self, streaming_audio_chunks, mock_progressive_whisper_responses
    ):
        """
        Test that multiple chunks can be processed concurrently without interference.

        Simulates scenario where chunks arrive faster than they can be processed.
        """
        from src.main_fastapi import app

        chunks, _sample_rate = streaming_audio_chunks

        with patch("src.routers.audio.audio_core.get_audio_coordinator") as mock_get_coordinator:
            mock_coordinator = AsyncMock()

            # Simulate slower processing (chunks arrive faster than processing)
            async def slow_process(*args, **kwargs):
                await asyncio.sleep(0.2)  # Simulate processing delay
                call_count = len(slow_process.calls)
                slow_process.calls.append(True)
                return {
                    "status": "processed",
                    "transcription": mock_progressive_whisper_responses[call_count]["text"],
                    "processing_time": 0.2,
                }

            slow_process.calls = []
            mock_coordinator.process_audio_file = slow_process
            mock_get_coordinator.return_value = mock_coordinator

            client = TestClient(app)
            session_id = "concurrent_streaming_test"

            # Send multiple chunks rapidly (simulating fast streaming)
            import concurrent.futures

            def send_chunk(i, chunk_audio):
                response = client.post(
                    "/api/audio/upload",
                    files={"audio": (f"chunk_{i}.webm", chunk_audio, "audio/wav")},
                    data={
                        "session_id": session_id,
                        "chunk_id": f"chunk_{i}",
                        "enable_transcription": "true",
                    },
                )
                return i, response

            # Send all chunks concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(send_chunk, i, chunk) for i, chunk in enumerate(chunks)]

                responses = [future.result() for future in concurrent.futures.as_completed(futures)]

            # Verify all processed successfully
            responses.sort(key=lambda x: x[0])  # Sort by chunk index

            for i, response in responses:
                assert response.status_code == 200, f"Concurrent chunk {i} failed: {response.text}"

            print("\n✅ CONCURRENT PROCESSING PASSED!")
            print(f"   Processed {len(responses)} chunks concurrently")
            print("   All chunks completed successfully")

    @pytest.mark.asyncio
    async def test_streaming_with_translations(
        self, streaming_audio_chunks, mock_progressive_whisper_responses
    ):
        """
        Test streaming with concurrent translations for each chunk.

        Simulates real use case where user wants real-time translated captions.
        """
        from src.main_fastapi import app

        chunks, _sample_rate = streaming_audio_chunks

        # Mock translation responses
        mock_translations = {
            "es": [
                "Hola,",
                "¿cómo estás",
                "hoy?",
                "Estoy probando",
                "el sistema de streaming.",
            ],
            "fr": [
                "Bonjour,",
                "comment allez-vous",
                "aujourd'hui?",
                "Je teste",
                "le système de streaming.",
            ],
        }

        with patch("src.routers.audio.audio_core.get_audio_coordinator") as mock_get_coordinator:
            mock_coordinator = AsyncMock()

            # Create results with translations
            chunk_results = []
            for i, whisper_resp in enumerate(mock_progressive_whisper_responses):
                chunk_results.append(
                    {
                        "status": "processed",
                        "transcription": whisper_resp["text"],
                        "language": "en",
                        "confidence": whisper_resp["confidence"],
                        "translations": {
                            "es": {
                                "text": mock_translations["es"][i],
                                "confidence": 0.92,
                            },
                            "fr": {
                                "text": mock_translations["fr"][i],
                                "confidence": 0.91,
                            },
                        },
                        "processing_time": 0.8,  # Longer with translations
                    }
                )

            mock_coordinator.process_audio_file = AsyncMock(side_effect=chunk_results)
            mock_get_coordinator.return_value = mock_coordinator

            client = TestClient(app)
            session_id = "streaming_with_translations"

            results = []
            for i, chunk_audio in enumerate(chunks):
                response = client.post(
                    "/api/audio/upload",
                    files={"audio": (f"chunk_{i}.webm", chunk_audio, "audio/wav")},
                    data={
                        "session_id": session_id,
                        "chunk_id": f"chunk_{i}",
                        "enable_transcription": "true",
                        "enable_translation": "true",
                        "target_languages": '["es", "fr"]',
                        "whisper_model": "whisper-base",
                    },
                )

                assert response.status_code == 200
                result = response.json()
                processing_result = result["processing_result"]

                # Verify translations present
                assert "translations" in processing_result
                assert "es" in processing_result["translations"]
                assert "fr" in processing_result["translations"]

                results.append(processing_result)

                print(f"✅ Chunk {i}:")
                print(f"   EN: {processing_result['transcription']}")
                print(f"   ES: {processing_result['translations']['es']['text']}")
                print(f"   FR: {processing_result['translations']['fr']['text']}")

            # Reconstruct full translations
            full_es = " ".join(r["translations"]["es"]["text"] for r in results)
            full_fr = " ".join(r["translations"]["fr"]["text"] for r in results)

            expected_es = "Hola, ¿cómo estás hoy? Estoy probando el sistema de streaming."
            expected_fr = (
                "Bonjour, comment allez-vous aujourd'hui? Je teste le système de streaming."
            )

            assert full_es == expected_es
            assert full_fr == expected_fr

            print("\n✅ STREAMING WITH TRANSLATIONS PASSED!")
            print(f"   Spanish: '{full_es}'")
            print(f"   French: '{full_fr}'")

    @pytest.mark.asyncio
    async def test_streaming_error_recovery(
        self, streaming_audio_chunks, mock_progressive_whisper_responses
    ):
        """
        Test that streaming continues after individual chunk failures.

        Verifies resilience: one chunk failing doesn't break the entire stream.
        """
        from src.main_fastapi import app

        chunks, _sample_rate = streaming_audio_chunks

        with patch("src.routers.audio.audio_core.get_audio_coordinator") as mock_get_coordinator:
            mock_coordinator = AsyncMock()

            # Simulate chunk 2 failing
            def process_with_failure(*args, **kwargs):
                call_count = len(process_with_failure.calls)
                process_with_failure.calls.append(True)

                if call_count == 2:  # Chunk 2 fails
                    return {
                        "status": "error",
                        "error": "Transcription service unavailable",
                        "processing_time": 0.1,
                    }
                else:
                    return {
                        "status": "processed",
                        "transcription": mock_progressive_whisper_responses[call_count]["text"],
                        "processing_time": 0.5,
                    }

            process_with_failure.calls = []
            mock_coordinator.process_audio_file = process_with_failure
            mock_get_coordinator.return_value = mock_coordinator

            client = TestClient(app)
            session_id = "streaming_error_recovery"

            results = []
            for i, chunk_audio in enumerate(chunks):
                response = client.post(
                    "/api/audio/upload",
                    files={"audio": (f"chunk_{i}.webm", chunk_audio, "audio/wav")},
                    data={
                        "session_id": session_id,
                        "chunk_id": f"chunk_{i}",
                        "enable_transcription": "true",
                    },
                )

                assert response.status_code == 200  # Even errors return 200 with error in body
                result = response.json()
                processing_result = result["processing_result"]

                if processing_result["status"] == "error":
                    print(f"⚠️  Chunk {i}: ERROR - {processing_result.get('error')}")
                else:
                    print(f"✅ Chunk {i}: '{processing_result['transcription']}'")

                results.append(processing_result)

            # Verify error was handled gracefully
            assert len(results) == 5
            assert results[2]["status"] == "error"  # Chunk 2 failed
            assert results[0]["status"] == "processed"  # Others succeeded
            assert results[1]["status"] == "processed"
            assert results[3]["status"] == "processed"
            assert results[4]["status"] == "processed"

            # Reconstruct partial transcription (skipping failed chunk)
            successful_transcripts = [
                r["transcription"] for r in results if r["status"] == "processed"
            ]

            assert len(successful_transcripts) == 4  # 4 out of 5 succeeded

            print("\n✅ ERROR RECOVERY PASSED!")
            print(f"   Total chunks: {len(results)}")
            print(f"   Successful: {len(successful_transcripts)}")
            print("   Failed: 1 (chunk 2)")
            print("   System continued streaming after failure")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
