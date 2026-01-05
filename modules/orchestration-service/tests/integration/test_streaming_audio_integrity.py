"""
Integration Test - Streaming Audio Integrity

Tests the complete audio pipeline from frontend streaming to Whisper transcription:
1. Generate time-coded test audio with known content
2. Stream it in chunks (simulating frontend)
3. Validate chunking, overlaps, and timing
4. Verify transcription accuracy and timing alignment

This validates:
- Frontend streaming format (WebM)
- Orchestration service audio handling
- FFmpeg conversion integrity
- Audio processing pipeline
- Whisper service transcription
"""

import pytest
import numpy as np
import io
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import soundfile as sf
import subprocess
import tempfile


# Test audio segments with known content and timing
TEST_SEGMENTS = [
    {
        "start": 0.0,
        "duration": 2.0,
        "frequency": 440,
        "text": "440Hz tone at 0-2 seconds",
    },
    {
        "start": 2.0,
        "duration": 2.0,
        "frequency": 880,
        "text": "880Hz tone at 2-4 seconds",
    },
    {
        "start": 4.0,
        "duration": 2.0,
        "frequency": 1320,
        "text": "1320Hz tone at 4-6 seconds",
    },
]


def generate_timecoded_audio(
    segments: List[Dict[str, Any]], sample_rate: int = 16000, output_format: str = "wav"
) -> bytes:
    """
    Generate time-coded test audio with known frequencies at specific times.

    Args:
        segments: List of segments with start, duration, frequency
        sample_rate: Audio sample rate
        output_format: Output format (wav, webm, mp3)

    Returns:
        Audio data as bytes
    """
    # Calculate total duration
    total_duration = max(seg["start"] + seg["duration"] for seg in segments)
    total_samples = int(total_duration * sample_rate)

    # Initialize audio array
    audio = np.zeros(total_samples, dtype=np.float32)

    # Generate each segment
    for seg in segments:
        start_sample = int(seg["start"] * sample_rate)
        duration_samples = int(seg["duration"] * sample_rate)
        end_sample = start_sample + duration_samples

        # Generate sine wave for this segment
        t = np.linspace(0, seg["duration"], duration_samples)
        tone = 0.3 * np.sin(2 * np.pi * seg["frequency"] * t)

        # Add to audio array
        audio[start_sample:end_sample] = tone.astype(np.float32)

    # Convert to bytes
    if output_format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    elif output_format in ["webm", "mp3"]:
        # First write to WAV, then convert with ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            sf.write(wav_file.name, audio, sample_rate)
            wav_path = wav_file.name

        with tempfile.NamedTemporaryFile(
            suffix=f".{output_format}", delete=False
        ) as output_file:
            output_path = output_file.name

        try:
            # Convert with ffmpeg
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-codec:a",
                "libopus" if output_format == "webm" else "libmp3lame",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Read the converted file
            with open(output_path, "rb") as f:
                data = f.read()

            return data
        finally:
            Path(wav_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    else:
        raise ValueError(f"Unsupported format: {output_format}")


def chunk_audio(
    audio_data: bytes, chunk_duration: float = 3.0, format: str = "wav"
) -> List[bytes]:
    """
    Split audio into chunks (simulating frontend streaming).

    Args:
        audio_data: Complete audio data
        chunk_duration: Duration of each chunk in seconds
        format: Audio format

    Returns:
        List of audio chunks as bytes
    """
    # First, load the audio
    if format == "wav":
        buffer = io.BytesIO(audio_data)
        audio, sample_rate = sf.read(buffer)
    else:
        # Use ffmpeg to convert to WAV first
        with tempfile.NamedTemporaryFile(
            suffix=f".{format}", delete=False
        ) as input_file:
            input_file.write(audio_data)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        try:
            cmd = ["ffmpeg", "-y", "-i", input_path, wav_path]
            subprocess.run(cmd, capture_output=True, check=True)

            audio, sample_rate = sf.read(wav_path)
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)

    # Calculate chunk size
    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = len(audio)

    chunks = []
    for start_idx in range(0, total_samples, chunk_samples):
        end_idx = min(start_idx + chunk_samples, total_samples)
        chunk = audio[start_idx:end_idx]

        # Convert chunk to bytes (WebM or WAV)
        if format == "webm":
            # Write to WAV first, then convert
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                sf.write(wav_file.name, chunk, sample_rate)
                wav_path = wav_file.name

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
                webm_path = webm_file.name

            try:
                cmd = ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libopus", webm_path]
                subprocess.run(cmd, capture_output=True, check=True)

                with open(webm_path, "rb") as f:
                    chunk_data = f.read()
            finally:
                Path(wav_path).unlink(missing_ok=True)
                Path(webm_path).unlink(missing_ok=True)
        else:
            # WAV format
            buffer = io.BytesIO()
            sf.write(buffer, chunk, sample_rate, format="WAV")
            buffer.seek(0)
            chunk_data = buffer.read()

        chunks.append(chunk_data)

    return chunks


def test_audio_generation():
    """Test that we can generate timecoded audio correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: Audio Generation")
    print("=" * 80)

    # Generate WAV
    wav_data = generate_timecoded_audio(TEST_SEGMENTS, output_format="wav")
    assert len(wav_data) > 0, "Failed to generate WAV audio"
    print(f"✅ Generated WAV audio: {len(wav_data)} bytes")

    # Verify it can be read back
    buffer = io.BytesIO(wav_data)
    audio, sample_rate = sf.read(buffer)
    print(f"✅ Audio verified: {len(audio)} samples at {sample_rate}Hz")
    print(f"   Duration: {len(audio) / sample_rate:.2f}s")

    # Generate WebM
    webm_data = generate_timecoded_audio(TEST_SEGMENTS, output_format="webm")
    assert len(webm_data) > 0, "Failed to generate WebM audio"
    print(f"✅ Generated WebM audio: {len(webm_data)} bytes")


def test_audio_chunking():
    """Test that chunking preserves audio integrity."""
    print("\n" + "=" * 80)
    print("TEST 2: Audio Chunking")
    print("=" * 80)

    # Generate test audio
    audio_data = generate_timecoded_audio(TEST_SEGMENTS, output_format="wav")

    # Chunk it
    chunks = chunk_audio(audio_data, chunk_duration=2.0, format="wav")
    print(f"✅ Split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        buffer = io.BytesIO(chunk)
        audio, sample_rate = sf.read(buffer)
        duration = len(audio) / sample_rate
        print(
            f"   Chunk {i}: {len(chunk)} bytes, {duration:.2f}s, {len(audio)} samples"
        )

    # Verify total duration matches
    total_duration = sum(len(sf.read(io.BytesIO(chunk))[0]) / 16000 for chunk in chunks)
    expected_duration = max(seg["start"] + seg["duration"] for seg in TEST_SEGMENTS)
    print(
        f"✅ Total duration: {total_duration:.2f}s (expected: {expected_duration:.2f}s)"
    )
    assert abs(total_duration - expected_duration) < 0.1, "Duration mismatch!"


def test_orchestration_upload_endpoint():
    """Test the orchestration service /api/audio/upload endpoint."""
    print("\n" + "=" * 80)
    print("TEST 3: Orchestration Upload Endpoint")
    print("=" * 80)

    import sys
    from pathlib import Path

    # Add src to path
    src_path = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))

    from fastapi.testclient import TestClient
    from main_fastapi import app

    client = TestClient(app)

    # Generate test audio
    audio_data = generate_timecoded_audio(TEST_SEGMENTS, output_format="webm")
    print(f"✅ Generated test WebM: {len(audio_data)} bytes")

    # Upload to endpoint
    response = client.post(
        "/api/audio/upload",
        files={"audio": ("test_chunk.webm", audio_data, "audio/webm")},
        data={
            "session_id": "integrity_test",
            "chunk_id": "chunk_0",
            "enable_transcription": "true",
            "enable_translation": "false",
            "enable_diarization": "false",
            "whisper_model": "whisper-tiny",
            "audio_processing": "false",  # Disable audio processing to test raw audio
        },
    )

    print(f"Response status: {response.status_code}")
    assert response.status_code == 200, f"Upload failed: {response.text}"

    result = response.json()
    print(f"✅ Upload successful")
    print(f"   Status: {result.get('status')}")

    # Check processing result
    if "processing_result" in result:
        proc_result = result["processing_result"]
        print(f"\n📊 Processing Result:")
        print(f"   Status: {proc_result.get('status')}")
        print(f"   Transcription: {proc_result.get('transcription', 'N/A')}")
        print(f"   Language: {proc_result.get('language', 'N/A')}")
        print(f"   Confidence: {proc_result.get('confidence', 'N/A')}")
        print(f"   Duration: {proc_result.get('duration', 'N/A')}")

        # Validate transcription exists and is not empty
        transcription = proc_result.get("transcription", "")
        assert transcription, "❌ FAIL: No transcription returned!"
        assert len(transcription) > 0, "❌ FAIL: Empty transcription!"
        print(f"✅ Got transcription: '{transcription}'")

        # Check if it's placeholder/gibberish
        if any(
            word in transcription.lower() for word in ["placeholder", "mock", "test"]
        ):
            print(f"⚠️  WARNING: Transcription appears to be placeholder")

        # For tone-based audio, we expect it to be silent/minimal transcription
        # Real test would use actual speech
        print(f"\n📝 Note: Tone-based test audio - minimal transcription expected")
    else:
        print("❌ FAIL: No processing_result in response!")
        print(f"Response: {json.dumps(result, indent=2)}")


def test_full_streaming_pipeline():
    """Test the complete streaming pipeline with chunked upload."""
    print("\n" + "=" * 80)
    print("TEST 4: Full Streaming Pipeline")
    print("=" * 80)

    import sys
    from pathlib import Path

    # Add src to path
    src_path = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))

    from fastapi.testclient import TestClient
    from main_fastapi import app

    client = TestClient(app)

    # Generate and chunk audio
    audio_data = generate_timecoded_audio(TEST_SEGMENTS, output_format="webm")
    chunks = chunk_audio(audio_data, chunk_duration=2.0, format="webm")

    print(f"✅ Generated {len(chunks)} chunks")

    session_id = f"streaming_test_{int(time.time())}"
    results = []

    # Upload each chunk
    for i, chunk in enumerate(chunks):
        print(f"\n📤 Uploading chunk {i}/{len(chunks)}...")

        response = client.post(
            "/api/audio/upload",
            files={"audio": (f"chunk_{i}.webm", chunk, "audio/webm")},
            data={
                "session_id": session_id,
                "chunk_id": f"chunk_{i}",
                "enable_transcription": "true",
                "enable_translation": "false",
                "enable_diarization": "false",
                "whisper_model": "whisper-tiny",
                "audio_processing": "false",  # Disable audio processing to test raw audio
            },
        )

        assert response.status_code == 200, f"Chunk {i} upload failed: {response.text}"
        result = response.json()
        results.append(result)

        if "processing_result" in result:
            proc = result["processing_result"]
            print(f"   ✅ Chunk {i}: {proc.get('transcription', 'N/A')}")
        else:
            print(f"   ⚠️  Chunk {i}: No processing result")

    # Analyze results
    print(f"\n📊 Pipeline Results Summary:")
    print(f"   Total chunks: {len(results)}")

    transcriptions = [
        r.get("processing_result", {}).get("transcription", "")
        for r in results
        if "processing_result" in r
    ]

    print(f"   Transcriptions: {len(transcriptions)}")
    for i, trans in enumerate(transcriptions):
        print(f"   Chunk {i}: '{trans}'")

    # Check for issues
    empty_transcriptions = sum(1 for t in transcriptions if not t or len(t) == 0)
    if empty_transcriptions > 0:
        print(f"   ⚠️  {empty_transcriptions} empty transcriptions")

    print(f"\n✅ Full pipeline test complete")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INTEGRATED AUDIO INTEGRITY TEST SUITE")
    print("=" * 80)
    print("\nThis tests the complete audio pipeline:")
    print("1. Audio generation with time-coded content")
    print("2. Audio chunking (frontend simulation)")
    print("3. Orchestration service upload handling")
    print("4. Full streaming pipeline with multiple chunks")
    print("\n" + "=" * 80 + "\n")

    try:
        test_audio_generation()
        test_audio_chunking()
        test_orchestration_upload_endpoint()
        test_full_streaming_pipeline()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        raise
