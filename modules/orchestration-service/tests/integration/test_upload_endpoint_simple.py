"""
Simple Integration Test - Verify Upload Endpoint Works

Tests the actual endpoint with minimal mocking to verify:
1. Endpoint accepts uploads
2. No placeholder responses
3. Real processing flow is called
"""

import pytest
import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock


def test_upload_endpoint_accepts_audio_and_returns_no_placeholder():
    """
    CRITICAL TEST: Verify upload endpoint returns real processing results.

    This is a simple end-to-end test that verifies the basic flow works.
    """
    from src.main_fastapi import app
    from src.dependencies import get_audio_coordinator

    # Create test audio
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    test_audio_wav = buffer.read()

    # Don't override - use real audio coordinator and whisper service
    try:
        client = TestClient(app)

        # Make request
        response = client.post(
            "/api/audio/upload",
            files={"audio": ("test.wav", test_audio_wav, "audio/wav")},
            data={
                "session_id": "test_simple",
                "enable_transcription": "true",
                "whisper_model": "whisper-tiny",
            },
        )

        # Verify response
        assert response.status_code == 200, f"Failed: {response.text}"

        result = response.json()
        processing_result = result.get("processing_result", {})

        # CRITICAL: Verify NO placeholder
        transcription = processing_result.get("transcription", "")
        assert "placeholder" not in transcription.lower(), (
            "REGRESSION: Found placeholder in response!"
        )

        # Verify we got real text (not empty, actual Whisper transcription)
        assert transcription is not None and len(transcription) > 0, (
            f"Expected real transcription, got empty or None"
        )

        # Verify processing status
        assert (
            result.get("status") == "processed"
            or processing_result.get("status") == "processed"
        ), f"Expected status 'processed', got: {result.get('status')}"

        print("✅ TEST PASSED: Upload endpoint works with real Whisper processing!")
        print(f"   Transcription: '{transcription}'")
        print(f"   Status: {processing_result.get('status')}")
        print(f"   Language: {processing_result.get('language')}")
        print(f"   Confidence: {processing_result.get('confidence')}")
        print(f"   Duration: {processing_result.get('duration')}")

    finally:
        pass  # No cleanup needed - using real services


if __name__ == "__main__":
    test_upload_endpoint_accepts_audio_and_returns_no_placeholder()
    print("\n🎉 All tests passed!")
