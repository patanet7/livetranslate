#!/usr/bin/env python3
"""
Test script to create real audio and test transcription through gateway
"""

import io
import time
import wave

import numpy as np
import requests


def create_test_wav_audio(duration=2, sample_rate=16000):
    """Create a test WAV file with a sine wave tone"""
    # Generate a simple sine wave tone
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3

    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.01, len(audio_data))
    audio_data = audio_data + noise

    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    wav_buffer.seek(0)
    return wav_buffer.read()


def test_direct_whisper():
    """Test direct connection to whisper service"""
    print("Testing direct whisper service with real audio...")

    audio_content = create_test_wav_audio()
    print(f"Created test WAV audio: {len(audio_content)} bytes")

    files = {"audio": ("test.wav", audio_content, "audio/wav")}

    try:
        response = requests.post(
            "http://localhost:5001/transcribe/whisper-base", files=files, timeout=30
        )
        print(f"Direct whisper - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Direct whisper - Text: {data.get('text', 'No text')}")
            print(f"Direct whisper - Device: {data.get('device_used', 'Unknown')}")
        else:
            print(f"Direct whisper - Response: {response.text[:200]}")
    except Exception as e:
        print(f"Direct whisper failed: {e}")


def test_gateway_forwarding():
    """Test forwarding through API gateway"""
    print("\nTesting API gateway forwarding with real audio...")

    audio_content = create_test_wav_audio()
    print(f"Created test WAV audio: {len(audio_content)} bytes")

    files = {"audio": ("test.wav", audio_content, "audio/wav")}

    try:
        response = requests.post(
            "http://localhost:3000/api/whisper/transcribe/whisper-base", files=files, timeout=30
        )
        print(f"Gateway - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Gateway - Text: {data.get('text', 'No text')}")
            print(f"Gateway - Device: {data.get('device_used', 'Unknown')}")
        else:
            print(f"Gateway - Response: {response.text[:200]}")
    except Exception as e:
        print(f"Gateway failed: {e}")


def test_orchestration_audio_endpoint():
    """Test the actual audio endpoint used by frontend"""
    print("\nTesting orchestration audio endpoint...")

    audio_content = create_test_wav_audio()
    print(f"Created test WAV audio: {len(audio_content)} bytes")

    # Test audio.js style request with model in form data
    files = {"audio": ("audio.mp4", audio_content, "audio/mp4")}  # Frontend sends as mp4

    try:
        # This mimics how the frontend calls the API
        response = requests.post(
            "http://localhost:3000/api/whisper/transcribe/whisper-base", files=files, timeout=30
        )
        print(f"Orchestration - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Orchestration - Text: {data.get('text', 'No text')}")
        else:
            print(f"Orchestration - Response: {response.text[:200]}")
    except Exception as e:
        print(f"Orchestration test failed: {e}")


if __name__ == "__main__":
    print("Testing audio transcription with real WAV data...")
    print("=" * 60)

    # Give services time to stabilize
    time.sleep(1)

    test_direct_whisper()
    test_gateway_forwarding()
    test_orchestration_audio_endpoint()

    print("\n" + "=" * 60)
    print("Test completed!")
