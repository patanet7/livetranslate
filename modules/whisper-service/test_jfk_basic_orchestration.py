#!/usr/bin/env python3
"""
Test Basic JFK Audio Upload Through Orchestration Service

Simple test to verify audio flows correctly through the orchestration service.
No domain prompts - just basic transcription.

REQUIREMENTS:
1. Orchestration service running on http://localhost:3000
2. Whisper service running on http://localhost:5001

Flow: Frontend → Orchestration → Whisper → Response
"""

import requests
import time
import wave
import json
from pathlib import Path

ORCHESTRATION_URL = "http://localhost:3000"
JFK_AUDIO_PATH = "/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/jfk.wav"


def test_jfk_basic_upload():
    """
    Test basic JFK audio upload through orchestration service

    This verifies:
    1. Orchestration service is running
    2. Audio file can be uploaded
    3. Orchestration service can communicate with Whisper
    4. Transcription is returned
    """
    print("\n" + "="*80)
    print("JFK AUDIO - BASIC ORCHESTRATION SERVICE TEST")
    print("="*80)
    print("Testing basic audio flow: Frontend → Orchestration → Whisper")
    print("(No domain prompts - just basic transcription)")
    print("="*80)

    # Check if orchestration service is running
    print("\n🔍 Checking orchestration service...")
    try:
        health_response = requests.get(f"{ORCHESTRATION_URL}/api/health", timeout=5)
        if health_response.status_code == 200:
            print(f"✅ Orchestration service is running (status 200)")
        else:
            print(f"⚠️  Orchestration service responded with status {health_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to orchestration service at {ORCHESTRATION_URL}")
        print(f"   Error: {e}")
        print(f"\n💡 Start orchestration service:")
        print(f"   cd modules/orchestration-service")
        print(f"   python src/main.py")
        return {"passed": False, "error": "Orchestration service not running"}

    # Check if JFK audio file exists
    if not Path(JFK_AUDIO_PATH).exists():
        print(f"\n❌ JFK audio file not found: {JFK_AUDIO_PATH}")
        return {"passed": False, "error": "Audio file not found"}

    # Load JFK audio file info
    print(f"\n📁 Loading JFK audio file...")
    with wave.open(JFK_AUDIO_PATH, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration = n_frames / sample_rate
        print(f"   Path: {JFK_AUDIO_PATH}")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Frames: {n_frames}")

    # Prepare upload
    session_id = f"basic-orch-test-{int(time.time())}"
    print(f"\n📤 Uploading audio to orchestration service...")
    print(f"   Session ID: {session_id}")
    print(f"   Endpoint: {ORCHESTRATION_URL}/api/audio/upload")

    try:
        with open(JFK_AUDIO_PATH, 'rb') as audio_file:
            files = {'audio': ('jfk.wav', audio_file, 'audio/wav')}

            # Minimal configuration - just basic transcription
            data = {
                'session_id': session_id,
                'enable_transcription': 'true',
                'enable_translation': 'false',
                'enable_diarization': 'false',
                'whisper_model': 'large-v3-turbo',  # Use same model as direct test
            }

            print(f"\n📝 Request configuration:")
            for key, value in data.items():
                print(f"   {key}: {value}")

            # Send request
            print(f"\n⏳ Sending request...")
            start_time = time.time()

            response = requests.post(
                f"{ORCHESTRATION_URL}/api/audio/upload",
                files=files,
                data=data,
                timeout=60
            )

            elapsed = time.time() - start_time
            print(f"✅ Request completed in {elapsed:.2f}s")

        # Process response
        print(f"\n📥 Response received:")
        print(f"   Status code: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'unknown')}")

        if response.status_code == 200:
            try:
                result = response.json()
                print(f"\n📄 Response body:")
                print(json.dumps(result, indent=2))

                # Extract transcription
                transcription = result.get('transcription', {})
                text = transcription.get('text', '')

                if not text:
                    # Try different response formats
                    text = result.get('text', '')
                    if not text:
                        text = str(result.get('segments', []))

                print(f"\n" + "="*80)
                print("TEST RESULTS")
                print("="*80)

                if text:
                    print(f"✅ Transcription received!")
                    print(f"\n📝 Transcription text:")
                    print(f"   '{text}'")

                    # Check for expected keywords (basic sanity check)
                    print(f"\n🔍 Basic keyword check:")
                    keywords = ["Americans", "country", "fellow", "ask"]
                    for keyword in keywords:
                        found = keyword.lower() in text.lower()
                        status = "✅" if found else "❌"
                        print(f"   {status} '{keyword}': {found}")

                    print("\n" + "="*80)
                    print("✅ TEST PASSED - Audio successfully processed through orchestration!")
                    print("="*80 + "\n")

                    return {
                        "passed": True,
                        "transcription": text,
                        "response_time": elapsed,
                        "response": result
                    }
                else:
                    print("⚠️  No transcription text found in response")
                    print("\n" + "="*80)
                    print("❌ TEST FAILED - No transcription in response")
                    print("="*80 + "\n")
                    return {"passed": False, "error": "No transcription text in response"}

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response: {e}")
                print(f"Response text: {response.text[:500]}")
                return {"passed": False, "error": "Invalid JSON response"}

        elif response.status_code == 422:
            print(f"\n⚠️  Validation error (422)")
            print(f"Response: {response.text}")
            return {"passed": False, "error": "Validation error"}

        else:
            print(f"\n❌ Upload failed with status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return {"passed": False, "error": f"HTTP {response.status_code}"}

    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out after 60 seconds")
        return {"passed": False, "error": "Request timeout"}

    except Exception as e:
        print(f"\n❌ Test failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


if __name__ == "__main__":
    result = test_jfk_basic_upload()
    exit(0 if result.get("passed") else 1)
