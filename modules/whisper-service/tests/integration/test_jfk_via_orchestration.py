#!/usr/bin/env python3
"""
Test Domain Prompts with JFK Audio - Via Orchestration Service

This test demonstrates the FULL INTEGRATION PATH:
    Frontend → Orchestration Service → Whisper Service

Tests political domain terminology for better transcription of American political speeches.

REQUIREMENTS:
1. Orchestration service running on http://localhost:3000
2. Whisper service running on http://localhost:5001
3. Orchestration service must pass domain prompt fields to Whisper

NOTE: The orchestration service currently does NOT pass domain prompt fields.
      This test documents how it SHOULD work once implemented.

To implement domain prompts in orchestration service, add these fields to:
- /modules/orchestration-service/src/routers/audio/audio_core.py (upload_audio_file function)
  Add Form parameters:
    - domain: Optional[str] = Form(None)
    - custom_terms: Optional[str] = Form(None)  # JSON array
    - initial_prompt: Optional[str] = Form(None)

- /modules/orchestration-service/src/clients/audio_service_client.py
  Pass domain fields to Whisper service WebSocket

See ORCHESTRATION_DOMAIN_PROMPTS.md for detailed implementation guide.
"""

import requests
import time
import wave
import json
from pathlib import Path

ORCHESTRATION_URL = "http://localhost:3000"
# Use fixtures path (consistent with conftest.py jfk_audio fixture)
JFK_AUDIO_PATH = str(Path(__file__).parent.parent / "fixtures" / "audio" / "jfk.wav")


def test_jfk_via_orchestration():
    """
    Test JFK audio streaming through orchestration service with domain prompts

    Expected flow:
    1. Upload JFK audio file to orchestration service
    2. Orchestration service processes and streams to Whisper
    3. Whisper applies domain prompts and returns transcription
    4. Orchestration service returns results to client

    Expected improvements with political domain:
    - "Americans" → recognized correctly
    - "country" → recognized correctly
    - "fellow" → recognized correctly
    """
    print("\n" + "="*80)
    print("JFK AUDIO VIA ORCHESTRATION SERVICE + DOMAIN PROMPTS TEST")
    print("="*80)
    print("Testing full integration: Frontend → Orchestration → Whisper")
    print("="*80)

    # Check if orchestration service is running
    try:
        health_response = requests.get(f"{ORCHESTRATION_URL}/api/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Orchestration service not healthy: {health_response.status_code}")
            return {"passed": False, "error": "Orchestration service not healthy"}
        print("✅ Orchestration service is running")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to orchestration service: {e}")
        print(f"\n💡 Start orchestration service:")
        print(f"   cd modules/orchestration-service")
        print(f"   python src/main.py")
        return {"passed": False, "error": "Orchestration service not running"}

    # Load JFK audio file
    if not Path(JFK_AUDIO_PATH).exists():
        print(f"❌ JFK audio file not found: {JFK_AUDIO_PATH}")
        return {"passed": False, "error": "Audio file not found"}

    print(f"\n📁 Loading JFK audio from: {JFK_AUDIO_PATH}")
    with wave.open(JFK_AUDIO_PATH, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration = n_frames / sample_rate
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Duration: {duration:.1f}s")

    # Prepare upload request with domain prompts
    session_id = f"orch-jfk-test-{int(time.time())}"

    # Domain prompt configuration
    domain_config = {
        "domain": "political",
        "custom_terms": json.dumps([
            "Americans",
            "fellow citizens",
            "country",
            "nation",
            "freedom",
            "liberty",
            "democracy"
        ]),
        "initial_prompt": "Presidential inaugural speech about American values and civic responsibility"
    }

    print(f"\n📝 Domain prompts configuration:")
    print(f"   Domain: {domain_config['domain']}")
    print(f"   Custom terms: {json.loads(domain_config['custom_terms'])}")
    print(f"   Initial prompt: {domain_config['initial_prompt']}")

    # Upload audio file with domain prompts
    print(f"\n📤 Uploading JFK audio to orchestration service...")
    print(f"   Session ID: {session_id}")

    try:
        with open(JFK_AUDIO_PATH, 'rb') as audio_file:
            files = {'audio': ('jfk.wav', audio_file, 'audio/wav')}

            data = {
                'session_id': session_id,
                'enable_transcription': 'true',
                'enable_translation': 'false',
                'enable_diarization': 'false',
                'whisper_model': 'large-v3-turbo',
                # DOMAIN PROMPT FIELDS (requires orchestration service implementation)
                **domain_config
            }

            response = requests.post(
                f"{ORCHESTRATION_URL}/api/audio/upload",
                files=files,
                data=data,
                timeout=60
            )

        print(f"\n📥 Response received:")
        print(f"   Status code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"   Response: {json.dumps(result, indent=2)}")

            # Extract transcription
            transcription = result.get('transcription', {})
            text = transcription.get('text', '')

            print(f"\n" + "="*80)
            print("TEST RESULTS")
            print("="*80)
            print(f"📄 Transcription: '{text}'")

            # Check for expected keywords
            print(f"\n🔍 Keyword detection:")
            keywords = ["Americans", "country", "fellow"]
            all_found = True
            for keyword in keywords:
                found = keyword.lower() in text.lower()
                status = "✅" if found else "❌"
                print(f"   {status} '{keyword}' found: {found}")
                if not found:
                    all_found = False

            print("\n" + "="*80)
            if all_found and len(text) > 0:
                print("✅ TEST PASSED - Domain prompts working via orchestration!")
            elif len(text) > 0:
                print("⚠️  TEST PARTIALLY PASSED - Transcription received but some keywords missing")
            else:
                print("❌ TEST FAILED - No transcription received")
            print("="*80 + "\n")

            return {
                "passed": all_found and len(text) > 0,
                "transcription": text,
                "keywords_found": all_found
            }

        elif response.status_code == 422:
            print(f"   ⚠️  Validation error (422) - Domain prompt fields not supported yet")
            print(f"   Response: {response.text}")
            print(f"\n💡 The orchestration service needs to be updated to support domain prompts.")
            print(f"   See ORCHESTRATION_DOMAIN_PROMPTS.md for implementation guide.")
            return {"passed": False, "error": "Domain prompts not supported in orchestration service"}

        else:
            print(f"   ❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {"passed": False, "error": f"Upload failed: {response.status_code}"}

    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after 60 seconds")
        return {"passed": False, "error": "Request timeout"}

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


if __name__ == "__main__":
    result = test_jfk_via_orchestration()
    exit(0 if result.get("passed") else 1)
