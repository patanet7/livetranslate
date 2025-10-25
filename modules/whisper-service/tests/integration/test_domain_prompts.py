#!/usr/bin/env python3
"""
Domain Prompt Integration Test

Demonstrates how the orchestration service can pass domain-specific terminology
to improve transcription accuracy.

Usage examples:
1. Built-in domain: domain="medical"
2. Custom terms: custom_terms=["Kubernetes", "Docker", "microservices"]
3. Custom prompt: initial_prompt="Technical discussion about cloud architecture"
"""

import socketio
import numpy as np
import base64
import time
import wave

SERVICE_URL = "http://localhost:5001"
JFK_AUDIO_PATH = "/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/jfk.wav"


def test_domain_prompt(domain=None, custom_terms=None, initial_prompt=None, test_name="Default"):
    """
    Test domain prompting with various configurations

    Args:
        domain: Built-in domain name ("medical", "legal", "technical", etc.)
        custom_terms: List of custom terminology
        initial_prompt: Custom prompt text
        test_name: Descriptive name for this test
    """
    print("\n" + "="*80)
    print(f"DOMAIN PROMPT TEST - {test_name}")
    print("="*80)
    print(f"Domain: {domain}")
    print(f"Custom Terms: {custom_terms}")
    print(f"Initial Prompt: {initial_prompt}")
    print("="*80)

    # Load JFK audio
    with wave.open(JFK_AUDIO_PATH, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = int(sample_rate * 5.0)  # First 5 seconds
        audio_bytes = wav_file.readframes(n_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0

    print(f"\n📁 Loaded {len(audio)/sample_rate:.1f}s of audio")

    # Connect to Socket.IO
    sio = socketio.Client()
    results = []

    @sio.on('connect')
    def on_connect():
        print("✅ Connected to Socket.IO")

    @sio.on('transcription_result')
    def on_result(data):
        results.append(data)
        text = data.get('text', '')
        is_draft = data.get('is_draft', False)
        is_final = data.get('is_final', False)

        status = "✏️ DRAFT" if is_draft else ("✅ FINAL" if is_final else "📝 UPDATE")
        print(f"\n{status} Result #{len(results)}:")
        print(f"   Text: '{text}'")

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        sio.connect(SERVICE_URL)
        time.sleep(0.5)

        session_id = f"domain-test-{int(time.time())}"
        sio.emit('join_session', {'session_id': session_id})
        time.sleep(0.5)

        print(f"\n🎙️  Starting transcription with domain prompts...")
        print(f"   Session ID: {session_id}")

        # Prepare emission data
        chunk_int16 = (audio * 32768.0).astype(np.int16)
        chunk_b64 = base64.b64encode(chunk_int16.tobytes()).decode('utf-8')

        # Build request with domain prompts
        request_data = {
            "session_id": session_id,
            "audio_data": chunk_b64,
            "model_name": "large-v3-turbo",
            "language": "en",
            "beam_size": 5,
            "sample_rate": sample_rate,
            "task": "transcribe",
            "target_language": "en",
            "enable_vad": True,
        }

        # Add domain prompt fields
        if domain:
            request_data["domain"] = domain
        if custom_terms:
            request_data["custom_terms"] = custom_terms
        if initial_prompt:
            request_data["initial_prompt"] = initial_prompt

        # Send audio with domain prompts
        sio.emit('transcribe_stream', request_data)

        print("\n⏳ Waiting for results...")
        time.sleep(8.0)

        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.3)
        sio.disconnect()

        print(f"\n📊 Test Results:")
        print(f"   Results received: {len(results)}")
        if results:
            full_text = ' '.join([r.get('text', '') for r in results])
            print(f"   Full transcription: '{full_text}'")

        return {"passed": len(results) > 0, "results": results}

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


def main():
    """Run domain prompt tests with different configurations"""
    print("\n" + "="*80)
    print("DOMAIN PROMPT INTEGRATION TESTS")
    print("="*80)

    tests = [
        # Test 1: Built-in medical domain
        {
            "name": "Medical Domain (Built-in)",
            "domain": "medical",
            "custom_terms": None,
            "initial_prompt": None
        },

        # Test 2: Custom technical terms
        {
            "name": "Custom Technical Terms",
            "domain": None,
            "custom_terms": ["Kubernetes", "Docker", "microservices", "API", "CI/CD"],
            "initial_prompt": None
        },

        # Test 3: Custom prompt
        {
            "name": "Custom Initial Prompt",
            "domain": None,
            "custom_terms": None,
            "initial_prompt": "This is a political speech about American values"
        },

        # Test 4: Combined (domain + custom terms)
        {
            "name": "Combined (Medical + Custom)",
            "domain": "medical",
            "custom_terms": ["hypertension", "cardiomyopathy"],
            "initial_prompt": "Medical consultation about cardiovascular health"
        },
    ]

    results_summary = []

    for test_config in tests:
        result = test_domain_prompt(
            domain=test_config["domain"],
            custom_terms=test_config["custom_terms"],
            initial_prompt=test_config["initial_prompt"],
            test_name=test_config["name"]
        )
        results_summary.append((test_config["name"], result))

        # Wait between tests
        time.sleep(3)

    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, result in results_summary:
        status = "✅ PASSED" if result.get("passed") else "❌ FAILED"
        print(f"\n{status} - {test_name}")
        print(f"   Results: {result.get('results', []).__len__()} transcriptions")
        if result.get("results"):
            text = ' '.join([r.get('text', '') for r in result.get('results', [])])
            print(f"   Text: '{text[:80]}'")

    print("\n" + "="*80)
    all_passed = all(r.get("passed") for _, r in results_summary)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
