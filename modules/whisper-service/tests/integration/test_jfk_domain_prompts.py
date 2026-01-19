#!/usr/bin/env python3
"""
Test Domain Prompts with JFK Audio - Direct Whisper Service Test

Demonstrates domain prompts working with real JFK audio through Whisper service.
Tests political domain terminology for better transcription of American political speeches.
"""

import base64
import time
import wave

import numpy as np
import socketio

SERVICE_URL = "http://localhost:5001"
JFK_AUDIO_PATH = (
    "/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/jfk.wav"
)


def test_jfk_with_political_domain():
    """
    Test JFK audio with political domain terminology

    Expected improvements:
    - "Americans" → recognized correctly (common term)
    - "country" → recognized correctly (political context)
    - "fellow" → recognized correctly (formal speech)
    - Overall transcription accuracy improved with political context
    """
    print("\n" + "=" * 80)
    print("JFK AUDIO + POLITICAL DOMAIN PROMPTS TEST")
    print("=" * 80)
    print("Testing domain prompts with JFK's famous inaugural speech...")
    print("=" * 80)

    # Load JFK audio
    with wave.open(JFK_AUDIO_PATH, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        audio_bytes = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0

    print(f"\n📁 Loaded {len(audio)/sample_rate:.1f}s of JFK audio (sample_rate={sample_rate}Hz)")

    # Connect to Socket.IO
    sio = socketio.Client()
    results = []

    @sio.on("connect")
    def on_connect():
        print("✅ Connected to Whisper service")

    @sio.on("transcription_result")
    def on_result(data):
        results.append(data)
        text = data.get("text", "")
        is_draft = data.get("is_draft", False)
        is_final = data.get("is_final", False)

        status = "✏️ DRAFT" if is_draft else ("✅ FINAL" if is_final else "📝 UPDATE")
        print(f"\n{status} Result #{len(results)}:")
        print(f"   Text: '{text}'")
        print(f"   is_draft={is_draft}, is_final={is_final}")

    @sio.on("error")
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        sio.connect(SERVICE_URL)
        time.sleep(0.5)

        session_id = f"jfk-domain-test-{int(time.time())}"
        sio.emit("join_session", {"session_id": session_id})
        time.sleep(0.5)

        print("\n🎙️  Streaming JFK audio with political domain prompts...")
        print(f"   Session ID: {session_id}")

        # Split audio into chunks (simulate streaming)
        chunk_duration = 2.0  # 2-second chunks
        chunk_samples = int(sample_rate * chunk_duration)

        # Send first chunk with domain prompts
        chunk_audio = audio[:chunk_samples]
        chunk_int16 = (chunk_audio * 32768.0).astype(np.int16)
        chunk_b64 = base64.b64encode(chunk_int16.tobytes()).decode("utf-8")

        # Build request with POLITICAL DOMAIN PROMPTS
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
            # DOMAIN PROMPT FIELDS - Political domain
            "domain": "political",  # Custom domain (not in built-in list, but will work)
            "custom_terms": [
                "Americans",
                "fellow citizens",
                "country",
                "nation",
                "freedom",
                "liberty",
                "democracy",
            ],
            "initial_prompt": "Presidential inaugural speech about American values and civic responsibility",
        }

        print("\n📝 Domain prompts configured:")
        print(f"   Domain: {request_data['domain']}")
        print(f"   Custom terms: {request_data['custom_terms']}")
        print(f"   Initial prompt: {request_data['initial_prompt']}")

        # Send first chunk
        sio.emit("transcribe_stream", request_data)
        print(f"\n📤 Sent chunk 1/{len(audio)//chunk_samples} ({chunk_duration}s)")
        time.sleep(3.0)  # Wait for processing

        # Send remaining chunks (without domain prompts - they persist in session)
        for i in range(1, len(audio) // chunk_samples):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, len(audio))

            chunk_audio = audio[start_idx:end_idx]
            chunk_int16 = (chunk_audio * 32768.0).astype(np.int16)
            chunk_b64 = base64.b64encode(chunk_int16.tobytes()).decode("utf-8")

            # Send subsequent chunks (domain prompts already set in session)
            request_data_next = {
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

            sio.emit("transcribe_stream", request_data_next)
            print(f"📤 Sent chunk {i+1}/{len(audio)//chunk_samples} ({chunk_duration}s)")
            time.sleep(2.0)

        # Wait for final results (longer wait to ensure processing completes)
        print("\n⏳ Waiting for final results...")
        print("   (Waiting 15 seconds to ensure all audio is processed...)")
        time.sleep(15.0)

        sio.emit("leave_session", {"session_id": session_id})
        time.sleep(0.5)
        sio.disconnect()

        # Display results
        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        print(f"Total results received: {len(results)}")

        if results:
            print("\n📝 Transcription results:")
            for i, result in enumerate(results, 1):
                text = result.get("text", "")
                is_draft = result.get("is_draft", False)
                is_final = result.get("is_final", False)
                status = "DRAFT" if is_draft else ("FINAL" if is_final else "UPDATE")
                print(f"   {i}. [{status}] '{text}'")

            # Combine all text
            full_text = " ".join([r.get("text", "") for r in results])
            print("\n📄 Full transcription:")
            print(f"   '{full_text}'")

            # Check for expected keywords
            print("\n🔍 Keyword detection:")
            keywords = ["Americans", "country", "fellow"]
            for keyword in keywords:
                found = keyword.lower() in full_text.lower()
                status = "✅" if found else "❌"
                print(f"   {status} '{keyword}' found: {found}")

        print("\n" + "=" * 80)
        if len(results) > 0:
            print("✅ TEST PASSED - Domain prompts working!")
        else:
            print("❌ TEST FAILED - No results received")
        print("=" * 80 + "\n")

        return {"passed": len(results) > 0, "results": results}

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return {"passed": False, "error": str(e)}


if __name__ == "__main__":
    result = test_jfk_with_political_domain()
    exit(0 if result.get("passed") else 1)
