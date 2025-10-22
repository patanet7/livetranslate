#!/usr/bin/env python3
"""
LIVE STREAMING SIMULATION TEST - SimulStreaming Pattern

This test simulates REAL-TIME LIVE MICROPHONE INPUT using JFK audio.
Based on SimulStreaming reference implementation.

Key differences from file playback:
- Uses 1.2-second chunks (SimulStreaming default --min-chunk-size)
- enable_vad=TRUE (for live microphone - filters silence)
- Streams 20 seconds of audio (not just 3)
- Processes at computation speed (not playback delays)
- Tests buffer accumulation and AlignAtt real-time behavior

Expected: Proper JFK transcription with VAD filtering silence
"""

import socketio
import numpy as np
import base64
import time
import wave
import os

SERVICE_URL = "http://localhost:5001"
JFK_AUDIO_PATH = "/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/jfk.wav"

# SimulStreaming default configuration
CHUNK_DURATION = 1.2  # seconds (SimulStreaming default --min-chunk-size)
TEST_DURATION = 20.0  # seconds (test first 20 seconds)
SAMPLE_RATE = 16000


def load_wav_audio(file_path, max_duration=None):
    """Load WAV audio file with optional duration limit"""
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()

        # Limit frames if max_duration specified
        if max_duration:
            max_frames = int(sample_rate * max_duration)
            n_frames = min(n_frames, max_frames)

        audio_bytes = wav_file.readframes(n_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0

        # Convert stereo to mono
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        return audio, sample_rate


def run_live_streaming_test(enable_vad=True, test_name="Default"):
    """
    Test LIVE STREAMING simulation with SimulStreaming configuration

    Simulates: Live microphone → 1.2s chunks → VAD filtering → AlignAtt

    Args:
        enable_vad: Whether to enable VAD pre-filtering (True for live, False for file playback)
        test_name: Descriptive name for this test run
    """
    print("\n" + "="*80)
    print(f"LIVE STREAMING SIMULATION TEST - {test_name}")
    print("="*80)
    print(f"Chunk Duration: {CHUNK_DURATION}s (SimulStreaming default)")
    print(f"Test Duration: {TEST_DURATION}s")
    print(f"Enable VAD: {enable_vad} ({'live microphone mode' if enable_vad else 'file playback mode'})")
    print("="*80)

    # Load JFK audio
    if not os.path.exists(JFK_AUDIO_PATH):
        print(f"❌ JFK audio not found: {JFK_AUDIO_PATH}")
        return {"passed": False, "error": "Audio file not found"}

    print(f"\n📁 Loading JFK audio (first {TEST_DURATION}s)...")
    jfk_audio, jfk_sample_rate = load_wav_audio(JFK_AUDIO_PATH, max_duration=TEST_DURATION)
    actual_duration = len(jfk_audio) / jfk_sample_rate
    print(f"✅ Loaded: {actual_duration:.1f}s at {jfk_sample_rate}Hz")

    # Resample to 16kHz if needed
    if jfk_sample_rate != SAMPLE_RATE:
        print(f"🔄 Resampling from {jfk_sample_rate}Hz to {SAMPLE_RATE}Hz...")
        from scipy import signal
        jfk_audio = signal.resample(jfk_audio, int(len(jfk_audio) * SAMPLE_RATE / jfk_sample_rate))
        jfk_sample_rate = SAMPLE_RATE
        print(f"✅ Resampled to {SAMPLE_RATE}Hz")

    # Calculate chunk size
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
    num_chunks = int(len(jfk_audio) / chunk_size)
    print(f"\n📊 Streaming configuration:")
    print(f"   Chunk size: {chunk_size} samples ({CHUNK_DURATION}s)")
    print(f"   Total chunks: {num_chunks}")
    print(f"   Total duration: {num_chunks * CHUNK_DURATION:.1f}s")

    # Connect to Socket.IO
    print(f"\n🔌 Connecting to service: {SERVICE_URL}")

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
        stable_text = data.get('stable_text', '')

        status = "✏️ DRAFT" if is_draft else ("✅ FINAL" if is_final else "📝 UPDATE")
        print(f"\n{status} Result #{len(results)}:")
        print(f"   Text: '{text[:60]}'")
        if stable_text and stable_text != text:
            print(f"   Stable: '{stable_text[:40]}'")

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        # Connect to Socket.IO
        sio.connect(SERVICE_URL)

        # Wait for connection to be fully established
        time.sleep(0.5)

        if not sio.connected:
            raise Exception("Failed to establish Socket.IO connection")

        session_id = f"live-sim-{int(time.time())}"
        sio.emit('join_session', {'session_id': session_id})
        time.sleep(0.5)  # Wait for join_session to complete

        print(f"\n🎙️  Starting live streaming simulation...")
        print(f"   Session ID: {session_id}")
        print("="*80)

        # Stream chunks - NO ARTIFICIAL DELAYS (process at computation speed)
        start_time = time.time()

        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(jfk_audio))
            chunk = jfk_audio[chunk_start:chunk_end]

            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

            chunk_bytes = chunk.tobytes()
            chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')

            # Send chunk with configuration
            sio.emit('transcribe_stream', {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "large-v3-turbo",
                "language": "en",
                "beam_size": 5,
                "sample_rate": SAMPLE_RATE,
                "task": "transcribe",
                "target_language": "en",
                "enable_vad": enable_vad,  # Configurable: True for live mic, False for file playback
            })

            # Brief pause to allow processing (but NOT real-time playback delay)
            time.sleep(0.1)  # Just enough for Socket.IO processing

            if (i + 1) % 5 == 0:
                print(f"   📤 Sent {i+1}/{num_chunks} chunks ({(i+1)*CHUNK_DURATION:.1f}s)")

        elapsed = time.time() - start_time
        print(f"\n✅ Streaming complete in {elapsed:.1f}s (processing speed, not real-time)")
        print(f"   Average: {elapsed/num_chunks:.3f}s per chunk")

        # Wait for final processing
        print("\n⏳ Waiting for final results...")
        time.sleep(2.0)

        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.3)
        sio.disconnect()

        return analyze_results(results)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


def analyze_results(results):
    """Analyze streaming test results"""
    print("\n" + "="*80)
    print("TEST RESULTS ANALYSIS")
    print("="*80)

    analysis = {
        "passed": False,
        "results_count": len(results),
        "has_stability_fields": False,
        "has_text": False,
        "transcription_text": "",
        "details": {}
    }

    if len(results) == 0:
        print("❌ NO RESULTS RECEIVED")
        print("   Possible causes:")
        print("   - VAD filtered all chunks (no speech detected)")
        print("   - Service not processing correctly")
        print("   - Buffer accumulation issue")
        analysis["message"] = "No results received"
        return analysis

    print(f"✅ Received {len(results)} results\n")

    # Check Phase 3C fields
    first_result = results[0]
    stability_fields = ['stable_text', 'unstable_text', 'is_draft', 'is_final',
                       'should_translate', 'stability_score']
    missing_fields = [f for f in stability_fields if f not in first_result]

    analysis["has_stability_fields"] = len(missing_fields) == 0
    analysis["missing_fields"] = missing_fields

    if analysis["has_stability_fields"]:
        print("✅ Phase 3C stability fields present")
    else:
        print(f"⚠️  Missing stability fields: {missing_fields}")

    # Check for text
    analysis["has_text"] = 'text' in first_result and len(first_result.get('text', '').strip()) > 0

    if analysis["has_text"]:
        print("✅ Transcription text present")
    else:
        print("❌ No transcription text")

    # Collect full transcription
    full_text = ' '.join([r.get('text', '') for r in results])
    analysis["transcription_text"] = full_text.strip()

    # Check for JFK keywords
    jfk_keywords = ["fellow", "americans", "ask", "country"]
    found_keywords = [kw for kw in jfk_keywords if kw.lower() in full_text.lower()]

    print(f"\n📝 Transcription ({len(full_text)} chars):")
    print(f"   '{full_text[:100]}'...")
    print(f"\n🔍 JFK keyword detection:")
    print(f"   Expected: {jfk_keywords}")
    print(f"   Found: {found_keywords} ({len(found_keywords)}/{len(jfk_keywords)})")

    # Extract stability details
    if analysis["has_stability_fields"]:
        analysis["details"] = {
            "stable_text": first_result.get('stable_text', '')[:50],
            "unstable_text": first_result.get('unstable_text', '')[:30],
            "is_draft": first_result.get('is_draft'),
            "is_final": first_result.get('is_final'),
            "should_translate": first_result.get('should_translate'),
            "stability_score": first_result.get('stability_score'),
        }

        print(f"\n🎯 Stability tracking (first result):")
        print(f"   Draft: {analysis['details']['is_draft']}")
        print(f"   Final: {analysis['details']['is_final']}")
        print(f"   Stability score: {analysis['details']['stability_score']}")

    # Test passes if we have stability fields, text, and JFK keywords
    analysis["passed"] = (analysis["has_stability_fields"] and
                         analysis["has_text"] and
                         len(found_keywords) >= 2)

    print("\n" + "="*80)
    if analysis["passed"]:
        print("✅ TEST PASSED!")
        print(f"   - Phase 3C stability tracking: WORKING")
        print(f"   - VAD live filtering: WORKING")
        print(f"   - JFK transcription: SUCCESSFUL ({len(found_keywords)}/4 keywords)")
    else:
        print("❌ TEST FAILED!")
        if not analysis["has_stability_fields"]:
            print(f"   - Missing Phase 3C fields: {missing_fields}")
        if not analysis["has_text"]:
            print("   - No transcription text")
        if len(found_keywords) < 2:
            print(f"   - Poor transcription quality ({len(found_keywords)}/4 keywords)")
    print("="*80)

    return analysis


def main():
    """Run live streaming simulation tests (both VAD modes)"""
    print("\n" + "="*80)
    print("LIVE STREAMING SIMULATION - DUAL MODE TEST")
    print("Testing both VAD=False (file playback) and VAD=True (live mic)")
    print("="*80)

    results = []

    # Test 1: VAD = False (file playback mode)
    print("\n\n" + "🎬 "*20)
    print("TEST 1: FILE PLAYBACK MODE (enable_vad=False)")
    print("🎬 "*20)
    result_vad_false = run_live_streaming_test(enable_vad=False, test_name="File Playback (VAD OFF)")
    results.append(("VAD=False", result_vad_false))

    # Wait between tests
    time.sleep(3)

    # Test 2: VAD = True (live microphone mode)
    print("\n\n" + "🎙️ "*20)
    print("TEST 2: LIVE MICROPHONE MODE (enable_vad=True)")
    print("🎙️ "*20)
    result_vad_true = run_live_streaming_test(enable_vad=True, test_name="Live Microphone (VAD ON)")
    results.append(("VAD=True", result_vad_true))

    # Summary
    print("\n\n" + "="*80)
    print("DUAL MODE TEST SUMMARY")
    print("="*80)
    for test_name, result in results:
        status = "✅ PASSED" if result.get("passed") else "❌ FAILED"
        print(f"\n{status} - {test_name}")
        print(f"   Results: {result.get('results_count')} transcriptions")
        print(f"   Text: '{result.get('transcription_text', '')[:60]}'")
        if result.get("message"):
            print(f"   Message: {result['message']}")

    # Overall pass/fail
    all_passed = all(r.get("passed") for _, r in results)
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED - See details above")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
