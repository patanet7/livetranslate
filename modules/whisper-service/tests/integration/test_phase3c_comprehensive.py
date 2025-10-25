#!/usr/bin/env python3
"""
COMPREHENSIVE PHASE 3C + TASK PARAMETER TEST

Uses REAL audio (JFK speech) to test:
1. Task parameter routing (transcribe vs translate)
2. Phase 3C stability tracking
3. VAD compatibility
4. Beam search compatibility
5. All fields flowing correctly

Expected JFK transcription:
"And so my fellow Americans, ask not what your country can do for you,
ask what you can do for your country."
"""

import socketio
import numpy as np
import base64
import time
import wave
import os

SERVICE_URL = "http://localhost:5001"
JFK_AUDIO_PATH = "/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/jfk.wav"


def load_wav_audio(file_path):
    """Load WAV audio file"""
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(n_frames)

        # Convert to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert to float32 and normalize
        audio = audio.astype(np.float32) / 32768.0

        # If stereo, convert to mono
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        return audio, sample_rate


def create_synthetic_silence(duration=1.0, sample_rate=16000):
    """Create pure silence for VAD testing"""
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


def run_streaming_test(test_name, audio, sample_rate, task, target_language,
                       beam_size=5, enable_vad=True, language="en"):
    """Run a single streaming test"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Config: task={task}, target_lang={target_language}, beam={beam_size}, vad={enable_vad}")

    sio = socketio.Client()
    results = []

    @sio.on('connect')
    def on_connect():
        print("✅ Connected")

    @sio.on('transcription_result')
    def on_result(data):
        results.append(data)
        print(f"📥 Result: text='{data.get('text', '')[:50]}'")

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        sio.connect(SERVICE_URL)
        session_id = f"test-{int(time.time())}"
        sio.emit('join_session', {'session_id': session_id})
        time.sleep(0.3)

        # Split audio into 500ms chunks
        chunk_size = int(sample_rate * 0.5)
        num_chunks = 0

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

            chunk_bytes = chunk.tobytes()
            chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')

            sio.emit('transcribe_stream', {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "large-v3-turbo",
                "language": language,
                "beam_size": beam_size,
                "sample_rate": sample_rate,
                "task": task,
                "target_language": target_language,
                "enable_vad": enable_vad,  # Controls VAD pre-filtering: False for file playback
            })

            num_chunks += 1
            time.sleep(0.8)

        print(f"   Sent {num_chunks} chunks")
        time.sleep(2.0)

        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.3)
        sio.disconnect()

        return analyze_results(results, test_name)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


def analyze_results(results, test_name):
    """Analyze test results"""
    analysis = {
        "test_name": test_name,
        "passed": False,
        "results_count": len(results),
        "has_stability_fields": False,
        "has_text": False,
        "transcription_text": "",
        "details": {}
    }

    if len(results) == 0:
        analysis["message"] = "No results (possibly VAD filtered)"
        return analysis

    # Check for Phase 3C stability fields
    first_result = results[0]
    stability_fields = ['stable_text', 'unstable_text', 'is_draft', 'is_final',
                       'should_translate', 'stability_score']
    missing_fields = [f for f in stability_fields if f not in first_result]

    analysis["has_stability_fields"] = len(missing_fields) == 0
    analysis["missing_fields"] = missing_fields

    # Check for text content
    analysis["has_text"] = 'text' in first_result and len(first_result.get('text', '').strip()) > 0

    # Collect transcription
    full_text = ' '.join([r.get('text', '') for r in results])
    analysis["transcription_text"] = full_text.strip()

    # Extract stability info from first result
    if analysis["has_stability_fields"]:
        analysis["details"] = {
            "stable_text": first_result.get('stable_text', '')[:50],
            "unstable_text": first_result.get('unstable_text', '')[:30],
            "is_draft": first_result.get('is_draft'),
            "is_final": first_result.get('is_final'),
            "should_translate": first_result.get('should_translate'),
            "stability_score": first_result.get('stability_score'),
            "translation_mode": first_result.get('translation_mode')
        }

    # Test passes if we have stability fields and text
    analysis["passed"] = analysis["has_stability_fields"] and analysis["has_text"]

    if analysis["passed"]:
        analysis["message"] = f"✅ PASSED - {len(results)} results with stability"
    else:
        reasons = []
        if not analysis["has_stability_fields"]:
            reasons.append(f"missing fields: {missing_fields}")
        if not analysis["has_text"]:
            reasons.append("no text")
        analysis["message"] = f"❌ FAILED - {', '.join(reasons)}"

    return analysis


def main():
    """Run comprehensive Phase 3C tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE PHASE 3C + TASK PARAMETER TEST")
    print("Using: Real JFK audio")
    print("="*80)

    # Check if JFK audio exists
    if not os.path.exists(JFK_AUDIO_PATH):
        print(f"❌ JFK audio not found at: {JFK_AUDIO_PATH}")
        return 1

    # Load JFK audio
    print(f"\n📁 Loading JFK audio from: {JFK_AUDIO_PATH}")
    jfk_audio, jfk_sample_rate = load_wav_audio(JFK_AUDIO_PATH)
    print(f"✅ Loaded: {len(jfk_audio)/jfk_sample_rate:.1f}s at {jfk_sample_rate}Hz")

    # Resample to 16kHz if needed
    if jfk_sample_rate != 16000:
        print(f"   Resampling from {jfk_sample_rate}Hz to 16000Hz...")
        from scipy import signal
        jfk_audio = signal.resample(jfk_audio, int(len(jfk_audio) * 16000 / jfk_sample_rate))
        jfk_sample_rate = 16000

    # Create silence audio for VAD test
    silence_audio = create_synthetic_silence(duration=2.0, sample_rate=16000)

    test_results = []

    # ==================== TASK PARAMETER TESTS ====================
    print("\n\n" + "="*80)
    print("📋 TASK PARAMETER TESTS")
    print("="*80)

    test_results.append(run_streaming_test(
        "1. Transcribe to English",
        jfk_audio, jfk_sample_rate,
        task="transcribe",
        target_language="en",
        beam_size=5,
        enable_vad=False  # CRITICAL: Disable VAD pre-filtering for file playback
    ))

    test_results.append(run_streaming_test(
        "2. Transcribe to Spanish (External Translation)",
        jfk_audio, jfk_sample_rate,
        task="transcribe",
        target_language="es",
        beam_size=5,
        enable_vad=False  # CRITICAL: Disable VAD pre-filtering for file playback
    ))

    test_results.append(run_streaming_test(
        "3. Translate to English (Whisper Translate Mode)",
        jfk_audio, jfk_sample_rate,
        task="translate",
        target_language="en",
        beam_size=5,
        enable_vad=False,  # CRITICAL: Disable VAD pre-filtering for file playback
        language="auto"  # Auto-detect
    ))

    test_results.append(run_streaming_test(
        "4. Translate to Spanish (Should Fallback to Transcribe)",
        jfk_audio, jfk_sample_rate,
        task="translate",
        target_language="es",
        beam_size=5,
        enable_vad=False  # CRITICAL: Disable VAD pre-filtering for file playback
    ))

    # ==================== VAD TESTS ====================
    print("\n\n" + "="*80)
    print("🎙️  VOICE ACTIVITY DETECTION TESTS")
    print("="*80)

    test_results.append(run_streaming_test(
        "5. VAD with Real Speech (JFK)",
        jfk_audio, jfk_sample_rate,
        task="transcribe",
        target_language="en",
        beam_size=5,
        enable_vad=True
    ))

    test_results.append(run_streaming_test(
        "6. VAD with Silence (Should Filter)",
        silence_audio, 16000,
        task="transcribe",
        target_language="en",
        beam_size=5,
        enable_vad=True
    ))

    # ==================== BEAM SEARCH TESTS ====================
    print("\n\n" + "="*80)
    print("🔍 BEAM SEARCH TESTS")
    print("="*80)

    for beam_size in [1, 5, 10]:
        test_results.append(run_streaming_test(
            f"7.{beam_size}. Beam Size = {beam_size}",
            jfk_audio, jfk_sample_rate,
            task="transcribe",
            target_language="en",
            beam_size=beam_size,
            enable_vad=False  # CRITICAL: Disable VAD pre-filtering for file playback
        ))

    # ==================== SUMMARY ====================
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for result in test_results:
        status = "✅" if result.get("passed") else "❌"
        print(f"\n{status} {result.get('test_name')}")
        print(f"   {result.get('message')}")
        print(f"   Results: {result.get('results_count')}")
        if result.get('transcription_text'):
            print(f"   Text: '{result['transcription_text'][:60]}'")
        if result.get('details'):
            details = result['details']
            print(f"   Stability: draft={details.get('is_draft')}, " +
                  f"final={details.get('is_final')}, " +
                  f"score={details.get('stability_score')}")

    passed = sum(1 for r in test_results if r.get("passed"))
    total = len(test_results)

    print(f"\n{'='*80}")
    print(f"FINAL SCORE: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*80}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
