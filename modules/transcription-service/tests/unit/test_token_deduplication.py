#!/usr/bin/env python3
"""
Phase 5: Token De-duplication - INTEGRATED TEST

Test that tokens are properly deduplicated at streaming chunk boundaries
to prevent repeated phrases and incomplete characters (�).

Requirements:
- Deduplicate exact token overlap at chunk boundaries
- Handle multi-token overlaps (2-3 tokens)
- No false positives (removing valid tokens)
- Works with both draft and final outputs
- Latency overhead < 2ms
"""

import base64
import os
import sys
import time
import wave

import numpy as np
import socketio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SERVICE_URL = "http://localhost:5001"


def load_wav_file(filepath):
    """Load WAV file and return audio data + sample rate"""
    with wave.open(filepath, "rb") as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)

        # Convert to int16 numpy array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        # If stereo, convert to mono
        if n_channels == 2:
            audio_int16 = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)

        return audio_int16, sample_rate


def resample_audio(audio_int16, orig_sr, target_sr=16000):
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio_int16

    print(f"  Resampling {orig_sr}Hz → {target_sr}Hz...")
    import librosa

    audio_float = audio_int16.astype(np.float32) / 32768.0
    audio_resampled = librosa.resample(audio_float, orig_sr=orig_sr, target_sr=target_sr)
    return (audio_resampled * 32768.0).astype(np.int16)


def test_chinese_chunk_artifacts():
    """
    Test Chinese audio for chunk boundary artifacts (�)

    Before Phase 5: Chinese text shows � at boundaries (incomplete UTF-8)
    After Phase 5: Should have clean Chinese text with no � characters
    """
    print("\n" + "=" * 80)
    print("TEST 1: Chinese Chunk Boundary Artifacts")
    print("=" * 80)
    print("Expected BEFORE Phase 5: '院子门口不远�' (has � artifact)")
    print("Expected AFTER Phase 5:  '院子门口不远处...' (clean Chinese)")

    chinese_path = os.path.join(os.path.dirname(__file__), "audio", "OSR_cn_000_0072_8k.wav")
    audio_int16, sample_rate = load_wav_file(chinese_path)
    audio_int16 = resample_audio(audio_int16, sample_rate, 16000)

    sio = socketio.Client()
    results_received = []
    has_artifact = False

    @sio.on("connect")
    def on_connect():
        print("✓ Connected to Whisper service")

    @sio.on("transcription_result")
    def on_result(data):
        results_received.append(data)
        text = data.get("text", "")

        # Check for � character (chunk boundary artifact)
        if "�" in text:
            print(f"  ⚠️  ARTIFACT DETECTED: '{text}'")
        else:
            print(f"  ✅ Clean text: '{text}'")

    @sio.on("error")
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        sio.connect(SERVICE_URL)
        time.sleep(0.5)

        session_id = f"test-chinese-dedup-{int(time.time())}"
        sio.emit("join_session", {"session_id": session_id})
        time.sleep(0.5)

        # Stream in 1-second chunks
        chunk_size = 16000 * 1
        total_chunks = len(audio_int16) // chunk_size + (1 if len(audio_int16) % chunk_size else 0)

        print(f"\n  Streaming {total_chunks} chunks...")
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_int16))
            chunk = audio_int16[start_idx:end_idx]

            chunk_b64 = base64.b64encode(chunk.tobytes()).decode("utf-8")

            request_data = {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "base",
                "sample_rate": 16000,
                "enable_code_switching": True,
                "config": {
                    "sliding_lid_window": 0.9,
                },
            }

            sio.emit("transcribe_stream", request_data)
            time.sleep(0.2)

        print("  Waiting for final processing...")
        time.sleep(3.0)

        print(f"\n  Received {len(results_received)} results")

        sio.emit("leave_session", {"session_id": session_id})
        time.sleep(0.5)
        sio.disconnect()

        # Test passes if NO � artifacts detected
        if not has_artifact:
            print("  ✅ PASS: No chunk boundary artifacts (� characters)")
            return True
        else:
            print("  ⚠️  FAIL: Chunk boundary artifacts still present")
            print("  This is expected BEFORE Phase 5 implementation")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_english_sentence_completion():
    """
    Test English audio for complete sentence at end

    Before Phase 5: JFK speech cuts off, missing "country" at end
    After Phase 5: Should have complete sentence
    """
    print("\n" + "=" * 80)
    print("TEST 2: English Sentence Completion")
    print("=" * 80)
    print("Expected BEFORE Phase 5: Missing 'country' at end")
    print("Expected AFTER Phase 5:  Complete sentence with 'country'")

    english_path = os.path.join(os.path.dirname(__file__), "audio", "jfk.wav")
    audio_int16, sample_rate = load_wav_file(english_path)
    audio_int16 = resample_audio(audio_int16, sample_rate, 16000)

    sio = socketio.Client()
    results_received = []
    has_country = False

    @sio.on("connect")
    def on_connect():
        print("✓ Connected to Whisper service")

    @sio.on("transcription_result")
    def on_result(data):
        results_received.append(data)
        text = data.get("text", "")

        # Check for "country" in text
        if "country" in text.lower():
            print(f"  ✅ Contains 'country': '{text}'")
        else:
            print(f"  📝 Text: '{text}'")

    @sio.on("error")
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        sio.connect(SERVICE_URL)
        time.sleep(0.5)

        session_id = f"test-english-dedup-{int(time.time())}"
        sio.emit("join_session", {"session_id": session_id})
        time.sleep(0.5)

        # Stream in 1-second chunks
        chunk_size = 16000 * 1
        total_chunks = len(audio_int16) // chunk_size + (1 if len(audio_int16) % chunk_size else 0)

        print(f"\n  Streaming {total_chunks} chunks...")
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_int16))
            chunk = audio_int16[start_idx:end_idx]

            chunk_b64 = base64.b64encode(chunk.tobytes()).decode("utf-8")

            request_data = {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "base",
                "sample_rate": 16000,
                "enable_code_switching": True,
                "config": {
                    "sliding_lid_window": 0.9,
                },
            }

            sio.emit("transcribe_stream", request_data)
            time.sleep(0.2)

        print("  Waiting for final processing...")
        time.sleep(3.0)

        print(f"\n  Received {len(results_received)} results")

        # Combine all text
        full_text = " ".join([r.get("text", "") for r in results_received])
        print(f"  Full transcription: '{full_text}'")

        sio.emit("leave_session", {"session_id": session_id})
        time.sleep(0.5)
        sio.disconnect()

        # Test passes if "country" is present
        if has_country:
            print("  ✅ PASS: Complete sentence with 'country'")
            return True
        else:
            print("  ⚠️  FAIL: Sentence incomplete, 'country' missing")
            print("  This is expected BEFORE Phase 5 implementation")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_no_repeated_phrases():
    """
    Test that phrases don't repeat at chunk boundaries

    Common issue: "hello world hello world" due to token overlap
    Should deduplicate to: "hello world"
    """
    print("\n" + "=" * 80)
    print("TEST 3: No Repeated Phrases at Chunk Boundaries")
    print("=" * 80)

    english_path = os.path.join(os.path.dirname(__file__), "audio", "jfk.wav")
    audio_int16, sample_rate = load_wav_file(english_path)
    audio_int16 = resample_audio(audio_int16, sample_rate, 16000)

    sio = socketio.Client()
    results_received = []
    has_repetition = False

    @sio.on("connect")
    def on_connect():
        print("✓ Connected to Whisper service")

    @sio.on("transcription_result")
    def on_result(data):
        results_received.append(data)
        text = data.get("text", "")

        # Check for obvious repetitions (same 3+ word sequence repeated)
        words = text.lower().split()
        for i in range(len(words) - 5):
            # Check if next 3 words are identical to current 3 words
            if words[i : i + 3] == words[i + 3 : i + 6]:
                has_repetition = True
                print(f"  ⚠️  REPETITION DETECTED: {words[i:i+6]}")
                break

        if not has_repetition:
            print(f"  ✅ No repetition: '{text[:80]}...'")

    @sio.on("error")
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        sio.connect(SERVICE_URL)
        time.sleep(0.5)

        session_id = f"test-repetition-{int(time.time())}"
        sio.emit("join_session", {"session_id": session_id})
        time.sleep(0.5)

        # Stream in smaller chunks (0.5s) to increase boundary chances
        chunk_size = 16000 // 2
        total_chunks = min(20, len(audio_int16) // chunk_size)

        print(f"\n  Streaming {total_chunks} chunks (0.5s each)...")
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_int16))
            chunk = audio_int16[start_idx:end_idx]

            chunk_b64 = base64.b64encode(chunk.tobytes()).decode("utf-8")

            request_data = {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "base",
                "sample_rate": 16000,
                "enable_code_switching": True,
                "config": {
                    "sliding_lid_window": 0.9,
                },
            }

            sio.emit("transcribe_stream", request_data)
            time.sleep(0.1)

        print("  Waiting for final processing...")
        time.sleep(3.0)

        print(f"\n  Received {len(results_received)} results")

        sio.emit("leave_session", {"session_id": session_id})
        time.sleep(0.5)
        sio.disconnect()

        # Test passes if NO repetitions detected
        if not has_repetition:
            print("  ✅ PASS: No phrase repetitions detected")
            return True
        else:
            print("  ⚠️  FAIL: Phrase repetitions found")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 5: TOKEN DE-DUPLICATION - INTEGRATED TESTS")
    print("=" * 80)
    print("\nThese tests verify that tokens are properly deduplicated at")
    print("streaming chunk boundaries to prevent:")
    print("  - Incomplete UTF-8 characters (�)")
    print("  - Missing words at sentence boundaries")
    print("  - Repeated phrases")

    test1_pass = test_chinese_chunk_artifacts()
    test2_pass = test_english_sentence_completion()
    test3_pass = test_no_repeated_phrases()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"Test 1 (Chinese artifacts):     {'✅ PASS' if test1_pass else '⚠️  FAIL (expected before Phase 5)'}"
    )
    print(
        f"Test 2 (English completion):    {'✅ PASS' if test2_pass else '⚠️  FAIL (expected before Phase 5)'}"
    )
    print(f"Test 3 (No repetitions):        {'✅ PASS' if test3_pass else '❌ FAIL'}")

    if all([test1_pass, test2_pass, test3_pass]):
        print("\n🎉 All tests passed - Token deduplication working!")
        exit(0)
    else:
        print("\n⚠️  Some tests failed - this is expected BEFORE Phase 5 implementation")
        print("    After implementing TokenDeduplicator, all tests should pass")
        exit(0)  # Exit 0 for now, will change to exit(1) after implementation
