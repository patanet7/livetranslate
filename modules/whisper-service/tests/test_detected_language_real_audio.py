#!/usr/bin/env python3
"""
Quick test to verify detected_language field with REAL audio files
"""

import sys
import os
import socketio
import time
import base64
import wave
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

SERVICE_URL = "http://localhost:5001"


def load_wav_file(filepath):
    """Load WAV file and return audio data + sample rate"""
    with wave.open(filepath, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)

        # Convert to int16 numpy array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        # If stereo, convert to mono
        if n_channels == 2:
            audio_int16 = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)

        print(f"  Loaded: {filepath}")
        print(f"  Sample rate: {sample_rate}Hz, Duration: {n_frames/sample_rate:.2f}s")

        return audio_int16, sample_rate


def test_english_audio():
    """Test with English audio (JFK)"""
    print("\n" + "="*80)
    print("TEST: English Audio (JFK)")
    print("="*80)

    audio_path = os.path.join(os.path.dirname(__file__), 'audio', 'jfk.wav')
    audio_int16, sample_rate = load_wav_file(audio_path)

    sio = socketio.Client()
    results_received = []

    @sio.on('connect')
    def on_connect():
        print("✓ Connected to Whisper service")

    @sio.on('transcription_result')
    def on_result(data):
        results_received.append(data)
        detected_lang = data.get('detected_language')
        text = data.get('text', '')
        is_final = data.get('is_final', False)
        print(f"  Result: detected_language={detected_lang}, is_final={is_final}")
        print(f"          text='{text}'")

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        sio.connect(SERVICE_URL)
        time.sleep(0.5)

        session_id = f"test-english-{int(time.time())}"
        sio.emit('join_session', {'session_id': session_id})
        time.sleep(0.5)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            print(f"  Resampling {sample_rate}Hz → 16000Hz...")
            import librosa
            audio_float = audio_int16.astype(np.float32) / 32768.0
            audio_resampled = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
            audio_int16 = (audio_resampled * 32768.0).astype(np.int16)

        # STREAM audio in chunks (not all at once!)
        chunk_size = 16000 * 1  # 1 second chunks
        total_chunks = len(audio_int16) // chunk_size + (1 if len(audio_int16) % chunk_size else 0)

        print(f"\n  Streaming {total_chunks} chunks (1s each)...")

        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_int16))
            chunk = audio_int16[start_idx:end_idx]

            chunk_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')

            request_data = {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "base",
                "sample_rate": 16000,
                "enable_code_switching": True,
                "config": {
                    "sliding_lid_window": 0.9,
                }
            }

            print(f"  Chunk {i+1}/{total_chunks}: {len(chunk)} samples ({len(chunk)/16000:.2f}s)")
            sio.emit('transcribe_stream', request_data)
            time.sleep(0.2)  # Small delay between chunks

        print("  Waiting for final processing...")
        time.sleep(3.0)  # Wait for final results

        print(f"\n  Received {len(results_received)} results")

        # Check for detected_language
        has_detected_language = any('detected_language' in r for r in results_received)
        if has_detected_language:
            print("  ✅ 'detected_language' field FOUND!")
            detected_langs = [r.get('detected_language') for r in results_received if 'detected_language' in r]
            print(f"  Languages detected: {set(detected_langs)}")
        else:
            print("  ⚠️  'detected_language' field NOT found")

        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.5)
        sio.disconnect()

        return has_detected_language

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chinese_audio():
    """Test with Chinese audio"""
    print("\n" + "="*80)
    print("TEST: Chinese Audio")
    print("="*80)

    audio_path = os.path.join(os.path.dirname(__file__), 'audio', 'OSR_cn_000_0072_8k.wav')
    audio_int16, sample_rate = load_wav_file(audio_path)

    sio = socketio.Client()
    results_received = []

    @sio.on('connect')
    def on_connect():
        print("✓ Connected to Whisper service")

    @sio.on('transcription_result')
    def on_result(data):
        results_received.append(data)
        detected_lang = data.get('detected_language')
        text = data.get('text', '')
        is_final = data.get('is_final', False)
        print(f"  Result: detected_language={detected_lang}, is_final={is_final}")
        print(f"          text='{text}'")

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        sio.connect(SERVICE_URL)
        time.sleep(0.5)

        session_id = f"test-chinese-{int(time.time())}"
        sio.emit('join_session', {'session_id': session_id})
        time.sleep(0.5)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            print(f"  Resampling {sample_rate}Hz → 16000Hz...")
            import librosa
            audio_float = audio_int16.astype(np.float32) / 32768.0
            audio_resampled = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
            audio_int16 = (audio_resampled * 32768.0).astype(np.int16)

        # STREAM audio in chunks (not all at once!)
        chunk_size = 16000 * 1  # 1 second chunks
        total_chunks = len(audio_int16) // chunk_size + (1 if len(audio_int16) % chunk_size else 0)

        print(f"\n  Streaming {total_chunks} chunks (1s each)...")

        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_int16))
            chunk = audio_int16[start_idx:end_idx]

            chunk_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')

            request_data = {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "base",
                "sample_rate": 16000,
                "enable_code_switching": True,
                "config": {
                    "sliding_lid_window": 0.9,
                }
            }

            print(f"  Chunk {i+1}/{total_chunks}: {len(chunk)} samples ({len(chunk)/16000:.2f}s)")
            sio.emit('transcribe_stream', request_data)
            time.sleep(0.2)  # Small delay between chunks

        print("  Waiting for final processing...")
        time.sleep(3.0)  # Wait for final results

        print(f"\n  Received {len(results_received)} results")

        # Check for detected_language
        has_detected_language = any('detected_language' in r for r in results_received)
        if has_detected_language:
            print("  ✅ 'detected_language' field FOUND!")
            detected_langs = [r.get('detected_language') for r in results_received if 'detected_language' in r]
            print(f"  Languages detected: {set(detected_langs)}")
        else:
            print("  ⚠️  'detected_language' field NOT found")

        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.5)
        sio.disconnect()

        return has_detected_language

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DETECTED LANGUAGE TEST - Real Audio Files")
    print("="*80)

    english_passed = test_english_audio()
    chinese_passed = test_chinese_audio()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"English (JFK): {'✅ PASS' if english_passed else '❌ FAIL'}")
    print(f"Chinese:       {'✅ PASS' if chinese_passed else '❌ FAIL'}")

    if english_passed and chinese_passed:
        print("\n🎉 Both tests passed - detected_language working!")
        exit(0)
    else:
        print("\n⚠️  Some tests failed")
        exit(1)
