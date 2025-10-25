#!/usr/bin/env python3
"""
Simple test for sustained language detection with direct whisper service
Tests that sustained detection works without errors
"""

import sys
import socketio
import base64
import numpy as np
import librosa
import time

WHISPER_URL = 'http://127.0.0.1:5001'
JFK_AUDIO = 'tests/audio/jfk.wav'
CHINESE_AUDIO = 'tests/audio/OSR_cn_000_0072_8k.wav'

sio = socketio.Client()
results = []

@sio.on('transcription_result')
def on_transcription(data):
    results.append(data)
    lang = data.get('detected_language', 'unknown')
    text = data.get('text', '')
    print(f"[RESULT] Language={lang}: {text}")

@sio.on('connect')
def on_connect():
    print("✅ Connected to whisper service")

@sio.on('disconnect')
def on_disconnect():
    print("❌ Disconnected from whisper service")

def send_audio_chunks(audio_path, num_chunks=5, chunk_duration=0.04):
    """Send audio in small chunks"""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    chunk_size = int(chunk_duration * sr)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(audio))
        chunk = audio[start_idx:end_idx]

        audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        data = {
            'session_id': 'test-sustained-detection',
            'audio_data': audio_b64,
            'model_name': 'large-v3-turbo',
            'sample_rate': 16000,
            'enable_code_switching': True,
            'chunk_index': i,
            'audio_start_time': i * chunk_duration,
            'audio_end_time': (i + 1) * chunk_duration,
            'chunk_duration': chunk_duration,
            'is_last_chunk': (i == num_chunks - 1)
        }

        sio.emit('transcribe_stream', data)
        print(f"[SENT] Chunk {i+1}/{num_chunks} from {audio_path}")
        time.sleep(0.05)

def main():
    print("=" * 80)
    print("SUSTAINED LANGUAGE DETECTION TEST")
    print("=" * 80)

    # Connect
    sio.connect(WHISPER_URL)
    time.sleep(1)

    # Test pattern: 4 English chunks → 4 Chinese chunks → 4 English chunks
    print("\n📊 Test Pattern: 4 JFK → 4 Chinese → 4 JFK")
    print("-" * 80)

    print("\n[PHASE 1] Sending 4 English (JFK) chunks...")
    send_audio_chunks(JFK_AUDIO, num_chunks=4)
    time.sleep(2)

    print("\n[PHASE 2] Sending 4 Chinese chunks...")
    send_audio_chunks(CHINESE_AUDIO, num_chunks=4)
    time.sleep(2)

    print("\n[PHASE 3] Sending 4 English (JFK) chunks...")
    send_audio_chunks(JFK_AUDIO, num_chunks=4)
    time.sleep(2)

    # Disconnect
    sio.disconnect()

    print("\n" + "=" * 80)
    print(f"✅ Test completed! Received {len(results)} results")
    print("=" * 80)

    # Show language distribution
    languages = [r.get('detected_language', 'unknown') for r in results]
    for lang in set(languages):
        count = languages.count(lang)
        print(f"  {lang}: {count} results")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
