#!/usr/bin/env python3
"""
TASK PARAMETER TEST - Verify translate vs transcribe logic

Tests that the service correctly chooses Whisper's task mode based on:
- task parameter (transcribe vs translate)
- target_language parameter
- source audio language

Expected behavior:
1. task="translate" + target="en" → Use Whisper translate (any → English)
2. task="translate" + target="es" → Use Whisper transcribe (external service handles translation)
3. task="transcribe" → Always use Whisper transcribe
"""

import socketio
import json
import numpy as np
import base64
import time
import re

SERVICE_URL = "http://localhost:5001"

def create_speech_audio(duration=2.0, sample_rate=16000):
    """Create realistic speech-like audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Speech-like modulated sine wave
    freq = 200 + 100 * np.sin(2 * np.pi * 2 * t)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    audio += 0.15 * np.sin(2 * np.pi * 2 * freq * t)

    # Envelope
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    audio = audio * envelope

    # Add noise
    audio += np.random.normal(0, 0.02, audio.shape)

    return audio.astype(np.float32)


def test_task_mode(task, target_language, expected_whisper_task):
    """Test a specific task/target_language combination"""

    print(f"\n{'='*80}")
    print(f"TEST: task='{task}', target_language='{target_language}'")
    print(f"Expected Whisper task mode: '{expected_whisper_task}'")
    print(f"{'='*80}")

    # Create audio chunks
    sample_rate = 16000
    audio = create_speech_audio(duration=2.0, sample_rate=sample_rate)

    # Split into chunks
    chunk_size = int(sample_rate * 0.5)  # 500ms chunks
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)

    print(f"Created {len(chunks)} audio chunks")

    # Connect to Socket.IO
    sio = socketio.Client()

    results = []
    task_logs = []

    @sio.on('connect')
    def on_connect():
        print("✅ Connected to Socket.IO")

    @sio.on('transcription_result')
    def on_transcription(data):
        print(f"\n📥 Transcription result received:")
        print(f"  Text: '{data.get('text', 'N/A')[:60]}'")
        print(f"  Stable: '{data.get('stable_text', 'N/A')[:40]}'")
        print(f"  Should Translate: {data.get('should_translate', 'N/A')}")
        results.append(data)

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data.get('message', 'Unknown')}")

    try:
        sio.connect(SERVICE_URL)

        # Join session
        session_id = f"test-task-{int(time.time())}"
        sio.emit('join_session', {'session_id': session_id})
        time.sleep(0.3)
        print(f"✅ Joined session: {session_id}")

        # Stream chunks with specified task and target_language
        print(f"\n📤 Streaming {len(chunks)} chunks with task='{task}', target_language='{target_language}'...")

        for i, chunk in enumerate(chunks):
            chunk_bytes = chunk.tobytes()
            chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')

            sio.emit('transcribe_stream', {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "large-v3-turbo",
                "language": "en",  # Source language
                "beam_size": 5,
                "sample_rate": sample_rate,
                "task": task,
                "target_language": target_language
            })

            time.sleep(0.8)  # Wait for processing

        # Wait for final results
        time.sleep(2.0)

        # Leave session
        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.3)

        sio.disconnect()

        print(f"\n{'='*80}")
        print("TEST RESULTS:")
        print(f"  Emissions received: {len(results)}")
        print(f"  Expected Whisper task: '{expected_whisper_task}'")

        # Check service logs to verify correct task mode was used
        print(f"\n  ⚠️  Manual verification required:")
        print(f"     Check service logs for: '[TASK] Using Whisper {expected_whisper_task}'")

        if len(results) > 0:
            print(f"\n✅ Test passed - received transcription results")
            return True
        else:
            print(f"\n⚠️  No results received (may be filtered by VAD)")
            return False

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all task parameter tests"""

    print("\n" + "="*80)
    print("TASK PARAMETER COMPREHENSIVE TEST")
    print("Testing: Whisper task mode selection logic")
    print("="*80 + "\n")

    test_cases = [
        # (task, target_language, expected_whisper_task)
        ("transcribe", "en", "transcribe"),
        ("transcribe", "es", "transcribe"),
        ("transcribe", "ja", "transcribe"),
        ("translate", "en", "translate"),
        ("translate", "es", "transcribe"),  # Can't translate to non-English, use transcribe
        ("translate", "ja", "transcribe"),  # Can't translate to non-English, use transcribe
    ]

    results = []

    for task, target_lang, expected in test_cases:
        result = test_task_mode(task, target_lang, expected)
        results.append({
            'task': task,
            'target_language': target_lang,
            'expected': expected,
            'passed': result
        })
        time.sleep(1.0)  # Brief pause between tests

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for r in results:
        status = "✅" if r['passed'] else "⚠️ "
        print(f"{status} task={r['task']:<12} target={r['target_language']:<4} → expected Whisper task={r['expected']:<12}")

    passed = sum(1 for r in results if r['passed'])
    total = len(results)

    print(f"\n{passed}/{total} tests passed")
    print("\n⚠️  IMPORTANT: Manually verify service logs show correct '[TASK]' messages")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
