#!/usr/bin/env python3
"""
REAL Streaming Code-Switching Test

Tests code-switching with ACTUAL Socket.IO streaming like production.
Uses real Chinese + English audio files, sends them as progressive chunks.

This tests the REAL system behavior, not batch transcription.
"""

import socketio
import numpy as np
import base64
import time
import soundfile as sf
import librosa

SERVICE_URL = "http://localhost:5001"
AUDIO_DIR = "tests/audio"


def load_and_resample(filepath: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to 16kHz"""
    print(f"   Loading: {filepath}")
    audio, sr = sf.read(filepath)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Resample if needed
    if sr != target_sr:
        print(f"   Resampling from {sr}Hz to {target_sr}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    duration = len(audio) / target_sr
    print(f"   ✓ Loaded {duration:.2f}s @ {target_sr}Hz")

    return audio.astype(np.float32)


def test_streaming_code_switching():
    """
    Test code-switching with REAL streaming chunks

    Simulates real production behavior:
    - Chinese audio (0-5s)
    - Short silence (5-5.5s)
    - English audio (5.5-10.5s)

    Sends as 2-second chunks progressively via Socket.IO.
    """
    print("\n" + "="*80)
    print("REAL STREAMING CODE-SWITCHING TEST")
    print("="*80)
    print("Testing code-switching with progressive audio chunks via Socket.IO")
    print("="*80)

    # Load audio files
    print("\n[1/5] Loading audio files...")
    chinese_audio = load_and_resample(f"{AUDIO_DIR}/OSR_cn_000_0072_8k.wav")
    english_audio = load_and_resample(f"{AUDIO_DIR}/jfk.wav")

    # Create mixed audio: Chinese (5s) + silence (0.5s) + English (5s)
    chinese_clip = chinese_audio[:int(5 * 16000)]
    english_clip = english_audio[:int(5 * 16000)]
    silence = np.zeros(int(0.5 * 16000), dtype=np.float32)

    mixed_audio = np.concatenate([chinese_clip, silence, english_clip])

    print(f"\n✓ Created mixed audio: {len(mixed_audio)/16000:.2f}s")
    print(f"   0.0s - 5.0s: Chinese")
    print(f"   5.0s - 5.5s: Silence")
    print(f"   5.5s - 10.5s: English")

    # Connect to Socket.IO
    print("\n[2/5] Connecting to Whisper service...")
    sio = socketio.Client()
    results = []

    @sio.on('connect')
    def on_connect():
        print("✅ Connected to Whisper service")

    @sio.on('transcription_result')
    def on_result(data):
        results.append(data)
        text = data.get('text', '')
        is_draft = data.get('is_draft', False)
        is_final = data.get('is_final', False)

        # Check for language metadata
        segments = data.get('segments', [])
        languages = set()
        if segments:
            for seg in segments:
                if isinstance(seg, dict) and 'detected_language' in seg:
                    languages.add(seg['detected_language'])

        status = "✏️ DRAFT" if is_draft else ("✅ FINAL" if is_final else "📝 UPDATE")
        print(f"\n{status} Result #{len(results)}:")
        print(f"   Text: '{text}'")
        if languages:
            print(f"   Languages detected: {languages}")
        print(f"   Segments: {len(segments)}")

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data.get('message')}")

    try:
        sio.connect(SERVICE_URL)
        time.sleep(0.5)

        session_id = f"code-switching-stream-test-{int(time.time())}"

        # Send configuration during join_session (not during transcribe_stream)
        join_config = {
            'model': 'base',
            'language': None,  # Auto-detect for code-switching
            'enable_vad': True,
            'enable_code_switching': True,  # ✅ CRITICAL: Must be in join_session config
            'target_language': 'en'
        }

        sio.emit('join_session', {
            'session_id': session_id,
            'config': join_config  # ✅ Pass config here, not in transcribe_stream
        })
        time.sleep(0.5)

        print(f"\n[3/5] Streaming mixed audio with code-switching enabled...")
        print(f"   Session ID: {session_id}")
        print(f"\n📝 Configuration:")
        print(f"   enable_code_switching: {join_config['enable_code_switching']}")
        print(f"   language: {join_config['language']} (auto-detect)")
        print(f"   model: {join_config['model']}")

        # Split audio into 2-second chunks (like production)
        chunk_duration = 2.0
        sample_rate = 16000
        chunk_samples = int(sample_rate * chunk_duration)
        num_chunks = int(np.ceil(len(mixed_audio) / chunk_samples))

        print(f"   Sending {num_chunks} chunks of {chunk_duration}s each")

        # Send chunks progressively
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, len(mixed_audio))

            chunk_audio = mixed_audio[start_idx:end_idx]
            chunk_int16 = (chunk_audio * 32768.0).astype(np.int16)
            chunk_b64 = base64.b64encode(chunk_int16.tobytes()).decode('utf-8')

            # Build request - config already set in session during join_session
            request_data = {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "sample_rate": sample_rate,
                "beam_size": 5,
            }

            chunk_start_time = start_idx / sample_rate
            chunk_end_time = end_idx / sample_rate

            sio.emit('transcribe_stream', request_data)
            print(f"📤 Sent chunk {i+1}/{num_chunks} ({chunk_start_time:.1f}s - {chunk_end_time:.1f}s)")
            time.sleep(2.5)  # Wait for processing

        # Wait for final results
        print("\n[4/5] Waiting for final results...")
        print("   (Waiting 10 seconds to ensure all audio is processed...)")
        time.sleep(10.0)

        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.5)
        sio.disconnect()

        # Analyze results
        print(f"\n[5/5] Analyzing results...")
        print("="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Total results received: {len(results)}")

        if results:
            print("\n📝 All transcription results:")
            for i, result in enumerate(results, 1):
                text = result.get('text', '')
                is_draft = result.get('is_draft', False)
                is_final = result.get('is_final', False)
                segments = result.get('segments', [])

                status = "DRAFT" if is_draft else ("FINAL" if is_final else "UPDATE")
                print(f"\n   Result {i}. [{status}]")
                print(f"      Text: '{text}'")
                print(f"      Segments: {len(segments)}")

                # Show segment details
                if segments:
                    for j, seg in enumerate(segments[:3], 1):  # Show first 3 segments
                        if isinstance(seg, dict):
                            seg_text = seg.get('text', '')
                            seg_lang = seg.get('detected_language', 'unknown')
                            seg_start = seg.get('start', 0)
                            seg_end = seg.get('end', 0)
                            lang_marker = "🇨🇳" if seg_lang == 'zh' else "🇺🇸" if seg_lang == 'en' else "❓"
                            print(f"         {lang_marker} [{j}] {seg_start:.1f}s-{seg_end:.1f}s | {seg_lang}: {seg_text[:40]}")

            # Combine all final/update results
            final_results = [r for r in results if not r.get('is_draft', False)]
            if final_results:
                full_text = ' '.join([r.get('text', '') for r in final_results])
                print(f"\n📄 Combined final transcription:")
                print(f"   '{full_text}'")

                # Check for both languages
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in full_text)
                has_english = any(word in full_text.lower() for word in ['and', 'so', 'fellow', 'american', 'ask', 'country'])

                print(f"\n🔍 Language detection:")
                print(f"   ✅ Chinese characters found: {has_chinese}")
                print(f"   ✅ English words found: {has_english}")

                # Check for code-switching in segments
                all_languages = set()
                for result in final_results:
                    for seg in result.get('segments', []):
                        if isinstance(seg, dict) and 'detected_language' in seg:
                            all_languages.add(seg['detected_language'])

                print(f"   Languages in segments: {all_languages}")

                # Verdict
                print(f"\n🎯 Code-switching verdict:")
                if has_chinese and has_english and len(all_languages) >= 2:
                    print(f"   ✅ SUCCESS: Both Chinese and English detected in streaming!")
                    print(f"   ✅ Code-switching WORKS in real streaming mode!")
                    success = True
                elif has_chinese and not has_english:
                    print(f"   ⚠️  PARTIAL: Only Chinese detected (English portion missing)")
                    print(f"   This suggests streaming stopped early or English chunks not processed")
                    success = False
                elif has_english and not has_chinese:
                    print(f"   ⚠️  PARTIAL: Only English detected (Chinese portion missing)")
                    success = False
                else:
                    print(f"   ❌ FAILED: Neither language properly detected")
                    success = False
            else:
                print("\n⚠️  No final results received (only drafts)")
                success = False
        else:
            print("\n❌ No results received at all")
            success = False

        print("\n" + "="*80)
        if success:
            print("✅ TEST PASSED - Streaming code-switching works!")
        else:
            print("❌ TEST FAILED - Code-switching not working in streaming mode")
        print("="*80 + "\n")

        return {"passed": success, "results": results}

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


if __name__ == "__main__":
    result = test_streaming_code_switching()
    exit(0 if result.get("passed") else 1)
