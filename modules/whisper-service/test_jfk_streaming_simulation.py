#!/usr/bin/env python3
"""
JFK Audio Streaming Simulation Test - Full Pipeline

This test simulates the complete streaming pipeline from orchestration → whisper service:
1. Loads JFK audio file
2. Splits into realistic streaming chunks (2-3 seconds)
3. Connects to Whisper service via Socket.IO (like orchestration service does)
4. Sends chunks with base64 encoding (exactly like orchestration)
5. Receives incremental results with draft/final markers
6. Validates VAC online processing and buffering system

This matches the exact flow used by:
- test_jfk_domain_prompts.py (Socket.IO streaming)
- Frontend Meeting Test Dashboard (orchestration → whisper)
- Google Meet bot integration (browser audio → orchestration → whisper)
"""

import socketio
import numpy as np
import base64
import time
import wave
import sys
import os

# Whisper service configuration
SERVICE_URL = "http://localhost:5001"
JFK_AUDIO_PATH = "/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/jfk.wav"

# Streaming configuration (matching real-world usage)
CHUNK_DURATION = 2.0  # 2-second chunks (realistic for live streaming)
ENABLE_VAD = True  # Enable Voice Activity Detection
ENABLE_DIARIZATION = False  # Not needed for JFK single-speaker test
MODEL_NAME = "large-v3-turbo"  # High-quality model


def load_jfk_audio():
    """
    Load JFK audio file and prepare for streaming

    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate
    """
    print("\n" + "="*80)
    print("LOADING JFK AUDIO")
    print("="*80)

    if not os.path.exists(JFK_AUDIO_PATH):
        print(f"❌ JFK file not found: {JFK_AUDIO_PATH}")
        return None, None

    try:
        with wave.open(JFK_AUDIO_PATH, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            n_channels = wav_file.getnchannels()
            duration = n_frames / sample_rate

            audio_bytes = wav_file.readframes(n_frames)
            audio = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert to float32 and normalize
            audio = audio.astype(np.float32) / 32768.0

            # Convert stereo to mono if needed
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
                print(f"✅ Converted stereo to mono")

            print(f"✅ Loaded JFK audio:")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Sample rate: {sample_rate}Hz")
            print(f"   Channels: {n_channels}")
            print(f"   Samples: {len(audio)}")

            return audio, sample_rate

    except Exception as e:
        print(f"❌ Error loading JFK audio: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def split_into_chunks(audio, sample_rate, chunk_duration=CHUNK_DURATION):
    """
    Split audio into streaming chunks (simulating real-time streaming)

    Args:
        audio: Audio data (numpy array)
        sample_rate: Sample rate in Hz
        chunk_duration: Chunk duration in seconds

    Returns:
        List of audio chunks
    """
    chunk_samples = int(sample_rate * chunk_duration)
    chunks = []

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) > 0:  # Only add non-empty chunks
            chunks.append(chunk)

    print(f"\n📊 Split audio into {len(chunks)} chunks of {chunk_duration}s each")
    return chunks


def test_streaming_simulation():
    """
    Test complete streaming simulation: orchestration → whisper service

    This test simulates the exact flow used in production:
    1. Socket.IO connection to whisper service
    2. Session creation with join_session
    3. Streaming chunks via transcribe_stream
    4. Receiving incremental results (draft/final)
    5. Session cleanup with leave_session
    """
    print("\n" + "="*80)
    print("JFK AUDIO STREAMING SIMULATION TEST")
    print("Simulating: Orchestration Service → Whisper Service")
    print("="*80)

    # Load JFK audio
    audio, sample_rate = load_jfk_audio()
    if audio is None:
        return {"passed": False, "error": "Failed to load JFK audio"}

    # Split into chunks (simulating streaming)
    chunks = split_into_chunks(audio, sample_rate, CHUNK_DURATION)

    # Connect to Socket.IO
    sio = socketio.Client()
    results = []
    connection_status = {"connected": False, "session_joined": False}

    @sio.on('connect')
    def on_connect():
        print(f"\n✅ Connected to Whisper service at {SERVICE_URL}")
        connection_status["connected"] = True

    @sio.on('session_joined')
    def on_session_joined(data):
        session_id = data.get('session_id')
        print(f"✅ Session joined: {session_id}")
        connection_status["session_joined"] = True

    @sio.on('transcription_result')
    def on_result(data):
        """Handle streaming transcription results (draft and final)"""
        results.append(data)

        text = data.get('text', '')
        is_draft = data.get('is_draft', False)
        is_final = data.get('is_final', False)
        stable_text = data.get('stable_text', '')
        unstable_text = data.get('unstable_text', '')
        stability_score = data.get('stability_score', 0.0)

        # Determine status
        if is_final:
            status = "✅ FINAL"
        elif is_draft:
            status = "✏️  DRAFT"
        else:
            status = "📝 UPDATE"

        print(f"\n{status} Result #{len(results)}:")
        if stable_text or unstable_text:
            # Phase 3C format (stable/unstable split)
            if stable_text:
                print(f"   Stable: '{stable_text}'")
            if unstable_text:
                print(f"   Unstable: '{unstable_text}' (score: {stability_score:.2f})")
        else:
            # Legacy format (full text)
            print(f"   Text: '{text}'")

        print(f"   is_draft={is_draft}, is_final={is_final}")

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data.get('message', data)}")

    @sio.on('disconnect')
    def on_disconnect():
        print(f"\n🔌 Disconnected from Whisper service")

    try:
        # Connect to service
        print(f"\n🔌 Connecting to Whisper service at {SERVICE_URL}...")
        sio.connect(SERVICE_URL)
        time.sleep(0.5)

        if not connection_status["connected"]:
            print("❌ Failed to connect to Whisper service")
            return {"passed": False, "error": "Connection failed"}

        # Create session
        session_id = f"jfk-streaming-test-{int(time.time())}"
        print(f"\n📋 Creating session: {session_id}")
        sio.emit('join_session', {'session_id': session_id})
        time.sleep(0.5)

        # Stream chunks
        print(f"\n🎙️  Streaming {len(chunks)} chunks...")
        print(f"   Model: {MODEL_NAME}")
        print(f"   Chunk duration: {CHUNK_DURATION}s")
        print(f"   VAD enabled: {ENABLE_VAD}")
        print("="*80)

        for i, chunk in enumerate(chunks):
            # Convert chunk to int16 and encode as base64 (like orchestration does)
            chunk_int16 = (chunk * 32768.0).astype(np.int16)
            chunk_b64 = base64.b64encode(chunk_int16.tobytes()).decode('utf-8')

            # Prepare request (matching orchestration service format)
            request_data = {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": MODEL_NAME,
                "language": "en",  # JFK speaks English
                "beam_size": 5,
                "sample_rate": sample_rate,
                "task": "transcribe",
                "target_language": "en",
                "enable_vad": ENABLE_VAD,
            }

            # Send chunk
            print(f"\n📤 Sending chunk {i+1}/{len(chunks)} ({len(chunk)/sample_rate:.2f}s, {len(chunk_int16.tobytes())} bytes)")
            sio.emit('transcribe_stream', request_data)

            # Wait for processing (simulate real-time streaming)
            # In real usage, chunks arrive at regular intervals
            time.sleep(1.5)  # Slightly less than chunk duration to simulate overlap

        # Wait for final results
        print(f"\n⏳ Waiting for final results...")
        print(f"   (Waiting 10 seconds to ensure all chunks are processed...)")
        time.sleep(10.0)

        # Leave session
        print(f"\n👋 Leaving session: {session_id}")
        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.5)

        # Disconnect
        sio.disconnect()

        # Analyze results
        print(f"\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Total results received: {len(results)}")

        if results:
            print(f"\n📝 Transcription results:")
            for i, result in enumerate(results, 1):
                text = result.get('text', '')
                stable_text = result.get('stable_text', '')
                unstable_text = result.get('unstable_text', '')
                is_draft = result.get('is_draft', False)
                is_final = result.get('is_final', False)

                status = "FINAL" if is_final else ("DRAFT" if is_draft else "UPDATE")

                if stable_text or unstable_text:
                    display_text = f"[STABLE] {stable_text} [UNSTABLE] {unstable_text}"
                else:
                    display_text = text

                print(f"   {i}. [{status}] '{display_text}'")

            # Combine all text
            full_text = ' '.join([
                r.get('stable_text') or r.get('text', '')
                for r in results
            ])

            print(f"\n📄 Full transcription:")
            print(f"   '{full_text}'")

            # Check for expected JFK keywords
            print(f"\n🔍 Keyword detection:")
            keywords = ["Americans", "country", "fellow", "ask"]
            found_count = 0
            for keyword in keywords:
                found = keyword.lower() in full_text.lower()
                status = "✅" if found else "❌"
                print(f"   {status} '{keyword}' found: {found}")
                if found:
                    found_count += 1

            # Check for incremental results (draft/final markers)
            draft_count = sum(1 for r in results if r.get('is_draft', False))
            final_count = sum(1 for r in results if r.get('is_final', False))

            print(f"\n📊 Streaming statistics:")
            print(f"   Total results: {len(results)}")
            print(f"   Draft results: {draft_count}")
            print(f"   Final results: {final_count}")
            print(f"   Keywords found: {found_count}/{len(keywords)}")

            # Success criteria
            passed = (
                len(results) > 0 and  # Got results
                found_count >= 2 and  # Found at least 2 JFK keywords
                (draft_count > 0 or final_count > 0)  # Got streaming markers
            )

            print(f"\n" + "="*80)
            if passed:
                print("✅ TEST PASSED - Streaming simulation successful!")
                print(f"   ✓ Received {len(results)} results")
                print(f"   ✓ Found {found_count}/{len(keywords)} JFK keywords")
                print(f"   ✓ Streaming markers present (draft={draft_count}, final={final_count})")
            else:
                print("❌ TEST FAILED - Streaming simulation incomplete")
                print(f"   Results: {len(results)}")
                print(f"   Keywords: {found_count}/{len(keywords)}")
                print(f"   Streaming markers: draft={draft_count}, final={final_count}")
            print("="*80 + "\n")

            return {
                "passed": passed,
                "results": results,
                "total_results": len(results),
                "draft_count": draft_count,
                "final_count": final_count,
                "keywords_found": found_count,
                "full_text": full_text
            }
        else:
            print("❌ No results received")
            print("="*80 + "\n")
            return {"passed": False, "error": "No results received"}

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


if __name__ == "__main__":
    result = test_streaming_simulation()
    exit(0 if result.get("passed") else 1)
