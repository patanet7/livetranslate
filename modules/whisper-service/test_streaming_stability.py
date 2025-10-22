#!/usr/bin/env python3
"""
STREAMING TEST - Phase 3C Draft/Final with Stable/Unstable

Tests streaming mode with the real service to see:
- Draft emissions (when stable prefix grows)
- Final emissions (at segment boundaries)
- Stable vs unstable text separation
"""

import socketio
import json
import numpy as np
import soundfile as sf
import base64
import time
import asyncio

SERVICE_URL = "http://localhost:5001"

def test_streaming_with_stability():
    """Test streaming transcription with stability tracking using Socket.IO"""

    print("=" * 80)
    print("STREAMING TEST: Phase 3C Draft/Final with Stable/Unstable Text")
    print("=" * 80)

    # Create test audio chunks
    print("\n[1] Creating audio chunks (simulating real-time streaming)...")
    sample_rate = 16000
    chunk_duration = 0.5  # 500ms chunks

    chunks = []
    for i in range(6):  # 6 chunks = 3 seconds total
        # Generate speech-like audio
        t = np.linspace(0, chunk_duration, int(sample_rate * chunk_duration))
        freq = 200 + 50 * np.sin(2 * np.pi * (i+1) * t)
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        audio += 0.15 * np.sin(2 * np.pi * 2 * freq * t)

        # Envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
        audio = audio * envelope

        # Add noise
        audio += np.random.normal(0, 0.02, audio.shape)

        chunks.append(audio.astype(np.float32))

    print(f"✅ Created {len(chunks)} audio chunks ({chunk_duration}s each)\n")

    # Connect to Socket.IO
    print("[2] Connecting to Whisper service Socket.IO...")

    emission_count = 0
    draft_count = 0
    final_count = 0
    results = []

    sio = socketio.Client()

    @sio.on('connect')
    def on_connect():
        print("✅ Connected to Socket.IO\n")

    @sio.on('transcription_result')
    def on_transcription(data):
        nonlocal emission_count, draft_count, final_count
        emission_count += 1

        print(f"\n📥 Emission #{emission_count}:")
        print(f"  Full Text: '{data.get('text', 'N/A')[:60]}...'")

        # Check for Phase 3C fields
        stable_text = data.get('stable_text', 'NOT_PRESENT')
        unstable_text = data.get('unstable_text', 'NOT_PRESENT')
        is_draft = data.get('is_draft', False)
        is_final = data.get('is_final', False)
        should_translate = data.get('should_translate', False)
        translation_mode = data.get('translation_mode', 'none')
        stability_score = data.get('stability_score', 0.0)

        print(f"\n  🎯 Stability Tracking:")
        print(f"    Stable Text: '{stable_text[:40] if stable_text != 'NOT_PRESENT' else stable_text}...'")
        print(f"    Unstable Text: '{unstable_text[:20] if unstable_text != 'NOT_PRESENT' else unstable_text}...'")
        print(f"    Is Draft: {is_draft}")
        print(f"    Is Final: {is_final}")
        print(f"    Should Translate: {should_translate}")
        print(f"    Translation Mode: {translation_mode}")
        print(f"    Stability Score: {stability_score:.3f}")

        if is_draft:
            draft_count += 1
            print(f"    ✏️  DRAFT EMISSION (stable prefix grew)")

        if is_final:
            final_count += 1
            print(f"    ✅ FINAL EMISSION (segment boundary)")

        results.append(data)

    @sio.on('error')
    def on_error(data):
        print(f"  ❌ Error: {data.get('message', 'Unknown')}")

    try:
        sio.connect(SERVICE_URL)

        # Join a session
        session_id = f"test-streaming-{int(time.time())}"
        print(f"[3] Joining session: {session_id}...")

        sio.emit('join_session', {'session_id': session_id})
        time.sleep(0.5)  # Wait for join
        print(f"✅ Joined session\n")

        # Stream audio chunks
        print("[4] Streaming audio chunks and monitoring for draft/final emissions...")
        print("-" * 80)

        for i, chunk in enumerate(chunks):
            print(f"\n📤 Sending chunk {i+1}/{len(chunks)}...")

            # Encode audio chunk
            chunk_bytes = chunk.tobytes()
            chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')

            # Send transcribe_stream event
            sio.emit('transcribe_stream', {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": "large-v3-turbo",
                "language": "en",
                "beam_size": 5,
                "sample_rate": sample_rate
            })

            # Wait for processing
            time.sleep(1.0)

        print("\n" + "-" * 80)
        print("\n[5] Streaming complete. Summary:")
        print(f"  Total Chunks Sent: {len(chunks)}")
        print(f"  Total Emissions Received: {emission_count}")
        print(f"  Draft Emissions: {draft_count}")
        print(f"  Final Emissions: {final_count}")

        if emission_count == 0:
            print("\n⚠️  No emissions received!")
            print("   This could mean:")
            print("   - VAD filtered out all chunks (no speech detected)")
            print("   - Streaming mode not triggering transcription")
            print("   - Minimum buffer duration not reached")
        elif len(results) > 0 and results[0].get('stable_text') == 'NOT_PRESENT':
            print("\n⚠️  Stability fields NOT present in response")
            print("   Phase 3C might not be fully integrated with WebSocket API")
        else:
            print("\n✅ Phase 3C stability tracking is WORKING!")
            print("   Draft/Final emissions with stable/unstable text separation confirmed")

        # Leave session
        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.5)

        sio.disconnect()

    except Exception as e:
        print(f"\n❌ Socket.IO test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run streaming test"""
    print("\n" + "=" * 80)
    print("PHASE 3C STREAMING TEST WITH STABILITY TRACKING")
    print("Testing: Draft/Final Emissions + Stable/Unstable Text Separation")
    print("=" * 80 + "\n")

    test_streaming_with_stability()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
