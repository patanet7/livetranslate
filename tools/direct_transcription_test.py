#!/usr/bin/env python3
"""
Direct Transcription Service Test

Tests the transcription WebSocket endpoint directly with proper audio format.
This bypasses orchestration to isolate transcription service behavior.

Key insight: api.py:596 expects float32 bytes, not int16!

Usage:
    # Start transcription service first:
    uv run python modules/transcription-service/src/main.py --registry modules/transcription-service/config/model_registry.local.yaml

    # Run this test:
    uv run python tools/direct_transcription_test.py
"""

import asyncio
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: uv pip install websockets")
    sys.exit(1)


TRANSCRIPTION_WS = "ws://localhost:5001/api/stream"
FIXTURES_DIR = Path(__file__).parent.parent / "modules" / "dashboard-service" / "tests" / "fixtures"


async def test_transcription_direct():
    """Send audio directly to transcription service and log all responses."""

    # Find a fixture
    fixture_path = FIXTURES_DIR / "meeting_zh_48k.wav"
    if not fixture_path.exists():
        fixture_path = list(FIXTURES_DIR.glob("*.wav"))[0] if FIXTURES_DIR.exists() else None

    if not fixture_path:
        print(f"ERROR: No fixtures found in {FIXTURES_DIR}")
        return False

    print(f"Using fixture: {fixture_path}")

    # Load audio
    sample_rate, audio_data = wavfile.read(fixture_path)
    print(f"Loaded: {len(audio_data)} samples @ {sample_rate}Hz = {len(audio_data)/sample_rate:.1f}s")

    # Convert to float32 normalized [-1, 1]
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Handle stereo → mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed (transcription expects 16kHz)
    if sample_rate != 16000:
        try:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
            print(f"Resampled to 16kHz: {len(audio_data)} samples = {len(audio_data)/16000:.1f}s")
        except ImportError:
            print("WARNING: librosa not available, using naive downsampling")
            factor = sample_rate // 16000
            audio_data = audio_data[::factor]
            sample_rate = 16000

    # RMS check before sending
    rms = np.sqrt(np.mean(audio_data ** 2))
    print(f"Audio RMS: {rms:.4f} (threshold is 0.02, speech should be > 0.02)")

    print(f"\nConnecting to {TRANSCRIPTION_WS}...")

    try:
        async with websockets.connect(TRANSCRIPTION_WS, ping_interval=30) as ws:
            print("✓ Connected")

            # Send config message
            config_msg = {
                "type": "config",
                "language": "zh",
                "lock_language": False,
            }
            await ws.send(json.dumps(config_msg))
            print(f"✓ Sent config: {config_msg}")

            # Track results
            segments_received = []
            errors_received = []

            # Start receiver task
            async def receiver():
                try:
                    while True:
                        msg = await ws.recv()
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            msg_type = data.get("type")

                            if msg_type == "segment":
                                segments_received.append(data)
                                text = data.get("text", "")[:60]
                                is_draft = data.get("is_draft", False)
                                seg_id = data.get("segment_id", "?")
                                print(f"  📝 Segment {seg_id} {'(draft)' if is_draft else '(final)'}: \"{text}\"")

                            elif msg_type == "error":
                                errors_received.append(data)
                                print(f"  ❌ Error: {data.get('message')}")

                            elif msg_type == "backend_switched":
                                print(f"  🔄 Backend: {data.get('backend')}:{data.get('model')}")

                            else:
                                print(f"  📨 {msg_type}: {data}")
                except websockets.ConnectionClosed:
                    pass

            receiver_task = asyncio.create_task(receiver())

            # Stream audio in chunks
            # VAC config: prebuffer=1s, stride=6s → need at least 1s for first inference
            # Send in 100ms chunks like the browser does
            chunk_duration = 0.1  # 100ms
            chunk_samples = int(sample_rate * chunk_duration)

            print(f"\nStreaming {len(audio_data)/sample_rate:.1f}s of audio in {chunk_duration*1000:.0f}ms chunks...")
            print(f"  - Prebuffer: 1.0s (first inference after ~1s)")
            print(f"  - Stride: 6.0s (subsequent inferences every ~6s)")

            chunks_sent = 0
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i : i + chunk_samples]

                # CRITICAL: Send as float32 bytes, not int16!
                chunk_bytes = chunk.astype(np.float32).tobytes()
                await ws.send(chunk_bytes)
                chunks_sent += 1

                # Pace at real-time (or slightly faster)
                await asyncio.sleep(chunk_duration * 0.5)

                # Progress every 2s
                if chunks_sent % 20 == 0:
                    elapsed = (i / sample_rate)
                    print(f"    ... {elapsed:.1f}s sent ({chunks_sent} chunks), {len(segments_received)} segments received")

            print(f"\n✓ Finished streaming {chunks_sent} chunks")

            # Send end message
            await ws.send(json.dumps({"type": "end"}))
            print("✓ Sent end message")

            # Wait for final results
            print("Waiting for final results...")
            await asyncio.sleep(5.0)

            # Cancel receiver
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

            # Summary
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Segments received: {len(segments_received)}")
            print(f"Errors received: {len(errors_received)}")

            if segments_received:
                print("\nAll segments:")
                for seg in segments_received:
                    seg_id = seg.get("segment_id", "?")
                    is_draft = seg.get("is_draft", False)
                    text = seg.get("text", "")
                    lang = seg.get("language", "?")
                    print(f"  [{seg_id}] {'D' if is_draft else 'F'} ({lang}): {text[:80]}")
                return True
            else:
                print("\n❌ No segments received!")
                if errors_received:
                    print("Errors:")
                    for err in errors_received:
                        print(f"  - {err}")
                return False

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def main():
    print("=" * 60)
    print("DIRECT TRANSCRIPTION SERVICE TEST")
    print("=" * 60)

    success = asyncio.run(test_transcription_direct())

    print(f"\n{'✓ Test passed' if success else '✗ Test failed'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
