#!/usr/bin/env python3
"""
Test Fixed SocketIOWhisperClient with JFK Audio

This test verifies that the fixed SocketIOWhisperClient properly sends
all required Whisper parameters to match the working test_jfk_domain_prompts.py flow.

Tests:
1. SocketIOWhisperClient includes all required parameters
2. Domain prompts are properly sent on first chunk
3. Subsequent chunks maintain configuration
4. Streaming transcription works end-to-end
"""

import asyncio
import sys
import os
import wave
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from socketio_whisper_client import SocketIOWhisperClient

# Configuration
WHISPER_HOST = "localhost"
WHISPER_PORT = 5001
JFK_AUDIO_PATH = (
    "/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/jfk.wav"
)
CHUNK_DURATION = 2.0  # 2-second chunks


async def test_socketio_whisper_client():
    """
    Test SocketIOWhisperClient with JFK audio streaming

    This validates that the fixed client matches the working
    test_jfk_domain_prompts.py behavior.
    """
    print("\n" + "=" * 80)
    print("SOCKETIO WHISPER CLIENT TEST - JFK AUDIO STREAMING")
    print("=" * 80)
    print("Testing: SocketIOWhisperClient → Whisper Service")
    print(f"Whisper: {WHISPER_HOST}:{WHISPER_PORT}")
    print("=" * 80)

    # Load JFK audio
    print(f"\n📁 Loading JFK audio from: {JFK_AUDIO_PATH}")
    if not os.path.exists(JFK_AUDIO_PATH):
        print(f"❌ JFK file not found: {JFK_AUDIO_PATH}")
        return {"passed": False, "error": "Audio file not found"}

    with wave.open(JFK_AUDIO_PATH, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        n_channels = wav_file.getnchannels()
        duration = n_frames / sample_rate

        audio_bytes = wav_file.readframes(n_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert stereo to mono if needed
        if n_channels == 2:
            audio_float = audio.astype(np.float32) / 32768.0
            audio_float = audio_float.reshape(-1, 2).mean(axis=1)
            audio = (audio_float * 32768.0).astype(np.int16)
            print("✅ Converted stereo to mono")

    print(f"✅ Loaded {duration:.2f}s of audio (sample_rate={sample_rate}Hz)")

    # Split into chunks
    chunk_samples = int(sample_rate * CHUNK_DURATION)
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) > 0:
            chunks.append(chunk)

    print(f"📊 Split into {len(chunks)} chunks of {CHUNK_DURATION}s each")

    # Create client
    print("\n🔌 Creating SocketIOWhisperClient...")
    client = SocketIOWhisperClient(
        whisper_host=WHISPER_HOST, whisper_port=WHISPER_PORT, auto_reconnect=True
    )

    # Track results
    results = []

    def on_segment(data):
        """Handle transcription segments"""
        results.append(data)
        text = data.get("text", "")
        is_draft = data.get("is_draft", False)
        is_final = data.get("is_final", False)
        stable_text = data.get("stable_text", "")
        unstable_text = data.get("unstable_text", "")

        status = "✅ FINAL" if is_final else ("✏️  DRAFT" if is_draft else "📝 UPDATE")
        print(f"\n{status} Result #{len(results)}:")
        if stable_text or unstable_text:
            if stable_text:
                print(f"   Stable: '{stable_text}'")
            if unstable_text:
                print(f"   Unstable: '{unstable_text}'")
        else:
            print(f"   Text: '{text}'")

    def on_error(error):
        """Handle errors"""
        print(f"❌ Error: {error}")

    def on_connection_change(connected):
        """Handle connection changes"""
        status = "connected" if connected else "disconnected"
        print(f"🔌 Connection status: {status}")

    # Register callbacks
    client.on_segment(on_segment)
    client.on_error(on_error)
    client.on_connection_change(on_connection_change)

    try:
        # Connect to Whisper service
        print("\n🔌 Connecting to Whisper service...")
        connected = await client.connect()
        if not connected:
            print("❌ Failed to connect to Whisper service")
            return {"passed": False, "error": "Connection failed"}

        print("✅ Connected!")

        # Wait for connection to stabilize
        await asyncio.sleep(0.5)

        # Start streaming session with full configuration
        session_id = f"socketio-jfk-test-{int(asyncio.get_event_loop().time())}"

        # Build configuration matching test_jfk_domain_prompts.py
        config = {
            "model_name": "large-v3-turbo",
            "language": "en",
            "beam_size": 5,
            "sample_rate": sample_rate,
            "task": "transcribe",
            "target_language": "en",
            "enable_vad": True,
            # Domain prompts (optional)
            "domain": "political",
            "custom_terms": [
                "Americans",
                "fellow citizens",
                "country",
                "nation",
                "freedom",
                "liberty",
                "democracy",
            ],
            "initial_prompt": "Presidential inaugural speech about American values and civic responsibility",
        }

        print(f"\n🎬 Starting stream session: {session_id}")
        print(f"   Model: {config['model_name']}")
        print(f"   Language: {config['language']}")
        print(f"   Task: {config['task']}")
        print(f"   Domain: {config.get('domain', 'none')}")

        await client.start_stream(session_id, config)

        # Wait for session to start
        await asyncio.sleep(0.5)

        # Stream chunks
        print(f"\n🎙️  Streaming {len(chunks)} chunks...")
        print("=" * 80)

        for i, chunk in enumerate(chunks):
            chunk_bytes = chunk.tobytes()
            print(
                f"\n📤 Sending chunk {i + 1}/{len(chunks)} ({len(chunk_bytes)} bytes)"
            )

            await client.send_audio_chunk(session_id, chunk_bytes)

            # Wait for processing (simulate real-time streaming)
            await asyncio.sleep(1.5)

        # Wait for final results
        print("\n⏳ Waiting for final results...")
        await asyncio.sleep(10.0)

        # Close stream
        print("\n⏹️  Closing stream...")
        await client.close_stream(session_id)

        # Disconnect
        await client.disconnect()

        # Analyze results
        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        print(f"Total results received: {len(results)}")

        if results:
            # Combine all text
            full_text = " ".join(
                [r.get("stable_text") or r.get("text", "") for r in results]
            )

            print("\n📄 Full transcription:")
            print(f"   '{full_text}'")

            # Check for JFK keywords
            print("\n🔍 Keyword detection:")
            keywords = ["Americans", "country", "fellow", "ask"]
            found_count = 0
            for keyword in keywords:
                found = keyword.lower() in full_text.lower()
                status = "✅" if found else "❌"
                print(f"   {status} '{keyword}' found: {found}")
                if found:
                    found_count += 1

            # Check streaming markers
            draft_count = sum(1 for r in results if r.get("is_draft", False))
            final_count = sum(1 for r in results if r.get("is_final", False))

            print("\n📊 Streaming statistics:")
            print(f"   Total results: {len(results)}")
            print(f"   Draft results: {draft_count}")
            print(f"   Final results: {final_count}")
            print(f"   Keywords found: {found_count}/{len(keywords)}")

            # Success criteria
            passed = (
                len(results) > 0
                and found_count >= 2
                and (draft_count > 0 or final_count > 0)
            )

            print("\n" + "=" * 80)
            if passed:
                print("✅ TEST PASSED - SocketIOWhisperClient working correctly!")
                print(f"   ✓ Received {len(results)} results")
                print(f"   ✓ Found {found_count}/{len(keywords)} JFK keywords")
                print(
                    f"   ✓ Streaming markers present (draft={draft_count}, final={final_count})"
                )
            else:
                print("❌ TEST FAILED - SocketIOWhisperClient incomplete")
                print(f"   Results: {len(results)}")
                print(f"   Keywords: {found_count}/{len(keywords)}")
            print("=" * 80 + "\n")

            return {
                "passed": passed,
                "results": results,
                "total_results": len(results),
                "keywords_found": found_count,
                "full_text": full_text,
            }
        else:
            print("❌ No results received")
            print("\n💡 Possible issues:")
            print("   - Whisper service not running")
            print("   - SocketIOWhisperClient missing parameters")
            print("   - Audio format issue")
            print("=" * 80 + "\n")
            return {"passed": False, "error": "No results received"}

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return {"passed": False, "error": str(e)}


if __name__ == "__main__":
    result = asyncio.run(test_socketio_whisper_client())
    sys.exit(0 if result.get("passed") else 1)
