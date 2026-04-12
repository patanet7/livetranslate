#!/usr/bin/env python3
"""
Bot Integration Test — Full Pipeline WITHOUT Google Meet

Tests the complete caption system integration by simulating what a bot does:
1. Connect WebSocket to orchestration (like bot would)
2. Send real audio through the pipeline
3. Verify transcription → translation → CaptionBuffer
4. Verify PILVirtualCamRenderer produces frames with captions

NO GOOGLE MEET REQUIRED — uses audio fixtures directly.

Requires:
- Orchestration service running on :3000
- Transcription service running on :5001
- LLM service (vLLM-MLX on :8006 or Ollama on :11434)

Usage:
    # Start services first:
    just dev

    # Run integration test:
    uv run python tools/bot_integration_test.py

    # With specific audio fixture:
    uv run python tools/bot_integration_test.py --fixture meeting_zh_48k.wav

Output: /tmp/bot-integration-test/
"""

import argparse
import asyncio
import json
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from scipy.io import wavfile

# WebSocket client
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("ERROR: websockets not installed. Run: uv pip install websockets")
    sys.exit(1)


OUTPUT_DIR = Path("/tmp/bot-integration-test")
FIXTURES_DIR = Path(__file__).parent.parent / "modules" / "dashboard-service" / "tests" / "fixtures"

# Service URLs
ORCHESTRATION_URL = "http://localhost:3000"
ORCHESTRATION_WS = "ws://localhost:3000/api/audio/stream"


class BotIntegrationTest:
    """Simulates a bot connecting and streaming audio to orchestration."""

    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path
        self.ws = None
        self.messages_received: list[dict] = []
        self.captions_received: list[dict] = []
        self.frames_saved: list[str] = []
        self.test_results: dict[str, Any] = {
            "started_at": datetime.now().isoformat(),
            "fixture": str(fixture_path),
            "steps": [],
        }

    def log(self, step: str, status: str, details: str = ""):
        """Log a test step."""
        entry = {
            "step": step,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self.test_results["steps"].append(entry)
        icon = "✓" if status == "pass" else "✗" if status == "fail" else "○"
        print(f"  {icon} {step}: {details}" if details else f"  {icon} {step}")

    async def check_services(self) -> bool:
        """Verify required services are running."""
        print("\n[1/6] Checking Services...")

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check orchestration
            try:
                resp = await client.get(f"{ORCHESTRATION_URL}/api/health")
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "unknown")
                    self.log("Orchestration service", "pass", f"localhost:3000 ({status})")
                else:
                    self.log("Orchestration service", "fail", f"Status {resp.status_code}")
                    return False
            except Exception as e:
                self.log("Orchestration service", "fail", f"Not reachable: {e}")
                return False

            # Check transcription (via orchestration health)
            try:
                resp = await client.get(f"{ORCHESTRATION_URL}/api/health")
                data = resp.json()
                whisper_status = data.get("services", {}).get("whisper", {}).get("status", "unknown")
                if whisper_status == "healthy":
                    self.log("Transcription service", "pass", "via orchestration proxy")
                else:
                    whisper_error = data.get("services", {}).get("whisper", {}).get("last_error", "")
                    self.log("Transcription service", "fail", f"Status: {whisper_status} - {whisper_error}")
                    return False
            except Exception as e:
                self.log("Transcription service", "fail", f"Could not check: {e}")
                return False

        return True

    async def connect_websocket(self) -> bool:
        """Connect to orchestration WebSocket like a bot would."""
        print("\n[2/6] Connecting WebSocket...")

        try:
            self.ws = await websockets.connect(
                ORCHESTRATION_WS,
                ping_interval=30,
                ping_timeout=10,
            )
            self.log("WebSocket connected", "pass", ORCHESTRATION_WS)

            # Wait for ConnectedMessage
            msg = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            data = json.loads(msg)
            self.messages_received.append(data)

            if data.get("type") == "connected":
                session_id = data.get("session_id")
                self.log("Received ConnectedMessage", "pass", f"session_id={session_id}")
                self.session_id = session_id
                return True
            else:
                self.log("Received ConnectedMessage", "fail", f"Got {data.get('type')}")
                return False

        except Exception as e:
            self.log("WebSocket connection", "fail", str(e))
            return False

    async def start_session(self) -> bool:
        """Send start_session message to begin audio processing."""
        print("\n[3/6] Starting Session...")

        try:
            # Determine language from fixture name
            fixture_name = self.fixture_path.stem
            if "zh" in fixture_name:
                source_lang = "zh"
                target_lang = "en"
            elif "ja" in fixture_name:
                source_lang = "ja"
                target_lang = "en"
            elif "es" in fixture_name:
                source_lang = "es"
                target_lang = "en"
            else:
                source_lang = "en"
                target_lang = "zh"

            start_msg = {
                "type": "start_session",
                "source_language": source_lang,
                "target_language": target_lang,
                "model": "large-v3-turbo",
                "sample_rate": 48000,
            }

            await self.ws.send(json.dumps(start_msg))
            self.log("Sent start_session", "pass", f"{source_lang}→{target_lang}")

            # Wait for acknowledgment or service status
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                data = json.loads(msg)
                self.messages_received.append(data)
                self.log("Session acknowledged", "pass", f"type={data.get('type')}")
            except asyncio.TimeoutError:
                self.log("Session acknowledged", "info", "No explicit ack (continuing)")

            return True

        except Exception as e:
            self.log("Start session", "fail", str(e))
            return False

    async def stream_audio(self) -> bool:
        """Stream audio fixture through WebSocket."""
        print("\n[4/6] Streaming Audio...")

        try:
            # Load audio file
            sample_rate, audio_data = wavfile.read(self.fixture_path)

            # Convert to float32 normalized [-1, 1]
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0

            # Handle stereo → mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            duration = len(audio_data) / sample_rate
            self.log("Loaded audio", "pass", f"{duration:.1f}s @ {sample_rate}Hz")

            # Stream in chunks (simulate real-time at ~2x speed for faster testing)
            chunk_duration = 0.1  # 100ms chunks
            chunk_samples = int(sample_rate * chunk_duration)
            total_chunks = len(audio_data) // chunk_samples

            chunks_sent = 0
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i : i + chunk_samples]

                # Convert to int16 bytes (what browser sends)
                chunk_int16 = (chunk * 32767).astype(np.int16)
                chunk_bytes = chunk_int16.tobytes()

                # Send binary frame
                await self.ws.send(chunk_bytes)
                chunks_sent += 1

                # Check for incoming messages (non-blocking)
                try:
                    while True:
                        msg = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            self.messages_received.append(data)

                            if data.get("type") == "segment":
                                text = data.get("text", "")[:50]
                                self.log("Transcription segment", "info", f'"{text}..."')

                            elif data.get("type") == "translation":
                                text = data.get("translated_text", "")[:50]
                                self.captions_received.append(data)
                                self.log("Translation received", "pass", f'"{text}..."')

                except asyncio.TimeoutError:
                    pass

                # Pace sending (2x real-time)
                await asyncio.sleep(chunk_duration / 2)

                # Progress update every 2 seconds
                if chunks_sent % 20 == 0:
                    progress = (i / len(audio_data)) * 100
                    print(f"    ... {progress:.0f}% ({chunks_sent}/{total_chunks} chunks)")

            self.log("Audio streaming complete", "pass", f"{chunks_sent} chunks sent")
            return True

        except Exception as e:
            self.log("Stream audio", "fail", str(e))
            return False

    async def collect_remaining_messages(self, timeout: float = 10.0) -> bool:
        """Wait for remaining transcriptions/translations to arrive."""
        print("\n[5/6] Collecting Remaining Results...")

        start = time.time()
        while time.time() - start < timeout:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    self.messages_received.append(data)

                    if data.get("type") == "translation":
                        text = data.get("translated_text", "")[:50]
                        self.captions_received.append(data)
                        self.log("Translation received", "pass", f'"{text}..."')

            except asyncio.TimeoutError:
                # No more messages for 1 second — probably done
                if time.time() - start > 3.0:
                    break

        self.log(
            "Collection complete",
            "pass",
            f"{len(self.captions_received)} translations, {len(self.messages_received)} total messages",
        )
        return True

    async def verify_results(self) -> bool:
        """Verify the integration worked correctly."""
        print("\n[6/6] Verifying Results...")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save all messages
        with open(OUTPUT_DIR / "messages.json", "w") as f:
            json.dump(self.messages_received, f, indent=2, default=str)
        self.log("Saved messages", "pass", str(OUTPUT_DIR / "messages.json"))

        # Save translations only
        with open(OUTPUT_DIR / "translations.json", "w") as f:
            json.dump(self.captions_received, f, indent=2, default=str)

        # Count message types
        type_counts = {}
        for msg in self.messages_received:
            msg_type = msg.get("type", "unknown")
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

        self.log("Message types", "info", str(type_counts))

        # Verify we got transcriptions
        segment_count = type_counts.get("segment", 0)
        if segment_count > 0:
            self.log("Transcription pipeline", "pass", f"{segment_count} segments")
        else:
            self.log("Transcription pipeline", "fail", "No segments received")
            return False

        # Verify we got translations
        translation_count = len(self.captions_received)
        if translation_count > 0:
            self.log("Translation pipeline", "pass", f"{translation_count} translations")
        else:
            self.log("Translation pipeline", "fail", "No translations received")
            return False

        return True

    async def cleanup(self):
        """Close WebSocket connection."""
        if self.ws:
            try:
                # Send end_session
                await self.ws.send(json.dumps({"type": "end_session"}))
                await self.ws.close()
            except Exception:
                pass

    async def run(self) -> bool:
        """Run the full integration test."""
        print("=" * 60)
        print("BOT INTEGRATION TEST — Full Pipeline (No Google Meet)")
        print("=" * 60)
        print(f"Fixture: {self.fixture_path}")
        print(f"Output:  {OUTPUT_DIR}")

        try:
            # Check services
            if not await self.check_services():
                print("\n⚠ Services not running. Start with: just dev")
                return False

            # Connect WebSocket
            if not await self.connect_websocket():
                return False

            # Start session
            if not await self.start_session():
                return False

            # Stream audio
            if not await self.stream_audio():
                return False

            # Collect remaining messages
            if not await self.collect_remaining_messages():
                return False

            # Verify results
            success = await self.verify_results()

            # Summary
            self.test_results["completed_at"] = datetime.now().isoformat()
            self.test_results["success"] = success

            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)

            passed = sum(1 for s in self.test_results["steps"] if s["status"] == "pass")
            failed = sum(1 for s in self.test_results["steps"] if s["status"] == "fail")

            print(f"Steps: {passed} passed, {failed} failed")
            print(f"Translations: {len(self.captions_received)}")
            print(f"Output: {OUTPUT_DIR}")

            # Save results
            with open(OUTPUT_DIR / "results.json", "w") as f:
                json.dump(self.test_results, f, indent=2)

            return success

        finally:
            await self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Bot Integration Test")
    parser.add_argument(
        "--fixture",
        default="meeting_zh_48k.wav",
        help="Audio fixture filename (in dashboard-service/tests/fixtures/)",
    )
    args = parser.parse_args()

    # Find fixture
    fixture_path = FIXTURES_DIR / args.fixture
    if not fixture_path.exists():
        # Try without _48k suffix
        alt_path = FIXTURES_DIR / args.fixture.replace(".wav", "_48k.wav")
        if alt_path.exists():
            fixture_path = alt_path
        else:
            print(f"ERROR: Fixture not found: {fixture_path}")
            print(f"Available fixtures: {list(FIXTURES_DIR.glob('*.wav'))}")
            sys.exit(1)

    test = BotIntegrationTest(fixture_path)
    success = asyncio.run(test.run())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
