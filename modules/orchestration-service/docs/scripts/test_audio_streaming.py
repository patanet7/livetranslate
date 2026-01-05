#!/usr/bin/env python3
"""
Audio Streaming Test Script

Tests the complete audio pipeline using ffmpeg input:
ffmpeg → orchestration service → Whisper service → transcription

This follows the SAME pattern as bot containers, ensuring we test the real production path.

Usage:
    # Test with microphone input (real-time)
    python test_audio_streaming.py --mic

    # Test with audio file
    python test_audio_streaming.py --file audio.wav

    # Test with custom ffmpeg input
    ffmpeg -f avfoundation -i ":0" -f s16le -ar 16000 -ac 1 - | python test_audio_streaming.py --stdin
"""

import asyncio
import websockets
import json
import sys
import argparse
import base64
import subprocess
import time
from datetime import datetime
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioStreamingTester:
    """
    Test client for orchestration service audio streaming.

    Follows the exact same pattern as bot containers:
    - Connects to orchestration service via WebSocket
    - Sends audio chunks with same format
    - Receives transcription segments
    """

    def __init__(
        self,
        orchestration_url: str = "ws://localhost:3000/api/audio/stream",
        chunk_duration_ms: int = 100,
        session_id: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize audio streaming tester

        Args:
            orchestration_url: WebSocket URL for orchestration service
            chunk_duration_ms: Audio chunk duration in milliseconds
            session_id: Session identifier (auto-generated if None)
            config: Whisper configuration (model, language, etc.)
        """
        self.orchestration_url = orchestration_url
        self.chunk_duration_ms = chunk_duration_ms
        self.session_id = session_id or f"test-{int(time.time())}"
        self.config = config or {
            "model": "large-v3-turbo",
            "language": "en",
            "enable_vad": True,
            "enable_diarization": True,
            "enable_cif": True,
            "enable_rolling_context": True,
        }

        self.ws = None
        self.connected = False
        self.authenticated = False
        self.session_started = False

        # Statistics
        self.chunks_sent = 0
        self.segments_received = 0
        self.start_time = None
        self.last_segment_time = None
        self.session_ended = False

    async def connect(self):
        """Connect to orchestration service WebSocket"""
        logger.info(f"🔌 Connecting to {self.orchestration_url}")

        try:
            self.ws = await websockets.connect(self.orchestration_url)
            self.connected = True
            logger.info("✅ Connected to orchestration service")

            # Wait for welcome message
            welcome = await self.ws.recv()
            welcome_msg = json.loads(welcome)
            logger.info(f"📨 Received: {welcome_msg}")

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            raise

    async def authenticate(self, user_id: str = "test-user", token: str = "test-token"):
        """Authenticate with orchestration service"""
        logger.info(f"🔐 Authenticating as {user_id}")

        auth_msg = {"type": "authenticate", "user_id": user_id, "token": token}

        await self.ws.send(json.dumps(auth_msg))

        # Wait for authenticated response
        response = await self.ws.recv()
        response_msg = json.loads(response)

        if response_msg.get("type") == "authenticated":
            self.authenticated = True
            logger.info(f"✅ Authenticated: {response_msg}")
        else:
            logger.error(f"❌ Authentication failed: {response_msg}")
            raise RuntimeError("Authentication failed")

    async def start_session(self):
        """Start streaming session"""
        logger.info(f"🎬 Starting session: {self.session_id}")
        logger.info(f"📝 Config: {self.config}")

        start_msg = {
            "type": "start_session",
            "session_id": self.session_id,
            "config": self.config,
        }

        await self.ws.send(json.dumps(start_msg))

        # Wait for session_started response
        response = await self.ws.recv()
        response_msg = json.loads(response)

        if response_msg.get("type") == "session_started":
            self.session_started = True
            self.start_time = time.time()
            logger.info(f"✅ Session started: {response_msg}")
        else:
            logger.error(f"❌ Session start failed: {response_msg}")
            raise RuntimeError("Session start failed")

    async def send_audio_chunk(self, audio_data: bytes):
        """
        Send audio chunk to orchestration service

        Args:
            audio_data: Raw audio bytes (16kHz, mono, S16LE)
        """
        if not self.session_started:
            raise RuntimeError("Session not started")

        # Encode audio to base64 (same as frontend and bots)
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        chunk_msg = {
            "type": "audio_chunk",
            "audio": audio_base64,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.ws.send(json.dumps(chunk_msg))
        self.chunks_sent += 1

        if self.chunks_sent % 10 == 0:
            elapsed = time.time() - self.start_time
            logger.debug(f"🎵 Sent {self.chunks_sent} chunks ({elapsed:.1f}s)")

    async def end_session(self):
        """End streaming session"""
        logger.info(f"⏹️ Ending session: {self.session_id}")

        end_msg = {"type": "end_session", "session_id": self.session_id}

        await self.ws.send(json.dumps(end_msg))

        # Don't wait for response here - let receive_messages() handle it
        # to avoid WebSocket concurrency error
        self.session_started = False
        logger.info(f"✅ Session end message sent")

    async def disconnect(self):
        """Disconnect from orchestration service"""
        if self.ws:
            await self.ws.close()
            self.connected = False
            logger.info("🔌 Disconnected from orchestration service")

    async def receive_messages(self):
        """
        Receive and process messages from orchestration service

        Runs in background to handle transcription segments
        """
        try:
            async for message in self.ws:
                msg = json.loads(message)
                msg_type = msg.get("type")

                if msg_type == "segment":
                    self.segments_received += 1
                    self.last_segment_time = time.time()
                    self._print_segment(msg)

                elif msg_type == "translation":
                    self._print_translation(msg)

                elif msg_type == "error":
                    logger.error(f"❌ Error from server: {msg.get('error')}")

                elif msg_type == "ping":
                    # Respond to ping
                    await self.ws.send(json.dumps({"type": "pong"}))

                elif msg_type == "session_ended":
                    logger.info(f"✅ Session ended: {msg}")

                else:
                    logger.debug(f"📨 Received: {msg_type}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Connection closed by server")
        except Exception as e:
            logger.error(f"❌ Error receiving messages: {e}")

    def _print_segment(self, segment: dict):
        """Pretty print transcription segment"""
        text = segment.get("text", "")
        speaker = segment.get("speaker", "Unknown")
        confidence = segment.get("confidence", 0) * 100
        is_final = segment.get("is_final", False)
        language = segment.get("detected_language", "unknown")

        status = "✅ FINAL" if is_final else "⏳ PARTIAL"

        print(f"\n{'=' * 80}")
        print(f"{status} | 👤 {speaker} | 🌐 {language.upper()} | 📊 {confidence:.1f}%")
        print(f"📝 {text}")
        print(f"{'=' * 80}")

    def _print_translation(self, translation: dict):
        """Pretty print translation"""
        text = translation.get("text", "")
        source_lang = translation.get("source_lang", "unknown")
        target_lang = translation.get("target_lang", "unknown")
        confidence = translation.get("confidence", 0) * 100

        print(f"\n{'=' * 80}")
        print(
            f"🌐 TRANSLATION | {source_lang.upper()} → {target_lang.upper()} | 📊 {confidence:.1f}%"
        )
        print(f"📝 {text}")
        print(f"{'=' * 80}")

    def print_statistics(self):
        """Print streaming statistics"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            chunks_per_sec = self.chunks_sent / elapsed if elapsed > 0 else 0

            print(f"\n{'=' * 80}")
            print(f"📊 STREAMING STATISTICS")
            print(f"{'=' * 80}")
            print(f"Duration:          {elapsed:.1f}s")
            print(f"Chunks sent:       {self.chunks_sent}")
            print(f"Segments received: {self.segments_received}")
            print(f"Chunk rate:        {chunks_per_sec:.1f} chunks/sec")
            print(f"{'=' * 80}\n")


async def stream_from_microphone(tester: AudioStreamingTester):
    """
    Stream audio from microphone using ffmpeg

    MacOS: ffmpeg -f avfoundation -i ":0" -f s16le -ar 16000 -ac 1 -
    Linux: ffmpeg -f alsa -i default -f s16le -ar 16000 -ac 1 -
    Windows: ffmpeg -f dshow -i audio="Microphone" -f s16le -ar 16000 -ac 1 -
    """
    logger.info("🎤 Starting microphone capture with ffmpeg")

    # Detect platform and choose appropriate ffmpeg input
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "avfoundation",
            "-i",
            ":0",  # Default audio input
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-loglevel",
            "error",
            "-",
        ]
    elif system == "Linux":
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "alsa",
            "-i",
            "default",
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-loglevel",
            "error",
            "-",
        ]
    elif system == "Windows":
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "dshow",
            "-i",
            "audio=Microphone",
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-loglevel",
            "error",
            "-",
        ]
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    logger.info(f"🎵 ffmpeg command: {' '.join(ffmpeg_cmd)}")

    # Start ffmpeg process
    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
    )

    # Calculate chunk size
    # 16kHz, 16-bit (2 bytes), mono = 32000 bytes/sec
    # For 100ms chunks: 32000 * 0.1 = 3200 bytes
    sample_rate = 16000
    bytes_per_sample = 2  # S16LE
    channels = 1
    chunk_size = int(
        sample_rate * bytes_per_sample * channels * tester.chunk_duration_ms / 1000
    )

    logger.info(f"📦 Chunk size: {chunk_size} bytes ({tester.chunk_duration_ms}ms)")

    try:
        while True:
            audio_chunk = process.stdout.read(chunk_size)

            if not audio_chunk:
                break

            if len(audio_chunk) < chunk_size:
                # Pad with zeros if incomplete chunk
                audio_chunk += b"\x00" * (chunk_size - len(audio_chunk))

            await tester.send_audio_chunk(audio_chunk)

            # Small delay to match real-time streaming
            await asyncio.sleep(tester.chunk_duration_ms / 1000)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping microphone capture")

    finally:
        process.terminate()
        process.wait()


async def stream_from_file(tester: AudioStreamingTester, audio_file: str):
    """
    Stream audio from file using ffmpeg

    Converts any audio format to 16kHz mono S16LE
    """
    logger.info(f"📁 Streaming from file: {audio_file}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        audio_file,
        "-f",
        "s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-loglevel",
        "error",
        "-",
    ]

    logger.info(f"🎵 ffmpeg command: {' '.join(ffmpeg_cmd)}")

    # Start ffmpeg process
    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
    )

    # Calculate chunk size
    sample_rate = 16000
    bytes_per_sample = 2  # S16LE
    channels = 1
    chunk_size = int(
        sample_rate * bytes_per_sample * channels * tester.chunk_duration_ms / 1000
    )

    logger.info(f"📦 Chunk size: {chunk_size} bytes ({tester.chunk_duration_ms}ms)")

    try:
        while True:
            audio_chunk = process.stdout.read(chunk_size)

            if not audio_chunk:
                break

            if len(audio_chunk) < chunk_size:
                # Pad with zeros if incomplete chunk
                audio_chunk += b"\x00" * (chunk_size - len(audio_chunk))

            await tester.send_audio_chunk(audio_chunk)

            # Real-time simulation: wait for chunk duration
            await asyncio.sleep(tester.chunk_duration_ms / 1000)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping file streaming")

    finally:
        process.terminate()
        process.wait()
        logger.info("✅ File streaming complete")


async def stream_from_stdin(tester: AudioStreamingTester):
    """
    Stream audio from stdin

    Expects raw S16LE audio at 16kHz mono
    Usage: ffmpeg -i input.mp3 -f s16le -ar 16000 -ac 1 - | python test_audio_streaming.py --stdin
    """
    logger.info("📥 Reading audio from stdin")

    # Calculate chunk size
    sample_rate = 16000
    bytes_per_sample = 2  # S16LE
    channels = 1
    chunk_size = int(
        sample_rate * bytes_per_sample * channels * tester.chunk_duration_ms / 1000
    )

    logger.info(f"📦 Chunk size: {chunk_size} bytes ({tester.chunk_duration_ms}ms)")

    try:
        while True:
            audio_chunk = sys.stdin.buffer.read(chunk_size)

            if not audio_chunk:
                break

            if len(audio_chunk) < chunk_size:
                # Pad with zeros if incomplete chunk
                audio_chunk += b"\x00" * (chunk_size - len(audio_chunk))

            await tester.send_audio_chunk(audio_chunk)

            # Real-time simulation: wait for chunk duration
            await asyncio.sleep(tester.chunk_duration_ms / 1000)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping stdin streaming")

    finally:
        logger.info("✅ Stdin streaming complete")


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(
        description="Test audio streaming through orchestration service (same pattern as bots)"
    )

    # Input source options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--mic", action="store_true", help="Stream from microphone"
    )
    input_group.add_argument("--file", type=str, help="Stream from audio file")
    input_group.add_argument("--stdin", action="store_true", help="Stream from stdin")

    # Configuration options
    parser.add_argument(
        "--url",
        default="ws://localhost:3000/api/audio/stream",
        help="Orchestration WebSocket URL",
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=100,
        help="Audio chunk duration in milliseconds",
    )
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model to use (default: large-v3-turbo)",
    )
    parser.add_argument("--language", default="en", help="Source language code")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session identifier (auto-generated if not provided)",
    )

    # Feature flags
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD")
    parser.add_argument(
        "--no-diarization", action="store_true", help="Disable speaker diarization"
    )
    parser.add_argument("--no-cif", action="store_true", help="Disable CIF")
    parser.add_argument(
        "--code-switching",
        action="store_true",
        help="Enable code-switching (multi-language detection)",
    )

    # Logging
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    config = {
        "model": args.model,
        "enable_vad": not args.no_vad,
        "enable_diarization": not args.no_diarization,
        "enable_cif": not args.no_cif,
        "enable_rolling_context": True,
        "enable_code_switching": args.code_switching,
    }

    # Code-switching requires language='auto' for proper detection
    if args.code_switching:
        config["language"] = "auto"
    else:
        config["language"] = args.language

    # Create tester
    tester = AudioStreamingTester(
        orchestration_url=args.url,
        chunk_duration_ms=args.chunk_duration,
        session_id=args.session_id,
        config=config,
    )

    try:
        # Connect to orchestration service
        await tester.connect()

        # Authenticate
        await tester.authenticate()

        # Start session
        await tester.start_session()

        # Start message receiver task
        receiver_task = asyncio.create_task(tester.receive_messages())

        # Stream audio based on input source
        if args.mic:
            await stream_from_microphone(tester)
        elif args.file:
            await stream_from_file(tester, args.file)
        elif args.stdin:
            await stream_from_stdin(tester)

        # End session
        await tester.end_session()

        # Wait for pending transcription segments to arrive
        # Keep waiting as long as segments are still arriving
        logger.info("⏳ Waiting for pending segments...")
        max_wait_time = 30  # Maximum 30 seconds total
        segment_timeout = 3  # 3 seconds without new segments = done
        wait_start = time.time()

        while True:
            elapsed = time.time() - wait_start
            if elapsed > max_wait_time:
                logger.info(f"⏱️ Reached max wait time ({max_wait_time}s)")
                break

            # Check if we received segments recently
            if tester.last_segment_time is not None:
                time_since_last_segment = time.time() - tester.last_segment_time
                if time_since_last_segment > segment_timeout:
                    logger.info(
                        f"✅ No segments for {segment_timeout}s, processing complete"
                    )
                    break

            await asyncio.sleep(0.5)  # Check every 500ms

        logger.info(f"📊 Received {tester.segments_received} total segments")

        # Print statistics
        tester.print_statistics()

        # Cancel receiver task
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupted by user")

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)

    finally:
        # Disconnect
        await tester.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
