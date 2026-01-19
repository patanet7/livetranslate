#!/usr/bin/env python3
"""
TCP Audio Streaming Server
Accepts raw PCM audio via TCP socket (like SimulStreaming)
Forwards to Whisper service via orchestration

Usage:
    # Start server
    python tcp_audio_server.py --port 43001

    # Stream from microphone
    ffmpeg -f avfoundation -i ":0" -ac 1 -ar 16000 -f s16le - | nc 127.0.0.1 43001

    # Stream from file
    ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - | nc 127.0.0.1 43001
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

# Add src to path
sys.path.insert(
    0,
    "/Users/thomaspatane/Documents/GitHub/livetranslate/modules/orchestration-service/src",
)
from socketio_whisper_client import SocketIOWhisperClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TCPAudioServer:
    def __init__(
        self,
        port: int = 43001,
        whisper_host: str = "localhost",
        whisper_port: int = 5001,
        chunk_size: int = 3200,  # 100ms at 16kHz mono
        model: str = "large-v3-turbo",
        language: str = "en",
    ):
        self.port = port
        self.chunk_size = chunk_size
        self.model = model
        self.language = language

        # Create Whisper client
        self.whisper_client = SocketIOWhisperClient(
            whisper_host=whisper_host, whisper_port=whisper_port, auto_reconnect=True
        )

        # Stats
        self.total_chunks = 0
        self.total_segments = 0
        self.start_time = None

    def on_segment(self, segment: dict):
        """Callback for Whisper segments"""
        self.total_segments += 1

        # Extract segment info
        text = segment.get("text", "")
        speaker = segment.get("speaker", "UNKNOWN")
        language = segment.get("language", "unknown").upper()
        confidence = segment.get("confidence", 0.0) * 100
        is_final = segment.get("is_final", False)

        # Print segment
        status = "✅ FINAL" if is_final else "⏳ PARTIAL"
        print("\n" + "=" * 80)
        print(f"{status} | 👤 {speaker} | 🌐 {language} | 📊 {confidence:.1f}%")
        print(f"📝 {text}")
        print("=" * 80)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming TCP connection"""
        addr = writer.get_extra_info("peername")
        session_id = f"tcp-{int(datetime.now(UTC).timestamp())}"

        logger.info(f"✅ Client connected from {addr}")
        logger.info(f"📝 Session ID: {session_id}")

        try:
            # Connect to Whisper service
            if not self.whisper_client.connected:
                logger.info("🔌 Connecting to Whisper service...")
                await self.whisper_client.connect()

            # Register segment callback
            self.whisper_client.on_segment(self.on_segment)

            # Start Whisper session
            logger.info(f"🎬 Starting Whisper session with model: {self.model}")
            await self.whisper_client.start_stream(
                session_id=session_id,
                config={
                    "model": self.model,
                    "language": self.language,
                    "enable_vad": True,
                    "enable_diarization": True,
                    "enable_cif": True,
                    "enable_rolling_context": True,
                },
            )

            logger.info("🎵 Streaming audio... (Ctrl+C to stop)")
            self.start_time = datetime.now(UTC)

            # Read audio chunks from TCP socket
            while True:
                chunk = await reader.read(self.chunk_size)

                if not chunk:
                    logger.info("📭 Client disconnected")
                    break

                self.total_chunks += 1

                # Forward to Whisper
                await self.whisper_client.send_audio_chunk(
                    session_id=session_id,
                    audio_data=chunk,
                    timestamp=datetime.now(UTC).isoformat(),
                )

                # Log progress every 100 chunks (10 seconds)
                if self.total_chunks % 100 == 0:
                    elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
                    logger.info(
                        f"📊 Sent {self.total_chunks} chunks ({elapsed:.1f}s, {self.total_segments} segments)"
                    )

        except Exception as e:
            logger.error(f"❌ Error handling client: {e}", exc_info=True)

        finally:
            # End Whisper session
            try:
                await self.whisper_client.close_stream(session_id)
            except Exception as e:
                logger.error(f"Error ending session: {e}")

            # Print stats
            if self.start_time:
                elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
                print("\n" + "=" * 80)
                print("📊 STREAMING STATISTICS")
                print("=" * 80)
                print(f"Duration:          {elapsed:.1f}s")
                print(f"Chunks sent:       {self.total_chunks}")
                print(f"Segments received: {self.total_segments}")
                if elapsed > 0:
                    print(f"Chunk rate:        {self.total_chunks / elapsed:.1f} chunks/sec")
                print("=" * 80 + "\n")

            writer.close()
            await writer.wait_closed()
            logger.info("🧹 Connection closed")

    async def start(self):
        """Start TCP server"""
        server = await asyncio.start_server(self.handle_client, "0.0.0.0", self.port)

        addr = server.sockets[0].getsockname()
        logger.info(f"🚀 TCP Audio Server listening on {addr[0]}:{addr[1]}")
        logger.info(f"📡 Model: {self.model} | Language: {self.language}")
        logger.info(
            f'💡 Test with: ffmpeg -f avfoundation -i ":0" -ac 1 -ar 16000 -f s16le - | nc 127.0.0.1 {self.port}'
        )

        async with server:
            await server.serve_forever()


async def main():
    parser = argparse.ArgumentParser(description="TCP Audio Streaming Server")
    parser.add_argument(
        "--port", type=int, default=43001, help="TCP port to listen on (default: 43001)"
    )
    parser.add_argument(
        "--whisper-host", type=str, default="localhost", help="Whisper service host"
    )
    parser.add_argument("--whisper-port", type=int, default=5001, help="Whisper service port")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3200,
        help="Chunk size in bytes (default: 3200 = 100ms)",
    )
    parser.add_argument("--model", type=str, default="large-v3-turbo", help="Whisper model")
    parser.add_argument("--language", type=str, default="en", help="Source language")
    args = parser.parse_args()

    server = TCPAudioServer(
        port=args.port,
        whisper_host=args.whisper_host,
        whisper_port=args.whisper_port,
        chunk_size=args.chunk_size,
        model=args.model,
        language=args.language,
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
