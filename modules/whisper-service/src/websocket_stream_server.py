#!/usr/bin/env python3
"""
WebSocket Stream Server for Real-Time Whisper Transcription

Implements WebSocket server for streaming audio transcription.
Integrates with StreamSessionManager for session lifecycle management.

Following Phase 3.1 architecture:
- WebSocket endpoint at /stream
- Message types: start_stream, audio_chunk, close_stream
- Real-time segment streaming with ISO 8601 timestamps
- Integration with model manager for transcription

Usage:
    server = WebSocketStreamServer(host="localhost", port=5001)
    await server.start()

    # Clients connect to ws://localhost:5001/stream
    # Send: {"action": "start_stream", "session_id": "...", "config": {...}}
    # Send: {"type": "audio_chunk", "session_id": "...", "audio": "<base64>", "timestamp": "..."}
    # Receive: {"type": "segment", "session_id": "...", "text": "...", ...}
"""

import asyncio
import json
import base64
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Set, Optional
import websockets
from websockets.asyncio.server import ServerConnection

from stream_session_manager import StreamSessionManager
from segment_timestamper import SegmentTimestamper

logger = logging.getLogger(__name__)


class WebSocketStreamServer:
    """
    WebSocket server for real-time audio streaming

    Manages WebSocket connections and routes messages to session manager
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5001,
        model_manager=None
    ):
        """
        Initialize WebSocket stream server

        Parameters:
            host (str): Host to bind to
            port (int): Port to listen on
            model_manager: ModelManager instance for transcription
        """
        self.host = host
        self.port = port
        self.model_manager = model_manager

        # Session management
        self.session_manager = StreamSessionManager(model_manager=model_manager)
        self.timestamper = SegmentTimestamper()

        # WebSocket connection tracking
        self.connections: Set[ServerConnection] = set()
        self.session_to_ws: Dict[str, ServerConnection] = {}

        # Server instance
        self.server = None
        self.running = False

        logger.info(f"WebSocket stream server initialized on {host}:{port}")

    async def start(self):
        """
        Start the WebSocket server

        Starts listening for connections at ws://{host}:{port}/stream
        """
        self.running = True

        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        ) as self.server:
            logger.info(f"✅ WebSocket server started on ws://{self.host}:{self.port}/stream")

            # Keep server running
            await asyncio.Future()  # Run forever

    async def stop(self):
        """
        Stop the WebSocket server

        Closes all connections and cleans up sessions
        """
        self.running = False

        # Close all connections
        for ws in list(self.connections):
            await ws.close()

        # Close all sessions
        for session_id in list(self.session_manager.sessions.keys()):
            await self.session_manager.close_session(session_id)

        logger.info("WebSocket server stopped")

    async def handle_connection(self, websocket: ServerConnection):
        """
        Handle a new WebSocket connection

        Parameters:
            websocket (ServerConnection): WebSocket connection
        """
        self.connections.add(websocket)
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"

        logger.info(f"New WebSocket connection from {client_id}")

        try:
            async for message in websocket:
                await self.handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {client_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection {client_id}: {e}")
        finally:
            self.connections.discard(websocket)

            # Clean up sessions for this connection
            sessions_to_close = [
                sid for sid, ws in self.session_to_ws.items()
                if ws == websocket
            ]
            for session_id in sessions_to_close:
                await self.session_manager.close_session(session_id)
                del self.session_to_ws[session_id]

    async def handle_message(self, websocket: ServerConnection, message: str):
        """
        Handle incoming WebSocket message

        Parameters:
            websocket (ServerConnection): WebSocket connection
            message (str): JSON message string
        """
        try:
            data = json.loads(message)

            # Route message based on action/type
            action = data.get("action") or data.get("type")

            if action == "start_stream":
                await self.handle_start_stream(websocket, data)
            elif action == "audio_chunk":
                await self.handle_audio_chunk(websocket, data)
            elif action == "close_stream":
                await self.handle_close_stream(websocket, data)
            else:
                # Unknown action
                await self.send_error(websocket, f"Unknown action: {action}")

        except json.JSONDecodeError as e:
            await self.send_error(websocket, f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_error(websocket, f"Server error: {e}")

    async def handle_start_stream(self, websocket: ServerConnection, data: Dict[str, Any]):
        """
        Handle start_stream action

        Creates a new streaming session

        Message format:
        {
            "action": "start_stream",
            "session_id": "session-123",
            "config": {
                "model": "large-v3",
                "language": "en",
                "enable_vad": True,
                "enable_cif": True
            }
        }

        Response:
        {
            "type": "session_started",
            "session_id": "session-123",
            "timestamp": "2025-01-15T10:30:00.000Z"
        }
        """
        session_id = data.get("session_id")
        config = data.get("config", {})

        if not session_id:
            await self.send_error(websocket, "Missing session_id")
            return

        # Create session
        session = await self.session_manager.create_session(
            session_id=session_id,
            config=config
        )

        # Track connection
        self.session_to_ws[session_id] = websocket

        # Send acknowledgement
        response = {
            "type": "session_started",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }

        await websocket.send(json.dumps(response))

        logger.info(f"Started streaming session: {session_id}")

    async def handle_audio_chunk(self, websocket: ServerConnection, data: Dict[str, Any]):
        """
        Handle audio_chunk message

        Receives audio data and queues for processing

        Message format:
        {
            "type": "audio_chunk",
            "session_id": "session-123",
            "audio": "<base64-encoded-audio>",
            "timestamp": "2025-01-15T10:30:00.000Z"
        }
        """
        session_id = data.get("session_id")
        audio_base64 = data.get("audio")
        timestamp_str = data.get("timestamp")

        if not session_id:
            await self.send_error(websocket, "Missing session_id")
            return

        if not audio_base64:
            await self.send_error(websocket, "Missing audio data")
            return

        # Parse timestamp
        if timestamp_str:
            timestamp = self.timestamper.parse_timestamp(timestamp_str)
        else:
            timestamp = datetime.now(timezone.utc)

        # Decode audio
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            await self.send_error(websocket, f"Invalid base64 audio: {e}")
            return

        # Add to session buffer
        success = await self.session_manager.add_audio_chunk(
            session_id=session_id,
            audio_data=audio_bytes,
            timestamp=timestamp
        )

        if not success:
            await self.send_error(websocket, f"Session not found: {session_id}")
            return

        # Process session (if enough audio buffered)
        segments = await self.session_manager.process_session(session_id)

        # Stream segments back to client
        for segment in segments:
            # Add absolute timestamps
            timestamped_segment = self.timestamper.add_absolute_timestamps(
                segment=segment,
                chunk_start_time=timestamp,
                is_final=False
            )

            # Send segment
            response = {
                "type": "segment",
                "session_id": session_id,
                **timestamped_segment
            }

            await websocket.send(json.dumps(response))

    async def handle_close_stream(self, websocket: ServerConnection, data: Dict[str, Any]):
        """
        Handle close_stream action

        Closes a streaming session

        Message format:
        {
            "action": "close_stream",
            "session_id": "session-123"
        }

        Response:
        {
            "type": "session_closed",
            "session_id": "session-123",
            "timestamp": "2025-01-15T10:30:00.000Z"
        }
        """
        session_id = data.get("session_id")

        if not session_id:
            await self.send_error(websocket, "Missing session_id")
            return

        # Close session
        await self.session_manager.close_session(session_id)

        # Remove from tracking
        self.session_to_ws.pop(session_id, None)

        # Send acknowledgement
        response = {
            "type": "session_closed",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }

        await websocket.send(json.dumps(response))

        logger.info(f"Closed streaming session: {session_id}")

    async def send_error(self, websocket: ServerConnection, error_message: str):
        """
        Send error message to client

        Parameters:
            websocket (ServerConnection): WebSocket connection
            error_message (str): Error description
        """
        response = {
            "type": "error",
            "error": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }

        await websocket.send(json.dumps(response))
        logger.warning(f"Sent error to client: {error_message}")

    def get_session(self, session_id: str):
        """
        Get session by ID

        Parameters:
            session_id (str): Session identifier

        Returns:
            StreamingSession or None: Session if exists
        """
        return self.session_manager.get_session(session_id)

    def get_active_sessions(self):
        """
        Get list of active session IDs

        Returns:
            List[str]: Active session IDs
        """
        return self.session_manager.get_active_sessions()


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_server():
        print("WebSocket Stream Server Test")
        print("=" * 50)

        # Create server
        server = WebSocketStreamServer(host="localhost", port=5001)

        print(f"\n[TEST] Starting server on ws://localhost:5001/stream")

        # Start server (would run forever in production)
        # In test, we'd need to connect with a client
        # await server.start()

        print("\n✅ WebSocket Stream Server Ready")
        print("\nTo test:")
        print("  1. Connect: ws://localhost:5001/stream")
        print("  2. Send: {\"action\": \"start_stream\", \"session_id\": \"test\", \"config\": {}}")
        print("  3. Send: {\"type\": \"audio_chunk\", \"session_id\": \"test\", \"audio\": \"<base64>\", \"timestamp\": \"...\"}")
        print("  4. Receive: {\"type\": \"segment\", ...}")

    asyncio.run(test_server())
