#!/usr/bin/env python3
"""
WebSocket Client for Orchestration → Whisper Communication

This client manages the WebSocket connection from the orchestration service
to the Whisper service's WebSocket server.

Architecture:
    Frontend ↔ Orchestration (this client) ↔ Whisper WebSocket Server

Features:
- Auto-reconnection with exponential backoff
- Session state tracking
- Audio chunk streaming
- Segment reception and callback handling
- Connection health monitoring
"""

import asyncio
import json
import base64
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


@dataclass
class WhisperSessionState:
    """Tracks state of a Whisper streaming session"""

    session_id: str
    config: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    chunks_sent: int = 0
    segments_received: int = 0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)


class WebSocketWhisperClient:
    """
    WebSocket client for orchestration service to connect to Whisper

    Usage:
        client = WebSocketWhisperClient(whisper_host="localhost", whisper_port=5001)
        await client.connect()

        session_id = await client.start_stream(
            session_id="my-session",
            config={"model": "large-v3", "language": "en"}
        )

        await client.send_audio_chunk(session_id, audio_data, timestamp)

        # Register callback for segments
        client.on_segment(lambda segment: print(segment))

        await client.close()
    """

    def __init__(
        self,
        whisper_host: str = "localhost",
        whisper_port: int = 5001,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 1.0,
    ):
        """
        Initialize Whisper WebSocket client

        Args:
            whisper_host: Whisper service hostname
            whisper_port: Whisper service WebSocket port
            auto_reconnect: Enable automatic reconnection
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Initial delay between reconnection attempts (exponential backoff)
        """
        self.whisper_host = whisper_host
        self.whisper_port = whisper_port
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.websocket: Optional[ClientConnection] = None
        self.connected = False
        self.reconnect_count = 0

        # Session management
        self.sessions: Dict[str, WhisperSessionState] = {}

        # Callbacks
        self.segment_callbacks: Set[Callable[[Dict[str, Any]], None]] = set()
        self.error_callbacks: Set[Callable[[str], None]] = set()
        self.connection_callbacks: Set[Callable[[bool], None]] = set()

        # Background tasks
        self.receiver_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None

    @property
    def whisper_url(self) -> str:
        """Get Whisper WebSocket URL"""
        return f"ws://{self.whisper_host}:{self.whisper_port}/stream"

    async def connect(self) -> bool:
        """
        Connect to Whisper WebSocket server

        Returns:
            bool: True if connected successfully
        """
        try:
            logger.info(f"🔌 Connecting to Whisper WebSocket at {self.whisper_url}")

            self.websocket = await websockets.connect(
                self.whisper_url, ping_interval=30, ping_timeout=10
            )

            self.connected = True
            self.reconnect_count = 0

            # Start background tasks
            self.receiver_task = asyncio.create_task(self._receive_messages())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

            logger.info(f"✅ Connected to Whisper WebSocket")
            self._notify_connection(True)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to Whisper: {e}")
            self.connected = False
            self._notify_connection(False)

            if (
                self.auto_reconnect
                and self.reconnect_count < self.max_reconnect_attempts
            ):
                await self._attempt_reconnect()

            return False

    async def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        self.reconnect_count += 1
        delay = self.reconnect_delay * (2 ** (self.reconnect_count - 1))

        logger.warning(
            f"🔄 Attempting reconnection {self.reconnect_count}/{self.max_reconnect_attempts} "
            f"in {delay:.1f}s..."
        )

        await asyncio.sleep(delay)
        await self.connect()

    async def _receive_messages(self):
        """Background task to receive messages from Whisper"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message from Whisper: {e}")
                    self._notify_error(f"Invalid JSON from Whisper: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection to Whisper closed")
            self.connected = False
            self._notify_connection(False)

            if (
                self.auto_reconnect
                and self.reconnect_count < self.max_reconnect_attempts
            ):
                await self._attempt_reconnect()

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming message from Whisper"""
        msg_type = data.get("type")
        session_id = data.get("session_id")

        # Update session activity
        if session_id and session_id in self.sessions:
            self.sessions[session_id].update_activity()

        if msg_type == "session_started":
            logger.info(f"📝 Session started: {session_id}")

        elif msg_type == "segment":
            logger.debug(f"📄 Received segment for session {session_id}")

            # Update session stats
            if session_id in self.sessions:
                self.sessions[session_id].segments_received += 1

            # Notify callbacks
            self._notify_segment(data)

        elif msg_type == "error":
            error_msg = data.get("error", "Unknown error")
            logger.error(f"❌ Error from Whisper: {error_msg}")
            self._notify_error(error_msg)

        else:
            logger.debug(f"Received message type: {msg_type}")

    async def _heartbeat_monitor(self):
        """Monitor connection health with periodic heartbeats"""
        while self.connected:
            try:
                await asyncio.sleep(30)

                if self.connected and self.websocket:
                    # WebSocket library handles ping/pong automatically
                    logger.debug("💓 Connection heartbeat OK")

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    async def start_stream(self, session_id: str, config: Dict[str, Any]) -> str:
        """
        Start a new streaming session on Whisper

        Args:
            session_id: Unique session identifier
            config: Whisper configuration (model, language, etc.)

        Returns:
            str: Session ID

        Raises:
            RuntimeError: If not connected to Whisper
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to Whisper WebSocket server")

        # Create session state
        session_state = WhisperSessionState(session_id=session_id, config=config)
        self.sessions[session_id] = session_state

        # Send start_stream message
        message = {"action": "start_stream", "session_id": session_id, "config": config}

        await self.websocket.send(json.dumps(message))
        logger.info(f"🎬 Started stream for session: {session_id}")

        return session_id

    async def send_audio_chunk(
        self, session_id: str, audio_data: bytes, timestamp: Optional[datetime] = None
    ):
        """
        Send audio chunk to Whisper for processing

        Args:
            session_id: Session identifier
            audio_data: Raw audio data (bytes)
            timestamp: Chunk timestamp (defaults to now)

        Raises:
            RuntimeError: If not connected or session not found
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to Whisper WebSocket server")

        if session_id not in self.sessions:
            raise RuntimeError(f"Session not found: {session_id}")

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        # Use current timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Format timestamp as ISO 8601 with 'Z' suffix
        timestamp_str = timestamp.isoformat().replace("+00:00", "Z")

        # Send audio chunk message
        message = {
            "type": "audio_chunk",
            "session_id": session_id,
            "audio": audio_base64,
            "timestamp": timestamp_str,
        }

        await self.websocket.send(json.dumps(message))

        # Update session stats
        session = self.sessions[session_id]
        session.chunks_sent += 1
        session.update_activity()

        logger.debug(
            f"🎵 Sent audio chunk for session {session_id} "
            f"(chunk #{session.chunks_sent})"
        )

    async def close_stream(self, session_id: str):
        """
        Close a streaming session

        Args:
            session_id: Session identifier
        """
        if not self.connected or not self.websocket:
            logger.warning("Cannot close stream - not connected")
            return

        if session_id not in self.sessions:
            logger.warning(f"Session not found: {session_id}")
            return

        # Send close message
        message = {"action": "close_stream", "session_id": session_id}

        await self.websocket.send(json.dumps(message))

        # Mark session as inactive
        self.sessions[session_id].is_active = False

        logger.info(f"⏹️ Closed stream for session: {session_id}")

    async def close(self):
        """Close WebSocket connection and cleanup"""
        logger.info("Closing Whisper WebSocket client...")

        # Cancel background tasks
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass

        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all active sessions
        for session_id in list(self.sessions.keys()):
            if self.sessions[session_id].is_active:
                await self.close_stream(session_id)

        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()

        self.connected = False
        self.websocket = None

        logger.info("✅ Whisper WebSocket client closed")

    # Callback registration methods

    def on_segment(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register callback for segment events

        Args:
            callback: Function to call when segment is received
        """
        self.segment_callbacks.add(callback)

    def on_error(self, callback: Callable[[str], None]):
        """
        Register callback for error events

        Args:
            callback: Function to call when error occurs
        """
        self.error_callbacks.add(callback)

    def on_connection_change(self, callback: Callable[[bool], None]):
        """
        Register callback for connection state changes

        Args:
            callback: Function to call when connection state changes (True=connected, False=disconnected)
        """
        self.connection_callbacks.add(callback)

    def _notify_segment(self, segment: Dict[str, Any]):
        """Notify all segment callbacks"""
        for callback in self.segment_callbacks:
            try:
                callback(segment)
            except Exception as e:
                logger.error(f"Error in segment callback: {e}")

    def _notify_error(self, error: str):
        """Notify all error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def _notify_connection(self, connected: bool):
        """Notify all connection callbacks"""
        for callback in self.connection_callbacks:
            try:
                callback(connected)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")

    # Status methods

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information

        Args:
            session_id: Session identifier

        Returns:
            Dict with session info or None if not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "config": session.config,
            "created_at": session.created_at.isoformat(),
            "is_active": session.is_active,
            "chunks_sent": session.chunks_sent,
            "segments_received": session.segments_received,
            "last_activity": session.last_activity.isoformat(),
        }

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all session information

        Returns:
            Dict mapping session_id to session info
        """
        return {
            session_id: self.get_session_info(session_id)
            for session_id in self.sessions
        }

    def is_connected(self) -> bool:
        """Check if connected to Whisper"""
        return self.connected

    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics

        Returns:
            Dict with connection stats
        """
        return {
            "connected": self.connected,
            "whisper_url": self.whisper_url,
            "reconnect_count": self.reconnect_count,
            "active_sessions": sum(1 for s in self.sessions.values() if s.is_active),
            "total_sessions": len(self.sessions),
            "total_chunks_sent": sum(s.chunks_sent for s in self.sessions.values()),
            "total_segments_received": sum(
                s.segments_received for s in self.sessions.values()
            ),
        }


# Example usage
async def example_usage():
    """Example of using the WebSocket Whisper client"""

    # Create client
    client = WebSocketWhisperClient(
        whisper_host="localhost", whisper_port=5001, auto_reconnect=True
    )

    # Register callbacks
    def on_segment_received(segment):
        print(f"📄 Segment: {segment.get('text', '')}")

    def on_error_occurred(error):
        print(f"❌ Error: {error}")

    def on_connection_changed(connected):
        print(f"🔌 Connection: {'connected' if connected else 'disconnected'}")

    client.on_segment(on_segment_received)
    client.on_error(on_error_occurred)
    client.on_connection_change(on_connection_changed)

    # Connect
    await client.connect()

    # Start streaming session
    session_id = await client.start_stream(
        session_id="example-session",
        config={"model": "large-v3", "language": "en", "enable_vad": True},
    )

    # Send audio chunks
    import numpy as np

    for i in range(10):
        # Generate test audio (1 second)
        test_audio = np.random.randn(16000).astype(np.float32)
        await client.send_audio_chunk(session_id, test_audio.tobytes())
        await asyncio.sleep(1)

    # Get session info
    info = client.get_session_info(session_id)
    print(f"Session info: {info}")

    # Close stream
    await client.close_stream(session_id)

    # Close client
    await client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
