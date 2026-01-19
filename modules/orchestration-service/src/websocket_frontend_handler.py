#!/usr/bin/env python3
"""
WebSocket Handler for Frontend ↔ Orchestration Communication

This handler manages WebSocket connections from the frontend (browser)
to the orchestration service.

Architecture:
    Frontend (Browser) ↔ Orchestration (this handler) ↔ Whisper WebSocket Server

Features:
- Connection management with authentication
- Session tracking per frontend connection
- Audio chunk reception from frontend
- Segment/transcript streaming to frontend
- Translation result streaming
- Connection health monitoring
"""

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

logger = logging.getLogger(__name__)


@dataclass
class FrontendConnection:
    """Tracks state of a frontend WebSocket connection"""

    connection_id: str
    websocket: ServerConnection
    user_id: str | None = None
    session_id: str | None = None
    connected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_authenticated: bool = False
    chunks_received: int = 0
    messages_sent: int = 0

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(UTC)


class WebSocketFrontendHandler:
    """
    WebSocket handler for frontend connections

    Usage:
        handler = WebSocketFrontendHandler()

        # Register callbacks
        handler.on_audio_chunk(lambda conn_id, audio_data: ...)
        handler.on_session_start(lambda conn_id, config: ...)

        # Handle connection
        await handler.handle_connection(websocket)

        # Send messages to frontend
        await handler.send_segment(connection_id, segment_data)
        await handler.send_translation(connection_id, translation_data)
    """

    def __init__(self, require_authentication: bool = False, heartbeat_interval: float = 30.0):
        """
        Initialize frontend WebSocket handler

        Args:
            require_authentication: Whether to require authentication
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.require_authentication = require_authentication
        self.heartbeat_interval = heartbeat_interval

        # Connection tracking
        self.connections: dict[str, FrontendConnection] = {}
        self.session_to_connections: dict[str, set[str]] = {}  # session_id -> connection_ids

        # Callbacks
        self.audio_chunk_callbacks: set[Callable[[str, bytes, datetime], None]] = set()
        self.session_start_callbacks: set[Callable[[str, str, dict[str, Any]], None]] = set()
        self.session_end_callbacks: set[Callable[[str, str], None]] = set()
        self.connection_callbacks: set[Callable[[str, bool], None]] = set()

    async def handle_connection(self, websocket: ServerConnection):
        """
        Handle a new frontend WebSocket connection

        Args:
            websocket: WebSocket connection from frontend
        """
        # Generate connection ID
        connection_id = f"frontend-{websocket.id}"

        logger.info(f"🔌 New frontend connection: {connection_id}")

        # Create connection state
        connection = FrontendConnection(connection_id=connection_id, websocket=websocket)
        self.connections[connection_id] = connection

        # Notify callbacks
        self._notify_connection(connection_id, True)

        try:
            # Start heartbeat monitor
            heartbeat_task = asyncio.create_task(self._heartbeat_monitor(connection_id))

            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(connection_id, data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from {connection_id}: {e}")
                    await self._send_error(connection_id, f"Invalid JSON: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Frontend connection closed: {connection_id}")

        finally:
            # Cleanup
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

            await self._cleanup_connection(connection_id)

    async def _handle_message(self, connection_id: str, data: dict[str, Any]):
        """Handle incoming message from frontend"""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        connection.update_activity()

        msg_type = data.get("type") or data.get("action")

        if msg_type == "authenticate":
            await self._handle_authenticate(connection_id, data)

        elif msg_type == "start_session":
            await self._handle_start_session(connection_id, data)

        elif msg_type == "audio_chunk":
            await self._handle_audio_chunk(connection_id, data)

        elif msg_type == "end_session":
            await self._handle_end_session(connection_id, data)

        elif msg_type == "ping":
            await self._send_message(connection_id, {"type": "pong"})

        else:
            logger.warning(f"Unknown message type from {connection_id}: {msg_type}")

    async def _handle_authenticate(self, connection_id: str, data: dict[str, Any]):
        """Handle authentication message"""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Extract auth info
        user_id = data.get("user_id")
        data.get("token")

        # TODO: Implement actual authentication
        # For now, accept all connections
        connection.is_authenticated = True
        connection.user_id = user_id

        logger.info(f"✅ Frontend {connection_id} authenticated (user: {user_id})")

        # Send acknowledgement
        await self._send_message(
            connection_id,
            {
                "type": "authenticated",
                "connection_id": connection_id,
                "user_id": user_id,
            },
        )

    async def _handle_start_session(self, connection_id: str, data: dict[str, Any]):
        """Handle session start message"""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Check authentication
        if self.require_authentication and not connection.is_authenticated:
            await self._send_error(connection_id, "Authentication required")
            return

        session_id = data.get("session_id")
        config = data.get("config", {})

        # Update connection state
        connection.session_id = session_id

        # Track session -> connection mapping
        if session_id not in self.session_to_connections:
            self.session_to_connections[session_id] = set()
        self.session_to_connections[session_id].add(connection_id)

        logger.info(f"🎬 Session started: {session_id} (connection: {connection_id})")

        # Notify callbacks
        self._notify_session_start(connection_id, session_id, config)

        # Send acknowledgement
        await self._send_message(
            connection_id,
            {
                "type": "session_started",
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
        )

    async def _handle_audio_chunk(self, connection_id: str, data: dict[str, Any]):
        """Handle audio chunk from frontend"""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Extract audio data
        audio_base64 = data.get("audio")
        timestamp_str = data.get("timestamp")

        if not audio_base64:
            await self._send_error(connection_id, "Missing audio data")
            return

        # Decode audio
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            await self._send_error(connection_id, f"Invalid audio data: {e}")
            return

        # Parse timestamp
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except Exception:
                timestamp = datetime.now(UTC)
        else:
            timestamp = datetime.now(UTC)

        # Update stats
        connection.chunks_received += 1

        logger.debug(f"🎵 Audio chunk from {connection_id} (chunk #{connection.chunks_received})")

        # Notify callbacks
        self._notify_audio_chunk(connection_id, audio_data, timestamp)

    async def _handle_end_session(self, connection_id: str, data: dict[str, Any]):
        """Handle session end message"""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        session_id = data.get("session_id") or connection.session_id

        logger.info(f"⏹️ Session ended: {session_id} (connection: {connection_id})")

        # Notify callbacks
        if session_id:
            self._notify_session_end(connection_id, session_id)

        # Cleanup session mapping
        if session_id and session_id in self.session_to_connections:
            self.session_to_connections[session_id].discard(connection_id)
            if not self.session_to_connections[session_id]:
                del self.session_to_connections[session_id]

        # Clear session from connection
        connection.session_id = None

        # Send acknowledgement
        await self._send_message(connection_id, {"type": "session_ended", "session_id": session_id})

    async def _heartbeat_monitor(self, connection_id: str):
        """Monitor connection health with periodic heartbeats"""
        while connection_id in self.connections:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                connection = self.connections.get(connection_id)
                if not connection:
                    break

                # Send ping
                await self._send_message(connection_id, {"type": "ping"})
                logger.debug(f"💓 Heartbeat sent to {connection_id}")

            except Exception as e:
                logger.error(f"Heartbeat error for {connection_id}: {e}")
                break

    async def _cleanup_connection(self, connection_id: str):
        """Cleanup connection state"""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Notify session end if active
        if connection.session_id:
            self._notify_session_end(connection_id, connection.session_id)

            # Cleanup session mapping
            if connection.session_id in self.session_to_connections:
                self.session_to_connections[connection.session_id].discard(connection_id)
                if not self.session_to_connections[connection.session_id]:
                    del self.session_to_connections[connection.session_id]

        # Remove connection
        del self.connections[connection_id]

        # Notify callbacks
        self._notify_connection(connection_id, False)

        logger.info(f"🔌 Frontend connection cleaned up: {connection_id}")

    # Sending methods

    async def _send_message(self, connection_id: str, data: dict[str, Any]):
        """Send message to frontend connection"""
        connection = self.connections.get(connection_id)
        if not connection:
            logger.warning(f"Cannot send message - connection not found: {connection_id}")
            return

        try:
            message = json.dumps(data)
            await connection.websocket.send(message)
            connection.messages_sent += 1
            connection.update_activity()

        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")

    async def _send_error(self, connection_id: str, error: str):
        """Send error message to frontend"""
        await self._send_message(
            connection_id,
            {
                "type": "error",
                "error": error,
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
        )

    async def send_segment(self, connection_id: str, segment: dict[str, Any]):
        """
        Send transcription segment to frontend

        Args:
            connection_id: Frontend connection ID
            segment: Segment data
        """
        await self._send_message(connection_id, {"type": "segment", **segment})

    async def send_translation(self, connection_id: str, translation: dict[str, Any]):
        """
        Send translation to frontend

        Args:
            connection_id: Frontend connection ID
            translation: Translation data
        """
        await self._send_message(connection_id, {"type": "translation", **translation})

    async def broadcast_to_session(self, session_id: str, data: dict[str, Any]):
        """
        Broadcast message to all connections in a session

        Args:
            session_id: Session ID
            data: Message data
        """
        connection_ids = self.session_to_connections.get(session_id, set())

        for connection_id in list(connection_ids):
            await self._send_message(connection_id, data)

    # Callback registration

    def on_audio_chunk(self, callback: Callable[[str, bytes, datetime], None]):
        """
        Register callback for audio chunks from frontend

        Args:
            callback: Function(connection_id, audio_data, timestamp)
        """
        self.audio_chunk_callbacks.add(callback)

    def on_session_start(self, callback: Callable[[str, str, dict[str, Any]], None]):
        """
        Register callback for session start

        Args:
            callback: Function(connection_id, session_id, config)
        """
        self.session_start_callbacks.add(callback)

    def on_session_end(self, callback: Callable[[str, str], None]):
        """
        Register callback for session end

        Args:
            callback: Function(connection_id, session_id)
        """
        self.session_end_callbacks.add(callback)

    def on_connection_change(self, callback: Callable[[str, bool], None]):
        """
        Register callback for connection state changes

        Args:
            callback: Function(connection_id, connected: bool)
        """
        self.connection_callbacks.add(callback)

    def _notify_audio_chunk(self, connection_id: str, audio_data: bytes, timestamp: datetime):
        """Notify all audio chunk callbacks"""
        for callback in self.audio_chunk_callbacks:
            try:
                callback(connection_id, audio_data, timestamp)
            except Exception as e:
                logger.error(f"Error in audio chunk callback: {e}")

    def _notify_session_start(self, connection_id: str, session_id: str, config: dict[str, Any]):
        """Notify all session start callbacks"""
        for callback in self.session_start_callbacks:
            try:
                callback(connection_id, session_id, config)
            except Exception as e:
                logger.error(f"Error in session start callback: {e}")

    def _notify_session_end(self, connection_id: str, session_id: str):
        """Notify all session end callbacks"""
        for callback in self.session_end_callbacks:
            try:
                callback(connection_id, session_id)
            except Exception as e:
                logger.error(f"Error in session end callback: {e}")

    def _notify_connection(self, connection_id: str, connected: bool):
        """Notify all connection callbacks"""
        for callback in self.connection_callbacks:
            try:
                callback(connection_id, connected)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")

    # Status methods

    def get_connection_info(self, connection_id: str) -> dict[str, Any] | None:
        """
        Get connection information

        Args:
            connection_id: Connection ID

        Returns:
            Dict with connection info or None
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return None

        return {
            "connection_id": connection.connection_id,
            "user_id": connection.user_id,
            "session_id": connection.session_id,
            "connected_at": connection.connected_at.isoformat(),
            "last_activity": connection.last_activity.isoformat(),
            "is_authenticated": connection.is_authenticated,
            "chunks_received": connection.chunks_received,
            "messages_sent": connection.messages_sent,
        }

    def get_all_connections(self) -> dict[str, dict[str, Any]]:
        """Get all connection information"""
        return {conn_id: self.get_connection_info(conn_id) for conn_id in self.connections}

    def get_session_connections(self, session_id: str) -> set[str]:
        """Get all connection IDs for a session"""
        return self.session_to_connections.get(session_id, set()).copy()

    def get_stats(self) -> dict[str, Any]:
        """Get handler statistics"""
        return {
            "total_connections": len(self.connections),
            "authenticated_connections": sum(
                1 for c in self.connections.values() if c.is_authenticated
            ),
            "active_sessions": len(self.session_to_connections),
            "total_chunks_received": sum(c.chunks_received for c in self.connections.values()),
            "total_messages_sent": sum(c.messages_sent for c in self.connections.values()),
        }


# Example usage
async def example_usage():
    """Example of using the frontend WebSocket handler"""

    handler = WebSocketFrontendHandler(require_authentication=False)

    # Register callbacks
    def on_audio(conn_id, audio_data, timestamp):
        print(f"🎵 Audio from {conn_id}: {len(audio_data)} bytes")

    def on_session_start(conn_id, session_id, config):
        print(f"🎬 Session started: {session_id} on {conn_id}")

    def on_session_end(conn_id, session_id):
        print(f"⏹️ Session ended: {session_id} on {conn_id}")

    handler.on_audio_chunk(on_audio)
    handler.on_session_start(on_session_start)
    handler.on_session_end(on_session_end)

    # Start WebSocket server
    async with websockets.serve(handler.handle_connection, "localhost", 8000):
        print("Frontend WebSocket server running on ws://localhost:8000")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
