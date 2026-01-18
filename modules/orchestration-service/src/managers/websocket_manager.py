"""
WebSocket Manager

Enterprise-grade WebSocket connection management for the orchestration service.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states"""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(Enum):
    """WebSocket message types"""

    PING = "ping"
    PONG = "pong"
    JOIN_SESSION = "join_session"
    LEAVE_SESSION = "leave_session"
    BROADCAST = "broadcast"
    DIRECT_MESSAGE = "direct_message"
    SERVICE_MESSAGE = "service_message"
    SERVICE_RESPONSE = "service_response"
    ERROR = "error"
    SYSTEM_MESSAGE = "system_message"


@dataclass
class ConnectionInfo:
    """WebSocket connection information"""

    connection_id: str
    websocket: WebSocket
    client_ip: str
    user_agent: str
    connected_at: float
    last_ping: float
    session_id: str | None = None
    user_id: str | None = None
    state: ConnectionState = ConnectionState.CONNECTING
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "connection_id": self.connection_id,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "connected_at": self.connected_at,
            "last_ping": self.last_ping,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "uptime": time.time() - self.connected_at,
            "metadata": self.metadata,
        }


@dataclass
class SessionInfo:
    """Session information"""

    session_id: str
    created_at: float
    last_activity: float
    connection_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "connection_count": len(self.connection_ids),
            "connection_ids": list(self.connection_ids),
            "uptime": time.time() - self.created_at,
            "metadata": self.metadata,
        }


class WebSocketManager:
    """
    Enterprise-grade WebSocket connection manager
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.max_connections = self.config.get("max_connections", 1000)
        self.heartbeat_interval = self.config.get("heartbeat_interval", 30)
        self.session_timeout = self.config.get("session_timeout", 1800)
        self.ping_timeout = self.config.get("ping_timeout", 10)

        # Connection storage
        self.connections: dict[str, ConnectionInfo] = {}
        self.sessions: dict[str, SessionInfo] = {}
        self.user_connections: dict[str, set[str]] = {}

        # Event handlers
        self.message_handlers: dict[MessageType, list[Callable]] = {
            msg_type: [] for msg_type in MessageType
        }

        # Background tasks
        self._heartbeat_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        # Statistics
        self.stats = {
            "total_connections": 0,
            "total_messages": 0,
            "total_sessions": 0,
            "total_errors": 0,
        }

    async def start(self):
        """Start the WebSocket manager"""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("WebSocket manager started")

    async def stop(self):
        """Stop the WebSocket manager"""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Close all connections
        await self._close_all_connections()

        logger.info("WebSocket manager stopped")

    async def connect(self, websocket: WebSocket, client_ip: str, user_agent: str) -> str:
        """
        Handle new WebSocket connection

        Returns:
            Connection ID
        """
        # Check connection limit
        if len(self.connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Connection limit exceeded")
            raise ConnectionError("Maximum connections exceeded")

        # Accept connection
        await websocket.accept()

        # Create connection info
        connection_id = str(uuid4())
        connection_info = ConnectionInfo(
            connection_id=connection_id,
            websocket=websocket,
            client_ip=client_ip,
            user_agent=user_agent,
            connected_at=time.time(),
            last_ping=time.time(),
            state=ConnectionState.CONNECTED,
        )

        # Store connection
        self.connections[connection_id] = connection_info
        self.stats["total_connections"] += 1

        logger.info(f"New WebSocket connection: {connection_id} from {client_ip}")

        # Send welcome message
        await self._send_message(
            connection_id,
            {
                "type": MessageType.SYSTEM_MESSAGE.value,
                "message": "Connected to orchestration service",
                "connection_id": connection_id,
                "timestamp": time.time(),
            },
        )

        return connection_id

    async def disconnect(self, connection_id: str, reason: str = "Client disconnected"):
        """
        Handle WebSocket disconnection
        """
        if connection_id not in self.connections:
            return

        connection_info = self.connections[connection_id]
        connection_info.state = ConnectionState.DISCONNECTING

        # Remove from session if part of one
        if connection_info.session_id:
            await self._leave_session(connection_id, connection_info.session_id)

        # Remove from user connections
        if connection_info.user_id and connection_info.user_id in self.user_connections:
            self.user_connections[connection_info.user_id].discard(connection_id)
            if not self.user_connections[connection_info.user_id]:
                del self.user_connections[connection_info.user_id]

        # Close WebSocket if not already closed
        try:
            from starlette.websockets import WebSocketState

            if connection_info.websocket.client_state not in [WebSocketState.DISCONNECTED]:
                await connection_info.websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket {connection_id}: {e}")

        # Remove connection
        del self.connections[connection_id]

        logger.info(f"WebSocket disconnected: {connection_id} - {reason}")

    async def handle_message(self, connection_id: str, message: dict[str, Any]):
        """
        Handle incoming WebSocket message
        """
        if connection_id not in self.connections:
            logger.warning(f"Message from unknown connection: {connection_id}")
            return

        self.stats["total_messages"] += 1

        try:
            message_type = MessageType(message.get("type", ""))

            # Update last activity
            connection_info = self.connections[connection_id]
            connection_info.last_ping = time.time()

            # Handle message based on type
            if message_type == MessageType.PING:
                await self._handle_ping(connection_id, message)
            elif message_type == MessageType.JOIN_SESSION:
                await self._handle_join_session(connection_id, message)
            elif message_type == MessageType.LEAVE_SESSION:
                await self._handle_leave_session(connection_id, message)
            elif message_type == MessageType.BROADCAST:
                await self._handle_broadcast(connection_id, message)
            elif message_type == MessageType.DIRECT_MESSAGE:
                await self._handle_direct_message(connection_id, message)
            elif message_type == MessageType.SERVICE_MESSAGE:
                await self._handle_service_message(connection_id, message)
            else:
                # Call registered handlers
                handlers = self.message_handlers.get(message_type, [])
                for handler in handlers:
                    try:
                        await handler(connection_id, message)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")

        except ValueError:
            # Invalid message type
            await self._send_error(connection_id, f"Invalid message type: {message.get('type')}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await self._send_error(connection_id, "Internal error processing message")
            self.stats["total_errors"] += 1

    async def _handle_ping(self, connection_id: str, message: dict[str, Any]):
        """Handle ping message"""
        await self._send_message(
            connection_id, {"type": MessageType.PONG.value, "timestamp": time.time()}
        )

    async def _handle_join_session(self, connection_id: str, message: dict[str, Any]):
        """Handle join session message"""
        session_id = message.get("session_id")
        if not session_id:
            await self._send_error(connection_id, "Missing session_id")
            return

        await self._join_session(connection_id, session_id, message.get("metadata", {}))

    async def _handle_leave_session(self, connection_id: str, message: dict[str, Any]):
        """Handle leave session message"""
        session_id = message.get("session_id")
        if not session_id:
            await self._send_error(connection_id, "Missing session_id")
            return

        await self._leave_session(connection_id, session_id)

    async def _handle_broadcast(self, connection_id: str, message: dict[str, Any]):
        """Handle broadcast message"""
        session_id = message.get("session_id")
        if not session_id:
            await self._send_error(connection_id, "Missing session_id")
            return

        await self.broadcast_to_session(
            session_id, message.get("data", {}), exclude_connection=connection_id
        )

    async def _handle_direct_message(self, connection_id: str, message: dict[str, Any]):
        """Handle direct message"""
        target_connection = message.get("target_connection")
        if not target_connection:
            await self._send_error(connection_id, "Missing target_connection")
            return

        await self._send_message(
            target_connection,
            {
                "type": MessageType.DIRECT_MESSAGE.value,
                "from_connection": connection_id,
                "data": message.get("data", {}),
                "timestamp": time.time(),
            },
        )

    async def _handle_service_message(self, connection_id: str, message: dict[str, Any]):
        """Handle service message (placeholder for service integration)"""
        target_service = message.get("target_service")
        if not target_service:
            await self._send_error(connection_id, "Missing target_service")
            return

        # This would integrate with service clients
        await self._send_message(
            connection_id,
            {
                "type": MessageType.SERVICE_RESPONSE.value,
                "source_service": target_service,
                "data": {"status": "service_integration_placeholder"},
                "timestamp": time.time(),
            },
        )

    async def _join_session(
        self, connection_id: str, session_id: str, metadata: dict[str, Any] | None = None
    ):
        """Join a session"""
        if connection_id not in self.connections:
            return

        connection_info = self.connections[connection_id]

        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionInfo(
                session_id=session_id,
                created_at=time.time(),
                last_activity=time.time(),
                metadata=metadata or {},
            )
            self.stats["total_sessions"] += 1

        # Add connection to session
        session_info = self.sessions[session_id]
        session_info.connection_ids.add(connection_id)
        session_info.last_activity = time.time()

        # Update connection info
        connection_info.session_id = session_id
        if metadata:
            connection_info.metadata.update(metadata)

        logger.info(f"Connection {connection_id} joined session {session_id}")

        # Notify connection
        await self._send_message(
            connection_id,
            {
                "type": MessageType.SYSTEM_MESSAGE.value,
                "message": f"Joined session: {session_id}",
                "session_id": session_id,
                "timestamp": time.time(),
            },
        )

    async def _leave_session(self, connection_id: str, session_id: str):
        """Leave a session"""
        if session_id not in self.sessions:
            return

        session_info = self.sessions[session_id]
        session_info.connection_ids.discard(connection_id)

        # Update connection info
        if connection_id in self.connections:
            self.connections[connection_id].session_id = None

        # Remove empty session
        if not session_info.connection_ids:
            del self.sessions[session_id]

        logger.info(f"Connection {connection_id} left session {session_id}")

        # Notify connection
        await self._send_message(
            connection_id,
            {
                "type": MessageType.SYSTEM_MESSAGE.value,
                "message": f"Left session: {session_id}",
                "timestamp": time.time(),
            },
        )

    async def broadcast_to_session(
        self, session_id: str, data: dict[str, Any], exclude_connection: str | None = None
    ):
        """Broadcast message to all connections in a session"""
        if session_id not in self.sessions:
            return

        session_info = self.sessions[session_id]
        message = {
            "type": MessageType.BROADCAST.value,
            "session_id": session_id,
            "data": data,
            "timestamp": time.time(),
        }

        # Send to all connections in session
        for connection_id in session_info.connection_ids:
            if connection_id != exclude_connection:
                await self._send_message(connection_id, message)

    async def broadcast_to_all(self, data: dict[str, Any]):
        """Broadcast message to all connections"""
        message = {
            "type": MessageType.BROADCAST.value,
            "data": data,
            "timestamp": time.time(),
        }

        # Send to all connections
        for connection_id in self.connections:
            await self._send_message(connection_id, message)

    async def _send_message(self, connection_id: str, message: dict[str, Any]):
        """Send message to specific connection"""
        if connection_id not in self.connections:
            return

        connection_info = self.connections[connection_id]

        try:
            await connection_info.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            await self.disconnect(connection_id, "Send error")

    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        await self._send_message(
            connection_id,
            {
                "type": MessageType.ERROR.value,
                "error": error_message,
                "timestamp": time.time(),
            },
        )

    async def _heartbeat_loop(self):
        """Background task for heartbeat monitoring"""
        while self._running:
            try:
                current_time = time.time()
                dead_connections = []

                # Check for dead connections
                for connection_id, connection_info in self.connections.items():
                    if (
                        current_time - connection_info.last_ping
                        > self.heartbeat_interval + self.ping_timeout
                    ):
                        dead_connections.append(connection_id)

                # Remove dead connections
                for connection_id in dead_connections:
                    await self.disconnect(connection_id, "Heartbeat timeout")

                # Send ping to all connections in frontend-compatible format
                for connection_id in list(self.connections.keys()):
                    await self._send_message(
                        connection_id,
                        {
                            "type": "system:heartbeat",
                            "data": {
                                "timestamp": int(current_time * 1000),
                                "server_time": datetime.fromtimestamp(current_time).isoformat(),
                            },
                            "timestamp": int(current_time * 1000),
                            "messageId": f"heartbeat-{int(current_time * 1000)}-{connection_id[:8]}",
                        },
                    )

                await asyncio.sleep(self.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)

    async def _cleanup_loop(self):
        """Background task for cleanup"""
        while self._running:
            try:
                current_time = time.time()

                # Clean up old sessions
                sessions_to_remove = []
                for session_id, session_info in self.sessions.items():
                    if (
                        current_time - session_info.last_activity > self.session_timeout
                        and not session_info.connection_ids
                    ):
                        sessions_to_remove.append(session_id)

                for session_id in sessions_to_remove:
                    del self.sessions[session_id]

                if sessions_to_remove:
                    logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

                await asyncio.sleep(300)  # Clean up every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)

    async def _close_all_connections(self):
        """Close all WebSocket connections"""
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id, "Server shutdown")

    def add_message_handler(self, message_type: MessageType, handler: Callable):
        """Add message handler for specific message type"""
        self.message_handlers[message_type].append(handler)

    def get_connection_info(self, connection_id: str) -> ConnectionInfo | None:
        """Get connection information"""
        return self.connections.get(connection_id)

    def get_session_info(self, session_id: str) -> SessionInfo | None:
        """Get session information"""
        return self.sessions.get(session_id)

    def get_stats(self) -> dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            **self.stats,
            "active_connections": len(self.connections),
            "active_sessions": len(self.sessions),
            "connections_by_state": {
                state.value: sum(1 for conn in self.connections.values() if conn.state == state)
                for state in ConnectionState
            },
        }

    def get_all_connections(self) -> list[dict[str, Any]]:
        """Get all connection information"""
        return [conn.to_dict() for conn in self.connections.values()]

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Get all session information"""
        return [session.to_dict() for session in self.sessions.values()]

    async def get_connection_stats(self) -> dict[str, Any]:
        """Get WebSocket connection statistics for analytics API"""
        current_time = time.time()

        # Calculate connection metrics
        total_connections = len(self.connections)
        active_connections = sum(
            1 for conn in self.connections.values() if conn.state == ConnectionState.CONNECTED
        )

        # Calculate session metrics
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for session in self.sessions.values() if session.connection_ids)

        # Calculate uptime metrics
        connection_uptimes = [
            current_time - conn.connected_at for conn in self.connections.values()
        ]
        avg_connection_uptime = (
            sum(connection_uptimes) / len(connection_uptimes) if connection_uptimes else 0
        )

        # Calculate message rates (approximate based on total messages)
        message_rate = self.stats.get("total_messages", 0) / max(
            1,
            current_time
            - (
                min(conn.connected_at for conn in self.connections.values())
                if self.connections
                else current_time
            ),
        )

        return {
            "total_connections": total_connections,
            "active_connections": active_connections,
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "max_connections": self.max_connections,
            "connection_utilization": active_connections / self.max_connections
            if self.max_connections > 0
            else 0,
            "avg_connection_uptime": avg_connection_uptime,
            "message_rate_per_second": message_rate,
            "total_messages_processed": self.stats.get("total_messages", 0),
            "total_errors": self.stats.get("total_errors", 0),
            "error_rate": self.stats.get("total_errors", 0)
            / max(1, self.stats.get("total_messages", 1)),
            "heartbeat_interval": self.heartbeat_interval,
            "session_timeout": self.session_timeout,
            "connections_by_state": {
                state.value: sum(1 for conn in self.connections.values() if conn.state == state)
                for state in ConnectionState
            },
            "timestamp": current_time,
        }
