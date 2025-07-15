#!/usr/bin/env python3
"""
Enterprise WebSocket Connection Manager

Enhanced connection management system using FastAPI WebSocket
with enterprise-grade features for real-time audio streaming and service coordination.
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, Set, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states"""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    STREAMING = "streaming"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ConnectionInfo:
    """Information about a FastAPI WebSocket connection"""

    connection_id: str
    websocket: WebSocket
    client_ip: str
    user_agent: str
    connected_at: float
    last_ping: float
    state: ConnectionState = ConnectionState.CONNECTED
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    room: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_ping = time.time()

    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """Check if connection has expired due to inactivity"""
        return (time.time() - self.last_ping) > timeout_seconds

    def get_connection_duration(self) -> float:
        """Get total connection duration in seconds"""
        return time.time() - self.connected_at

    def is_connected(self) -> bool:
        """Check if WebSocket is still connected"""
        return self.websocket.client_state == WebSocketState.CONNECTED

    async def send_message(self, message: Dict[str, Any]):
        """Send message through WebSocket"""
        try:
            if self.is_connected():
                await self.websocket.send_text(json.dumps(message))
                self.messages_sent += 1
                self.bytes_sent += len(json.dumps(message))
                self.update_activity()
            else:
                logger.warning(f"Cannot send message - WebSocket {self.connection_id} is not connected")
        except Exception as e:
            logger.error(f"Failed to send message to {self.connection_id}: {e}")
            self.errors += 1
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "connection_id": self.connection_id,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "connected_at": self.connected_at,
            "last_ping": self.last_ping,
            "state": self.state.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "room": self.room,
            "metadata": self.metadata,
            "connection_duration": self.get_connection_duration(),
            "is_connected": self.is_connected(),
            "statistics": {
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
                "errors": self.errors,
            },
        }


class EnterpriseConnectionManager:
    """Enhanced WebSocket connection manager with enterprise features using FastAPI WebSocket"""

    def __init__(
        self,
        max_connections: int = 10000,
        connection_timeout: int = 1800,  # 30 minutes
        max_connections_per_ip: int = 50,
        cleanup_interval: int = 300,  # 5 minutes
        heartbeat_interval: int = 30,
    ):
        """
        Initialize enterprise connection manager

        Args:
            max_connections: Maximum total connections
            connection_timeout: Timeout for inactive connections (seconds)
            max_connections_per_ip: Maximum connections per IP address
            cleanup_interval: Interval for cleanup tasks (seconds)
            heartbeat_interval: Heartbeat interval (seconds)
        """
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.max_connections_per_ip = max_connections_per_ip
        self.cleanup_interval = cleanup_interval
        self.heartbeat_interval = heartbeat_interval

        # Connection tracking
        self.connections: Dict[str, ConnectionInfo] = {}
        self.sessions: Dict[str, Set[str]] = defaultdict(
            set
        )  # session_id -> set of connection_ids
        self.rooms: Dict[str, Set[str]] = defaultdict(set)  # room -> set of connection_ids
        self.ip_connections: Dict[str, Set[str]] = defaultdict(set)  # ip -> set of connection_ids

        # Enterprise features
        self.connection_pool = (
            weakref.WeakValueDictionary()
        )  # Weak references to connections
        self.message_buffer: Dict[str, List[Dict]] = defaultdict(
            list
        )  # Session-based message buffering

        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._heartbeat_task = None
        self._running = False

        # Statistics
        self.stats = {
            "total_connections": 0,
            "total_disconnections": 0,
            "total_timeouts": 0,
            "total_errors": 0,
            "peak_connections": 0,
            "active_connections": 0,
            "active_sessions": 0,
            "active_rooms": 0
        }

        logger.info(
            f"Enterprise WebSocket manager initialized (max: {max_connections})"
        )

    async def start(self):
        """Start the connection manager"""
        async with self._lock:
            if self._running:
                return

            self._running = True

            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info("Enterprise WebSocket connection manager started")

    async def stop(self):
        """Stop the connection manager"""
        async with self._lock:
            self._running = False
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            logger.info("Enterprise WebSocket connection manager stopped")

    async def add_connection(self, websocket: WebSocket, client_ip: str, user_agent: str) -> str:
        """Add a new WebSocket connection"""
        try:
            # Check connection limits
            if len(self.connections) >= self.max_connections:
                logger.warning(f"Max connections reached ({self.max_connections})")
                raise Exception("Maximum connections reached")

            if len(self.ip_connections[client_ip]) >= self.max_connections_per_ip:
                logger.warning(f"IP connection limit exceeded for {client_ip}")
                raise Exception("Too many connections from this IP")

            # Generate unique connection ID
            connection_id = str(uuid.uuid4())

            # Create connection info
            connection = ConnectionInfo(
                connection_id=connection_id,
                websocket=websocket,
                client_ip=client_ip,
                user_agent=user_agent,
                connected_at=time.time(),
                last_ping=time.time(),
            )

            # Add to tracking structures
            async with self._lock:
                self.connections[connection_id] = connection
                self.ip_connections[client_ip].add(connection_id)
                self.connection_pool[connection_id] = connection

                self.stats["total_connections"] += 1
                self.stats["active_connections"] = len(self.connections)
                if len(self.connections) > self.stats["peak_connections"]:
                    self.stats["peak_connections"] = len(self.connections)

            logger.info(f"WebSocket connected: {connection_id} from {client_ip}")

            # Send connection confirmation
            await connection.send_message({
                "type": "connection:established",
                "data": {
                    "connectionId": connection_id,
                    "serverTime": int(time.time() * 1000),
                },
                "timestamp": int(time.time() * 1000),
                "messageId": f"msg-{int(time.time() * 1000)}-{connection_id[:8]}",
            })

            return connection_id

        except Exception as e:
            logger.error(f"Connection handling failed: {e}")
            raise

    async def remove_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Remove a WebSocket connection"""
        try:
            async with self._lock:
                connection = self.connections.pop(connection_id, None)
                if not connection:
                    return None

                # Remove from tracking structures
                self.ip_connections[connection.client_ip].discard(connection_id)
                if not self.ip_connections[connection.client_ip]:
                    del self.ip_connections[connection.client_ip]

                # Remove from sessions and rooms
                if connection.session_id:
                    self.sessions[connection.session_id].discard(connection_id)
                    if not self.sessions[connection.session_id]:
                        del self.sessions[connection.session_id]

                if connection.room:
                    self.rooms[connection.room].discard(connection_id)
                    if not self.rooms[connection.room]:
                        del self.rooms[connection.room]

                # Remove from connection pool
                self.connection_pool.pop(connection_id, None)

                connection.state = ConnectionState.DISCONNECTED
                self.stats["total_disconnections"] += 1
                self.stats["active_connections"] = len(self.connections)

                logger.info(
                    f"WebSocket disconnected: {connection_id} (duration: {connection.get_connection_duration():.2f}s)"
                )

                return connection

        except Exception as e:
            logger.error(f"Connection removal failed: {e}")
            return None

    async def handle_message(self, connection_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebSocket message"""
        try:
            message_type = message.get("type")
            
            # Update connection activity
            await self.update_connection_activity(connection_id, messages_received=1)

            if message_type == "connection:ping":
                return await self._handle_heartbeat(connection_id)
            elif message_type == "session:join":
                return await self._handle_join_session(connection_id, message.get("data", {}).get("sessionId"))
            elif message_type == "session:leave":
                return await self._handle_leave_session(connection_id)
            elif message_type == "service:request":
                return await self._handle_service_request(connection_id, message.get("data", {}))
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return {"error": f"Unknown message type: {message_type}"}

        except Exception as e:
            logger.error(f"Message handling failed: {e}")
            return {"error": str(e)}

    async def _handle_heartbeat(self, connection_id: str) -> Dict[str, Any]:
        """Handle heartbeat message"""
        async with self._lock:
            connection = self.connections.get(connection_id)
            if connection:
                connection.update_activity()

        # Respond with server heartbeat
        if connection:
            await connection.send_message({
                "type": "connection:pong",
                "data": {
                    "timestamp": int(time.time() * 1000),
                    "server_time": time.time(),
                },
                "timestamp": int(time.time() * 1000),
                "messageId": f"pong-{int(time.time() * 1000)}-{connection_id[:8]}",
            })

        return {"status": "heartbeat_received"}

    async def _handle_join_session(
        self, connection_id: str, session_id: Optional[str]
    ) -> Dict[str, Any]:
        """Handle session join request"""
        if not session_id:
            return {"error": "Missing session_id"}

        success = await self.join_session(connection_id, session_id)
        if success:
            # Buffer any pending messages for this session
            buffered_messages = self.message_buffer.get(session_id, [])
            connection = self.connections.get(connection_id)
            if buffered_messages and connection:
                for message in buffered_messages:
                    await connection.send_message({
                        "type": "session:buffered_message",
                        "data": message,
                        "timestamp": int(time.time() * 1000),
                    })
                # Clear buffer after delivery
                self.message_buffer[session_id] = []

            return {"status": "joined", "session_id": session_id}
        else:
            return {"error": "Failed to join session"}

    async def _handle_leave_session(self, connection_id: str) -> Dict[str, Any]:
        """Handle session leave request"""
        success = await self.leave_session(connection_id)
        if success:
            return {"status": "left"}
        else:
            return {"error": "Not in session or failed to leave"}

    async def _handle_service_request(
        self, connection_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle service-specific request"""
        target_service = data.get("target_service")
        action = data.get("action")
        
        # This would route to the appropriate service through the API gateway
        # For now, just acknowledge
        logger.info(
            f"Service request from {connection_id} to {target_service}: {action}"
        )

        connection = self.connections.get(connection_id)
        if connection:
            await connection.send_message({
                "type": "service:response",
                "data": {
                    "source_service": target_service,
                    "response_to": action,
                    "status": "acknowledged",
                    "timestamp": time.time(),
                },
                "timestamp": int(time.time() * 1000),
                "messageId": f"service-{int(time.time() * 1000)}-{connection_id[:8]}",
            })

        return {"status": "routed", "target_service": target_service}

    async def update_connection_activity(
        self,
        connection_id: str,
        bytes_received: int = 0,
        bytes_sent: int = 0,
        messages_received: int = 0,
        messages_sent: int = 0,
    ):
        """Update connection activity and statistics"""
        async with self._lock:
            connection = self.connections.get(connection_id)
            if connection:
                connection.update_activity()
                connection.bytes_received += bytes_received
                connection.bytes_sent += bytes_sent
                connection.messages_received += messages_received
                connection.messages_sent += messages_sent

    async def join_session(self, connection_id: str, session_id: str) -> bool:
        """Join a session"""
        async with self._lock:
            connection = self.connections.get(connection_id)
            if not connection:
                return False

            # Leave previous session if any
            if connection.session_id:
                await self.leave_session(connection_id)

            # Join new session
            connection.session_id = session_id
            self.sessions[session_id].add(connection_id)

            # Update stats
            self.stats["active_sessions"] = len(self.sessions)

            logger.info(f"Connection {connection_id} joined session {session_id}")
            return True

    async def leave_session(self, connection_id: str) -> bool:
        """Leave current session"""
        async with self._lock:
            connection = self.connections.get(connection_id)
            if not connection or not connection.session_id:
                return False

            session_id = connection.session_id
            self.sessions[session_id].discard(connection_id)
            if not self.sessions[session_id]:
                del self.sessions[session_id]

            connection.session_id = None

            # Update stats
            self.stats["active_sessions"] = len(self.sessions)

            logger.info(f"Connection {connection_id} left session {session_id}")
            return True

    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections in a session"""
        async with self._lock:
            connections = self.sessions.get(session_id, set())
            if connections:
                for connection_id in connections:
                    connection = self.connections.get(connection_id)
                    if connection:
                        try:
                            await connection.send_message({
                                "type": "session:broadcast",
                                "data": message,
                                "timestamp": int(time.time() * 1000),
                            })
                            await self.update_connection_activity(connection_id, messages_sent=1)
                        except Exception as e:
                            logger.error(f"Failed to broadcast to {connection_id}: {e}")
                            
                logger.debug(
                    f"Broadcasted to {len(connections)} connections in session {session_id}"
                )
            else:
                # Buffer message if no active connections
                self.message_buffer[session_id].append(message)
                logger.debug(f"Buffered message for session {session_id}")

    async def cleanup_expired_connections(self) -> List[str]:
        """Clean up expired connections and return their IDs"""
        expired_ids = []

        async with self._lock:
            for connection_id, connection in list(self.connections.items()):
                if connection.is_expired(self.connection_timeout):
                    expired_ids.append(connection_id)
                    await self.remove_connection(connection_id)
                    self.stats["total_timeouts"] += 1

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired connections")

        return expired_ids

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive connection manager statistics"""
        # Calculate per-state counts
        state_counts = defaultdict(int)
        for connection in self.connections.values():
            state_counts[connection.state.value] += 1

        # Calculate IP distribution
        ip_distribution = {
            ip: len(connection_ids) for ip, connection_ids in self.ip_connections.items()
        }

        # Calculate message buffer stats
        buffered_sessions = len(self.message_buffer)
        total_buffered_messages = sum(
            len(msgs) for msgs in self.message_buffer.values()
        )

        # Update current stats
        self.stats["active_connections"] = len(self.connections)
        self.stats["active_sessions"] = len(self.sessions)
        self.stats["active_rooms"] = len(self.rooms)

        return {
            **self.stats,
            "state_distribution": dict(state_counts),
            "ip_distribution": ip_distribution,
            "message_buffering": {
                "buffered_sessions": buffered_sessions,
                "total_buffered_messages": total_buffered_messages,
            },
            "configuration": {
                "max_connections": self.max_connections,
                "connection_timeout": self.connection_timeout,
                "max_connections_per_ip": self.max_connections_per_ip,
                "cleanup_interval": self.cleanup_interval,
                "heartbeat_interval": self.heartbeat_interval,
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics (alias for get_statistics)"""
        return self.get_statistics()

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await self.cleanup_expired_connections()

                # Clean up empty message buffers
                async with self._lock:
                    empty_buffers = [
                        session_id
                        for session_id, msgs in self.message_buffer.items()
                        if not msgs
                    ]
                    for session_id in empty_buffers:
                        del self.message_buffer[session_id]

                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.cleanup_interval)

    async def _heartbeat_loop(self):
        """Background heartbeat monitoring loop"""
        while self._running:
            try:
                current_time = time.time()
                stale_connections = []

                async with self._lock:
                    for connection_id, connection in list(self.connections.items()):
                        if current_time - connection.last_ping > self.heartbeat_interval * 2:
                            stale_connections.append(connection_id)

                # Remove stale connections
                for connection_id in stale_connections:
                    logger.warning(f"Removing stale connection: {connection_id}")
                    await self.remove_connection(connection_id)

                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
