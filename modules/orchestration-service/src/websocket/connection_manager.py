#!/usr/bin/env python3
"""
Enterprise WebSocket Connection Manager

Enhanced connection management system based on the existing websocket-service
with enterprise-grade features for real-time audio streaming and service coordination.
"""

import asyncio
import logging
import time
import uuid
import threading
from typing import Dict, Set, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import weakref

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
    """Information about a WebSocket connection"""

    sid: str
    client_ip: str
    user_agent: str
    connected_at: datetime
    last_activity: datetime
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
        self.last_activity = datetime.now()

    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """Check if connection has expired due to inactivity"""
        return (datetime.now() - self.last_activity).total_seconds() > timeout_seconds

    def get_connection_duration(self) -> timedelta:
        """Get total connection duration"""
        return datetime.now() - self.connected_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "sid": self.sid,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "state": self.state.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "room": self.room,
            "metadata": self.metadata,
            "connection_duration": str(self.get_connection_duration()),
            "statistics": {
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
                "errors": self.errors,
            },
        }


class EnterpriseConnectionManager:
    """Enhanced WebSocket connection manager with enterprise features"""

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
        )  # session_id -> set of sids
        self.rooms: Dict[str, Set[str]] = defaultdict(set)  # room -> set of sids
        self.ip_connections: Dict[str, Set[str]] = defaultdict(set)  # ip -> set of sids

        # Enterprise features
        self.connection_pool = (
            weakref.WeakValueDictionary()
        )  # Weak references to connections
        self.message_buffer: Dict[str, List[Dict]] = defaultdict(
            list
        )  # Session-based message buffering
        self.heartbeat_tracking: Dict[str, float] = {}  # SID -> last heartbeat time

        self._lock = threading.RLock()
        self._cleanup_task = None
        self._heartbeat_task = None
        self._running = False
        self._socketio = None  # Will be set by orchestration service

        # Statistics
        self.total_connections = 0
        self.total_disconnections = 0
        self.total_timeouts = 0
        self.total_errors = 0
        self.peak_connections = 0

        logger.info(
            f"Enterprise WebSocket manager initialized (max: {max_connections})"
        )

    def set_socketio(self, socketio):
        """Set the SocketIO instance for real-time operations"""
        self._socketio = socketio
        logger.info("SocketIO instance attached to connection manager")

    async def start(self):
        """Start the connection manager"""
        with self._lock:
            if self._running:
                return

            self._running = True

            # Start background tasks
            self._cleanup_task = threading.Thread(
                target=self._cleanup_loop, daemon=True
            )
            self._cleanup_task.start()

            self._heartbeat_task = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_task.start()

            logger.info("Enterprise WebSocket connection manager started")

    async def stop(self):
        """Stop the connection manager"""
        with self._lock:
            self._running = False
            if self._cleanup_task:
                self._cleanup_task.join(timeout=5)
            if self._heartbeat_task:
                self._heartbeat_task.join(timeout=5)
            logger.info("Enterprise WebSocket connection manager stopped")

    def handle_connect(self, request) -> Dict[str, Any]:
        """Handle new WebSocket connection"""
        try:
            sid = request.sid
            client_ip = request.environ.get("REMOTE_ADDR", "unknown")
            user_agent = request.environ.get("HTTP_USER_AGENT", "")

            # Check connection limits
            if len(self.connections) >= self.max_connections:
                logger.warning(f"Max connections reached ({self.max_connections})")
                return {"error": "Maximum connections reached"}

            if len(self.ip_connections[client_ip]) >= self.max_connections_per_ip:
                logger.warning(f"IP connection limit exceeded for {client_ip}")
                return {"error": "Too many connections from this IP"}

            # Create connection info
            connection = ConnectionInfo(
                sid=sid,
                client_ip=client_ip,
                user_agent=user_agent,
                connected_at=datetime.now(),
                last_activity=datetime.now(),
            )

            # Add to tracking structures
            with self._lock:
                self.connections[sid] = connection
                self.ip_connections[client_ip].add(sid)
                self.connection_pool[sid] = connection
                self.heartbeat_tracking[sid] = time.time()

                self.total_connections += 1
                if len(self.connections) > self.peak_connections:
                    self.peak_connections = len(self.connections)

            logger.info(f"WebSocket connected: {sid} from {client_ip}")

            # Emit connection confirmation
            if self._socketio:
                self._socketio.emit(
                    "connected",
                    {
                        "session_id": sid,
                        "status": "connected",
                        "server_time": time.time(),
                    },
                    room=sid,
                )

            return {"status": "connected", "session_id": sid}

        except Exception as e:
            logger.error(f"Connection handling failed: {e}")
            return {"error": str(e)}

    def handle_disconnect(self, request) -> Dict[str, Any]:
        """Handle WebSocket disconnection"""
        try:
            sid = request.sid

            connection = self.remove_connection(sid)
            if connection:
                logger.info(
                    f"WebSocket disconnected: {sid} (duration: {connection.get_connection_duration()})"
                )
                return {
                    "status": "disconnected",
                    "duration": str(connection.get_connection_duration()),
                }
            else:
                logger.warning(f"Disconnect for unknown connection: {sid}")
                return {"status": "unknown"}

        except Exception as e:
            logger.error(f"Disconnect handling failed: {e}")
            return {"error": str(e)}

    def handle_service_message(self, request, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle service-specific WebSocket messages"""
        try:
            sid = request.sid
            message_type = data.get("type")
            target_service = data.get("target_service")

            # Update connection activity
            self.update_connection_activity(sid, messages_received=1)

            if message_type == "heartbeat":
                return self._handle_heartbeat(sid)
            elif message_type == "join_session":
                return self._handle_join_session(sid, data.get("session_id"))
            elif message_type == "leave_session":
                return self._handle_leave_session(sid)
            elif message_type == "service_request":
                return self._handle_service_request(sid, target_service, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return {"error": f"Unknown message type: {message_type}"}

        except Exception as e:
            logger.error(f"Service message handling failed: {e}")
            return {"error": str(e)}

    def _handle_heartbeat(self, sid: str) -> Dict[str, Any]:
        """Handle heartbeat message"""
        with self._lock:
            self.heartbeat_tracking[sid] = time.time()
            connection = self.connections.get(sid)
            if connection:
                connection.update_activity()

        # Respond with server heartbeat
        if self._socketio:
            self._socketio.emit(
                "heartbeat_response",
                {"server_time": time.time(), "status": "alive"},
                room=sid,
            )

        return {"status": "heartbeat_received"}

    def _handle_join_session(
        self, sid: str, session_id: Optional[str]
    ) -> Dict[str, Any]:
        """Handle session join request"""
        if not session_id:
            return {"error": "Missing session_id"}

        success = self.join_session(sid, session_id)
        if success:
            # Buffer any pending messages for this session
            buffered_messages = self.message_buffer.get(session_id, [])
            if buffered_messages and self._socketio:
                for message in buffered_messages:
                    self._socketio.emit("buffered_message", message, room=sid)
                # Clear buffer after delivery
                self.message_buffer[session_id] = []

            return {"status": "joined", "session_id": session_id}
        else:
            return {"error": "Failed to join session"}

    def _handle_leave_session(self, sid: str) -> Dict[str, Any]:
        """Handle session leave request"""
        success = self.leave_session(sid)
        if success:
            return {"status": "left"}
        else:
            return {"error": "Not in session or failed to leave"}

    def _handle_service_request(
        self, sid: str, target_service: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle service-specific request"""
        # This would route to the appropriate service through the API gateway
        # For now, just acknowledge
        logger.info(
            f"Service request from {sid} to {target_service}: {data.get('action')}"
        )

        if self._socketio:
            self._socketio.emit(
                "service_response",
                {
                    "source_service": target_service,
                    "response_to": data.get("action"),
                    "status": "acknowledged",
                    "timestamp": time.time(),
                },
                room=sid,
            )

        return {"status": "routed", "target_service": target_service}

    def remove_connection(self, sid: str) -> Optional[ConnectionInfo]:
        """Remove a connection"""
        with self._lock:
            connection = self.connections.pop(sid, None)
            if not connection:
                return None

            # Remove from tracking structures
            self.ip_connections[connection.client_ip].discard(sid)
            if not self.ip_connections[connection.client_ip]:
                del self.ip_connections[connection.client_ip]

            # Remove from sessions and rooms
            if connection.session_id:
                self.sessions[connection.session_id].discard(sid)
                if not self.sessions[connection.session_id]:
                    del self.sessions[connection.session_id]

            if connection.room:
                self.rooms[connection.room].discard(sid)
                if not self.rooms[connection.room]:
                    del self.rooms[connection.room]

            # Remove from tracking
            self.heartbeat_tracking.pop(sid, None)
            self.connection_pool.pop(sid, None)

            connection.state = ConnectionState.DISCONNECTED
            self.total_disconnections += 1

            return connection

    def update_connection_activity(
        self,
        sid: str,
        bytes_received: int = 0,
        bytes_sent: int = 0,
        messages_received: int = 0,
        messages_sent: int = 0,
    ):
        """Update connection activity and statistics"""
        with self._lock:
            connection = self.connections.get(sid)
            if connection:
                connection.update_activity()
                connection.bytes_received += bytes_received
                connection.bytes_sent += bytes_sent
                connection.messages_received += messages_received
                connection.messages_sent += messages_sent

    def join_session(self, sid: str, session_id: str) -> bool:
        """Join a session"""
        with self._lock:
            connection = self.connections.get(sid)
            if not connection:
                return False

            # Leave previous session if any
            if connection.session_id:
                self.leave_session(sid)

            # Join new session
            connection.session_id = session_id
            self.sessions[session_id].add(sid)

            logger.info(f"Connection {sid} joined session {session_id}")
            return True

    def leave_session(self, sid: str) -> bool:
        """Leave current session"""
        with self._lock:
            connection = self.connections.get(sid)
            if not connection or not connection.session_id:
                return False

            session_id = connection.session_id
            self.sessions[session_id].discard(sid)
            if not self.sessions[session_id]:
                del self.sessions[session_id]

            connection.session_id = None

            logger.info(f"Connection {sid} left session {session_id}")
            return True

    def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections in a session"""
        if not self._socketio:
            logger.warning("No SocketIO instance available for broadcast")
            return

        with self._lock:
            connections = self.sessions.get(session_id, set())
            if connections:
                for sid in connections:
                    self._socketio.emit("session_broadcast", message, room=sid)
                    self.update_connection_activity(sid, messages_sent=1)
                logger.debug(
                    f"Broadcasted to {len(connections)} connections in session {session_id}"
                )
            else:
                # Buffer message if no active connections
                self.message_buffer[session_id].append(message)
                logger.debug(f"Buffered message for session {session_id}")

    def cleanup_expired_connections(self) -> List[str]:
        """Clean up expired connections and return their IDs"""
        expired_sids = []

        with self._lock:
            for sid, connection in list(self.connections.items()):
                if connection.is_expired(self.connection_timeout):
                    expired_sids.append(sid)
                    self.remove_connection(sid)
                    self.total_timeouts += 1

        if expired_sids:
            logger.info(f"Cleaned up {len(expired_sids)} expired connections")

        return expired_sids

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive connection manager statistics"""
        with self._lock:
            active_connections = len(self.connections)
            active_sessions = len(self.sessions)
            active_rooms = len(self.rooms)

            # Calculate per-state counts
            state_counts = defaultdict(int)
            for connection in self.connections.values():
                state_counts[connection.state.value] += 1

            # Calculate IP distribution
            ip_distribution = {
                ip: len(sids) for ip, sids in self.ip_connections.items()
            }

            # Calculate message buffer stats
            buffered_sessions = len(self.message_buffer)
            total_buffered_messages = sum(
                len(msgs) for msgs in self.message_buffer.values()
            )

            return {
                "active_connections": active_connections,
                "active_sessions": active_sessions,
                "active_rooms": active_rooms,
                "peak_connections": self.peak_connections,
                "total_connections": self.total_connections,
                "total_disconnections": self.total_disconnections,
                "total_timeouts": self.total_timeouts,
                "total_errors": self.total_errors,
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

    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                self.cleanup_expired_connections()

                # Clean up empty message buffers
                with self._lock:
                    empty_buffers = [
                        session_id
                        for session_id, msgs in self.message_buffer.items()
                        if not msgs
                    ]
                    for session_id in empty_buffers:
                        del self.message_buffer[session_id]

                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(self.cleanup_interval)

    def _heartbeat_loop(self):
        """Background heartbeat monitoring loop"""
        while self._running:
            try:
                current_time = time.time()
                stale_connections = []

                with self._lock:
                    for sid, last_heartbeat in self.heartbeat_tracking.items():
                        if current_time - last_heartbeat > self.heartbeat_interval * 2:
                            stale_connections.append(sid)

                # Remove stale connections
                for sid in stale_connections:
                    logger.warning(f"Removing stale connection: {sid}")
                    self.remove_connection(sid)

                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(self.heartbeat_interval)
