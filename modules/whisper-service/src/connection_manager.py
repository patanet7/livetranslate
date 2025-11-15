#!/usr/bin/env python3
"""
WebSocket Connection Manager

Enhanced connection management system for real-time audio streaming.
Provides connection tracking, timeout handling, session management, and health monitoring.
"""

import logging
import time
from typing import Dict, Set, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
import weakref
import gc

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
    
    # Performance optimizations
    _message_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    _last_gc: float = field(default_factory=time.time)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        
        # Periodic garbage collection for long-lived connections
        if time.time() - self._last_gc > 300:  # Every 5 minutes
            gc.collect()
            self._last_gc = time.time()
    
    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """Check if connection has expired due to inactivity"""
        return (datetime.now() - self.last_activity).total_seconds() > timeout_seconds
    
    def get_connection_duration(self) -> timedelta:
        """Get total connection duration"""
        return datetime.now() - self.connected_at
    
    def add_to_buffer(self, message: Any):
        """Add message to buffer (for debugging/monitoring)"""
        self._message_buffer.append({
            'timestamp': time.time(),
            'message': str(message)[:100]  # Truncate for memory
        })
    
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
                "errors": self.errors
            }
        }

class ConnectionPool:
    """Connection pooling for efficient resource management"""
    
    def __init__(self, max_pool_size: int = 1000):
        self.max_pool_size = max_pool_size
        self._pool: deque = deque(maxlen=max_pool_size)
        self._lock = threading.Lock()
        
    def get_connection_info(self, sid: str, client_ip: str, user_agent: str = "", 
                           user_id: Optional[str] = None) -> ConnectionInfo:
        """Get a connection info object from pool or create new"""
        with self._lock:
            if self._pool:
                connection = self._pool.popleft()
                # Reset the connection info
                connection.sid = sid
                connection.client_ip = client_ip
                connection.user_agent = user_agent
                connection.connected_at = datetime.now()
                connection.last_activity = datetime.now()
                connection.state = ConnectionState.CONNECTED
                connection.session_id = None
                connection.user_id = user_id
                connection.room = None
                connection.metadata.clear()
                connection.messages_sent = 0
                connection.messages_received = 0
                connection.bytes_sent = 0
                connection.bytes_received = 0
                connection.errors = 0
                connection._message_buffer.clear()
                return connection
            else:
                return ConnectionInfo(
                    sid=sid,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    connected_at=datetime.now(),
                    last_activity=datetime.now(),
                    user_id=user_id
                )
    
    def return_connection_info(self, connection: ConnectionInfo):
        """Return connection info to pool for reuse"""
        with self._lock:
            if len(self._pool) < self.max_pool_size:
                self._pool.append(connection)

class ConnectionManager:
    """Enhanced WebSocket connection manager with performance optimizations"""
    
    def __init__(self, 
                 connection_timeout: int = 300,
                 max_connections_per_ip: int = 10,
                 cleanup_interval: int = 60,
                 enable_connection_pooling: bool = True,
                 batch_size: int = 50):
        """
        Initialize connection manager
        
        Args:
            connection_timeout: Timeout for inactive connections (seconds)
            max_connections_per_ip: Maximum connections per IP address
            cleanup_interval: Interval for cleanup tasks (seconds)
            enable_connection_pooling: Enable connection object pooling
            batch_size: Batch size for bulk operations
        """
        self.connections: Dict[str, ConnectionInfo] = {}
        self.sessions: Dict[str, Set[str]] = defaultdict(set)  # session_id -> set of sids
        self.rooms: Dict[str, Set[str]] = defaultdict(set)     # room -> set of sids
        self.ip_connections: Dict[str, Set[str]] = defaultdict(set)  # ip -> set of sids
        
        self.connection_timeout = connection_timeout
        self.max_connections_per_ip = max_connections_per_ip
        self.cleanup_interval = cleanup_interval
        self.batch_size = batch_size
        
        # Performance optimizations
        self.connection_pool = ConnectionPool() if enable_connection_pooling else None
        self._weak_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
        self._lock = threading.RLock()
        self._cleanup_task = None
        self._running = False
        
        # Statistics
        self.total_connections = 0
        self.total_disconnections = 0
        self.total_timeouts = 0
        self.total_errors = 0
        self.pool_hits = 0
        self.pool_misses = 0
        
    def start(self):
        """Start the connection manager"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._cleanup_task = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_task.start()
            logger.info("Connection manager started")
    
    def stop(self):
        """Stop the connection manager"""
        with self._lock:
            self._running = False
            if self._cleanup_task:
                self._cleanup_task.join(timeout=5)
            logger.info("Connection manager stopped")
    
    def add_connection(self, 
                      sid: str, 
                      client_ip: str, 
                      user_agent: str = "",
                      user_id: Optional[str] = None) -> bool:
        """
        Add a new connection
        
        Args:
            sid: Socket ID
            client_ip: Client IP address
            user_agent: User agent string
            user_id: Optional user ID
            
        Returns:
            True if connection was added, False if rejected
        """
        with self._lock:
            # Check IP connection limit
            if len(self.ip_connections[client_ip]) >= self.max_connections_per_ip:
                logger.warning(f"Connection limit exceeded for IP {client_ip}")
                return False
            
            # Create connection info
            if self.connection_pool:
                connection = self.connection_pool.get_connection_info(
                    sid=sid,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    user_id=user_id
                )
                self.pool_hits += 1
            else:
                connection = ConnectionInfo(
                    sid=sid,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    connected_at=datetime.now(),
                    last_activity=datetime.now(),
                    user_id=user_id
                )
                self.pool_misses += 1
            
            # Add to tracking structures
            self.connections[sid] = connection
            self.ip_connections[client_ip].add(sid)
            self._weak_refs[sid] = connection
            
            self.total_connections += 1
            
            logger.info(f"Connection added: {sid} from {client_ip}")
            return True
    
    def remove_connection(self, sid: str) -> Optional[ConnectionInfo]:
        """
        Remove a connection with optimized cleanup
        
        Args:
            sid: Socket ID
            
        Returns:
            ConnectionInfo if found, None otherwise
        """
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
            
            connection.state = ConnectionState.DISCONNECTED
            self.total_disconnections += 1
            
            # Return connection to pool for reuse
            if self.connection_pool:
                self.connection_pool.return_connection_info(connection)
            
            # Remove from weak references
            self._weak_refs.pop(sid, None)
            
            logger.info(f"Connection removed: {sid}")
            return connection
    
    def get_connection(self, sid: str) -> Optional[ConnectionInfo]:
        """Get connection info by socket ID"""
        with self._lock:
            return self.connections.get(sid)
    
    def update_connection_activity(self, sid: str, 
                                 bytes_received: int = 0, 
                                 bytes_sent: int = 0,
                                 messages_received: int = 0,
                                 messages_sent: int = 0):
        """Update connection activity and statistics"""
        with self._lock:
            connection = self.connections.get(sid)
            if connection:
                connection.update_activity()
                connection.bytes_received += bytes_received
                connection.bytes_sent += bytes_sent
                connection.messages_received += messages_received
                connection.messages_sent += messages_sent
    
    def set_connection_state(self, sid: str, state: ConnectionState):
        """Set connection state"""
        with self._lock:
            connection = self.connections.get(sid)
            if connection:
                connection.state = state
                logger.debug(f"Connection {sid} state changed to {state.value}")
    
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
    
    def join_room(self, sid: str, room: str) -> bool:
        """Join a room"""
        with self._lock:
            connection = self.connections.get(sid)
            if not connection:
                return False
            
            # Leave previous room if any
            if connection.room:
                self.leave_room(sid)
            
            # Join new room
            connection.room = room
            self.rooms[room].add(sid)
            
            logger.debug(f"Connection {sid} joined room {room}")
            return True
    
    def leave_room(self, sid: str) -> bool:
        """Leave current room"""
        with self._lock:
            connection = self.connections.get(sid)
            if not connection or not connection.room:
                return False
            
            room = connection.room
            self.rooms[room].discard(sid)
            if not self.rooms[room]:
                del self.rooms[room]
            
            connection.room = None
            
            logger.debug(f"Connection {sid} left room {room}")
            return True
    
    def get_session_connections(self, session_id: str) -> List[str]:
        """Get all connection IDs in a session"""
        with self._lock:
            return list(self.sessions.get(session_id, set()))
    
    def get_room_connections(self, room: str) -> List[str]:
        """Get all connection IDs in a room"""
        with self._lock:
            return list(self.rooms.get(room, set()))
    
    def get_active_connections(self) -> List[str]:
        """Get all active connection IDs"""
        with self._lock:
            return list(self.connections.keys())
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        with self._lock:
            return len(self.connections)
    
    def get_session_count(self) -> int:
        """Get total number of active sessions"""
        with self._lock:
            return len(self.sessions)
    
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
        """Get comprehensive connection manager statistics with performance metrics"""
        with self._lock:
            active_connections = len(self.connections)
            active_sessions = len(self.sessions)
            active_rooms = len(self.rooms)
            
            # Calculate per-state counts and performance metrics
            state_counts = defaultdict(int)
            total_bytes_sent = 0
            total_bytes_received = 0
            total_messages_sent = 0
            total_messages_received = 0
            
            for connection in self.connections.values():
                state_counts[connection.state.value] += 1
                total_bytes_sent += connection.bytes_sent
                total_bytes_received += connection.bytes_received
                total_messages_sent += connection.messages_sent
                total_messages_received += connection.messages_received
            
            # Calculate IP distribution
            ip_distribution = {ip: len(sids) for ip, sids in self.ip_connections.items()}
            
            # Pool efficiency metrics
            pool_efficiency = 0
            if (self.pool_hits + self.pool_misses) > 0:
                pool_efficiency = (self.pool_hits / (self.pool_hits + self.pool_misses)) * 100
            
            return {
                "active_connections": active_connections,
                "active_sessions": active_sessions,
                "active_rooms": active_rooms,
                "total_connections": self.total_connections,
                "total_disconnections": self.total_disconnections,
                "total_timeouts": self.total_timeouts,
                "total_errors": self.total_errors,
                "state_distribution": dict(state_counts),
                "ip_distribution": ip_distribution,
                "performance_metrics": {
                    "total_bytes_sent": total_bytes_sent,
                    "total_bytes_received": total_bytes_received,
                    "total_messages_sent": total_messages_sent,
                    "total_messages_received": total_messages_received,
                    "pool_hits": self.pool_hits,
                    "pool_misses": self.pool_misses,
                    "pool_efficiency_percent": round(pool_efficiency, 2),
                    "memory_usage": {
                        "connections": len(self.connections),
                        "weak_refs": len(self._weak_refs),
                        "pool_size": len(self.connection_pool._pool) if self.connection_pool else 0,
                        "pool_max_size": self.connection_pool.max_pool_size if self.connection_pool else 0
                    }
                },
                "configuration": {
                    "connection_timeout": self.connection_timeout,
                    "max_connections_per_ip": self.max_connections_per_ip,
                    "cleanup_interval": self.cleanup_interval,
                    "batch_size": self.batch_size,
                    "connection_pooling_enabled": self.connection_pool is not None
                }
            }
    
    def get_detailed_connections(self) -> List[Dict[str, Any]]:
        """Get detailed information about all connections"""
        with self._lock:
            return [conn.to_dict() for conn in self.connections.values()]
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                self.cleanup_expired_connections()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(self.cleanup_interval)

# Global connection manager instance
connection_manager = ConnectionManager() 