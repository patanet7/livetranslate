#!/usr/bin/env python3
"""
Reconnection Manager

Handles client reconnections gracefully with session persistence,
message buffering, and state recovery for WebSocket connections.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session states for reconnection handling"""

    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


@dataclass
class BufferedMessage:
    """A message buffered for disconnected clients"""

    message_id: str
    message_type: str
    data: Any
    timestamp: datetime
    priority: int = 0  # Higher priority messages sent first
    attempts: int = 0
    max_attempts: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "attempts": self.attempts,
        }


@dataclass
class SessionInfo:
    """Information about a client session for reconnection"""

    session_id: str
    user_id: str | None
    original_connection_id: str
    current_connection_id: str | None
    state: SessionState
    created_at: datetime
    last_activity: datetime
    disconnected_at: datetime | None = None
    reconnected_at: datetime | None = None

    # Session data
    session_data: dict[str, Any] = field(default_factory=dict)
    buffered_messages: list[BufferedMessage] = field(default_factory=list)

    # Configuration
    max_buffer_size: int = 100
    session_timeout_minutes: int = 30

    def is_expired(self) -> bool:
        """Check if session has expired"""
        if self.state == SessionState.EXPIRED:
            return True

        timeout = timedelta(minutes=self.session_timeout_minutes)
        if self.disconnected_at:
            return datetime.now() - self.disconnected_at > timeout

        return False

    def add_buffered_message(self, message: BufferedMessage) -> bool:
        """Add a message to the buffer"""
        if len(self.buffered_messages) >= self.max_buffer_size:
            # Remove oldest low-priority message
            low_priority_msgs = [m for m in self.buffered_messages if m.priority <= 0]
            if low_priority_msgs:
                oldest = min(low_priority_msgs, key=lambda m: m.timestamp)
                self.buffered_messages.remove(oldest)
            else:
                # Remove oldest message if all are high priority
                oldest = min(self.buffered_messages, key=lambda m: m.timestamp)
                self.buffered_messages.remove(oldest)

        self.buffered_messages.append(message)
        # Sort by priority (high to low) then by timestamp (old to new)
        self.buffered_messages.sort(key=lambda m: (-m.priority, m.timestamp))
        return True

    def get_pending_messages(self) -> list[BufferedMessage]:
        """Get messages that haven't exceeded max attempts"""
        return [m for m in self.buffered_messages if m.attempts < m.max_attempts]

    def clear_delivered_messages(self, delivered_ids: list[str]):
        """Remove delivered messages from buffer"""
        self.buffered_messages = [
            m for m in self.buffered_messages if m.message_id not in delivered_ids
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "original_connection_id": self.original_connection_id,
            "current_connection_id": self.current_connection_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "disconnected_at": self.disconnected_at.isoformat() if self.disconnected_at else None,
            "reconnected_at": self.reconnected_at.isoformat() if self.reconnected_at else None,
            "session_data": self.session_data,
            "buffered_message_count": len(self.buffered_messages),
            "pending_message_count": len(self.get_pending_messages()),
            "max_buffer_size": self.max_buffer_size,
            "session_timeout_minutes": self.session_timeout_minutes,
        }


class ReconnectionManager:
    """Manages client reconnections and session persistence"""

    def __init__(self):
        self.sessions: dict[str, SessionInfo] = {}  # session_id -> SessionInfo
        self.connection_sessions: dict[str, str] = {}  # connection_id -> session_id
        self.user_sessions: dict[str, list[str]] = {}  # user_id -> [session_ids]

        # Configuration
        self.default_session_timeout = 30  # minutes
        self.default_buffer_size = 100
        self.cleanup_interval = 300  # 5 minutes
        self.max_reconnection_attempts = 5

        # Statistics
        self.total_sessions_created = 0
        self.total_reconnections = 0
        self.total_expired_sessions = 0
        self.total_buffered_messages = 0

        # Thread safety
        self.lock = threading.RLock()

        # Start cleanup thread
        self._start_cleanup_thread()

        logger.info("Reconnection manager initialized")

    def create_session(
        self,
        connection_id: str,
        user_id: str | None = None,
        session_data: dict[str, Any] | None = None,
    ) -> str:
        """Create a new session for a connection"""
        with self.lock:
            session_id = str(uuid.uuid4())

            session_info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                original_connection_id=connection_id,
                current_connection_id=connection_id,
                state=SessionState.ACTIVE,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                session_data=session_data or {},
            )

            self.sessions[session_id] = session_info
            self.connection_sessions[connection_id] = session_id

            if user_id:
                if user_id not in self.user_sessions:
                    self.user_sessions[user_id] = []
                self.user_sessions[user_id].append(session_id)

            self.total_sessions_created += 1

            logger.info(f"Created session {session_id} for connection {connection_id}")
            return session_id

    def handle_disconnection(self, connection_id: str) -> SessionInfo | None:
        """Handle client disconnection"""
        with self.lock:
            session_id = self.connection_sessions.get(connection_id)
            if not session_id:
                return None

            session_info = self.sessions.get(session_id)
            if not session_info:
                return None

            # Update session state
            session_info.state = SessionState.DISCONNECTED
            session_info.disconnected_at = datetime.now()
            session_info.current_connection_id = None

            # Remove connection mapping
            self.connection_sessions.pop(connection_id, None)

            logger.info(f"Session {session_id} disconnected (connection {connection_id})")
            return session_info

    def attempt_reconnection(
        self, connection_id: str, session_id: str, user_id: str | None = None
    ) -> bool:
        """Attempt to reconnect a client to an existing session"""
        with self.lock:
            session_info = self.sessions.get(session_id)
            if not session_info:
                logger.warning(f"Reconnection failed: session {session_id} not found")
                return False

            if session_info.is_expired():
                logger.warning(f"Reconnection failed: session {session_id} expired")
                session_info.state = SessionState.EXPIRED
                return False

            # Verify user identity if provided
            if user_id and session_info.user_id != user_id:
                logger.warning(f"Reconnection failed: user mismatch for session {session_id}")
                return False

            # Update session for reconnection
            session_info.current_connection_id = connection_id
            session_info.state = SessionState.ACTIVE
            session_info.reconnected_at = datetime.now()
            session_info.last_activity = datetime.now()

            # Update mappings
            self.connection_sessions[connection_id] = session_id

            self.total_reconnections += 1

            logger.info(
                f"Successfully reconnected session {session_id} to connection {connection_id}"
            )
            return True

    def buffer_message(
        self, session_id: str, message_type: str, data: Any, priority: int = 0
    ) -> bool:
        """Buffer a message for a disconnected session"""
        with self.lock:
            session_info = self.sessions.get(session_id)
            if not session_info:
                return False

            if session_info.state not in [SessionState.DISCONNECTED, SessionState.RECONNECTING]:
                # Session is active, no need to buffer
                return False

            message = BufferedMessage(
                message_id=str(uuid.uuid4()),
                message_type=message_type,
                data=data,
                timestamp=datetime.now(),
                priority=priority,
            )

            success = session_info.add_buffered_message(message)
            if success:
                self.total_buffered_messages += 1

            return success

    def get_session_by_connection(self, connection_id: str) -> SessionInfo | None:
        """Get session info by connection ID"""
        with self.lock:
            session_id = self.connection_sessions.get(connection_id)
            if session_id:
                return self.sessions.get(session_id)
            return None

    def get_session(self, session_id: str) -> SessionInfo | None:
        """Get session info by session ID"""
        with self.lock:
            return self.sessions.get(session_id)

    def update_session_data(self, session_id: str, data: dict[str, Any]) -> bool:
        """Update session data"""
        with self.lock:
            session_info = self.sessions.get(session_id)
            if not session_info:
                return False

            session_info.session_data.update(data)
            session_info.last_activity = datetime.now()
            return True

    def get_pending_messages(self, session_id: str) -> list[BufferedMessage]:
        """Get pending messages for a session"""
        with self.lock:
            session_info = self.sessions.get(session_id)
            if not session_info:
                return []

            return session_info.get_pending_messages()

    def mark_messages_delivered(self, session_id: str, message_ids: list[str]):
        """Mark messages as delivered"""
        with self.lock:
            session_info = self.sessions.get(session_id)
            if session_info:
                session_info.clear_delivered_messages(message_ids)

    def mark_message_failed(self, session_id: str, message_id: str):
        """Mark a message delivery as failed"""
        with self.lock:
            session_info = self.sessions.get(session_id)
            if not session_info:
                return

            for message in session_info.buffered_messages:
                if message.message_id == message_id:
                    message.attempts += 1
                    break

    def get_user_sessions(self, user_id: str) -> list[SessionInfo]:
        """Get all sessions for a user"""
        with self.lock:
            session_ids = self.user_sessions.get(user_id, [])
            return [self.sessions[sid] for sid in session_ids if sid in self.sessions]

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        with self.lock:
            expired_sessions = []

            for session_id, session_info in self.sessions.items():
                if session_info.is_expired():
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                self._remove_session(session_id)
                self.total_expired_sessions += 1

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def _remove_session(self, session_id: str):
        """Remove a session and all its references"""
        session_info = self.sessions.pop(session_id, None)
        if not session_info:
            return

        # Remove connection mapping
        if session_info.current_connection_id:
            self.connection_sessions.pop(session_info.current_connection_id, None)

        # Remove from user sessions
        if session_info.user_id:
            user_sessions = self.user_sessions.get(session_info.user_id, [])
            if session_id in user_sessions:
                user_sessions.remove(session_id)
                if not user_sessions:
                    self.user_sessions.pop(session_info.user_id, None)

    def _start_cleanup_thread(self):
        """Start the cleanup thread"""

        def cleanup_loop():
            while True:
                try:
                    self.cleanup_expired_sessions()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
                    time.sleep(60)  # Retry after 1 minute on error

        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info("Started session cleanup thread")

    def get_statistics(self) -> dict[str, Any]:
        """Get reconnection manager statistics"""
        with self.lock:
            active_sessions = len(
                [s for s in self.sessions.values() if s.state == SessionState.ACTIVE]
            )
            disconnected_sessions = len(
                [s for s in self.sessions.values() if s.state == SessionState.DISCONNECTED]
            )
            total_buffered = sum(len(s.buffered_messages) for s in self.sessions.values())

            return {
                "total_sessions": len(self.sessions),
                "active_sessions": active_sessions,
                "disconnected_sessions": disconnected_sessions,
                "total_sessions_created": self.total_sessions_created,
                "total_reconnections": self.total_reconnections,
                "total_expired_sessions": self.total_expired_sessions,
                "total_buffered_messages": self.total_buffered_messages,
                "current_buffered_messages": total_buffered,
                "unique_users": len(self.user_sessions),
                "default_session_timeout_minutes": self.default_session_timeout,
                "default_buffer_size": self.default_buffer_size,
            }


# Global reconnection manager instance
reconnection_manager = ReconnectionManager()
