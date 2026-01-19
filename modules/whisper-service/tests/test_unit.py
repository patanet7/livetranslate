#!/usr/bin/env python3
"""
Unit Tests for WebSocket Server Components

Tests individual components and modules in isolation.
Updated to match current API signatures.
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from connection_manager import ConnectionInfo, ConnectionManager, ConnectionState
from error_handler import ErrorCategory, ErrorHandler, ErrorInfo, ErrorSeverity
from heartbeat_manager import HeartbeatInfo, HeartbeatManager, HeartbeatState
from message_router import MessageRouter, MessageType
from reconnection_manager import ReconnectionManager
from simple_auth import SimpleAuth, UserRole


class TestConnectionManager:
    """Unit tests for ConnectionManager"""

    def test_connection_manager_initialization(self):
        """Test ConnectionManager initialization"""
        cm = ConnectionManager()

        assert cm.connections == {}
        assert cm.ip_connections == {}
        assert cm.rooms == {}
        assert cm.sessions == {}
        assert cm.total_connections == 0
        assert cm.total_disconnections == 0

    def test_add_connection(self):
        """Test adding a connection"""
        cm = ConnectionManager()

        success = cm.add_connection(
            sid="test_sid", client_ip="127.0.0.1", user_agent="test_agent", user_id="test_user"
        )

        assert success is True
        assert "test_sid" in cm.connections
        assert cm.total_connections == 1

        connection = cm.connections["test_sid"]
        assert connection.sid == "test_sid"
        assert connection.client_ip == "127.0.0.1"
        assert connection.user_agent == "test_agent"
        assert connection.user_id == "test_user"
        assert connection.state == ConnectionState.CONNECTED

    def test_remove_connection(self):
        """Test removing a connection"""
        cm = ConnectionManager()

        # Add connection first
        cm.add_connection("test_sid", "127.0.0.1", "test_agent", "test_user")

        # Remove connection
        removed = cm.remove_connection("test_sid")

        assert removed is not None
        assert removed.sid == "test_sid"
        assert "test_sid" not in cm.connections
        assert cm.total_disconnections == 1

    def test_get_connection(self):
        """Test getting a connection"""
        cm = ConnectionManager()

        # Add connection
        cm.add_connection("test_sid", "127.0.0.1", "test_agent", "test_user")

        # Get connection
        connection = cm.get_connection("test_sid")
        assert connection is not None
        assert connection.sid == "test_sid"

        # Get non-existent connection
        non_existent = cm.get_connection("non_existent")
        assert non_existent is None

    def test_ip_rate_limiting(self):
        """Test IP-based rate limiting"""
        cm = ConnectionManager(max_connections_per_ip=1)

        # Add first connection from IP
        success1 = cm.add_connection("sid1", "127.0.0.1", "agent1", "user1")
        assert success1 is True

        # Try to add second connection from same IP (should fail due to rate limit)
        success2 = cm.add_connection("sid2", "127.0.0.1", "agent2", "user2")
        assert success2 is False

    def test_room_management(self):
        """Test room join/leave functionality"""
        cm = ConnectionManager()
        cm.add_connection("test_sid", "127.0.0.1", "test_agent", "test_user")

        # Join room
        cm.join_room("test_sid", "test_room")
        assert "test_room" in cm.rooms
        assert "test_sid" in cm.rooms["test_room"]

        # Leave room (uses connection's current room)
        cm.leave_room("test_sid")
        assert "test_sid" not in cm.rooms.get("test_room", set())

    def test_connection_statistics(self):
        """Test connection statistics"""
        cm = ConnectionManager()

        # Add some connections
        cm.add_connection("sid1", "127.0.0.1", "agent1", "user1")
        cm.add_connection("sid2", "192.168.1.1", "agent2", "user2")

        stats = cm.get_statistics()

        assert stats["active_connections"] == 2
        assert stats["total_connections"] == 2
        assert stats["active_rooms"] == 0
        assert "127.0.0.1" in stats["ip_distribution"]
        assert "192.168.1.1" in stats["ip_distribution"]


class TestErrorHandler:
    """Unit tests for ErrorHandler"""

    def test_error_handler_initialization(self):
        """Test ErrorHandler initialization"""
        eh = ErrorHandler()
        assert eh.error_counts == {}
        assert isinstance(eh.recovery_handlers, dict)

    def test_handle_connection_error(self):
        """Test handling connection errors"""
        eh = ErrorHandler()

        error = ErrorInfo(
            category=ErrorCategory.CONNECTION_FAILED,
            severity=ErrorSeverity.MEDIUM,
            message="Connection failed",
            details="timeout",
            connection_id="test_sid",
        )

        response = eh.handle_error(error)

        assert response.category == ErrorCategory.CONNECTION_FAILED
        assert response.message == "Connection failed"
        assert eh.error_counts.get(ErrorCategory.CONNECTION_FAILED, 0) > 0

    def test_handle_authentication_error(self):
        """Test handling authentication errors"""
        eh = ErrorHandler()

        error = ErrorInfo(
            category=ErrorCategory.AUTHENTICATION_FAILED,
            severity=ErrorSeverity.MEDIUM,
            message="Invalid token",
            details="token invalid",
        )

        response = eh.handle_error(error)

        assert response.category == ErrorCategory.AUTHENTICATION_FAILED
        assert response.message == "Invalid token"

    def test_error_statistics(self):
        """Test error statistics collection"""
        eh = ErrorHandler()

        # Generate some errors
        for i in range(5):
            error = ErrorInfo(
                category=ErrorCategory.INTERNAL_ERROR,
                severity=ErrorSeverity.LOW,
                message=f"Error {i}",
            )
            eh.handle_error(error)

        stats = eh.get_error_statistics()
        assert stats["total_errors"] == 5
        assert stats["category_distribution"]["internal_error"] == 5


class TestHeartbeatManager:
    """Unit tests for HeartbeatManager"""

    def test_heartbeat_manager_initialization(self):
        """Test HeartbeatManager initialization"""
        hm = HeartbeatManager(ping_interval=30, pong_timeout=10)
        assert hm.ping_interval == 30
        assert hm.pong_timeout == 10
        assert hm.heartbeats == {}

    def test_add_connection_heartbeat(self):
        """Test adding a connection for heartbeat monitoring"""
        hm = HeartbeatManager(ping_interval=30)

        hm.add_connection("test_sid")
        assert "test_sid" in hm.heartbeats

        heartbeat_info = hm.heartbeats["test_sid"]
        assert isinstance(heartbeat_info, HeartbeatInfo)
        assert heartbeat_info.connection_id == "test_sid"
        assert heartbeat_info.state == HeartbeatState.HEALTHY

    def test_update_heartbeat(self):
        """Test updating heartbeat for a connection"""
        hm = HeartbeatManager(ping_interval=30)

        hm.add_connection("test_sid")
        initial_time = hm.heartbeats["test_sid"].last_heartbeat

        # Wait a bit and handle pong
        time.sleep(0.01)
        hm.handle_pong("test_sid")

        updated_time = hm.heartbeats["test_sid"].last_pong
        assert updated_time >= initial_time
        assert hm.heartbeats["test_sid"].missed_heartbeats == 0

    def test_remove_connection_heartbeat(self):
        """Test removing a connection from heartbeat monitoring"""
        hm = HeartbeatManager(ping_interval=30)

        hm.add_connection("test_sid")
        hm.remove_connection("test_sid")

        assert "test_sid" not in hm.heartbeats


class TestMessageRouter:
    """Unit tests for MessageRouter"""

    def test_message_router_initialization(self):
        """Test MessageRouter initialization"""
        mr = MessageRouter()
        assert isinstance(mr.routes, dict)
        assert isinstance(mr.global_middleware, list)

    def test_register_route(self):
        """Test registering a new route"""
        mr = MessageRouter()

        async def test_handler(data, context):
            return {"status": "success"}

        mr.register_route(MessageType.PING, test_handler)

        assert MessageType.PING in mr.routes
        assert mr.routes[MessageType.PING].handler == test_handler

    def test_route_message(self):
        """Test routing a message"""
        mr = MessageRouter()

        # Handler takes MessageContext as single argument
        def test_handler(context):
            return {"status": "success", "data": context.data.get("value")}

        mr.register_route(MessageType.PING, test_handler)

        result = mr.route_message("test_sid", "ping", {"value": "test_data"})

        assert result is not None
        assert result.get("status") == "success"


class TestReconnectionManager:
    """Unit tests for ReconnectionManager"""

    def test_reconnection_manager_initialization(self):
        """Test ReconnectionManager initialization"""
        rm = ReconnectionManager()
        assert rm.sessions == {}
        assert rm.connection_sessions == {}

    def test_create_session(self):
        """Test creating a reconnection session"""
        rm = ReconnectionManager()

        session_id = rm.create_session("test_sid", "test_user", {"key": "value"})

        assert session_id in rm.sessions
        assert rm.sessions[session_id].original_connection_id == "test_sid"
        assert rm.sessions[session_id].user_id == "test_user"

    def test_get_session(self):
        """Test getting a session"""
        rm = ReconnectionManager()

        session_id = rm.create_session("test_sid", "test_user")

        session = rm.get_session(session_id)
        assert session is not None
        assert session.original_connection_id == "test_sid"

    def test_buffer_message(self):
        """Test buffering messages for disconnected session"""
        rm = ReconnectionManager()

        session_id = rm.create_session("test_sid", "test_user")

        # Handle disconnection first (buffer_message only works for disconnected sessions)
        rm.handle_disconnection("test_sid")

        # Buffer a message for the disconnected session
        result = rm.buffer_message(session_id, "test_type", {"data": "test_data"})

        assert result is True
        session = rm.get_session(session_id)
        assert len(session.buffered_messages) == 1


class TestSimpleAuth:
    """Unit tests for SimpleAuth"""

    def test_simple_auth_initialization(self):
        """Test SimpleAuth initialization"""
        auth = SimpleAuth()
        assert isinstance(auth.tokens, dict)
        assert isinstance(auth.users, dict)
        # Default users should be created
        assert "admin" in auth.users
        assert "user" in auth.users

    def test_create_guest_token(self):
        """Test guest token creation"""
        auth = SimpleAuth()

        token = auth.create_guest_token()

        assert token is not None
        assert token.token in auth.tokens
        assert token.role == UserRole.GUEST

    def test_validate_token(self):
        """Test token validation"""
        auth = SimpleAuth()

        # Create a guest token
        auth_token = auth.create_guest_token()

        # Validate the token
        validated = auth.validate_token(auth_token.token)
        assert validated is not None
        assert validated.user_id == auth_token.user_id

        # Test invalid token
        invalid = auth.validate_token("invalid_token_string")
        assert invalid is None

    def test_revoke_token(self):
        """Test token revocation"""
        auth = SimpleAuth()

        # Create and revoke token
        auth_token = auth.create_guest_token()
        token_str = auth_token.token

        auth.revoke_token(token_str)

        # Token should no longer be valid
        validated = auth.validate_token(token_str)
        assert validated is None
        assert token_str not in auth.tokens


class TestConnectionInfo:
    """Unit tests for ConnectionInfo class"""

    def test_connection_info_initialization(self):
        """Test ConnectionInfo initialization"""
        now = datetime.now()
        conn_info = ConnectionInfo(
            sid="test_sid",
            client_ip="127.0.0.1",
            user_agent="test_agent",
            connected_at=now,
            last_activity=now,
            user_id="test_user",
        )

        assert conn_info.sid == "test_sid"
        assert conn_info.client_ip == "127.0.0.1"
        assert conn_info.user_agent == "test_agent"
        assert conn_info.user_id == "test_user"
        assert conn_info.state == ConnectionState.CONNECTED
        assert conn_info.connected_at == now
        assert conn_info.last_activity == now

    def test_connection_info_update_activity(self):
        """Test activity update in ConnectionInfo"""
        now = datetime.now()
        conn_info = ConnectionInfo(
            sid="test_sid",
            client_ip="127.0.0.1",
            user_agent="test_agent",
            connected_at=now,
            last_activity=now,
        )

        time.sleep(0.01)
        conn_info.update_activity()

        assert conn_info.last_activity > now

    def test_connection_info_expiry(self):
        """Test connection expiry check"""
        now = datetime.now()
        conn_info = ConnectionInfo(
            sid="test_sid",
            client_ip="127.0.0.1",
            user_agent="test_agent",
            connected_at=now,
            last_activity=now,
        )

        # Should not be expired immediately
        assert conn_info.is_expired(timeout_seconds=300) is False

        # Should be expired with very short timeout
        assert conn_info.is_expired(timeout_seconds=0) is True


# Integration tests for component interactions
class TestComponentIntegration:
    """Integration tests for component interactions"""

    def test_connection_manager_error_handler_integration(self):
        """Test integration between ConnectionManager and ErrorHandler"""
        cm = ConnectionManager(max_connections_per_ip=1)
        eh = ErrorHandler()

        # Add first connection (should succeed)
        success1 = cm.add_connection("sid1", "127.0.0.1", "agent1", "user1")
        assert success1 is True

        # Try to add second connection from same IP (should fail due to IP limit)
        success2 = cm.add_connection("sid2", "127.0.0.1", "agent2", "user2")
        assert success2 is False

        # Create and handle error
        error = ErrorInfo(
            category=ErrorCategory.CONNECTION_FAILED,
            severity=ErrorSeverity.MEDIUM,
            message="Connection limit exceeded",
        )
        response = eh.handle_error(error)

        assert response.category == ErrorCategory.CONNECTION_FAILED

    def test_heartbeat_connection_manager_integration(self):
        """Test integration between HeartbeatManager and ConnectionManager"""
        cm = ConnectionManager()
        hm = HeartbeatManager(ping_interval=30)

        # Add connection to both managers
        cm.add_connection("test_sid", "127.0.0.1", "test_agent", "test_user")
        hm.add_connection("test_sid")

        # Verify connection exists in both
        assert cm.get_connection("test_sid") is not None
        assert "test_sid" in hm.heartbeats

        # Remove connection from both
        cm.remove_connection("test_sid")
        hm.remove_connection("test_sid")

        # Verify connection removed from both
        assert cm.get_connection("test_sid") is None
        assert "test_sid" not in hm.heartbeats


# Performance tests for critical components
class TestComponentPerformance:
    """Performance tests for critical components"""

    def test_connection_manager_performance(self):
        """Test ConnectionManager performance with many connections"""
        cm = ConnectionManager(max_connections_per_ip=1000)

        start_time = time.time()

        # Add 100 connections from different IPs
        for i in range(100):
            success = cm.add_connection(f"sid_{i}", f"192.168.1.{i % 256}", "agent", f"user_{i}")
            assert success is True

        add_time = time.time() - start_time

        # Adding 100 connections should be fast
        assert add_time < 1.0  # Less than 1 second

        # Test statistics generation performance
        start_time = time.time()
        stats = cm.get_statistics()
        stats_time = time.time() - start_time

        assert stats["active_connections"] == 100
        assert stats_time < 0.1  # Less than 100ms

    def test_error_handler_performance(self):
        """Test ErrorHandler performance with many errors"""
        eh = ErrorHandler()

        start_time = time.time()

        # Handle 1000 errors
        for i in range(1000):
            error = ErrorInfo(
                category=ErrorCategory.INTERNAL_ERROR,
                severity=ErrorSeverity.LOW,
                message=f"Error {i}",
            )
            response = eh.handle_error(error)
            assert response.category == ErrorCategory.INTERNAL_ERROR

        handle_time = time.time() - start_time

        # Handling 1000 errors should be fast
        assert handle_time < 2.0  # Less than 2 seconds

        # Verify all errors were counted
        stats = eh.get_error_statistics()
        assert stats["total_errors"] == 1000
