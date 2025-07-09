#!/usr/bin/env python3
"""
Unit Tests for WebSocket Server Components

Tests individual components and modules in isolation.
"""

import pytest
import asyncio
import json
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Import the modules we want to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from connection_manager import ConnectionManager, ConnectionInfo, ConnectionState
from error_handler import ErrorHandler, ErrorType, WebSocketError
from heartbeat_manager import HeartbeatManager
from message_router import MessageRouter
from reconnection_manager import ReconnectionManager
from simple_auth import SimpleAuth

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
            sid="test_sid",
            client_ip="127.0.0.1",
            user_agent="test_agent",
            user_id="test_user"
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
    
    def test_connection_limit(self):
        """Test connection limit enforcement"""
        cm = ConnectionManager(max_connections=2)
        
        # Add first connection
        success1 = cm.add_connection("sid1", "127.0.0.1", "agent1", "user1")
        assert success1 is True
        
        # Add second connection
        success2 = cm.add_connection("sid2", "127.0.0.1", "agent2", "user2")
        assert success2 is True
        
        # Try to add third connection (should fail)
        success3 = cm.add_connection("sid3", "127.0.0.1", "agent3", "user3")
        assert success3 is False
    
    def test_ip_rate_limiting(self):
        """Test IP-based rate limiting"""
        cm = ConnectionManager(max_connections_per_ip=1)
        
        # Add first connection from IP
        success1 = cm.add_connection("sid1", "127.0.0.1", "agent1", "user1")
        assert success1 is True
        
        # Try to add second connection from same IP (should fail)
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
        
        # Leave room
        cm.leave_room("test_sid", "test_room")
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
        assert len(eh.error_handlers) > 0
    
    def test_handle_connection_error(self):
        """Test handling connection errors"""
        eh = ErrorHandler()
        
        error = WebSocketError(
            error_type=ErrorType.CONNECTION_ERROR,
            message="Connection failed",
            details={"reason": "timeout"}
        )
        
        response = eh.handle_error(error, {"sid": "test_sid"})
        
        assert response["type"] == "error"
        assert response["error_type"] == "connection_error"
        assert "Connection failed" in response["message"]
        assert eh.error_counts.get("connection_error", 0) > 0
    
    def test_handle_authentication_error(self):
        """Test handling authentication errors"""
        eh = ErrorHandler()
        
        error = WebSocketError(
            error_type=ErrorType.AUTHENTICATION_ERROR,
            message="Invalid token",
            details={"token": "invalid"}
        )
        
        response = eh.handle_error(error, {"sid": "test_sid"})
        
        assert response["type"] == "error"
        assert response["error_type"] == "authentication_error"
        assert "Invalid token" in response["message"]
    
    def test_error_statistics(self):
        """Test error statistics collection"""
        eh = ErrorHandler()
        
        # Generate some errors
        for i in range(5):
            error = WebSocketError(ErrorType.MESSAGE_ERROR, f"Error {i}")
            eh.handle_error(error, {"sid": f"sid_{i}"})
        
        stats = eh.get_statistics()
        assert stats["total_errors"] == 5
        assert stats["error_counts"]["message_error"] == 5

class TestHeartbeatManager:
    """Unit tests for HeartbeatManager"""
    
    @pytest.mark.asyncio
    async def test_heartbeat_manager_initialization(self):
        """Test HeartbeatManager initialization"""
        hm = HeartbeatManager(interval=1.0)
        assert hm.interval == 1.0
        assert hm.connections == {}
        assert not hm.is_running
    
    @pytest.mark.asyncio
    async def test_add_connection_heartbeat(self):
        """Test adding a connection for heartbeat monitoring"""
        hm = HeartbeatManager(interval=1.0)
        
        hm.add_connection("test_sid")
        assert "test_sid" in hm.connections
        
        connection_info = hm.connections["test_sid"]
        assert connection_info["last_heartbeat"] is not None
        assert connection_info["missed_heartbeats"] == 0
    
    @pytest.mark.asyncio
    async def test_update_heartbeat(self):
        """Test updating heartbeat for a connection"""
        hm = HeartbeatManager(interval=1.0)
        
        hm.add_connection("test_sid")
        initial_time = hm.connections["test_sid"]["last_heartbeat"]
        
        # Wait a bit and update
        await asyncio.sleep(0.1)
        hm.update_heartbeat("test_sid")
        
        updated_time = hm.connections["test_sid"]["last_heartbeat"]
        assert updated_time > initial_time
        assert hm.connections["test_sid"]["missed_heartbeats"] == 0
    
    @pytest.mark.asyncio
    async def test_remove_connection_heartbeat(self):
        """Test removing a connection from heartbeat monitoring"""
        hm = HeartbeatManager(interval=1.0)
        
        hm.add_connection("test_sid")
        hm.remove_connection("test_sid")
        
        assert "test_sid" not in hm.connections

class TestMessageRouter:
    """Unit tests for MessageRouter"""
    
    def test_message_router_initialization(self):
        """Test MessageRouter initialization"""
        mr = MessageRouter()
        assert len(mr.routes) > 0
        assert len(mr.middleware) >= 0
    
    def test_register_route(self):
        """Test registering a new route"""
        mr = MessageRouter()
        
        @mr.route("test_message")
        async def test_handler(message, context):
            return {"status": "success"}
        
        assert "test_message" in mr.routes
    
    @pytest.mark.asyncio
    async def test_route_message(self):
        """Test routing a message"""
        mr = MessageRouter()
        
        @mr.route("test_message")
        async def test_handler(message, context):
            return {"status": "success", "data": message.get("data")}
        
        message = {"type": "test_message", "data": "test_data"}
        context = {"sid": "test_sid"}
        
        response = await mr.route_message(message, context)
        
        assert response["status"] == "success"
        assert response["data"] == "test_data"
    
    @pytest.mark.asyncio
    async def test_unknown_message_type(self):
        """Test handling unknown message types"""
        mr = MessageRouter()
        
        message = {"type": "unknown_message"}
        context = {"sid": "test_sid"}
        
        response = await mr.route_message(message, context)
        
        assert response["type"] == "error"
        assert "unknown_message_type" in response["error_type"]

class TestReconnectionManager:
    """Unit tests for ReconnectionManager"""
    
    def test_reconnection_manager_initialization(self):
        """Test ReconnectionManager initialization"""
        rm = ReconnectionManager()
        assert rm.sessions == {}
        assert rm.message_buffers == {}
    
    def test_create_session(self):
        """Test creating a reconnection session"""
        rm = ReconnectionManager()
        
        session_id = rm.create_session("test_sid", {"user_id": "test_user"})
        
        assert session_id in rm.sessions
        assert rm.sessions[session_id]["sid"] == "test_sid"
        assert rm.sessions[session_id]["metadata"]["user_id"] == "test_user"
    
    def test_restore_session(self):
        """Test restoring a session"""
        rm = ReconnectionManager()
        
        # Create session
        session_id = rm.create_session("old_sid", {"user_id": "test_user"})
        
        # Restore session with new sid
        restored = rm.restore_session(session_id, "new_sid")
        
        assert restored is True
        assert rm.sessions[session_id]["sid"] == "new_sid"
    
    def test_buffer_message(self):
        """Test buffering messages for disconnected session"""
        rm = ReconnectionManager()
        
        session_id = rm.create_session("test_sid", {"user_id": "test_user"})
        
        # Buffer a message
        message = {"type": "test", "data": "test_data"}
        rm.buffer_message(session_id, message)
        
        assert session_id in rm.message_buffers
        assert len(rm.message_buffers[session_id]) == 1
        assert rm.message_buffers[session_id][0]["message"] == message
    
    def test_get_buffered_messages(self):
        """Test retrieving buffered messages"""
        rm = ReconnectionManager()
        
        session_id = rm.create_session("test_sid", {"user_id": "test_user"})
        
        # Buffer some messages
        for i in range(3):
            message = {"type": "test", "data": f"test_data_{i}"}
            rm.buffer_message(session_id, message)
        
        # Get buffered messages
        messages = rm.get_buffered_messages(session_id)
        
        assert len(messages) == 3
        assert messages[0]["message"]["data"] == "test_data_0"
        assert messages[2]["message"]["data"] == "test_data_2"
    
    def test_session_expiry(self):
        """Test session expiry"""
        rm = ReconnectionManager(session_timeout=0.1)  # 0.1 second timeout
        
        session_id = rm.create_session("test_sid", {"user_id": "test_user"})
        
        # Wait for session to expire
        time.sleep(0.2)
        
        # Clean up expired sessions
        rm.cleanup_expired_sessions()
        
        assert session_id not in rm.sessions

class TestSimpleAuth:
    """Unit tests for SimpleAuth"""
    
    def test_simple_auth_initialization(self):
        """Test SimpleAuth initialization"""
        auth = SimpleAuth()
        assert auth.valid_tokens == set()
    
    def test_generate_token(self):
        """Test token generation"""
        auth = SimpleAuth()
        
        token = auth.generate_token("test_user")
        
        assert token is not None
        assert len(token) > 0
        assert token in auth.valid_tokens
    
    def test_validate_token(self):
        """Test token validation"""
        auth = SimpleAuth()
        
        # Generate a valid token
        token = auth.generate_token("test_user")
        
        # Validate the token
        user_id = auth.validate_token(token)
        assert user_id == "test_user"
        
        # Test invalid token
        invalid_user = auth.validate_token("invalid_token")
        assert invalid_user is None
    
    def test_revoke_token(self):
        """Test token revocation"""
        auth = SimpleAuth()
        
        # Generate and revoke token
        token = auth.generate_token("test_user")
        auth.revoke_token(token)
        
        # Token should no longer be valid
        user_id = auth.validate_token(token)
        assert user_id is None
        assert token not in auth.valid_tokens

class TestConnectionInfo:
    """Unit tests for ConnectionInfo class"""
    
    def test_connection_info_initialization(self):
        """Test ConnectionInfo initialization"""
        conn_info = ConnectionInfo(
            sid="test_sid",
            client_ip="127.0.0.1",
            user_agent="test_agent",
            user_id="test_user"
        )
        
        assert conn_info.sid == "test_sid"
        assert conn_info.client_ip == "127.0.0.1"
        assert conn_info.user_agent == "test_agent"
        assert conn_info.user_id == "test_user"
        assert conn_info.state == ConnectionState.CONNECTING
        assert conn_info.connected_at is not None
        assert conn_info.rooms == set()
    
    def test_connection_info_room_management(self):
        """Test room management in ConnectionInfo"""
        conn_info = ConnectionInfo("test_sid", "127.0.0.1", "test_agent", "test_user")
        
        # Add rooms
        conn_info.rooms.add("room1")
        conn_info.rooms.add("room2")
        
        assert "room1" in conn_info.rooms
        assert "room2" in conn_info.rooms
        assert len(conn_info.rooms) == 2
        
        # Remove room
        conn_info.rooms.remove("room1")
        assert "room1" not in conn_info.rooms
        assert len(conn_info.rooms) == 1

class TestWebSocketError:
    """Unit tests for WebSocketError class"""
    
    def test_websocket_error_initialization(self):
        """Test WebSocketError initialization"""
        error = WebSocketError(
            error_type=ErrorType.CONNECTION_ERROR,
            message="Test error",
            details={"key": "value"}
        )
        
        assert error.error_type == ErrorType.CONNECTION_ERROR
        assert error.message == "Test error"
        assert error.details == {"key": "value"}
        assert error.timestamp is not None
    
    def test_websocket_error_str_representation(self):
        """Test string representation of WebSocketError"""
        error = WebSocketError(
            error_type=ErrorType.AUTHENTICATION_ERROR,
            message="Auth failed"
        )
        
        error_str = str(error)
        assert "authentication_error" in error_str
        assert "Auth failed" in error_str

# Integration tests for component interactions
class TestComponentIntegration:
    """Integration tests for component interactions"""
    
    def test_connection_manager_error_handler_integration(self):
        """Test integration between ConnectionManager and ErrorHandler"""
        cm = ConnectionManager()
        eh = ErrorHandler()
        
        # Try to add connection that exceeds limit
        cm.max_connections = 1
        
        # Add first connection (should succeed)
        success1 = cm.add_connection("sid1", "127.0.0.1", "agent1", "user1")
        assert success1 is True
        
        # Try to add second connection (should fail and generate error)
        success2 = cm.add_connection("sid2", "127.0.0.1", "agent2", "user2")
        assert success2 is False
        
        # Create and handle error
        error = WebSocketError(
            error_type=ErrorType.CONNECTION_ERROR,
            message="Connection limit exceeded"
        )
        response = eh.handle_error(error, {"sid": "sid2"})
        
        assert response["type"] == "error"
        assert "connection_error" in response["error_type"]
    
    @pytest.mark.asyncio
    async def test_heartbeat_connection_manager_integration(self):
        """Test integration between HeartbeatManager and ConnectionManager"""
        cm = ConnectionManager()
        hm = HeartbeatManager(interval=0.1)
        
        # Add connection to both managers
        cm.add_connection("test_sid", "127.0.0.1", "test_agent", "test_user")
        hm.add_connection("test_sid")
        
        # Verify connection exists in both
        assert cm.get_connection("test_sid") is not None
        assert "test_sid" in hm.connections
        
        # Remove connection from connection manager
        cm.remove_connection("test_sid")
        hm.remove_connection("test_sid")
        
        # Verify connection removed from both
        assert cm.get_connection("test_sid") is None
        assert "test_sid" not in hm.connections

# Performance tests for critical components
class TestComponentPerformance:
    """Performance tests for critical components"""
    
    def test_connection_manager_performance(self):
        """Test ConnectionManager performance with many connections"""
        cm = ConnectionManager(max_connections=1000)
        
        start_time = time.time()
        
        # Add 100 connections
        for i in range(100):
            success = cm.add_connection(f"sid_{i}", "127.0.0.1", "agent", f"user_{i}")
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
            error = WebSocketError(
                error_type=ErrorType.MESSAGE_ERROR,
                message=f"Error {i}"
            )
            response = eh.handle_error(error, {"sid": f"sid_{i}"})
            assert response["type"] == "error"
        
        handle_time = time.time() - start_time
        
        # Handling 1000 errors should be fast
        assert handle_time < 2.0  # Less than 2 seconds
        
        # Verify all errors were counted
        stats = eh.get_statistics()
        assert stats["total_errors"] == 1000 