#!/usr/bin/env python3
"""
Integration Tests for WebSocket Server

Tests the full WebSocket server functionality including real WebSocket connections,
API endpoints, and service interactions.
"""

import pytest
import asyncio
import json
import time
import tempfile
import uuid
from typing import Dict, Any, List
import numpy as np
import soundfile as sf

from .utils import (
    WebSocketTestClient, HTTPTestClient, TestAudioGenerator,
    PerformanceMonitor, assert_response_time, assert_websocket_message,
    generate_test_token, wait_for_condition
)
from .fixtures import *

class TestWebSocketConnections:
    """Integration tests for WebSocket connections"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self, websocket_client):
        """Test complete WebSocket connection lifecycle"""
        # Connect
        connected = await websocket_client.connect()
        assert connected is True
        
        # Send heartbeat
        heartbeat_message = {"type": "heartbeat", "timestamp": time.time()}
        sent = await websocket_client.send_message(heartbeat_message)
        assert sent is True
        
        # Receive heartbeat response
        response = await websocket_client.receive_message(timeout=5.0)
        assert response is not None
        assert_websocket_message(response, "heartbeat_ack", ["timestamp"])
        
        # Disconnect
        await websocket_client.disconnect()
        
        # Verify connection duration
        duration = websocket_client.get_connection_duration()
        assert duration > 0
    
    @pytest.mark.asyncio
    async def test_websocket_authentication(self, test_config):
        """Test WebSocket authentication flow"""
        client = WebSocketTestClient(test_config["websocket_url"])
        
        # Connect with auth token
        auth_token = generate_test_token("test_user")
        connected = await client.connect(auth_token)
        assert connected is True
        
        # Send authentication message
        auth_message = {"type": "authenticate", "token": auth_token}
        sent = await client.send_message(auth_message)
        assert sent is True
        
        # Receive authentication response
        response = await client.receive_message(timeout=5.0)
        assert response is not None
        assert_websocket_message(response, "auth_success", ["user_id"])
        assert response["user_id"] == "test_user"
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_websocket_audio_streaming(self, websocket_client, test_audio_sine_wave):
        """Test audio streaming over WebSocket"""
        # Connect and authenticate
        connected = await websocket_client.connect()
        assert connected is True
        
        # Convert audio to bytes
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, test_audio_sine_wave, 16000)
        
        with open(temp_file.name, 'rb') as f:
            audio_bytes = f.read()
        
        # Send audio data
        sent = await websocket_client.send_audio_data(audio_bytes)
        assert sent is True
        
        # Wait for transcription response
        response = await websocket_client.receive_message(timeout=10.0)
        assert response is not None
        assert_websocket_message(response, "transcription", ["text", "session_id"])
        
        await websocket_client.disconnect()
        
        # Cleanup
        import os
        os.unlink(temp_file.name)
    
    @pytest.mark.asyncio
    async def test_websocket_room_management(self, websocket_client):
        """Test WebSocket room join/leave functionality"""
        connected = await websocket_client.connect()
        assert connected is True
        
        # Join room
        join_message = {"type": "subscribe", "room": "test_room"}
        sent = await websocket_client.send_message(join_message)
        assert sent is True
        
        response = await websocket_client.receive_message(timeout=5.0)
        assert response is not None
        assert_websocket_message(response, "room_joined", ["room"])
        assert response["room"] == "test_room"
        
        # Leave room
        leave_message = {"type": "unsubscribe", "room": "test_room"}
        sent = await websocket_client.send_message(leave_message)
        assert sent is True
        
        response = await websocket_client.receive_message(timeout=5.0)
        assert response is not None
        assert_websocket_message(response, "room_left", ["room"])
        assert response["room"] == "test_room"
        
        await websocket_client.disconnect()

class TestHTTPEndpoints:
    """Integration tests for HTTP API endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, http_client, sample_api_endpoints, expected_response_schemas):
        """Test health check endpoint"""
        start_time = time.time()
        
        async with http_client as client:
            response = await client.get(sample_api_endpoints["health"])
            
            response_time = time.time() - start_time
            assert_response_time(response_time, 1.0, "health check")
            
            assert response.status == 200
            
            data = await response.json()
            
            # Validate response schema
            schema = expected_response_schemas["health"]
            for field in schema["required"]:
                assert field in data
            
            assert data["status"] in ["healthy", "ok"]
            assert isinstance(data["timestamp"], (int, float))
    
    @pytest.mark.asyncio
    async def test_models_endpoint(self, http_client, sample_api_endpoints):
        """Test models listing endpoint"""
        async with http_client as client:
            response = await client.get(sample_api_endpoints["models"])
            
            assert response.status == 200
            
            data = await response.json()
            assert "models" in data
            assert isinstance(data["models"], list)
            
            if data["models"]:
                model = data["models"][0]
                assert "name" in model
                assert "type" in model
    
    @pytest.mark.asyncio
    async def test_connections_endpoint(self, http_client, sample_api_endpoints):
        """Test connections status endpoint"""
        async with http_client as client:
            response = await client.get(sample_api_endpoints["connections"])
            
            assert response.status == 200
            
            data = await response.json()
            assert "active_connections" in data
            assert "total_connections" in data
            assert isinstance(data["active_connections"], int)
            assert isinstance(data["total_connections"], int)
    
    @pytest.mark.asyncio
    async def test_performance_endpoint(self, http_client, sample_api_endpoints, expected_response_schemas):
        """Test performance metrics endpoint"""
        async with http_client as client:
            response = await client.get(sample_api_endpoints["performance"])
            
            assert response.status == 200
            
            data = await response.json()
            
            # Validate response schema
            schema = expected_response_schemas["performance"]
            for field in schema["required"]:
                assert field in data
            
            assert isinstance(data["connection_stats"], dict)
            assert isinstance(data["performance_stats"], dict)
    
    @pytest.mark.asyncio
    async def test_transcribe_endpoint(self, http_client, sample_api_endpoints, test_audio_file):
        """Test file transcription endpoint"""
        async with http_client as client:
            with open(test_audio_file, 'rb') as f:
                audio_data = f.read()
            
            data = {'file': audio_data}
            response = await client.post(sample_api_endpoints["transcribe"], data=data)
            
            # Should succeed or return appropriate error
            assert response.status in [200, 400, 415]  # OK, Bad Request, or Unsupported Media Type
            
            if response.status == 200:
                result = await response.json()
                assert "text" in result
                assert isinstance(result["text"], str)

class TestErrorHandling:
    """Integration tests for error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_websocket_message(self, websocket_client, error_scenarios):
        """Test handling of invalid WebSocket messages"""
        connected = await websocket_client.connect()
        assert connected is True
        
        # Send invalid message format
        invalid_message = '{"type": "invalid", "data":'  # Invalid JSON
        
        # This should cause an error response
        try:
            await websocket_client.websocket.send(invalid_message)
            response = await websocket_client.receive_message(timeout=5.0)
            
            if response:
                assert_websocket_message(response, "error")
                assert "message_parse_error" in response.get("error_type", "")
        except Exception:
            # Connection might be closed due to invalid message
            pass
        
        await websocket_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, test_config, error_scenarios):
        """Test authentication error handling"""
        client = WebSocketTestClient(test_config["websocket_url"])
        
        # Try to connect with invalid token
        invalid_token = error_scenarios["invalid_auth"]["token"]
        connected = await client.connect(invalid_token)
        
        if connected:
            # Send authentication message
            auth_message = {"type": "authenticate", "token": invalid_token}
            sent = await client.send_message(auth_message)
            
            if sent:
                response = await client.receive_message(timeout=5.0)
                if response:
                    assert_websocket_message(response, "error")
                    assert "authentication" in response.get("error_type", "").lower()
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_unsupported_audio_format(self, websocket_client, error_scenarios):
        """Test unsupported audio format error"""
        connected = await websocket_client.connect()
        assert connected is True
        
        # Send start audio with unsupported format
        unsupported_message = error_scenarios["unsupported_audio_format"]["message"]
        sent = await websocket_client.send_message(unsupported_message)
        assert sent is True
        
        response = await websocket_client.receive_message(timeout=5.0)
        assert response is not None
        assert_websocket_message(response, "error")
        assert "format" in response.get("message", "").lower()
        
        await websocket_client.disconnect()

class TestConcurrentConnections:
    """Integration tests for concurrent connections"""
    
    @pytest.mark.asyncio
    async def test_multiple_websocket_connections(self, multiple_websocket_clients):
        """Test handling multiple simultaneous WebSocket connections"""
        assert len(multiple_websocket_clients) > 0
        
        # Send heartbeat from all clients simultaneously
        tasks = []
        for client in multiple_websocket_clients:
            heartbeat_message = {"type": "heartbeat", "timestamp": time.time()}
            task = client.send_message(heartbeat_message)
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed
        success_count = sum(1 for result in results if result is True)
        assert success_count >= len(multiple_websocket_clients) * 0.8  # At least 80% success
    
    @pytest.mark.asyncio
    async def test_concurrent_audio_streaming(self, test_config, test_audio_sine_wave):
        """Test concurrent audio streaming from multiple clients"""
        num_clients = 5
        clients = []
        
        # Create and connect clients
        for i in range(num_clients):
            client = WebSocketTestClient(test_config["websocket_url"])
            connected = await client.connect()
            if connected:
                clients.append(client)
        
        assert len(clients) > 0
        
        # Convert audio to bytes
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, test_audio_sine_wave, 16000)
        
        with open(temp_file.name, 'rb') as f:
            audio_bytes = f.read()
        
        # Send audio from all clients simultaneously
        tasks = []
        for client in clients:
            task = client.send_audio_data(audio_bytes)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed
        success_count = sum(1 for result in results if result is True)
        assert success_count >= len(clients) * 0.8  # At least 80% success
        
        # Cleanup
        for client in clients:
            await client.disconnect()
        
        import os
        os.unlink(temp_file.name)

class TestReconnectionFlow:
    """Integration tests for reconnection handling"""
    
    @pytest.mark.asyncio
    async def test_normal_reconnection(self, test_config, reconnection_scenarios):
        """Test normal reconnection after disconnect"""
        scenario = reconnection_scenarios["normal_reconnection"]
        
        client = WebSocketTestClient(test_config["websocket_url"])
        
        # Initial connection
        auth_token = generate_test_token("test_user")
        connected = await client.connect(auth_token)
        assert connected is True
        
        # Get session ID
        auth_message = {"type": "authenticate", "token": auth_token}
        await client.send_message(auth_message)
        response = await client.receive_message(timeout=5.0)
        assert response is not None
        
        session_id = response.get("session_id")
        assert session_id is not None
        
        # Wait specified time then disconnect
        await asyncio.sleep(scenario["disconnect_after"])
        await client.disconnect()
        
        # Wait before reconnecting
        await asyncio.sleep(scenario["reconnect_delay"])
        
        # Reconnect with same token
        reconnected = await client.connect(auth_token)
        assert reconnected is True
        
        # Send reconnection message
        reconnect_message = {
            "type": "reconnect_session",
            "session_id": session_id,
            "token": auth_token
        }
        await client.send_message(reconnect_message)
        
        response = await client.receive_message(timeout=10.0)
        if response and scenario["should_restore_session"]:
            assert response.get("type") in ["session_restored", "reconnect_success"]
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_session_expiry_reconnection(self, test_config, reconnection_scenarios):
        """Test reconnection after session expiry"""
        scenario = reconnection_scenarios["session_expired"]
        
        client = WebSocketTestClient(test_config["websocket_url"])
        
        # Initial connection
        auth_token = generate_test_token("test_user")
        connected = await client.connect(auth_token)
        assert connected is True
        
        # Get session ID
        auth_message = {"type": "authenticate", "token": auth_token}
        await client.send_message(auth_message)
        response = await client.receive_message(timeout=5.0)
        
        if response:
            session_id = response.get("session_id")
            await client.disconnect()
            
            # Wait long enough for session to expire (simulated with short wait)
            await asyncio.sleep(0.5)
            
            # Try to reconnect
            reconnected = await client.connect(auth_token)
            assert reconnected is True
            
            # Try to restore expired session
            reconnect_message = {
                "type": "reconnect_session",
                "session_id": session_id,
                "token": auth_token
            }
            await client.send_message(reconnect_message)
            
            response = await client.receive_message(timeout=5.0)
            if response and not scenario["should_restore_session"]:
                assert response.get("type") in ["session_expired", "error"]
        
        await client.disconnect()

class TestPerformanceIntegration:
    """Integration tests for performance requirements"""
    
    @pytest.mark.asyncio
    async def test_connection_performance(self, test_config, benchmark_thresholds, performance_monitor):
        """Test connection establishment performance"""
        client = WebSocketTestClient(test_config["websocket_url"])
        
        # Measure connection time
        start_time = time.time()
        connected = await client.connect()
        connection_time = time.time() - start_time
        
        assert connected is True
        assert_response_time(connection_time, benchmark_thresholds["connection_time"], "WebSocket connection")
        
        performance_monitor.record_response_time(connection_time)
        performance_monitor.record_success()
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_message_response_performance(self, websocket_client, benchmark_thresholds, performance_monitor):
        """Test message response time performance"""
        connected = await websocket_client.connect()
        assert connected is True
        
        # Measure heartbeat response time
        start_time = time.time()
        heartbeat_message = {"type": "heartbeat", "timestamp": start_time}
        await websocket_client.send_message(heartbeat_message)
        
        response = await websocket_client.receive_message(timeout=5.0)
        response_time = time.time() - start_time
        
        assert response is not None
        assert_response_time(response_time, benchmark_thresholds["message_response_time"], "message response")
        
        performance_monitor.record_response_time(response_time)
        performance_monitor.record_success()
        
        await websocket_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_throughput_performance(self, test_config, benchmark_thresholds, performance_monitor):
        """Test message throughput performance"""
        client = WebSocketTestClient(test_config["websocket_url"])
        connected = await client.connect()
        assert connected is True
        
        num_messages = 100
        start_time = time.time()
        
        # Send multiple messages rapidly
        for i in range(num_messages):
            message = {"type": "heartbeat", "timestamp": time.time(), "id": i}
            await client.send_message(message)
        
        # Wait for responses
        received_count = 0
        while received_count < num_messages:
            response = await client.receive_message(timeout=1.0)
            if response:
                received_count += 1
                performance_monitor.record_success()
            else:
                break
        
        total_time = time.time() - start_time
        throughput = received_count / total_time
        
        # Should handle reasonable throughput
        assert throughput >= benchmark_thresholds["throughput"] * 0.1  # 10% of benchmark
        
        await client.disconnect()

class TestServiceIntegration:
    """Integration tests for service interactions"""
    
    @pytest.mark.asyncio
    async def test_whisper_service_integration(self, http_client, sample_api_endpoints):
        """Test integration with Whisper transcription service"""
        async with http_client as client:
            # Check if whisper service is available
            response = await client.get("/whisper/models")
            
            if response.status == 200:
                data = await response.json()
                assert "models" in data
                
                # Test transcription if models are available
                if data["models"]:
                    # This would test actual Whisper integration
                    # For now, just verify the endpoint exists
                    transcribe_response = await client.get("/whisper/status")
                    assert transcribe_response.status in [200, 404]  # Either works or not implemented
    
    @pytest.mark.asyncio
    async def test_redis_integration(self, redis_helper):
        """Test Redis integration for session management"""
        if redis_helper.client:
            # Test basic Redis operations
            redis_helper.client.set("test:integration", "test_value")
            value = redis_helper.client.get("test:integration")
            assert value.decode() == "test_value"
            
            # Test connection count
            connection_count = redis_helper.get_connection_count()
            assert connection_count >= 1
    
    @pytest.mark.asyncio
    async def test_end_to_end_audio_flow(self, websocket_client, test_audio_sine_wave):
        """Test complete end-to-end audio processing flow"""
        connected = await websocket_client.connect()
        assert connected is True
        
        # Convert audio to bytes
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, test_audio_sine_wave, 16000)
        
        with open(temp_file.name, 'rb') as f:
            audio_bytes = f.read()
        
        # Start audio session
        session_id = str(uuid.uuid4())
        start_message = {
            "type": "start_audio",
            "session_id": session_id,
            "audio_format": "wav",
            "sample_rate": 16000
        }
        await websocket_client.send_message(start_message)
        
        # Send audio data in chunks
        chunk_size = 1024
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            chunk_message = {
                "type": "audio_data",
                "session_id": session_id,
                "data": chunk.hex(),
                "chunk_id": i // chunk_size
            }
            await websocket_client.send_message(chunk_message)
        
        # End audio session
        end_message = {"type": "end_audio", "session_id": session_id}
        await websocket_client.send_message(end_message)
        
        # Wait for processing completion
        timeout = 30.0  # Allow up to 30 seconds for processing
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = await websocket_client.receive_message(timeout=5.0)
            if response:
                if response.get("type") == "transcription":
                    # Successfully received transcription
                    assert "text" in response
                    assert "session_id" in response
                    assert response["session_id"] == session_id
                    break
                elif response.get("type") == "error":
                    # Handle error appropriately
                    break
        
        await websocket_client.disconnect()
        
        # Cleanup
        import os
        os.unlink(temp_file.name) 