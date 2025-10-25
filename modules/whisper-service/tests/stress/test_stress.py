#!/usr/bin/env python3
"""
Stress Tests for WebSocket Server

Tests the WebSocket server under high load and stress conditions.
"""

import pytest
import asyncio
import time
import uuid
import random
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .utils import (
    WebSocketTestClient, HTTPTestClient, TestAudioGenerator,
    PerformanceMonitor, assert_response_time, generate_test_token
)
from .fixtures import *

class TestConnectionStress:
    """Stress tests for connection handling"""
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_high_connection_count(self, test_config, stress_test_config, performance_monitor):
        """Test handling many simultaneous connections"""
        max_connections = min(stress_test_config["max_connections"], 50)  # Limit for CI
        clients = []
        
        try:
            # Ramp up connections
            ramp_up_interval = stress_test_config["ramp_up_time"] / max_connections
            
            for i in range(max_connections):
                client = WebSocketTestClient(test_config["websocket_url"], timeout=10)
                start_time = time.time()
                
                connected = await client.connect()
                connection_time = time.time() - start_time
                
                if connected:
                    clients.append(client)
                    performance_monitor.record_response_time(connection_time)
                    performance_monitor.record_success()
                else:
                    performance_monitor.record_error()
                
                # Small delay to ramp up gradually
                await asyncio.sleep(ramp_up_interval)
            
            # Verify we achieved reasonable connection success rate
            success_rate = len(clients) / max_connections
            assert success_rate >= 0.8, f"Only {success_rate:.1%} connections succeeded"
            
            # Hold connections for test duration
            await asyncio.sleep(5)  # Hold for 5 seconds
            
            # Test that all connections are still active
            active_count = 0
            for client in clients:
                heartbeat_message = {"type": "heartbeat", "timestamp": time.time()}
                if await client.send_message(heartbeat_message):
                    response = await client.receive_message(timeout=2.0)
                    if response:
                        active_count += 1
            
            active_rate = active_count / len(clients) if clients else 0
            assert active_rate >= 0.9, f"Only {active_rate:.1%} connections remained active"
            
        finally:
            # Cleanup all connections
            cleanup_tasks = [client.disconnect() for client in clients]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_rapid_connect_disconnect(self, test_config, stress_test_config, performance_monitor):
        """Test rapid connection and disconnection cycles"""
        cycles = 20  # Reduced for CI
        
        for i in range(cycles):
            client = WebSocketTestClient(test_config["websocket_url"], timeout=5)
            
            # Connect
            start_time = time.time()
            connected = await client.connect()
            connect_time = time.time() - start_time
            
            if connected:
                performance_monitor.record_response_time(connect_time)
                performance_monitor.record_success()
                
                # Send a quick message
                message = {"type": "heartbeat", "timestamp": time.time()}
                await client.send_message(message)
                
                # Disconnect immediately
                await client.disconnect()
            else:
                performance_monitor.record_error()
            
            # Small delay between cycles
            await asyncio.sleep(0.1)
        
        # Verify performance
        stats = performance_monitor.get_statistics()
        assert stats["success_rate"] >= 0.9, f"Success rate too low: {stats['success_rate']:.1%}"
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_connection_timeout_handling(self, test_config):
        """Test handling of connection timeouts and failures"""
        # Try to connect to non-existent endpoints to test timeout handling
        invalid_urls = [
            "ws://localhost:9999",  # Non-existent port
            "ws://invalid-host:5001",  # Invalid host
        ]
        
        for url in invalid_urls:
            client = WebSocketTestClient(url, timeout=2)  # Short timeout
            
            start_time = time.time()
            connected = await client.connect()
            connect_time = time.time() - start_time
            
            # Should fail quickly and not hang
            assert connected is False
            assert connect_time < 5.0, f"Connection attempt took too long: {connect_time:.2f}s"

class TestMessageStress:
    """Stress tests for message handling"""
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_high_message_throughput(self, websocket_client, stress_test_config, performance_monitor):
        """Test handling high message throughput"""
        connected = await websocket_client.connect()
        assert connected is True
        
        num_messages = stress_test_config["messages_per_connection"]
        messages_sent = 0
        messages_received = 0
        
        start_time = time.time()
        
        # Send messages rapidly
        send_tasks = []
        for i in range(num_messages):
            message = {
                "type": "heartbeat",
                "timestamp": time.time(),
                "id": i,
                "data": f"test_data_{i}"
            }
            task = websocket_client.send_message(message)
            send_tasks.append(task)
            messages_sent += 1
        
        # Wait for all sends to complete
        send_results = await asyncio.gather(*send_tasks, return_exceptions=True)
        successful_sends = sum(1 for result in send_results if result is True)
        
        # Receive responses
        received_messages = []
        while len(received_messages) < successful_sends:
            response = await websocket_client.receive_message(timeout=1.0)
            if response:
                received_messages.append(response)
                performance_monitor.record_success()
            else:
                break
        
        total_time = time.time() - start_time
        throughput = len(received_messages) / total_time
        
        # Verify throughput
        assert throughput >= 10, f"Throughput too low: {throughput:.1f} msg/sec"
        
        # Verify message integrity
        response_rate = len(received_messages) / successful_sends
        assert response_rate >= 0.9, f"Response rate too low: {response_rate:.1%}"
        
        await websocket_client.disconnect()
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_large_message_handling(self, websocket_client):
        """Test handling of large messages"""
        connected = await websocket_client.connect()
        assert connected is True
        
        # Test progressively larger messages
        sizes = [1024, 10240, 102400, 1048576]  # 1KB to 1MB
        
        for size in sizes:
            large_data = "x" * size
            message = {
                "type": "test_large",
                "data": large_data,
                "size": size
            }
            
            start_time = time.time()
            sent = await websocket_client.send_message(message)
            send_time = time.time() - start_time
            
            if sent:
                response = await websocket_client.receive_message(timeout=10.0)
                response_time = time.time() - start_time
                
                # Should handle reasonably sized messages
                if size <= 102400:  # Up to 100KB should work
                    assert response is not None, f"No response for {size} byte message"
                    assert_response_time(response_time, 5.0, f"{size} byte message")
            
        await websocket_client.disconnect()
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_concurrent_message_streams(self, test_config, stress_test_config):
        """Test multiple concurrent message streams"""
        num_clients = 5
        messages_per_client = 50
        
        clients = []
        
        # Create clients
        for i in range(num_clients):
            client = WebSocketTestClient(test_config["websocket_url"])
            connected = await client.connect()
            if connected:
                clients.append(client)
        
        assert len(clients) >= 3, "Need at least 3 clients for concurrent test"
        
        async def send_message_stream(client, client_id):
            """Send a stream of messages from one client"""
            sent_count = 0
            received_count = 0
            
            for i in range(messages_per_client):
                message = {
                    "type": "heartbeat",
                    "client_id": client_id,
                    "message_id": i,
                    "timestamp": time.time()
                }
                
                sent = await client.send_message(message)
                if sent:
                    sent_count += 1
                
                # Brief delay to avoid overwhelming
                await asyncio.sleep(0.01)
            
            # Collect responses
            timeout_time = time.time() + 10  # 10 second timeout
            while received_count < sent_count and time.time() < timeout_time:
                response = await client.receive_message(timeout=1.0)
                if response:
                    received_count += 1
            
            return sent_count, received_count
        
        # Run concurrent streams
        tasks = [
            send_message_stream(client, i)
            for i, client in enumerate(clients)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        total_sent = 0
        total_received = 0
        for result in results:
            if isinstance(result, tuple):
                sent, received = result
                total_sent += sent
                total_received += received
        
        success_rate = total_received / total_sent if total_sent > 0 else 0
        assert success_rate >= 0.8, f"Concurrent message success rate too low: {success_rate:.1%}"
        
        # Cleanup
        for client in clients:
            await client.disconnect()

class TestAudioStress:
    """Stress tests for audio processing"""
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_concurrent_audio_streams(self, test_config, test_audio_sine_wave):
        """Test multiple concurrent audio streams"""
        num_clients = 3  # Reduced for CI
        clients = []
        
        # Create clients
        for i in range(num_clients):
            client = WebSocketTestClient(test_config["websocket_url"])
            connected = await client.connect()
            if connected:
                clients.append(client)
        
        assert len(clients) >= 2, "Need at least 2 clients for concurrent audio test"
        
        # Convert audio to bytes
        import tempfile
        import soundfile as sf
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, test_audio_sine_wave, 16000)
        
        with open(temp_file.name, 'rb') as f:
            audio_bytes = f.read()
        
        async def send_audio_stream(client, client_id):
            """Send audio stream from one client"""
            session_id = f"stress_session_{client_id}_{uuid.uuid4()}"
            
            try:
                # Start audio session
                start_message = {
                    "type": "start_audio",
                    "session_id": session_id,
                    "audio_format": "wav",
                    "sample_rate": 16000
                }
                await client.send_message(start_message)
                
                # Send audio in chunks
                chunk_size = 1024
                chunks_sent = 0
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    chunk_message = {
                        "type": "audio_data",
                        "session_id": session_id,
                        "data": chunk.hex(),
                        "chunk_id": chunks_sent
                    }
                    
                    sent = await client.send_message(chunk_message)
                    if sent:
                        chunks_sent += 1
                    
                    # Small delay to avoid overwhelming
                    await asyncio.sleep(0.01)
                
                # End audio session
                end_message = {"type": "end_audio", "session_id": session_id}
                await client.send_message(end_message)
                
                return chunks_sent
                
            except Exception as e:
                return 0
        
        # Run concurrent audio streams
        start_time = time.time()
        tasks = [
            send_audio_stream(client, i)
            for i, client in enumerate(clients)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_streams = sum(1 for result in results if isinstance(result, int) and result > 0)
        success_rate = successful_streams / len(clients)
        
        assert success_rate >= 0.7, f"Concurrent audio success rate too low: {success_rate:.1%}"
        assert total_time < 60, f"Concurrent audio processing took too long: {total_time:.1f}s"
        
        # Cleanup
        for client in clients:
            await client.disconnect()
        
        import os
        os.unlink(temp_file.name)
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_large_audio_file_processing(self, websocket_client, audio_generator):
        """Test processing of large audio files"""
        connected = await websocket_client.connect()
        assert connected is True
        
        # Generate longer audio files
        durations = [5.0, 10.0, 20.0]  # 5, 10, 20 seconds
        
        for duration in durations:
            # Generate audio
            audio_data = audio_generator.generate_sine_wave(duration=duration, frequency=440)
            
            # Save to temporary file
            import tempfile
            import soundfile as sf
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, audio_data, 16000)
            
            with open(temp_file.name, 'rb') as f:
                audio_bytes = f.read()
            
            # Send large audio file
            session_id = f"large_audio_{duration}_{uuid.uuid4()}"
            
            start_time = time.time()
            
            # Start audio session
            start_message = {
                "type": "start_audio",
                "session_id": session_id,
                "audio_format": "wav",
                "sample_rate": 16000
            }
            await websocket_client.send_message(start_message)
            
            # Send audio in chunks
            chunk_size = 4096  # Larger chunks for efficiency
            chunks_sent = 0
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                chunk_message = {
                    "type": "audio_data",
                    "session_id": session_id,
                    "data": chunk.hex(),
                    "chunk_id": chunks_sent
                }
                
                sent = await websocket_client.send_message(chunk_message)
                if sent:
                    chunks_sent += 1
                else:
                    break
                
                # Small delay
                await asyncio.sleep(0.001)
            
            # End audio session
            end_message = {"type": "end_audio", "session_id": session_id}
            await websocket_client.send_message(end_message)
            
            processing_time = time.time() - start_time
            
            # Should complete within reasonable time
            max_time = duration * 2 + 10  # Allow 2x duration + 10 seconds overhead
            assert processing_time < max_time, f"{duration}s audio took {processing_time:.1f}s to process"
            
            # Cleanup
            import os
            os.unlink(temp_file.name)
            
            # Wait between tests
            await asyncio.sleep(1)
        
        await websocket_client.disconnect()

class TestMemoryStress:
    """Stress tests for memory usage"""
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_memory_leak_detection(self, test_config):
        """Test for memory leaks during repeated operations"""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform repeated operations
        iterations = 10  # Reduced for CI
        
        for i in range(iterations):
            client = WebSocketTestClient(test_config["websocket_url"])
            
            # Connect, send messages, disconnect
            connected = await client.connect()
            if connected:
                # Send several messages
                for j in range(10):
                    message = {
                        "type": "heartbeat",
                        "data": "x" * 1024,  # 1KB of data
                        "iteration": i,
                        "message": j
                    }
                    await client.send_message(message)
                
                await client.disconnect()
            
            # Force garbage collection
            gc.collect()
            
            # Check memory every 5 iterations
            if i % 5 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Allow some memory increase but not excessive
                max_increase = 50  # 50MB max increase
                assert memory_increase < max_increase, f"Excessive memory increase: {memory_increase:.1f}MB"
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_connection_pool_stress(self, test_config):
        """Test connection pool under stress"""
        connections_created = 0
        max_iterations = 20  # Reduced for CI
        
        for i in range(max_iterations):
            batch_size = 5
            clients = []
            
            # Create batch of connections
            for j in range(batch_size):
                client = WebSocketTestClient(test_config["websocket_url"])
                connected = await client.connect()
                if connected:
                    clients.append(client)
                    connections_created += 1
            
            # Use connections briefly
            for client in clients:
                heartbeat = {"type": "heartbeat", "timestamp": time.time()}
                await client.send_message(heartbeat)
            
            # Disconnect all
            for client in clients:
                await client.disconnect()
            
            # Brief pause
            await asyncio.sleep(0.1)
        
        # Should have been able to create reasonable number of connections
        assert connections_created >= max_iterations * batch_size * 0.8

class TestErrorStress:
    """Stress tests for error handling"""
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_rapid_invalid_messages(self, websocket_client):
        """Test handling of many rapid invalid messages"""
        connected = await websocket_client.connect()
        assert connected is True
        
        # Send many invalid messages rapidly
        invalid_messages = [
            '{"type": "invalid"}',
            '{"invalid": "json"',
            '{}',
            '{"type": "unknown_type"}',
            '{"type": "heartbeat", "invalid_field": "value"}'
        ]
        
        errors_received = 0
        messages_sent = 0
        
        # Send invalid messages
        for i in range(20):  # Reduced for CI
            message = random.choice(invalid_messages)
            try:
                await websocket_client.websocket.send(message)
                messages_sent += 1
            except:
                pass
            
            # Small delay
            await asyncio.sleep(0.01)
        
        # Collect error responses
        timeout_time = time.time() + 5
        while time.time() < timeout_time:
            response = await websocket_client.receive_message(timeout=0.5)
            if response and response.get("type") == "error":
                errors_received += 1
        
        # Should handle errors gracefully and connection should remain
        assert websocket_client.websocket is not None
        
        # Should still be able to send valid message
        valid_message = {"type": "heartbeat", "timestamp": time.time()}
        sent = await websocket_client.send_message(valid_message)
        assert sent is True
        
        await websocket_client.disconnect()
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_connection_failure_recovery(self, test_config):
        """Test recovery from connection failures"""
        recovery_attempts = 5
        successful_recoveries = 0
        
        for i in range(recovery_attempts):
            client = WebSocketTestClient(test_config["websocket_url"])
            
            # Connect
            connected = await client.connect()
            if not connected:
                continue
            
            # Send some messages
            for j in range(5):
                message = {"type": "heartbeat", "timestamp": time.time()}
                await client.send_message(message)
            
            # Simulate abrupt disconnection
            if client.websocket:
                await client.websocket.close()
            
            # Attempt to reconnect
            await asyncio.sleep(0.5)
            reconnected = await client.connect()
            
            if reconnected:
                # Verify connection works
                test_message = {"type": "heartbeat", "timestamp": time.time()}
                sent = await client.send_message(test_message)
                if sent:
                    successful_recoveries += 1
                
                await client.disconnect()
        
        # Should achieve reasonable recovery rate
        recovery_rate = successful_recoveries / recovery_attempts
        assert recovery_rate >= 0.7, f"Recovery rate too low: {recovery_rate:.1%}"

# Performance benchmarking
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_baseline_performance_benchmark(self, test_config, benchmark_thresholds, performance_monitor):
        """Establish baseline performance metrics"""
        client = WebSocketTestClient(test_config["websocket_url"])
        
        # Connection benchmark
        start_time = time.time()
        connected = await client.connect()
        connection_time = time.time() - start_time
        
        assert connected is True
        assert_response_time(connection_time, benchmark_thresholds["connection_time"], "connection")
        
        # Message response benchmark
        num_tests = 10
        response_times = []
        
        for i in range(num_tests):
            start_time = time.time()
            message = {"type": "heartbeat", "timestamp": start_time, "test": i}
            await client.send_message(message)
            
            response = await client.receive_message(timeout=5.0)
            response_time = time.time() - start_time
            
            if response:
                response_times.append(response_time)
                performance_monitor.record_response_time(response_time)
                performance_monitor.record_success()
        
        # Analyze results
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            assert avg_response_time <= benchmark_thresholds["message_response_time"]
            assert max_response_time <= benchmark_thresholds["message_response_time"] * 2
        
        await client.disconnect()
        
        # Generate performance report
        stats = performance_monitor.get_statistics()
        print(f"\nPerformance Benchmark Results:")
        print(f"Connection Time: {connection_time:.3f}s")
        print(f"Avg Response Time: {stats.get('avg_response_time', 0):.3f}s")
        print(f"Success Rate: {stats.get('success_rate', 0):.1%}")
        print(f"Throughput: {stats.get('throughput', 0):.1f} ops/sec") 