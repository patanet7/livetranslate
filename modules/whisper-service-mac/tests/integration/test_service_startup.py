#!/usr/bin/env python3
"""
Integration tests for service startup and configuration

Tests the complete service initialization process.
"""

import pytest
import time
import requests
import subprocess
import signal
import os
import threading
from pathlib import Path


class TestServiceStartup:
    """Test service startup and configuration"""
    
    @pytest.fixture
    def service_process(self):
        """Start the service in a subprocess for testing"""
        # Change to project directory
        project_dir = Path(__file__).parent.parent.parent
        
        # Start service with test configuration
        env = os.environ.copy()
        env.update({
            "TESTING": "true",
            "PORT": "5003",  # Use different port for testing
            "HOST": "127.0.0.1",
            "DEBUG": "true"
        })
        
        process = subprocess.Popen(
            ["python3", "src/main.py", "--port", "5003", "--host", "127.0.0.1"],
            cwd=project_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        # Wait for service to start
        time.sleep(3)
        
        yield process
        
        # Cleanup: terminate the process group
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    
    def test_service_starts_successfully(self, service_process):
        """Test that the service starts and responds to health checks"""
        # Check if service is running
        assert service_process.poll() is None, "Service process should be running"
        
        # Test health endpoint
        response = requests.get("http://127.0.0.1:5003/health", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "whisper-service-mac"
    
    def test_service_loads_configuration(self, service_process):
        """Test that service loads configuration correctly"""
        response = requests.get("http://127.0.0.1:5003/api/device-info", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert "platform" in data
        assert "capabilities" in data
    
    def test_service_lists_models(self, service_process):
        """Test that service can list available models"""
        response = requests.get("http://127.0.0.1:5003/api/models", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert "available_models" in data
        
        # Should have at least some models if they exist
        if data["available_models"]:
            assert isinstance(data["available_models"], list)
    
    def test_service_handles_cors(self, service_process):
        """Test that service has CORS headers enabled"""
        response = requests.get("http://127.0.0.1:5003/health", timeout=10)
        
        # Check for CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
    
    def test_service_error_handling(self, service_process):
        """Test service error handling for invalid requests"""
        # Test 404 for unknown endpoint
        response = requests.get("http://127.0.0.1:5003/invalid-endpoint", timeout=10)
        assert response.status_code == 404
        
        # Test 405 for wrong method
        response = requests.post("http://127.0.0.1:5003/health", timeout=10)
        assert response.status_code == 405


class TestConfigurationLoading:
    """Test configuration loading and validation"""
    
    def test_loads_default_config(self):
        """Test loading default configuration"""
        from core.whisper_cpp_engine import WhisperCppEngine
        
        engine = WhisperCppEngine()
        
        # Should have reasonable defaults
        assert engine.threads >= 1
        assert engine.models_dir is not None
        assert engine.enable_metal in [True, False]  # Boolean value
        assert engine.enable_coreml in [True, False]  # Boolean value
    
    def test_config_file_loading(self):
        """Test loading configuration from file"""
        config_path = Path(__file__).parent.parent.parent / "config" / "mac_config.yaml"
        
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
                
            # Check basic structure
            assert "whisper" in config
            assert "models" in config["whisper"]
            assert "hardware" in config["whisper"]


class TestServiceCompatibility:
    """Test compatibility with orchestration service expectations"""
    
    def test_orchestration_endpoints_present(self, service_process):
        """Test that all orchestration-required endpoints are present"""
        base_url = "http://127.0.0.1:5003"
        
        # Critical endpoints for orchestration
        endpoints = [
            "/health",
            "/api/models", 
            "/api/device-info",
            "/api/process-chunk"
        ]
        
        for endpoint in endpoints:
            if endpoint == "/api/process-chunk":
                # POST endpoint - test with OPTIONS for CORS
                response = requests.options(f"{base_url}{endpoint}", timeout=5)
                assert response.status_code in [200, 405]  # Either works or method not allowed
            else:
                # GET endpoints
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                assert response.status_code == 200, f"Endpoint {endpoint} should return 200"
    
    def test_response_format_compatibility(self, service_process):
        """Test that response formats match orchestration expectations"""
        base_url = "http://127.0.0.1:5003"
        
        # Test health response format
        response = requests.get(f"{base_url}/health")
        data = response.json()
        
        required_health_fields = ["status", "service", "version", "uptime"]
        for field in required_health_fields:
            assert field in data, f"Health response missing required field: {field}"
        
        # Test models response format
        response = requests.get(f"{base_url}/api/models")
        data = response.json()
        
        required_models_fields = ["available_models", "current_model"]
        for field in required_models_fields:
            assert field in data, f"Models response missing required field: {field}"
        
        # Test device info response format
        response = requests.get(f"{base_url}/api/device-info")
        data = response.json()
        
        required_device_fields = ["platform", "architecture", "capabilities"]
        for field in required_device_fields:
            assert field in data, f"Device info response missing required field: {field}"


class TestPerformance:
    """Test basic performance characteristics"""
    
    def test_health_endpoint_performance(self, service_process):
        """Test health endpoint response time"""
        start_time = time.time()
        response = requests.get("http://127.0.0.1:5003/health", timeout=5)
        end_time = time.time()
        
        assert response.status_code == 200
        
        response_time = end_time - start_time
        assert response_time < 1.0, f"Health endpoint too slow: {response_time:.3f}s"
    
    def test_concurrent_requests(self, service_process):
        """Test handling multiple concurrent requests"""
        def make_request():
            return requests.get("http://127.0.0.1:5003/health", timeout=10)
        
        # Start multiple threads making requests
        threads = []
        results = []
        
        for _ in range(5):
            def thread_func():
                try:
                    response = make_request()
                    results.append(response.status_code)
                except Exception as e:
                    results.append(str(e))
            
            thread = threading.Thread(target=thread_func)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=15)
        
        # Check results
        assert len(results) == 5, "Not all requests completed"
        for result in results:
            assert result == 200, f"Request failed: {result}"