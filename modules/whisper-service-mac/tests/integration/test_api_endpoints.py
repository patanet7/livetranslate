#!/usr/bin/env python3
"""
Integration tests for whisper-service-mac API endpoints

Tests all API endpoints for compatibility with orchestration service.
"""

import json
import pytest
import io
import base64
from unittest.mock import patch, MagicMock


class TestHealthEndpoint:
    """Test /health endpoint"""
    
    def test_health_check_success(self, client):
        """Test health endpoint returns success"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data["status"] == "healthy"
        assert data["service"] == "whisper-service-mac"
        assert "version" in data
        assert "uptime" in data
        assert "timestamp" in data
        
    def test_health_check_includes_capabilities(self, client):
        """Test health endpoint includes hardware capabilities"""
        response = client.get("/health")
        data = json.loads(response.data)
        
        assert "capabilities" in data
        capabilities = data["capabilities"]
        
        # Check for macOS-specific capabilities
        expected_caps = ["metal", "coreml", "ane", "unified_memory", "neon"]
        for cap in expected_caps:
            assert cap in capabilities


class TestModelsEndpoint:
    """Test /models and /api/models endpoints"""
    
    def test_models_list(self, client):
        """Test models listing endpoint"""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "models" in data
        assert len(data["models"]) >= 1  # At least some models
        
        # Check model structure
        if data["models"]:
            model = data["models"][0]
            assert "name" in model
            assert "file_name" in model
            assert "size" in model
            assert "format" in model
        
    def test_api_models_endpoint(self, client):
        """Test /api/models endpoint (orchestration compatibility)"""
        response = client.get("/api/models")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "available_models" in data
        assert "current_model" in data
        assert len(data["available_models"]) >= 1  # At least some models


class TestDeviceInfoEndpoint:
    """Test /api/device-info endpoint"""
    
    def test_device_info(self, client):
        """Test device info endpoint"""
        response = client.get("/api/device-info")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data["platform"] == "macOS-15.5-arm64-arm-64bit"
        assert data["architecture"] == "arm64"
        assert data["device_type"] == "Apple Silicon"
        assert "capabilities" in data
        assert "acceleration" in data
        
        capabilities = data["capabilities"]
        assert capabilities["metal"] is True
        assert capabilities["coreml"] is True
        
        acceleration = data["acceleration"]
        assert acceleration["metal"] is True
        assert acceleration["coreml"] is True


class TestTranscribeEndpoint:
    """Test /transcribe endpoint"""
    
    def test_transcribe_with_file_upload(self, client, sample_audio_file):
        """Test transcription with file upload"""
        
        with open(sample_audio_file, 'rb') as f:
            data = {
                'audio': (f, 'test.wav', 'audio/wav'),
                'model': 'base',
                'language': 'en'
            }
            
            response = client.post('/transcribe', data=data, content_type='multipart/form-data')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        assert "text" in result
        assert "language" in result
        assert "processing_time" in result
        assert result["text"] == "This is a test transcription."
        
    def test_transcribe_with_json_data(self, client, sample_audio_data, json_headers):
        """Test transcription with JSON audio data"""
        
        audio_data, sample_rate = sample_audio_data
        audio_b64 = base64.b64encode(audio_data.tobytes()).decode()
        
        data = {
            'audio_data': audio_b64,
            'sample_rate': sample_rate,
            'model': 'base',
            'language': 'en',
            'task': 'transcribe'
        }
        
        response = client.post('/transcribe', json=data, headers=json_headers)
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        assert "text" in result
        assert "segments" in result
        assert result["language"] == "en"
        
    def test_transcribe_missing_audio(self, client, json_headers):
        """Test transcription with missing audio data"""
        data = {
            'model': 'base',
            'language': 'en'
        }
        
        response = client.post('/transcribe', json=data, headers=json_headers)
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert "error" in result


class TestProcessChunkEndpoint:
    """Test /api/process-chunk endpoint (orchestration compatibility)"""
    
    @patch('api.api_server.whisper_engine')
    def test_process_chunk_success(self, mock_engine, client, sample_audio_data, mock_whisper_engine, json_headers):
        """Test audio chunk processing"""
        mock_engine.transcribe.return_value = mock_whisper_engine.transcribe(None)
        
        audio_data, sample_rate = sample_audio_data
        audio_b64 = base64.b64encode(audio_data.tobytes()).decode()
        
        data = {
            'audio_data': audio_b64,
            'sample_rate': sample_rate,
            'model': 'base',
            'chunk_id': 'test-chunk-001',
            'session_id': 'test-session'
        }
        
        response = client.post('/api/process-chunk', json=data, headers=json_headers)
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Check orchestration-compatible response format
        assert "transcription" in result
        assert "chunk_id" in result
        assert "session_id" in result
        assert "processing_time" in result
        assert "model_used" in result
        
        assert result["chunk_id"] == "test-chunk-001"
        assert result["session_id"] == "test-session"
        assert result["model_used"] == "base"
        
    @patch('api.api_server.whisper_engine')
    def test_process_chunk_with_model_conversion(self, mock_engine, client, sample_audio_data, mock_whisper_engine, json_headers):
        """Test chunk processing with model name conversion"""
        mock_engine.transcribe.return_value = mock_whisper_engine.transcribe(None)
        
        audio_data, sample_rate = sample_audio_data
        audio_b64 = base64.b64encode(audio_data.tobytes()).decode()
        
        # Test with orchestration model name format
        data = {
            'audio_data': audio_b64,
            'sample_rate': sample_rate,
            'model': 'whisper-base',  # Orchestration format
            'chunk_id': 'test-chunk-002'
        }
        
        response = client.post('/api/process-chunk', json=data, headers=json_headers)
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should convert whisper-base -> base for GGML
        mock_engine.transcribe.assert_called_once()
        call_args = mock_engine.transcribe.call_args
        assert "model" in call_args.kwargs
        assert call_args.kwargs["model"] == "base"  # Converted name
        
    def test_process_chunk_missing_data(self, client, json_headers):
        """Test chunk processing with missing audio data"""
        data = {
            'model': 'base',
            'chunk_id': 'test-chunk-003'
        }
        
        response = client.post('/api/process-chunk', json=data, headers=json_headers)
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert "error" in result
        
    def test_process_chunk_invalid_audio_data(self, client, json_headers):
        """Test chunk processing with invalid base64 audio data"""
        data = {
            'audio_data': 'invalid-base64-data',
            'sample_rate': 16000,
            'model': 'base',
            'chunk_id': 'test-chunk-004'
        }
        
        response = client.post('/api/process-chunk', json=data, headers=json_headers)
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert "error" in result


class TestMacOSSpecificEndpoints:
    """Test macOS-specific API endpoints"""
    
    @patch('api.api_server.whisper_engine')
    def test_metal_status(self, mock_engine, client):
        """Test Metal GPU status endpoint"""
        mock_engine.get_capabilities.return_value = {
            "metal": True,
            "metal_performance_shaders": True,
            "unified_memory": True
        }
        
        response = client.get("/api/metal/status")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "metal_available" in data
        assert "metal_performance_shaders" in data
        assert "unified_memory" in data
        
    @patch('api.api_server.whisper_engine')
    def test_coreml_models(self, mock_engine, client):
        """Test Core ML models endpoint"""
        mock_coreml_models = [
            {"name": "whisper-base-coreml", "path": "/fake/path/base.mlmodelc"},
            {"name": "whisper-small-coreml", "path": "/fake/path/small.mlmodelc"}
        ]
        
        with patch('api.api_server.list_coreml_models', return_value=mock_coreml_models):
            response = client.get("/api/coreml/models")
            
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "coreml_models" in data
        assert len(data["coreml_models"]) == 2
        assert data["coreml_models"][0]["name"] == "whisper-base-coreml"


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_404_for_unknown_endpoint(self, client):
        """Test 404 response for unknown endpoints"""
        response = client.get("/api/unknown-endpoint")
        assert response.status_code == 404
        
    def test_405_for_wrong_method(self, client):
        """Test 405 response for wrong HTTP method"""
        response = client.post("/health")
        assert response.status_code == 405
        
    @patch('api.api_server.whisper_engine')
    def test_transcribe_engine_error(self, mock_engine, client, sample_audio_data, json_headers):
        """Test transcription when engine raises error"""
        mock_engine.transcribe.side_effect = Exception("Engine error")
        
        audio_data, sample_rate = sample_audio_data
        audio_b64 = base64.b64encode(audio_data.tobytes()).decode()
        
        data = {
            'audio_data': audio_b64,
            'sample_rate': sample_rate,
            'model': 'base'
        }
        
        response = client.post('/transcribe', json=data, headers=json_headers)
        
        assert response.status_code == 500
        result = json.loads(response.data)
        assert "error" in result
        
    def test_invalid_json_request(self, client, json_headers):
        """Test handling of invalid JSON in request"""
        response = client.post('/transcribe', data='invalid json', headers=json_headers)
        
        assert response.status_code == 400


class TestCORSHeaders:
    """Test CORS headers for browser compatibility"""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present"""
        response = client.get("/health")
        
        # Check for common CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        
    def test_options_request(self, client):
        """Test OPTIONS request for CORS preflight"""
        response = client.options("/api/models")
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Methods" in response.headers


class TestPerformanceMetrics:
    """Test performance tracking and metrics"""
    
    @patch('api.api_server.whisper_engine')
    def test_performance_tracking_in_response(self, mock_engine, client, sample_audio_data, mock_whisper_engine, json_headers):
        """Test that performance metrics are included in responses"""
        mock_engine.transcribe.return_value = mock_whisper_engine.transcribe(None)
        
        audio_data, sample_rate = sample_audio_data
        audio_b64 = base64.b64encode(audio_data.tobytes()).decode()
        
        data = {
            'audio_data': audio_b64,
            'sample_rate': sample_rate,
            'model': 'base'
        }
        
        response = client.post('/transcribe', json=data, headers=json_headers)
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        assert "processing_time" in result
        assert isinstance(result["processing_time"], (int, float))
        assert result["processing_time"] >= 0