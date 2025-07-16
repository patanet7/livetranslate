#!/usr/bin/env python3
"""
Test configuration and fixtures for whisper-service-mac
"""

import os
import sys
import time
import pytest
import tempfile
import numpy as np
from pathlib import Path

# Add src to Python path
test_dir = Path(__file__).parent
project_dir = test_dir.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

@pytest.fixture
def app():
    """Create Flask test application"""
    # Set test environment
    os.environ["TESTING"] = "true"
    os.environ["WHISPER_SERVICE_TYPE"] = "mac"
    
    # Import and patch the global whisper_engine
    import api.api_server as api_module
    from unittest.mock import MagicMock
    
    # Create mock engine that matches actual WhisperCppEngine API
    mock_engine = MagicMock()
    
    # Mock capabilities (matches _detect_capabilities + macOS specific)
    mock_engine.capabilities = {
        "metal": True,
        "coreml": True, 
        "accelerate": True,
        "neon": True,
        "avx": False,
        "word_timestamps": True,
        "quantization": True,
        "ane": True,  # Apple Neural Engine via Core ML
        "unified_memory": True  # Apple Silicon unified memory
    }
    mock_engine.get_capabilities.return_value = mock_engine.capabilities
    
    # Mock current model (should be a dict, not string)
    mock_engine.current_model = {
        "name": "base",
        "file_path": "/fake/path/ggml-base.bin",
        "size": "142MB"
    }
    
    # Mock available models list
    mock_engine.available_models = [
        {
            "name": "tiny", 
            "file_path": "/fake/path/ggml-tiny.bin", 
            "file_name": "ggml-tiny.bin",
            "size": "39MB",
            "quantization": None,
            "coreml_available": True
        },
        {
            "name": "base", 
            "file_path": "/fake/path/ggml-base.bin", 
            "file_name": "ggml-base.bin",
            "size": "142MB",
            "quantization": None,
            "coreml_available": True
        },
        {
            "name": "small", 
            "file_path": "/fake/path/ggml-small.bin", 
            "file_name": "ggml-small.bin",
            "size": "488MB",
            "quantization": "q5_0",
            "coreml_available": False
        }
    ]
    
    # Mock methods
    mock_engine.get_available_models.return_value = mock_engine.available_models
    mock_engine.load_model.return_value = True
    mock_engine.get_device_info.return_value = {
        "platform": "macOS-15.5-arm64-arm-64bit",
        "architecture": "arm64",
        "apple_silicon": True,
        "macos_version": "15.5",
        "python_version": "3.12.0",
        "capabilities": {
            "metal": True,
            "coreml": True,
            "ane": True,
            "unified_memory": True,
            "neon": True
        },
        "whisper_cpp_available": True
    }
    
    # Mock transcription method
    def mock_transcribe(audio_data, **kwargs):
        from core.whisper_cpp_engine import WhisperResult
        return WhisperResult(
            text="This is a test transcription.",
            language="en",
            segments=[{
                "start": 0.0,
                "end": 2.0,
                "text": "This is a test transcription."
            }],
            word_timestamps=[
                {"word": "This", "start": 0.0, "end": 0.3},
                {"word": "is", "start": 0.3, "end": 0.4},
                {"word": "a", "start": 0.4, "end": 0.5},
                {"word": "test", "start": 0.5, "end": 0.8},
                {"word": "transcription.", "start": 0.8, "end": 2.0}
            ],
            processing_time=0.5,
            model_name=kwargs.get('model', 'base'),
            confidence=0.95
        )
    
    mock_engine.transcribe = mock_transcribe
    
    # Replace global engine with mock
    original_engine = api_module.whisper_engine
    api_module.whisper_engine = mock_engine
    
    # Get the app instance
    app = api_module.app
    app.config["TESTING"] = True
    app.config["start_time"] = time.time()
    
    with app.app_context():
        yield app
    
    # Restore original engine
    api_module.whisper_engine = original_engine


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing"""
    # Generate 1 second of sine wave at 16kHz
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return audio_data, sample_rate


@pytest.fixture
def sample_audio_file(sample_audio_data):
    """Create temporary audio file for testing"""
    import soundfile as sf
    
    audio_data, sample_rate = sample_audio_data
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_data, sample_rate)
        yield f.name
        
    # Cleanup
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.fixture
def mock_whisper_engine():
    """Mock WhisperCppEngine for testing"""
    class MockWhisperEngine:
        def __init__(self):
            self.models = {
                "tiny": {"name": "tiny", "file_path": "/fake/path/ggml-tiny.bin", "size": "39MB"},
                "base": {"name": "base", "file_path": "/fake/path/ggml-base.bin", "size": "142MB"},
                "small": {"name": "small", "file_path": "/fake/path/ggml-small.bin", "size": "488MB"}
            }
            self.current_model = "base"
            
        def list_models(self):
            return list(self.models.keys())
            
        def get_model_info(self, model_name):
            return self.models.get(model_name)
            
        def load_model(self, model_name):
            if model_name in self.models:
                self.current_model = model_name
                return True
            return False
            
        def transcribe(self, audio_data, **kwargs):
            # Mock transcription result
            from core.whisper_cpp_engine import WhisperResult
            return WhisperResult(
                text="This is a test transcription.",
                language="en",
                segments=[{
                    "start": 0.0,
                    "end": 2.0,
                    "text": "This is a test transcription."
                }],
                word_timestamps=[
                    {"word": "This", "start": 0.0, "end": 0.3},
                    {"word": "is", "start": 0.3, "end": 0.4},
                    {"word": "a", "start": 0.4, "end": 0.5},
                    {"word": "test", "start": 0.5, "end": 0.8},
                    {"word": "transcription.", "start": 0.8, "end": 2.0}
                ],
                processing_time=0.5,
                model_name=self.current_model,
                confidence=0.95
            )
            
        def get_capabilities(self):
            return {
                "metal": True,
                "coreml": True,
                "ane": True,
                "unified_memory": True,
                "neon": True
            }
            
        def get_device_info(self):
            return {
                "platform": "macOS-15.5-arm64-arm-64bit",
                "architecture": "arm64",
                "python_version": "3.12.0",
                "apple_silicon": True,
                "macos_version": "15.5",
                "capabilities": self.get_capabilities(),
                "whisper_cpp_available": True
            }
    
    return MockWhisperEngine()


@pytest.fixture
def json_headers():
    """Common JSON headers for API requests"""
    return {"Content-Type": "application/json"}


@pytest.fixture
def multipart_headers():
    """Headers for multipart form data"""
    return {"Content-Type": "multipart/form-data"}