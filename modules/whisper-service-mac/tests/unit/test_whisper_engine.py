#!/usr/bin/env python3
"""
Unit tests for WhisperCppEngine

Tests the core whisper.cpp integration and Apple Silicon optimizations.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from core.whisper_cpp_engine import WhisperCppEngine, WhisperResult, WhisperCppError


class TestWhisperCppEngine:
    """Test WhisperCppEngine functionality"""
    
    def test_engine_initialization(self):
        """Test engine initialization with default settings"""
        engine = WhisperCppEngine()
        
        assert engine.whisper_binary is not None
        assert engine.models_dir is not None
        assert engine.enable_metal is True
        assert engine.enable_coreml is True
        assert engine.threads == 4
        
    def test_engine_initialization_with_config(self):
        """Test engine initialization with custom configuration"""
        config = {
            "whisper_binary": "/custom/path/whisper",
            "models_dir": "/custom/models",
            "enable_metal": False,
            "enable_coreml": False,
            "threads": 8
        }
        
        engine = WhisperCppEngine(**config)
        
        assert engine.whisper_binary == "/custom/path/whisper"
        assert str(engine.models_dir) == "/custom/models"
        assert engine.enable_metal is False
        assert engine.enable_coreml is False
        assert engine.threads == 8
        
    @patch('subprocess.run')
    def test_detect_capabilities(self, mock_run):
        """Test hardware capability detection"""
        # Mock whisper.cpp output showing capabilities
        mock_output = """whisper.cpp
METAL = 1
COREML = 1
NEON = 1
usage: ./whisper [options] file0.wav file1.wav ..."""
        
        mock_run.return_value.stdout = mock_output
        mock_run.return_value.returncode = 0
        
        engine = WhisperCppEngine()
        capabilities = engine._detect_capabilities()
        
        assert capabilities["metal"] is True
        assert capabilities["coreml"] is True
        assert capabilities["neon"] is True
        
    @patch('subprocess.run')
    def test_detect_capabilities_no_acceleration(self, mock_run):
        """Test capability detection when no acceleration available"""
        mock_output = """whisper.cpp
METAL = 0
COREML = 0
usage: ./whisper [options] file0.wav file1.wav ..."""
        
        mock_run.return_value.stdout = mock_output
        mock_run.return_value.returncode = 0
        
        engine = WhisperCppEngine()
        capabilities = engine._detect_capabilities()
        
        assert capabilities["metal"] is False
        assert capabilities["coreml"] is False
        
    @patch('os.listdir')
    @patch('os.path.exists')
    def test_list_models(self, mock_exists, mock_listdir):
        """Test model listing functionality"""
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "ggml-tiny.bin",
            "ggml-base.bin", 
            "ggml-small.bin",
            "other-file.txt"  # Should be ignored
        ]
        
        engine = WhisperCppEngine()
        models = engine.list_models()
        
        assert "tiny" in models
        assert "base" in models
        assert "small" in models
        assert len(models) == 3
        
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_get_model_info(self, mock_getsize, mock_exists):
        """Test getting model information"""
        mock_exists.return_value = True
        mock_getsize.return_value = 142000000  # 142MB
        
        engine = WhisperCppEngine()
        info = engine.get_model_info("base")
        
        assert info["name"] == "base"
        assert "file_path" in info
        assert "size" in info
        assert info["file_path"].endswith("ggml-base.bin")
        
    @patch('os.path.exists')
    def test_get_model_info_not_found(self, mock_exists):
        """Test getting info for non-existent model"""
        mock_exists.return_value = False
        
        engine = WhisperCppEngine()
        info = engine.get_model_info("nonexistent")
        
        assert info is None
        
    @patch('os.path.exists')
    def test_load_model_success(self, mock_exists):
        """Test successful model loading"""
        mock_exists.return_value = True
        
        engine = WhisperCppEngine()
        result = engine.load_model("base")
        
        assert result is True
        assert engine.current_model == "base"
        
    @patch('os.path.exists')
    def test_load_model_not_found(self, mock_exists):
        """Test loading non-existent model"""
        mock_exists.return_value = False
        
        engine = WhisperCppEngine()
        
        with pytest.raises(WhisperCppError):
            engine.load_model("nonexistent")
            
    def test_temp_audio_file_context_manager(self):
        """Test temporary audio file creation and cleanup"""
        engine = WhisperCppEngine()
        
        # Create sample audio data
        audio_data = np.random.rand(16000).astype(np.float32)
        sample_rate = 16000
        
        temp_path = None
        with engine._temp_audio_file(audio_data, sample_rate) as path:
            temp_path = path
            assert os.path.exists(path)
            assert path.endswith('.wav')
            
        # File should be cleaned up after context
        assert not os.path.exists(temp_path)
        
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_run_whisper_cpp_success(self, mock_exists, mock_run):
        """Test successful whisper.cpp execution"""
        mock_exists.return_value = True
        
        # Mock successful whisper.cpp output
        mock_output = '''[
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": " This is a test transcription."
    }
  ],
  "text": "This is a test transcription."
}
]'''
        
        mock_run.return_value.stdout = mock_output
        mock_run.return_value.returncode = 0
        
        engine = WhisperCppEngine()
        engine.current_model = "base"
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            result = engine._run_whisper_cpp(
                model_path="/fake/path/ggml-base.bin",
                audio_file=temp_file.name,
                language="en"
            )
            
        assert isinstance(result, WhisperResult)
        assert result.text == "This is a test transcription."
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0]["text"] == " This is a test transcription."
        
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_run_whisper_cpp_with_word_timestamps(self, mock_exists, mock_run):
        """Test whisper.cpp execution with word-level timestamps"""
        mock_exists.return_value = True
        
        # Mock output with word timestamps
        mock_output = '''[
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": " This is a test.",
      "words": [
        {"word": "This", "start": 0.0, "end": 0.3},
        {"word": "is", "start": 0.3, "end": 0.4},
        {"word": "a", "start": 0.4, "end": 0.5},
        {"word": "test.", "start": 0.5, "end": 0.8}
      ]
    }
  ],
  "text": "This is a test."
}
]'''
        
        mock_run.return_value.stdout = mock_output
        mock_run.return_value.returncode = 0
        
        engine = WhisperCppEngine()
        engine.current_model = "base"
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            result = engine._run_whisper_cpp(
                model_path="/fake/path/ggml-base.bin",
                audio_file=temp_file.name,
                word_timestamps=True
            )
            
        assert result.word_timestamps is not None
        assert len(result.word_timestamps) == 4
        assert result.word_timestamps[0]["word"] == "This"
        assert result.word_timestamps[0]["start"] == 0.0
        
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_run_whisper_cpp_failure(self, mock_exists, mock_run):
        """Test whisper.cpp execution failure"""
        mock_exists.return_value = True
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Error: Model file not found"
        
        engine = WhisperCppEngine()
        engine.current_model = "base"
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            with pytest.raises(WhisperCppError):
                engine._run_whisper_cpp(
                    model_path="/fake/path/ggml-base.bin",
                    audio_file=temp_file.name
                )
                
    @patch.object(WhisperCppEngine, '_run_whisper_cpp')
    @patch('os.path.exists')
    def test_transcribe_with_file_path(self, mock_exists, mock_run_whisper):
        """Test transcription with audio file path"""
        mock_exists.return_value = True
        
        # Mock whisper result
        mock_result = WhisperResult(
            text="Test transcription",
            language="en",
            segments=[{"start": 0.0, "end": 1.0, "text": "Test transcription"}]
        )
        mock_run_whisper.return_value = mock_result
        
        engine = WhisperCppEngine()
        engine.current_model = "base"
        
        result = engine.transcribe("/fake/audio.wav", model="base")
        
        assert result.text == "Test transcription"
        assert result.model_name == "base"
        mock_run_whisper.assert_called_once()
        
    @patch.object(WhisperCppEngine, '_run_whisper_cpp')
    @patch.object(WhisperCppEngine, '_temp_audio_file')
    @patch('os.path.exists')
    def test_transcribe_with_audio_data(self, mock_exists, mock_temp_file, mock_run_whisper):
        """Test transcription with raw audio data"""
        mock_exists.return_value = True
        
        # Mock temporary file
        mock_temp_file.return_value.__enter__.return_value = "/tmp/audio.wav"
        mock_temp_file.return_value.__exit__.return_value = None
        
        # Mock whisper result
        mock_result = WhisperResult(
            text="Test transcription",
            language="en",
            segments=[{"start": 0.0, "end": 1.0, "text": "Test transcription"}]
        )
        mock_run_whisper.return_value = mock_result
        
        engine = WhisperCppEngine()
        engine.current_model = "base"
        
        # Create sample audio data
        audio_data = np.random.rand(16000).astype(np.float32)
        
        result = engine.transcribe(audio_data, model="base", sample_rate=16000)
        
        assert result.text == "Test transcription"
        mock_temp_file.assert_called_once_with(audio_data, 16000)
        mock_run_whisper.assert_called_once()
        
    def test_transcribe_no_model_loaded(self):
        """Test transcription when no model is loaded"""
        engine = WhisperCppEngine()
        
        with pytest.raises(WhisperCppError, match="No model loaded"):
            engine.transcribe("/fake/audio.wav")
            
    @patch('platform.platform')
    @patch('platform.machine')
    @patch('platform.mac_ver')
    def test_get_device_info(self, mock_mac_ver, mock_machine, mock_platform):
        """Test device information gathering"""
        mock_platform.return_value = "macOS-15.5-arm64-arm-64bit"
        mock_machine.return_value = "arm64"
        mock_mac_ver.return_value = ("15.5", "", "")
        
        engine = WhisperCppEngine()
        
        with patch.object(engine, '_detect_capabilities') as mock_caps:
            mock_caps.return_value = {"metal": True, "coreml": True}
            
            device_info = engine.get_device_info()
            
        assert device_info["platform"] == "macOS-15.5-arm64-arm64"
        assert device_info["architecture"] == "arm64"
        assert device_info["apple_silicon"] is True
        assert device_info["macos_version"] == "15.5"
        assert device_info["capabilities"]["metal"] is True
        
    def test_performance_tracking(self):
        """Test performance metrics tracking"""
        engine = WhisperCppEngine()
        
        # Initially no times recorded
        assert len(engine.inference_times) == 0
        assert engine.get_average_inference_time() == 0.0
        
        # Add some times
        engine.inference_times.extend([1.0, 2.0, 3.0])
        
        assert engine.get_average_inference_time() == 2.0
        
        # Test max times limit
        engine.max_inference_times = 2
        engine.inference_times.extend([4.0, 5.0])  # Should trim to last 2
        
        assert len(engine.inference_times) == 2
        assert engine.inference_times == [4.0, 5.0]


class TestWhisperResult:
    """Test WhisperResult dataclass"""
    
    def test_whisper_result_creation(self):
        """Test creating WhisperResult with required fields"""
        result = WhisperResult(
            text="Test text",
            language="en",
            segments=[{"start": 0.0, "end": 1.0, "text": "Test text"}]
        )
        
        assert result.text == "Test text"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.word_timestamps is None
        assert result.processing_time == 0.0
        assert result.confidence == 0.0
        
    def test_whisper_result_with_optional_fields(self):
        """Test creating WhisperResult with all fields"""
        word_timestamps = [
            {"word": "Test", "start": 0.0, "end": 0.5},
            {"word": "text", "start": 0.5, "end": 1.0}
        ]
        
        result = WhisperResult(
            text="Test text",
            language="en",
            segments=[{"start": 0.0, "end": 1.0, "text": "Test text"}],
            word_timestamps=word_timestamps,
            processing_time=1.5,
            model_name="base",
            confidence=0.95
        )
        
        assert result.word_timestamps == word_timestamps
        assert result.processing_time == 1.5
        assert result.model_name == "base"
        assert result.confidence == 0.95


class TestModelNameConversion:
    """Test model name conversion for orchestration compatibility"""
    
    def test_convert_orchestration_model_names(self):
        """Test conversion from orchestration format to GGML format"""
        engine = WhisperCppEngine()
        
        # Test orchestration -> GGML conversion
        assert engine._convert_model_name("whisper-tiny") == "tiny"
        assert engine._convert_model_name("whisper-base") == "base"
        assert engine._convert_model_name("whisper-small") == "small"
        assert engine._convert_model_name("whisper-medium") == "medium"
        assert engine._convert_model_name("whisper-large-v3") == "large-v3"
        
        # Test already GGML format (no conversion)
        assert engine._convert_model_name("tiny") == "tiny"
        assert engine._convert_model_name("base") == "base"
        assert engine._convert_model_name("large-v3") == "large-v3"
        
    def test_get_ggml_model_path(self):
        """Test GGML model path generation"""
        engine = WhisperCppEngine()
        
        path = engine._get_ggml_model_path("base")
        assert path.name == "ggml-base.bin"
        assert "ggml" in str(path)
        
        # Test with already prefixed name
        path = engine._get_ggml_model_path("ggml-base")
        assert path.name == "ggml-base.bin"