#!/usr/bin/env python3
"""
Basic Audio Models Tests

Tests for the basic audio models that are currently implemented.
These tests focus on what's actually available in the codebase.
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports - go up 4 levels to get to orchestration-service root
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

from src.audio.models import (
    AudioChunkMetadata,
    AudioChunkingConfig,
    SourceType,
    ProcessingStatus,
    create_audio_chunk_metadata,
    get_default_chunking_config,
)

from src.audio.config import (
    AudioProcessingConfig,
    get_default_audio_processing_config,
    AudioPreset,
)


class TestAudioChunkMetadata(unittest.TestCase):
    """Test AudioChunkMetadata model."""
    
    def test_valid_chunk_metadata_creation(self):
        """Test creating valid chunk metadata."""
        chunk = AudioChunkMetadata(
            session_id="test_session",
            file_path="/tmp/test.wav",
            file_name="test.wav",
            file_size=1024,
            duration_seconds=2.0,
            chunk_sequence=1,
            chunk_start_time=0.0,
            chunk_end_time=2.0,
            source_type=SourceType.BOT_AUDIO
        )
        
        assert chunk.session_id == "test_session"
        assert chunk.file_path == "/tmp/test.wav"
        assert chunk.file_name == "test.wav"
        assert chunk.file_size == 1024
        assert chunk.duration_seconds == 2.0
        assert chunk.chunk_sequence == 1
        assert chunk.chunk_start_time == 0.0
        assert chunk.chunk_end_time == 2.0
        assert chunk.source_type == SourceType.BOT_AUDIO
    
    def test_chunk_metadata_with_factory_function(self):
        """Test creating chunk metadata with factory function."""
        chunk = create_audio_chunk_metadata(
            session_id="test_session",
            file_path="/tmp/test.wav",
            file_size=1024,
            duration_seconds=2.0,
            chunk_sequence=1,
            chunk_start_time=0.0,
            source_type=SourceType.BOT_AUDIO
        )
        
        assert chunk.session_id == "test_session"
        assert chunk.duration_seconds == 2.0
        assert chunk.chunk_end_time == 2.0  # Should be calculated
        assert chunk.file_name == "test.wav"  # Should be extracted from path


class TestAudioChunkingConfig(unittest.TestCase):
    """Test AudioChunkingConfig model."""
    
    def test_default_chunking_config(self):
        """Test default chunking configuration."""
        config = get_default_chunking_config()
        
        assert isinstance(config, AudioChunkingConfig)
        assert config.chunk_duration > 0
        assert config.overlap_duration >= 0
        assert config.overlap_duration < config.chunk_duration
        assert config.buffer_duration > config.chunk_duration
    
    def test_custom_chunking_config(self):
        """Test custom chunking configuration."""
        config = AudioChunkingConfig(
            chunk_duration=4.0,
            overlap_duration=0.8,
            processing_interval=3.5,
            buffer_duration=15.0
        )
        
        assert config.chunk_duration == 4.0
        assert config.overlap_duration == 0.8
        assert config.processing_interval == 3.5
        assert config.buffer_duration == 15.0
    
    def test_chunking_config_validation(self):
        """Test chunking configuration validation."""
        # This should work - overlap less than chunk duration
        config = AudioChunkingConfig(
            chunk_duration=3.0,
            overlap_duration=1.0
        )
        assert config.chunk_duration == 3.0
        assert config.overlap_duration == 1.0


class TestAudioProcessingConfig(unittest.TestCase):
    """Test AudioProcessingConfig model."""
    
    def test_default_audio_processing_config(self):
        """Test default audio processing configuration."""
        config = get_default_audio_processing_config()
        
        assert isinstance(config, AudioProcessingConfig)
        assert config.sample_rate == 16000
        assert config.preset_name == "default"
        assert len(config.enabled_stages) > 0
    
    def test_custom_audio_processing_config(self):
        """Test custom audio processing configuration."""
        config = AudioProcessingConfig(
            preset_name="test_preset",
            enabled_stages=["vad", "noise_reduction"],
            sample_rate=44100,
            real_time_priority=False
        )
        
        assert config.preset_name == "test_preset"
        assert config.enabled_stages == ["vad", "noise_reduction"]
        assert config.sample_rate == 44100
        assert config.real_time_priority == False


class TestEnums(unittest.TestCase):
    """Test enum types."""
    
    def test_source_type_enum(self):
        """Test SourceType enum."""
        assert SourceType.BOT_AUDIO == "bot_audio"
        assert SourceType.FRONTEND_TEST == "frontend_test"
        assert SourceType.GOOGLE_MEET == "google_meet"
        assert SourceType.MANUAL_UPLOAD == "manual_upload"
    
    def test_processing_status_enum(self):
        """Test ProcessingStatus enum."""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
        assert ProcessingStatus.TIMEOUT == "timeout"
    
    def test_audio_preset_enum(self):
        """Test AudioPreset enum."""
        assert AudioPreset.DEFAULT == "default"
        assert AudioPreset.VOICE_OPTIMIZED == "voice"
        assert AudioPreset.NOISY_ENVIRONMENT == "noisy"


class TestModelSerialization(unittest.TestCase):
    """Test model serialization and deserialization."""
    
    def test_chunk_metadata_json_serialization(self):
        """Test chunk metadata JSON serialization."""
        chunk = create_audio_chunk_metadata(
            session_id="test_session",
            file_path="/tmp/test.wav",
            file_size=1024,
            duration_seconds=2.0,
            chunk_sequence=1,
            chunk_start_time=0.0
        )
        
        # Serialize to JSON
        json_data = chunk.json()
        assert isinstance(json_data, str)
        assert "test_session" in json_data
        assert "test.wav" in json_data
    
    def test_chunking_config_dict_conversion(self):
        """Test chunking config dictionary conversion."""
        config = AudioChunkingConfig(
            chunk_duration=4.0,
            overlap_duration=0.8
        )
        
        # Convert to dictionary
        config_dict = config.dict()
        assert isinstance(config_dict, dict)
        assert config_dict["chunk_duration"] == 4.0
        assert config_dict["overlap_duration"] == 0.8
        
        # Create from dictionary
        new_config = AudioChunkingConfig(**config_dict)
        assert new_config.chunk_duration == 4.0
        assert new_config.overlap_duration == 0.8


if __name__ == "__main__":
    # Run tests directly without pytest to avoid configuration issues
    import unittest
    
    # Convert pytest classes to unittest and run
    suite = unittest.TestSuite()
    
    # Add test methods manually
    test_loader = unittest.TestLoader()
    suite.addTests(test_loader.loadTestsFromTestCase(TestAudioChunkMetadata))
    suite.addTests(test_loader.loadTestsFromTestCase(TestAudioChunkingConfig))
    suite.addTests(test_loader.loadTestsFromTestCase(TestAudioProcessingConfig))
    suite.addTests(test_loader.loadTestsFromTestCase(TestEnums))
    suite.addTests(test_loader.loadTestsFromTestCase(TestModelSerialization))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, failure in result.failures:
            print(f"  {test}: {failure}")
    
    if result.errors:
        print("\nErrors:")
        for test, error in result.errors:
            print(f"  {test}: {error}")