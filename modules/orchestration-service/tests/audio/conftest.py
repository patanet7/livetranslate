#!/usr/bin/env python3
"""
Audio Testing Configuration and Fixtures

Basic test configuration and fixtures for audio processing components.
Provides test data, mock services, and utility functions for available modules.
"""

import asyncio
import os
import tempfile
import pytest
import pytest_asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
import json
from datetime import datetime, timedelta

# Import the audio components we're testing
import sys

# Add multiple possible paths to ensure we can find the src directory
current_dir = Path(__file__).parent
test_root = current_dir.parent
service_root = test_root.parent
src_path = service_root / "src"

# Add the src path to Python path
if src_path.exists():
    sys.path.insert(0, str(src_path.parent))
    sys.path.insert(0, str(src_path))

# Only import modules that actually exist
try:
    from src.audio.models import (
        AudioChunkMetadata,
        AudioChunkingConfig,
        QualityMetrics,
        SourceType,
        ProcessingStatus,
        create_audio_chunk_metadata,
        get_default_chunking_config,
    )
except ImportError as e:
    print(f"Warning: Could not import audio models: {e}")

try:
    from src.audio.config import (
        AudioPreset,
        AudioProcessingConfig,
        get_default_audio_processing_config,
    )
except ImportError as e:
    print(f"Warning: Could not import audio config: {e}")


# Test Configuration
TEST_SAMPLE_RATE = 16000
TEST_DURATION = 3.0  # seconds
TEST_CHANNELS = 1


class AudioTestFixtures:
    """Test audio data generation and management."""
    
    @staticmethod
    def generate_sine_wave(frequency: float, duration: float, amplitude: float = 0.1, sample_rate: int = TEST_SAMPLE_RATE) -> np.ndarray:
        """Generate a clean sine wave for testing."""
        t = np.arange(int(duration * sample_rate)) / sample_rate
        return amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    @staticmethod
    def generate_noise(duration: float, amplitude: float = 0.02, sample_rate: int = TEST_SAMPLE_RATE) -> np.ndarray:
        """Generate white noise for testing."""
        samples = int(duration * sample_rate)
        return (amplitude * np.random.randn(samples)).astype(np.float32)
    
    @staticmethod
    def generate_voice_like_audio(duration: float, fundamental_freq: float = 120, sample_rate: int = TEST_SAMPLE_RATE) -> np.ndarray:
        """Generate voice-like audio with harmonics."""
        t = np.arange(int(duration * sample_rate)) / sample_rate
        
        # Fundamental frequency
        signal = 0.3 * np.sin(2 * np.pi * fundamental_freq * t)
        
        # Add harmonics
        signal += 0.2 * np.sin(2 * np.pi * fundamental_freq * 2 * t)
        signal += 0.1 * np.sin(2 * np.pi * fundamental_freq * 3 * t)
        signal += 0.05 * np.sin(2 * np.pi * fundamental_freq * 4 * t)
        
        # Add formants
        signal += 0.1 * np.sin(2 * np.pi * 800 * t)  # First formant
        signal += 0.05 * np.sin(2 * np.pi * 1200 * t)  # Second formant
        
        # Add some noise for realism
        signal += 0.01 * np.random.randn(len(signal))
        
        return signal.astype(np.float32)
    
    @staticmethod
    def generate_silence(duration: float, sample_rate: int = TEST_SAMPLE_RATE) -> np.ndarray:
        """Generate silence for testing."""
        samples = int(duration * sample_rate)
        return np.zeros(samples, dtype=np.float32)


# Pytest Fixtures

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_audio_fixtures():
    """Provide audio test data fixtures."""
    return AudioTestFixtures()


@pytest.fixture
def sample_audio_data(test_audio_fixtures):
    """Generate sample audio data for testing."""
    return {
        "sine_440": test_audio_fixtures.generate_sine_wave(440, TEST_DURATION),
        "voice_like": test_audio_fixtures.generate_voice_like_audio(TEST_DURATION),
        "noise": test_audio_fixtures.generate_noise(TEST_DURATION),
        "silence": test_audio_fixtures.generate_silence(TEST_DURATION),
    }


@pytest.fixture
async def mock_database_adapter():
    """Provide a mock database adapter for testing."""
    mock_adapter = AsyncMock()
    
    # Mock successful operations
    mock_adapter.initialize.return_value = True
    mock_adapter.store_audio_chunk.return_value = "test_chunk_id"
    mock_adapter.store_transcript.return_value = "test_transcript_id"
    mock_adapter.store_translation.return_value = "test_translation_id"
    mock_adapter.store_speaker_correlation.return_value = "test_correlation_id"
    mock_adapter.get_session_audio_chunks.return_value = []
    mock_adapter.get_session_transcripts.return_value = []
    mock_adapter.get_speaker_correlations.return_value = []
    
    return mock_adapter


# Utility Functions for Testing

def assert_audio_quality(audio_data: np.ndarray, min_quality: float = 0.0, max_quality: float = 1.0):
    """Assert audio quality is within expected range."""
    # Basic quality checks
    assert not np.isnan(audio_data).any(), "Audio contains NaN values"
    assert not np.isinf(audio_data).any(), "Audio contains infinite values"
    assert len(audio_data) > 0, "Audio data is empty"
    
    # RMS level check
    rms = np.sqrt(np.mean(audio_data ** 2))
    assert 0.0 <= rms <= 1.0, f"RMS level {rms} out of range [0, 1]"
    
    # Peak level check
    peak = np.max(np.abs(audio_data))
    assert 0.0 <= peak <= 1.0, f"Peak level {peak} out of range [0, 1]"