"""
Sample unit test demonstrating fixture usage.

This is a template for creating new unit tests in the whisper-service.
"""

import numpy as np
import pytest


def test_audio_duration(hello_world_audio):
    """Test that audio fixture has correct duration."""
    audio, sr = hello_world_audio
    duration = len(audio) / sr
    assert duration == pytest.approx(3.0, abs=0.1)


def test_audio_properties(short_speech_audio):
    """Test audio properties."""
    audio, sr = short_speech_audio

    # Check dtype
    assert audio.dtype == np.float32

    # Check sample rate
    assert sr == 16000

    # Check amplitude range
    assert np.max(audio) <= 1.0
    assert np.min(audio) >= -1.0


@pytest.mark.slow
def test_slow_operation(long_speech_audio):
    """Example of a slow test that can be skipped."""
    audio, sr = long_speech_audio

    # Simulate slow processing
    result = np.fft.fft(audio)
    assert len(result) == len(audio)


def test_with_temp_dir(temp_audio_dir, hello_world_audio):
    """Example of using temporary directory fixture."""
    import soundfile as sf

    audio, sr = hello_world_audio

    # Write to temp directory
    output_file = temp_audio_dir / "output.wav"
    sf.write(output_file, audio, sr)

    assert output_file.exists()

    # Read back and verify
    audio_read, sr_read = sf.read(output_file, dtype=np.float32)
    assert sr_read == sr
    assert np.allclose(audio_read, audio)


def test_with_mock_model(mock_whisper_model):
    """Example of using mock Whisper model."""
    # Test the mock model
    result = mock_whisper_model.transcribe(np.zeros(16000))

    assert "text" in result
    assert result["text"] == "Hello world"
    assert "segments" in result
    assert len(result["segments"]) == 2


def test_with_config(default_whisper_config):
    """Example of using default configuration."""
    config = default_whisper_config

    assert config["model_name"] == "base"
    assert config["sample_rate"] == 16000
    assert config["device"] == "cpu"

    # Modify config for test
    config["temperature"] = 0.5
    assert config["temperature"] == 0.5
