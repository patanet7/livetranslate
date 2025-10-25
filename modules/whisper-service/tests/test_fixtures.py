"""
Test that fixtures are working correctly.

This file verifies that all audio fixtures are generated and can be loaded.
"""

import numpy as np
import pytest
from pathlib import Path


def test_hello_world_fixture(hello_world_audio):
    """Test hello_world.wav fixture loads correctly."""
    audio, sr = hello_world_audio
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == 16000
    assert len(audio) > 0
    assert len(audio) / sr == pytest.approx(3.0, abs=0.1)  # ~3 seconds


def test_silence_fixture(silence_audio):
    """Test silence.wav fixture loads correctly."""
    audio, sr = silence_audio
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == 16000
    assert len(audio) > 0
    assert np.allclose(audio, 0.0)  # All zeros


def test_noisy_fixture(noisy_audio):
    """Test noisy.wav fixture loads correctly."""
    audio, sr = noisy_audio
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == 16000
    assert len(audio) > 0
    assert len(audio) / sr == pytest.approx(3.0, abs=0.1)  # ~3 seconds


def test_short_speech_fixture(short_speech_audio):
    """Test short_speech.wav fixture loads correctly."""
    audio, sr = short_speech_audio
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == 16000
    assert len(audio) > 0
    assert len(audio) / sr == pytest.approx(1.0, abs=0.1)  # ~1 second


def test_long_speech_fixture(long_speech_audio):
    """Test long_speech.wav fixture loads correctly."""
    audio, sr = long_speech_audio
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == 16000
    assert len(audio) > 0
    assert len(audio) / sr == pytest.approx(5.0, abs=0.1)  # ~5 seconds


def test_white_noise_fixture(white_noise_audio):
    """Test white_noise.wav fixture loads correctly."""
    audio, sr = white_noise_audio
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == 16000
    assert len(audio) > 0
    assert not np.allclose(audio, 0.0)  # Not all zeros


def test_all_fixtures(all_audio_fixtures):
    """Test that all fixtures are loaded."""
    expected_fixtures = {
        "hello_world",
        "silence",
        "noisy",
        "short_speech",
        "long_speech",
        "white_noise",
    }

    loaded_fixtures = set(all_audio_fixtures.keys())
    assert loaded_fixtures == expected_fixtures, f"Missing: {expected_fixtures - loaded_fixtures}"

    # Verify each fixture
    for name, (audio, sr) in all_audio_fixtures.items():
        assert isinstance(audio, np.ndarray), f"{name} audio is not ndarray"
        assert audio.dtype == np.float32, f"{name} audio dtype is not float32"
        assert sr == 16000, f"{name} sample rate is not 16000"
        assert len(audio) > 0, f"{name} audio is empty"


def test_fixture_files_exist():
    """Test that fixture files were created on disk."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "audio"

    expected_files = [
        "hello_world.wav",
        "silence.wav",
        "noisy.wav",
        "short_speech.wav",
        "long_speech.wav",
        "white_noise.wav",
    ]

    for filename in expected_files:
        filepath = fixtures_dir / filename
        assert filepath.exists(), f"Fixture file missing: {filepath}"
        assert filepath.stat().st_size > 0, f"Fixture file is empty: {filepath}"


def test_default_config(default_whisper_config):
    """Test default Whisper configuration fixture."""
    assert isinstance(default_whisper_config, dict)
    assert "model_name" in default_whisper_config
    assert "device" in default_whisper_config
    assert "sample_rate" in default_whisper_config
    assert default_whisper_config["sample_rate"] == 16000


def test_hardware_detection(has_openvino, has_gpu, device_type):
    """Test hardware detection fixtures."""
    assert isinstance(has_openvino, bool)
    assert isinstance(has_gpu, bool)
    assert device_type in ["openvino", "cuda", "cpu"]

    # Device type should match hardware availability
    if has_openvino:
        assert device_type == "openvino"
    elif has_gpu:
        assert device_type == "cuda"
    else:
        assert device_type == "cpu"


def test_temp_audio_dir(temp_audio_dir):
    """Test temporary audio directory fixture."""
    assert temp_audio_dir.exists()
    assert temp_audio_dir.is_dir()
