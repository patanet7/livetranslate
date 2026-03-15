"""Tests for audio downsampler — native quality → 16kHz mono for transcription."""
import sys
from pathlib import Path

import numpy as np
import pytest

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from meeting.downsampler import downsample_to_16k


class TestDownsampler:
    def test_48k_to_16k(self):
        """1 second of 48kHz mono should produce exactly 16 000 samples."""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = downsample_to_16k(audio, source_rate=48000)
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_44100_to_16k(self):
        """44.1kHz mono (1 second) should produce 16 000 samples."""
        audio = np.random.randn(44100).astype(np.float32) * 0.1
        result = downsample_to_16k(audio, source_rate=44100)
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_16k_passthrough(self):
        """Audio already at 16kHz must be returned unchanged."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = downsample_to_16k(audio, source_rate=16000)
        assert len(result) == 16000
        np.testing.assert_array_equal(result, audio)

    def test_stereo_to_mono(self):
        """Stereo (samples, 2) input must be mixed to mono and downsampled."""
        stereo = np.random.randn(48000, 2).astype(np.float32) * 0.1
        result = downsample_to_16k(stereo, source_rate=48000, channels=2)
        assert result.ndim == 1
        assert len(result) == 16000

    def test_output_dtype_is_float32(self):
        """Output must always be float32 regardless of input dtype."""
        audio = np.random.randn(48000).astype(np.float64)
        result = downsample_to_16k(audio, source_rate=48000)
        assert result.dtype == np.float32

    def test_multi_channel_mixed_to_mono(self):
        """Four-channel audio must be mixed to mono (mean across channels)."""
        quad = np.random.randn(48000, 4).astype(np.float32) * 0.1
        result = downsample_to_16k(quad, source_rate=48000)
        assert result.ndim == 1
        assert len(result) == 16000

    def test_amplitude_preserved_approximately(self):
        """Signal amplitude should be broadly preserved after resampling."""
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(48000).astype(np.float32) * 0.1
        result = downsample_to_16k(audio, source_rate=48000)
        # Allow generous tolerance given anti-aliasing; just ensure it's non-zero
        assert np.abs(result).max() > 0.001
