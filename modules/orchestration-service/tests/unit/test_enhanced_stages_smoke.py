#!/usr/bin/env python3
"""
Smoke tests for enhanced audio processing stages.

These tests verify that enhanced stages can be imported, instantiated,
and perform basic processing without the enhanced libraries installed.

If libraries are available, more comprehensive tests are run.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from audio.config import (
    LUFSNormalizationConfig,
    LUFSNormalizationMode,
    CompressionConfig,
    CompressionMode,
    LimiterConfig,
)


class TestEnhancedStagesAvailability:
    """Test that enhanced stages can be imported based on available libraries."""

    def test_stages_enhanced_module_imports(self):
        """Test that the stages_enhanced module can be imported."""
        try:
            from audio import stages_enhanced

            assert hasattr(stages_enhanced, "AVAILABLE_FEATURES")
            assert hasattr(stages_enhanced, "__version__")
            print(
                f"✓ stages_enhanced module imported (version {stages_enhanced.__version__})"
            )
        except ImportError as e:
            pytest.fail(f"Could not import stages_enhanced module: {e}")

    def test_feature_flags(self):
        """Test that feature flags are properly set."""
        from audio import stages_enhanced

        features = stages_enhanced.AVAILABLE_FEATURES
        assert isinstance(features, dict)
        assert "lufs_normalization" in features
        assert "compression" in features
        assert "limiter" in features

        print(f"\nAvailable enhanced features:")
        for feature, available in features.items():
            status = "✓" if available else "✗"
            print(f"  {status} {feature}: {available}")


class TestLUFSNormalizationEnhanced:
    """Tests for enhanced LUFS normalization stage."""

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio for testing (1 second of 440Hz sine wave)."""
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32)

    @pytest.fixture
    def lufs_config(self):
        """Create default LUFS config."""
        return LUFSNormalizationConfig(
            enabled=True,
            mode=LUFSNormalizationMode.STREAMING,
            target_lufs=-14.0,
            true_peak_limiting=True,
        )

    def test_lufs_import(self):
        """Test that LUFS stage can be imported if library is available."""
        try:
            from audio.stages_enhanced import LUFSNormalizationStageEnhanced

            print("✓ LUFSNormalizationStageEnhanced imported successfully")
            return True
        except ImportError as e:
            print(f"✗ LUFSNormalizationStageEnhanced not available: {e}")
            pytest.skip("pyloudnorm not installed")
            return False

    def test_lufs_initialization(self, lufs_config):
        """Test that LUFS stage can be initialized."""
        try:
            from audio.stages_enhanced import LUFSNormalizationStageEnhanced

            stage = LUFSNormalizationStageEnhanced(lufs_config, sample_rate=16000)
            assert stage.stage_name == "lufs_normalization_enhanced"
            assert stage.is_initialized
            print("✓ LUFS stage initialized successfully")
        except ImportError:
            pytest.skip("pyloudnorm not installed")

    def test_lufs_processing(self, lufs_config, sample_audio):
        """Test basic LUFS processing."""
        try:
            from audio.stages_enhanced import LUFSNormalizationStageEnhanced

            stage = LUFSNormalizationStageEnhanced(lufs_config, sample_rate=16000)

            # Process audio
            result = stage.process(sample_audio)

            # Verify result structure
            assert result.stage_name == "lufs_normalization_enhanced"
            assert result.processed_audio is not None
            assert len(result.processed_audio) == len(sample_audio)
            assert "output_lufs" in result.metadata
            assert "gain_applied_db" in result.metadata

            print(f"✓ LUFS processing successful")
            print(f"  Input LUFS: {result.metadata['input_lufs']:.1f}")
            print(f"  Output LUFS: {result.metadata['output_lufs']:.1f}")
            print(f"  Gain applied: {result.metadata['gain_applied_db']:.1f} dB")

        except ImportError:
            pytest.skip("pyloudnorm not installed")


class TestCompressionEnhanced:
    """Tests for enhanced compression stage."""

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio with dynamic range."""
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create audio with varying amplitude
        audio = np.sin(2 * np.pi * 440 * t) * (0.3 + 0.7 * np.sin(2 * np.pi * 2 * t))
        return audio.astype(np.float32)

    @pytest.fixture
    def comp_config(self):
        """Create default compression config."""
        return CompressionConfig(
            enabled=True,
            mode=CompressionMode.SOFT_KNEE,
            threshold=-20,
            ratio=3.0,
            attack_time=5.0,
            release_time=100.0,
        )

    def test_compression_import(self):
        """Test that compression stage can be imported."""
        try:
            from audio.stages_enhanced import CompressionStageEnhanced

            print("✓ CompressionStageEnhanced imported successfully")
            return True
        except ImportError as e:
            print(f"✗ CompressionStageEnhanced not available: {e}")
            pytest.skip("pedalboard not installed")
            return False

    def test_compression_initialization(self, comp_config):
        """Test that compression stage can be initialized."""
        try:
            from audio.stages_enhanced import CompressionStageEnhanced

            stage = CompressionStageEnhanced(comp_config, sample_rate=16000)
            assert stage.stage_name == "compression_enhanced"
            assert stage.is_initialized
            print("✓ Compression stage initialized successfully")
        except ImportError:
            pytest.skip("pedalboard not installed")

    def test_compression_processing(self, comp_config, sample_audio):
        """Test basic compression processing."""
        try:
            from audio.stages_enhanced import CompressionStageEnhanced

            stage = CompressionStageEnhanced(comp_config, sample_rate=16000)

            # Process audio
            result = stage.process(sample_audio)

            # Verify result structure
            assert result.stage_name == "compression_enhanced"
            assert result.processed_audio is not None
            assert len(result.processed_audio) == len(sample_audio)
            assert "gain_reduction_db" in result.metadata
            assert "threshold_db" in result.metadata

            print(f"✓ Compression processing successful")
            print(f"  Gain reduction: {result.metadata['gain_reduction_db']:.1f} dB")
            print(f"  Threshold: {result.metadata['threshold_db']:.1f} dB")

        except ImportError:
            pytest.skip("pedalboard not installed")


class TestLimiterEnhanced:
    """Tests for enhanced limiter stage."""

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio with peaks."""
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create audio with some peaks
        audio = np.sin(2 * np.pi * 440 * t)
        # Add occasional peaks
        peaks = np.random.rand(len(t)) > 0.95
        audio[peaks] *= 2.0
        return audio.astype(np.float32)

    @pytest.fixture
    def limiter_config(self):
        """Create default limiter config."""
        return LimiterConfig(
            enabled=True, threshold=-1.0, release_time=50.0, soft_clip=True
        )

    def test_limiter_import(self):
        """Test that limiter stage can be imported."""
        try:
            from audio.stages_enhanced import LimiterStageEnhanced

            print("✓ LimiterStageEnhanced imported successfully")
            return True
        except ImportError as e:
            print(f"✗ LimiterStageEnhanced not available: {e}")
            pytest.skip("pedalboard not installed")
            return False

    def test_limiter_initialization(self, limiter_config):
        """Test that limiter stage can be initialized."""
        try:
            from audio.stages_enhanced import LimiterStageEnhanced

            stage = LimiterStageEnhanced(limiter_config, sample_rate=16000)
            assert stage.stage_name == "limiter_enhanced"
            assert stage.is_initialized
            print("✓ Limiter stage initialized successfully")
        except ImportError:
            pytest.skip("pedalboard not installed")

    def test_limiter_processing(self, limiter_config, sample_audio):
        """Test basic limiter processing."""
        try:
            from audio.stages_enhanced import LimiterStageEnhanced

            stage = LimiterStageEnhanced(limiter_config, sample_rate=16000)

            # Process audio
            result = stage.process(sample_audio)

            # Verify result structure
            assert result.stage_name == "limiter_enhanced"
            assert result.processed_audio is not None
            assert len(result.processed_audio) == len(sample_audio)
            assert "output_peak_db" in result.metadata
            assert "limiting_engaged" in result.metadata

            print(f"✓ Limiter processing successful")
            print(f"  Output peak: {result.metadata['output_peak_db']:.1f} dB")
            print(f"  Limiting engaged: {result.metadata['limiting_engaged']}")

        except ImportError:
            pytest.skip("pedalboard not installed")


class TestIntegration:
    """Integration tests for enhanced stages pipeline."""

    @pytest.fixture
    def sample_audio(self):
        """Generate realistic test audio."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Voice-like signal with varying amplitude
        fundamental = np.sin(2 * np.pi * 200 * t)
        harmonic1 = 0.5 * np.sin(2 * np.pi * 400 * t)
        harmonic2 = 0.3 * np.sin(2 * np.pi * 600 * t)
        envelope = 0.3 + 0.7 * np.sin(2 * np.pi * 3 * t)
        audio = (fundamental + harmonic1 + harmonic2) * envelope
        return audio.astype(np.float32)

    def test_full_pipeline(self, sample_audio):
        """Test processing through all enhanced stages."""
        try:
            from audio.stages_enhanced import (
                LUFSNormalizationStageEnhanced,
                CompressionStageEnhanced,
                LimiterStageEnhanced,
            )

            # Create stages
            lufs_stage = LUFSNormalizationStageEnhanced(
                LUFSNormalizationConfig(enabled=True, target_lufs=-16.0),
                sample_rate=16000,
            )
            comp_stage = CompressionStageEnhanced(
                CompressionConfig(enabled=True, threshold=-20, ratio=3.0),
                sample_rate=16000,
            )
            limiter_stage = LimiterStageEnhanced(
                LimiterConfig(enabled=True, threshold=-1.0), sample_rate=16000
            )

            # Process through pipeline
            result1 = lufs_stage.process(sample_audio)
            result2 = comp_stage.process(result1.processed_audio)
            result3 = limiter_stage.process(result2.processed_audio)

            # Verify final output
            assert result3.processed_audio is not None
            assert len(result3.processed_audio) == len(sample_audio)

            print("✓ Full pipeline processing successful")
            print(
                f"  Total latency: {result1.processing_time_ms + result2.processing_time_ms + result3.processing_time_ms:.1f} ms"
            )

        except ImportError:
            pytest.skip("Enhanced libraries not installed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
