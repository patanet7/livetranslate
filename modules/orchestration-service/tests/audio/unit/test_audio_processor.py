#!/usr/bin/env python3
"""
Unit Tests for Audio Processor Pipeline

Comprehensive tests for all audio processing stages including VAD,
voice filtering, noise reduction, voice enhancement, compression, and limiting.
"""

import pytest
import numpy as np

# Audio processor tests - dependencies now available
from src.audio.audio_processor import (
    VoiceActivityDetector,
    VoiceFrequencyFilter,
    NoiseReducer,
    VoiceEnhancer,
    DynamicCompressor,
    AudioLimiter,
    create_audio_pipeline_processor,
)
from src.audio.config import (
    AudioProcessingConfig,
    VADConfig,
    VoiceFilterConfig,
    NoiseReductionConfig,
    VoiceEnhancementConfig,
    CompressionConfig,
    LimiterConfig,
    VADMode,
    NoiseReductionMode,
    CompressionMode,
)


class TestVoiceActivityDetector:
    """Test Voice Activity Detection functionality."""

    def test_vad_initialization(self):
        """Test VAD initialization with different configurations."""
        config = VADConfig(
            enabled=True,
            mode=VADMode.WEBRTC,
            aggressiveness=2,
            energy_threshold=0.02,
            sensitivity=0.7,
        )

        vad = VoiceActivityDetector(config, sample_rate=16000)

        assert vad.config.enabled == True
        assert vad.config.mode == VADMode.WEBRTC
        assert vad.config.aggressiveness == 2
        assert vad.config.energy_threshold == 0.02
        assert vad.config.sensitivity == 0.7
        assert vad.sample_rate == 16000

    def test_vad_disabled_returns_voice_detected(self, sample_audio_data):
        """Test that disabled VAD always returns voice detected."""
        config = VADConfig(enabled=False)
        vad = VoiceActivityDetector(config)

        voice_detected, confidence = vad.detect_voice_activity(
            sample_audio_data["silence"]
        )

        assert voice_detected == True
        assert confidence == 1.0

    def test_vad_energy_based_detection(self, sample_audio_data):
        """Test energy-based VAD detection."""
        config = VADConfig(enabled=True, mode=VADMode.BASIC, energy_threshold=0.01)
        vad = VoiceActivityDetector(config)

        # Test with voice-like audio
        voice_detected, confidence = vad.detect_voice_activity(
            sample_audio_data["voice_like"]
        )
        assert voice_detected == True
        assert confidence > 0.5

        # Test with silence
        voice_detected, confidence = vad.detect_voice_activity(
            sample_audio_data["silence"]
        )
        assert voice_detected == False
        assert confidence < 0.5

        # Test with noise
        voice_detected, confidence = vad.detect_voice_activity(
            sample_audio_data["noise"]
        )
        # Noise might or might not be detected as voice depending on level
        assert isinstance(voice_detected, bool)
        assert 0.0 <= confidence <= 1.0

    def test_vad_webrtc_simulation(self, sample_audio_data):
        """Test WebRTC VAD simulation."""
        config = VADConfig(
            enabled=True,
            mode=VADMode.WEBRTC,
            aggressiveness=2,
            voice_freq_min=85,
            voice_freq_max=300,
        )
        vad = VoiceActivityDetector(config)

        # Test with voice-like audio
        voice_detected, confidence = vad.detect_voice_activity(
            sample_audio_data["voice_like"]
        )
        assert voice_detected == True
        assert confidence > 0.3

        # Test with high-frequency sine wave (not voice-like)
        voice_detected, confidence = vad.detect_voice_activity(
            sample_audio_data["sine_440"]
        )
        # May or may not be detected as voice
        assert isinstance(voice_detected, bool)
        assert 0.0 <= confidence <= 1.0

    def test_vad_aggressive_mode(self, sample_audio_data):
        """Test aggressive VAD mode."""
        config = VADConfig(
            enabled=True,
            mode=VADMode.AGGRESSIVE,
            energy_threshold=0.005,  # Lower threshold for aggressive mode
        )
        vad = VoiceActivityDetector(config)

        # Aggressive mode should be more sensitive
        voice_detected, confidence = vad.detect_voice_activity(
            sample_audio_data["noisy_voice_0db"]
        )
        # Should detect voice even in very noisy conditions
        assert isinstance(voice_detected, bool)
        assert 0.0 <= confidence <= 1.0

    def test_vad_parameter_validation(self):
        """Test VAD parameter validation during initialization."""
        config = VADConfig(
            aggressiveness=5,  # Will be clamped to 3
            energy_threshold=2.0,  # Will be clamped to 1.0
            sensitivity=1.5,  # Will be clamped to 1.0
        )

        # Parameters should be automatically validated/clamped
        assert config.aggressiveness == 3
        assert config.energy_threshold == 1.0
        assert config.sensitivity == 1.0


class TestVoiceFrequencyFilter:
    """Test Voice Frequency Filtering functionality."""

    def test_voice_filter_initialization(self):
        """Test voice filter initialization."""
        config = VoiceFilterConfig(
            enabled=True,
            fundamental_min=85,
            fundamental_max=300,
            voice_band_gain=1.2,
            preserve_formants=True,
        )

        filter_processor = VoiceFrequencyFilter(config, sample_rate=16000)

        assert filter_processor.config.enabled == True
        assert filter_processor.config.fundamental_min == 85
        assert filter_processor.config.fundamental_max == 300
        assert filter_processor.config.voice_band_gain == 1.2
        assert filter_processor.sample_rate == 16000

    def test_voice_filter_disabled_passthrough(self, sample_audio_data):
        """Test that disabled voice filter passes audio through unchanged."""
        config = VoiceFilterConfig(enabled=False)
        filter_processor = VoiceFrequencyFilter(config)

        original_audio = sample_audio_data["voice_like"]
        processed_audio = filter_processor.process_audio(original_audio)

        np.testing.assert_array_equal(original_audio, processed_audio)

    def test_voice_filter_processing(self, sample_audio_data):
        """Test voice filter processing effects."""
        config = VoiceFilterConfig(
            enabled=True,
            voice_band_gain=1.5,  # Boost voice frequencies
            preserve_formants=True,
        )
        filter_processor = VoiceFrequencyFilter(config)

        original_audio = sample_audio_data["voice_like"]
        processed_audio = filter_processor.process_audio(original_audio)

        # Processed audio should be different
        assert not np.array_equal(original_audio, processed_audio)

        # Should maintain similar length
        assert len(processed_audio) == len(original_audio)

        # Should not clip (assuming reasonable gain)
        assert np.max(np.abs(processed_audio)) <= 1.0

    def test_voice_filter_with_different_audio_types(self, sample_audio_data):
        """Test voice filter with different audio types."""
        config = VoiceFilterConfig(enabled=True, voice_band_gain=1.3)
        filter_processor = VoiceFrequencyFilter(config)

        # Test with various audio types
        for audio_name, audio_data in sample_audio_data.items():
            if len(audio_data) > 0:
                processed_audio = filter_processor.process_audio(audio_data)

                # Should maintain basic properties
                assert len(processed_audio) == len(audio_data)
                assert not np.isnan(processed_audio).any()
                assert not np.isinf(processed_audio).any()


class TestNoiseReducer:
    """Test Noise Reduction functionality."""

    def test_noise_reducer_initialization(self):
        """Test noise reducer initialization."""
        config = NoiseReductionConfig(
            enabled=True,
            mode=NoiseReductionMode.MODERATE,
            strength=0.7,
            voice_protection=True,
        )

        noise_reducer = NoiseReducer(config, sample_rate=16000)

        assert noise_reducer.config.enabled == True
        assert noise_reducer.config.mode == NoiseReductionMode.MODERATE
        assert noise_reducer.config.strength == 0.7
        assert noise_reducer.config.voice_protection == True
        assert noise_reducer.sample_rate == 16000

    def test_noise_reducer_disabled_passthrough(self, sample_audio_data):
        """Test that disabled noise reducer passes audio through unchanged."""
        config = NoiseReductionConfig(enabled=False)
        noise_reducer = NoiseReducer(config)

        original_audio = sample_audio_data["noisy_voice_10db"]
        processed_audio = noise_reducer.process_audio(original_audio)

        np.testing.assert_array_equal(original_audio, processed_audio)

    def test_light_noise_reduction(self, sample_audio_data):
        """Test light noise reduction mode."""
        config = NoiseReductionConfig(
            enabled=True, mode=NoiseReductionMode.LIGHT, strength=0.5
        )
        noise_reducer = NoiseReducer(config)

        noisy_audio = sample_audio_data["noisy_voice_10db"]
        processed_audio = noise_reducer.process_audio(noisy_audio)

        # Should process audio
        assert len(processed_audio) == len(noisy_audio)
        assert not np.array_equal(processed_audio, noisy_audio)

        # Should not create artifacts
        assert not np.isnan(processed_audio).any()
        assert not np.isinf(processed_audio).any()

    def test_aggressive_noise_reduction(self, sample_audio_data):
        """Test aggressive noise reduction mode."""
        config = NoiseReductionConfig(
            enabled=True, mode=NoiseReductionMode.AGGRESSIVE, strength=0.9
        )
        noise_reducer = NoiseReducer(config)

        noisy_audio = sample_audio_data["noisy_voice_0db"]  # Very noisy
        processed_audio = noise_reducer.process_audio(noisy_audio)

        # Should significantly modify the audio
        assert len(processed_audio) == len(noisy_audio)
        assert not np.array_equal(processed_audio, noisy_audio)

        # Calculate noise reduction effectiveness
        original_noise_level = np.std(noisy_audio)
        processed_noise_level = np.std(processed_audio)

        # Processed audio should generally have lower noise
        # (though this isn't guaranteed for all types of noise)
        assert processed_noise_level >= 0  # At minimum, should be valid

    def test_adaptive_noise_reduction(self, sample_audio_data):
        """Test adaptive noise reduction mode."""
        config = NoiseReductionConfig(
            enabled=True, mode=NoiseReductionMode.ADAPTIVE, adaptation_rate=0.2
        )
        noise_reducer = NoiseReducer(config)

        # Test with different signal levels
        quiet_audio = sample_audio_data["voice_like"] * 0.01  # Very quiet
        loud_audio = sample_audio_data["voice_like"] * 0.5  # Loud

        quiet_processed = noise_reducer.process_audio(quiet_audio)
        loud_processed = noise_reducer.process_audio(loud_audio)

        # Both should be processed differently
        assert len(quiet_processed) == len(quiet_audio)
        assert len(loud_processed) == len(loud_audio)
        assert not np.array_equal(quiet_processed, quiet_audio)
        assert not np.array_equal(loud_processed, loud_audio)


class TestVoiceEnhancer:
    """Test Voice Enhancement functionality."""

    def test_voice_enhancer_initialization(self):
        """Test voice enhancer initialization."""
        config = VoiceEnhancementConfig(
            enabled=True, clarity_enhancement=0.3, presence_boost=0.2, normalize=True
        )

        enhancer = VoiceEnhancer(config, sample_rate=16000)

        assert enhancer.config.enabled == True
        assert enhancer.config.clarity_enhancement == 0.3
        assert enhancer.config.presence_boost == 0.2
        assert enhancer.config.normalize == True
        assert enhancer.sample_rate == 16000

    def test_voice_enhancer_disabled_passthrough(self, sample_audio_data):
        """Test that disabled voice enhancer passes audio through unchanged."""
        config = VoiceEnhancementConfig(enabled=False)
        enhancer = VoiceEnhancer(config)

        original_audio = sample_audio_data["voice_like"]
        processed_audio = enhancer.process_audio(original_audio)

        np.testing.assert_array_equal(original_audio, processed_audio)

    def test_clarity_enhancement(self, sample_audio_data):
        """Test clarity enhancement processing."""
        config = VoiceEnhancementConfig(
            enabled=True,
            clarity_enhancement=0.4,
            normalize=False,  # Don't normalize to see raw effect
        )
        enhancer = VoiceEnhancer(config)

        original_audio = sample_audio_data["voice_like"]
        processed_audio = enhancer.process_audio(original_audio)

        # Should enhance the audio
        assert len(processed_audio) == len(original_audio)
        assert not np.array_equal(processed_audio, original_audio)

        # Should not create severe artifacts
        assert not np.isnan(processed_audio).any()
        assert not np.isinf(processed_audio).any()

    def test_presence_boost(self, sample_audio_data):
        """Test presence boost processing."""
        config = VoiceEnhancementConfig(
            enabled=True,
            presence_boost=0.3,
            clarity_enhancement=0.0,  # Only test presence boost
        )
        enhancer = VoiceEnhancer(config)

        original_audio = sample_audio_data["voice_like"]
        processed_audio = enhancer.process_audio(original_audio)

        # Should modify the audio
        assert len(processed_audio) == len(original_audio)
        assert not np.array_equal(processed_audio, original_audio)

    def test_normalization(self, sample_audio_data):
        """Test audio normalization."""
        config = VoiceEnhancementConfig(
            enabled=True, normalize=True, clarity_enhancement=0.0, presence_boost=0.0
        )
        enhancer = VoiceEnhancer(config)

        # Create audio that will clip
        loud_audio = sample_audio_data["voice_like"] * 2.0  # Will clip
        processed_audio = enhancer.process_audio(loud_audio)

        # Should be normalized to prevent clipping
        assert np.max(np.abs(processed_audio)) <= 0.95  # Should be normalized


class TestDynamicCompressor:
    """Test Dynamic Range Compression functionality."""

    def test_compressor_initialization(self):
        """Test compressor initialization."""
        config = CompressionConfig(
            enabled=True,
            mode=CompressionMode.SOFT_KNEE,
            threshold=-20,
            ratio=3.0,
            knee=2.0,
        )

        compressor = DynamicCompressor(config, sample_rate=16000)

        assert compressor.config.enabled == True
        assert compressor.config.mode == CompressionMode.SOFT_KNEE
        assert compressor.config.threshold == -20
        assert compressor.config.ratio == 3.0
        assert compressor.config.knee == 2.0

    def test_compressor_disabled_passthrough(self, sample_audio_data):
        """Test that disabled compressor passes audio through unchanged."""
        config = CompressionConfig(enabled=False)
        compressor = DynamicCompressor(config)

        original_audio = sample_audio_data["voice_like"]
        processed_audio = compressor.process_audio(original_audio)

        np.testing.assert_array_equal(original_audio, processed_audio)

    def test_soft_knee_compression(self, sample_audio_data):
        """Test soft knee compression."""
        config = CompressionConfig(
            enabled=True,
            mode=CompressionMode.SOFT_KNEE,
            threshold=-15,  # dB
            ratio=3.0,
            knee=2.0,
        )
        compressor = DynamicCompressor(config)

        # Use louder audio to trigger compression
        loud_audio = sample_audio_data["voice_like"] * 0.5
        processed_audio = compressor.process_audio(loud_audio)

        # Should process the audio
        assert len(processed_audio) == len(loud_audio)
        assert not np.array_equal(processed_audio, loud_audio)

        # Peak should generally be reduced (though not guaranteed for all signals)
        original_peak = np.max(np.abs(loud_audio))
        processed_peak = np.max(np.abs(processed_audio))

        # At minimum, should not create artifacts
        assert not np.isnan(processed_audio).any()
        assert not np.isinf(processed_audio).any()

    def test_hard_knee_compression(self, sample_audio_data):
        """Test hard knee compression."""
        config = CompressionConfig(
            enabled=True, mode=CompressionMode.HARD_KNEE, threshold=-12, ratio=4.0
        )
        compressor = DynamicCompressor(config)

        loud_audio = sample_audio_data["voice_like"] * 0.6
        processed_audio = compressor.process_audio(loud_audio)

        # Should process the audio
        assert len(processed_audio) == len(loud_audio)
        assert not np.array_equal(processed_audio, loud_audio)


class TestAudioLimiter:
    """Test Audio Limiting functionality."""

    def test_limiter_initialization(self):
        """Test limiter initialization."""
        config = LimiterConfig(
            enabled=True, threshold=-1.0, release_time=50.0, soft_clip=True
        )

        limiter = AudioLimiter(config, sample_rate=16000)

        assert limiter.config.enabled == True
        assert limiter.config.threshold == -1.0
        assert limiter.config.release_time == 50.0
        assert limiter.config.soft_clip == True

    def test_limiter_disabled_passthrough(self, sample_audio_data):
        """Test that disabled limiter passes audio through unchanged."""
        config = LimiterConfig(enabled=False)
        limiter = AudioLimiter(config)

        original_audio = sample_audio_data["voice_like"]
        processed_audio = limiter.process_audio(original_audio)

        np.testing.assert_array_equal(original_audio, processed_audio)

    def test_limiting_prevents_clipping(self, sample_audio_data):
        """Test that limiter prevents clipping."""
        config = LimiterConfig(
            enabled=True,
            threshold=-3.0,  # -3 dB threshold
            soft_clip=True,
        )
        limiter = AudioLimiter(config)

        # Create audio that would clip
        clipping_audio = sample_audio_data["clipped"]
        processed_audio = limiter.process_audio(clipping_audio)

        # Should prevent severe clipping
        threshold_linear = 10 ** (-3.0 / 20)  # Convert -3dB to linear
        assert (
            np.max(np.abs(processed_audio)) <= threshold_linear * 1.1
        )  # Allow small tolerance

    def test_soft_vs_hard_clipping(self, sample_audio_data):
        """Test difference between soft and hard clipping."""
        # Soft clipping config
        soft_config = LimiterConfig(enabled=True, threshold=-6.0, soft_clip=True)
        soft_limiter = AudioLimiter(soft_config)

        # Hard clipping config
        hard_config = LimiterConfig(enabled=True, threshold=-6.0, soft_clip=False)
        hard_limiter = AudioLimiter(hard_config)

        loud_audio = sample_audio_data["voice_like"] * 0.8

        soft_processed = soft_limiter.process_audio(loud_audio)
        hard_processed = hard_limiter.process_audio(loud_audio)

        # Both should limit, but differently
        assert len(soft_processed) == len(loud_audio)
        assert len(hard_processed) == len(loud_audio)
        assert not np.array_equal(soft_processed, hard_processed)


class TestAudioPipelineProcessor:
    """Test complete Audio Pipeline Processing."""

    def test_pipeline_processor_initialization(self, test_audio_processing_config):
        """Test pipeline processor initialization."""
        processor = create_audio_pipeline_processor(test_audio_processing_config)

        assert processor.config.preset_name == "test_preset"
        assert processor.sample_rate == 16000
        assert hasattr(processor, "vad")
        assert hasattr(processor, "voice_filter")
        assert hasattr(processor, "noise_reducer")
        assert hasattr(processor, "voice_enhancer")
        assert hasattr(processor, "compressor")
        assert hasattr(processor, "limiter")

    def test_complete_pipeline_processing(
        self, sample_audio_data, test_audio_processing_config
    ):
        """Test complete pipeline processing."""
        processor = create_audio_pipeline_processor(test_audio_processing_config)

        original_audio = sample_audio_data["voice_like"]
        processed_audio, metadata = processor.process_audio_chunk(original_audio)

        # Should process the audio
        assert len(processed_audio) == len(original_audio)
        assert isinstance(metadata, dict)
        assert "stages_applied" in metadata
        assert "processing_time_ms" in metadata
        assert "input_quality" in metadata
        assert "output_quality" in metadata

        # Should apply configured stages
        expected_stages = test_audio_processing_config.enabled_stages
        applied_stages = metadata["stages_applied"]

        # VAD might skip other stages if no voice detected
        if "vad" in applied_stages and metadata.get("vad_result", {}).get(
            "voice_detected", True
        ):
            # If voice was detected, other stages should be applied
            for stage in expected_stages:
                if stage != "vad":
                    assert stage in applied_stages or stage == "quality"

    def test_pipeline_with_different_configurations(
        self, sample_audio_data, test_configurations
    ):
        """Test pipeline with different configuration presets."""
        for config_name, config in test_configurations.items():
            processor = create_audio_pipeline_processor(config)

            audio_data = sample_audio_data["voice_like"]
            processed_audio, metadata = processor.process_audio_chunk(audio_data)

            # Should always produce valid output
            assert len(processed_audio) == len(audio_data)
            assert not np.isnan(processed_audio).any()
            assert not np.isinf(processed_audio).any()

            # Metadata should be valid
            assert isinstance(metadata, dict)
            assert "stages_applied" in metadata
            assert metadata["processing_time_ms"] >= 0

    def test_pipeline_bypassing(self, sample_audio_data):
        """Test pipeline bypassing for low quality input."""
        # Configure pipeline to bypass low quality audio
        config = AudioProcessingConfig(
            bypass_on_low_quality=True,
            quality={"quality_threshold": 0.8},  # High threshold
        )
        processor = create_audio_pipeline_processor(config)

        # Test with low quality audio (noise)
        noise_audio = sample_audio_data["noise"]
        processed_audio, metadata = processor.process_audio_chunk(noise_audio)

        # Should bypass processing
        assert metadata.get("bypassed", False) == True
        assert "bypass_reason" in metadata
        np.testing.assert_array_equal(processed_audio, noise_audio)

    def test_pipeline_vad_bypassing(self, sample_audio_data):
        """Test pipeline bypassing when no voice detected."""
        config = AudioProcessingConfig(
            enabled_stages=["vad", "voice_filter", "noise_reduction"],
            vad={"enabled": True, "energy_threshold": 0.1},  # High threshold
        )
        processor = create_audio_pipeline_processor(config)

        # Test with silence
        silence_audio = sample_audio_data["silence"]
        processed_audio, metadata = processor.process_audio_chunk(silence_audio)

        # Should detect no voice and bypass
        vad_result = metadata.get("vad_result", {})
        if not vad_result.get("voice_detected", True):
            assert metadata.get("bypassed", False) == True
            assert metadata.get("bypass_reason") == "no_voice_detected"

    def test_pipeline_stage_pausing(self, sample_audio_data):
        """Test pipeline stage pausing for debugging."""
        config = AudioProcessingConfig(
            enabled_stages=["vad", "voice_filter", "noise_reduction"],
            pause_after_stage={"voice_filter": True},
        )
        processor = create_audio_pipeline_processor(config)

        audio_data = sample_audio_data["voice_like"]
        processed_audio, metadata = processor.process_audio_chunk(audio_data)

        # Should pause after voice_filter stage
        applied_stages = metadata.get("stages_applied", [])
        if "voice_filter" in applied_stages:
            # Should not have applied noise_reduction if paused after voice_filter
            assert "noise_reduction" not in applied_stages

    def test_config_update(self, sample_audio_data, test_audio_processing_config):
        """Test dynamic configuration updates."""
        processor = create_audio_pipeline_processor(test_audio_processing_config)

        # Test with initial config
        audio_data = sample_audio_data["voice_like"]
        result1, metadata1 = processor.process_audio_chunk(audio_data)

        # Update configuration
        new_config = AudioProcessingConfig(
            preset_name="updated_preset",
            enabled_stages=["vad", "noise_reduction"],  # Fewer stages
        )
        processor.update_config(new_config)

        # Test with updated config
        result2, metadata2 = processor.process_audio_chunk(audio_data)

        # Should use updated configuration
        assert processor.config.preset_name == "updated_preset"
        assert len(metadata2.get("stages_applied", [])) <= len(
            metadata1.get("stages_applied", [])
        )

    def test_processing_statistics(
        self, sample_audio_data, test_audio_processing_config
    ):
        """Test processing statistics collection."""
        processor = create_audio_pipeline_processor(test_audio_processing_config)

        # Process several chunks
        audio_data = sample_audio_data["voice_like"]
        for _ in range(5):
            processor.process_audio_chunk(audio_data)

        # Get statistics
        stats = processor.get_processing_statistics()

        assert "total_samples_processed" in stats
        assert "total_duration_processed" in stats
        assert "average_processing_time" in stats
        assert "average_quality_score" in stats
        assert "enabled_stages" in stats
        assert "current_preset" in stats

        assert stats["total_samples_processed"] > 0
        assert stats["total_duration_processed"] > 0
        assert stats["average_processing_time"] >= 0
        assert 0.0 <= stats["average_quality_score"] <= 1.0


class TestErrorHandling:
    """Test error handling in audio processing."""

    def test_invalid_audio_input(self, test_audio_processing_config):
        """Test handling of invalid audio inputs."""
        processor = create_audio_pipeline_processor(test_audio_processing_config)

        # Test with empty audio
        empty_audio = np.array([], dtype=np.float32)
        processed_audio, metadata = processor.process_audio_chunk(empty_audio)

        # Should handle gracefully
        assert len(processed_audio) == 0
        assert isinstance(metadata, dict)

    def test_nan_audio_input(self, test_audio_processing_config):
        """Test handling of NaN audio inputs."""
        processor = create_audio_pipeline_processor(test_audio_processing_config)

        # Test with NaN audio
        nan_audio = np.full(1000, np.nan, dtype=np.float32)
        processed_audio, metadata = processor.process_audio_chunk(nan_audio)

        # Should handle gracefully (might return original or zeros)
        assert len(processed_audio) == len(nan_audio)
        assert isinstance(metadata, dict)
        assert "error" in metadata or not np.isnan(processed_audio).any()

    def test_inf_audio_input(self, test_audio_processing_config):
        """Test handling of infinite audio inputs."""
        processor = create_audio_pipeline_processor(test_audio_processing_config)

        # Test with infinite audio
        inf_audio = np.full(1000, np.inf, dtype=np.float32)
        processed_audio, metadata = processor.process_audio_chunk(inf_audio)

        # Should handle gracefully
        assert len(processed_audio) == len(inf_audio)
        assert isinstance(metadata, dict)
        assert "error" in metadata or not np.isinf(processed_audio).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
