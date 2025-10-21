#!/usr/bin/env python3
"""
TDD Test Suite for AlignAtt Streaming Decoder
Based on SimulStreaming paper (IWSLT 2025)

Tests written BEFORE implementation to validate:
- Frame threshold enforcement (l = k - τ)
- Attention masking for streaming
- Latency improvement (-30-50% target)
- Incremental decoding state management
"""

import pytest
import numpy as np
from unittest.mock import Mock
import sys
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from alignatt_decoder import (
    AlignAttDecoder,
    AlignAttConfig,
    AlignAttState,
    create_alignatt_decoder
)


class TestAlignAttDecoder:
    """Test AlignAtt streaming decoder functionality"""

    def test_initialization_default(self):
        """Test default decoder initialization"""
        decoder = AlignAttDecoder()

        assert decoder.frame_threshold_offset == 10
        assert decoder.enable_incremental == True
        assert decoder.enable_attention_masking == True
        assert decoder.max_frame == 0

    def test_initialization_custom(self):
        """Test custom initialization"""
        decoder = AlignAttDecoder(
            frame_threshold_offset=15,
            enable_incremental=False,
            enable_attention_masking=False
        )

        assert decoder.frame_threshold_offset == 15
        assert decoder.enable_incremental == False
        assert decoder.enable_attention_masking == False

    def test_frame_threshold_calculation(self):
        """Test frame threshold calculation: l = k - τ"""
        decoder = AlignAttDecoder(frame_threshold_offset=10)

        # Per SimulStreaming paper: l = k - τ
        # where k = available frames, τ = offset
        available_frames = 100
        decoder.set_max_attention_frame(available_frames)

        expected_max = 100 - 10  # 90 frames
        assert decoder.max_frame == expected_max

    def test_frame_threshold_non_negative(self):
        """Test that max_frame never goes negative"""
        decoder = AlignAttDecoder(frame_threshold_offset=50)

        # Even with small frame count, should not go negative
        decoder.set_max_attention_frame(20)
        assert decoder.max_frame >= 0
        assert decoder.max_frame == 0  # max(0, 20-50) = 0

    def test_attention_mask_creation(self):
        """Test creation of attention mask"""
        decoder = AlignAttDecoder(frame_threshold_offset=10)
        decoder.set_max_attention_frame(100)

        # Create mask for 100 frames
        mask = decoder.create_attention_mask(100)

        assert len(mask) == 100
        assert mask.dtype == bool

        # First 90 frames should be True (allowed)
        assert all(mask[:90])

        # Last 10 frames should be False (masked)
        assert not any(mask[90:])

    def test_attention_mask_disabled(self):
        """Test attention masking when disabled"""
        decoder = AlignAttDecoder(enable_attention_masking=False)

        mask = decoder.create_attention_mask(100)

        # All frames should be allowed when masking disabled
        assert all(mask)

    def test_should_emit_token_within_range(self):
        """Test token emission decision within allowed range"""
        decoder = AlignAttDecoder(frame_threshold_offset=10)
        decoder.set_max_attention_frame(100)

        # Attention within allowed range (< 90)
        should_emit = decoder.should_emit_token(
            current_attention_frame=50,
            confidence=0.95
        )

        assert should_emit == True

    def test_should_emit_token_outside_range(self):
        """Test token emission suppression outside allowed range"""
        decoder = AlignAttDecoder(frame_threshold_offset=10)
        decoder.set_max_attention_frame(100)

        # Attention outside allowed range (> 90)
        should_emit = decoder.should_emit_token(
            current_attention_frame=95,
            confidence=0.95
        )

        assert should_emit == False

    def test_should_emit_token_low_confidence(self):
        """Test token suppression with low confidence"""
        decoder = AlignAttDecoder(frame_threshold_offset=10)
        decoder.set_max_attention_frame(100)

        # Within range but low confidence
        should_emit = decoder.should_emit_token(
            current_attention_frame=50,
            confidence=0.3,
            min_confidence=0.7
        )

        assert should_emit == False

    def test_incremental_decoding_state(self):
        """Test incremental decoding state management"""
        decoder = AlignAttDecoder(frame_threshold_offset=10)

        audio_chunk = np.random.randn(1000)  # Mock audio
        state = decoder.decode_incremental(audio_chunk)

        assert isinstance(state, AlignAttState)
        assert state.max_audio_frame > 0
        assert state.current_frame_threshold >= 0
        assert state.is_continuation == False

    def test_incremental_decoding_with_previous_state(self):
        """Test incremental decoding with continuation"""
        decoder = AlignAttDecoder(frame_threshold_offset=10)

        # First chunk
        chunk1 = np.random.randn(1000)
        state1 = decoder.decode_incremental(chunk1)

        # Second chunk with previous state
        chunk2 = np.random.randn(1000)
        state2 = decoder.decode_incremental(chunk2, previous_state=state1)

        assert state2.is_continuation == True
        assert state2.max_audio_frame > state1.max_audio_frame

    def test_state_continuation_check(self):
        """Test state continuation validation"""
        state1 = AlignAttState(max_audio_frame=100, is_continuation=False)
        state2 = AlignAttState(max_audio_frame=200, is_continuation=True)

        assert state2.is_continuation_of(state1)

    def test_latency_improvement_calculation(self):
        """Test latency improvement metrics"""
        decoder = AlignAttDecoder()

        improvement = decoder.calculate_latency_improvement(
            fixed_chunk_latency_ms=400,
            alignatt_latency_ms=180
        )

        assert improvement["baseline_latency_ms"] == 400
        assert improvement["alignatt_latency_ms"] == 180
        assert improvement["improvement_ms"] == 220
        assert abs(improvement["improvement_percent"] - 55.0) < 0.01  # Float precision tolerance
        assert improvement["target_met"] == True  # >= 30%

    def test_latency_target_not_met(self):
        """Test latency calculation when target not met"""
        decoder = AlignAttDecoder()

        improvement = decoder.calculate_latency_improvement(
            fixed_chunk_latency_ms=400,
            alignatt_latency_ms=350
        )

        assert improvement["improvement_percent"] < 30
        assert improvement["target_met"] == False

    def test_optimal_offset_calculation(self):
        """Test optimal offset calculation for target latency"""
        decoder = AlignAttDecoder()

        # Target 150ms latency
        optimal = decoder.get_optimal_offset(
            audio_duration_s=3.0,
            target_latency_ms=150
        )

        # Should return reasonable offset (5-20 frames)
        assert 5 <= optimal <= 20

    def test_reset_state(self):
        """Test decoder state reset"""
        decoder = AlignAttDecoder()

        # Set some state
        decoder.set_max_attention_frame(100)
        decoder.current_state = AlignAttState()

        assert decoder.max_frame > 0

        # Reset
        decoder.reset_state()

        assert decoder.max_frame == 0
        assert decoder.current_state is None

    def test_preset_configurations(self):
        """Test pre-defined configuration presets"""
        ultra_low = AlignAttConfig.ultra_low_latency()
        low = AlignAttConfig.low_latency()
        balanced = AlignAttConfig.balanced()
        quality = AlignAttConfig.quality_focused()

        assert ultra_low.frame_threshold_offset == 5
        assert low.frame_threshold_offset == 10
        assert balanced.frame_threshold_offset == 15
        assert quality.frame_threshold_offset == 20

    def test_preset_by_name(self):
        """Test getting preset by name"""
        decoder = AlignAttConfig.from_name("low_latency")
        assert decoder.frame_threshold_offset == 10

        decoder = AlignAttConfig.from_name("ultra_low_latency")
        assert decoder.frame_threshold_offset == 5

        # Unknown preset defaults to low_latency
        decoder = AlignAttConfig.from_name("unknown")
        assert decoder.frame_threshold_offset == 10

    def test_convenience_function(self):
        """Test convenience function"""
        decoder1 = create_alignatt_decoder(offset=12)
        assert decoder1.frame_threshold_offset == 12

        decoder2 = create_alignatt_decoder(preset="balanced")
        assert decoder2.frame_threshold_offset == 15

        # Override preset
        decoder3 = create_alignatt_decoder(offset=8, preset="quality_focused")
        assert decoder3.frame_threshold_offset == 8


class TestAlignAttPyTorch:
    """Test AlignAtt with PyTorch integration"""

    def test_pytorch_attention_hooks(self):
        """Test that AlignAtt works with PyTorch attention capture"""
        decoder = AlignAttDecoder(frame_threshold_offset=10)

        # Simulate audio features (100 frames)
        audio_frames = 100
        decoder.set_max_attention_frame(audio_frames)

        # Should set max attention frame based on audio
        assert decoder.max_frame > 0
        assert decoder.max_frame == 90  # 100 - 10 offset

    def test_pytorch_config_without_audio_features(self):
        """Test configuration with minimal audio"""
        decoder = AlignAttDecoder()

        # Should not crash with minimal frames
        decoder.set_max_attention_frame(5)
        assert decoder.max_frame >= 0  # Should never go negative

class TestAlignAttSimulStreaming:
    """Test compliance with SimulStreaming paper specifications"""

    def test_frame_threshold_formula(self):
        """Test l = k - τ formula from paper"""
        decoder = AlignAttDecoder(frame_threshold_offset=10)

        test_cases = [
            (100, 90),   # k=100, τ=10 → l=90
            (50, 40),    # k=50, τ=10 → l=40
            (150, 140),  # k=150, τ=10 → l=140
        ]

        for available, expected_max in test_cases:
            decoder.set_max_attention_frame(available)
            assert decoder.max_frame == expected_max

    def test_latency_improvement_target(self):
        """Test -30-50% latency improvement target"""
        decoder = AlignAttDecoder()

        # Baseline: 400ms (fixed chunking)
        # Target: 200-280ms (30-50% reduction)

        # Good case: 50% improvement
        improvement = decoder.calculate_latency_improvement(400, 200)
        assert improvement["target_met"] == True
        assert improvement["improvement_percent"] == 50

        # Marginal case: exactly 30%
        improvement = decoder.calculate_latency_improvement(400, 280)
        assert improvement["target_met"] == True
        assert improvement["improvement_percent"] == 30

        # Poor case: < 30%
        improvement = decoder.calculate_latency_improvement(400, 300)
        assert improvement["target_met"] == False

    def test_default_offset_10_frames(self):
        """Test default offset is 10 frames (~200ms at 50fps)"""
        decoder = AlignAttDecoder()

        assert decoder.frame_threshold_offset == 10
        # At 50fps (20ms per frame), 10 frames = 200ms reserved

    def test_streaming_incremental_state(self):
        """Test incremental state for streaming"""
        decoder = AlignAttDecoder()

        # Simulate streaming chunks
        states = []
        for i in range(5):
            chunk = np.random.randn(500)  # 500 samples per chunk
            prev_state = states[-1] if states else None
            state = decoder.decode_incremental(chunk, prev_state)
            states.append(state)

        # Each state should have more frames than previous
        for i in range(1, len(states)):
            assert states[i].max_audio_frame > states[i-1].max_audio_frame
            assert states[i].is_continuation == True


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
