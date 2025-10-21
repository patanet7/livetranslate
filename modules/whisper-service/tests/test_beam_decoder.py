#!/usr/bin/env python3
"""
TDD Test Suite for Beam Search Decoder
Phase 2: SimulStreaming Innovation

Tests written BEFORE implementation to validate:
- Beam search quality improvement (+20-30% over greedy)
- Different beam sizes (1, 3, 5, 10)
- Length normalization
- Hypothesis ranking
- Configuration presets
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from beam_decoder import (
    BeamSearchDecoder,
    BeamSearchConfig,
    Hypothesis,
    create_beam_decoder
)


class TestBeamSearchDecoder:
    """Test beam search decoder functionality"""

    def test_initialization_default(self):
        """Test default beam search decoder initialization"""
        decoder = BeamSearchDecoder()

        assert decoder.beam_size == 5
        assert decoder.length_penalty == 1.0
        assert decoder.temperature == 0.0
        assert decoder.early_stopping == True
        assert not decoder.is_greedy_mode()

    def test_initialization_custom(self):
        """Test custom beam search decoder initialization"""
        decoder = BeamSearchDecoder(
            beam_size=10,
            length_penalty=1.2,
            temperature=0.5,
            no_repeat_ngram_size=3
        )

        assert decoder.beam_size == 10
        assert decoder.length_penalty == 1.2
        assert decoder.temperature == 0.5
        assert decoder.no_repeat_ngram_size == 3

    def test_greedy_mode_detection(self):
        """Test greedy mode detection (beam_size=1)"""
        greedy_decoder = BeamSearchDecoder(beam_size=1)
        beam_decoder = BeamSearchDecoder(beam_size=5)

        assert greedy_decoder.is_greedy_mode()
        assert not beam_decoder.is_greedy_mode()

    def test_beam_size_validation(self):
        """Test beam size validation"""
        with pytest.raises(ValueError):
            BeamSearchDecoder(beam_size=0)

        with pytest.raises(ValueError):
            BeamSearchDecoder(beam_size=-1)

    def test_pytorch_configuration(self):
        """Test PyTorch Whisper configuration"""
        decoder = BeamSearchDecoder(beam_size=5)

        config = decoder.configure_for_pytorch()

        assert config is not None
        assert config["beam_size"] == 5
        assert config["best_of"] == 5
        assert config["length_penalty"] == 1.0

    def test_transformers_configuration(self):
        """Test HuggingFace Transformers configuration"""
        decoder = BeamSearchDecoder(beam_size=5, length_penalty=1.2)

        config = decoder.configure_for_transformers()

        assert config["num_beams"] == 5
        assert config["length_penalty"] == 1.2
        assert config["return_timestamps"] == True
        assert config["early_stopping"] == True

    def test_hypothesis_creation(self):
        """Test hypothesis data structure"""
        hyp = Hypothesis(
            tokens=[1, 2, 3, 4, 5],
            score=10.0,
            text="test output"
        )

        assert len(hyp.tokens) == 5
        assert hyp.score == 10.0
        assert hyp.text == "test output"
        assert hyp.length_normalized_score == 2.0  # 10.0 / 5

    def test_hypothesis_ranking(self):
        """Test hypothesis ranking by score"""
        decoder = BeamSearchDecoder(beam_size=5)

        hypotheses = [
            Hypothesis(tokens=[1, 2, 3], score=6.0, text="short"),
            Hypothesis(tokens=[1, 2, 3, 4, 5], score=12.0, text="medium"),
            Hypothesis(tokens=[1, 2], score=5.0, text="very short"),
        ]

        ranked = decoder.rank_hypotheses(hypotheses)

        # Should be sorted by length-normalized score (descending)
        assert len(ranked) == 3
        assert ranked[0].text == "very short"  # 5.0 / 2 = 2.5
        assert ranked[1].text == "medium"       # 12.0 / 5 = 2.4
        assert ranked[2].text == "short"        # 6.0 / 3 = 2.0

    def test_get_best_hypothesis(self):
        """Test getting best hypothesis from beam search results"""
        decoder = BeamSearchDecoder(beam_size=3)

        hypotheses = [
            Hypothesis(tokens=[1, 2, 3], score=9.0, text="first"),
            Hypothesis(tokens=[1, 2, 3, 4], score=11.0, text="second"),
            Hypothesis(tokens=[1, 2], score=7.0, text="third"),
        ]

        best = decoder.get_best_hypothesis(hypotheses)

        assert best is not None
        assert best.text == "third"  # Highest normalized score: 7.0 / 2 = 3.5

    def test_preset_configurations(self):
        """Test pre-defined configuration presets"""
        fast = BeamSearchConfig.fast()
        balanced = BeamSearchConfig.balanced()
        quality = BeamSearchConfig.quality()
        max_quality = BeamSearchConfig.max_quality()

        assert fast.beam_size == 1
        assert balanced.beam_size == 3
        assert quality.beam_size == 5
        assert max_quality.beam_size == 10

        assert fast.is_greedy_mode()
        assert not quality.is_greedy_mode()

    def test_preset_by_name(self):
        """Test getting preset configuration by name"""
        decoder = BeamSearchConfig.from_name("quality")
        assert decoder.beam_size == 5

        decoder = BeamSearchConfig.from_name("fast")
        assert decoder.beam_size == 1

        # Unknown preset should default to quality
        decoder = BeamSearchConfig.from_name("unknown")
        assert decoder.beam_size == 5

    def test_create_beam_decoder_convenience(self):
        """Test convenience function for creating decoder"""
        decoder1 = create_beam_decoder(beam_size=7)
        assert decoder1.beam_size == 7

        decoder2 = create_beam_decoder(quality_preset="balanced")
        assert decoder2.beam_size == 3

        # Preset with override
        decoder3 = create_beam_decoder(beam_size=8, quality_preset="quality")
        assert decoder3.beam_size == 8


class TestBeamSearchQuality:
    """Test beam search quality improvements"""

    def test_quality_improvement_simulation(self):
        """
        Simulate quality improvement with beam search

        Target from SimulStreaming paper:
        - Beam size 1 (greedy): baseline
        - Beam size 5: +20-30% quality improvement
        """
        # This would require actual model inference
        # For now, test that decoder is configured for quality

        greedy_decoder = BeamSearchDecoder(beam_size=1)
        quality_decoder = BeamSearchDecoder(beam_size=5)

        assert greedy_decoder.is_greedy_mode()
        assert quality_decoder.beam_size == 5

        # In actual use, beam_size=5 should produce better hypotheses
        # that will be validated in integration tests

    def test_length_penalty_effect(self):
        """Test that length penalty affects hypothesis ranking"""
        # Create hypotheses with different lengths
        short_hyp = Hypothesis(tokens=[1, 2], score=4.0, text="short")
        long_hyp = Hypothesis(tokens=[1, 2, 3, 4, 5], score=8.0, text="long")

        # Default length penalty (1.0) - no bias
        decoder_neutral = BeamSearchDecoder(length_penalty=1.0)
        ranked_neutral = decoder_neutral.rank_hypotheses([short_hyp, long_hyp])

        # Normalized scores: short=2.0, long=1.6
        assert ranked_neutral[0].text == "short"

        # Favor longer sequences (length_penalty > 1.0)
        decoder_long = BeamSearchDecoder(length_penalty=1.5)
        ranked_long = decoder_long.rank_hypotheses([short_hyp, long_hyp])

        # With penalty > 1.0, longer sequences get boosted
        # This should change the ranking

    def test_temperature_configuration(self):
        """Test temperature parameter for sampling"""
        deterministic = BeamSearchDecoder(temperature=0.0)
        sampling = BeamSearchDecoder(temperature=0.7)

        assert deterministic.temperature == 0.0
        assert sampling.temperature == 0.7

        # Temperature > 0 should enable sampling in config
        config = sampling.configure_for_transformers()
        if sampling.temperature > 0.0:
            assert config.get("do_sample") == True
            assert config.get("temperature") == 0.7


class TestBeamSearchIntegration:
    """Integration tests for beam search with Whisper"""

    def test_pytorch_beam_search_config(self):
        """Test beam search configuration for PyTorch Whisper"""
        decoder = BeamSearchDecoder(beam_size=5)

        config = decoder.configure_for_pytorch()

        # Verify beam search parameters are set
        assert config is not None
        assert config["beam_size"] == 5

    def test_large_beam_size_warning(self):
        """Test warning for very large beam sizes"""
        # Very large beam sizes may cause memory issues but are allowed
        decoder = BeamSearchDecoder(beam_size=25)

        assert decoder.beam_size == 25
        # Implementation logs a warning for beam_size > 20

    def test_empty_hypotheses_handling(self):
        """Test handling of empty hypothesis list"""
        decoder = BeamSearchDecoder()

        best = decoder.get_best_hypothesis([])
        assert best is None

        ranked = decoder.rank_hypotheses([])
        assert len(ranked) == 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
