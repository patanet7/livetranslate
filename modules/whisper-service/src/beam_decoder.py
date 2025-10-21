#!/usr/bin/env python3
"""
Beam Search Decoder for Whisper Large-v3
Phase 2: SimulStreaming Innovation

Implements beam search decoding for improved transcription quality.
Target: +20-30% quality improvement over greedy decoding

Reference: SimulStreaming paper (IWSLT 2025)
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A single hypothesis in beam search"""
    tokens: List[int]
    score: float
    text: str = ""
    length_normalized_score: float = 0.0

    def __post_init__(self):
        """Calculate length-normalized score"""
        if len(self.tokens) > 0:
            self.length_normalized_score = self.score / len(self.tokens)
        else:
            self.length_normalized_score = self.score


class BeamSearchDecoder:
    """
    Beam search decoder for Whisper models

    Beam search explores multiple decoding paths simultaneously,
    keeping the top-k most promising hypotheses at each step.
    This improves quality over greedy decoding (beam_size=1).

    Quality improvement targets (from SimulStreaming paper):
    - Beam size 1 (greedy): baseline
    - Beam size 3: +10-15% quality
    - Beam size 5: +20-30% quality (recommended)
    - Beam size 10: +25-35% quality (diminishing returns)
    """

    def __init__(
        self,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        temperature: float = 0.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = True
    ):
        """
        Initialize beam search decoder

        Args:
            beam_size: Number of hypotheses to keep (1=greedy, 5=default quality)
            length_penalty: Length normalization factor (>1 favors longer, <1 favors shorter)
            temperature: Sampling temperature (0.0=deterministic, >0 adds randomness)
            no_repeat_ngram_size: Prevent repeating n-grams (0=disabled)
            early_stopping: Stop when all beams end in EOS token
        """
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.early_stopping = early_stopping

        logger.info(f"BeamSearchDecoder initialized: beam_size={beam_size}, "
                   f"length_penalty={length_penalty}, temperature={temperature}")

        # Validate beam size
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if beam_size > 20:
            logger.warning(f"beam_size={beam_size} is very large and may cause memory issues. "
                         "Recommended: 1-10")

    def is_greedy_mode(self) -> bool:
        """Check if decoder is in greedy mode (beam_size=1)"""
        return self.beam_size == 1

    def configure_for_pytorch(self) -> Dict[str, Any]:
        """
        Configure PyTorch Whisper (openai-whisper) for beam search

        Returns:
            Dictionary of generation parameters for PyTorch Whisper
        """
        config = {
            "beam_size": self.beam_size,
            "best_of": self.beam_size,  # Number of candidates for beam search
            "patience": 1.0,  # Beam search patience
            "length_penalty": self.length_penalty,
            "temperature": self.temperature if self.temperature > 0.0 else 0.0,
        }

        logger.info(f"[BEAM_SEARCH] Configured PyTorch Whisper with beam_size={self.beam_size}")

        return config

    def configure_for_transformers(self) -> Dict[str, Any]:
        """
        Configure HuggingFace Transformers pipeline for beam search

        Returns:
            Dictionary of generation parameters for Transformers
        """
        config = {
            "num_beams": self.beam_size,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "return_timestamps": True
        }

        # Temperature (for sampling)
        if self.temperature > 0.0:
            config["do_sample"] = True
            config["temperature"] = self.temperature

        logger.info(f"[BEAM_SEARCH] Configured Transformers with beam_size={self.beam_size}")

        return config

    def rank_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Rank hypotheses by length-normalized score

        Args:
            hypotheses: List of beam search hypotheses

        Returns:
            Sorted hypotheses (best first)
        """
        # Apply length penalty
        for hyp in hypotheses:
            if self.length_penalty != 1.0:
                hyp.length_normalized_score = hyp.score / (len(hyp.tokens) ** self.length_penalty)

        # Sort by normalized score (descending)
        ranked = sorted(hypotheses, key=lambda h: h.length_normalized_score, reverse=True)

        return ranked

    def get_best_hypothesis(self, hypotheses: List[Hypothesis]) -> Hypothesis:
        """
        Get the best hypothesis from beam search results

        Args:
            hypotheses: List of final hypotheses

        Returns:
            Best hypothesis based on length-normalized score
        """
        ranked = self.rank_hypotheses(hypotheses)
        return ranked[0] if ranked else None

    def log_beam_statistics(self, hypotheses: List[Hypothesis]):
        """Log statistics about beam search results"""
        if not hypotheses:
            logger.warning("[BEAM_SEARCH] No hypotheses generated")
            return

        ranked = self.rank_hypotheses(hypotheses)

        logger.info(f"[BEAM_SEARCH] Generated {len(ranked)} hypotheses:")
        for i, hyp in enumerate(ranked[:3]):  # Show top 3
            logger.info(f"  Rank {i+1}: score={hyp.score:.3f}, "
                       f"norm_score={hyp.length_normalized_score:.3f}, "
                       f"tokens={len(hyp.tokens)}, "
                       f"text='{hyp.text[:50]}...'")


class BeamSearchConfig:
    """
    Pre-defined beam search configurations for different quality/speed tradeoffs
    """

    @staticmethod
    def fast() -> BeamSearchDecoder:
        """Fast mode: Greedy decoding (beam_size=1)"""
        return BeamSearchDecoder(
            beam_size=1,
            length_penalty=1.0,
            temperature=0.0
        )

    @staticmethod
    def balanced() -> BeamSearchDecoder:
        """Balanced mode: Good quality/speed tradeoff (beam_size=3)"""
        return BeamSearchDecoder(
            beam_size=3,
            length_penalty=1.0,
            temperature=0.0,
            no_repeat_ngram_size=3
        )

    @staticmethod
    def quality() -> BeamSearchDecoder:
        """Quality mode: Best quality (beam_size=5) - DEFAULT"""
        return BeamSearchDecoder(
            beam_size=5,
            length_penalty=1.0,
            temperature=0.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    @staticmethod
    def max_quality() -> BeamSearchDecoder:
        """Max quality mode: Maximum quality (beam_size=10)"""
        return BeamSearchDecoder(
            beam_size=10,
            length_penalty=1.0,
            temperature=0.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    @staticmethod
    def from_name(name: str) -> BeamSearchDecoder:
        """Get pre-defined config by name"""
        configs = {
            "fast": BeamSearchConfig.fast,
            "balanced": BeamSearchConfig.balanced,
            "quality": BeamSearchConfig.quality,
            "max_quality": BeamSearchConfig.max_quality
        }

        if name not in configs:
            logger.warning(f"Unknown beam search config '{name}', using 'quality'")
            return BeamSearchConfig.quality()

        return configs[name]()


# Convenience functions
def create_beam_decoder(
    beam_size: int = 5,
    quality_preset: Optional[str] = None
) -> BeamSearchDecoder:
    """
    Create beam search decoder with smart defaults

    Args:
        beam_size: Manual beam size (overrides preset)
        quality_preset: "fast", "balanced", "quality", "max_quality"

    Returns:
        Configured BeamSearchDecoder
    """
    if quality_preset:
        decoder = BeamSearchConfig.from_name(quality_preset)
        if beam_size != 5:  # Override beam size if specified
            decoder.beam_size = beam_size
        return decoder
    else:
        return BeamSearchDecoder(beam_size=beam_size)


if __name__ == "__main__":
    # Test beam search decoder creation
    print("Beam Search Decoder - Phase 2 Implementation")
    print("=" * 60)

    # Test different configurations
    configs = ["fast", "balanced", "quality", "max_quality"]

    for config_name in configs:
        decoder = BeamSearchConfig.from_name(config_name)
        print(f"\n{config_name.upper()} Configuration:")
        print(f"  Beam size: {decoder.beam_size}")
        print(f"  Length penalty: {decoder.length_penalty}")
        print(f"  Temperature: {decoder.temperature}")
        print(f"  Is greedy: {decoder.is_greedy_mode()}")

    # Test custom configuration
    print("\n\nCUSTOM Configuration:")
    custom_decoder = create_beam_decoder(beam_size=7)
    print(f"  Beam size: {custom_decoder.beam_size}")
