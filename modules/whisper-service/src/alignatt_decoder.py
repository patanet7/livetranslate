#!/usr/bin/env python3
"""
AlignAtt Streaming Policy for Whisper Large-v3
Phase 2: SimulStreaming Innovation

Implements attention-guided streaming policy for reduced latency
Target: -30-50% latency reduction vs fixed chunking

Reference: SimulStreaming paper (IWSLT 2025) - Section 3.2
"We guide the decoder to attend to only the first l frames of the audio
by setting a frame threshold offset τ, where l = k - τ"
"""

import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlignAttState:
    """State tracking for incremental decoding"""
    max_audio_frame: int = 0
    current_frame_threshold: int = 0
    tokens_generated: int = 0
    last_attention_frame: int = 0
    is_continuation: bool = False

    def is_continuation_of(self, previous_state: 'AlignAttState') -> bool:
        """Check if this state continues from previous state"""
        return (
            self.is_continuation and
            self.max_audio_frame >= previous_state.max_audio_frame
        )


class AlignAttDecoder:
    """
    Attention-guided streaming decoder for Whisper

    From SimulStreaming paper (Section 3.2):
    "Our AlignAtt policy uses cross-attention to determine when to emit tokens.
    We guide the decoder to attend to only the first l frames by setting a
    frame threshold offset τ, where l = k - τ"

    Key concepts:
    - Frame threshold: Maximum audio frame the decoder can attend to
    - Frame offset (τ): Frames reserved for future streaming (default: 10)
    - Incremental decoding: Process audio as it arrives, emit tokens early

    Latency improvement targets:
    - Fixed chunking baseline: 200-500ms
    - AlignAtt target: <150ms (-30-50% latency)
    """

    def __init__(
        self,
        frame_threshold_offset: int = 10,
        enable_incremental: bool = True,
        enable_attention_masking: bool = True
    ):
        """
        Initialize AlignAtt decoder

        Args:
            frame_threshold_offset: Frames to reserve for streaming (τ in paper)
                                   Default: 10 frames (≈200ms at 50fps)
            enable_incremental: Enable incremental decoding
            enable_attention_masking: Enable attention mask enforcement
        """
        self.frame_threshold_offset = frame_threshold_offset
        self.enable_incremental = enable_incremental
        self.enable_attention_masking = enable_attention_masking

        # State tracking
        self.max_frame = 0
        self.current_state: Optional[AlignAttState] = None

        logger.info(f"AlignAttDecoder initialized: offset={frame_threshold_offset}, "
                   f"incremental={enable_incremental}, masking={enable_attention_masking}")

    def set_max_attention_frame(self, available_frames: int):
        """
        Set maximum frame the decoder can attend to

        Per SimulStreaming paper:
        l = k - τ
        where:
        - k = total available frames
        - τ = frame_threshold_offset
        - l = maximum frame for attention

        Args:
            available_frames: Total audio frames available (k)
        """
        self.max_frame = max(0, available_frames - self.frame_threshold_offset)

        logger.debug(f"[ALIGNATT] Max attention frame set: {self.max_frame} "
                    f"(available: {available_frames}, offset: {self.frame_threshold_offset})")

    def configure_for_pytorch(self, audio_features: np.ndarray) -> Dict[str, Any]:
        """
        Configure PyTorch Whisper for AlignAtt streaming

        Args:
            audio_features: Audio feature array or tensor

        Returns:
            Configuration dict for generation
        """
        try:
            # Calculate frame threshold from audio features
            if hasattr(audio_features, 'shape'):
                available_frames = audio_features.shape[1] if len(audio_features.shape) > 1 else len(audio_features)
            else:
                available_frames = len(audio_features) // 160  # 10ms frames at 16kHz

            self.set_max_attention_frame(available_frames)

            # Create configuration for PyTorch Whisper
            config = {
                "max_new_tokens": 448,
                "return_timestamps": True,
                "max_attention_frame": self.max_frame
            }

            logger.info(f"[ALIGNATT] Configured PyTorch for streaming (max_frame: {self.max_frame})")

            return config

        except Exception as e:
            logger.warning(f"Failed to configure PyTorch AlignAtt: {e}")
            return {}

    def create_attention_mask(
        self,
        audio_length: int,
        max_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Create attention mask for frame threshold enforcement

        From SimulStreaming paper:
        "We guide the decoder to attend to only the first l frames"

        Args:
            audio_length: Total audio length in frames
            max_length: Maximum length (uses max_frame if not specified)

        Returns:
            Boolean mask array (True = allowed, False = masked)
        """
        if not self.enable_attention_masking:
            return np.ones(audio_length, dtype=bool)

        threshold = max_length if max_length is not None else self.max_frame

        # Create mask: True for frames <= threshold, False for others
        mask = np.arange(audio_length) < threshold

        masked_count = np.sum(~mask)
        logger.debug(f"[ALIGNATT] Attention mask: {threshold}/{audio_length} frames allowed, "
                    f"{masked_count} frames masked")

        return mask

    def decode_incremental(
        self,
        audio_chunk: np.ndarray,
        previous_state: Optional[AlignAttState] = None
    ) -> AlignAttState:
        """
        Perform incremental decoding with AlignAtt policy

        Args:
            audio_chunk: New audio features
            previous_state: State from previous decoding step

        Returns:
            New decoder state
        """
        if not self.enable_incremental:
            logger.warning("[ALIGNATT] Incremental decoding disabled")
            return AlignAttState()

        # Calculate frames
        chunk_frames = len(audio_chunk)
        total_frames = chunk_frames

        if previous_state:
            total_frames += previous_state.max_audio_frame

        # Set frame threshold
        self.set_max_attention_frame(total_frames)

        # Create new state
        new_state = AlignAttState(
            max_audio_frame=total_frames,
            current_frame_threshold=self.max_frame,
            is_continuation=previous_state is not None
        )

        logger.info(f"[ALIGNATT] Incremental decode: total_frames={total_frames}, "
                   f"threshold={self.max_frame}, continuation={new_state.is_continuation}")

        return new_state

    def should_emit_token(
        self,
        current_attention_frame: int,
        confidence: float = 1.0,
        min_confidence: float = 0.7
    ) -> bool:
        """
        Determine if decoder should emit token based on attention

        Per SimulStreaming paper:
        "We emit tokens when the decoder's attention is within the allowed frame range"

        Args:
            current_attention_frame: Frame the decoder is currently attending to
            confidence: Token confidence score
            min_confidence: Minimum confidence required to emit

        Returns:
            True if token should be emitted
        """
        # Check if attention is within allowed range
        within_range = current_attention_frame <= self.max_frame

        # Check confidence threshold
        sufficient_confidence = confidence >= min_confidence

        should_emit = within_range and sufficient_confidence

        if not should_emit:
            logger.debug(f"[ALIGNATT] Token suppressed: frame={current_attention_frame}, "
                        f"max={self.max_frame}, conf={confidence:.2f}")

        return should_emit

    def calculate_latency_improvement(
        self,
        fixed_chunk_latency_ms: float,
        alignatt_latency_ms: float
    ) -> Dict[str, float]:
        """
        Calculate latency improvement over fixed chunking

        Args:
            fixed_chunk_latency_ms: Baseline latency with fixed chunking
            alignatt_latency_ms: Latency with AlignAtt

        Returns:
            Dictionary with improvement metrics
        """
        improvement_ms = fixed_chunk_latency_ms - alignatt_latency_ms
        improvement_percent = (improvement_ms / fixed_chunk_latency_ms) * 100 if fixed_chunk_latency_ms > 0 else 0

        return {
            "baseline_latency_ms": fixed_chunk_latency_ms,
            "alignatt_latency_ms": alignatt_latency_ms,
            "improvement_ms": improvement_ms,
            "improvement_percent": improvement_percent,
            "target_met": improvement_percent >= 30  # SimulStreaming target: -30-50%
        }

    def get_optimal_offset(self, audio_duration_s: float, target_latency_ms: float = 150) -> int:
        """
        Calculate optimal frame threshold offset for target latency

        Args:
            audio_duration_s: Audio duration in seconds
            target_latency_ms: Target latency in milliseconds

        Returns:
            Optimal frame offset (τ)
        """
        # Whisper uses 20ms frames (50 fps)
        frames_per_second = 50

        # Calculate frames for target latency
        latency_frames = int((target_latency_ms / 1000.0) * frames_per_second)

        # Ensure reasonable offset (5-20 frames)
        optimal_offset = max(5, min(20, latency_frames))

        logger.info(f"[ALIGNATT] Optimal offset: {optimal_offset} frames "
                   f"(target: {target_latency_ms}ms, audio: {audio_duration_s}s)")

        return optimal_offset

    def reset_state(self):
        """Reset decoder state for new session"""
        self.max_frame = 0
        self.current_state = None
        logger.debug("[ALIGNATT] Decoder state reset")


class AlignAttConfig:
    """Pre-defined AlignAtt configurations"""

    @staticmethod
    def ultra_low_latency() -> AlignAttDecoder:
        """Ultra-low latency mode (offset=5, ~100ms)"""
        return AlignAttDecoder(
            frame_threshold_offset=5,
            enable_incremental=True,
            enable_attention_masking=True
        )

    @staticmethod
    def low_latency() -> AlignAttDecoder:
        """Low latency mode (offset=10, ~200ms) - DEFAULT"""
        return AlignAttDecoder(
            frame_threshold_offset=10,
            enable_incremental=True,
            enable_attention_masking=True
        )

    @staticmethod
    def balanced() -> AlignAttDecoder:
        """Balanced mode (offset=15, ~300ms)"""
        return AlignAttDecoder(
            frame_threshold_offset=15,
            enable_incremental=True,
            enable_attention_masking=True
        )

    @staticmethod
    def quality_focused() -> AlignAttDecoder:
        """Quality-focused mode (offset=20, ~400ms)"""
        return AlignAttDecoder(
            frame_threshold_offset=20,
            enable_incremental=True,
            enable_attention_masking=True
        )

    @staticmethod
    def from_name(name: str) -> AlignAttDecoder:
        """Get config by name"""
        configs = {
            "ultra_low_latency": AlignAttConfig.ultra_low_latency,
            "low_latency": AlignAttConfig.low_latency,
            "balanced": AlignAttConfig.balanced,
            "quality_focused": AlignAttConfig.quality_focused
        }

        if name not in configs:
            logger.warning(f"Unknown AlignAtt config '{name}', using 'low_latency'")
            return AlignAttConfig.low_latency()

        return configs[name]()


# Convenience function
def create_alignatt_decoder(
    offset: Optional[int] = None,
    preset: Optional[str] = None
) -> AlignAttDecoder:
    """
    Create AlignAtt decoder with smart defaults

    Args:
        offset: Manual frame offset (overrides preset)
        preset: "ultra_low_latency", "low_latency", "balanced", "quality_focused"

    Returns:
        Configured AlignAttDecoder
    """
    if preset:
        decoder = AlignAttConfig.from_name(preset)
        if offset is not None:
            decoder.frame_threshold_offset = offset
        return decoder
    else:
        return AlignAttDecoder(frame_threshold_offset=offset or 10)


if __name__ == "__main__":
    # Test AlignAtt decoder
    print("AlignAtt Streaming Decoder - Phase 2 Implementation")
    print("=" * 60)

    # Test different configurations
    presets = ["ultra_low_latency", "low_latency", "balanced", "quality_focused"]

    for preset_name in presets:
        decoder = AlignAttConfig.from_name(preset_name)
        print(f"\n{preset_name.upper()}:")
        print(f"  Frame offset: {decoder.frame_threshold_offset}")
        print(f"  Latency: ~{decoder.frame_threshold_offset * 20}ms")

        # Test with 100 frames (2 seconds of audio)
        decoder.set_max_attention_frame(100)
        print(f"  Max attention frame: {decoder.max_frame}/100")

        # Create attention mask
        mask = decoder.create_attention_mask(100)
        print(f"  Masked frames: {np.sum(~mask)}/100")

    # Test latency calculation
    print("\n\nLATENCY IMPROVEMENT:")
    decoder = AlignAttConfig.low_latency()
    improvement = decoder.calculate_latency_improvement(
        fixed_chunk_latency_ms=350,
        alignatt_latency_ms=180
    )
    print(f"  Baseline: {improvement['baseline_latency_ms']}ms")
    print(f"  AlignAtt: {improvement['alignatt_latency_ms']}ms")
    print(f"  Improvement: {improvement['improvement_percent']:.1f}%")
    print(f"  Target met: {improvement['target_met']}")
