#!/usr/bin/env python3
"""
Auto Gain Control Stage

Modular AGC implementation that can be used independently
or as part of the complete audio processing pipeline.
"""

from typing import Any

import numpy as np

from ..config import AGCConfig, AGCMode
from ..stage_components import BaseAudioStage


class AGCStage(BaseAudioStage):
    """Auto Gain Control stage component."""

    def __init__(self, config: AGCConfig, sample_rate: int = 16000):
        super().__init__("agc", config, sample_rate)

        # AGC state
        self.current_gain = 1.0
        self.target_gain = 1.0
        self.signal_level = 0.0
        self.noise_gate_open = False
        self.hold_counter = 0

        # Convert time constants to samples
        self.attack_samples = max(1, int(config.attack_time * sample_rate / 1000))
        self.release_samples = max(1, int(config.release_time * sample_rate / 1000))
        self.hold_samples = int(config.hold_time * sample_rate / 1000)
        self.lookahead_samples = int(config.lookahead_time * sample_rate / 1000)

        # Convert dB to linear
        self.target_level_linear = 10 ** (config.target_level / 20)
        self.max_gain_linear = 10 ** (config.max_gain / 20)
        self.min_gain_linear = 10 ** (config.min_gain / 20)
        self.noise_gate_threshold_linear = 10 ** (config.noise_gate_threshold / 20)

        # Lookahead buffer
        self.lookahead_buffer = (
            np.zeros(self.lookahead_samples) if self.lookahead_samples > 0 else None
        )
        self.buffer_index = 0

        # Adaptation state
        self.adaptation_counter = 0
        self.level_history = []
        self.max_history_length = int(sample_rate * 0.1)  # 100ms history

        self.is_initialized = True

    def _process_audio(self, audio_data: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Process audio through AGC."""
        if self.config.mode == AGCMode.DISABLED:
            return audio_data, {"mode": "disabled", "gain_applied": 1.0}

        try:
            processed = np.zeros_like(audio_data)
            gain_values = []
            level_values = []

            for i, sample in enumerate(audio_data):
                # Calculate current signal level (RMS over recent samples)
                self.level_history.append(abs(sample))
                if len(self.level_history) > self.max_history_length:
                    self.level_history.pop(0)

                current_level = np.sqrt(np.mean(np.array(self.level_history) ** 2))
                level_values.append(current_level)

                # Noise gate
                if current_level > self.noise_gate_threshold_linear:
                    self.noise_gate_open = True
                elif current_level < self.noise_gate_threshold_linear * 0.5:  # Hysteresis
                    self.noise_gate_open = False

                if not self.noise_gate_open:
                    processed[i] = sample * self.current_gain
                    gain_values.append(self.current_gain)
                    continue

                # Calculate desired gain based on mode
                if self.config.mode == AGCMode.FAST:
                    desired_gain = self._calculate_gain_fast(current_level)
                elif self.config.mode == AGCMode.MEDIUM:
                    desired_gain = self._calculate_gain_medium(current_level)
                elif self.config.mode == AGCMode.SLOW:
                    desired_gain = self._calculate_gain_slow(current_level)
                elif self.config.mode == AGCMode.ADAPTIVE:
                    desired_gain = self._calculate_gain_adaptive(current_level)
                else:
                    desired_gain = self.current_gain

                # Apply gain limits
                desired_gain = max(self.min_gain_linear, min(self.max_gain_linear, desired_gain))

                # Apply attack/release with hold
                if desired_gain < self.current_gain:
                    # Attack (gain reduction)
                    self.current_gain = self._apply_attack(desired_gain)
                    self.hold_counter = self.hold_samples
                elif desired_gain > self.current_gain and self.hold_counter <= 0:
                    # Release (gain increase)
                    self.current_gain = self._apply_release(desired_gain)

                # Update hold counter
                if self.hold_counter > 0:
                    self.hold_counter -= 1

                # Apply lookahead if enabled
                if self.lookahead_buffer is not None:
                    delayed_sample = self.lookahead_buffer[self.buffer_index]
                    self.lookahead_buffer[self.buffer_index] = sample
                    self.buffer_index = (self.buffer_index + 1) % len(self.lookahead_buffer)
                    processed[i] = delayed_sample * self.current_gain
                else:
                    processed[i] = sample * self.current_gain

                gain_values.append(self.current_gain)

                # Update adaptation counter
                self.adaptation_counter += 1

            # Calculate metadata
            metadata = {
                "mode": self.config.mode.value,
                "final_gain": self.current_gain,
                "gain_applied_db": 20 * np.log10(self.current_gain)
                if self.current_gain > 0
                else -80,
                "average_level": np.mean(level_values) if level_values else 0.0,
                "noise_gate_open": self.noise_gate_open,
                "target_level_db": self.config.target_level,
                "gain_range_db": [
                    20 * np.log10(self.min_gain_linear),
                    20 * np.log10(self.max_gain_linear),
                ],
                "adaptation_counter": self.adaptation_counter,
            }

            return processed, metadata

        except Exception as e:
            raise Exception(f"AGC processing failed: {e}") from e

    def _calculate_gain_fast(self, level: float) -> float:
        """Calculate gain for fast mode."""
        if level < 1e-10:  # Avoid division by zero
            return self.current_gain

        return self.target_level_linear / level

    def _calculate_gain_medium(self, level: float) -> float:
        """Calculate gain for medium mode with smooth transitions."""
        if level < 1e-10:
            return self.current_gain

        target_gain = self.target_level_linear / level

        # Smooth transition using knee
        knee_factor = self.config.knee_width / 20  # Convert dB to linear factor
        if abs(target_gain - self.current_gain) > knee_factor:
            # Large change, use fast response
            return target_gain
        else:
            # Small change, use smooth response
            return self.current_gain + (target_gain - self.current_gain) * 0.1

    def _calculate_gain_slow(self, level: float) -> float:
        """Calculate gain for slow mode with gentle adjustments."""
        if level < 1e-10:
            return self.current_gain

        target_gain = self.target_level_linear / level

        # Very gradual adjustment
        adjustment_rate = 0.05
        return self.current_gain + (target_gain - self.current_gain) * adjustment_rate

    def _calculate_gain_adaptive(self, level: float) -> float:
        """Calculate gain for adaptive mode based on signal characteristics."""
        if level < 1e-10:
            return self.current_gain

        # Analyze signal characteristics
        level_variance = np.var(self.level_history) if len(self.level_history) > 10 else 0

        # Adaptive adjustment rate based on signal stability
        if level_variance < 0.01:
            # Stable signal - use slow adaptation
            adjustment_rate = 0.02
        elif level_variance < 0.1:
            # Moderate variation - use medium adaptation
            adjustment_rate = 0.1
        else:
            # High variation - use fast adaptation
            adjustment_rate = 0.3

        # Apply adaptation rate
        rate = self.config.adaptation_rate * adjustment_rate
        target_gain = self.target_level_linear / level

        return self.current_gain + (target_gain - self.current_gain) * rate

    def _apply_attack(self, desired_gain: float) -> float:
        """Apply attack time constant."""
        attack_coeff = 1.0 / self.attack_samples
        return self.current_gain + (desired_gain - self.current_gain) * attack_coeff

    def _apply_release(self, desired_gain: float) -> float:
        """Apply release time constant."""
        release_coeff = 1.0 / self.release_samples
        return self.current_gain + (desired_gain - self.current_gain) * release_coeff

    def _get_stage_config(self) -> dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "mode": self.config.mode.value,
            "target_level": self.config.target_level,
            "max_gain": self.config.max_gain,
            "min_gain": self.config.min_gain,
            "attack_time": self.config.attack_time,
            "release_time": self.config.release_time,
            "hold_time": self.config.hold_time,
            "knee_width": self.config.knee_width,
            "lookahead_time": self.config.lookahead_time,
            "adaptation_rate": self.config.adaptation_rate,
            "noise_gate_threshold": self.config.noise_gate_threshold,
        }

    def get_gain_state(self) -> dict[str, Any]:
        """Get current AGC gain state."""
        return {
            "current_gain": self.current_gain,
            "current_gain_db": 20 * np.log10(self.current_gain) if self.current_gain > 0 else -80,
            "target_level_linear": self.target_level_linear,
            "noise_gate_open": self.noise_gate_open,
            "hold_counter": self.hold_counter,
            "adaptation_counter": self.adaptation_counter,
            "level_history_length": len(self.level_history),
        }
