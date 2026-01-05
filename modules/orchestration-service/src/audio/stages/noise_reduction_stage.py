#!/usr/bin/env python3
"""
Noise Reduction Stage

Modular noise reduction implementation that can be used independently
or as part of the complete audio processing pipeline.
"""

import numpy as np
from scipy.fft import rfft, irfft
from typing import Dict, Any, Tuple
from ..stage_components import BaseAudioStage
from ..config import NoiseReductionConfig, NoiseReductionMode


class NoiseReductionStage(BaseAudioStage):
    """Noise reduction stage component."""

    def __init__(self, config: NoiseReductionConfig, sample_rate: int = 16000):
        super().__init__("noise_reduction", config, sample_rate)

        # Noise reduction state
        self.noise_profile = None
        self.noise_history = []
        self.adaptation_counter = 0

        # Parameters
        self.frame_size = 1024
        self.overlap = 0.5
        self.hop_size = int(self.frame_size * (1 - self.overlap))

        self.is_initialized = True

    def _process_audio(
        self, audio_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio through noise reduction."""
        try:
            # Apply input gain
            if abs(self.config.gain_in) > 0.1:
                input_gain_linear = 10 ** (self.config.gain_in / 20)
                processed_audio = audio_data * input_gain_linear
            else:
                processed_audio = audio_data.copy()

            # Apply noise reduction based on mode
            if not self.config.enabled:
                processed = processed_audio
            elif self.config.mode == NoiseReductionMode.LIGHT:
                processed = self._light_noise_reduction(processed_audio)
            elif self.config.mode == NoiseReductionMode.MODERATE:
                processed = self._moderate_noise_reduction(processed_audio)
            elif self.config.mode == NoiseReductionMode.AGGRESSIVE:
                processed = self._aggressive_noise_reduction(processed_audio)
            elif self.config.mode == NoiseReductionMode.ADAPTIVE:
                processed = self._adaptive_noise_reduction(processed_audio)
            else:
                processed = processed_audio

            # Apply output gain
            if abs(self.config.gain_out) > 0.1:
                output_gain_linear = 10 ** (self.config.gain_out / 20)
                processed = processed * output_gain_linear

            # Calculate noise reduction metrics
            input_noise_floor = self._estimate_noise_floor(audio_data)
            output_noise_floor = self._estimate_noise_floor(processed)
            noise_reduction_db = 20 * np.log10(
                input_noise_floor / max(output_noise_floor, 1e-10)
            )

            metadata = {
                "mode": self.config.mode.value,
                "strength": self.config.strength,
                "noise_reduction_db": noise_reduction_db,
                "input_noise_floor": input_noise_floor,
                "output_noise_floor": output_noise_floor,
                "voice_protection": self.config.voice_protection,
                "adaptation_counter": self.adaptation_counter,
                "noise_profile_available": self.noise_profile is not None,
                "gain_in_db": self.config.gain_in,
                "gain_out_db": self.config.gain_out,
                "enabled": self.config.enabled,
            }

            return processed, metadata

        except Exception as e:
            raise Exception(f"Noise reduction failed: {e}")

    def _light_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Light noise reduction using simple spectral subtraction."""
        return self._spectral_subtraction(audio_data, alpha=0.3)

    def _moderate_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Moderate noise reduction with voice protection."""
        return self._spectral_subtraction(audio_data, alpha=0.7, voice_protection=True)

    def _aggressive_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Aggressive noise reduction with maximum suppression."""
        return self._spectral_subtraction(audio_data, alpha=0.95, voice_protection=True)

    def _adaptive_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Adaptive noise reduction that adjusts based on signal characteristics."""
        # Analyze signal characteristics
        signal_energy = np.mean(audio_data**2)
        signal_variance = np.var(audio_data)

        # Adapt strength based on signal characteristics
        if signal_variance < 0.01:
            # Low variance - likely steady signal, use moderate reduction
            alpha = 0.5
        elif signal_energy < 0.001:
            # Low energy - likely noise, use aggressive reduction
            alpha = 0.9
        else:
            # Normal signal - use standard reduction
            alpha = 0.7

        return self._spectral_subtraction(
            audio_data, alpha=alpha, voice_protection=True
        )

    def _spectral_subtraction(
        self, audio_data: np.ndarray, alpha: float = 0.7, voice_protection: bool = False
    ) -> np.ndarray:
        """Spectral subtraction noise reduction."""
        if len(audio_data) < self.frame_size:
            return audio_data

        # Pad audio to ensure we can process full frames
        padded_length = len(audio_data) + self.frame_size
        padded_audio = np.pad(
            audio_data, (0, padded_length - len(audio_data)), mode="constant"
        )
        processed = np.zeros_like(padded_audio)

        # Process overlapping frames
        for i in range(0, len(padded_audio) - self.frame_size, self.hop_size):
            frame = padded_audio[i : i + self.frame_size]

            # Apply window
            windowed_frame = frame * np.hanning(self.frame_size)

            # FFT
            spectrum = rfft(windowed_frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Update noise profile
            self._update_noise_profile(magnitude)

            # Apply spectral subtraction
            if self.noise_profile is not None:
                # Calculate noise reduction - ensure it's an array matching magnitude shape
                noise_reduction_factor = alpha * self.config.strength

                # Apply voice protection if enabled
                if voice_protection and self.config.voice_protection:
                    # Protect voice frequencies (200-3000 Hz)
                    freqs = np.fft.rfftfreq(self.frame_size, 1 / self.sample_rate)
                    voice_mask = (freqs >= 200) & (freqs <= 3000)

                    # Create protection factor array matching magnitude shape
                    protection_factor = np.ones_like(magnitude)
                    protection_factor[voice_mask] *= (
                        0.5  # Reduce noise reduction in voice range
                    )

                    # Apply protection by modifying the reduction strength per frequency
                    final_reduction_factor = noise_reduction_factor * protection_factor
                else:
                    # No voice protection - use uniform reduction
                    final_reduction_factor = noise_reduction_factor

                # Spectral subtraction
                reduced_magnitude = (
                    magnitude - final_reduction_factor * self.noise_profile
                )

                # Ensure we don't over-subtract
                reduced_magnitude = np.maximum(reduced_magnitude, 0.1 * magnitude)

                # Reconstruct spectrum
                processed_spectrum = reduced_magnitude * np.exp(1j * phase)

                # IFFT
                processed_frame = irfft(processed_spectrum)

                # Apply window and overlap-add
                processed[i : i + self.frame_size] += processed_frame * np.hanning(
                    self.frame_size
                )
            else:
                # No noise profile yet, just add original frame
                processed[i : i + self.frame_size] += windowed_frame

        return processed[: len(audio_data)]

    def _update_noise_profile(self, magnitude: np.ndarray):
        """Update noise profile for spectral subtraction."""
        self.adaptation_counter += 1

        # Initialize noise profile
        if self.noise_profile is None:
            self.noise_profile = magnitude.copy()
            return

        # Adaptive update
        is_noise = np.mean(magnitude) < np.mean(self.noise_profile) * 1.5

        if is_noise or self.adaptation_counter % 10 == 0:  # Update periodically
            rate = self.config.adaptation_rate
            self.noise_profile = (1 - rate) * self.noise_profile + rate * magnitude

    def _estimate_noise_floor(self, audio_data: np.ndarray) -> float:
        """Estimate noise floor level."""
        if len(audio_data) == 0:
            return 0.0

        # Sort audio levels and take bottom 10% as noise floor estimate
        sorted_levels = np.sort(np.abs(audio_data))
        noise_floor_index = int(len(sorted_levels) * 0.1)
        return np.mean(sorted_levels[: max(1, noise_floor_index)])

    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "gain_in": self.config.gain_in,
            "gain_out": self.config.gain_out,
            "mode": self.config.mode.value,
            "strength": self.config.strength,
            "voice_protection": self.config.voice_protection,
            "stationary_noise_reduction": self.config.stationary_noise_reduction,
            "non_stationary_noise_reduction": self.config.non_stationary_noise_reduction,
            "noise_floor_db": self.config.noise_floor_db,
            "adaptation_rate": self.config.adaptation_rate,
        }

    def get_noise_profile(self) -> Dict[str, Any]:
        """Get current noise profile information."""
        return {
            "has_noise_profile": self.noise_profile is not None,
            "adaptation_counter": self.adaptation_counter,
            "noise_profile_shape": self.noise_profile.shape
            if self.noise_profile is not None
            else None,
            "noise_profile_mean": np.mean(self.noise_profile)
            if self.noise_profile is not None
            else 0.0,
            "noise_profile_std": np.std(self.noise_profile)
            if self.noise_profile is not None
            else 0.0,
        }

    def reset_noise_profile(self):
        """Reset the noise profile to start fresh."""
        self.noise_profile = None
        self.adaptation_counter = 0
        self.noise_history.clear()

    def reset_state(self):
        """Reset all noise reduction state."""
        self.reset_noise_profile()
