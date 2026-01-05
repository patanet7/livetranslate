#!/usr/bin/env python3
"""
Voice Enhancement Stage

Modular voice enhancement implementation that can be used independently
or as part of the complete audio processing pipeline.
"""

import numpy as np
import scipy.signal
from typing import Dict, Any, Tuple
from ..stage_components import BaseAudioStage
from ..config import VoiceEnhancementConfig


class VoiceEnhancementStage(BaseAudioStage):
    """Voice enhancement stage component."""

    def __init__(self, config: VoiceEnhancementConfig, sample_rate: int = 16000):
        super().__init__("voice_enhancement", config, sample_rate)

        # Design enhancement filters
        self._design_enhancement_filters()

        self.is_initialized = True

    def _design_enhancement_filters(self):
        """Design filters for voice enhancement."""
        nyquist = self.sample_rate / 2

        # Presence boost filter (2-5 kHz)
        if self.config.presence_boost > 0:
            presence_freq = [2000 / nyquist, 5000 / nyquist]
            presence_freq[1] = min(presence_freq[1], 0.99)
            self.presence_filter = scipy.signal.butter(2, presence_freq, btype="band")
        else:
            self.presence_filter = None

        # Warmth filter (low-mid frequencies)
        if abs(self.config.warmth_adjustment) > 0.01:
            warmth_freq = 500 / nyquist
            self.warmth_filter = scipy.signal.butter(2, warmth_freq, btype="low")
        else:
            self.warmth_filter = None

        # Brightness filter (high frequencies)
        if abs(self.config.brightness_adjustment) > 0.01:
            brightness_freq = 5000 / nyquist
            brightness_freq = min(brightness_freq, 0.99)
            self.brightness_filter = scipy.signal.butter(
                2, brightness_freq, btype="high"
            )
        else:
            self.brightness_filter = None

    def _process_audio(
        self, audio_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio through voice enhancement."""
        try:
            processed = audio_data.copy()
            enhancements_applied = []

            # Clarity enhancement using harmonic enhancement
            if self.config.clarity_enhancement > 0:
                processed = self._enhance_clarity(processed)
                enhancements_applied.append("clarity")

            # Presence boost
            if self.config.presence_boost > 0 and self.presence_filter is not None:
                presence_enhanced = scipy.signal.filtfilt(
                    self.presence_filter[0], self.presence_filter[1], processed
                )
                processed = processed + presence_enhanced * self.config.presence_boost
                enhancements_applied.append("presence")

            # Warmth adjustment
            if (
                abs(self.config.warmth_adjustment) > 0.01
                and self.warmth_filter is not None
            ):
                warmth_component = scipy.signal.filtfilt(
                    self.warmth_filter[0], self.warmth_filter[1], processed
                )
                processed = processed + warmth_component * self.config.warmth_adjustment
                enhancements_applied.append("warmth")

            # Brightness adjustment
            if (
                abs(self.config.brightness_adjustment) > 0.01
                and self.brightness_filter is not None
            ):
                brightness_component = scipy.signal.filtfilt(
                    self.brightness_filter[0], self.brightness_filter[1], processed
                )
                processed = (
                    processed + brightness_component * self.config.brightness_adjustment
                )
                enhancements_applied.append("brightness")

            # Sibilance control
            if self.config.sibilance_control > 0:
                processed = self._control_sibilance(processed)
                enhancements_applied.append("sibilance")

            # Normalize if requested
            if self.config.normalize:
                processed = self._normalize_audio(processed)
                enhancements_applied.append("normalize")

            metadata = {
                "enhancements_applied": enhancements_applied,
                "clarity_enhancement": self.config.clarity_enhancement,
                "presence_boost": self.config.presence_boost,
                "warmth_adjustment": self.config.warmth_adjustment,
                "brightness_adjustment": self.config.brightness_adjustment,
                "sibilance_control": self.config.sibilance_control,
                "normalize": self.config.normalize,
            }

            return processed, metadata

        except Exception as e:
            raise Exception(f"Voice enhancement failed: {e}")

    def _enhance_clarity(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance clarity using harmonic enhancement."""
        # Simple harmonic enhancement
        enhanced = audio_data.copy()

        # Add subtle second harmonic
        if len(enhanced) > 1:
            # Simple harmonic generation (placeholder)
            harmonic_gain = self.config.clarity_enhancement * 0.1
            enhanced = enhanced + harmonic_gain * np.tanh(enhanced * 2)

        return enhanced

    def _control_sibilance(self, audio_data: np.ndarray) -> np.ndarray:
        """Control harsh sibilants."""
        # Simple sibilance control using high-frequency limiting
        if len(audio_data) < 64:
            return audio_data

        # De-esser frequency range (6-10 kHz)
        nyquist = self.sample_rate / 2
        sibilance_freq = [6000 / nyquist, min(10000 / nyquist, 0.99)]

        try:
            # Design sibilance filter
            sibilance_filter = scipy.signal.butter(2, sibilance_freq, btype="band")

            # Extract sibilance frequencies
            sibilance_component = scipy.signal.filtfilt(
                sibilance_filter[0], sibilance_filter[1], audio_data
            )

            # Apply gentle compression to sibilance
            compressed_sibilance = sibilance_component * (
                1 - self.config.sibilance_control
            )

            # Reconstruct audio
            return audio_data - sibilance_component + compressed_sibilance

        except Exception:
            return audio_data

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio levels."""
        if len(audio_data) == 0:
            return audio_data

        # Peak normalization
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            return audio_data / peak * 0.95  # Leave some headroom

        return audio_data

    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "normalize": self.config.normalize,
            "clarity_enhancement": self.config.clarity_enhancement,
            "presence_boost": self.config.presence_boost,
            "warmth_adjustment": self.config.warmth_adjustment,
            "brightness_adjustment": self.config.brightness_adjustment,
            "sibilance_control": self.config.sibilance_control,
        }

    def update_config(self, new_config: VoiceEnhancementConfig):
        """Update configuration and redesign filters."""
        super().update_config(new_config)
        self._design_enhancement_filters()
