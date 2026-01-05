#!/usr/bin/env python3
"""
Voice Frequency Filter Stage

Modular voice filtering implementation that can be used independently
or as part of the complete audio processing pipeline.
"""

import numpy as np
import scipy.signal
from typing import Dict, Any, Tuple
from ..stage_components import BaseAudioStage
from ..config import VoiceFilterConfig


class VoiceFilterStage(BaseAudioStage):
    """Voice frequency filtering stage component."""

    def __init__(self, config: VoiceFilterConfig, sample_rate: int = 16000):
        super().__init__("voice_filter", config, sample_rate)

        # Design filters
        self._design_filters()

        self.is_initialized = True

    def _design_filters(self):
        """Design voice frequency filters."""
        nyquist = self.sample_rate / 2

        # Fundamental frequency enhancement filter
        if (
            self.config.fundamental_min < nyquist
            and self.config.fundamental_max < nyquist
        ):
            fundamental_low = self.config.fundamental_min / nyquist
            fundamental_high = min(self.config.fundamental_max / nyquist, 0.99)

            self.fundamental_filter = scipy.signal.butter(
                2, [fundamental_low, fundamental_high], btype="band"
            )
        else:
            self.fundamental_filter = None

        # Formant filters if enabled
        if self.config.preserve_formants:
            formant1_low = self.config.formant1_min / nyquist
            formant1_high = min(self.config.formant1_max / nyquist, 0.99)

            formant2_low = self.config.formant2_min / nyquist
            formant2_high = min(self.config.formant2_max / nyquist, 0.99)

            self.formant1_filter = scipy.signal.butter(
                2, [formant1_low, formant1_high], btype="band"
            )
            self.formant2_filter = scipy.signal.butter(
                2, [formant2_low, formant2_high], btype="band"
            )
        else:
            self.formant1_filter = None
            self.formant2_filter = None

        # High frequency rolloff
        if self.config.high_freq_rolloff < nyquist:
            rolloff_freq = self.config.high_freq_rolloff / nyquist
            self.rolloff_filter = scipy.signal.butter(4, rolloff_freq, btype="low")
        else:
            self.rolloff_filter = None

    def _process_audio(
        self, audio_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio through voice filtering."""
        try:
            processed = audio_data.copy()

            # Track filter applications
            filters_applied = []

            # Apply fundamental frequency enhancement
            if self.fundamental_filter is not None:
                fundamental_enhanced = scipy.signal.filtfilt(
                    self.fundamental_filter[0], self.fundamental_filter[1], processed
                )
                processed = processed + (
                    fundamental_enhanced * (self.config.voice_band_gain - 1.0)
                )
                filters_applied.append("fundamental")

            # Apply formant preservation
            if self.config.preserve_formants and self.formant1_filter is not None:
                formant1_enhanced = scipy.signal.filtfilt(
                    self.formant1_filter[0], self.formant1_filter[1], processed
                )
                formant2_enhanced = scipy.signal.filtfilt(
                    self.formant2_filter[0], self.formant2_filter[1], processed
                )

                # Blend formant enhancements
                formant_boost = 0.1 * (formant1_enhanced + formant2_enhanced)
                processed = processed + formant_boost
                filters_applied.append("formants")

            # Apply high frequency rolloff
            if self.rolloff_filter is not None:
                processed = scipy.signal.filtfilt(
                    self.rolloff_filter[0], self.rolloff_filter[1], processed
                )
                filters_applied.append("rolloff")

            # Calculate frequency response metrics
            metadata = {
                "filters_applied": filters_applied,
                "voice_band_gain": self.config.voice_band_gain,
                "fundamental_range_hz": [
                    self.config.fundamental_min,
                    self.config.fundamental_max,
                ],
                "formant1_range_hz": [
                    self.config.formant1_min,
                    self.config.formant1_max,
                ],
                "formant2_range_hz": [
                    self.config.formant2_min,
                    self.config.formant2_max,
                ],
                "high_freq_rolloff_hz": self.config.high_freq_rolloff,
                "preserve_formants": self.config.preserve_formants,
            }

            return processed, metadata

        except Exception as e:
            raise Exception(f"Voice filtering failed: {e}")

    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "fundamental_min": self.config.fundamental_min,
            "fundamental_max": self.config.fundamental_max,
            "formant1_min": self.config.formant1_min,
            "formant1_max": self.config.formant1_max,
            "formant2_min": self.config.formant2_min,
            "formant2_max": self.config.formant2_max,
            "preserve_formants": self.config.preserve_formants,
            "voice_band_gain": self.config.voice_band_gain,
            "high_freq_rolloff": self.config.high_freq_rolloff,
        }

    def update_config(self, new_config: VoiceFilterConfig):
        """Update configuration and redesign filters."""
        super().update_config(new_config)
        self._design_filters()

    def get_frequency_response(self, frequencies: np.ndarray = None) -> Dict[str, Any]:
        """Get frequency response of the voice filter."""
        if frequencies is None:
            frequencies = np.logspace(1, 4, 1000)  # 10Hz to 10kHz

        try:
            # Calculate frequency response for each filter
            response_data = {}

            if self.fundamental_filter is not None:
                w, h = scipy.signal.freqz(
                    self.fundamental_filter[0],
                    self.fundamental_filter[1],
                    worN=frequencies,
                    fs=self.sample_rate,
                )
                response_data["fundamental"] = {
                    "frequencies": frequencies,
                    "magnitude_db": 20 * np.log10(np.abs(h)),
                    "phase": np.angle(h),
                }

            if self.rolloff_filter is not None:
                w, h = scipy.signal.freqz(
                    self.rolloff_filter[0],
                    self.rolloff_filter[1],
                    worN=frequencies,
                    fs=self.sample_rate,
                )
                response_data["rolloff"] = {
                    "frequencies": frequencies,
                    "magnitude_db": 20 * np.log10(np.abs(h)),
                    "phase": np.angle(h),
                }

            return response_data

        except Exception as e:
            return {"error": str(e)}
