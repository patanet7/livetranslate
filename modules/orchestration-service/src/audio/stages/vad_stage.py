#!/usr/bin/env python3
"""
Voice Activity Detection Stage

Modular VAD implementation that can be used independently
or as part of the complete audio processing pipeline.
"""

from typing import Any

import numpy as np

from ..config import VADConfig, VADMode
from ..stage_components import BaseAudioStage


class VADStage(BaseAudioStage):
    """Voice Activity Detection stage component."""

    def __init__(self, config: VADConfig, sample_rate: int = 16000):
        super().__init__("vad", config, sample_rate)

        # VAD state
        self.previous_energy = 0.0
        self.noise_floor = 0.01
        self.adaptation_rate = 0.1

        # WebRTC VAD simulation (simplified)
        self.webrtc_history = []
        self.webrtc_threshold = 0.5

        self.is_initialized = True

    def _process_audio(self, audio_data: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Process audio through VAD."""
        try:
            if self.config.mode == VADMode.BASIC:
                voice_detected, confidence = self._energy_based_vad(audio_data)
            elif self.config.mode == VADMode.WEBRTC:
                voice_detected, confidence = self._webrtc_vad_simulation(audio_data)
            elif self.config.mode == VADMode.AGGRESSIVE:
                voice_detected, confidence = self._aggressive_vad(audio_data)
            else:
                voice_detected, confidence = self._energy_based_vad(audio_data)

            metadata = {
                "voice_detected": voice_detected,
                "confidence": confidence,
                "energy_threshold": self.config.energy_threshold,
                "noise_floor": self.noise_floor,
                "mode": self.config.mode.value,
                "aggressiveness": self.config.aggressiveness,
            }

            return audio_data, metadata

        except Exception as e:
            raise Exception(f"VAD processing failed: {e}") from e

    def _energy_based_vad(self, audio_data: np.ndarray) -> tuple[bool, float]:
        """Energy-based voice activity detection."""
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_data**2))

        # Adaptive noise floor
        if rms_energy < self.config.energy_threshold:
            self.noise_floor = (
                1 - self.adaptation_rate
            ) * self.noise_floor + self.adaptation_rate * rms_energy

        # Voice detection
        voice_threshold = self.noise_floor + self.config.energy_threshold
        voice_detected = rms_energy > voice_threshold

        # Confidence calculation
        if voice_detected:
            confidence = min(1.0, rms_energy / voice_threshold)
        else:
            confidence = max(0.0, rms_energy / voice_threshold)

        # Apply sensitivity
        confidence = confidence * self.config.sensitivity

        return voice_detected, confidence

    def _webrtc_vad_simulation(self, audio_data: np.ndarray) -> tuple[bool, float]:
        """WebRTC VAD simulation."""
        # Calculate energy in voice frequency range
        voice_energy = self._calculate_voice_frequency_energy(audio_data)

        # Simulate WebRTC aggressiveness
        threshold_multiplier = 1.0 + (self.config.aggressiveness * 0.2)
        adjusted_threshold = self.config.energy_threshold * threshold_multiplier

        voice_detected = voice_energy > adjusted_threshold

        # Confidence based on energy ratio
        confidence = min(1.0, voice_energy / adjusted_threshold) if voice_detected else 0.0

        # Apply sensitivity
        confidence = confidence * self.config.sensitivity

        return voice_detected, confidence

    def _aggressive_vad(self, audio_data: np.ndarray) -> tuple[bool, float]:
        """Aggressive VAD with stricter criteria."""
        # Use both energy and frequency analysis
        rms_energy = np.sqrt(np.mean(audio_data**2))
        voice_freq_energy = self._calculate_voice_frequency_energy(audio_data)

        # More aggressive thresholds
        energy_threshold = self.config.energy_threshold * 0.5
        voice_threshold = self.noise_floor + energy_threshold

        # Both conditions must be met
        energy_detected = rms_energy > voice_threshold
        freq_detected = voice_freq_energy > energy_threshold

        voice_detected = energy_detected and freq_detected

        # Confidence calculation
        if voice_detected:
            energy_conf = min(1.0, rms_energy / voice_threshold)
            freq_conf = min(1.0, voice_freq_energy / energy_threshold)
            confidence = (energy_conf + freq_conf) / 2.0
        else:
            confidence = 0.0

        # Apply sensitivity
        confidence = confidence * self.config.sensitivity

        return voice_detected, confidence

    def _calculate_voice_frequency_energy(self, audio_data: np.ndarray) -> float:
        """Calculate energy in voice frequency range."""
        if len(audio_data) < 64:  # Not enough samples for meaningful analysis
            return 0.0

        # Simple frequency analysis using FFT
        fft = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1 / self.sample_rate)

        # Find indices for voice frequency range
        voice_mask = (freqs >= self.config.voice_freq_min) & (freqs <= self.config.voice_freq_max)

        if not np.any(voice_mask):
            return 0.0

        # Calculate energy in voice frequency range
        voice_energy = np.sum(np.abs(fft[voice_mask]) ** 2)
        total_energy = np.sum(np.abs(fft) ** 2)

        if total_energy == 0:
            return 0.0

        return voice_energy / total_energy

    def _get_stage_config(self) -> dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "mode": self.config.mode.value,
            "aggressiveness": self.config.aggressiveness,
            "energy_threshold": self.config.energy_threshold,
            "voice_freq_min": self.config.voice_freq_min,
            "voice_freq_max": self.config.voice_freq_max,
            "frame_duration_ms": self.config.frame_duration_ms,
            "sensitivity": self.config.sensitivity,
        }

    def get_voice_detection_result(self, audio_data: np.ndarray) -> dict[str, Any]:
        """Get voice detection result without processing audio."""
        result = self.process(audio_data)
        return {
            "voice_detected": result.metadata.get("voice_detected", False),
            "confidence": result.metadata.get("confidence", 0.0),
            "processing_time_ms": result.processing_time_ms,
            "energy_threshold": result.metadata.get("energy_threshold", 0.0),
            "noise_floor": result.metadata.get("noise_floor", 0.0),
        }
