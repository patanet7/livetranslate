#!/usr/bin/env python3
"""
Audio Processing Pipeline - Orchestration Service

Modular audio processing pipeline that uses individual stage components
for flexible configuration and comprehensive monitoring.

Features:
- Modular stage architecture for independent testing
- Real-time performance monitoring and database storage
- Configurable performance targets per stage
- Comprehensive error handling and recovery
- WebSocket integration for real-time updates
- Database aggregation of performance metrics
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
from dataclasses import asdict

from .config import (
    AudioProcessingConfig,
    AudioConfigurationManager,
    VADConfig,
    VoiceFilterConfig,
    NoiseReductionConfig,
    VoiceEnhancementConfig,
    AGCConfig,
    CompressionConfig,
    LimiterConfig,
    QualityConfig,
    VADMode,
    NoiseReductionMode,
    CompressionMode,
    AGCMode,
)
from .models import QualityMetrics
from .stage_components import ModularAudioPipeline, StagePerformanceTarget
from .stages import (
    VADStage,
    VoiceFilterStage,
    NoiseReductionStage,
    VoiceEnhancementStage,
    AGCStage,
    CompressionStage,
    LimiterStage
)
from ..database.processing_metrics import get_metrics_manager

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Voice Activity Detection with multiple algorithm support.
    Integrates WebRTC VAD, energy-based VAD, and frequency domain analysis.
    """
    
    def __init__(self, config: VADConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # VAD state
        self.previous_energy = 0.0
        self.noise_floor = 0.01
        self.adaptation_rate = 0.1
        
        # WebRTC VAD simulation (simplified)
        self.webrtc_history = []
        self.webrtc_threshold = 0.5
        
        logger.debug(f"VoiceActivityDetector initialized: mode={config.mode}, sensitivity={config.sensitivity}")
    
    def detect_voice_activity(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect voice activity in audio data.
        
        Returns:
            Tuple of (voice_detected, confidence)
        """
        if not self.config.enabled:
            return True, 1.0  # Assume voice if VAD disabled
        
        try:
            if self.config.mode == VADMode.BASIC:
                return self._energy_based_vad(audio_data)
            elif self.config.mode == VADMode.WEBRTC:
                return self._webrtc_vad_simulation(audio_data)
            elif self.config.mode == VADMode.AGGRESSIVE:
                return self._aggressive_vad(audio_data)
            else:
                return self._energy_based_vad(audio_data)
                
        except Exception as e:
            logger.error(f"VAD detection failed: {e}")
            return True, 0.5  # Default to voice detected with low confidence
    
    def _energy_based_vad(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Energy-based voice activity detection."""
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Adaptive noise floor
        if rms_energy < self.config.energy_threshold:
            self.noise_floor = (1 - self.adaptation_rate) * self.noise_floor + self.adaptation_rate * rms_energy
        
        # Voice detection
        voice_threshold = max(self.config.energy_threshold, self.noise_floor * 3)
        voice_detected = rms_energy > voice_threshold
        
        # Confidence based on energy ratio
        confidence = min(1.0, rms_energy / voice_threshold) if voice_threshold > 0 else 0.0
        confidence = max(0.0, confidence * self.config.sensitivity)
        
        return voice_detected, confidence
    
    def _webrtc_vad_simulation(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Simplified WebRTC VAD simulation."""
        # Energy analysis
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
        zcr = zero_crossings / len(audio_data) if len(audio_data) > 0 else 0.0
        
        # Spectral features (simplified)
        if len(audio_data) > 256:
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            spectrum = np.abs(np.fft.rfft(audio_data))
            
            # Voice frequency energy (85-300 Hz)
            voice_mask = (freqs >= self.config.voice_freq_min) & (freqs <= self.config.voice_freq_max)
            voice_energy = np.sum(spectrum[voice_mask]) / (np.sum(spectrum) + 1e-10)
        else:
            voice_energy = 0.5
        
        # Combine features
        energy_score = min(1.0, rms_energy / self.config.energy_threshold)
        zcr_score = min(1.0, zcr / 0.1)  # Normalize ZCR
        spectral_score = voice_energy
        
        # Weighted combination based on aggressiveness
        if self.config.aggressiveness <= 1:
            # Less aggressive
            combined_score = 0.6 * energy_score + 0.2 * zcr_score + 0.2 * spectral_score
            threshold = 0.3
        elif self.config.aggressiveness == 2:
            # Moderate
            combined_score = 0.5 * energy_score + 0.3 * zcr_score + 0.2 * spectral_score
            threshold = 0.5
        else:
            # More aggressive
            combined_score = 0.4 * energy_score + 0.3 * zcr_score + 0.3 * spectral_score
            threshold = 0.7
        
        voice_detected = combined_score > threshold
        confidence = combined_score * self.config.sensitivity
        
        return voice_detected, min(1.0, confidence)
    
    def _aggressive_vad(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Aggressive VAD with higher sensitivity."""
        voice_detected, confidence = self._webrtc_vad_simulation(audio_data)
        
        # Lower thresholds for aggressive mode
        if not voice_detected:
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            if rms_energy > self.config.energy_threshold * 0.5:
                voice_detected = True
                confidence = max(confidence, 0.3)
        
        return voice_detected, confidence


class VoiceFrequencyFilter:
    """
    Voice frequency filtering with formant preservation.
    Enhances human voice frequencies while attenuating others.
    """
    
    def __init__(self, config: VoiceFilterConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # Design filters
        self._design_filters()
        
        logger.debug(f"VoiceFrequencyFilter initialized: gain={config.voice_band_gain}")
    
    def _design_filters(self):
        """Design voice frequency filters."""
        nyquist = self.sample_rate / 2
        
        # Fundamental frequency band filter
        fundamental_low = self.config.fundamental_min / nyquist
        fundamental_high = self.config.fundamental_max / nyquist
        
        if fundamental_high < 1.0:
            self.fundamental_filter = scipy.signal.butter(
                4, [fundamental_low, fundamental_high], btype='band'
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
                2, [formant1_low, formant1_high], btype='band'
            )
            self.formant2_filter = scipy.signal.butter(
                2, [formant2_low, formant2_high], btype='band'
            )
        else:
            self.formant1_filter = None
            self.formant2_filter = None
        
        # High frequency rolloff
        if self.config.high_freq_rolloff < nyquist:
            rolloff_freq = self.config.high_freq_rolloff / nyquist
            self.rolloff_filter = scipy.signal.butter(4, rolloff_freq, btype='low')
        else:
            self.rolloff_filter = None
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply voice frequency filtering."""
        if not self.config.enabled:
            return audio_data
        
        try:
            processed = audio_data.copy()
            
            # Apply fundamental frequency enhancement
            if self.fundamental_filter is not None:
                fundamental_enhanced = scipy.signal.filtfilt(
                    self.fundamental_filter[0], self.fundamental_filter[1], processed
                )
                processed = processed + (fundamental_enhanced * (self.config.voice_band_gain - 1.0))
            
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
            
            # Apply high frequency rolloff
            if self.rolloff_filter is not None:
                processed = scipy.signal.filtfilt(
                    self.rolloff_filter[0], self.rolloff_filter[1], processed
                )
            
            return processed
            
        except Exception as e:
            logger.error(f"Voice filtering failed: {e}")
            return audio_data


class NoiseReducer:
    """
    Advanced noise reduction with voice protection.
    Implements spectral subtraction with musical noise suppression.
    """
    
    def __init__(self, config: NoiseReductionConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # Noise reduction state
        self.noise_profile = None
        self.noise_history = []
        self.adaptation_counter = 0
        
        # Parameters
        self.frame_size = 1024
        self.overlap = 0.5
        self.hop_size = int(self.frame_size * (1 - self.overlap))
        
        logger.debug(f"NoiseReducer initialized: mode={config.mode}, strength={config.strength}")
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction."""
        if not self.config.enabled:
            return audio_data
        
        try:
            if self.config.mode == NoiseReductionMode.LIGHT:
                return self._light_noise_reduction(audio_data)
            elif self.config.mode == NoiseReductionMode.MODERATE:
                return self._moderate_noise_reduction(audio_data)
            elif self.config.mode == NoiseReductionMode.AGGRESSIVE:
                return self._aggressive_noise_reduction(audio_data)
            elif self.config.mode == NoiseReductionMode.ADAPTIVE:
                return self._adaptive_noise_reduction(audio_data)
            else:
                return audio_data
                
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return audio_data
    
    def _light_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Light noise reduction using simple spectral gating."""
        if len(audio_data) < self.frame_size:
            return audio_data
        
        # Simple spectral gating
        spectrum = rfft(audio_data)
        magnitude = np.abs(spectrum)
        
        # Estimate noise floor
        noise_floor = np.percentile(magnitude, 20)
        
        # Apply gating
        gate_threshold = noise_floor * (1 + self.config.strength)
        mask = magnitude > gate_threshold
        
        # Soft gating to avoid artifacts
        soft_mask = np.where(mask, 1.0, 0.3)
        spectrum_processed = spectrum * soft_mask
        
        return irfft(spectrum_processed, len(audio_data))
    
    def _moderate_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Moderate noise reduction with voice protection."""
        return self._spectral_subtraction(audio_data, strength_factor=1.0)
    
    def _aggressive_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Aggressive noise reduction."""
        return self._spectral_subtraction(audio_data, strength_factor=1.5)
    
    def _adaptive_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Adaptive noise reduction that adjusts based on signal characteristics."""
        # Analyze signal characteristics
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Adapt strength based on signal level
        if rms_energy < 0.01:
            # Very quiet signal - use aggressive reduction
            adapted_strength = self.config.strength * 1.5
        elif rms_energy > 0.1:
            # Loud signal - use lighter reduction
            adapted_strength = self.config.strength * 0.7
        else:
            # Normal signal - use configured strength
            adapted_strength = self.config.strength
        
        return self._spectral_subtraction(audio_data, strength_factor=adapted_strength)
    
    def _spectral_subtraction(self, audio_data: np.ndarray, strength_factor: float = 1.0) -> np.ndarray:
        """Spectral subtraction with musical noise suppression."""
        if len(audio_data) < self.frame_size:
            return audio_data
        
        # Windowing and FFT
        window = np.hanning(self.frame_size)
        processed = np.zeros_like(audio_data)
        
        # Process overlapping frames
        for i in range(0, len(audio_data) - self.frame_size, self.hop_size):
            frame = audio_data[i:i + self.frame_size] * window
            spectrum = rfft(frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            # Update noise profile
            self._update_noise_profile(magnitude)
            
            if self.noise_profile is not None:
                # Spectral subtraction
                alpha = self.config.strength * strength_factor
                beta = 0.1  # Over-subtraction factor
                
                # Calculate gain
                gain = 1.0 - alpha * (self.noise_profile / (magnitude + 1e-10))
                
                # Voice protection - preserve voice frequencies
                if self.config.voice_protection:
                    freqs = np.fft.rfftfreq(self.frame_size, 1/self.sample_rate)
                    voice_mask = ((freqs >= 85) & (freqs <= 3000))  # Voice frequency range
                    gain[voice_mask] = np.maximum(gain[voice_mask], 0.3)  # Minimum gain for voice
                
                # Apply floor to prevent musical noise
                gain = np.maximum(gain, 0.1)
                
                # Apply gain
                processed_spectrum = magnitude * gain * np.exp(1j * phase)
                processed_frame = irfft(processed_spectrum, self.frame_size) * window
                
                # Overlap-add
                processed[i:i + self.frame_size] += processed_frame
            else:
                # No noise profile yet, just add original frame
                processed[i:i + self.frame_size] += frame
        
        return processed
    
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


class VoiceEnhancer:
    """
    Voice enhancement processor for clarity, presence, and tonal adjustments.
    """
    
    def __init__(self, config: VoiceEnhancementConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # Design enhancement filters
        self._design_enhancement_filters()
        
        logger.debug(f"VoiceEnhancer initialized: clarity={config.clarity_enhancement}")
    
    def _design_enhancement_filters(self):
        """Design filters for voice enhancement."""
        nyquist = self.sample_rate / 2
        
        # Presence boost filter (2-5 kHz)
        if self.config.presence_boost > 0:
            presence_freq = [2000 / nyquist, 5000 / nyquist]
            presence_freq[1] = min(presence_freq[1], 0.99)
            self.presence_filter = scipy.signal.butter(2, presence_freq, btype='band')
        else:
            self.presence_filter = None
        
        # Warmth filter (low-mid frequencies)
        if abs(self.config.warmth_adjustment) > 0.01:
            warmth_freq = 500 / nyquist
            self.warmth_filter = scipy.signal.butter(2, warmth_freq, btype='low')
        else:
            self.warmth_filter = None
        
        # Brightness filter (high frequencies)
        if abs(self.config.brightness_adjustment) > 0.01:
            brightness_freq = 5000 / nyquist
            brightness_freq = min(brightness_freq, 0.99)
            self.brightness_filter = scipy.signal.butter(2, brightness_freq, btype='high')
        else:
            self.brightness_filter = None
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply voice enhancement."""
        if not self.config.enabled:
            return audio_data
        
        try:
            processed = audio_data.copy()
            
            # Clarity enhancement using harmonic enhancement
            if self.config.clarity_enhancement > 0:
                processed = self._enhance_clarity(processed)
            
            # Presence boost
            if self.config.presence_boost > 0 and self.presence_filter is not None:
                presence_enhanced = scipy.signal.filtfilt(
                    self.presence_filter[0], self.presence_filter[1], processed
                )
                processed = processed + presence_enhanced * self.config.presence_boost
            
            # Warmth adjustment
            if abs(self.config.warmth_adjustment) > 0.01 and self.warmth_filter is not None:
                warmth_component = scipy.signal.filtfilt(
                    self.warmth_filter[0], self.warmth_filter[1], processed
                )
                processed = processed + warmth_component * self.config.warmth_adjustment
            
            # Brightness adjustment
            if abs(self.config.brightness_adjustment) > 0.01 and self.brightness_filter is not None:
                brightness_component = scipy.signal.filtfilt(
                    self.brightness_filter[0], self.brightness_filter[1], processed
                )
                processed = processed + brightness_component * self.config.brightness_adjustment
            
            # Sibilance control
            if self.config.sibilance_control > 0:
                processed = self._control_sibilance(processed)
            
            # Normalization if enabled
            if self.config.normalize:
                processed = self._normalize_audio(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Voice enhancement failed: {e}")
            return audio_data
    
    def _enhance_clarity(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance clarity using harmonic enhancement."""
        if len(audio_data) < 256:
            return audio_data
        
        # Simple harmonic enhancement
        spectrum = rfft(audio_data)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Enhance harmonics
        enhanced_magnitude = magnitude * (1 + self.config.clarity_enhancement * 0.5)
        
        # Reconstruct signal
        enhanced_spectrum = enhanced_magnitude * np.exp(1j * phase)
        return irfft(enhanced_spectrum, len(audio_data))
    
    def _control_sibilance(self, audio_data: np.ndarray) -> np.ndarray:
        """Control harsh sibilant sounds."""
        # Simple sibilance reduction using high-frequency limiting
        nyquist = self.sample_rate / 2
        sibilant_freq = 6000 / nyquist
        
        if sibilant_freq < 1.0:
            # Design de-esser filter
            sos = scipy.signal.butter(4, sibilant_freq, btype='high', output='sos')
            sibilant_component = scipy.signal.sosfilt(sos, audio_data)
            
            # Apply reduction
            reduction_factor = 1.0 - self.config.sibilance_control
            return audio_data - sibilant_component * (1.0 - reduction_factor)
        
        return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        peak = np.max(np.abs(audio_data))
        if peak > 0.95:
            return audio_data * (0.95 / peak)
        return audio_data


class DynamicCompressor:
    """
    Dynamic range compressor with multiple modes.
    """
    
    def __init__(self, config: CompressionConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # Compressor state
        self.envelope = 0.0
        self.gain_reduction = 0.0
        
        # Convert time constants to samples
        self.attack_samples = int(self.config.attack_time * sample_rate / 1000)
        self.release_samples = int(self.config.release_time * sample_rate / 1000)
        
        logger.debug(f"DynamicCompressor initialized: threshold={config.threshold}dB, ratio={config.ratio}")
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression."""
        if not self.config.enabled:
            return audio_data
        
        try:
            # Convert threshold from dB to linear
            threshold_linear = 10 ** (self.config.threshold / 20)
            
            processed = np.zeros_like(audio_data)
            
            for i, sample in enumerate(audio_data):
                # Envelope detection
                sample_abs = abs(sample)
                
                if sample_abs > self.envelope:
                    # Attack
                    self.envelope += (sample_abs - self.envelope) / self.attack_samples
                else:
                    # Release
                    self.envelope += (sample_abs - self.envelope) / self.release_samples
                
                # Gain reduction calculation
                if self.envelope > threshold_linear:
                    # Above threshold - apply compression
                    excess = self.envelope / threshold_linear
                    if self.config.mode == CompressionMode.SOFT_KNEE:
                        # Soft knee compression
                        knee_ratio = min(1.0, excess / (10 ** (self.config.knee / 20)))
                        gain_reduction = 1.0 - (1.0 - 1.0/self.config.ratio) * knee_ratio
                    else:
                        # Hard knee compression
                        gain_reduction = 1.0 / self.config.ratio + (1.0 - 1.0/self.config.ratio) / excess
                    
                    self.gain_reduction = gain_reduction
                else:
                    # Below threshold - no compression
                    self.gain_reduction = 1.0
                
                # Apply gain reduction with makeup gain
                makeup_gain_linear = 10 ** (self.config.makeup_gain / 20)
                processed[i] = sample * self.gain_reduction * makeup_gain_linear
            
            return processed
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return audio_data


class AudioLimiter:
    """
    Final stage limiter to prevent clipping.
    """
    
    def __init__(self, config: LimiterConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # Limiter state
        self.delay_buffer = np.zeros(int(config.lookahead * sample_rate / 1000))
        self.buffer_index = 0
        self.gain_reduction = 1.0
        
        # Convert threshold and release time
        self.threshold_linear = 10 ** (config.threshold / 20)
        self.release_samples = int(config.release_time * sample_rate / 1000)
        
        logger.debug(f"AudioLimiter initialized: threshold={config.threshold}dB")
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply limiting."""
        if not self.config.enabled:
            return audio_data
        
        try:
            processed = np.zeros_like(audio_data)
            
            for i, sample in enumerate(audio_data):
                # Add sample to delay buffer
                delayed_sample = self.delay_buffer[self.buffer_index]
                self.delay_buffer[self.buffer_index] = sample
                self.buffer_index = (self.buffer_index + 1) % len(self.delay_buffer)
                
                # Peak detection on current sample
                peak = abs(sample)
                
                if peak > self.threshold_linear:
                    # Calculate required gain reduction
                    required_gain = self.threshold_linear / peak
                    self.gain_reduction = min(self.gain_reduction, required_gain)
                else:
                    # Release
                    self.gain_reduction += (1.0 - self.gain_reduction) / self.release_samples
                    self.gain_reduction = min(1.0, self.gain_reduction)
                
                # Apply gain reduction to delayed sample
                if self.config.soft_clip and abs(delayed_sample * self.gain_reduction) > self.threshold_linear:
                    # Soft clipping
                    sign = 1 if delayed_sample >= 0 else -1
                    processed[i] = sign * self.threshold_linear * np.tanh(abs(delayed_sample * self.gain_reduction) / self.threshold_linear)
                else:
                    processed[i] = delayed_sample * self.gain_reduction
            
            return processed
            
        except Exception as e:
            logger.error(f"Limiting failed: {e}")
            return audio_data


class AutoGainControl:
    """
    Auto Gain Control processor for maintaining consistent audio levels.
    Implements adaptive gain adjustment with multiple control modes.
    """
    
    def __init__(self, config: AGCConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
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
        self.lookahead_buffer = np.zeros(self.lookahead_samples) if self.lookahead_samples > 0 else None
        self.buffer_index = 0
        
        # Adaptation state
        self.adaptation_counter = 0
        self.level_history = []
        self.max_history_length = int(sample_rate * 0.1)  # 100ms history
        
        logger.debug(f"AutoGainControl initialized: mode={config.mode}, target={config.target_level}dB")
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply auto gain control."""
        if not self.config.enabled or self.config.mode == AGCMode.DISABLED:
            return audio_data
        
        try:
            processed = np.zeros_like(audio_data)
            
            for i, sample in enumerate(audio_data):
                # Calculate current signal level (RMS over recent samples)
                self.level_history.append(abs(sample))
                if len(self.level_history) > self.max_history_length:
                    self.level_history.pop(0)
                
                current_level = np.sqrt(np.mean(np.array(self.level_history) ** 2))
                
                # Noise gate
                if current_level > self.noise_gate_threshold_linear:
                    self.noise_gate_open = True
                elif current_level < self.noise_gate_threshold_linear * 0.5:  # Hysteresis
                    self.noise_gate_open = False
                
                if not self.noise_gate_open:
                    processed[i] = sample * self.current_gain
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
                
                # Update adaptation counter
                self.adaptation_counter += 1
            
            return processed
            
        except Exception as e:
            logger.error(f"Auto gain control failed: {e}")
            return audio_data
    
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


class AudioPipelineProcessor:
    """
    Modular audio processing pipeline that uses individual stage components
    for flexible configuration and comprehensive monitoring.
    """
    
    def __init__(self, config: AudioProcessingConfig, sample_rate: int = 16000, database_url: str = None):
        self.config = config
        self.sample_rate = sample_rate
        
        # Initialize modular pipeline
        self.pipeline = ModularAudioPipeline(sample_rate)
        
        # Initialize database metrics manager
        self.metrics_manager = get_metrics_manager(database_url) if database_url else None
        
        # Initialize stage components
        self._initialize_stages()
        
        # Set optional performance targets
        self._set_performance_targets()
        
        logger.info(f"Modular AudioPipelineProcessor initialized with preset: {config.preset_name}")
    
    def _initialize_stages(self):
        """Initialize all processing stages."""
        # Create stage instances
        vad_stage = VADStage(self.config.vad, self.sample_rate)
        voice_filter_stage = VoiceFilterStage(self.config.voice_filter, self.sample_rate)
        noise_reduction_stage = NoiseReductionStage(self.config.noise_reduction, self.sample_rate)
        voice_enhancement_stage = VoiceEnhancementStage(self.config.voice_enhancement, self.sample_rate)
        agc_stage = AGCStage(self.config.agc, self.sample_rate)
        compression_stage = CompressionStage(self.config.compression, self.sample_rate)
        limiter_stage = LimiterStage(self.config.limiter, self.sample_rate)
        
        # Add stages to pipeline in order
        self.pipeline.add_stage(vad_stage)
        self.pipeline.add_stage(voice_filter_stage)
        self.pipeline.add_stage(noise_reduction_stage)
        self.pipeline.add_stage(voice_enhancement_stage)
        self.pipeline.add_stage(agc_stage)
        self.pipeline.add_stage(compression_stage)
        self.pipeline.add_stage(limiter_stage)
        
        # Enable/disable stages based on config
        for stage_name in self.config.enabled_stages:
            self.pipeline.enable_stage(stage_name, True)
        
        # Disable stages not in enabled list
        all_stages = ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "agc", "compression", "limiter"]
        for stage_name in all_stages:
            if stage_name not in self.config.enabled_stages:
                self.pipeline.enable_stage(stage_name, False)
    
    def _set_performance_targets(self):
        """Set optional performance targets for each stage."""
        # These are optional targets that can be configured
        # Default targets based on real-time processing requirements
        
        performance_targets = {
            "vad": StagePerformanceTarget(target_latency_ms=5.0, max_latency_ms=10.0),
            "voice_filter": StagePerformanceTarget(target_latency_ms=8.0, max_latency_ms=15.0),
            "noise_reduction": StagePerformanceTarget(target_latency_ms=15.0, max_latency_ms=25.0),
            "voice_enhancement": StagePerformanceTarget(target_latency_ms=10.0, max_latency_ms=20.0),
            "agc": StagePerformanceTarget(target_latency_ms=12.0, max_latency_ms=20.0),
            "compression": StagePerformanceTarget(target_latency_ms=8.0, max_latency_ms=15.0),
            "limiter": StagePerformanceTarget(target_latency_ms=6.0, max_latency_ms=12.0)
        }
        
        # Apply targets to stages
        for stage_name, target in performance_targets.items():
            self.pipeline.set_stage_performance_target(stage_name, target)
    
    def update_config(self, config: AudioProcessingConfig):
        """Update processing configuration."""
        self.config = config
        
        # Update individual stage configs
        if self.pipeline.get_stage("vad"):
            self.pipeline.get_stage("vad").update_config(config.vad)
        if self.pipeline.get_stage("voice_filter"):
            self.pipeline.get_stage("voice_filter").update_config(config.voice_filter)
        if self.pipeline.get_stage("noise_reduction"):
            self.pipeline.get_stage("noise_reduction").update_config(config.noise_reduction)
        if self.pipeline.get_stage("voice_enhancement"):
            self.pipeline.get_stage("voice_enhancement").update_config(config.voice_enhancement)
        if self.pipeline.get_stage("agc"):
            self.pipeline.get_stage("agc").update_config(config.agc)
        if self.pipeline.get_stage("compression"):
            self.pipeline.get_stage("compression").update_config(config.compression)
        if self.pipeline.get_stage("limiter"):
            self.pipeline.get_stage("limiter").update_config(config.limiter)
        
        # Update stage enable/disable status
        for stage_name in self.config.enabled_stages:
            self.pipeline.enable_stage(stage_name, True)
        
        all_stages = ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "agc", "compression", "limiter"]
        for stage_name in all_stages:
            if stage_name not in self.config.enabled_stages:
                self.pipeline.enable_stage(stage_name, False)
        
        logger.info(f"Audio pipeline config updated to preset: {config.preset_name}")
    
    def process_audio_chunk(self, audio_data: np.ndarray, session_id: str = None, chunk_id: str = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process audio chunk through the modular pipeline with comprehensive monitoring.
        
        Args:
            audio_data: Input audio data
            session_id: Optional session identifier for tracking
            chunk_id: Optional chunk identifier for tracking
            
        Returns:
            Tuple of (processed_audio, processing_metadata)
        """
        try:
            # Process through modular pipeline
            pipeline_result = self.pipeline.process_chunk(audio_data)
            
            # Store metrics in database if manager is available
            if self.metrics_manager:
                try:
                    self.metrics_manager.store_pipeline_metrics(pipeline_result, session_id, chunk_id)
                except Exception as e:
                    logger.warning(f"Failed to store pipeline metrics: {e}")
            
            # Extract processed audio and metadata
            processed_audio = pipeline_result["final_audio"]
            processing_metadata = pipeline_result["pipeline_metadata"]
            
            # Add stage results for compatibility
            processing_metadata["stage_results"] = pipeline_result["stage_results"]
            
            # Add legacy fields for backward compatibility
            processing_metadata["stages_applied"] = processing_metadata.get("stages_processed", [])
            processing_metadata["vad_result"] = self._extract_vad_result(pipeline_result)
            processing_metadata["quality_metrics"] = self._calculate_quality_metrics(audio_data, processed_audio)
            processing_metadata["bypassed"] = len(processing_metadata.get("stages_with_errors", [])) > 0
            
            return processed_audio, processing_metadata
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            
            # Return original audio with error metadata
            error_metadata = {
                "stages_applied": [],
                "stage_results": {},
                "vad_result": None,
                "quality_metrics": None,
                "total_processing_time_ms": 0,
                "bypassed": True,
                "error": str(e),
                "pipeline_metadata": {
                    "stages_processed": [],
                    "stages_bypassed": [],
                    "stages_with_errors": ["pipeline"],
                    "performance_warnings": [{"type": "pipeline_error", "message": str(e)}]
                }
            }
            
            return audio_data, error_metadata
    
    def _extract_vad_result(self, pipeline_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract VAD result from pipeline results."""
        stage_results = pipeline_result.get("stage_results", {})
        vad_result = stage_results.get("vad")
        
        if vad_result and vad_result.metadata:
            return {
                "voice_detected": vad_result.metadata.get("voice_detected", False),
                "confidence": vad_result.metadata.get("confidence", 0.0)
            }
        
        return None
    
    def _calculate_quality_metrics(self, input_audio: np.ndarray, output_audio: np.ndarray) -> Dict[str, Any]:
        """Calculate quality metrics for processed audio."""
        try:
            # Calculate basic quality metrics
            input_rms = np.sqrt(np.mean(input_audio ** 2))
            output_rms = np.sqrt(np.mean(output_audio ** 2))
            
            # SNR estimation (simplified)
            signal_power = np.mean(output_audio ** 2)
            noise_power = np.mean((output_audio - input_audio) ** 2)
            snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
            
            return {
                "input_rms": input_rms,
                "output_rms": output_rms,
                "level_change_db": 20 * np.log10(output_rms / max(input_rms, 1e-10)),
                "estimated_snr_db": snr,
                "dynamic_range": np.max(output_audio) - np.min(output_audio)
            }
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            return {}
    
    def process_single_stage(self, stage_name: str, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process audio through a single stage only."""
        result = self.pipeline.process_single_stage(stage_name, audio_data)
        
        if result:
            return {
                "processed_audio": result.processed_audio,
                "stage_result": result,
                "processing_time_ms": result.processing_time_ms,
                "metadata": result.metadata
            }
        
        return None
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return self.pipeline.get_pipeline_statistics()
    
    def get_stage_statistics(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific stage."""
        stage = self.pipeline.get_stage(stage_name)
        return stage.get_statistics() if stage else None
    
    def reset_all_statistics(self):
        """Reset all pipeline and stage statistics."""
        self.pipeline.reset_all_statistics()
    
    def get_database_statistics(self, hours: int = 24, session_id: str = None) -> Dict[str, Any]:
        """Get processing statistics from database."""
        if self.metrics_manager:
            return self.metrics_manager.get_processing_statistics(hours, session_id)
        return {"error": "Database metrics not available"}
    
    def get_real_time_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get real-time metrics from database."""
        if self.metrics_manager:
            return self.metrics_manager.get_real_time_metrics(minutes)
        return {"error": "Database metrics not available"}
    
    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Clean up old database metrics."""
        if self.metrics_manager:
            self.metrics_manager.cleanup_old_metrics(days_to_keep)
    
    def set_stage_performance_target(self, stage_name: str, target: StagePerformanceTarget):
        """Set performance target for a specific stage."""
        self.pipeline.set_stage_performance_target(stage_name, target)
    
    def enable_stage(self, stage_name: str, enabled: bool = True):
        """Enable or disable a specific stage."""
        self.pipeline.enable_stage(stage_name, enabled)
    
    def get_stage_config(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get current configuration for a specific stage."""
        stage = self.pipeline.get_stage(stage_name)
        if stage:
            return stage._get_stage_config()
        return None
                processing_metadata["bypassed"] = True
                processing_metadata["bypass_reason"] = "low_input_quality"
                return audio_data, processing_metadata
            
            current_audio = audio_data.copy()
            
            # Stage 1: Voice Activity Detection
            if self.config.is_stage_enabled("vad"):
                stage_start_time = time.time()
                voice_detected, vad_confidence = self.vad.detect_voice_activity(current_audio)
                stage_time_ms = (time.time() - stage_start_time) * 1000
                
                processing_metadata["vad_result"] = {
                    "voice_detected": voice_detected,
                    "confidence": vad_confidence
                }
                processing_metadata["stages_applied"].append("vad")
                processing_metadata["stage_timings"]["vad"] = stage_time_ms
                
                # Check VAD latency target (5ms)
                if stage_time_ms > 5.0:
                    processing_metadata["performance_metrics"]["stage_latency_warnings"].append({
                        "stage": "vad",
                        "actual_ms": stage_time_ms,
                        "target_ms": 5.0,
                        "exceeded_by_ms": stage_time_ms - 5.0
                    })
                
                # Skip further processing if no voice detected
                if not voice_detected and vad_confidence < 0.3:
                    processing_metadata["bypassed"] = True
                    processing_metadata["bypass_reason"] = "no_voice_detected"
                    return current_audio, processing_metadata
            
            # Stage 2: Voice Frequency Filtering
            if self.config.is_stage_enabled("voice_filter"):
                current_audio = self.voice_filter.process_audio(current_audio)
                processing_metadata["stages_applied"].append("voice_filter")
                
                if self.config.pause_after_stage.get("voice_filter", False):
                    return current_audio, processing_metadata
            
            # Stage 3: Noise Reduction
            if self.config.is_stage_enabled("noise_reduction"):
                current_audio = self.noise_reducer.process_audio(current_audio)
                processing_metadata["stages_applied"].append("noise_reduction")
                
                if self.config.pause_after_stage.get("noise_reduction", False):
                    return current_audio, processing_metadata
            
            # Stage 4: Voice Enhancement
            if self.config.is_stage_enabled("voice_enhancement"):
                current_audio = self.voice_enhancer.process_audio(current_audio)
                processing_metadata["stages_applied"].append("voice_enhancement")
                
                if self.config.pause_after_stage.get("voice_enhancement", False):
                    return current_audio, processing_metadata
            
            # Stage 5: Auto Gain Control
            if self.config.is_stage_enabled("agc"):
                current_audio = self.agc.process_audio(current_audio)
                processing_metadata["stages_applied"].append("agc")
                
                if self.config.pause_after_stage.get("agc", False):
                    return current_audio, processing_metadata
            
            # Stage 6: Dynamic Range Compression
            if self.config.is_stage_enabled("compression"):
                current_audio = self.compressor.process_audio(current_audio)
                processing_metadata["stages_applied"].append("compression")
                
                if self.config.pause_after_stage.get("compression", False):
                    return current_audio, processing_metadata
            
            # Stage 7: Final Limiting
            if self.config.is_stage_enabled("limiter"):
                current_audio = self.limiter.process_audio(current_audio)
                processing_metadata["stages_applied"].append("limiter")
            
            # Final quality analysis
            output_quality = self._analyze_quality(current_audio)
            processing_metadata["output_quality"] = output_quality.overall_quality_score
            processing_metadata["quality_improvement"] = (
                output_quality.overall_quality_score - input_quality.overall_quality_score
            )
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            processing_metadata["processing_time_ms"] = processing_time
            
            self.total_samples_processed += len(audio_data)
            self.processing_times.append(processing_time)
            self.quality_history.append(output_quality.overall_quality_score)
            
            # Keep history limited
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
            if len(self.quality_history) > 100:
                self.quality_history = self.quality_history[-100:]
            
            return current_audio, processing_metadata
            
        except Exception as e:
            logger.error(f"Audio pipeline processing failed: {e}")
            processing_metadata["error"] = str(e)
            processing_metadata["processing_time_ms"] = (time.time() - start_time) * 1000
            return audio_data, processing_metadata
    
    def _analyze_quality(self, audio_data: np.ndarray) -> QualityMetrics:
        """Analyze audio quality."""
        # Reuse quality analysis from chunk_manager
        try:
            # Basic quality metrics
            rms_level = np.sqrt(np.mean(audio_data ** 2))
            peak_level = np.max(np.abs(audio_data))
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            zcr = zero_crossings / len(audio_data) if len(audio_data) > 0 else 0.0
            
            # SNR estimation
            signal_power = np.mean(audio_data ** 2)
            noise_estimate = np.percentile(np.abs(audio_data), 10) ** 2
            snr = 10 * np.log10(signal_power / max(noise_estimate, 1e-10)) if signal_power > 0 else -60
            
            # Voice activity
            voice_activity = rms_level > self.config.quality.silence_threshold and zcr > 0.01
            
            # Overall quality score
            quality_score = min(1.0, (rms_level / 0.1 + min(snr + 10, 30) / 40) / 2)
            
            return QualityMetrics(
                rms_level=float(rms_level),
                peak_level=float(peak_level),
                signal_to_noise_ratio=float(snr),
                zero_crossing_rate=float(zcr),
                voice_activity_detected=voice_activity,
                voice_activity_confidence=min(1.0, rms_level / self.config.quality.silence_threshold),
                speaking_time_ratio=float(np.sum(np.abs(audio_data) > self.config.quality.silence_threshold) / len(audio_data)),
                clipping_detected=peak_level > self.config.quality.clipping_threshold,
                distortion_level=0.0,
                noise_level=min(1.0, noise_estimate / 0.01),
                overall_quality_score=float(quality_score),
                quality_factors={},
                analysis_method="pipeline_analyzer"
            )
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return QualityMetrics(
                rms_level=0.0,
                peak_level=0.0,
                signal_to_noise_ratio=-60.0,
                zero_crossing_rate=0.0,
                voice_activity_detected=False,
                voice_activity_confidence=0.0,
                speaking_time_ratio=0.0,
                clipping_detected=False,
                distortion_level=0.0,
                noise_level=1.0,
                overall_quality_score=0.0,
                quality_factors={},
                analysis_method="failed"
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_samples_processed": self.total_samples_processed,
            "total_duration_processed": self.total_samples_processed / self.sample_rate,
            "average_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "max_processing_time": np.max(self.processing_times) if self.processing_times else 0,
            "average_quality_score": np.mean(self.quality_history) if self.quality_history else 0,
            "enabled_stages": self.config.enabled_stages,
            "current_preset": self.config.preset_name,
            "config_version": self.config.version,
        }


# Factory function
def create_audio_pipeline_processor(
    config: AudioProcessingConfig,
    sample_rate: int = 16000
) -> AudioPipelineProcessor:
    """Create and return an AudioPipelineProcessor instance."""
    return AudioPipelineProcessor(config, sample_rate)


# Example usage and testing
async def main():
    """Example usage of the audio pipeline processor."""
    from .config import create_audio_processing_config_from_preset
    
    # Create configuration
    config = create_audio_processing_config_from_preset("voice")
    if not config:
        config = AudioProcessingConfig()
    
    # Create processor
    processor = create_audio_pipeline_processor(config)
    
    # Generate test audio
    sample_rate = 16000
    duration = 3.0  # 3 seconds
    t = np.arange(int(duration * sample_rate)) / sample_rate
    
    # Test audio: sine wave + noise
    test_audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.randn(len(t))
    
    # Process audio
    processed_audio, metadata = processor.process_audio_chunk(test_audio.astype(np.float32))
    
    print(f"Processing metadata: {metadata}")
    print(f"Stages applied: {metadata['stages_applied']}")
    print(f"Processing time: {metadata['processing_time_ms']:.2f}ms")
    print(f"Quality improvement: {metadata.get('quality_improvement', 0):.3f}")
    
    # Get statistics
    stats = processor.get_processing_statistics()
    print(f"Processing statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())