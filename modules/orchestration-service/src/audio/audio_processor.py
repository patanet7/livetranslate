#!/usr/bin/env python3
"""
Audio Processing Pipeline - Orchestration Service

Complete audio processing pipeline with VAD, noise reduction, voice enhancement, 
and compression. Integrates with the configuration system for persistent settings
and provides the same functionality as the frontend processing pipeline.

Features:
- Voice Activity Detection (WebRTC VAD, Silero VAD, Energy-based)
- Voice frequency filtering with formant preservation
- Advanced noise reduction with voice protection
- Voice enhancement (clarity, presence, warmth, brightness)
- Dynamic range compression with multiple modes
- Quality analysis and dynamic adjustment
- Real-time processing with configurable parameters
- Integration with configuration management system
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import scipy.signal
from scipy.fft import fft, ifft, rfft, irfft
from dataclasses import asdict

from .config import (
    AudioProcessingConfig,
    AudioConfigurationManager,
    VADConfig,
    VoiceFilterConfig,
    NoiseReductionConfig,
    VoiceEnhancementConfig,
    CompressionConfig,
    LimiterConfig,
    QualityConfig,
    VADMode,
    NoiseReductionMode,
    CompressionMode,
)
from .models import QualityMetrics

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


class AudioPipelineProcessor:
    """
    Complete audio processing pipeline that integrates all processing stages.
    Configurable through the AudioProcessingConfig system.
    """
    
    def __init__(self, config: AudioProcessingConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # Initialize processing stages
        self.vad = VoiceActivityDetector(config.vad, sample_rate)
        self.voice_filter = VoiceFrequencyFilter(config.voice_filter, sample_rate)
        self.noise_reducer = NoiseReducer(config.noise_reduction, sample_rate)
        self.voice_enhancer = VoiceEnhancer(config.voice_enhancement, sample_rate)
        self.compressor = DynamicCompressor(config.compression, sample_rate)
        self.limiter = AudioLimiter(config.limiter, sample_rate)
        
        # Processing statistics
        self.total_samples_processed = 0
        self.processing_times = []
        self.quality_history = []
        
        logger.info(f"AudioPipelineProcessor initialized with preset: {config.preset_name}")
    
    def update_config(self, config: AudioProcessingConfig):
        """Update processing configuration."""
        self.config = config
        
        # Update individual processors
        self.vad.config = config.vad
        self.voice_filter.config = config.voice_filter
        self.noise_reducer.config = config.noise_reduction
        self.voice_enhancer.config = config.voice_enhancement
        self.compressor.config = config.compression
        self.limiter.config = config.limiter
        
        # Reinitialize filters if needed
        self.voice_filter._design_filters()
        self.voice_enhancer._design_enhancement_filters()
        
        logger.info(f"Audio pipeline config updated to preset: {config.preset_name}")
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process audio chunk through the complete pipeline.
        
        Returns:
            Tuple of (processed_audio, processing_metadata)
        """
        start_time = time.time()
        processing_metadata = {
            "stages_applied": [],
            "stage_results": {},
            "vad_result": None,
            "quality_metrics": None,
            "processing_time_ms": 0,
            "bypassed": False,
        }
        
        try:
            # Input quality analysis
            input_quality = self._analyze_quality(audio_data)
            processing_metadata["input_quality"] = input_quality.overall_quality_score
            
            # Check if we should bypass processing for low quality input
            if (self.config.bypass_on_low_quality and 
                input_quality.overall_quality_score < self.config.quality.quality_threshold):
                processing_metadata["bypassed"] = True
                processing_metadata["bypass_reason"] = "low_input_quality"
                return audio_data, processing_metadata
            
            current_audio = audio_data.copy()
            
            # Stage 1: Voice Activity Detection
            if self.config.is_stage_enabled("vad"):
                voice_detected, vad_confidence = self.vad.detect_voice_activity(current_audio)
                processing_metadata["vad_result"] = {
                    "voice_detected": voice_detected,
                    "confidence": vad_confidence
                }
                processing_metadata["stages_applied"].append("vad")
                
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
            
            # Stage 5: Dynamic Range Compression
            if self.config.is_stage_enabled("compression"):
                current_audio = self.compressor.process_audio(current_audio)
                processing_metadata["stages_applied"].append("compression")
                
                if self.config.pause_after_stage.get("compression", False):
                    return current_audio, processing_metadata
            
            # Stage 6: Final Limiting
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