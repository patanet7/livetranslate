#!/usr/bin/env python3
"""
Conventional Denoising Stage

Time-domain noise reduction using classical signal processing techniques.
Provides fast, low-latency noise reduction with various filtering algorithms.
Complements spectral denoising for optimal noise reduction performance.
"""

import numpy as np
import scipy.signal
from typing import Dict, Any, Tuple, Optional
from ..stage_components import BaseAudioStage
from ..config import ConventionalDenoisingConfig, ConventionalDenoisingMode


class ConventionalDenoisingStage(BaseAudioStage):
    """Conventional (time-domain) noise reduction stage component."""
    
    def __init__(self, config: ConventionalDenoisingConfig, sample_rate: int = 16000):
        super().__init__("conventional_denoising", config, sample_rate)
        
        # Processing state
        self.history_buffer = np.zeros(config.window_size * 2)
        self.noise_estimate = 0.0
        self.signal_estimate = 0.0
        self.adaptation_state = np.zeros(config.window_size)
        
        # Adaptive filter coefficients
        self.filter_coeffs = np.ones(config.window_size) / config.window_size
        self.reference_signal = np.zeros(config.window_size)
        
        # Transient detection
        self.prev_energy = 0.0
        self.energy_history = np.zeros(5)
        
        # High frequency emphasis filter
        if config.high_freq_emphasis > 0:
            # Simple high-pass filter for emphasis
            cutoff = 3000  # Hz
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            self.emphasis_b, self.emphasis_a = scipy.signal.butter(2, normalized_cutoff, btype='high')
            self.emphasis_zi = None
        else:
            self.emphasis_b = self.emphasis_a = self.emphasis_zi = None
        
        # Wavelet setup (if available)
        self.wavelet_available = False
        try:
            import pywt
            self.pywt = pywt
            self.wavelet_available = True
        except ImportError:
            self.pywt = None
        
        self.is_initialized = True
    
    def _process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio through conventional denoising."""
        try:
            # Apply input gain
            if abs(self.config.gain_in) > 0.1:
                input_gain_linear = 10 ** (self.config.gain_in / 20)
                processed = audio_data * input_gain_linear
            else:
                processed = audio_data.copy()
            
            if self.config.mode == ConventionalDenoisingMode.DISABLED:
                # Apply output gain and return
                if abs(self.config.gain_out) > 0.1:
                    output_gain_linear = 10 ** (self.config.gain_out / 20)
                    processed = processed * output_gain_linear
                
                return processed, {
                    "mode": "disabled",
                    "gain_in_db": self.config.gain_in,
                    "gain_out_db": self.config.gain_out
                }
            
            # Choose denoising algorithm
            if self.config.mode == ConventionalDenoisingMode.MEDIAN_FILTER:
                denoised, stats = self._median_filter_denoising(processed)
            elif self.config.mode == ConventionalDenoisingMode.GAUSSIAN_FILTER:
                denoised, stats = self._gaussian_filter_denoising(processed)
            elif self.config.mode == ConventionalDenoisingMode.BILATERAL_FILTER:
                denoised, stats = self._bilateral_filter_denoising(processed)
            elif self.config.mode == ConventionalDenoisingMode.WAVELET_DENOISING:
                denoised, stats = self._wavelet_denoising(processed)
            elif self.config.mode == ConventionalDenoisingMode.ADAPTIVE_FILTER:
                denoised, stats = self._adaptive_filter_denoising(processed)
            elif self.config.mode == ConventionalDenoisingMode.RNR_FILTER:
                denoised, stats = self._rnr_filter_denoising(processed)
            else:
                # Fallback to median filter
                denoised, stats = self._median_filter_denoising(processed)
            
            # Apply output gain
            if abs(self.config.gain_out) > 0.1:
                output_gain_linear = 10 ** (self.config.gain_out / 20)
                denoised = denoised * output_gain_linear
            
            # Calculate quality metrics
            input_rms = np.sqrt(np.mean(processed ** 2))
            output_rms = np.sqrt(np.mean(denoised ** 2))
            noise_reduction_db = 20 * np.log10(input_rms / max(output_rms, 1e-10))
            
            metadata = {
                "mode": self.config.mode.value,
                "strength": self.config.strength,
                "window_size": self.config.window_size,
                "noise_reduction_db": noise_reduction_db,
                "input_rms": input_rms,
                "output_rms": output_rms,
                "gain_in_db": self.config.gain_in,
                "gain_out_db": self.config.gain_out,
                **stats
            }
            
            return denoised, metadata
            
        except Exception as e:
            raise Exception(f"Conventional denoising failed: {e}")
    
    def _median_filter_denoising(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply median filtering for impulse noise removal."""
        kernel_size = self.config.median_kernel_size
        
        # Apply median filter
        filtered = scipy.signal.medfilt(audio_data, kernel_size=kernel_size)
        
        # Blend with original based on strength
        denoised = (1 - self.config.strength) * audio_data + self.config.strength * filtered
        
        stats = {
            "kernel_size": kernel_size,
            "filter_type": "median"
        }
        
        return denoised, stats
    
    def _gaussian_filter_denoising(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply Gaussian filtering for noise smoothing."""
        from scipy.ndimage import gaussian_filter1d
        
        # Apply Gaussian filter
        filtered = gaussian_filter1d(audio_data, sigma=self.config.gaussian_sigma)
        
        # Preserve transients if enabled
        if self.config.preserve_transients:
            # Detect transients (rapid energy changes)
            energy = audio_data ** 2
            energy_smooth = gaussian_filter1d(energy, sigma=2.0)
            transient_mask = energy > (energy_smooth * 2.0)
            
            # Preserve original signal during transients
            filtered[transient_mask] = audio_data[transient_mask]
        
        # Blend with original based on strength
        denoised = (1 - self.config.strength) * audio_data + self.config.strength * filtered
        
        stats = {
            "gaussian_sigma": self.config.gaussian_sigma,
            "transients_preserved": self.config.preserve_transients,
            "filter_type": "gaussian"
        }
        
        return denoised, stats
    
    def _bilateral_filter_denoising(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply bilateral filtering for edge-preserving denoising."""
        # Simplified bilateral filter implementation for 1D audio
        window_size = self.config.window_size
        sigma_color = self.config.bilateral_sigma_color
        sigma_space = self.config.bilateral_sigma_space
        
        denoised = np.zeros_like(audio_data)
        half_window = window_size // 2
        
        # Pad audio for edge handling
        padded = np.pad(audio_data, half_window, mode='reflect')
        
        for i in range(len(audio_data)):
            center_val = audio_data[i]
            window = padded[i:i + window_size]
            
            # Spatial weights (distance from center)
            spatial_weights = np.exp(-0.5 * ((np.arange(window_size) - half_window) ** 2) / (sigma_space ** 2))
            
            # Color weights (intensity difference)
            color_weights = np.exp(-0.5 * ((window - center_val) ** 2) / (sigma_color ** 2))
            
            # Combined weights
            weights = spatial_weights * color_weights
            weights = weights / np.sum(weights)
            
            # Weighted average
            denoised[i] = np.sum(window * weights)
        
        # Blend with original based on strength
        denoised = (1 - self.config.strength) * audio_data + self.config.strength * denoised
        
        stats = {
            "sigma_color": sigma_color,
            "sigma_space": sigma_space,
            "window_size": window_size,
            "filter_type": "bilateral"
        }
        
        return denoised, stats
    
    def _wavelet_denoising(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply wavelet denoising."""
        if not self.wavelet_available:
            # Fallback to Gaussian filter
            return self._gaussian_filter_denoising(audio_data)
        
        try:
            # Wavelet decomposition
            coeffs = self.pywt.wavedec(audio_data, self.config.wavelet_type, level=self.config.wavelet_levels)
            
            # Estimate noise level from detail coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # Threshold calculation (Donoho-Johnstone)
            threshold = sigma * np.sqrt(2 * np.log(len(audio_data))) * self.config.strength
            
            # Apply thresholding to detail coefficients
            coeffs_thresh = coeffs.copy()
            for i in range(1, len(coeffs)):  # Skip approximation coefficients
                if self.config.wavelet_threshold_mode == "soft":
                    coeffs_thresh[i] = self.pywt.threshold(coeffs[i], threshold, mode='soft')
                else:
                    coeffs_thresh[i] = self.pywt.threshold(coeffs[i], threshold, mode='hard')
            
            # Wavelet reconstruction
            denoised = self.pywt.waverec(coeffs_thresh, self.config.wavelet_type)
            
            # Ensure same length as input
            if len(denoised) != len(audio_data):
                denoised = denoised[:len(audio_data)]
            
            stats = {
                "wavelet_type": self.config.wavelet_type,
                "decomposition_levels": self.config.wavelet_levels,
                "threshold_mode": self.config.wavelet_threshold_mode,
                "threshold_value": threshold,
                "noise_sigma": sigma,
                "filter_type": "wavelet"
            }
            
            return denoised, stats
            
        except Exception as e:
            # Fallback to Gaussian filter on error
            return self._gaussian_filter_denoising(audio_data)
    
    def _adaptive_filter_denoising(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply adaptive filtering with LMS algorithm."""
        window_size = self.config.window_size
        mu = self.config.adaptation_rate  # Learning rate
        
        denoised = np.zeros_like(audio_data)
        
        # Initialize if needed
        if len(self.filter_coeffs) != window_size:
            self.filter_coeffs = np.ones(window_size) / window_size
            self.reference_signal = np.zeros(window_size)
        
        for i in range(len(audio_data)):
            # Update reference signal buffer
            self.reference_signal = np.roll(self.reference_signal, -1)
            self.reference_signal[-1] = audio_data[i]
            
            # Filter output
            y = np.dot(self.filter_coeffs, self.reference_signal)
            
            # Error calculation (assume noise estimate)
            if i > window_size:
                # Simple noise estimation based on local variance
                local_var = np.var(audio_data[max(0, i-window_size):i])
                noise_threshold = self.config.threshold * np.sqrt(local_var)
                
                if abs(audio_data[i] - y) > noise_threshold:
                    # Likely noise, adapt filter
                    error = audio_data[i] - y
                    # LMS update
                    self.filter_coeffs += mu * error * self.reference_signal
                    denoised[i] = y  # Use filtered output
                else:
                    # Likely signal, use original
                    denoised[i] = audio_data[i]
            else:
                denoised[i] = audio_data[i]
        
        # Blend with original based on strength
        denoised = (1 - self.config.strength) * audio_data + self.config.strength * denoised
        
        stats = {
            "adaptation_rate": mu,
            "window_size": window_size,
            "filter_coeffs_mean": np.mean(self.filter_coeffs),
            "filter_type": "adaptive_lms"
        }
        
        return denoised, stats
    
    def _rnr_filter_denoising(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply Reduce Noise and Reverb filtering."""
        # Combine multiple techniques for comprehensive noise reduction
        
        # 1. High-frequency emphasis (if enabled)
        if self.emphasis_b is not None:
            if self.emphasis_zi is None:
                # Initialize filter state
                self.emphasis_zi = scipy.signal.lfilter_zi(self.emphasis_b, self.emphasis_a)
                emphasized, self.emphasis_zi = scipy.signal.lfilter(
                    self.emphasis_b, self.emphasis_a, audio_data, zi=self.emphasis_zi
                )
            else:
                emphasized, self.emphasis_zi = scipy.signal.lfilter(
                    self.emphasis_b, self.emphasis_a, audio_data, zi=self.emphasis_zi
                )
            
            # Blend emphasis
            audio_with_emphasis = (1 - self.config.high_freq_emphasis) * audio_data + \
                                 self.config.high_freq_emphasis * emphasized
        else:
            audio_with_emphasis = audio_data
        
        # 2. Median filtering for impulse noise
        median_filtered = scipy.signal.medfilt(audio_with_emphasis, kernel_size=3)
        
        # 3. Gaussian smoothing for general noise
        from scipy.ndimage import gaussian_filter1d
        gaussian_filtered = gaussian_filter1d(median_filtered, sigma=1.0)
        
        # 4. Energy-based gating
        # Calculate local energy
        window_size = max(3, self.config.window_size // 2)
        energy = np.convolve(audio_data ** 2, np.ones(window_size) / window_size, mode='same')
        
        # Adaptive threshold based on energy distribution
        energy_threshold = np.percentile(energy, 30) * (1 + self.config.threshold)
        
        # Gate based on energy
        gate_mask = energy > energy_threshold
        
        # Apply gating
        gated = np.where(gate_mask, gaussian_filtered, gaussian_filtered * 0.3)
        
        # Final blend with original
        strength = self.config.strength
        denoised = (1 - strength) * audio_data + strength * gated
        
        stats = {
            "high_freq_emphasis": self.config.high_freq_emphasis,
            "energy_threshold": energy_threshold,
            "gated_samples": np.sum(~gate_mask),
            "total_samples": len(audio_data),
            "filter_type": "rnr_composite"
        }
        
        return denoised, stats
    
    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "gain_in": self.config.gain_in,
            "gain_out": self.config.gain_out,
            "mode": self.config.mode.value,
            "strength": self.config.strength,
            "window_size": self.config.window_size,
            "threshold": self.config.threshold,
            "adaptation_rate": self.config.adaptation_rate,
            "preserve_transients": self.config.preserve_transients,
            "high_freq_emphasis": self.config.high_freq_emphasis,
            "median_kernel_size": self.config.median_kernel_size,
            "gaussian_sigma": self.config.gaussian_sigma,
            "bilateral_sigma_color": self.config.bilateral_sigma_color,
            "bilateral_sigma_space": self.config.bilateral_sigma_space,
            "wavelet_type": self.config.wavelet_type,
            "wavelet_levels": self.config.wavelet_levels,
            "wavelet_threshold_mode": self.config.wavelet_threshold_mode,
            "wavelet_available": self.wavelet_available
        }
    
    def reset_state(self):
        """Reset conventional denoising state."""
        self.history_buffer.fill(0)
        self.noise_estimate = 0.0
        self.signal_estimate = 0.0
        self.adaptation_state.fill(0)
        self.filter_coeffs = np.ones(self.config.window_size) / self.config.window_size
        self.reference_signal.fill(0)
        self.prev_energy = 0.0
        self.energy_history.fill(0)
        if self.emphasis_zi is not None:
            self.emphasis_zi = scipy.signal.lfilter_zi(self.emphasis_b, self.emphasis_a) * 0
    
    def update_config(self, new_config: ConventionalDenoisingConfig):
        """Update configuration and reset state if necessary."""
        old_window_size = self.config.window_size
        super().update_config(new_config)
        
        # Reset if window size changed
        if new_config.window_size != old_window_size:
            self.history_buffer = np.zeros(new_config.window_size * 2)
            self.adaptation_state = np.zeros(new_config.window_size)
            self.filter_coeffs = np.ones(new_config.window_size) / new_config.window_size
            self.reference_signal = np.zeros(new_config.window_size)
        
        # Update high frequency emphasis filter if needed
        if new_config.high_freq_emphasis != self.config.high_freq_emphasis:
            if new_config.high_freq_emphasis > 0:
                cutoff = 3000  # Hz
                nyquist = self.sample_rate / 2
                normalized_cutoff = cutoff / nyquist
                self.emphasis_b, self.emphasis_a = scipy.signal.butter(2, normalized_cutoff, btype='high')
                self.emphasis_zi = None
            else:
                self.emphasis_b = self.emphasis_a = self.emphasis_zi = None