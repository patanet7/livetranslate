#!/usr/bin/env python3
"""
Spectral Denoising Stage

Advanced frequency-domain noise reduction using spectral subtraction,
Wiener filtering, and adaptive noise estimation. Can be used independently
or as part of the complete modular audio processing pipeline.
"""

import numpy as np
import scipy.signal
from typing import Dict, Any, Tuple
from ..stage_components import BaseAudioStage
from ..config import SpectralDenoisingConfig, SpectralDenoisingMode


class SpectralDenoisingStage(BaseAudioStage):
    """Advanced spectral domain noise reduction stage component."""
    
    def __init__(self, config: SpectralDenoisingConfig, sample_rate: int = 16000):
        super().__init__("spectral_denoising", config, sample_rate)
        
        # FFT parameters
        self.fft_size = config.fft_size
        self.hop_size = self.fft_size // 4  # 75% overlap
        self.window = np.hanning(self.fft_size)
        
        # Noise estimation
        self.noise_spectrum = None
        self.noise_estimation_frames = 0
        self.noise_update_rate = config.noise_update_rate
        
        # Spectral processing buffers
        self.input_buffer = np.zeros(self.fft_size)
        self.output_buffer = np.zeros(self.fft_size)
        self.overlap_buffer = np.zeros(self.fft_size - self.hop_size)
        
        # Smoothing filters
        self.alpha_smooth = config.smoothing_factor
        self.prev_magnitude = None
        self.prev_phase = None
        
        # Wiener filter state
        self.signal_power_est = None
        self.noise_power_est = None
        
        self.is_initialized = True
    
    def _process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio through spectral denoising."""
        try:
            # Apply input gain
            if abs(self.config.gain_in) > 0.1:
                input_gain_linear = 10 ** (self.config.gain_in / 20)
                processed = audio_data * input_gain_linear
            else:
                processed = audio_data.copy()
            
            # Process in overlapping frames
            output_frames = []
            total_noise_reduced = 0.0
            spectral_floor_applied = 0
            frames_processed = 0
            
            # Pad input for processing
            padded_input = np.concatenate([self.input_buffer, processed])
            
            for i in range(0, len(processed), self.hop_size):
                if i + self.fft_size <= len(padded_input):
                    frame = padded_input[i:i + self.fft_size]
                    
                    # Process frame through spectral denoising
                    denoised_frame, frame_stats = self._process_spectral_frame(frame)
                    output_frames.append(denoised_frame)
                    
                    total_noise_reduced += frame_stats.get("noise_reduction_db", 0.0)
                    spectral_floor_applied += frame_stats.get("spectral_floor_applied", 0)
                    frames_processed += 1
            
            # Reconstruct output
            if output_frames:
                output_audio = self._reconstruct_audio(output_frames)
                # Trim to original length
                if len(output_audio) > len(processed):
                    output_audio = output_audio[:len(processed)]
                elif len(output_audio) < len(processed):
                    # Pad if necessary
                    padding = np.zeros(len(processed) - len(output_audio))
                    output_audio = np.concatenate([output_audio, padding])
                processed = output_audio
            
            # Update input buffer for next chunk
            if len(padded_input) >= self.fft_size:
                self.input_buffer = padded_input[-self.fft_size:]
            
            # Apply output gain
            if abs(self.config.gain_out) > 0.1:
                output_gain_linear = 10 ** (self.config.gain_out / 20)
                processed = processed * output_gain_linear
            
            # Calculate denoising metrics
            avg_noise_reduction = total_noise_reduced / max(frames_processed, 1)
            spectral_floor_percent = (spectral_floor_applied / max(frames_processed, 1)) * 100
            
            metadata = {
                "mode": self.config.mode.value,
                "fft_size": self.fft_size,
                "frames_processed": frames_processed,
                "avg_noise_reduction_db": avg_noise_reduction,
                "spectral_floor_percent": spectral_floor_percent,
                "noise_estimation_frames": self.noise_estimation_frames,
                "gain_in_db": self.config.gain_in,
                "gain_out_db": self.config.gain_out,
                "smoothing_factor": self.config.smoothing_factor,
                "noise_floor_db": self.config.noise_floor_db
            }
            
            return processed, metadata
            
        except Exception as e:
            raise Exception(f"Spectral denoising failed: {e}")
    
    def _process_spectral_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame through spectral denoising."""
        # Apply window
        windowed_frame = frame * self.window
        
        # FFT
        spectrum = np.fft.rfft(windowed_frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Initialize noise estimation if needed
        if self.noise_spectrum is None:
            self.noise_spectrum = magnitude.copy()
            self.signal_power_est = magnitude ** 2
            self.noise_power_est = magnitude ** 2
        
        # Update noise estimation
        self._update_noise_estimation(magnitude)
        
        # Apply spectral denoising based on mode
        if self.config.mode == SpectralDenoisingMode.SPECTRAL_SUBTRACTION:
            denoised_magnitude = self._spectral_subtraction(magnitude)
        elif self.config.mode == SpectralDenoisingMode.WIENER_FILTER:
            denoised_magnitude = self._wiener_filtering(magnitude)
        elif self.config.mode == SpectralDenoisingMode.ADAPTIVE:
            denoised_magnitude = self._adaptive_denoising(magnitude)
        else:  # MINIMAL
            denoised_magnitude = self._minimal_denoising(magnitude)
        
        # Apply smoothing
        if self.prev_magnitude is not None:
            denoised_magnitude = (self.alpha_smooth * self.prev_magnitude + 
                                (1 - self.alpha_smooth) * denoised_magnitude)
        
        self.prev_magnitude = denoised_magnitude.copy()
        
        # Reconstruct spectrum
        denoised_spectrum = denoised_magnitude * np.exp(1j * phase)
        
        # IFFT
        denoised_frame = np.fft.irfft(denoised_spectrum, n=self.fft_size)
        
        # Apply window for reconstruction
        denoised_frame = denoised_frame * self.window
        
        # Calculate frame statistics
        noise_reduction_db = 20 * np.log10(np.mean(magnitude) / max(np.mean(denoised_magnitude), 1e-10))
        spectral_floor_applied = np.sum(denoised_magnitude <= (magnitude * self.config.spectral_floor))
        
        frame_stats = {
            "noise_reduction_db": noise_reduction_db,
            "spectral_floor_applied": spectral_floor_applied
        }
        
        return denoised_frame, frame_stats
    
    def _update_noise_estimation(self, magnitude: np.ndarray):
        """Update noise spectrum estimation."""
        # Simple voice activity detection based on energy
        frame_energy = np.mean(magnitude ** 2)
        energy_threshold = np.mean(self.noise_power_est) * self.config.vad_threshold
        
        if frame_energy < energy_threshold or self.noise_estimation_frames < 10:
            # Update noise estimate
            update_rate = self.noise_update_rate
            self.noise_spectrum = (update_rate * self.noise_spectrum + 
                                 (1 - update_rate) * magnitude)
            self.noise_power_est = (update_rate * self.noise_power_est + 
                                  (1 - update_rate) * magnitude ** 2)
            self.noise_estimation_frames += 1
        else:
            # Update signal estimate
            self.signal_power_est = (0.9 * self.signal_power_est + 
                                   0.1 * magnitude ** 2)
    
    def _spectral_subtraction(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction denoising."""
        # Calculate noise reduction factor
        alpha = self.config.reduction_strength
        
        # Spectral subtraction formula
        denoised_magnitude = magnitude - alpha * self.noise_spectrum
        
        # Apply spectral floor
        spectral_floor = self.config.spectral_floor * magnitude
        denoised_magnitude = np.maximum(denoised_magnitude, spectral_floor)
        
        return denoised_magnitude
    
    def _wiener_filtering(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply Wiener filtering."""
        # Calculate SNR estimate
        signal_power = magnitude ** 2
        noise_power = self.noise_spectrum ** 2
        snr = signal_power / (noise_power + 1e-10)
        
        # Wiener filter gain
        wiener_gain = snr / (snr + 1)
        
        # Apply gain reduction based on strength setting
        wiener_gain = wiener_gain ** self.config.reduction_strength
        
        # Apply spectral floor
        min_gain = self.config.spectral_floor
        wiener_gain = np.maximum(wiener_gain, min_gain)
        
        denoised_magnitude = magnitude * wiener_gain
        
        return denoised_magnitude
    
    def _adaptive_denoising(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply adaptive denoising combining multiple methods."""
        # Calculate noise level
        noise_level = np.mean(self.noise_spectrum)
        signal_level = np.mean(magnitude)
        snr_estimate = signal_level / (noise_level + 1e-10)
        
        # Choose method based on SNR
        if snr_estimate > 10:  # High SNR - use minimal processing
            alpha = 0.3 * self.config.reduction_strength
            denoised_magnitude = magnitude - alpha * self.noise_spectrum
        elif snr_estimate > 3:  # Medium SNR - use Wiener filtering
            denoised_magnitude = self._wiener_filtering(magnitude)
        else:  # Low SNR - use aggressive spectral subtraction
            alpha = 1.5 * self.config.reduction_strength
            denoised_magnitude = magnitude - alpha * self.noise_spectrum
        
        # Apply spectral floor
        spectral_floor = self.config.spectral_floor * magnitude
        denoised_magnitude = np.maximum(denoised_magnitude, spectral_floor)
        
        return denoised_magnitude
    
    def _minimal_denoising(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply minimal denoising for high-quality sources."""
        # Very gentle noise reduction
        alpha = 0.2 * self.config.reduction_strength
        denoised_magnitude = magnitude - alpha * self.noise_spectrum
        
        # High spectral floor to preserve quality
        spectral_floor = 0.8 * magnitude
        denoised_magnitude = np.maximum(denoised_magnitude, spectral_floor)
        
        return denoised_magnitude
    
    def _reconstruct_audio(self, frames: list) -> np.ndarray:
        """Reconstruct audio from processed frames with overlap-add."""
        if not frames:
            return np.array([])
        
        total_length = len(frames) * self.hop_size + self.fft_size - self.hop_size
        output = np.zeros(total_length)
        
        for i, frame in enumerate(frames):
            start_idx = i * self.hop_size
            end_idx = start_idx + self.fft_size
            if end_idx <= len(output):
                output[start_idx:end_idx] += frame
        
        return output
    
    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "gain_in": self.config.gain_in,
            "gain_out": self.config.gain_out,
            "mode": self.config.mode.value,
            "reduction_strength": self.config.reduction_strength,
            "spectral_floor": self.config.spectral_floor,
            "fft_size": self.config.fft_size,
            "smoothing_factor": self.config.smoothing_factor,
            "noise_update_rate": self.config.noise_update_rate,
            "vad_threshold": self.config.vad_threshold,
            "noise_floor_db": self.config.noise_floor_db
        }
    
    def reset_state(self):
        """Reset spectral denoising state."""
        self.noise_spectrum = None
        self.noise_estimation_frames = 0
        self.input_buffer.fill(0)
        self.output_buffer.fill(0)
        self.overlap_buffer.fill(0)
        self.prev_magnitude = None
        self.prev_phase = None
        self.signal_power_est = None
        self.noise_power_est = None
    
    def get_noise_spectrum(self) -> Dict[str, Any]:
        """Get current noise spectrum for visualization."""
        if self.noise_spectrum is None:
            return {"error": "Noise spectrum not yet estimated"}
        
        # Calculate frequency bins
        freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)
        
        # Convert to dB
        noise_spectrum_db = 20 * np.log10(self.noise_spectrum + 1e-10)
        
        return {
            "frequencies": freqs.tolist(),
            "noise_spectrum_db": noise_spectrum_db.tolist(),
            "estimation_frames": self.noise_estimation_frames,
            "is_converged": self.noise_estimation_frames >= 20
        }
    
    def update_config(self, new_config: SpectralDenoisingConfig):
        """Update configuration and reset state if FFT size changed."""
        old_fft_size = self.config.fft_size
        super().update_config(new_config)
        
        # Reset if FFT parameters changed
        if new_config.fft_size != old_fft_size:
            self.fft_size = new_config.fft_size
            self.hop_size = self.fft_size // 4
            self.window = np.hanning(self.fft_size)
            self.reset_state()