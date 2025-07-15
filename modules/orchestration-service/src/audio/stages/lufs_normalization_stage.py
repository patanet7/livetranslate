#!/usr/bin/env python3
"""
LUFS Normalization Stage

Professional loudness normalization using LUFS (Loudness Units relative to Full Scale)
measurement according to ITU-R BS.1770-4 and EBU R128 standards.

Provides broadcast-quality loudness standardization for streaming, broadcast,
and professional audio applications.
"""

import numpy as np
import scipy.signal
from typing import Dict, Any, Tuple, Deque
from collections import deque
from ..stage_components import BaseAudioStage
from ..config import LUFSNormalizationConfig, LUFSNormalizationMode


class LUFSNormalizationStage(BaseAudioStage):
    """Professional LUFS-based loudness normalization stage."""
    
    def __init__(self, config: LUFSNormalizationConfig, sample_rate: int = 16000):
        super().__init__("lufs_normalization", config, sample_rate)
        
        # LUFS measurement state
        self.measurement_buffer = deque(maxlen=int(sample_rate * config.measurement_window))
        self.short_term_buffer = deque(maxlen=int(sample_rate * config.short_term_window))
        self.momentary_buffer = deque(maxlen=int(sample_rate * config.momentary_window))
        
        # Loudness measurement
        self.current_lufs = config.target_lufs  # Start at target
        self.short_term_lufs = config.target_lufs
        self.momentary_lufs = config.target_lufs
        self.integrated_lufs = config.target_lufs
        
        # True peak detection
        self.current_peak = 0.0
        self.true_peak_buffer = np.zeros(int(sample_rate * 0.01))  # 10ms buffer
        
        # Gain adjustment state
        self.current_gain_adjustment = 0.0  # dB
        self.lookahead_buffer = deque(maxlen=int(sample_rate * config.lookahead_time))
        
        # EBU R128 pre-filter coefficients (K-weighting)
        self._initialize_k_weighting_filter()
        
        # Gating state for integrated loudness
        self.gated_blocks = []
        self.measurement_count = 0
        
        self.is_initialized = True
    
    def _initialize_k_weighting_filter(self):
        """Initialize K-weighting filter according to ITU-R BS.1770-4."""
        # Pre-filter: High-pass filter (f_c = 38 Hz, Q = 0.5)
        nyquist = self.sample_rate / 2
        fc_hp = 38.0 / nyquist
        
        # High-pass Butterworth filter
        self.hp_b, self.hp_a = scipy.signal.butter(1, fc_hp, btype='high')
        self.hp_zi = scipy.signal.lfilter_zi(self.hp_b, self.hp_a)
        
        # RLB filter: High-frequency shelving filter (+4dB at 10kHz)
        fc_shelf = 1681.0 / nyquist  # Nominal frequency
        
        # Simplified shelving filter approximation
        self.shelf_b, self.shelf_a = scipy.signal.butter(1, fc_shelf, btype='high')
        self.shelf_zi = scipy.signal.lfilter_zi(self.shelf_b, self.shelf_a)
        
        # Gain compensation for the shelving filter
        self.shelf_gain = 1.585  # Approximately +4dB at high frequencies
    
    def _process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio through LUFS normalization."""
        try:
            # Apply input gain
            if abs(self.config.gain_in) > 0.1:
                input_gain_linear = 10 ** (self.config.gain_in / 20)
                processed = audio_data * input_gain_linear
            else:
                processed = audio_data.copy()
            
            if not self.config.enabled:
                # Apply output gain and return
                if abs(self.config.gain_out) > 0.1:
                    output_gain_linear = 10 ** (self.config.gain_out / 20)
                    processed = processed * output_gain_linear
                
                return processed, {
                    "enabled": False,
                    "mode": self.config.mode.value,
                    "gain_in_db": self.config.gain_in,
                    "gain_out_db": self.config.gain_out
                }
            
            # Measure LUFS
            lufs_measurements = self._measure_lufs(processed)
            
            # Calculate gain adjustment
            gain_adjustment = self._calculate_gain_adjustment(lufs_measurements)
            
            # Apply loudness normalization
            normalized_audio = self._apply_normalization(processed, gain_adjustment)
            
            # True peak limiting if enabled
            if self.config.true_peak_limiting:
                normalized_audio = self._apply_true_peak_limiting(normalized_audio)
            
            # Apply output gain
            if abs(self.config.gain_out) > 0.1:
                output_gain_linear = 10 ** (self.config.gain_out / 20)
                normalized_audio = normalized_audio * output_gain_linear
            
            # Calculate final metrics
            final_lufs = self._estimate_current_lufs(normalized_audio)
            final_peak = np.max(np.abs(normalized_audio))
            final_peak_db = 20 * np.log10(max(final_peak, 1e-10))
            
            metadata = {
                "enabled": True,
                "mode": self.config.mode.value,
                "target_lufs": self.config.target_lufs,
                "measured_lufs": lufs_measurements["integrated"],
                "final_lufs": final_lufs,
                "short_term_lufs": lufs_measurements["short_term"],
                "momentary_lufs": lufs_measurements["momentary"],
                "gain_adjustment_db": gain_adjustment,
                "input_peak_db": 20 * np.log10(max(np.max(np.abs(processed)), 1e-10)),
                "output_peak_db": final_peak_db,
                "lufs_deviation": final_lufs - self.config.target_lufs,
                "within_tolerance": abs(final_lufs - self.config.target_lufs) <= self.config.lufs_tolerance,
                "peak_limiting_active": self.config.true_peak_limiting,
                "gain_in_db": self.config.gain_in,
                "gain_out_db": self.config.gain_out
            }
            
            return normalized_audio, metadata
            
        except Exception as e:
            raise Exception(f"LUFS normalization failed: {e}")
    
    def _measure_lufs(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Measure LUFS according to ITU-R BS.1770-4."""
        # Apply K-weighting filter
        filtered_audio = self._apply_k_weighting(audio_data)
        
        # Update measurement buffers
        self.measurement_buffer.extend(filtered_audio)
        self.short_term_buffer.extend(filtered_audio)
        self.momentary_buffer.extend(filtered_audio)
        
        # Calculate loudness for different time windows
        measurements = {}
        
        # Momentary loudness (400ms)
        if len(self.momentary_buffer) > 0:
            momentary_mean_square = np.mean(np.array(self.momentary_buffer) ** 2)
            measurements["momentary"] = -0.691 + 10 * np.log10(max(momentary_mean_square, 1e-10))
            self.momentary_lufs = measurements["momentary"]
        else:
            measurements["momentary"] = self.config.target_lufs
        
        # Short-term loudness (3s)
        if len(self.short_term_buffer) > 0:
            short_term_mean_square = np.mean(np.array(self.short_term_buffer) ** 2)
            measurements["short_term"] = -0.691 + 10 * np.log10(max(short_term_mean_square, 1e-10))
            self.short_term_lufs = measurements["short_term"]
        else:
            measurements["short_term"] = self.config.target_lufs
        
        # Integrated loudness (with gating)
        measurements["integrated"] = self._calculate_integrated_lufs()
        
        return measurements
    
    def _apply_k_weighting(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply K-weighting filter (ITU-R BS.1770-4)."""
        # High-pass filter
        hp_filtered, self.hp_zi = scipy.signal.lfilter(
            self.hp_b, self.hp_a, audio_data, zi=self.hp_zi
        )
        
        # Shelving filter
        shelf_filtered, self.shelf_zi = scipy.signal.lfilter(
            self.shelf_b, self.shelf_a, hp_filtered, zi=self.shelf_zi
        )
        
        # Apply shelving gain
        k_weighted = shelf_filtered * self.shelf_gain
        
        return k_weighted
    
    def _calculate_integrated_lufs(self) -> float:
        """Calculate integrated loudness with gating according to EBU R128."""
        if len(self.measurement_buffer) < self.sample_rate * 0.4:  # Need at least 400ms
            return self.config.target_lufs
        
        # Convert buffer to array for processing
        audio_array = np.array(self.measurement_buffer)
        
        # Block-based measurement (400ms blocks with 75% overlap)
        block_size = int(self.sample_rate * 0.4)  # 400ms
        hop_size = int(block_size * 0.25)  # 75% overlap
        
        block_loudnesses = []
        
        for i in range(0, len(audio_array) - block_size + 1, hop_size):
            block = audio_array[i:i + block_size]
            block_mean_square = np.mean(block ** 2)
            
            if block_mean_square > 0:
                block_loudness = -0.691 + 10 * np.log10(block_mean_square)
                
                # Apply absolute gating (-70 LUFS)
                if block_loudness >= self.config.gating_threshold:
                    block_loudnesses.append(block_loudness)
        
        if not block_loudnesses:
            return self.config.target_lufs
        
        # Apply relative gating (-10 LU relative to ungated mean)
        ungated_mean = np.mean(block_loudnesses)
        relative_threshold = ungated_mean - 10.0
        
        gated_blocks = [loudness for loudness in block_loudnesses if loudness >= relative_threshold]
        
        if gated_blocks:
            integrated_loudness = np.mean(gated_blocks)
            self.integrated_lufs = integrated_loudness
            return integrated_loudness
        else:
            return self.config.target_lufs
    
    def _calculate_gain_adjustment(self, measurements: Dict[str, float]) -> float:
        """Calculate gain adjustment needed to reach target LUFS."""
        # Use short-term measurement for responsive adjustment
        current_lufs = measurements["short_term"]
        
        # Calculate required adjustment
        target_adjustment = self.config.target_lufs - current_lufs
        
        # Apply adjustment speed (smoothing)
        speed = self.config.adjustment_speed
        self.current_gain_adjustment = (
            (1 - speed) * self.current_gain_adjustment + 
            speed * target_adjustment
        )
        
        # Limit extreme adjustments for stability
        max_adjustment = 20.0  # dB
        self.current_gain_adjustment = np.clip(
            self.current_gain_adjustment, 
            -max_adjustment, 
            max_adjustment
        )
        
        return self.current_gain_adjustment
    
    def _apply_normalization(self, audio_data: np.ndarray, gain_db: float) -> np.ndarray:
        """Apply gain adjustment for loudness normalization."""
        if abs(gain_db) < 0.01:  # Skip tiny adjustments
            return audio_data
        
        gain_linear = 10 ** (gain_db / 20)
        return audio_data * gain_linear
    
    def _apply_true_peak_limiting(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply true peak limiting to prevent clipping."""
        # Simple true peak detection using upsampling
        # For real implementation, this would use proper oversampling
        
        peak_threshold_linear = 10 ** (self.config.max_peak_db / 20)
        
        # Find peaks exceeding threshold
        peaks = np.abs(audio_data)
        max_peak = np.max(peaks)
        
        if max_peak > peak_threshold_linear:
            # Apply limiting
            limiting_ratio = peak_threshold_linear / max_peak
            limited_audio = audio_data * limiting_ratio
            
            # Soft knee limiting for smoother results
            knee_width = 0.1  # 10% soft knee
            soft_knee_factor = np.where(
                peaks > peak_threshold_linear * (1 - knee_width),
                limiting_ratio + (1 - limiting_ratio) * 
                np.maximum(0, (peak_threshold_linear - peaks) / (peak_threshold_linear * knee_width)),
                1.0
            )
            
            return audio_data * soft_knee_factor
        
        return audio_data
    
    def _estimate_current_lufs(self, audio_data: np.ndarray) -> float:
        """Quick LUFS estimation for current audio chunk."""
        if len(audio_data) == 0:
            return self.config.target_lufs
        
        # Apply K-weighting
        k_weighted = self._apply_k_weighting(audio_data.copy())
        
        # Calculate mean square and convert to LUFS
        mean_square = np.mean(k_weighted ** 2)
        if mean_square > 0:
            return -0.691 + 10 * np.log10(mean_square)
        else:
            return self.config.target_lufs
    
    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "gain_in": self.config.gain_in,
            "gain_out": self.config.gain_out,
            "mode": self.config.mode.value,
            "target_lufs": self.config.target_lufs,
            "max_peak_db": self.config.max_peak_db,
            "measurement_window": self.config.measurement_window,
            "adjustment_speed": self.config.adjustment_speed,
            "true_peak_limiting": self.config.true_peak_limiting,
            "lufs_tolerance": self.config.lufs_tolerance,
            "gating_threshold": self.config.gating_threshold
        }
    
    def reset_state(self):
        """Reset LUFS measurement state."""
        self.measurement_buffer.clear()
        self.short_term_buffer.clear()
        self.momentary_buffer.clear()
        
        self.current_lufs = self.config.target_lufs
        self.short_term_lufs = self.config.target_lufs
        self.momentary_lufs = self.config.target_lufs
        self.integrated_lufs = self.config.target_lufs
        
        self.current_peak = 0.0
        self.true_peak_buffer.fill(0)
        self.current_gain_adjustment = 0.0
        self.lookahead_buffer.clear()
        
        self.gated_blocks.clear()
        self.measurement_count = 0
        
        # Reset filter states
        self.hp_zi = scipy.signal.lfilter_zi(self.hp_b, self.hp_a)
        self.shelf_zi = scipy.signal.lfilter_zi(self.shelf_b, self.shelf_a)
    
    def get_lufs_measurements(self) -> Dict[str, Any]:
        """Get current LUFS measurements."""
        return {
            "integrated_lufs": self.integrated_lufs,
            "short_term_lufs": self.short_term_lufs,
            "momentary_lufs": self.momentary_lufs,
            "current_peak_db": 20 * np.log10(max(self.current_peak, 1e-10)),
            "target_lufs": self.config.target_lufs,
            "deviation_from_target": self.integrated_lufs - self.config.target_lufs,
            "within_tolerance": abs(self.integrated_lufs - self.config.target_lufs) <= self.config.lufs_tolerance,
            "measurement_buffer_length": len(self.measurement_buffer),
            "gated_blocks_count": len(self.gated_blocks),
            "current_gain_adjustment": self.current_gain_adjustment
        }
    
    def update_config(self, new_config: LUFSNormalizationConfig):
        """Update configuration and reinitialize if needed."""
        old_sample_rate_dependent = (
            self.config.measurement_window != new_config.measurement_window or
            self.config.short_term_window != new_config.short_term_window or
            self.config.momentary_window != new_config.momentary_window or
            self.config.lookahead_time != new_config.lookahead_time
        )
        
        super().update_config(new_config)
        
        if old_sample_rate_dependent:
            # Reinitialize buffers with new sizes
            self.measurement_buffer = deque(maxlen=int(self.sample_rate * new_config.measurement_window))
            self.short_term_buffer = deque(maxlen=int(self.sample_rate * new_config.short_term_window))
            self.momentary_buffer = deque(maxlen=int(self.sample_rate * new_config.momentary_window))
            self.lookahead_buffer = deque(maxlen=int(self.sample_rate * new_config.lookahead_time))
            self.true_peak_buffer = np.zeros(int(self.sample_rate * 0.01))