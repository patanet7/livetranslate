#!/usr/bin/env python3
"""
Limiter Stage

Modular limiter implementation that can be used independently
or as part of the complete audio processing pipeline.
"""

import numpy as np
from typing import Dict, Any, Tuple
from ..stage_components import BaseAudioStage
from ..config import LimiterConfig


class LimiterStage(BaseAudioStage):
    """Final limiting stage component."""
    
    def __init__(self, config: LimiterConfig, sample_rate: int = 16000):
        super().__init__("limiter", config, sample_rate)
        
        # Limiter state
        self.delay_buffer = np.zeros(int(config.lookahead * sample_rate / 1000))
        self.buffer_index = 0
        self.gain_reduction = 1.0
        
        # Convert threshold and release time
        self.threshold_linear = 10 ** (config.threshold / 20)
        self.release_samples = int(config.release_time * sample_rate / 1000)
        
        self.is_initialized = True
    
    def _process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio through limiting."""
        try:
            processed = np.zeros_like(audio_data)
            peak_values = []
            gain_reductions = []
            clipping_events = 0
            
            for i, sample in enumerate(audio_data):
                # Add sample to delay buffer
                delayed_sample = self.delay_buffer[self.buffer_index]
                self.delay_buffer[self.buffer_index] = sample
                self.buffer_index = (self.buffer_index + 1) % len(self.delay_buffer)
                
                # Peak detection on current sample
                peak = abs(sample)
                peak_values.append(peak)
                
                if peak > self.threshold_linear:
                    # Calculate required gain reduction
                    required_gain = self.threshold_linear / peak
                    self.gain_reduction = min(self.gain_reduction, required_gain)
                    clipping_events += 1
                else:
                    # Release
                    self.gain_reduction += (1.0 - self.gain_reduction) / self.release_samples
                    self.gain_reduction = min(1.0, self.gain_reduction)
                
                # Apply gain reduction to delayed sample
                if self.config.soft_clip and abs(delayed_sample * self.gain_reduction) > self.threshold_linear:
                    # Soft clipping using tanh
                    sign = 1 if delayed_sample >= 0 else -1
                    processed[i] = sign * self.threshold_linear * np.tanh(
                        abs(delayed_sample * self.gain_reduction) / self.threshold_linear
                    )
                else:
                    processed[i] = delayed_sample * self.gain_reduction
                
                gain_reductions.append(self.gain_reduction)
            
            # Calculate limiting metrics
            max_peak = np.max(peak_values) if peak_values else 0.0
            average_gain_reduction = np.mean(gain_reductions)
            limiting_active_percent = (clipping_events / len(audio_data)) * 100 if len(audio_data) > 0 else 0.0
            
            metadata = {
                "threshold_db": self.config.threshold,
                "max_peak_linear": max_peak,
                "max_peak_db": 20 * np.log10(max(max_peak, 1e-10)),
                "average_gain_reduction": average_gain_reduction,
                "average_gain_reduction_db": 20 * np.log10(average_gain_reduction),
                "clipping_events": clipping_events,
                "limiting_active_percent": limiting_active_percent,
                "soft_clip": self.config.soft_clip,
                "release_time_ms": self.config.release_time,
                "lookahead_ms": self.config.lookahead
            }
            
            return processed, metadata
            
        except Exception as e:
            raise Exception(f"Limiting failed: {e}")
    
    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "threshold": self.config.threshold,
            "release_time": self.config.release_time,
            "lookahead": self.config.lookahead,
            "soft_clip": self.config.soft_clip
        }
    
    def get_limiting_curve(self, input_levels_db: np.ndarray = None) -> Dict[str, Any]:
        """Get limiting curve for visualization."""
        if input_levels_db is None:
            input_levels_db = np.linspace(-20, 6, 100)  # -20dB to +6dB
        
        output_levels_db = []
        
        for input_db in input_levels_db:
            input_linear = 10 ** (input_db / 20)
            
            if input_linear > self.threshold_linear:
                if self.config.soft_clip:
                    # Soft clipping curve
                    output_linear = self.threshold_linear * np.tanh(input_linear / self.threshold_linear)
                else:
                    # Hard limiting
                    output_linear = self.threshold_linear
            else:
                # Below threshold - no limiting
                output_linear = input_linear
            
            output_db = 20 * np.log10(max(output_linear, 1e-10))
            output_levels_db.append(output_db)
        
        return {
            "input_levels_db": input_levels_db.tolist(),
            "output_levels_db": output_levels_db,
            "threshold_db": self.config.threshold,
            "soft_clip": self.config.soft_clip
        }
    
    def reset_state(self):
        """Reset limiter state."""
        self.delay_buffer.fill(0)
        self.buffer_index = 0
        self.gain_reduction = 1.0