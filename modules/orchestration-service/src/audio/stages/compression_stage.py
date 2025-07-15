#!/usr/bin/env python3
"""
Compression Stage

Modular compression implementation that can be used independently
or as part of the complete audio processing pipeline.
"""

import numpy as np
from typing import Dict, Any, Tuple
from ..stage_components import BaseAudioStage
from ..config import CompressionConfig, CompressionMode


class CompressionStage(BaseAudioStage):
    """Dynamic range compression stage component."""
    
    def __init__(self, config: CompressionConfig, sample_rate: int = 16000):
        super().__init__("compression", config, sample_rate)
        
        # Compression state
        self.gain_reduction = 1.0
        self.envelope = 0.0
        
        # Convert parameters
        self.threshold_linear = 10 ** (config.threshold / 20)
        self.attack_coeff = np.exp(-1.0 / (config.attack_time * sample_rate / 1000))
        self.release_coeff = np.exp(-1.0 / (config.release_time * sample_rate / 1000))
        
        self.is_initialized = True
    
    def _process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio through compression."""
        try:
            processed = np.zeros_like(audio_data)
            gain_reductions = []
            
            for i, sample in enumerate(audio_data):
                # Calculate envelope
                sample_level = abs(sample)
                
                if sample_level > self.envelope:
                    # Attack
                    self.envelope = sample_level + self.attack_coeff * (self.envelope - sample_level)
                else:
                    # Release
                    self.envelope = sample_level + self.release_coeff * (self.envelope - sample_level)
                
                # Calculate gain reduction
                if self.envelope > self.threshold_linear:
                    # Calculate excess over threshold
                    excess = self.envelope / self.threshold_linear
                    
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
                gain_reductions.append(self.gain_reduction)
            
            # Calculate compression metrics
            average_gain_reduction = np.mean(gain_reductions)
            max_gain_reduction = np.min(gain_reductions)  # Min because reduction < 1.0
            
            metadata = {
                "mode": self.config.mode.value,
                "threshold_db": self.config.threshold,
                "ratio": self.config.ratio,
                "makeup_gain_db": self.config.makeup_gain,
                "average_gain_reduction": average_gain_reduction,
                "average_gain_reduction_db": 20 * np.log10(average_gain_reduction),
                "max_gain_reduction": max_gain_reduction,
                "max_gain_reduction_db": 20 * np.log10(max_gain_reduction),
                "knee_width": self.config.knee,
                "attack_time_ms": self.config.attack_time,
                "release_time_ms": self.config.release_time
            }
            
            return processed, metadata
            
        except Exception as e:
            raise Exception(f"Compression failed: {e}")
    
    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        return {
            "enabled": self.config.enabled,
            "mode": self.config.mode.value,
            "threshold": self.config.threshold,
            "ratio": self.config.ratio,
            "knee": self.config.knee,
            "attack_time": self.config.attack_time,
            "release_time": self.config.release_time,
            "makeup_gain": self.config.makeup_gain,
            "lookahead": self.config.lookahead
        }
    
    def get_compression_curve(self, input_levels_db: np.ndarray = None) -> Dict[str, Any]:
        """Get compression curve for visualization."""
        if input_levels_db is None:
            input_levels_db = np.linspace(-60, 0, 100)
        
        output_levels_db = []
        
        for input_db in input_levels_db:
            input_linear = 10 ** (input_db / 20)
            
            if input_linear > self.threshold_linear:
                # Above threshold
                excess = input_linear / self.threshold_linear
                
                if self.config.mode == CompressionMode.SOFT_KNEE:
                    knee_ratio = min(1.0, excess / (10 ** (self.config.knee / 20)))
                    gain_reduction = 1.0 - (1.0 - 1.0/self.config.ratio) * knee_ratio
                else:
                    gain_reduction = 1.0 / self.config.ratio + (1.0 - 1.0/self.config.ratio) / excess
                
                output_linear = input_linear * gain_reduction
            else:
                # Below threshold
                output_linear = input_linear
            
            # Apply makeup gain
            makeup_gain_linear = 10 ** (self.config.makeup_gain / 20)
            output_linear *= makeup_gain_linear
            
            output_db = 20 * np.log10(max(output_linear, 1e-10))
            output_levels_db.append(output_db)
        
        return {
            "input_levels_db": input_levels_db.tolist(),
            "output_levels_db": output_levels_db,
            "threshold_db": self.config.threshold,
            "ratio": self.config.ratio,
            "knee_width_db": self.config.knee,
            "makeup_gain_db": self.config.makeup_gain
        }