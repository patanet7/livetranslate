#!/usr/bin/env python3
"""
Equalizer Stage

Modular equalizer implementation that can be used independently
or as part of the complete audio processing pipeline.
"""

import numpy as np
import scipy.signal
from typing import Dict, Any, Tuple, List
from ..stage_components import BaseAudioStage
from ..config import EqualizerConfig, EqualizerBand


class EqualizerStage(BaseAudioStage):
    """Multi-band parametric equalizer stage component."""
    
    def __init__(self, config: EqualizerConfig, sample_rate: int = 16000):
        super().__init__("equalizer", config, sample_rate)
        
        # Equalizer state
        self.filters = []
        self.filter_states = []
        
        # Design filters for each band
        self._design_filters()
        
        self.is_initialized = True
    
    def _design_filters(self):
        """Design filters for each equalizer band."""
        self.filters = []
        self.filter_states = []
        
        nyquist = self.sample_rate / 2
        
        for band in self.config.bands:
            if not band.enabled or abs(band.gain) < 0.1:
                # Skip disabled or minimal gain bands
                self.filters.append(None)
                self.filter_states.append(None)
                continue
            
            try:
                # Design filter based on band type
                if band.filter_type == "peaking":
                    filter_coeff = self._design_peaking_filter(band, nyquist)
                elif band.filter_type == "low_shelf":
                    filter_coeff = self._design_shelf_filter(band, nyquist, "low")
                elif band.filter_type == "high_shelf":
                    filter_coeff = self._design_shelf_filter(band, nyquist, "high")
                elif band.filter_type == "low_pass":
                    filter_coeff = self._design_basic_filter(band, nyquist, "low")
                elif band.filter_type == "high_pass":
                    filter_coeff = self._design_basic_filter(band, nyquist, "high")
                else:
                    # Default to peaking
                    filter_coeff = self._design_peaking_filter(band, nyquist)
                
                self.filters.append(filter_coeff)
                # Initialize filter state (zeros for IIR filter)
                self.filter_states.append(np.zeros(max(len(filter_coeff[0]), len(filter_coeff[1])) - 1))
                
            except Exception as e:
                # Skip problematic bands
                self.filters.append(None)
                self.filter_states.append(None)
    
    def _design_peaking_filter(self, band: EqualizerBand, nyquist: float) -> Tuple[np.ndarray, np.ndarray]:
        """Design peaking filter for mid-frequency bands."""
        freq_norm = band.frequency / nyquist
        freq_norm = max(0.001, min(0.999, freq_norm))  # Clamp to valid range
        
        # Convert gain from dB to linear
        gain_linear = 10 ** (band.gain / 20)
        
        # Calculate Q from bandwidth (octaves)
        # Q = fc / bandwidth_hz
        bandwidth_hz = band.frequency * (2 ** band.bandwidth - 1) / (2 ** band.bandwidth + 1)
        q_factor = band.frequency / max(bandwidth_hz, 1.0)
        q_factor = max(0.1, min(q_factor, 30.0))  # Reasonable Q range
        
        # Design peaking filter using scipy
        if abs(band.gain) > 0.1:
            b, a = scipy.signal.iirpeak(freq_norm, q_factor)
            
            # Apply gain adjustment
            if band.gain > 0:
                # Boost
                b = b * gain_linear
            else:
                # Cut - invert the filter response
                b = b / gain_linear
        else:
            # No gain - pass through
            b, a = np.array([1.0]), np.array([1.0])
        
        return b, a
    
    def _design_shelf_filter(self, band: EqualizerBand, nyquist: float, shelf_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Design shelf filter for low/high frequency ranges."""
        freq_norm = band.frequency / nyquist
        freq_norm = max(0.001, min(0.999, freq_norm))
        
        # Convert gain from dB to linear
        gain_linear = 10 ** (band.gain / 20)
        
        # Use butter filter and apply gain
        if shelf_type == "low":
            b, a = scipy.signal.butter(2, freq_norm, btype='low')
        else:  # high
            b, a = scipy.signal.butter(2, freq_norm, btype='high')
        
        # Apply gain to the filter
        if band.gain != 0:
            b = b * (gain_linear if band.gain > 0 else 1/abs(gain_linear))
        
        return b, a
    
    def _design_basic_filter(self, band: EqualizerBand, nyquist: float, filter_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Design basic low/high pass filter."""
        freq_norm = band.frequency / nyquist
        freq_norm = max(0.001, min(0.999, freq_norm))
        
        # Design basic filter
        b, a = scipy.signal.butter(2, freq_norm, btype=filter_type)
        
        return b, a
    
    def _process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio through equalizer."""
        try:
            processed = audio_data.copy()
            
            # Apply input gain
            if abs(self.config.gain_in) > 0.1:
                input_gain_linear = 10 ** (self.config.gain_in / 20)
                processed = processed * input_gain_linear
            
            bands_applied = []
            band_gains = []
            
            # Apply each enabled filter band
            for i, (band, filter_coeff, filter_state) in enumerate(zip(self.config.bands, self.filters, self.filter_states)):
                if filter_coeff is None or not band.enabled:
                    continue
                
                try:
                    # Apply filter with state management
                    if len(processed) > 0:
                        if filter_state is not None:
                            # Use lfilter with initial conditions for real-time processing
                            filtered_audio, new_state = scipy.signal.lfilter(
                                filter_coeff[0], filter_coeff[1], processed, zi=filter_state
                            )
                            self.filter_states[i] = new_state
                        else:
                            # Fallback to filtfilt for offline processing
                            filtered_audio = scipy.signal.filtfilt(
                                filter_coeff[0], filter_coeff[1], processed
                            )
                        
                        # Mix with original based on gain
                        if band.gain > 0:
                            # Boost - add filtered signal
                            mix_ratio = min(abs(band.gain) / 20.0, 1.0)  # Scale gain
                            processed = processed + (filtered_audio - processed) * mix_ratio
                        else:
                            # Cut - subtract filtered signal
                            mix_ratio = min(abs(band.gain) / 20.0, 1.0)
                            processed = processed - (filtered_audio - processed) * mix_ratio
                        
                        bands_applied.append(f"{band.filter_type}_{band.frequency}Hz")
                        band_gains.append(band.gain)
                
                except Exception as e:
                    # Skip problematic bands but continue processing
                    continue
            
            # Apply output gain
            if abs(self.config.gain_out) > 0.1:
                output_gain_linear = 10 ** (self.config.gain_out / 20)
                processed = processed * output_gain_linear
            
            # Calculate EQ metrics
            input_rms = np.sqrt(np.mean(audio_data ** 2)) if len(audio_data) > 0 else 0.0
            output_rms = np.sqrt(np.mean(processed ** 2)) if len(processed) > 0 else 0.0
            
            metadata = {
                "bands_applied": bands_applied,
                "band_gains_db": band_gains,
                "total_bands": len(self.config.bands),
                "enabled_bands": len(bands_applied),
                "gain_in_db": self.config.gain_in,
                "gain_out_db": self.config.gain_out,
                "input_rms": input_rms,
                "output_rms": output_rms,
                "level_change_db": 20 * np.log10(output_rms / max(input_rms, 1e-10)),
                "eq_preset": getattr(self.config, 'preset_name', 'custom')
            }
            
            return processed, metadata
            
        except Exception as e:
            raise Exception(f"Equalization failed: {e}")
    
    def _get_stage_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        band_configs = []
        for band in self.config.bands:
            band_configs.append({
                "enabled": band.enabled,
                "frequency": band.frequency,
                "gain": band.gain,
                "bandwidth": band.bandwidth,
                "filter_type": band.filter_type
            })
        
        return {
            "enabled": self.config.enabled,
            "gain_in": self.config.gain_in,
            "gain_out": self.config.gain_out,
            "bands": band_configs,
            "preset_name": getattr(self.config, 'preset_name', 'custom')
        }
    
    def get_frequency_response(self, frequencies: np.ndarray = None) -> Dict[str, Any]:
        """Get frequency response of the equalizer."""
        if frequencies is None:
            frequencies = np.logspace(1, 4, 100)  # 10 Hz to 10 kHz
        
        # Calculate combined frequency response
        w = 2 * np.pi * frequencies / self.sample_rate
        combined_response = np.ones(len(frequencies), dtype=complex)
        
        band_responses = []
        
        for i, (band, filter_coeff) in enumerate(zip(self.config.bands, self.filters)):
            if filter_coeff is None or not band.enabled:
                band_responses.append(np.ones(len(frequencies)))
                continue
            
            try:
                # Calculate frequency response for this band
                _, h = scipy.signal.freqs(filter_coeff[0], filter_coeff[1], w)
                combined_response *= h
                
                # Store individual band response
                band_responses.append(np.abs(h))
                
            except Exception:
                band_responses.append(np.ones(len(frequencies)))
        
        # Convert to dB
        magnitude_db = 20 * np.log10(np.abs(combined_response))
        phase_deg = np.angle(combined_response) * 180 / np.pi
        
        return {
            "frequencies": frequencies.tolist(),
            "magnitude_db": magnitude_db.tolist(),
            "phase_deg": phase_deg.tolist(),
            "band_responses": band_responses,
            "overall_gain_db": self.config.overall_gain
        }
    
    def update_band(self, band_index: int, frequency: float = None, gain: float = None, 
                   bandwidth: float = None, enabled: bool = None):
        """Update a specific equalizer band."""
        if 0 <= band_index < len(self.config.bands):
            band = self.config.bands[band_index]
            
            if frequency is not None:
                band.frequency = frequency
            if gain is not None:
                band.gain = gain
            if bandwidth is not None:
                band.bandwidth = bandwidth
            if enabled is not None:
                band.enabled = enabled
            
            # Redesign filters with updated configuration
            self._design_filters()
    
    def update_config(self, new_config: EqualizerConfig):
        """Update configuration and redesign filters."""
        super().update_config(new_config)
        self._design_filters()
    
    def reset_state(self):
        """Reset equalizer state."""
        for i in range(len(self.filter_states)):
            if self.filter_states[i] is not None:
                self.filter_states[i].fill(0)
    
    def apply_preset(self, preset_name: str):
        """Apply a predefined equalizer preset."""
        presets = {
            "flat": self._create_flat_preset(),
            "voice_enhance": self._create_voice_enhance_preset(),
            "bass_boost": self._create_bass_boost_preset(),
            "treble_boost": self._create_treble_boost_preset(),
            "mid_cut": self._create_mid_cut_preset(),
            "broadcast": self._create_broadcast_preset()
        }
        
        if preset_name in presets:
            preset_config = presets[preset_name]
            self.config.bands = preset_config["bands"]
            self.config.gain_in = preset_config.get("gain_in", 0.0)
            self.config.gain_out = preset_config.get("gain_out", 0.0)
            self._design_filters()
    
    def _create_flat_preset(self) -> Dict[str, Any]:
        """Create flat/neutral EQ preset."""
        return {
            "bands": [
                EqualizerBand(enabled=False, frequency=80, gain=0.0, bandwidth=1.0, filter_type="high_pass"),
                EqualizerBand(enabled=False, frequency=200, gain=0.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=False, frequency=800, gain=0.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=False, frequency=3200, gain=0.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=False, frequency=8000, gain=0.0, bandwidth=1.0, filter_type="high_shelf")
            ],
            "gain_in": 0.0,
            "gain_out": 0.0
        }
    
    def _create_voice_enhance_preset(self) -> Dict[str, Any]:
        """Create voice enhancement EQ preset."""
        return {
            "bands": [
                EqualizerBand(enabled=True, frequency=80, gain=-6.0, bandwidth=1.0, filter_type="high_pass"),
                EqualizerBand(enabled=True, frequency=200, gain=-3.0, bandwidth=1.5, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=1000, gain=2.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=3000, gain=3.0, bandwidth=1.2, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=8000, gain=1.0, bandwidth=1.0, filter_type="high_shelf")
            ],
            "gain_in": 0.0,
            "gain_out": 0.0
        }
    
    def _create_bass_boost_preset(self) -> Dict[str, Any]:
        """Create bass boost EQ preset."""
        return {
            "bands": [
                EqualizerBand(enabled=True, frequency=60, gain=4.0, bandwidth=1.0, filter_type="low_shelf"),
                EqualizerBand(enabled=True, frequency=120, gain=3.0, bandwidth=1.5, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=250, gain=1.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=False, frequency=1000, gain=0.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=False, frequency=4000, gain=0.0, bandwidth=1.0, filter_type="peaking")
            ],
            "gain_in": 0.0,
            "gain_out": -2.0  # Compensate for boost
        }
    
    def _create_treble_boost_preset(self) -> Dict[str, Any]:
        """Create treble boost EQ preset."""
        return {
            "bands": [
                EqualizerBand(enabled=False, frequency=100, gain=0.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=False, frequency=400, gain=0.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=2000, gain=2.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=5000, gain=3.0, bandwidth=1.2, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=10000, gain=4.0, bandwidth=1.0, filter_type="high_shelf")
            ],
            "gain_in": 0.0,
            "gain_out": -1.0
        }
    
    def _create_mid_cut_preset(self) -> Dict[str, Any]:
        """Create mid-cut EQ preset (scooped sound)."""
        return {
            "bands": [
                EqualizerBand(enabled=True, frequency=100, gain=2.0, bandwidth=1.0, filter_type="low_shelf"),
                EqualizerBand(enabled=True, frequency=400, gain=-4.0, bandwidth=2.0, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=1200, gain=-3.0, bandwidth=1.5, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=3000, gain=-2.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=8000, gain=3.0, bandwidth=1.0, filter_type="high_shelf")
            ],
            "gain_in": 0.0,
            "gain_out": 0.0
        }
    
    def _create_broadcast_preset(self) -> Dict[str, Any]:
        """Create broadcast/radio EQ preset."""
        return {
            "bands": [
                EqualizerBand(enabled=True, frequency=80, gain=-12.0, bandwidth=1.0, filter_type="high_pass"),
                EqualizerBand(enabled=True, frequency=200, gain=-2.0, bandwidth=1.0, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=800, gain=1.0, bandwidth=1.5, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=2500, gain=3.0, bandwidth=1.2, filter_type="peaking"),
                EqualizerBand(enabled=True, frequency=7000, gain=-6.0, bandwidth=1.0, filter_type="low_pass")
            ],
            "gain_in": 0.0,
            "gain_out": 0.0
        }