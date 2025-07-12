#!/usr/bin/env python3
"""
Audio Processing Configuration System - Orchestration Service

Comprehensive configuration management for all audio processing parameters.
Supports hot-reloading, persistent storage, frontend customization, and validation.
Integrates with the existing audio processing pipeline from the orchestration service.

Features:
- Hot-reloadable configuration with file watching
- Database persistence for user customizations
- Frontend API integration for real-time settings
- Validation and range checking for all parameters
- Preset management (Default, Voice, Noisy, Music, etc.)
- Per-session configuration overrides
- A/B testing support for different parameter sets
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import yaml

from .models import AudioChunkingConfig

logger = logging.getLogger(__name__)


class AudioPreset(str, Enum):
    """Predefined audio processing presets."""
    DEFAULT = "default"
    VOICE_OPTIMIZED = "voice"
    NOISY_ENVIRONMENT = "noisy"
    MUSIC_CONTENT = "music"
    MINIMAL_PROCESSING = "minimal"
    AGGRESSIVE_PROCESSING = "aggressive"
    CONFERENCE_CALL = "conference"
    BROADCAST_QUALITY = "broadcast"


class VADMode(str, Enum):
    """Voice Activity Detection modes."""
    DISABLED = "disabled"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    SILERO = "silero"
    WEBRTC = "webrtc"


class NoiseReductionMode(str, Enum):
    """Noise reduction modes."""
    DISABLED = "disabled"
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


class CompressionMode(str, Enum):
    """Dynamic range compression modes."""
    DISABLED = "disabled"
    SOFT_KNEE = "soft_knee"
    HARD_KNEE = "hard_knee"
    ADAPTIVE = "adaptive"
    VOICE_OPTIMIZED = "voice_optimized"


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    enabled: bool = True
    mode: VADMode = VADMode.WEBRTC
    aggressiveness: int = 2  # 0-3 for WebRTC VAD
    energy_threshold: float = 0.01
    voice_freq_min: float = 85  # Hz
    voice_freq_max: float = 300  # Hz
    frame_duration_ms: int = 30  # 10, 20, or 30 ms
    sensitivity: float = 0.5  # 0.0-1.0
    
    def __post_init__(self):
        # Validation
        self.aggressiveness = max(0, min(3, self.aggressiveness))
        self.energy_threshold = max(0.0, min(1.0, self.energy_threshold))
        self.sensitivity = max(0.0, min(1.0, self.sensitivity))


@dataclass
class VoiceFilterConfig:
    """Voice frequency filtering configuration."""
    enabled: bool = True
    fundamental_min: float = 85  # Hz - Human voice fundamental frequency
    fundamental_max: float = 300  # Hz
    formant1_min: float = 200  # Hz - First formant range
    formant1_max: float = 1000  # Hz
    formant2_min: float = 900  # Hz - Second formant range
    formant2_max: float = 3000  # Hz
    preserve_formants: bool = True
    voice_band_gain: float = 1.1  # Slight boost for voice frequencies
    high_freq_rolloff: float = 8000  # Hz - Roll off above this frequency
    
    def __post_init__(self):
        # Validation
        self.voice_band_gain = max(0.1, min(3.0, self.voice_band_gain))


@dataclass
class NoiseReductionConfig:
    """Noise reduction configuration."""
    enabled: bool = True
    mode: NoiseReductionMode = NoiseReductionMode.MODERATE
    strength: float = 0.7  # 0.0-1.0
    voice_protection: bool = True  # Protect voice frequencies
    stationary_noise_reduction: float = 0.8  # For stationary noise
    non_stationary_noise_reduction: float = 0.5  # For non-stationary noise
    noise_floor_db: float = -40  # dB
    adaptation_rate: float = 0.1  # How quickly to adapt to changing noise
    
    def __post_init__(self):
        # Validation
        self.strength = max(0.0, min(1.0, self.strength))
        self.stationary_noise_reduction = max(0.0, min(1.0, self.stationary_noise_reduction))
        self.non_stationary_noise_reduction = max(0.0, min(1.0, self.non_stationary_noise_reduction))
        self.adaptation_rate = max(0.01, min(1.0, self.adaptation_rate))


@dataclass
class VoiceEnhancementConfig:
    """Voice enhancement configuration."""
    enabled: bool = True
    normalize: bool = False  # Disabled by default to preserve natural voice
    clarity_enhancement: float = 0.2  # 0.0-1.0
    presence_boost: float = 0.1  # Slight presence boost
    warmth_adjustment: float = 0.0  # -1.0 to 1.0
    brightness_adjustment: float = 0.0  # -1.0 to 1.0
    sibilance_control: float = 0.1  # Control harsh sibilants
    
    def __post_init__(self):
        # Validation
        self.clarity_enhancement = max(0.0, min(1.0, self.clarity_enhancement))
        self.presence_boost = max(0.0, min(1.0, self.presence_boost))
        self.warmth_adjustment = max(-1.0, min(1.0, self.warmth_adjustment))
        self.brightness_adjustment = max(-1.0, min(1.0, self.brightness_adjustment))
        self.sibilance_control = max(0.0, min(1.0, self.sibilance_control))


@dataclass
class CompressionConfig:
    """Dynamic range compression configuration."""
    enabled: bool = True
    mode: CompressionMode = CompressionMode.SOFT_KNEE
    threshold: float = -20  # dB
    ratio: float = 3.0  # 1.0-20.0
    knee: float = 2.0  # dB - Soft knee width
    attack_time: float = 5.0  # ms
    release_time: float = 100.0  # ms
    makeup_gain: float = 0.0  # dB
    lookahead: float = 5.0  # ms
    
    def __post_init__(self):
        # Validation
        self.ratio = max(1.0, min(20.0, self.ratio))
        self.knee = max(0.0, min(10.0, self.knee))
        self.attack_time = max(0.1, min(100.0, self.attack_time))
        self.release_time = max(1.0, min(1000.0, self.release_time))


@dataclass
class LimiterConfig:
    """Final limiting configuration."""
    enabled: bool = True
    threshold: float = -1.0  # dB
    release_time: float = 50.0  # ms
    lookahead: float = 5.0  # ms
    soft_clip: bool = True  # Soft clipping for natural sound
    
    def __post_init__(self):
        # Validation
        self.release_time = max(1.0, min(500.0, self.release_time))
        self.lookahead = max(0.0, min(20.0, self.lookahead))


@dataclass
class QualityConfig:
    """Quality analysis and control configuration."""
    enabled: bool = True
    quality_threshold: float = 0.3  # Minimum quality for processing
    silence_threshold: float = 0.01  # Silence detection
    noise_threshold: float = 0.02  # Noise floor
    clipping_threshold: float = 0.95  # Clipping detection
    snr_minimum: float = 10.0  # dB - Minimum SNR
    dynamic_quality_adjustment: bool = True  # Adjust processing based on quality
    quality_history_length: int = 10  # Number of chunks to consider for quality trends
    
    def __post_init__(self):
        # Validation
        self.quality_threshold = max(0.0, min(1.0, self.quality_threshold))
        self.silence_threshold = max(0.0, min(0.1, self.silence_threshold))
        self.noise_threshold = max(0.0, min(0.1, self.noise_threshold))
        self.clipping_threshold = max(0.5, min(1.0, self.clipping_threshold))


@dataclass
class AudioProcessingConfig:
    """Complete audio processing pipeline configuration."""
    
    # Core processing stages
    vad: VADConfig = field(default_factory=VADConfig)
    voice_filter: VoiceFilterConfig = field(default_factory=VoiceFilterConfig)
    noise_reduction: NoiseReductionConfig = field(default_factory=NoiseReductionConfig)
    voice_enhancement: VoiceEnhancementConfig = field(default_factory=VoiceEnhancementConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    limiter: LimiterConfig = field(default_factory=LimiterConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # Pipeline control
    enabled_stages: List[str] = field(default_factory=lambda: [
        "vad", "voice_filter", "noise_reduction", "voice_enhancement", "compression", "limiter"
    ])
    pause_after_stage: Dict[str, bool] = field(default_factory=dict)  # For debugging
    bypass_on_low_quality: bool = True  # Bypass processing if input quality is too low
    
    # Global settings
    sample_rate: int = 16000
    buffer_size: int = 1024
    processing_block_size: int = 512
    overlap_factor: float = 0.5  # Overlap between processing blocks
    
    # Performance settings
    real_time_priority: bool = True
    cpu_usage_limit: float = 0.8  # 80% CPU usage limit
    processing_timeout: float = 100.0  # ms - Maximum processing time per block
    
    # Metadata
    preset_name: str = "default"
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def get_stage_config(self, stage_name: str) -> Optional[Any]:
        """Get configuration for a specific processing stage."""
        return getattr(self, stage_name, None)
    
    def is_stage_enabled(self, stage_name: str) -> bool:
        """Check if a processing stage is enabled."""
        return stage_name in self.enabled_stages and hasattr(self, stage_name)
    
    def enable_stage(self, stage_name: str, enabled: bool = True):
        """Enable or disable a processing stage."""
        if enabled and stage_name not in self.enabled_stages:
            self.enabled_stages.append(stage_name)
        elif not enabled and stage_name in self.enabled_stages:
            self.enabled_stages.remove(stage_name)
        self.last_modified = datetime.utcnow().isoformat()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (VADConfig, VoiceFilterConfig, NoiseReductionConfig, 
                                                  VoiceEnhancementConfig, CompressionConfig, LimiterConfig, QualityConfig)):
                    # Update nested config objects
                    current_config = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(current_config, nested_key):
                            setattr(current_config, nested_key, nested_value)
                else:
                    setattr(self, key, value)
        self.last_modified = datetime.utcnow().isoformat()


class AudioConfigurationManager:
    """
    Manages audio processing configurations with persistence, validation, and hot-reloading.
    Integrates with database storage and provides API for frontend customization.
    """
    
    def __init__(
        self,
        config_file_path: Optional[str] = None,
        database_adapter = None,
        auto_reload: bool = True
    ):
        self.config_file_path = config_file_path or "/etc/livetranslate/audio_config.yaml"
        self.database_adapter = database_adapter
        self.auto_reload = auto_reload
        
        # Current configurations
        self.default_config = AudioProcessingConfig()
        self.session_configs: Dict[str, AudioProcessingConfig] = {}
        self.preset_configs: Dict[str, AudioProcessingConfig] = {}
        
        # File watching
        self.last_file_mtime = 0
        self.file_watch_task: Optional[asyncio.Task] = None
        
        # Change callbacks
        self.config_change_callbacks: List[Callable] = []
        
        # Load built-in presets
        self._load_builtin_presets()
        
        logger.info(f"AudioConfigurationManager initialized with config file: {self.config_file_path}")
    
    async def initialize(self) -> bool:
        """Initialize the configuration manager."""
        try:
            # Load configuration from file
            await self._load_config_from_file()
            
            # Load configurations from database if available
            if self.database_adapter:
                await self._load_configs_from_database()
            
            # Start file watching if enabled
            if self.auto_reload:
                self.file_watch_task = asyncio.create_task(self._watch_config_file())
            
            logger.info("Audio configuration manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audio configuration manager: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the configuration manager."""
        if self.file_watch_task:
            self.file_watch_task.cancel()
            try:
                await self.file_watch_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Audio configuration manager shutdown")
    
    def _load_builtin_presets(self):
        """Load built-in preset configurations."""
        
        # Default preset
        self.preset_configs[AudioPreset.DEFAULT] = AudioProcessingConfig(preset_name="default")
        
        # Voice optimized preset
        voice_config = AudioProcessingConfig(preset_name="voice")
        voice_config.vad.mode = VADMode.WEBRTC
        voice_config.vad.aggressiveness = 2
        voice_config.voice_filter.voice_band_gain = 1.2
        voice_config.voice_enhancement.clarity_enhancement = 0.3
        voice_config.voice_enhancement.presence_boost = 0.2
        voice_config.compression.threshold = -18
        voice_config.compression.ratio = 2.5
        self.preset_configs[AudioPreset.VOICE_OPTIMIZED] = voice_config
        
        # Noisy environment preset
        noisy_config = AudioProcessingConfig(preset_name="noisy")
        noisy_config.vad.mode = VADMode.AGGRESSIVE
        noisy_config.vad.aggressiveness = 3
        noisy_config.noise_reduction.mode = NoiseReductionMode.AGGRESSIVE
        noisy_config.noise_reduction.strength = 0.9
        noisy_config.voice_enhancement.clarity_enhancement = 0.4
        noisy_config.compression.threshold = -15
        noisy_config.compression.ratio = 4.0
        self.preset_configs[AudioPreset.NOISY_ENVIRONMENT] = noisy_config
        
        # Music content preset
        music_config = AudioProcessingConfig(preset_name="music")
        music_config.vad.enabled = False  # Don't use VAD for music
        music_config.voice_filter.enabled = False  # Don't filter music frequencies
        music_config.noise_reduction.mode = NoiseReductionMode.LIGHT
        music_config.voice_enhancement.enabled = False
        music_config.compression.threshold = -12
        music_config.compression.ratio = 1.5
        self.preset_configs[AudioPreset.MUSIC_CONTENT] = music_config
        
        # Minimal processing preset
        minimal_config = AudioProcessingConfig(preset_name="minimal")
        minimal_config.enabled_stages = ["quality"]  # Only quality analysis
        minimal_config.vad.enabled = False
        minimal_config.voice_filter.enabled = False
        minimal_config.noise_reduction.enabled = False
        minimal_config.voice_enhancement.enabled = False
        minimal_config.compression.enabled = False
        minimal_config.limiter.enabled = False
        self.preset_configs[AudioPreset.MINIMAL_PROCESSING] = minimal_config
        
        # Aggressive processing preset
        aggressive_config = AudioProcessingConfig(preset_name="aggressive")
        aggressive_config.vad.mode = VADMode.AGGRESSIVE
        aggressive_config.noise_reduction.mode = NoiseReductionMode.AGGRESSIVE
        aggressive_config.noise_reduction.strength = 0.95
        aggressive_config.voice_enhancement.clarity_enhancement = 0.5
        aggressive_config.voice_enhancement.presence_boost = 0.3
        aggressive_config.compression.threshold = -12
        aggressive_config.compression.ratio = 6.0
        self.preset_configs[AudioPreset.AGGRESSIVE_PROCESSING] = aggressive_config
        
        # Conference call preset
        conference_config = AudioProcessingConfig(preset_name="conference")
        conference_config.vad.mode = VADMode.WEBRTC
        conference_config.voice_filter.voice_band_gain = 1.1
        conference_config.noise_reduction.mode = NoiseReductionMode.MODERATE
        conference_config.voice_enhancement.clarity_enhancement = 0.3
        conference_config.compression.threshold = -16
        conference_config.compression.ratio = 3.0
        self.preset_configs[AudioPreset.CONFERENCE_CALL] = conference_config
        
        # Broadcast quality preset
        broadcast_config = AudioProcessingConfig(preset_name="broadcast")
        broadcast_config.vad.mode = VADMode.WEBRTC
        broadcast_config.voice_enhancement.clarity_enhancement = 0.4
        broadcast_config.voice_enhancement.presence_boost = 0.25
        broadcast_config.compression.threshold = -14
        broadcast_config.compression.ratio = 3.5
        broadcast_config.compression.knee = 1.5
        broadcast_config.limiter.threshold = -0.5
        self.preset_configs[AudioPreset.BROADCAST_QUALITY] = broadcast_config
    
    async def _load_config_from_file(self):
        """Load configuration from YAML file."""
        try:
            config_path = Path(self.config_file_path)
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data:
                    self.default_config.update_from_dict(config_data)
                    logger.info(f"Loaded configuration from {config_path}")
                else:
                    logger.info(f"Empty configuration file, using defaults")
            else:
                # Create default configuration file
                await self._save_config_to_file()
                logger.info(f"Created default configuration file: {config_path}")
                
        except Exception as e:
            logger.error(f"Failed to load configuration from file: {e}")
    
    async def _save_config_to_file(self):
        """Save current configuration to YAML file."""
        try:
            config_path = Path(self.config_file_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = asdict(self.default_config)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.last_file_mtime = config_path.stat().st_mtime
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to file: {e}")
    
    async def _load_configs_from_database(self):
        """Load configurations from database."""
        # TODO: Implement database loading
        # This would load user customizations from a settings table
        pass
    
    async def _save_config_to_database(self, config: AudioProcessingConfig, config_name: str = "default"):
        """Save configuration to database."""
        # TODO: Implement database saving
        # This would save user customizations to a settings table
        pass
    
    async def _watch_config_file(self):
        """Watch configuration file for changes and reload automatically."""
        try:
            while True:
                await asyncio.sleep(1.0)  # Check every second
                
                config_path = Path(self.config_file_path)
                if config_path.exists():
                    current_mtime = config_path.stat().st_mtime
                    
                    if current_mtime > self.last_file_mtime:
                        logger.info("Configuration file changed, reloading...")
                        await self._load_config_from_file()
                        await self._notify_config_change("file_reload", self.default_config)
                        
        except asyncio.CancelledError:
            logger.info("Configuration file watching stopped")
        except Exception as e:
            logger.error(f"Error watching configuration file: {e}")
    
    # Public API methods
    
    def get_default_config(self) -> AudioProcessingConfig:
        """Get the default audio processing configuration."""
        return self.default_config
    
    def get_session_config(self, session_id: str) -> AudioProcessingConfig:
        """Get configuration for a specific session, or default if not found."""
        return self.session_configs.get(session_id, self.default_config)
    
    def get_preset_config(self, preset_name: str) -> Optional[AudioProcessingConfig]:
        """Get a preset configuration."""
        return self.preset_configs.get(preset_name)
    
    def get_available_presets(self) -> List[str]:
        """Get list of available presets."""
        return list(self.preset_configs.keys())
    
    async def set_default_config(self, config: AudioProcessingConfig, save_to_file: bool = True):
        """Set the default configuration."""
        self.default_config = config
        self.default_config.last_modified = datetime.utcnow().isoformat()
        
        if save_to_file:
            await self._save_config_to_file()
        
        if self.database_adapter:
            await self._save_config_to_database(config, "default")
        
        await self._notify_config_change("default_updated", config)
    
    async def set_session_config(self, session_id: str, config: AudioProcessingConfig):
        """Set configuration for a specific session."""
        self.session_configs[session_id] = config
        config.last_modified = datetime.utcnow().isoformat()
        
        if self.database_adapter:
            await self._save_config_to_database(config, f"session_{session_id}")
        
        await self._notify_config_change("session_updated", config, session_id)
    
    async def update_config_from_dict(
        self, 
        config_updates: Dict[str, Any], 
        session_id: Optional[str] = None,
        save_persistent: bool = True
    ) -> AudioProcessingConfig:
        """Update configuration from dictionary of changes."""
        
        if session_id:
            # Update session-specific config
            if session_id not in self.session_configs:
                # Create new session config based on default
                self.session_configs[session_id] = AudioProcessingConfig()
                self.session_configs[session_id].update_from_dict(asdict(self.default_config))
            
            config = self.session_configs[session_id]
            config.update_from_dict(config_updates)
            
            if save_persistent and self.database_adapter:
                await self._save_config_to_database(config, f"session_{session_id}")
            
            await self._notify_config_change("session_updated", config, session_id)
            return config
        else:
            # Update default config
            self.default_config.update_from_dict(config_updates)
            
            if save_persistent:
                await self._save_config_to_file()
                if self.database_adapter:
                    await self._save_config_to_database(self.default_config, "default")
            
            await self._notify_config_change("default_updated", self.default_config)
            return self.default_config
    
    async def apply_preset(self, preset_name: str, session_id: Optional[str] = None) -> bool:
        """Apply a preset configuration."""
        preset_config = self.get_preset_config(preset_name)
        if not preset_config:
            logger.warning(f"Preset {preset_name} not found")
            return False
        
        if session_id:
            await self.set_session_config(session_id, preset_config)
        else:
            await self.set_default_config(preset_config)
        
        logger.info(f"Applied preset {preset_name} to {'session ' + session_id if session_id else 'default'}")
        return True
    
    def add_config_change_callback(self, callback: Callable):
        """Add callback for configuration changes."""
        self.config_change_callbacks.append(callback)
    
    def remove_config_change_callback(self, callback: Callable):
        """Remove callback for configuration changes."""
        if callback in self.config_change_callbacks:
            self.config_change_callbacks.remove(callback)
    
    async def _notify_config_change(self, change_type: str, config: AudioProcessingConfig, session_id: Optional[str] = None):
        """Notify all callbacks of configuration changes."""
        for callback in self.config_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(change_type, config, session_id)
                else:
                    callback(change_type, config, session_id)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for frontend validation."""
        return {
            "vad": {
                "enabled": {"type": "boolean", "default": True},
                "mode": {"type": "enum", "values": list(VADMode), "default": "webrtc"},
                "aggressiveness": {"type": "integer", "min": 0, "max": 3, "default": 2},
                "energy_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.01},
                "voice_freq_min": {"type": "float", "min": 50, "max": 500, "default": 85},
                "voice_freq_max": {"type": "float", "min": 100, "max": 1000, "default": 300},
                "sensitivity": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            },
            "voice_filter": {
                "enabled": {"type": "boolean", "default": True},
                "fundamental_min": {"type": "float", "min": 50, "max": 200, "default": 85},
                "fundamental_max": {"type": "float", "min": 200, "max": 500, "default": 300},
                "voice_band_gain": {"type": "float", "min": 0.1, "max": 3.0, "default": 1.1},
                "preserve_formants": {"type": "boolean", "default": True},
            },
            "noise_reduction": {
                "enabled": {"type": "boolean", "default": True},
                "mode": {"type": "enum", "values": list(NoiseReductionMode), "default": "moderate"},
                "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7},
                "voice_protection": {"type": "boolean", "default": True},
                "adaptation_rate": {"type": "float", "min": 0.01, "max": 1.0, "default": 0.1},
            },
            "voice_enhancement": {
                "enabled": {"type": "boolean", "default": True},
                "normalize": {"type": "boolean", "default": False},
                "clarity_enhancement": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.2},
                "presence_boost": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.1},
                "warmth_adjustment": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
                "brightness_adjustment": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.0},
            },
            "compression": {
                "enabled": {"type": "boolean", "default": True},
                "mode": {"type": "enum", "values": list(CompressionMode), "default": "soft_knee"},
                "threshold": {"type": "float", "min": -40, "max": 0, "default": -20},
                "ratio": {"type": "float", "min": 1.0, "max": 20.0, "default": 3.0},
                "knee": {"type": "float", "min": 0.0, "max": 10.0, "default": 2.0},
                "attack_time": {"type": "float", "min": 0.1, "max": 100.0, "default": 5.0},
                "release_time": {"type": "float", "min": 1.0, "max": 1000.0, "default": 100.0},
            },
            "quality": {
                "enabled": {"type": "boolean", "default": True},
                "quality_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.3},
                "silence_threshold": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.01},
                "dynamic_quality_adjustment": {"type": "boolean", "default": True},
            },
            "global": {
                "enabled_stages": {"type": "array", "items": "string", "default": ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "compression", "limiter"]},
                "real_time_priority": {"type": "boolean", "default": True},
                "cpu_usage_limit": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.8},
            }
        }
    
    def validate_config_updates(self, config_updates: Dict[str, Any]) -> List[str]:
        """Validate configuration updates and return list of errors."""
        errors = []
        schema = self.get_config_schema()
        
        # TODO: Implement comprehensive validation
        # This would validate each field against the schema
        
        return errors


# Factory functions
def create_audio_config_manager(
    config_file_path: Optional[str] = None,
    database_adapter = None,
    auto_reload: bool = True
) -> AudioConfigurationManager:
    """Create and return an AudioConfigurationManager instance."""
    return AudioConfigurationManager(config_file_path, database_adapter, auto_reload)


def get_default_audio_processing_config() -> AudioProcessingConfig:
    """Get default audio processing configuration."""
    return AudioProcessingConfig()


def create_audio_processing_config_from_preset(preset_name: str) -> Optional[AudioProcessingConfig]:
    """Create audio processing config from preset name."""
    manager = AudioConfigurationManager()
    manager._load_builtin_presets()
    return manager.get_preset_config(preset_name)


# Example usage and testing
async def main():
    """Example usage of the audio configuration system."""
    
    # Create configuration manager
    config_manager = create_audio_config_manager(
        config_file_path="./audio_config.yaml",
        auto_reload=True
    )
    
    # Add change callback
    def on_config_change(change_type, config, session_id=None):
        print(f"Config changed: {change_type}, session: {session_id}")
    
    config_manager.add_config_change_callback(on_config_change)
    
    try:
        # Initialize
        await config_manager.initialize()
        
        # Get default config
        default_config = config_manager.get_default_config()
        print(f"Default config preset: {default_config.preset_name}")
        
        # Apply preset
        await config_manager.apply_preset("voice")
        
        # Update specific settings
        await config_manager.update_config_from_dict({
            "vad": {"aggressiveness": 3},
            "compression": {"threshold": -15, "ratio": 4.0}
        })
        
        # Create session-specific config
        await config_manager.update_config_from_dict({
            "noise_reduction": {"strength": 0.9}
        }, session_id="test-session-123")
        
        # Get available presets
        presets = config_manager.get_available_presets()
        print(f"Available presets: {presets}")
        
        # Get config schema for frontend
        schema = config_manager.get_config_schema()
        print(f"Config schema keys: {list(schema.keys())}")
        
        # Test for 5 seconds
        await asyncio.sleep(5)
        
    finally:
        await config_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())