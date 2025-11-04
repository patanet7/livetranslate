#!/usr/bin/env python3
"""
Whisper Service Configuration System

Centralized configuration for all whisper-service components with validation.
Extracts magic numbers and provides environment variable support.

Usage:
    from config import WhisperConfig, VADConfig, LIDConfig

    config = WhisperConfig.from_env()
    vad_config = VADConfig.from_env()
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Voice Activity Detection Configuration"""

    # Core VAD parameters
    threshold: float = 0.5
    sampling_rate: int = 16000
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 100

    # Silence detection thresholds
    silence_threshold_chunks: int = 10  # No output for 10 chunks (~5s at 0.5s/chunk)

    # Buffer size (Silero VAD requirement)
    vad_chunk_size: int = 512  # Silero VAD requires exactly 512 samples

    def __post_init__(self):
        """Validate configuration"""
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError(f"VAD threshold must be in [0.0, 1.0], got {self.threshold}")

        if self.sampling_rate not in [8000, 16000]:
            raise ValueError(f"VAD sampling_rate must be 8000 or 16000, got {self.sampling_rate}")

        if self.min_silence_duration_ms < 100:
            raise ValueError(f"min_silence_duration_ms must be >= 100ms, got {self.min_silence_duration_ms}")

        if self.silence_threshold_chunks < 1:
            raise ValueError(f"silence_threshold_chunks must be >= 1, got {self.silence_threshold_chunks}")

        logger.debug(f"VADConfig initialized: threshold={self.threshold}, min_silence={self.min_silence_duration_ms}ms")

    @classmethod
    def from_env(cls) -> 'VADConfig':
        """Load configuration from environment variables"""
        return cls(
            threshold=float(os.getenv('VAD_THRESHOLD', '0.5')),
            sampling_rate=int(os.getenv('VAD_SAMPLING_RATE', '16000')),
            min_silence_duration_ms=int(os.getenv('VAD_MIN_SILENCE_MS', '500')),
            speech_pad_ms=int(os.getenv('VAD_SPEECH_PAD_MS', '100')),
            silence_threshold_chunks=int(os.getenv('VAD_SILENCE_THRESHOLD_CHUNKS', '10'))
        )


@dataclass
class LIDConfig:
    """Language ID Detection Configuration"""

    # Frame-level LID parameters
    lid_hop_ms: int = 100  # 10Hz frame rate
    smoothing_enabled: bool = True
    smoothing_window_size: int = 5  # Median smoothing window

    # Sustained detection parameters (from FEEDBACK.md)
    confidence_margin: float = 0.2  # P(new) - P(old) threshold
    min_dwell_frames: int = 6  # Minimum consecutive frames
    min_dwell_ms: float = 250.0  # Minimum dwell time in ms

    # Viterbi smoother parameters
    viterbi_transition_cost: float = 0.3  # Prefer staying in same language
    viterbi_window_size: int = 5  # 500ms window at 10Hz

    def __post_init__(self):
        """Validate configuration"""
        if self.confidence_margin < 0.1 or self.confidence_margin > 0.5:
            raise ValueError(f"confidence_margin must be in [0.1, 0.5], got {self.confidence_margin}")

        if self.min_dwell_ms < 100:
            raise ValueError(f"min_dwell_ms must be >= 100ms, got {self.min_dwell_ms}")

        if self.min_dwell_frames < 1:
            raise ValueError(f"min_dwell_frames must be >= 1, got {self.min_dwell_frames}")

        if self.lid_hop_ms < 50 or self.lid_hop_ms > 200:
            logger.warning(f"lid_hop_ms={self.lid_hop_ms} is outside recommended range [50, 200]")

        logger.debug(f"LIDConfig initialized: margin={self.confidence_margin}, min_dwell={self.min_dwell_ms}ms")

    @classmethod
    def from_env(cls) -> 'LIDConfig':
        """Load configuration from environment variables"""
        return cls(
            lid_hop_ms=int(os.getenv('LID_HOP_MS', '100')),
            smoothing_enabled=os.getenv('LID_SMOOTHING_ENABLED', 'true').lower() == 'true',
            smoothing_window_size=int(os.getenv('LID_SMOOTHING_WINDOW', '5')),
            confidence_margin=float(os.getenv('LID_CONFIDENCE_MARGIN', '0.2')),
            min_dwell_frames=int(os.getenv('LID_MIN_DWELL_FRAMES', '6')),
            min_dwell_ms=float(os.getenv('LID_MIN_DWELL_MS', '250.0')),
            viterbi_transition_cost=float(os.getenv('LID_VITERBI_TRANSITION_COST', '0.3')),
            viterbi_window_size=int(os.getenv('LID_VITERBI_WINDOW', '5'))
        )


@dataclass
class WhisperConfig:
    """Whisper Transcription Configuration"""

    # Model configuration
    model_path: str = ""  # Required, no default
    models_dir: Optional[str] = None

    # Decoder configuration
    decoder_type: str = "greedy"  # "greedy" or "beam"
    beam_size: int = 1

    # Streaming configuration
    online_chunk_size: float = 1.2  # Seconds
    sampling_rate: int = 16000

    # Language configuration
    target_languages: List[str] = field(default_factory=lambda: ['en', 'zh'])

    # Audio processing
    audio_min_len: float = 1.0  # Minimum audio length in seconds

    # Performance tuning
    n_mels: int = 80  # Mel spectrogram bands (128 for large-v3, 80 for older models)

    def __post_init__(self):
        """Validate configuration"""
        if not self.model_path:
            raise ValueError("model_path is required")

        if self.decoder_type not in ["greedy", "beam"]:
            raise ValueError(f"decoder_type must be 'greedy' or 'beam', got {self.decoder_type}")

        if self.decoder_type == "greedy" and self.beam_size != 1:
            logger.warning(f"decoder_type='greedy' requires beam_size=1, overriding beam_size={self.beam_size}")
            self.beam_size = 1

        if self.beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {self.beam_size}")

        if self.online_chunk_size < 0.5 or self.online_chunk_size > 5.0:
            logger.warning(f"online_chunk_size={self.online_chunk_size} is outside recommended range [0.5, 5.0]")

        if self.sampling_rate not in [8000, 16000]:
            raise ValueError(f"sampling_rate must be 8000 or 16000, got {self.sampling_rate}")

        if not self.target_languages:
            raise ValueError("target_languages cannot be empty")

        # Set default models_dir if not provided
        if self.models_dir is None:
            self.models_dir = str(Path.home() / ".whisper" / "models")

        logger.debug(f"WhisperConfig initialized: model={self.model_path}, decoder={self.decoder_type}")

    @classmethod
    def from_env(cls, model_path: str) -> 'WhisperConfig':
        """Load configuration from environment variables"""
        languages_str = os.getenv('WHISPER_LANGUAGES', 'en,zh')
        languages = [lang.strip() for lang in languages_str.split(',')]

        return cls(
            model_path=model_path,
            models_dir=os.getenv('WHISPER_MODELS_DIR'),
            decoder_type=os.getenv('WHISPER_DECODER_TYPE', 'greedy'),
            beam_size=int(os.getenv('WHISPER_BEAM_SIZE', '1')),
            online_chunk_size=float(os.getenv('WHISPER_CHUNK_SIZE', '1.2')),
            sampling_rate=int(os.getenv('WHISPER_SAMPLING_RATE', '16000')),
            target_languages=languages,
            audio_min_len=float(os.getenv('WHISPER_AUDIO_MIN_LEN', '1.0')),
            n_mels=int(os.getenv('WHISPER_N_MELS', '80'))
        )


@dataclass
class SessionConfig:
    """Session-Restart Transcriber Configuration"""

    # Combine all sub-configs
    whisper: WhisperConfig
    vad: VADConfig
    lid: LIDConfig

    # Logging configuration
    log_level: str = "INFO"
    enable_performance_logging: bool = True
    enable_debug_audio_stats: bool = False  # Set to True for detailed audio debugging

    def __post_init__(self):
        """Validate configuration"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {self.log_level}")

        logger.debug(f"SessionConfig initialized with log_level={self.log_level}")

    @classmethod
    def from_env(cls, model_path: str) -> 'SessionConfig':
        """Load complete configuration from environment variables"""
        return cls(
            whisper=WhisperConfig.from_env(model_path),
            vad=VADConfig.from_env(),
            lid=LIDConfig.from_env(),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            enable_performance_logging=os.getenv('ENABLE_PERF_LOGGING', 'true').lower() == 'true',
            enable_debug_audio_stats=os.getenv('ENABLE_DEBUG_AUDIO', 'false').lower() == 'true'
        )

    def configure_logging(self):
        """Configure logging based on settings"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Adjust specific loggers for high-frequency operations
        if self.log_level == "INFO":
            # Reduce noise from high-frequency loggers
            logging.getLogger('vad_detector').setLevel(logging.WARNING)
            logging.getLogger('sustained_detector').setLevel(logging.INFO)
            logging.getLogger('lid_detector').setLevel(logging.INFO)

        logger.info(f"Logging configured: level={self.log_level}")


# Convenience function for quick setup
def load_config(model_path: str) -> SessionConfig:
    """
    Load complete configuration from environment variables.

    Args:
        model_path: Path to Whisper model

    Returns:
        SessionConfig with all settings loaded

    Example:
        config = load_config("/path/to/model.pt")
        config.configure_logging()
    """
    config = SessionConfig.from_env(model_path)
    return config
