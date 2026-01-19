"""
Whisper Service Settings

Type-safe configuration using Pydantic BaseSettings.
All settings can be overridden via environment variables.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Server configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="WHISPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5001, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    workers: int = Field(default=1, description="Number of worker processes")


class ModelSettings(BaseSettings):
    """Whisper model configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="WHISPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / ".models",
        description="Directory containing Whisper models",
    )
    default_model: str = Field(
        default="large-v3-turbo",
        description="Default Whisper model to use",
    )
    model_precision: Literal["fp32", "fp16", "int8"] = Field(
        default="fp16",
        description="Model precision for inference",
    )

    @field_validator("models_dir", mode="before")
    @classmethod
    def resolve_models_dir(cls, v: str | Path) -> Path:
        """Resolve models directory path."""
        path = Path(v).expanduser()
        if not path.is_absolute():
            path = Path(__file__).parent.parent / path
        return path


class AudioSettings(BaseSettings):
    """Audio processing configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    buffer_duration: float = Field(
        default=4.0,
        description="Audio buffer duration in seconds",
    )
    inference_interval: float = Field(
        default=3.0,
        description="Interval between inference runs in seconds",
    )
    overlap_duration: float = Field(
        default=0.2,
        description="Overlap duration between audio chunks in seconds",
    )
    enable_vad: bool = Field(
        default=True,
        alias="ENABLE_VAD",
        description="Enable Voice Activity Detection",
    )
    min_inference_interval: float = Field(
        default=0.2,
        alias="MIN_INFERENCE_INTERVAL",
        description="Minimum interval between inference runs",
    )


class DeviceSettings(BaseSettings):
    """Hardware device configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    device: Literal["npu", "gpu", "cpu", "auto"] = Field(
        default="auto",
        alias="DEVICE",
        description="Inference device (npu, gpu, cpu, auto)",
    )
    openvino_device: str | None = Field(
        default=None,
        alias="OPENVINO_DEVICE",
        description="OpenVINO device specification",
    )
    cuda_visible_devices: str | None = Field(
        default=None,
        alias="CUDA_VISIBLE_DEVICES",
        description="CUDA visible devices",
    )


class OrchestrationSettings(BaseSettings):
    """Orchestration integration settings."""

    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: bool = Field(
        default=False,
        description="Enable orchestration mode",
    )
    endpoint: str = Field(
        default="http://localhost:3000/api/audio",
        description="Orchestration service endpoint",
    )


class PerformanceSettings(BaseSettings):
    """Performance tuning settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_concurrent_requests: int = Field(
        default=10,
        alias="MAX_CONCURRENT_REQUESTS",
        description="Maximum concurrent transcription requests",
    )
    request_timeout: float = Field(
        default=30.0,
        alias="REQUEST_TIMEOUT",
        description="Request timeout in seconds",
    )
    max_audio_size_mb: int = Field(
        default=100,
        alias="MAX_AUDIO_SIZE_MB",
        description="Maximum audio file size in MB",
    )


class WhisperSettings(BaseSettings):
    """
    Main Whisper service settings.

    Aggregates all settings categories for easy access.

    Usage:
        from settings import get_settings

        settings = get_settings()
        print(settings.server.port)
        print(settings.model.default_model)
        print(settings.audio.sample_rate)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    device: DeviceSettings = Field(default_factory=DeviceSettings)
    orchestration: OrchestrationSettings = Field(default_factory=OrchestrationSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)


# Singleton instance
_settings: WhisperSettings | None = None


def get_settings() -> WhisperSettings:
    """
    Get the singleton settings instance.

    Returns:
        WhisperSettings: The settings instance.
    """
    global _settings
    if _settings is None:
        _settings = WhisperSettings()
    return _settings


def reload_settings() -> WhisperSettings:
    """
    Force reload settings from environment.

    Returns:
        WhisperSettings: The new settings instance.
    """
    global _settings
    _settings = WhisperSettings()
    return _settings
