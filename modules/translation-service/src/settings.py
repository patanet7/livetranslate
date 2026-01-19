"""
Translation Service Settings

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
        env_prefix="TRANSLATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5003, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Flask secret key",
    )


class ModelSettings(BaseSettings):
    """Translation model configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="TRANSLATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "huggingface",
        description="Directory containing translation models",
    )
    default_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Default translation model",
    )
    fallback_model: str = Field(
        default="llama3.1:8b",
        description="Fallback Ollama model",
    )


class DeviceSettings(BaseSettings):
    """Hardware device configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    device: Literal["gpu", "cpu", "auto"] = Field(
        default="auto",
        alias="DEVICE",
        description="Inference device (gpu, cpu, auto)",
    )
    cuda_visible_devices: str | None = Field(
        default=None,
        alias="CUDA_VISIBLE_DEVICES",
        description="CUDA visible devices",
    )
    gpu_memory_utilization: float = Field(
        default=0.85,
        alias="GPU_MEMORY_UTILIZATION",
        description="GPU memory utilization fraction (0.0-1.0)",
    )

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_gpu_memory(cls, v: float) -> float:
        """Validate GPU memory utilization is in valid range."""
        if not 0.0 < v <= 1.0:
            raise ValueError("GPU memory utilization must be between 0 and 1")
        return v


class BackendSettings(BaseSettings):
    """Translation backend configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # vLLM settings
    vllm_enabled: bool = Field(
        default=False,
        alias="VLLM_ENABLED",
        description="Enable vLLM backend",
    )
    vllm_url: str = Field(
        default="http://localhost:8000",
        alias="VLLM_URL",
        description="vLLM server URL",
    )
    vllm_tensor_parallel_size: int = Field(
        default=1,
        alias="VLLM_TENSOR_PARALLEL_SIZE",
        description="vLLM tensor parallel size",
    )

    # Ollama settings
    ollama_enabled: bool = Field(
        default=True,
        alias="OLLAMA_ENABLED",
        description="Enable Ollama backend",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_URL",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        alias="OLLAMA_MODEL",
        description="Ollama model to use",
    )

    # Triton settings
    triton_enabled: bool = Field(
        default=False,
        alias="TRITON_ENABLED",
        description="Enable Triton backend",
    )
    triton_url: str = Field(
        default="localhost:8001",
        alias="TRITON_URL",
        description="Triton server URL",
    )

    # Groq settings
    groq_enabled: bool = Field(
        default=False,
        alias="GROQ_ENABLED",
        description="Enable Groq API backend",
    )
    groq_api_key: str | None = Field(
        default=None,
        alias="GROQ_API_KEY",
        description="Groq API key",
    )
    groq_model: str = Field(
        default="llama-3.1-70b-versatile",
        alias="GROQ_MODEL",
        description="Groq model to use",
    )


class ServiceIntegrationSettings(BaseSettings):
    """Service integration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    orchestration_url: str = Field(
        default="http://localhost:3000",
        alias="ORCHESTRATION_URL",
        description="Orchestration service URL",
    )
    whisper_url: str = Field(
        default="http://localhost:5001",
        alias="WHISPER_URL",
        description="Whisper service URL",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
        description="Redis connection URL",
    )


class QualitySettings(BaseSettings):
    """Translation quality settings."""

    model_config = SettingsConfigDict(
        env_prefix="TRANSLATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    quality_threshold: float = Field(
        default=0.7,
        description="Minimum translation quality threshold",
    )
    enable_validation: bool = Field(
        default=True,
        description="Enable translation validation",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum translation retries on failure",
    )


class PerformanceSettings(BaseSettings):
    """Performance tuning settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_batch_size: int = Field(
        default=32,
        alias="MAX_BATCH_SIZE",
        description="Maximum batch size for translation",
    )
    max_tokens: int = Field(
        default=1024,
        alias="MAX_TOKENS",
        description="Maximum tokens per translation",
    )
    temperature: float = Field(
        default=0.3,
        alias="TEMPERATURE",
        description="Model temperature for generation",
    )
    request_timeout: float = Field(
        default=30.0,
        alias="REQUEST_TIMEOUT",
        description="Request timeout in seconds",
    )


class TranslationSettings(BaseSettings):
    """
    Main Translation service settings.

    Aggregates all settings categories for easy access.

    Usage:
        from settings import get_settings

        settings = get_settings()
        print(settings.server.port)
        print(settings.model.default_model)
        print(settings.backend.ollama_url)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    device: DeviceSettings = Field(default_factory=DeviceSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    integration: ServiceIntegrationSettings = Field(default_factory=ServiceIntegrationSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)


# Singleton instance
_settings: TranslationSettings | None = None


def get_settings() -> TranslationSettings:
    """
    Get the singleton settings instance.

    Returns:
        TranslationSettings: The settings instance.
    """
    global _settings
    if _settings is None:
        _settings = TranslationSettings()
    return _settings


def reload_settings() -> TranslationSettings:
    """
    Force reload settings from environment.

    Returns:
        TranslationSettings: The new settings instance.
    """
    global _settings
    _settings = TranslationSettings()
    return _settings
