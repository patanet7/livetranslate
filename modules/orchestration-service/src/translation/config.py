"""Translation module configuration.

All settings can be overridden via environment variables with LLM_ prefix:
  LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT_S,
  LLM_CONTEXT_WINDOW_SIZE, LLM_MAX_CONTEXT_TOKENS, LLM_MAX_QUEUE_DEPTH
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class TranslationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3.5:7b"
    temperature: float = 0.3
    timeout_s: int = 5
    context_window_size: int = 5
    max_context_tokens: int = 500
    max_queue_depth: int = 10

    @property
    def llm_base_url(self) -> str:
        """Alias for backward compatibility — callers use config.llm_base_url."""
        return self.base_url

    @classmethod
    def from_env(cls) -> TranslationConfig:
        """Backward-compatible factory — pydantic-settings reads env vars automatically."""
        return cls()
