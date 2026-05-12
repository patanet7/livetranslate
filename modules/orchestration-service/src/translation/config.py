"""Translation pipeline behavioral configuration.

This config covers *how the translation pipeline behaves* — rolling context
window size, token budgets, queue depth, draft timeouts. It does NOT cover
*which LLM to call* — endpoint identity (base_url, api_key, model) and
sampling parameters (temperature, top_p, ...) live on `LLMConnection` and
are resolved via `services.llm_resolver`.

All settings can be overridden via environment variables with `LLM_` prefix:
  LLM_CONTEXT_WINDOW_SIZE, LLM_MAX_CONTEXT_TOKENS,
  LLM_CROSS_DIRECTION_MAX_TOKENS, LLM_MAX_QUEUE_DEPTH,
  LLM_DRAFT_MAX_TOKENS, LLM_DRAFT_TIMEOUT_S
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class TranslationConfig(BaseSettings):
    # extra="ignore" keeps the env-var → config path resilient: stray legacy
    # env vars like LLM_BASE_URL or kwargs from old call sites no longer
    # populate fields here (they live on LLMConnection now) but they don't
    # crash construction either.
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    context_window_size: int = 5
    max_context_tokens: int = 800
    cross_direction_max_tokens: int = 200
    max_queue_depth: int = 10
    draft_max_tokens: int = 256
    draft_timeout_s: int = 4

    @classmethod
    def from_env(cls) -> TranslationConfig:
        """Backward-compatible factory — pydantic-settings reads env vars automatically."""
        return cls()
