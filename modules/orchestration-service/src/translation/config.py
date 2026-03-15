"""Translation module configuration.

All settings can be overridden via environment variables:
  LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT_S,
  LLM_CONTEXT_WINDOW_SIZE, LLM_MAX_CONTEXT_TOKENS
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TranslationConfig:
    llm_base_url: str = "http://thomas-pc:11434/v1"
    model: str = "qwen3.5:7b"
    temperature: float = 0.3
    timeout_s: int = 5
    context_window_size: int = 5
    max_context_tokens: int = 500
    max_queue_depth: int = 10

    @classmethod
    def from_env(cls) -> TranslationConfig:
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            model=os.getenv("LLM_MODEL", cls.model),
            temperature=float(os.getenv("LLM_TEMPERATURE", str(cls.temperature))),
            timeout_s=int(os.getenv("LLM_TIMEOUT_S", str(cls.timeout_s))),
            context_window_size=int(os.getenv("LLM_CONTEXT_WINDOW_SIZE", str(cls.context_window_size))),
            max_context_tokens=int(os.getenv("LLM_MAX_CONTEXT_TOKENS", str(cls.max_context_tokens))),
            max_queue_depth=int(os.getenv("LLM_MAX_QUEUE_DEPTH", str(cls.max_queue_depth))),
        )
