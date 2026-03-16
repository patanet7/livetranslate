"""Conftest for unit tests.

Inherits testcontainer fixtures (Postgres, Redis) from the parent conftest.py
so that integration tests in this directory work correctly.

LLM config is centralised here — all test files use the `llm_config` fixture
instead of hardcoding URLs.
"""
import os

import pytest

from translation.config import TranslationConfig


@pytest.fixture
def llm_config():
    """Shared LLM config — reads from env vars, defaults to local vllm-mlx."""
    return TranslationConfig(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8006/v1"),
        model=os.getenv("LLM_MODEL", "mlx-community/Qwen3-4B-4bit"),
        temperature=0.7,
        timeout_s=15,
    )
