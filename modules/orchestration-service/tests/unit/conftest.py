"""Conftest for unit tests.

Inherits testcontainer fixtures (Postgres, Redis) from the parent conftest.py
so that integration tests in this directory work correctly.

After the LLM/translation config consolidation:
- `translation_behavioral` returns the (reduced) `TranslationConfig` for
  pipeline behavior knobs (context window, queue depth, draft timing).
- `llm_connection` returns the `LLMConnection` value object for endpoint
  identity + default sampling parameters.

`llm_config` is kept as a legacy alias of `translation_behavioral`; the
`extra='ignore'` setting on `TranslationConfig` means old call sites passing
`base_url=`/`model=`/`temperature=` still construct without crashing — those
kwargs are silently dropped. Tests asserting on the old shape are updated
phase-by-phase.
"""
import os

import pytest

from livetranslate_common.models.llm import LLMConnection
from translation.config import TranslationConfig


@pytest.fixture
def translation_behavioral() -> TranslationConfig:
    """Pipeline behavior knobs only — no endpoint or sampling info."""
    return TranslationConfig()


@pytest.fixture
def llm_connection() -> LLMConnection:
    """Endpoint + default sampling — reads from env vars, defaults to local vLLM-MLX.

    The env-var read here mirrors the resolver's step-4 bootstrap path so
    unit tests can opt into a real backend by setting LLM_BASE_URL/LLM_MODEL.
    """
    return LLMConnection(
        engine="openai_compatible",
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8006/v1"),
        model=os.getenv("LLM_MODEL", "mlx-community/Qwen3-4B-4bit"),
        api_key=os.getenv("LLM_API_KEY", ""),
        temperature=0.7,
        timeout_s=15.0,
    )


@pytest.fixture
def llm_config(translation_behavioral: TranslationConfig) -> TranslationConfig:
    """Legacy alias — pre-consolidation tests requested `llm_config`.

    Returns the reduced TranslationConfig. Endpoint/sampling kwargs that old
    tests pass to `TranslationConfig(base_url=..., model=..., temperature=...)`
    are absorbed by extra='ignore' but do nothing.
    """
    return translation_behavioral
