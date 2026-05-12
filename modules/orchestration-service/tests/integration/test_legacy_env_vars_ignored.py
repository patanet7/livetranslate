"""Legacy env vars no longer populate any production config.

After consolidation, only `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` are
honored as bootstrap fallback (with one-shot WARN). Everything else is
silently ignored by the new TranslationConfig (extra='ignore').
"""

from __future__ import annotations

import pytest

from translation.config import TranslationConfig


pytestmark = [pytest.mark.integration]


def test_legacy_intelligence_env_vars_do_not_appear_on_translation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INTELLIGENCE_DIRECT_LLM_* were removed entirely."""
    monkeypatch.setenv("INTELLIGENCE_DIRECT_LLM_URL", "http://nope:1")
    monkeypatch.setenv("INTELLIGENCE_DIRECT_LLM_MODEL", "ghost")
    cfg = TranslationConfig()
    # These never lived on TranslationConfig — verify cleanly that they don't
    # get magicked in.
    for attr in ("direct_llm_url", "direct_llm_model", "intelligence_direct_llm_url"):
        assert not hasattr(cfg, attr)


def test_legacy_ollama_env_vars_do_not_appear_on_translation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gone:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "ghost-model")
    cfg = TranslationConfig()
    for attr in ("ollama_base_url", "ollama_model", "base_url", "model"):
        assert not hasattr(cfg, attr)


def test_legacy_translation_service_url_does_not_populate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TRANSLATION_SERVICE_URL was used for proxy mode; not read by the new config."""
    monkeypatch.setenv("TRANSLATION_SERVICE_URL", "http://nope:5003")
    cfg = TranslationConfig()
    assert not hasattr(cfg, "translation_service_url")
    assert not hasattr(cfg, "base_url")


def test_legacy_llm_temperature_env_ignored_by_behavioral_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_TEMPERATURE used to live on TranslationConfig — now on LLMConnection."""
    monkeypatch.setenv("LLM_TEMPERATURE", "0.99")
    cfg = TranslationConfig()
    assert not hasattr(cfg, "temperature")
