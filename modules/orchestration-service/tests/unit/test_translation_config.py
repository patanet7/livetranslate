"""TranslationConfig is reduced to behavioral pipeline knobs only.

Endpoint identity (base_url, api_key, model) and sampling parameters
(temperature, max_tokens, top_p, ...) now live on `LLMConnection`.
This config stays for things that govern how the translation *pipeline*
behaves: rolling context window size, token budgets, queue depth, draft
timeouts.
"""

from __future__ import annotations

import pytest

from translation.config import TranslationConfig


# ---------------------------------------------------------------------------
# Field surface — what TranslationConfig still owns
# ---------------------------------------------------------------------------


BEHAVIORAL_FIELDS = {
    "context_window_size",
    "max_context_tokens",
    "cross_direction_max_tokens",
    "max_queue_depth",
    "draft_max_tokens",
    "draft_timeout_s",
}


REMOVED_FIELDS = {
    # endpoint identity — now on LLMConnection
    "base_url",
    "api_key",
    "model",
    # sampling — now on LLMConnection / LLMParameterOverrides
    "temperature",
    "max_tokens",
    "timeout_s",
    "top_p",
    "top_k",
    "repetition_penalty",
    "presence_penalty",
}


class TestTranslationConfigFieldSurface:
    def test_only_behavioral_fields_remain(self) -> None:
        fields = set(TranslationConfig.model_fields.keys())
        assert fields == BEHAVIORAL_FIELDS, (
            f"Expected exactly behavioral fields; got extra={fields - BEHAVIORAL_FIELDS} "
            f"missing={BEHAVIORAL_FIELDS - fields}"
        )

    @pytest.mark.parametrize("removed", sorted(REMOVED_FIELDS))
    def test_removed_field_not_a_pydantic_field(self, removed: str) -> None:
        assert removed not in TranslationConfig.model_fields

    @pytest.mark.parametrize("removed", sorted(REMOVED_FIELDS))
    def test_removed_field_not_attribute_on_instance(self, removed: str) -> None:
        cfg = TranslationConfig()
        assert not hasattr(cfg, removed), (
            f"{removed!r} should be gone but instance still exposes it"
        )


# ---------------------------------------------------------------------------
# Construction & env-var reading
# ---------------------------------------------------------------------------


class TestTranslationConfigEnv:
    def test_default_construction(self) -> None:
        cfg = TranslationConfig()
        assert cfg.context_window_size == 5
        assert cfg.max_context_tokens == 800
        assert cfg.cross_direction_max_tokens == 200
        assert cfg.max_queue_depth == 10
        assert cfg.draft_max_tokens == 256
        assert cfg.draft_timeout_s == 4

    def test_env_var_reads_behavioral_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CONTEXT_WINDOW_SIZE", "7")
        monkeypatch.setenv("LLM_DRAFT_MAX_TOKENS", "128")
        cfg = TranslationConfig()
        assert cfg.context_window_size == 7
        assert cfg.draft_max_tokens == 128

    def test_legacy_env_var_does_not_populate_anything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM_BASE_URL / LLM_MODEL / LLM_API_KEY are read by the resolver,
        not by TranslationConfig. Setting them must not fabricate fields here."""
        monkeypatch.setenv("LLM_BASE_URL", "http://elsewhere:1234/v1")
        monkeypatch.setenv("LLM_MODEL", "ghost-model")
        monkeypatch.setenv("LLM_API_KEY", "secret")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
        cfg = TranslationConfig()
        for legacy in ("base_url", "model", "api_key", "temperature"):
            assert not hasattr(cfg, legacy)

    def test_extra_kwargs_silently_ignored(self) -> None:
        """`extra='ignore'` keeps the migration painless — old call sites
        passing base_url/model/temperature still construct, just without effect."""
        cfg = TranslationConfig(  # type: ignore[call-arg]
            base_url="http://old:1/v1",
            model="legacy",
            temperature=0.5,
            timeout_s=99,
            context_window_size=3,
        )
        assert cfg.context_window_size == 3
        assert not hasattr(cfg, "base_url")

    def test_from_env_classmethod_still_works(self) -> None:
        """Backward compat: existing callers use `TranslationConfig.from_env()`."""
        cfg = TranslationConfig.from_env()
        assert isinstance(cfg, TranslationConfig)
        assert cfg.context_window_size == 5
