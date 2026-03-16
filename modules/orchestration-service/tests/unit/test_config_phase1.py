"""Tests for Phase 1 config additions."""
from translation.config import TranslationConfig


class TestTranslationConfigPhase1:
    def test_max_context_tokens_default(self):
        config = TranslationConfig()
        assert config.max_context_tokens == 800

    def test_cross_direction_max_tokens_default(self):
        config = TranslationConfig()
        assert config.cross_direction_max_tokens == 200

    def test_cross_direction_max_tokens_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_CROSS_DIRECTION_MAX_TOKENS", "300")
        config = TranslationConfig()
        assert config.cross_direction_max_tokens == 300
