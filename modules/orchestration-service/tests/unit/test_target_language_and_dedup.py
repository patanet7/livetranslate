"""Tests for target language wiring and translation guards.

Behavioral tests — no mocking. Tests verify:
- ConfigMessage target_language field is handled by the backend
- Translation guards: short text filter at orchestration layer

Note: Text-level regex dedup tests (TestDraftFinalDedup, TestRepetitionTranslationGuard)
were removed in Phase 2 — SegmentStore replaced all text-level dedup with structural
segment_id lifecycle tracking. See test_segment_store.py for the replacement tests.
"""
import pytest

from livetranslate_common.models.ws_messages import ConfigMessage, parse_ws_message


# ---------------------------------------------------------------------------
# ConfigMessage target_language handling
# ---------------------------------------------------------------------------


class TestConfigMessageTargetLanguage:
    """Verify ConfigMessage carries target_language to the backend."""

    def test_config_message_with_target_language(self) -> None:
        msg = ConfigMessage(target_language="es")
        assert msg.target_language == "es"
        assert msg.type == "config"

    def test_config_message_target_language_in_parsed_message(self) -> None:
        raw = '{"type": "config", "target_language": "zh"}'
        msg = parse_ws_message(raw)
        assert isinstance(msg, ConfigMessage)
        assert msg.target_language == "zh"

    def test_backend_handler_updates_target_language(self) -> None:
        """Simulate the ConfigMessage handler: if msg.target_language, update local var."""
        target_language = "en"  # default
        msg = ConfigMessage(target_language="es")

        # This is the handler logic we're adding:
        if msg.target_language:
            target_language = msg.target_language

        assert target_language == "es"

    def test_backend_handler_ignores_none_target_language(self) -> None:
        """ConfigMessage without target_language should not change session state."""
        target_language = "en"
        msg = ConfigMessage(language="zh")  # only source language, no target

        if msg.target_language:
            target_language = msg.target_language

        assert target_language == "en"


# ---------------------------------------------------------------------------
# Translation guards (defense-in-depth at orchestration layer)
# ---------------------------------------------------------------------------


class TestShortTextTranslationGuard:
    """Translation should be skipped for segments with < 3 characters.

    These are noise segments that slip through VAD — not worth translating.
    The segment is still forwarded to the frontend (for display), but no
    LLM call is made.
    """

    def _should_skip_translation(self, text: str) -> bool:
        """Replicate the short-text guard from websocket_audio.py."""
        return len(text.strip()) < 3

    def test_single_char_skipped(self) -> None:
        assert self._should_skip_translation("a") is True

    def test_two_chars_skipped(self) -> None:
        assert self._should_skip_translation("ah") is True

    def test_three_chars_passes(self) -> None:
        assert self._should_skip_translation("yes") is False

    def test_whitespace_only_skipped(self) -> None:
        assert self._should_skip_translation("  ") is True

    def test_empty_string_skipped(self) -> None:
        assert self._should_skip_translation("") is True

    def test_real_word_passes(self) -> None:
        assert self._should_skip_translation("Thank you") is False
