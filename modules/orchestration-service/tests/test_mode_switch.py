"""Tests for interpreter ↔ split/transcript mode switching.

Verifies that source_language is saved before entering interpreter mode
and restored on exit, and that lock_language config is sent correctly.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestModeSwitchLanguageRestore:
    """Test that interpreter ↔ split transitions preserve source_language."""

    def test_entering_interpreter_saves_source_language(self):
        """When entering interpreter mode, the previous source_language
        should be saved so it can be restored on exit."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()

        # Set up: user has source_language = "en" in split mode
        state.source_language = "en"
        state.interpreter_languages = None

        # Enter interpreter mode
        state.enter_interpreter(("zh", "en"))

        assert state.interpreter_languages == ("zh", "en")
        assert state.source_language is None  # auto-detect forced
        assert state._pre_interpreter_source_language == "en"

    def test_leaving_interpreter_restores_source_language(self):
        """When leaving interpreter mode, source_language should be
        restored to the value it had before entering interpreter."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()

        # Set up: user had source_language = "en"
        state.source_language = "en"
        state.enter_interpreter(("zh", "en"))

        # Leave interpreter mode
        state.leave_interpreter()

        assert state.interpreter_languages is None
        assert state.source_language == "en"

    def test_split_interpreter_split_roundtrip(self):
        """Full cycle: split(en→zh) → interpreter(zh/en) → split restores en→zh."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()

        # Start in split mode with en→zh
        state.source_language = "en"
        state.target_language = "zh"

        # Enter interpreter
        state.enter_interpreter(("zh", "en"))
        assert state.source_language is None
        assert state.interpreter_languages == ("zh", "en")

        # Exit interpreter
        state.leave_interpreter()
        assert state.source_language == "en"
        assert state.target_language == "zh"  # target preserved
        assert state.interpreter_languages is None

    def test_entering_interpreter_clears_text_buffers(self):
        """Mode transition should clear stable text buffers."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()

        state._stable_text_buffer = "some accumulated text"
        state._last_translated_stable = "some translated text"
        state._last_translation_direction = "en→zh"

        state.enter_interpreter(("zh", "en"))

        assert state._stable_text_buffer == ""
        assert state._last_translated_stable == ""
        assert state._last_translation_direction is None

    def test_leaving_interpreter_clears_text_buffers(self):
        """Leaving interpreter should also clear text buffers."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()
        state.source_language = "en"
        state.enter_interpreter(("zh", "en"))

        state._stable_text_buffer = "interpreter text"
        state._last_translated_stable = "translated text"
        state._last_translation_direction = "zh→en"

        state.leave_interpreter()

        assert state._stable_text_buffer == ""
        assert state._last_translated_stable == ""
        assert state._last_translation_direction is None

    def test_interpreter_direction_flip(self):
        """In interpreter mode, detected language determines translation direction."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()
        state.enter_interpreter(("zh", "en"))

        # Segment detected as zh → translate to en
        target = state.get_effective_target("zh")
        assert target == "en"

        # Segment detected as en → translate to zh
        target = state.get_effective_target("en")
        assert target == "zh"

        # Segment detected as neither → no translation
        target = state.get_effective_target("ja")
        assert target is None

    def test_split_mode_effective_target(self):
        """In split mode, target_language is always the effective target."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()
        state.source_language = "en"
        state.target_language = "zh"

        target = state.get_effective_target("en")
        assert target == "zh"

        # Even if detected language differs, target stays the same
        target = state.get_effective_target("ja")
        assert target == "zh"

    def test_no_pre_interpreter_language_when_auto_detect(self):
        """If source was already auto-detect (None) before interpreter,
        leaving interpreter should restore None."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()
        state.source_language = None  # auto-detect
        state.enter_interpreter(("zh", "en"))

        state.leave_interpreter()
        assert state.source_language is None


class TestLockLanguageConfig:
    """Test that explicit source language sends lock_language to transcription."""

    def test_explicit_source_sets_lock(self):
        """Setting an explicit source language should set lock_language=True."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()
        state.set_source_language("en")

        assert state.source_language == "en"
        assert state.lock_language is True

    def test_auto_detect_clears_lock(self):
        """Setting source to None (auto-detect) clears lock_language."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()
        state.set_source_language("en")
        state.set_source_language(None)

        assert state.source_language is None
        assert state.lock_language is False

    def test_interpreter_mode_clears_lock(self):
        """Entering interpreter mode clears lock_language."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()
        state.set_source_language("en")
        assert state.lock_language is True

        state.enter_interpreter(("zh", "en"))
        assert state.lock_language is False

    def test_leaving_interpreter_restores_lock(self):
        """Leaving interpreter restores lock_language if source was explicit."""
        from routers.audio.websocket_audio import _make_config_handler

        state = _make_config_handler()
        state.set_source_language("en")
        state.enter_interpreter(("zh", "en"))
        state.leave_interpreter()

        assert state.lock_language is True
        assert state.source_language == "en"
