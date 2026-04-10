"""Tests for CommandDispatcher — routes chat commands to MeetingSessionConfig."""

import pytest

from services.command_dispatcher import CommandDispatcher
from services.meeting_session_config import MeetingSessionConfig


class TestCommandDispatcher:
    def setup_method(self):
        self.config = MeetingSessionConfig(session_id="test-123")
        self.dispatcher = CommandDispatcher(self.config)

    def test_set_language_single(self):
        result = self.dispatcher.dispatch("/lang zh", sender="Alice")
        assert result.response_text == "✓ Translating: auto-detect → zh"
        assert self.config.target_lang == "zh"
        assert self.config.source_lang == "auto"

    def test_set_language_pair(self):
        result = self.dispatcher.dispatch("/lang zh-en", sender="Alice")
        assert result.response_text == "✓ Translating: zh → en"
        assert self.config.source_lang == "zh"
        assert self.config.target_lang == "en"

    def test_font_up(self):
        result = self.dispatcher.dispatch("/font up", sender="Alice")
        assert self.config.font_size == 28  # 24 + 4
        assert "28" in result.response_text

    def test_font_down(self):
        result = self.dispatcher.dispatch("/font down", sender="Alice")
        assert self.config.font_size == 20  # 24 - 4
        assert "20" in result.response_text

    def test_font_exact(self):
        result = self.dispatcher.dispatch("/font 32", sender="Alice")
        assert self.config.font_size == 32

    def test_mode_change(self):
        result = self.dispatcher.dispatch("/mode split", sender="Alice")
        assert self.config.display_mode == "split"
        assert "split" in result.response_text

    def test_theme_change(self):
        result = self.dispatcher.dispatch("/theme light", sender="Alice")
        assert self.config.theme == "light"

    def test_theme_contrast_alias(self):
        result = self.dispatcher.dispatch("/theme contrast", sender="Alice")
        assert self.config.theme == "high_contrast"

    def test_speakers_toggle(self):
        result = self.dispatcher.dispatch("/speakers off", sender="Alice")
        assert self.config.show_speakers is False

    def test_original_toggle(self):
        result = self.dispatcher.dispatch("/original on", sender="Alice")
        assert self.config.show_original is True

    def test_source_switch(self):
        result = self.dispatcher.dispatch("/source fireflies", sender="Alice")
        assert self.config.caption_source == "fireflies"

    def test_translate_toggle(self):
        result = self.dispatcher.dispatch("/translate off", sender="Alice")
        assert self.config.translation_enabled is False

    def test_status_query(self):
        result = self.dispatcher.dispatch("/status", sender="Alice")
        assert "subtitle" in result.response_text  # default mode
        assert "en" in result.response_text  # default target lang

    def test_help_query(self):
        result = self.dispatcher.dispatch("/help", sender="Alice")
        assert "/lang" in result.response_text
        assert "/font" in result.response_text

    def test_unknown_command(self):
        result = self.dispatcher.dispatch("/unknown blah", sender="Alice")
        assert "unknown" in result.response_text.lower() or "/help" in result.response_text

    def test_non_command_ignored(self):
        result = self.dispatcher.dispatch("hello everyone", sender="Alice")
        assert result is None

    def test_returns_changed_fields(self):
        result = self.dispatcher.dispatch("/lang zh", sender="Alice")
        assert "target_lang" in result.changed_fields

    def test_demo_without_manager(self):
        result = self.dispatcher.dispatch("/demo", sender="Alice")
        assert "not available" in result.response_text

    def test_demo_with_manager(self):
        dispatcher = CommandDispatcher(self.config, demo_manager=object())
        result = dispatcher.dispatch("/demo replay", sender="Alice")
        assert result.demo_action == "replay"
        assert "Starting" in result.response_text

    def test_demo_fireflies_alias(self):
        dispatcher = CommandDispatcher(self.config, demo_manager=object())
        result = dispatcher.dispatch("/demo", sender="Alice")
        assert result.demo_action == "replay"

    def test_demo_stop(self):
        dispatcher = CommandDispatcher(self.config, demo_manager=object())
        result = dispatcher.dispatch("/demo stop", sender="Alice")
        assert result.demo_action == "stop"
        assert "Stopping" in result.response_text
