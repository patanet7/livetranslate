"""Unit tests for WebSocket message models.

Focuses on ConfigMessage target_language field and serialization.
"""

from __future__ import annotations

import json

from livetranslate_common.models.ws_messages import ConfigMessage, parse_ws_message


class TestConfigMessageTargetLanguage:
    """ConfigMessage must accept and round-trip a target_language field."""

    def test_config_message_accepts_target_language(self) -> None:
        msg = ConfigMessage(target_language="es")
        assert msg.target_language == "es"

    def test_config_message_target_language_defaults_none(self) -> None:
        msg = ConfigMessage()
        assert msg.target_language is None

    def test_config_message_round_trips_target_language_json(self) -> None:
        msg = ConfigMessage(target_language="zh", language="en")
        serialized = msg.model_dump_json()
        data = json.loads(serialized)
        assert data["target_language"] == "zh"
        assert data["language"] == "en"
        assert data["type"] == "config"

    def test_parse_ws_message_config_with_target_language(self) -> None:
        raw = json.dumps({"type": "config", "target_language": "ja"})
        msg = parse_ws_message(raw)
        assert isinstance(msg, ConfigMessage)
        assert msg.target_language == "ja"

    def test_parse_ws_message_config_without_target_language(self) -> None:
        raw = json.dumps({"type": "config", "language": "en"})
        msg = parse_ws_message(raw)
        assert isinstance(msg, ConfigMessage)
        assert msg.target_language is None


class TestConfigMessageInterpreterLanguages:
    """ConfigMessage must accept and round-trip an interpreter_languages field."""

    def test_config_message_accepts_interpreter_languages(self) -> None:
        msg = ConfigMessage(interpreter_languages=["zh", "en"])
        assert msg.interpreter_languages == ["zh", "en"]

    def test_config_message_interpreter_languages_defaults_none(self) -> None:
        msg = ConfigMessage()
        assert msg.interpreter_languages is None

    def test_config_message_round_trips_interpreter_languages_json(self) -> None:
        msg = ConfigMessage(interpreter_languages=["zh", "en"])
        serialized = msg.model_dump_json()
        data = json.loads(serialized)
        assert data["interpreter_languages"] == ["zh", "en"]
        assert data["type"] == "config"

    def test_parse_ws_message_config_with_interpreter_languages(self) -> None:
        raw = json.dumps({"type": "config", "interpreter_languages": ["ja", "en"]})
        msg = parse_ws_message(raw)
        assert isinstance(msg, ConfigMessage)
        assert msg.interpreter_languages == ["ja", "en"]

    def test_parse_ws_message_config_without_interpreter_languages(self) -> None:
        raw = json.dumps({"type": "config", "language": "en"})
        msg = parse_ws_message(raw)
        assert isinstance(msg, ConfigMessage)
        assert msg.interpreter_languages is None

    def test_config_message_both_target_and_interpreter(self) -> None:
        """Both fields can coexist (though semantically only one is active)."""
        msg = ConfigMessage(target_language="es", interpreter_languages=["zh", "en"])
        assert msg.target_language == "es"
        assert msg.interpreter_languages == ["zh", "en"]
