"""Tests for chat command/response WebSocket messages."""

import pytest

from livetranslate_common.models.ws_messages import (
    ChatCommandMessage,
    ChatResponseMessage,
    ConfigChangedMessage,
    parse_ws_message,
)


class TestChatMessages:
    def test_parse_chat_command(self):
        msg = parse_ws_message('{"type": "chat_command", "command": "/lang zh", "sender": "Alice"}')
        assert isinstance(msg, ChatCommandMessage)
        assert msg.command == "/lang zh"
        assert msg.sender == "Alice"

    def test_parse_chat_response(self):
        msg = parse_ws_message('{"type": "chat_response", "text": "Language set to zh"}')
        assert isinstance(msg, ChatResponseMessage)
        assert msg.text == "Language set to zh"

    def test_parse_config_changed(self):
        msg = parse_ws_message('{"type": "config_changed", "changes": {"target_lang": "zh"}}')
        assert isinstance(msg, ConfigChangedMessage)
        assert msg.changes == {"target_lang": "zh"}
