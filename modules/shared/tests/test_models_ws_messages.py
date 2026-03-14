"""Behavioral tests for WebSocket message schemas and parse_ws_message."""

from __future__ import annotations

import json

import pytest

from livetranslate_common.models.ws_messages import (
    PROTOCOL_VERSION,
    BackendSwitchedMessage,
    ConfigMessage,
    ConnectedMessage,
    EndMessage,
    EndMeetingMessage,
    EndSessionMessage,
    InterimMessage,
    LanguageDetectedMessage,
    MeetingStartedMessage,
    PromoteToMeetingMessage,
    RecordingStatusMessage,
    SegmentMessage,
    ServiceStatusMessage,
    StartSessionMessage,
    TranslationMessage,
    parse_ws_message,
)


class TestClientMessages:
    def test_start_session(self) -> None:
        msg = StartSessionMessage(sample_rate=16000, channels=1)
        assert msg.type == "start_session"
        assert msg.sample_rate == 16000
        assert msg.channels == 1
        assert msg.device_id is None

        msg_with_device = StartSessionMessage(sample_rate=48000, channels=2, device_id="usb-mic-0")
        assert msg_with_device.device_id == "usb-mic-0"

    def test_end_session(self) -> None:
        msg = EndSessionMessage()
        assert msg.type == "end_session"

    def test_promote_to_meeting(self) -> None:
        msg = PromoteToMeetingMessage()
        assert msg.type == "promote_to_meeting"

    def test_end_meeting(self) -> None:
        msg = EndMeetingMessage()
        assert msg.type == "end_meeting"


class TestTranscriptionServiceMessages:
    def test_config_message(self) -> None:
        msg = ConfigMessage(model="whisper-large-v3", language="en")
        assert msg.type == "config"
        assert msg.model == "whisper-large-v3"
        assert msg.language == "en"
        assert msg.initial_prompt is None
        assert msg.glossary_terms is None

    def test_config_message_auto_detect(self) -> None:
        """All fields None means auto-detect / keep current."""
        msg = ConfigMessage()
        assert msg.model is None
        assert msg.language is None

    def test_end_message(self) -> None:
        msg = EndMessage()
        assert msg.type == "end"

    def test_language_detected(self) -> None:
        msg = LanguageDetectedMessage(language="fr", confidence=0.98)
        assert msg.type == "language_detected"
        assert msg.language == "fr"
        assert msg.confidence == pytest.approx(0.98)

    def test_backend_switched(self) -> None:
        msg = BackendSwitchedMessage(backend="faster-whisper", model="whisper-base", language="zh")
        assert msg.type == "backend_switched"
        assert msg.backend == "faster-whisper"
        assert msg.model == "whisper-base"
        assert msg.language == "zh"


class TestServerMessages:
    def test_connected(self) -> None:
        msg = ConnectedMessage(session_id="sess-abc-123")
        assert msg.type == "connected"
        assert msg.session_id == "sess-abc-123"
        assert msg.protocol_version == PROTOCOL_VERSION

    def test_segment(self) -> None:
        msg = SegmentMessage(
            text="Hello world",
            language="en",
            confidence=0.95,
            stable_text="Hello world",
            unstable_text="",
            is_final=True,
            speaker_id="SPEAKER_00",
        )
        assert msg.type == "segment"
        assert msg.is_final is True
        assert msg.speaker_id == "SPEAKER_00"

    def test_interim(self) -> None:
        msg = InterimMessage(text="Hel...", confidence=0.6)
        assert msg.type == "interim"
        assert msg.text == "Hel..."

    def test_translation(self) -> None:
        msg = TranslationMessage(
            text="Bonjour",
            source_lang="en",
            target_lang="fr",
            transcript_id=7,
            context_used=3,
        )
        assert msg.type == "translation"
        assert msg.context_used == 3

    def test_meeting_started(self) -> None:
        msg = MeetingStartedMessage(session_id="meet-xyz", started_at="2026-03-14T10:00:00Z")
        assert msg.type == "meeting_started"
        assert msg.session_id == "meet-xyz"

    def test_recording_status(self) -> None:
        msg = RecordingStatusMessage(recording=True, chunks_written=42)
        assert msg.type == "recording_status"
        assert msg.recording is True
        assert msg.chunks_written == 42

    def test_service_status(self) -> None:
        msg = ServiceStatusMessage(transcription="up", translation="down")
        assert msg.type == "service_status"
        assert msg.transcription == "up"
        assert msg.translation == "down"


class TestParseMessage:
    def test_parse_start_session(self) -> None:
        raw = json.dumps({"type": "start_session", "sample_rate": 16000, "channels": 1})
        result = parse_ws_message(raw)
        assert isinstance(result, StartSessionMessage)
        assert result.sample_rate == 16000

    def test_parse_unknown_type_returns_none(self) -> None:
        raw = json.dumps({"type": "totally_unknown_message", "data": 123})
        result = parse_ws_message(raw)
        assert result is None

    def test_parse_malformed_json_returns_none(self) -> None:
        result = parse_ws_message("not valid json {{{{")
        assert result is None

    def test_parse_connected(self) -> None:
        raw = json.dumps(
            {"type": "connected", "session_id": "s-001", "protocol_version": PROTOCOL_VERSION}
        )
        result = parse_ws_message(raw)
        assert isinstance(result, ConnectedMessage)
        assert result.session_id == "s-001"
