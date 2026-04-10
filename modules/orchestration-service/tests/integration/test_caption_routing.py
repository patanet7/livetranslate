"""Tests for CaptionSourceAdapter protocol and source routing."""

import asyncio

import pytest

from services.pipeline.adapters.source_adapter import (
    BotAudioCaptionSource,
    CaptionEvent,
    CaptionSourceAdapter,
)
from services.meeting_session_config import MeetingSessionConfig


class TestCaptionEvent:
    def test_create_caption_event(self):
        event = CaptionEvent(
            event_type="added",
            caption_id="c1",
            text="Hello world",
            speaker_name="Alice",
            speaker_color="#4CAF50",
            source_lang="en",
            confidence=0.95,
            is_draft=False,
        )
        assert event.event_type == "added"
        assert event.text == "Hello world"
        assert event.translated_text is None
        assert event.target_lang is None

    def test_caption_event_defaults(self):
        event = CaptionEvent(
            event_type="added",
            caption_id="c2",
            text="Test",
        )
        assert event.speaker_color == "#4CAF50"
        assert event.source_lang == "auto"
        assert event.confidence == 1.0
        assert event.is_draft is False
        assert event.timestamp is not None


@pytest.mark.asyncio
class TestBotAudioCaptionSource:
    async def test_implements_protocol(self):
        source = BotAudioCaptionSource()
        assert isinstance(source, CaptionSourceAdapter)

    async def test_start_stop_lifecycle(self):
        source = BotAudioCaptionSource()
        config = MeetingSessionConfig(session_id="test-123")
        await source.start(config)
        assert source.is_running
        await source.stop()
        assert not source.is_running

    async def test_emits_events_to_callback(self):
        events = []
        source = BotAudioCaptionSource()
        source.on_caption = lambda e: events.append(e)

        config = MeetingSessionConfig(session_id="test-123")
        await source.start(config)

        await source.handle_transcription(
            text="Hello",
            speaker_name="Alice",
            source_lang="en",
            confidence=0.9,
            is_final=True,
        )

        assert len(events) == 1
        assert events[0].text == "Hello"
        assert events[0].event_type == "added"
        assert events[0].speaker_name == "Alice"
        assert events[0].is_draft is False

        await source.stop()

    async def test_draft_when_not_final(self):
        events = []
        source = BotAudioCaptionSource()
        source.on_caption = lambda e: events.append(e)

        config = MeetingSessionConfig(session_id="test-123")
        await source.start(config)

        await source.handle_transcription(
            text="Hel...",
            speaker_name="Alice",
            is_final=False,
        )

        assert events[0].is_draft is True

        await source.stop()

    async def test_speaker_color_assignment(self):
        events = []
        source = BotAudioCaptionSource()
        source.on_caption = lambda e: events.append(e)

        config = MeetingSessionConfig(session_id="test-123")
        await source.start(config)

        await source.handle_transcription(text="Hi", speaker_name="Alice")
        await source.handle_transcription(text="Hey", speaker_name="Bob")
        await source.handle_transcription(text="Hello", speaker_name="Alice")

        # Alice should get same color both times
        assert events[0].speaker_color == events[2].speaker_color
        # Bob should get different color
        assert events[1].speaker_color != events[0].speaker_color

        await source.stop()

    async def test_does_not_emit_when_stopped(self):
        events = []
        source = BotAudioCaptionSource()
        source.on_caption = lambda e: events.append(e)

        # Don't start — should not emit
        await source.handle_transcription(text="Hello", speaker_name="Alice")
        assert len(events) == 0

    async def test_does_not_emit_without_callback(self):
        source = BotAudioCaptionSource()
        config = MeetingSessionConfig(session_id="test-123")
        await source.start(config)

        # No callback set — should not error
        await source.handle_transcription(text="Hello", speaker_name="Alice")

        await source.stop()
