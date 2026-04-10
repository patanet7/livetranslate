"""Tests for CaptionBuffer multi-subscriber extension."""

import pytest

from services.caption_buffer import CaptionBuffer


class TestCaptionBufferSubscribers:
    def test_subscribe_receives_add_events(self):
        events = []
        buffer = CaptionBuffer()
        buffer.subscribe(lambda event_type, caption: events.append((event_type, caption.translated_text)))

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")
        assert len(events) == 1
        assert events[0] == ("added", "Hello")

    def test_subscribe_receives_expire_events(self):
        events = []
        buffer = CaptionBuffer(default_duration=0.0)  # Expire immediately
        buffer.subscribe(lambda event_type, caption: events.append(event_type))

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")

        import time
        time.sleep(0.05)
        buffer.cleanup_expired()

        event_types = [e for e in events]
        assert "added" in event_types
        assert "expired" in event_types

    def test_multiple_subscribers(self):
        events_a, events_b = [], []
        buffer = CaptionBuffer()
        buffer.subscribe(lambda et, c: events_a.append(et))
        buffer.subscribe(lambda et, c: events_b.append(et))

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")
        assert len(events_a) == 1
        assert len(events_b) == 1

    def test_unsubscribe(self):
        events = []
        buffer = CaptionBuffer()
        callback = lambda et, c: events.append(et)
        buffer.subscribe(callback)
        buffer.unsubscribe(callback)

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")
        assert len(events) == 0

    def test_legacy_callback_still_works(self):
        """Backward compat: on_caption_added constructor param still fires."""
        added = []
        buffer = CaptionBuffer(on_caption_added=lambda c: added.append(c.translated_text))

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")
        assert added == ["Hello"]

    def test_legacy_and_subscriber_both_fire(self):
        """Both old callback AND new subscriber should fire."""
        legacy_events = []
        subscriber_events = []

        buffer = CaptionBuffer(on_caption_added=lambda c: legacy_events.append(c.translated_text))
        buffer.subscribe(lambda et, c: subscriber_events.append((et, c.translated_text)))

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")
        assert legacy_events == ["Hello"]
        assert len(subscriber_events) == 1
        assert subscriber_events[0] == ("added", "Hello")
