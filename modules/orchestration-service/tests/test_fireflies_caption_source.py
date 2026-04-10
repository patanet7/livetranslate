"""Tests for FirefliesCaptionSource — wraps Fireflies as CaptionSourceAdapter."""

import pytest
from services.pipeline.adapters.fireflies_caption_source import FirefliesCaptionSource
from services.pipeline.adapters.source_adapter import CaptionEvent, CaptionSourceAdapter


class TestFirefliesCaptionSource:
    def test_implements_protocol(self):
        source = FirefliesCaptionSource()
        assert isinstance(source, CaptionSourceAdapter)

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        source = FirefliesCaptionSource()
        assert not source.is_running
        await source.start(config=None)
        assert source.is_running
        await source.stop()
        assert not source.is_running

    @pytest.mark.asyncio
    async def test_handle_chunk_emits_caption_event(self):
        source = FirefliesCaptionSource()
        events: list[CaptionEvent] = []
        source.on_caption = lambda e: events.append(e)
        await source.start(config=None)

        await source.handle_chunk({
            "type": "transcript",
            "transcript_id": "t1",
            "chunk_id": "c1",
            "text": "Hello world",
            "speaker_name": "Alice",
            "start_time": 0.0,
            "end_time": 1.25,
        })

        assert len(events) == 1
        assert events[0].text == "Hello world"
        assert events[0].speaker_name == "Alice"
        assert events[0].event_type == "added"

    @pytest.mark.asyncio
    async def test_speaker_colors_assigned(self):
        source = FirefliesCaptionSource()
        events: list[CaptionEvent] = []
        source.on_caption = lambda e: events.append(e)
        await source.start(config=None)

        await source.handle_chunk({"text": "Hi", "speaker_name": "Alice", "chunk_id": "c1", "start_time": 0, "end_time": 1})
        await source.handle_chunk({"text": "Hey", "speaker_name": "Bob", "chunk_id": "c2", "start_time": 1, "end_time": 2})
        await source.handle_chunk({"text": "Again", "speaker_name": "Alice", "chunk_id": "c3", "start_time": 2, "end_time": 3})

        assert events[0].speaker_color == events[2].speaker_color
        assert events[0].speaker_color != events[1].speaker_color

    @pytest.mark.asyncio
    async def test_ignores_events_when_stopped(self):
        source = FirefliesCaptionSource()
        events: list[CaptionEvent] = []
        source.on_caption = lambda e: events.append(e)

        await source.handle_chunk({"text": "Hi", "speaker_name": "Alice", "chunk_id": "c1", "start_time": 0, "end_time": 1})
        assert len(events) == 0
