"""Tests for SourceOrchestrator — manages clean source switching."""

import asyncio
import pytest
from services.source_orchestrator import SourceOrchestrator
from services.pipeline.adapters.source_adapter import BotAudioCaptionSource, CaptionEvent
from services.pipeline.adapters.fireflies_caption_source import FirefliesCaptionSource
from services.meeting_session_config import MeetingSessionConfig


class TestSourceOrchestrator:
    def setup_method(self):
        self.config = MeetingSessionConfig(session_id="test-123")
        self.events: list[CaptionEvent] = []
        self.orchestrator = SourceOrchestrator(
            config=self.config,
            on_caption=lambda e: self.events.append(e),
        )

    @pytest.mark.asyncio
    async def test_start_with_bot_audio(self):
        await self.orchestrator.start()
        assert self.orchestrator.active_source is not None
        assert self.orchestrator.active_source.is_running

    @pytest.mark.asyncio
    async def test_start_with_fireflies(self):
        self.config.update(caption_source="fireflies")
        await self.orchestrator.start()
        assert isinstance(self.orchestrator.active_source, FirefliesCaptionSource)
        assert self.orchestrator.active_source.is_running

    @pytest.mark.asyncio
    async def test_switch_source_stops_old_starts_new(self):
        await self.orchestrator.start()
        old_source = self.orchestrator.active_source
        assert isinstance(old_source, BotAudioCaptionSource)

        await self.orchestrator.switch_source("fireflies")
        assert not old_source.is_running
        assert isinstance(self.orchestrator.active_source, FirefliesCaptionSource)
        assert self.orchestrator.active_source.is_running

    @pytest.mark.asyncio
    async def test_switch_back_to_bot_audio(self):
        self.config.update(caption_source="fireflies")
        await self.orchestrator.start()
        await self.orchestrator.switch_source("bot_audio")
        assert isinstance(self.orchestrator.active_source, BotAudioCaptionSource)

    @pytest.mark.asyncio
    async def test_switch_to_same_source_is_noop(self):
        await self.orchestrator.start()
        source_before = self.orchestrator.active_source
        await self.orchestrator.switch_source("bot_audio")
        assert self.orchestrator.active_source is source_before

    @pytest.mark.asyncio
    async def test_events_routed_after_switch(self):
        await self.orchestrator.start()
        await self.orchestrator.switch_source("fireflies")

        ff_source = self.orchestrator.active_source
        assert isinstance(ff_source, FirefliesCaptionSource)
        await ff_source.handle_chunk({"text": "test", "speaker_name": "A", "chunk_id": "c1", "start_time": 0, "end_time": 1})

        assert len(self.events) == 1
        assert self.events[0].text == "test"

    @pytest.mark.asyncio
    async def test_stop_stops_active_source(self):
        await self.orchestrator.start()
        source = self.orchestrator.active_source
        await self.orchestrator.stop()
        assert not source.is_running
        assert self.orchestrator.active_source is None

    @pytest.mark.asyncio
    async def test_config_subscriber_triggers_switch(self):
        await self.orchestrator.start()
        assert isinstance(self.orchestrator.active_source, BotAudioCaptionSource)

        self.config.update(caption_source="fireflies")
        await asyncio.sleep(0.05)

        assert isinstance(self.orchestrator.active_source, FirefliesCaptionSource)


class TestSourceOrchestratorResilience:
    def setup_method(self):
        self.config = MeetingSessionConfig(session_id="test-resilience")
        self.events: list[CaptionEvent] = []
        self.orchestrator = SourceOrchestrator(
            config=self.config,
            on_caption=lambda e: self.events.append(e),
        )

    @pytest.mark.asyncio
    async def test_restart_on_source_error(self):
        """If active source crashes, orchestrator restarts it."""
        await self.orchestrator.start()
        source = self.orchestrator.active_source
        assert source.is_running

        # Simulate source crash
        await source.stop()
        assert not source.is_running

        # Orchestrator detects and restarts
        await self.orchestrator.health_check()
        assert self.orchestrator.active_source.is_running

    @pytest.mark.asyncio
    async def test_health_check_noop_when_healthy(self):
        """Health check doesn't restart a healthy source."""
        await self.orchestrator.start()
        source_before = self.orchestrator.active_source
        await self.orchestrator.health_check()
        assert self.orchestrator.active_source is source_before
