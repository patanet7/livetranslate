"""SourceOrchestrator — manages clean source switching.

Subscribes to MeetingSessionConfig changes. When caption_source changes,
cleanly stops the old source and starts the new one.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from livetranslate_common.logging import get_logger

from services.meeting_session_config import MeetingSessionConfig
from services.pipeline.adapters.fireflies_caption_source import FirefliesCaptionSource
from services.pipeline.adapters.source_adapter import (
    BotAudioCaptionSource,
    CaptionEvent,
    CaptionSourceAdapter,
)

logger = get_logger()

_SOURCE_FACTORIES: dict[str, type] = {
    "bot_audio": BotAudioCaptionSource,
    "fireflies": FirefliesCaptionSource,
}


class SourceOrchestrator:
    """Manages active caption source lifecycle and clean switching."""

    def __init__(
        self,
        config: MeetingSessionConfig,
        on_caption: Callable[[CaptionEvent], Any],
    ) -> None:
        self._config = config
        self._on_caption = on_caption
        self.active_source: CaptionSourceAdapter | None = None
        self._config.subscribe(self._on_config_changed)

    async def start(self) -> None:
        source_type = self._config.caption_source
        self.active_source = self._create_source(source_type)
        self.active_source.on_caption = self._on_caption
        await self.active_source.start(self._config)
        logger.info("source_orchestrator_started", source=source_type)

    async def stop(self) -> None:
        if self.active_source:
            await self.active_source.stop()
            self.active_source = None
        self._config.unsubscribe(self._on_config_changed)
        logger.info("source_orchestrator_stopped")

    async def switch_source(self, new_source_type: str) -> None:
        if self.active_source and self._current_source_type() == new_source_type:
            return

        if self.active_source:
            await self.active_source.stop()
            # Brief drain to let in-flight caption callbacks complete
            await asyncio.sleep(0.1)

        self.active_source = self._create_source(new_source_type)
        self.active_source.on_caption = self._on_caption
        await self.active_source.start(self._config)
        logger.info("source_switched", new_source=new_source_type)

    async def health_check(self) -> None:
        """Check if active source is healthy. Restart if crashed."""
        if self.active_source and not self.active_source.is_running:
            source_type = self._current_source_type()
            logger.warning("source_crashed_restarting", source=source_type)
            self.active_source = self._create_source(source_type)
            self.active_source.on_caption = self._on_caption
            await self.active_source.start(self._config)
            logger.info("source_restarted", source=source_type)

    def _create_source(self, source_type: str) -> Any:
        factory = _SOURCE_FACTORIES.get(source_type, BotAudioCaptionSource)
        return factory()

    def _current_source_type(self) -> str:
        if isinstance(self.active_source, FirefliesCaptionSource):
            return "fireflies"
        return "bot_audio"

    def _on_config_changed(self, changed_fields: set[str]) -> None:
        if "caption_source" in changed_fields:
            new_source = self._config.caption_source
            asyncio.ensure_future(self.switch_source(new_source))
