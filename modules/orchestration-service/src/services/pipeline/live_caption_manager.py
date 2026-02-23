"""
Live Caption Manager for Fireflies Real-Time Pipeline.

Manages caption display filtering based on PipelineConfig settings:
- display_mode: controls what gets sent to clients ("english", "translated", "both")
- enable_interim_captions: gates word-by-word updates during ASR refinement

This wraps the raw WebSocket broadcast callbacks with config-aware filtering,
so caption clients only receive what the current display_mode dictates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Awaitable

from livetranslate_common.logging import get_logger

if TYPE_CHECKING:
    from src.models.fireflies import FirefliesChunk
    from src.services.pipeline.config import PipelineConfig

logger = get_logger()

# Type alias for the broadcast function
BroadcastFn = Callable[[str, dict[str, Any]], Awaitable[None]]


class LiveCaptionManager:
    """Manages caption display lifecycle with config-driven filtering.

    Usage:
        manager = LiveCaptionManager(config=pipeline_config, broadcast=ws_broadcast)

        # Use as interim caption handler (replaces raw handle_live_update)
        client.on_live_update = manager.handle_interim_update

        # Use as caption event handler (wraps _broadcast_caption_to_ws)
        coordinator.on_caption_event(manager.handle_caption_event)
    """

    def __init__(
        self,
        config: PipelineConfig,
        broadcast: BroadcastFn,
        session_id: str = "",
    ):
        self._config = config
        self._broadcast = broadcast
        self._session_id = session_id or config.session_id
        self._interim_updates_sent: int = 0
        self._interim_updates_filtered: int = 0
        self._captions_sent: int = 0

    @property
    def display_mode(self) -> str:
        """Current display mode from config (reads live value)."""
        return self._config.display_mode

    @property
    def interim_enabled(self) -> bool:
        """Whether interim captions are enabled (reads live config value)."""
        return self._config.enable_interim_captions

    @property
    def stats(self) -> dict[str, int]:
        return {
            "interim_updates_sent": self._interim_updates_sent,
            "interim_updates_filtered": self._interim_updates_filtered,
            "captions_sent": self._captions_sent,
        }

    async def handle_interim_update(self, chunk: FirefliesChunk, is_final: bool) -> None:
        """Handle word-by-word interim caption updates from the Fireflies client.

        Respects config.enable_interim_captions:
        - When enabled: broadcasts all interim updates to WebSocket clients
        - When disabled: only broadcasts finalized chunks (is_final=True)

        Respects config.display_mode:
        - When "translated": skip interim updates entirely (no original text to show)
        - When "english" or "both": send interim updates
        """
        # If display mode is "translated", interim captions are useless
        # (they're raw ASR text in the source language, not translations)
        if self.display_mode == "translated" and not is_final:
            self._interim_updates_filtered += 1
            return

        # If interim captions disabled, only send finals
        if not self.interim_enabled and not is_final:
            self._interim_updates_filtered += 1
            return

        await self._broadcast(
            self._session_id,
            {
                "event": "interim_caption",
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "speaker_name": chunk.speaker_name,
                "speaker_color": None,
                "is_final": is_final,
            },
        )
        self._interim_updates_sent += 1

    async def handle_caption_event(self, event_type: str, caption: Any) -> None:
        """Handle caption lifecycle events from the pipeline coordinator.

        Applies display_mode filtering:
        - "english": sends original_text only, clears translated_text
        - "translated": sends translated_text only, clears original_text
        - "both": sends both (no filtering)
        """
        if event_type == "caption_expired":
            await self._broadcast(
                self._session_id,
                {"event": "caption_expired", "caption_id": caption.id},
            )
            return

        # Get caption dict and apply display_mode filtering
        caption_dict = caption.to_dict()

        if self.display_mode == "english":
            caption_dict["translated_text"] = ""
        elif self.display_mode == "translated":
            caption_dict["original_text"] = ""
        # "both" sends everything as-is

        await self._broadcast(
            self._session_id,
            {
                "event": event_type,
                "caption": caption_dict,
            },
        )
        self._captions_sent += 1
