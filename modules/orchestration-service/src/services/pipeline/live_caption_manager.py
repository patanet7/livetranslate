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

        # Grow filter state: chunk_id -> last text broadcast
        self._displayed_text: dict[str, str] = {}
        self._interim_shrinks_suppressed: int = 0
        self._interim_duplicates_suppressed: int = 0

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
            "interim_shrinks_suppressed": self._interim_shrinks_suppressed,
            "interim_duplicates_suppressed": self._interim_duplicates_suppressed,
            "captions_sent": self._captions_sent,
            "displayed_text_entries": len(self._displayed_text),
        }

    async def handle_interim_update(self, chunk, is_final: bool) -> None:
        """Handle interim caption with grow-only filter.

        Only broadcasts when:
        - is_final=True (always)
        - Text is new (first time for this chunk_id)
        - Text grew (starts with previous text)
        - Text was corrected but is longer

        Suppresses:
        - Pure duplicates (same text)
        - Shrinks (text got shorter without being a grow/correction)
        """
        # Display mode gate
        if self.display_mode == "translated" and not is_final:
            self._interim_updates_filtered += 1
            return

        # Interim enabled gate
        if not self.interim_enabled and not is_final:
            self._interim_updates_filtered += 1
            return

        chunk_id = chunk.chunk_id
        new_text = chunk.text

        # Final always broadcasts and cleans up
        if is_final:
            self._displayed_text.pop(chunk_id, None)
            await self._broadcast(
                self._session_id,
                {
                    "event": "interim_caption",
                    "chunk_id": chunk_id,
                    "text": new_text,
                    "speaker_name": chunk.speaker_name,
                    "speaker_color": None,
                    "is_final": True,
                    "type": "final",
                },
            )
            self._interim_updates_sent += 1
            return

        last_text = self._displayed_text.get(chunk_id, "")

        # Pure duplicate — skip
        if new_text == last_text:
            self._interim_duplicates_suppressed += 1
            return

        # Determine update type
        if not last_text:
            update_type = "grow"  # First text for this chunk
        elif new_text.startswith(last_text):
            update_type = "grow"  # Pure append
        elif len(new_text) > len(last_text):
            update_type = "correction"  # Rewritten but longer
        else:
            # Shrink — suppress
            self._interim_shrinks_suppressed += 1
            return

        self._displayed_text[chunk_id] = new_text
        await self._broadcast(
            self._session_id,
            {
                "event": "interim_caption",
                "chunk_id": chunk_id,
                "text": new_text,
                "speaker_name": chunk.speaker_name,
                "speaker_color": None,
                "is_final": False,
                "type": update_type,
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

    def cleanup_stale_displayed_text(self, max_age_seconds: float = 30.0) -> int:
        """Remove displayed text entries older than max_age. Call periodically."""
        # In practice, entries are cleaned on finalization. This is a safety net.
        count = len(self._displayed_text)
        self._displayed_text.clear()
        return count
