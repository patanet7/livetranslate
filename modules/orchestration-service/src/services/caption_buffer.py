"""
Caption Buffer Service

Manages a display queue of translated captions with:
- Time-based expiration (configurable duration per caption)
- Speaker color assignment (consistent colors per speaker)
- Maximum caption limit (FIFO overflow handling)
- Priority support for important messages

The CaptionBuffer bridges RollingWindowTranslator output to UI display,
ensuring captions appear and disappear at the right times with proper
visual styling.

Reference: FIREFLIES_ADAPTATION_PLAN.md Section "Caption Output"
"""

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default speaker colors (material design palette)
DEFAULT_SPEAKER_COLORS = [
    "#4CAF50",  # Green
    "#2196F3",  # Blue
    "#FF9800",  # Orange
    "#9C27B0",  # Purple
    "#F44336",  # Red
    "#00BCD4",  # Cyan
    "#E91E63",  # Pink
    "#FFEB3B",  # Yellow
    "#795548",  # Brown
    "#607D8B",  # Blue Grey
]

# Default configuration
DEFAULT_CAPTION_DURATION_SECONDS = 4.0  # Bubble visible for 4 sec total
DEFAULT_MAX_CAPTIONS = 5
DEFAULT_MIN_DISPLAY_TIME_SECONDS = 1.0
DEFAULT_MAX_CAPTION_CHARS = 250  # ~3-4 lines at 32 chars/line (CEA-608 standard)
DEFAULT_MAX_AGGREGATION_TIME = 3.0  # Force new bubble after 3 sec


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Caption:
    """A single caption entry for display."""

    id: str
    original_text: str | None
    translated_text: str
    speaker_name: str
    speaker_color: str
    target_language: str
    confidence: float
    created_at: datetime
    expires_at: datetime
    priority: int = 0  # Higher = more important, stays longer

    @property
    def is_expired(self) -> bool:
        """Check if caption has expired."""
        return datetime.now(UTC) >= self.expires_at

    @property
    def time_remaining_seconds(self) -> float:
        """Get seconds until expiration."""
        remaining = (self.expires_at - datetime.now(UTC)).total_seconds()
        return max(0.0, remaining)

    @property
    def display_duration_seconds(self) -> float:
        """Get total display duration."""
        return (self.expires_at - self.created_at).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "original_text": self.original_text,
            "translated_text": str(self.translated_text),
            "speaker_name": str(self.speaker_name),
            "speaker_color": str(self.speaker_color),
            "target_language": str(self.target_language),
            "confidence": float(self.confidence),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "duration_seconds": float(self.display_duration_seconds),
            "time_remaining_seconds": float(self.time_remaining_seconds),
            "priority": int(self.priority),
        }


@dataclass
class CaptionBufferStats:
    """Statistics for a caption buffer."""

    total_captions_added: int = 0
    total_captions_expired: int = 0
    total_captions_overflow: int = 0
    speakers_seen: int = 0
    current_caption_count: int = 0
    average_display_time_seconds: float = 0.0
    last_caption_time: datetime | None = None


# =============================================================================
# Caption Buffer
# =============================================================================


class CaptionBuffer:
    """
    Manages caption display queue with timing and styling.

    The buffer maintains an ordered collection of captions, automatically
    expiring old ones and enforcing a maximum count. Each speaker gets
    a consistent color assignment.

    Usage:
        buffer = CaptionBuffer(
            max_captions=5,
            default_duration=8.0,
        )

        # Add caption from translation result
        caption = buffer.add_caption(
            translated_text="Hola, mundo!",
            speaker_name="Alice",
            original_text="Hello, world!",
            target_language="es",
        )

        # Get current captions for display
        active = buffer.get_active_captions()

        # Cleanup expired
        buffer.cleanup_expired()
    """

    def __init__(
        self,
        max_captions: int = DEFAULT_MAX_CAPTIONS,
        default_duration: float = DEFAULT_CAPTION_DURATION_SECONDS,
        min_display_time: float = DEFAULT_MIN_DISPLAY_TIME_SECONDS,
        max_caption_chars: int = DEFAULT_MAX_CAPTION_CHARS,
        max_aggregation_time: float = DEFAULT_MAX_AGGREGATION_TIME,
        speaker_colors: list[str] | None = None,
        show_original: bool = True,
        on_caption_added: Callable[[Caption], None] | None = None,
        on_caption_expired: Callable[[Caption], None] | None = None,
        on_caption_updated: Callable[[Caption], None] | None = None,
        aggregate_speaker_text: bool = True,
        speaker_aggregation_window: float = 5.0,
    ):
        """
        Initialize the caption buffer.

        Args:
            max_captions: Maximum number of captions to display at once
            default_duration: Default caption display duration in seconds
            min_display_time: Minimum time a caption is shown
            max_caption_chars: Maximum characters per caption (oldest text truncated when exceeded)
            max_aggregation_time: Max seconds a caption can aggregate before forcing new bubble
            speaker_colors: Custom color palette for speakers
            show_original: Whether to include original text in captions
            on_caption_added: Callback when caption is added
            on_caption_expired: Callback when caption expires
            on_caption_updated: Callback when caption is updated (text appended)
            aggregate_speaker_text: If True, append text from same speaker to existing caption
            speaker_aggregation_window: Time window (seconds) for aggregating same-speaker text
        """
        self.max_captions = max_captions
        self.default_duration = default_duration
        self.min_display_time = min_display_time
        self.max_caption_chars = max_caption_chars
        self.max_aggregation_time = max_aggregation_time
        self.speaker_colors = speaker_colors or DEFAULT_SPEAKER_COLORS
        self.show_original = show_original
        self.on_caption_added = on_caption_added
        self.on_caption_expired = on_caption_expired
        self.on_caption_updated = on_caption_updated
        self.aggregate_speaker_text = aggregate_speaker_text
        self.speaker_aggregation_window = speaker_aggregation_window

        # Caption storage (ordered by creation time)
        self._captions: OrderedDict[str, Caption] = OrderedDict()

        # Speaker -> current caption ID mapping (for aggregation)
        self._speaker_current_caption: dict[str, str] = {}

        # Track last speaker for out-of-order detection
        self._last_speaker: str | None = None

        # Speaker -> color mapping
        self._speaker_color_map: dict[str, str] = {}
        self._next_color_index: int = 0

        # Statistics
        self._stats = CaptionBufferStats()

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"CaptionBuffer initialized: max_captions={max_captions}, "
            f"duration={default_duration}s, aggregate={aggregate_speaker_text}"
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def add_caption(
        self,
        translated_text: str,
        speaker_name: str,
        original_text: str | None = None,
        target_language: str = "es",
        confidence: float = 1.0,
        duration: float | None = None,
        priority: int = 0,
        caption_id: str | None = None,
    ) -> tuple:
        """
        Add a caption to the display queue.

        If speaker aggregation is enabled and the same speaker has an active
        caption within the aggregation window, the new text is appended to
        the existing caption instead of creating a new one.

        Args:
            translated_text: Translated text to display
            speaker_name: Speaker name
            original_text: Optional original text
            target_language: Target language code
            confidence: Translation confidence (0.0-1.0)
            duration: Custom duration (uses default if None)
            priority: Priority level (higher = stays longer)
            caption_id: Custom caption ID (auto-generated if None)

        Returns:
            Tuple of (Caption, was_updated: bool)
            - was_updated=True if text was appended to existing caption
            - was_updated=False if new caption was created
        """
        with self._lock:
            now = datetime.now(UTC)
            actual_duration = duration if duration is not None else self.default_duration
            actual_duration += priority * 2.0

            # Check for speaker aggregation
            if self.aggregate_speaker_text:
                existing_caption = self._get_active_speaker_caption(speaker_name)
                if existing_caption:
                    # Append text to existing caption
                    new_translated = existing_caption.translated_text + " " + translated_text

                    # Truncate from beginning if exceeds max chars (roll-up style)
                    if len(new_translated) > self.max_caption_chars:
                        # Find a word boundary to truncate at
                        excess = len(new_translated) - self.max_caption_chars
                        # Add buffer to ensure we're under limit
                        truncate_at = excess + 20
                        # Find next space after truncate point
                        space_idx = new_translated.find(" ", truncate_at)
                        if space_idx > 0:
                            new_translated = "..." + new_translated[space_idx + 1 :]
                        else:
                            new_translated = "..." + new_translated[truncate_at:]

                    existing_caption.translated_text = new_translated

                    # Same for original text
                    if original_text and existing_caption.original_text:
                        new_original = existing_caption.original_text + " " + original_text
                        if len(new_original) > self.max_caption_chars:
                            excess = len(new_original) - self.max_caption_chars
                            truncate_at = excess + 20
                            space_idx = new_original.find(" ", truncate_at)
                            if space_idx > 0:
                                new_original = "..." + new_original[space_idx + 1 :]
                            else:
                                new_original = "..." + new_original[truncate_at:]
                        existing_caption.original_text = new_original
                    elif original_text:
                        existing_caption.original_text = original_text

                    # Extend expiration, but cap at max_aggregation_time + buffer from creation
                    # This keeps bubble visible while speaker talks, but forces new bubble eventually
                    max_expires_at = existing_caption.created_at + timedelta(
                        seconds=self.max_aggregation_time + 4.0  # 4 sec buffer for fade
                    )
                    new_expires_at = now + timedelta(seconds=actual_duration)
                    existing_caption.expires_at = min(new_expires_at, max_expires_at)

                    # Update confidence (weighted average)
                    existing_caption.confidence = (existing_caption.confidence + confidence) / 2

                    logger.debug(
                        f"Caption updated (aggregated): id={existing_caption.id}, "
                        f"speaker={speaker_name}, chars={len(existing_caption.translated_text)}"
                    )

                    # Track last speaker
                    self._last_speaker = speaker_name

                    # Callback for update
                    if self.on_caption_updated:
                        try:
                            self.on_caption_updated(existing_caption)
                        except Exception as e:
                            logger.warning(f"Caption updated callback error: {e}")

                    return (existing_caption, True)

            # Get speaker color
            speaker_color = self._get_speaker_color(speaker_name)

            # Create new caption
            caption = Caption(
                id=caption_id or str(uuid4()),
                original_text=original_text if self.show_original else None,
                translated_text=translated_text,
                speaker_name=speaker_name,
                speaker_color=speaker_color,
                target_language=target_language,
                confidence=confidence,
                created_at=now,
                expires_at=now + timedelta(seconds=actual_duration),
                priority=priority,
            )

            # Handle overflow (remove oldest, non-priority captions)
            while len(self._captions) >= self.max_captions:
                self._remove_oldest_caption()

            # Add caption
            self._captions[caption.id] = caption

            # Track speaker's current caption for aggregation
            self._speaker_current_caption[speaker_name] = caption.id

            # Track last speaker for out-of-order detection
            self._last_speaker = speaker_name

            # Update stats
            self._stats.total_captions_added += 1
            self._stats.current_caption_count = len(self._captions)
            self._stats.last_caption_time = now

            # Callback
            if self.on_caption_added:
                try:
                    self.on_caption_added(caption)
                except Exception as e:
                    logger.warning(f"Caption added callback error: {e}")

            logger.debug(
                f"Caption added: id={caption.id}, speaker={speaker_name}, "
                f"duration={actual_duration}s"
            )

            return (caption, False)

    def _get_active_speaker_caption(self, speaker_name: str) -> Caption | None:
        """
        Get the active caption for a speaker if within aggregation window.

        Returns None if:
        - No active caption exists
        - Caption is expired
        - Caption has been alive longer than max_aggregation_time (force new bubble)
        - A different speaker has spoken since (out-of-order detection)
        """
        caption_id = self._speaker_current_caption.get(speaker_name)
        if not caption_id:
            return None

        caption = self._captions.get(caption_id)
        if not caption:
            # Caption was removed
            del self._speaker_current_caption[speaker_name]
            return None

        if caption.is_expired:
            # Caption has expired
            del self._speaker_current_caption[speaker_name]
            return None

        # Check if caption has been alive too long - force new bubble
        age = (datetime.now(UTC) - caption.created_at).total_seconds()
        if age > self.max_aggregation_time:
            logger.debug(
                f"Caption {caption_id} exceeded max_aggregation_time ({age:.1f}s > {self.max_aggregation_time}s), "
                f"forcing new bubble for speaker {speaker_name}"
            )
            return None

        # Out-of-order detection: if someone else spoke since this speaker, force new bubble
        if self._last_speaker and self._last_speaker != speaker_name:
            logger.debug(
                f"Out-of-order: {speaker_name} spoke after {self._last_speaker}, "
                f"forcing new bubble"
            )
            return None

        return caption

    def get_active_captions(self) -> list[Caption]:
        """
        Get all active (non-expired) captions.

        Returns:
            List of active Caption objects, ordered by creation time
        """
        with self._lock:
            # Cleanup expired first
            self._cleanup_expired_internal()

            return list(self._captions.values())

    def get_caption(self, caption_id: str) -> Caption | None:
        """
        Get a specific caption by ID.

        Args:
            caption_id: Caption ID

        Returns:
            Caption object or None if not found/expired
        """
        with self._lock:
            caption = self._captions.get(caption_id)
            if caption and not caption.is_expired:
                return caption
            return None

    def remove_caption(self, caption_id: str) -> bool:
        """
        Remove a caption by ID.

        Args:
            caption_id: Caption ID to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if caption_id in self._captions:
                del self._captions[caption_id]
                self._stats.current_caption_count = len(self._captions)
                return True
            return False

    def clear(self) -> int:
        """
        Clear all captions.

        Returns:
            Number of captions cleared
        """
        with self._lock:
            count = len(self._captions)
            self._captions.clear()
            self._stats.current_caption_count = 0
            logger.info(f"Cleared {count} captions")
            return count

    def cleanup_expired(self) -> int:
        """
        Remove expired captions.

        Returns:
            Number of captions removed
        """
        with self._lock:
            return self._cleanup_expired_internal()

    def extend_caption(self, caption_id: str, additional_seconds: float) -> bool:
        """
        Extend a caption's display time.

        Args:
            caption_id: Caption ID
            additional_seconds: Seconds to add to expiration

        Returns:
            True if extended, False if not found
        """
        with self._lock:
            caption = self._captions.get(caption_id)
            if caption:
                caption.expires_at += timedelta(seconds=additional_seconds)
                logger.debug(f"Extended caption {caption_id} by {additional_seconds}s")
                return True
            return False

    def get_stats(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            self._stats.current_caption_count = len(self._captions)
            self._stats.speakers_seen = len(self._speaker_color_map)

            return {
                "total_captions_added": self._stats.total_captions_added,
                "total_captions_expired": self._stats.total_captions_expired,
                "total_captions_overflow": self._stats.total_captions_overflow,
                "speakers_seen": self._stats.speakers_seen,
                "current_caption_count": self._stats.current_caption_count,
                "max_captions": self.max_captions,
                "default_duration_seconds": self.default_duration,
                "last_caption_time": (
                    self._stats.last_caption_time.isoformat()
                    if self._stats.last_caption_time
                    else None
                ),
                "speaker_colors": dict(self._speaker_color_map),
            }

    def get_speaker_color(self, speaker_name: str) -> str:
        """
        Get the assigned color for a speaker.

        Args:
            speaker_name: Speaker name

        Returns:
            Hex color string
        """
        with self._lock:
            return self._get_speaker_color(speaker_name)

    def set_speaker_color(self, speaker_name: str, color: str) -> None:
        """
        Manually set a speaker's color.

        Args:
            speaker_name: Speaker name
            color: Hex color string (e.g., "#FF0000")
        """
        with self._lock:
            self._speaker_color_map[speaker_name] = color
            logger.debug(f"Set speaker color: {speaker_name} -> {color}")

    def reset_speaker_colors(self) -> None:
        """Reset speaker color assignments."""
        with self._lock:
            self._speaker_color_map.clear()
            self._next_color_index = 0
            logger.info("Reset speaker color assignments")

    # =========================================================================
    # Iteration and Display
    # =========================================================================

    def get_display_dict(self) -> dict:
        """
        Get display-ready representation for UI.

        Returns:
            Dictionary with captions and metadata
        """
        with self._lock:
            captions = self.get_active_captions()
            return {
                "captions": [c.to_dict() for c in captions],
                "count": len(captions),
                "max_captions": self.max_captions,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def __len__(self) -> int:
        """Get number of active captions."""
        with self._lock:
            return len(self._captions)

    def __iter__(self):
        """Iterate over active captions."""
        return iter(self.get_active_captions())

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_speaker_color(self, speaker_name: str) -> str:
        """Get or assign color for speaker."""
        if speaker_name not in self._speaker_color_map:
            # Assign next color in palette
            color_index = self._next_color_index % len(self.speaker_colors)
            self._speaker_color_map[speaker_name] = self.speaker_colors[color_index]
            self._next_color_index += 1
            logger.debug(
                f"Assigned color {self._speaker_color_map[speaker_name]} "
                f"to speaker {speaker_name}"
            )
        return self._speaker_color_map[speaker_name]

    def _remove_oldest_caption(self) -> Caption | None:
        """Remove the oldest caption (respecting priority)."""
        if not self._captions:
            return None

        # Find lowest priority caption (prefer expired ones)
        candidates = list(self._captions.values())

        # Sort by: expired first, then priority (low first), then age (old first)
        candidates.sort(
            key=lambda c: (
                not c.is_expired,  # Expired first
                c.priority,  # Low priority first
                c.created_at,  # Oldest first
            )
        )

        victim = candidates[0]

        # Don't remove if it hasn't met minimum display time
        display_time = (datetime.now(UTC) - victim.created_at).total_seconds()
        if display_time < self.min_display_time and not victim.is_expired:
            # Skip this one, try next
            if len(candidates) > 1:
                victim = candidates[1]
            else:
                # Force remove oldest anyway
                pass

        del self._captions[victim.id]
        self._stats.total_captions_overflow += 1

        logger.debug(f"Removed caption due to overflow: id={victim.id}")

        return victim

    def _cleanup_expired_internal(self) -> int:
        """Internal cleanup without lock (caller must hold lock)."""
        expired_ids = [cid for cid, caption in self._captions.items() if caption.is_expired]

        for cid in expired_ids:
            caption = self._captions.pop(cid)
            self._stats.total_captions_expired += 1

            if self.on_caption_expired:
                try:
                    self.on_caption_expired(caption)
                except Exception as e:
                    logger.warning(f"Caption expired callback error: {e}")

        if expired_ids:
            self._stats.current_caption_count = len(self._captions)
            logger.debug(f"Cleaned up {len(expired_ids)} expired captions")

        return len(expired_ids)


# =============================================================================
# Session Caption Manager
# =============================================================================


class SessionCaptionManager:
    """
    Manages caption buffers for multiple sessions.

    Each session gets its own CaptionBuffer with independent state.
    Useful for managing captions across multiple concurrent translation sessions.
    """

    def __init__(
        self,
        default_max_captions: int = DEFAULT_MAX_CAPTIONS,
        default_duration: float = DEFAULT_CAPTION_DURATION_SECONDS,
    ):
        """
        Initialize the session manager.

        Args:
            default_max_captions: Default max captions per session
            default_duration: Default caption duration
        """
        self.default_max_captions = default_max_captions
        self.default_duration = default_duration
        self._sessions: dict[str, CaptionBuffer] = {}
        self._lock = threading.RLock()

    def get_buffer(self, session_id: str) -> CaptionBuffer:
        """
        Get or create a caption buffer for a session.

        Args:
            session_id: Session ID

        Returns:
            CaptionBuffer for the session
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = CaptionBuffer(
                    max_captions=self.default_max_captions,
                    default_duration=self.default_duration,
                )
                logger.info(f"Created caption buffer for session: {session_id}")
            return self._sessions[session_id]

    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session's caption buffer.

        Args:
            session_id: Session ID

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()
                del self._sessions[session_id]
                logger.info(f"Removed caption buffer for session: {session_id}")
                return True
            return False

    def get_all_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def cleanup_all(self) -> dict[str, int]:
        """
        Cleanup expired captions in all sessions.

        Returns:
            Dict mapping session_id -> expired_count
        """
        with self._lock:
            results = {}
            for session_id, buffer in self._sessions.items():
                results[session_id] = buffer.cleanup_expired()
            return results


# =============================================================================
# Factory Function
# =============================================================================


def create_caption_buffer(
    config: dict | None = None,
) -> CaptionBuffer:
    """
    Factory function to create a CaptionBuffer with configuration.

    Args:
        config: Optional configuration dict with keys:
            - max_captions: int (default: 5)
            - default_duration: float (default: 8.0)
            - min_display_time: float (default: 2.0)
            - max_caption_chars: int (default: 250, ~3-4 lines)
            - max_aggregation_time: float (default: 8.0, max seconds before new bubble)
            - speaker_colors: List[str] (default: material palette)
            - show_original: bool (default: True)

    Returns:
        Configured CaptionBuffer instance
    """
    config = config or {}

    return CaptionBuffer(
        max_captions=config.get("max_captions", DEFAULT_MAX_CAPTIONS),
        default_duration=config.get("default_duration", DEFAULT_CAPTION_DURATION_SECONDS),
        min_display_time=config.get("min_display_time", DEFAULT_MIN_DISPLAY_TIME_SECONDS),
        max_caption_chars=config.get("max_caption_chars", DEFAULT_MAX_CAPTION_CHARS),
        max_aggregation_time=config.get("max_aggregation_time", DEFAULT_MAX_AGGREGATION_TIME),
        speaker_colors=config.get("speaker_colors"),
        show_original=config.get("show_original", True),
    )
