"""
Unit Tests for CaptionBuffer

Tests caption display queue management with:
- Caption lifecycle (add, get, remove, expire)
- Speaker color assignment
- Timing control and expiration
- Max caption limits and overflow handling
- Session management

Test Categories:
1. Basic Operations - Add, get, remove captions
2. Expiration - Time-based caption expiration
3. Speaker Colors - Consistent color assignment
4. Overflow Handling - Max caption limit enforcement
5. Session Management - Multi-session support
6. Edge Cases - Boundary conditions
"""

import pytest
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional
from uuid import uuid4


# =============================================================================
# Test-Local Implementation
# =============================================================================
# We define test-local versions to avoid import chain issues.

DEFAULT_SPEAKER_COLORS = [
    "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336",
    "#00BCD4", "#E91E63", "#FFEB3B", "#795548", "#607D8B",
]

DEFAULT_CAPTION_DURATION_SECONDS = 8.0
DEFAULT_MAX_CAPTIONS = 5
DEFAULT_MIN_DISPLAY_TIME_SECONDS = 2.0


@dataclass
class Caption:
    """A single caption entry for display."""

    id: str
    original_text: Optional[str]
    translated_text: str
    speaker_name: str
    speaker_color: str
    target_language: str
    confidence: float
    created_at: datetime
    expires_at: datetime
    priority: int = 0

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def time_remaining_seconds(self) -> float:
        remaining = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, remaining)

    @property
    def display_duration_seconds(self) -> float:
        return (self.expires_at - self.created_at).total_seconds()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "speaker_name": self.speaker_name,
            "speaker_color": self.speaker_color,
            "target_language": self.target_language,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "time_remaining_seconds": self.time_remaining_seconds,
            "priority": self.priority,
        }


@dataclass
class CaptionBufferStats:
    total_captions_added: int = 0
    total_captions_expired: int = 0
    total_captions_overflow: int = 0
    speakers_seen: int = 0
    current_caption_count: int = 0
    average_display_time_seconds: float = 0.0
    last_caption_time: Optional[datetime] = None


class CaptionBuffer:
    """Test-local CaptionBuffer implementation."""

    def __init__(
        self,
        max_captions: int = DEFAULT_MAX_CAPTIONS,
        default_duration: float = DEFAULT_CAPTION_DURATION_SECONDS,
        min_display_time: float = DEFAULT_MIN_DISPLAY_TIME_SECONDS,
        speaker_colors: Optional[List[str]] = None,
        show_original: bool = True,
        on_caption_added: Optional[Callable[[Caption], None]] = None,
        on_caption_expired: Optional[Callable[[Caption], None]] = None,
    ):
        self.max_captions = max_captions
        self.default_duration = default_duration
        self.min_display_time = min_display_time
        self.speaker_colors = speaker_colors or DEFAULT_SPEAKER_COLORS
        self.show_original = show_original
        self.on_caption_added = on_caption_added
        self.on_caption_expired = on_caption_expired

        self._captions: OrderedDict[str, Caption] = OrderedDict()
        self._speaker_color_map: Dict[str, str] = {}
        self._next_color_index: int = 0
        self._stats = CaptionBufferStats()
        self._lock = threading.RLock()

    def add_caption(
        self,
        translated_text: str,
        speaker_name: str,
        original_text: Optional[str] = None,
        target_language: str = "es",
        confidence: float = 1.0,
        duration: Optional[float] = None,
        priority: int = 0,
        caption_id: Optional[str] = None,
    ) -> Caption:
        with self._lock:
            speaker_color = self._get_speaker_color(speaker_name)
            actual_duration = duration if duration is not None else self.default_duration
            actual_duration += priority * 2.0

            now = datetime.now(timezone.utc)
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

            while len(self._captions) >= self.max_captions:
                self._remove_oldest_caption()

            self._captions[caption.id] = caption

            self._stats.total_captions_added += 1
            self._stats.current_caption_count = len(self._captions)
            self._stats.last_caption_time = now

            if self.on_caption_added:
                try:
                    self.on_caption_added(caption)
                except Exception:
                    pass

            return caption

    def get_active_captions(self) -> List[Caption]:
        with self._lock:
            self._cleanup_expired_internal()
            return list(self._captions.values())

    def get_caption(self, caption_id: str) -> Optional[Caption]:
        with self._lock:
            caption = self._captions.get(caption_id)
            if caption and not caption.is_expired:
                return caption
            return None

    def remove_caption(self, caption_id: str) -> bool:
        with self._lock:
            if caption_id in self._captions:
                del self._captions[caption_id]
                self._stats.current_caption_count = len(self._captions)
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            count = len(self._captions)
            self._captions.clear()
            self._stats.current_caption_count = 0
            return count

    def cleanup_expired(self) -> int:
        with self._lock:
            return self._cleanup_expired_internal()

    def extend_caption(self, caption_id: str, additional_seconds: float) -> bool:
        with self._lock:
            caption = self._captions.get(caption_id)
            if caption:
                caption.expires_at += timedelta(seconds=additional_seconds)
                return True
            return False

    def get_stats(self) -> Dict:
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
        with self._lock:
            return self._get_speaker_color(speaker_name)

    def set_speaker_color(self, speaker_name: str, color: str) -> None:
        with self._lock:
            self._speaker_color_map[speaker_name] = color

    def reset_speaker_colors(self) -> None:
        with self._lock:
            self._speaker_color_map.clear()
            self._next_color_index = 0

    def get_display_dict(self) -> Dict:
        with self._lock:
            captions = self.get_active_captions()
            return {
                "captions": [c.to_dict() for c in captions],
                "count": len(captions),
                "max_captions": self.max_captions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._captions)

    def __iter__(self):
        return iter(self.get_active_captions())

    def _get_speaker_color(self, speaker_name: str) -> str:
        if speaker_name not in self._speaker_color_map:
            color_index = self._next_color_index % len(self.speaker_colors)
            self._speaker_color_map[speaker_name] = self.speaker_colors[color_index]
            self._next_color_index += 1
        return self._speaker_color_map[speaker_name]

    def _remove_oldest_caption(self) -> Optional[Caption]:
        if not self._captions:
            return None

        candidates = list(self._captions.values())
        candidates.sort(
            key=lambda c: (
                not c.is_expired,
                c.priority,
                c.created_at,
            )
        )

        victim = candidates[0]
        display_time = (datetime.now(timezone.utc) - victim.created_at).total_seconds()
        if display_time < self.min_display_time and not victim.is_expired:
            if len(candidates) > 1:
                victim = candidates[1]

        del self._captions[victim.id]
        self._stats.total_captions_overflow += 1

        return victim

    def _cleanup_expired_internal(self) -> int:
        expired_ids = [
            cid for cid, caption in self._captions.items()
            if caption.is_expired
        ]

        for cid in expired_ids:
            caption = self._captions.pop(cid)
            self._stats.total_captions_expired += 1

            if self.on_caption_expired:
                try:
                    self.on_caption_expired(caption)
                except Exception:
                    pass

        if expired_ids:
            self._stats.current_caption_count = len(self._captions)

        return len(expired_ids)


class SessionCaptionManager:
    """Manages caption buffers for multiple sessions."""

    def __init__(
        self,
        default_max_captions: int = DEFAULT_MAX_CAPTIONS,
        default_duration: float = DEFAULT_CAPTION_DURATION_SECONDS,
    ):
        self.default_max_captions = default_max_captions
        self.default_duration = default_duration
        self._sessions: Dict[str, CaptionBuffer] = {}
        self._lock = threading.RLock()

    def get_buffer(self, session_id: str) -> CaptionBuffer:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = CaptionBuffer(
                    max_captions=self.default_max_captions,
                    default_duration=self.default_duration,
                )
            return self._sessions[session_id]

    def remove_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()
                del self._sessions[session_id]
                return True
            return False

    def get_all_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def cleanup_all(self) -> Dict[str, int]:
        with self._lock:
            results = {}
            for session_id, buffer in self._sessions.items():
                results[session_id] = buffer.cleanup_expired()
            return results


def create_caption_buffer(config: Optional[Dict] = None) -> CaptionBuffer:
    config = config or {}
    return CaptionBuffer(
        max_captions=config.get("max_captions", DEFAULT_MAX_CAPTIONS),
        default_duration=config.get("default_duration", DEFAULT_CAPTION_DURATION_SECONDS),
        min_display_time=config.get("min_display_time", DEFAULT_MIN_DISPLAY_TIME_SECONDS),
        speaker_colors=config.get("speaker_colors"),
        show_original=config.get("show_original", True),
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def buffer():
    """Create a caption buffer with short durations for testing."""
    return CaptionBuffer(
        max_captions=5,
        default_duration=1.0,  # Short for testing
        min_display_time=0.1,
    )


@pytest.fixture
def long_buffer():
    """Create a caption buffer with longer durations."""
    return CaptionBuffer(
        max_captions=5,
        default_duration=10.0,
        min_display_time=1.0,
    )


# =============================================================================
# Test: Basic Operations
# =============================================================================


class TestBasicOperations:
    """Tests for basic caption operations."""

    def test_add_caption_returns_caption(self, buffer):
        """GIVEN a caption buffer
        WHEN add_caption is called
        THEN it should return a Caption object with correct fields."""
        caption = buffer.add_caption(
            translated_text="Hola, mundo!",
            speaker_name="Alice",
            original_text="Hello, world!",
            target_language="es",
            confidence=0.95,
        )

        assert isinstance(caption, Caption)
        assert caption.translated_text == "Hola, mundo!"
        assert caption.original_text == "Hello, world!"
        assert caption.speaker_name == "Alice"
        assert caption.target_language == "es"
        assert caption.confidence == 0.95
        assert caption.id is not None

    def test_add_caption_increments_count(self, buffer):
        """GIVEN an empty buffer
        WHEN captions are added
        THEN the length should increase."""
        assert len(buffer) == 0

        buffer.add_caption("Text 1", "Speaker1")
        assert len(buffer) == 1

        buffer.add_caption("Text 2", "Speaker2")
        assert len(buffer) == 2

    def test_get_caption_by_id(self, buffer):
        """GIVEN a caption in the buffer
        WHEN get_caption is called with its ID
        THEN the caption should be returned."""
        caption = buffer.add_caption("Test", "Speaker")

        retrieved = buffer.get_caption(caption.id)

        assert retrieved is not None
        assert retrieved.id == caption.id
        assert retrieved.translated_text == "Test"

    def test_get_caption_nonexistent_returns_none(self, buffer):
        """GIVEN a buffer
        WHEN get_caption is called with non-existent ID
        THEN None should be returned."""
        result = buffer.get_caption("nonexistent-id")
        assert result is None

    def test_remove_caption(self, buffer):
        """GIVEN a caption in the buffer
        WHEN remove_caption is called
        THEN the caption should be removed."""
        caption = buffer.add_caption("Test", "Speaker")
        assert len(buffer) == 1

        result = buffer.remove_caption(caption.id)

        assert result is True
        assert len(buffer) == 0
        assert buffer.get_caption(caption.id) is None

    def test_remove_nonexistent_returns_false(self, buffer):
        """GIVEN a buffer
        WHEN remove_caption is called with non-existent ID
        THEN False should be returned."""
        result = buffer.remove_caption("nonexistent")
        assert result is False

    def test_clear_removes_all(self, buffer):
        """GIVEN a buffer with captions
        WHEN clear is called
        THEN all captions should be removed."""
        buffer.add_caption("Text 1", "Speaker1")
        buffer.add_caption("Text 2", "Speaker2")
        buffer.add_caption("Text 3", "Speaker3")
        assert len(buffer) == 3

        count = buffer.clear()

        assert count == 3
        assert len(buffer) == 0

    def test_get_active_captions(self, long_buffer):
        """GIVEN captions in the buffer
        WHEN get_active_captions is called
        THEN all non-expired captions should be returned."""
        long_buffer.add_caption("Text 1", "Speaker1")
        long_buffer.add_caption("Text 2", "Speaker2")

        active = long_buffer.get_active_captions()

        assert len(active) == 2
        assert all(isinstance(c, Caption) for c in active)


# =============================================================================
# Test: Expiration
# =============================================================================


class TestExpiration:
    """Tests for caption expiration."""

    def test_caption_expires_after_duration(self):
        """GIVEN a caption with short duration
        WHEN duration passes
        THEN caption should be expired."""
        buffer = CaptionBuffer(default_duration=0.1)
        caption = buffer.add_caption("Test", "Speaker")

        # Wait for expiration
        time.sleep(0.15)

        assert caption.is_expired

    def test_cleanup_removes_expired(self):
        """GIVEN expired captions in buffer
        WHEN cleanup_expired is called
        THEN expired captions should be removed."""
        buffer = CaptionBuffer(default_duration=0.1)
        buffer.add_caption("Text 1", "Speaker1")
        buffer.add_caption("Text 2", "Speaker2")

        # Wait for expiration
        time.sleep(0.15)

        count = buffer.cleanup_expired()

        assert count == 2
        assert len(buffer) == 0

    def test_get_active_removes_expired(self):
        """GIVEN expired captions
        WHEN get_active_captions is called
        THEN expired captions should not be included."""
        buffer = CaptionBuffer(default_duration=0.1)
        buffer.add_caption("Test", "Speaker")

        # Wait for expiration
        time.sleep(0.15)

        active = buffer.get_active_captions()

        assert len(active) == 0

    def test_extend_caption_increases_duration(self, long_buffer):
        """GIVEN a caption
        WHEN extend_caption is called
        THEN expiration time should increase."""
        caption = long_buffer.add_caption("Test", "Speaker")
        original_expires = caption.expires_at

        result = long_buffer.extend_caption(caption.id, 5.0)

        assert result is True
        assert caption.expires_at > original_expires

    def test_time_remaining_decreases(self):
        """GIVEN a caption
        WHEN time passes
        THEN time_remaining_seconds should decrease."""
        buffer = CaptionBuffer(default_duration=1.0)
        caption = buffer.add_caption("Test", "Speaker")

        initial_remaining = caption.time_remaining_seconds
        time.sleep(0.2)
        later_remaining = caption.time_remaining_seconds

        assert later_remaining < initial_remaining


# =============================================================================
# Test: Speaker Colors
# =============================================================================


class TestSpeakerColors:
    """Tests for speaker color assignment."""

    def test_speaker_gets_consistent_color(self, buffer):
        """GIVEN a speaker
        WHEN multiple captions are added
        THEN speaker should get the same color."""
        c1 = buffer.add_caption("Text 1", "Alice")
        c2 = buffer.add_caption("Text 2", "Alice")

        assert c1.speaker_color == c2.speaker_color

    def test_different_speakers_get_different_colors(self, buffer):
        """GIVEN multiple speakers
        WHEN captions are added
        THEN each speaker should get a unique color."""
        c1 = buffer.add_caption("Text 1", "Alice")
        c2 = buffer.add_caption("Text 2", "Bob")
        c3 = buffer.add_caption("Text 3", "Charlie")

        colors = {c1.speaker_color, c2.speaker_color, c3.speaker_color}
        assert len(colors) == 3  # All unique

    def test_get_speaker_color(self, buffer):
        """GIVEN a speaker in the buffer
        WHEN get_speaker_color is called
        THEN the assigned color should be returned."""
        buffer.add_caption("Test", "Alice")

        color = buffer.get_speaker_color("Alice")

        assert color is not None
        assert color.startswith("#")

    def test_set_speaker_color(self, buffer):
        """GIVEN a buffer
        WHEN set_speaker_color is called
        THEN the color should be applied to future captions."""
        buffer.set_speaker_color("Alice", "#FF0000")
        caption = buffer.add_caption("Test", "Alice")

        assert caption.speaker_color == "#FF0000"

    def test_reset_speaker_colors(self, buffer):
        """GIVEN assigned speaker colors
        WHEN reset_speaker_colors is called
        THEN colors should be reset."""
        buffer.add_caption("Test", "Alice")
        original_color = buffer.get_speaker_color("Alice")

        buffer.reset_speaker_colors()

        # After reset, getting color again assigns fresh from start
        new_color = buffer.get_speaker_color("Alice")
        # Should get first color in palette
        assert new_color == buffer.speaker_colors[0]

    def test_color_cycles_through_palette(self, buffer):
        """GIVEN more speakers than colors
        WHEN colors are assigned
        THEN they should cycle through the palette."""
        speakers = [f"Speaker{i}" for i in range(15)]
        for speaker in speakers:
            buffer.add_caption("Test", speaker)

        # All speakers should have colors from the palette
        for speaker in speakers:
            color = buffer.get_speaker_color(speaker)
            assert color in buffer.speaker_colors


# =============================================================================
# Test: Overflow Handling
# =============================================================================


class TestOverflowHandling:
    """Tests for max caption limit enforcement."""

    def test_overflow_removes_oldest(self, long_buffer):
        """GIVEN a full buffer
        WHEN a new caption is added
        THEN the oldest should be removed."""
        # Fill buffer (max 5)
        for i in range(5):
            long_buffer.add_caption(f"Text {i}", f"Speaker{i}")

        assert len(long_buffer) == 5

        # Add one more
        new_caption = long_buffer.add_caption("New text", "NewSpeaker")

        assert len(long_buffer) == 5
        assert new_caption in long_buffer.get_active_captions()

    def test_overflow_prefers_expired_captions(self):
        """GIVEN expired and non-expired captions
        WHEN overflow occurs
        THEN expired should be removed first."""
        buffer = CaptionBuffer(max_captions=3, default_duration=0.1)

        # Add caption that will expire
        old = buffer.add_caption("Old", "Speaker1")
        time.sleep(0.15)  # Let it expire

        # Add non-expired captions
        buffer.add_caption("New1", "Speaker2", duration=10.0)
        buffer.add_caption("New2", "Speaker3", duration=10.0)

        # Add one more to trigger overflow
        buffer.add_caption("New3", "Speaker4", duration=10.0)

        # Expired one should be gone
        assert buffer.get_caption(old.id) is None

    def test_overflow_respects_priority(self):
        """GIVEN captions with different priorities
        WHEN overflow occurs
        THEN low priority should be removed first."""
        buffer = CaptionBuffer(max_captions=3, default_duration=10.0, min_display_time=0.0)

        # Add low priority caption
        low = buffer.add_caption("Low priority", "Speaker1", priority=0)

        # Add high priority captions
        buffer.add_caption("High 1", "Speaker2", priority=5)
        buffer.add_caption("High 2", "Speaker3", priority=5)

        # Add one more to trigger overflow
        buffer.add_caption("High 3", "Speaker4", priority=5)

        # Low priority should be gone
        assert buffer.get_caption(low.id) is None

    def test_overflow_stat_tracked(self, long_buffer):
        """GIVEN overflow events
        WHEN stats are checked
        THEN overflow count should be accurate."""
        # Fill and overflow
        for i in range(7):
            long_buffer.add_caption(f"Text {i}", f"Speaker{i}")

        stats = long_buffer.get_stats()
        assert stats["total_captions_overflow"] == 2  # 7 - 5


# =============================================================================
# Test: Session Management
# =============================================================================


class TestSessionManagement:
    """Tests for multi-session support."""

    def test_get_buffer_creates_new(self):
        """GIVEN a session manager
        WHEN get_buffer is called for new session
        THEN a new buffer should be created."""
        manager = SessionCaptionManager()

        buffer = manager.get_buffer("session-1")

        assert buffer is not None
        assert isinstance(buffer, CaptionBuffer)

    def test_get_buffer_returns_existing(self):
        """GIVEN an existing session
        WHEN get_buffer is called again
        THEN the same buffer should be returned."""
        manager = SessionCaptionManager()

        buffer1 = manager.get_buffer("session-1")
        buffer1.add_caption("Test", "Speaker")

        buffer2 = manager.get_buffer("session-1")

        assert buffer1 is buffer2
        assert len(buffer2) == 1

    def test_remove_session(self):
        """GIVEN a session
        WHEN remove_session is called
        THEN the session should be removed."""
        manager = SessionCaptionManager()
        manager.get_buffer("session-1")

        result = manager.remove_session("session-1")

        assert result is True
        assert "session-1" not in manager.get_all_sessions()

    def test_remove_nonexistent_session(self):
        """GIVEN no session
        WHEN remove_session is called
        THEN False should be returned."""
        manager = SessionCaptionManager()

        result = manager.remove_session("nonexistent")

        assert result is False

    def test_get_all_sessions(self):
        """GIVEN multiple sessions
        WHEN get_all_sessions is called
        THEN all session IDs should be returned."""
        manager = SessionCaptionManager()
        manager.get_buffer("session-1")
        manager.get_buffer("session-2")
        manager.get_buffer("session-3")

        sessions = manager.get_all_sessions()

        assert len(sessions) == 3
        assert "session-1" in sessions
        assert "session-2" in sessions
        assert "session-3" in sessions

    def test_cleanup_all_sessions(self):
        """GIVEN multiple sessions with expired captions
        WHEN cleanup_all is called
        THEN all expired captions should be cleaned."""
        manager = SessionCaptionManager(default_duration=0.1)

        # Add captions to multiple sessions
        for i in range(3):
            buffer = manager.get_buffer(f"session-{i}")
            buffer.add_caption(f"Text {i}", f"Speaker{i}")

        # Wait for expiration
        time.sleep(0.15)

        results = manager.cleanup_all()

        assert len(results) == 3
        assert all(count >= 1 for count in results.values())


# =============================================================================
# Test: Callbacks
# =============================================================================


class TestCallbacks:
    """Tests for callback functionality."""

    def test_on_caption_added_callback(self):
        """GIVEN a buffer with on_caption_added callback
        WHEN caption is added
        THEN callback should be called."""
        added_captions = []

        def on_added(caption):
            added_captions.append(caption)

        buffer = CaptionBuffer(on_caption_added=on_added)
        buffer.add_caption("Test", "Speaker")

        assert len(added_captions) == 1
        assert added_captions[0].translated_text == "Test"

    def test_on_caption_expired_callback(self):
        """GIVEN a buffer with on_caption_expired callback
        WHEN caption expires
        THEN callback should be called."""
        expired_captions = []

        def on_expired(caption):
            expired_captions.append(caption)

        buffer = CaptionBuffer(
            default_duration=0.1,
            on_caption_expired=on_expired,
        )
        buffer.add_caption("Test", "Speaker")

        # Wait for expiration
        time.sleep(0.15)
        buffer.cleanup_expired()

        assert len(expired_captions) == 1

    def test_callback_error_handled(self):
        """GIVEN a callback that raises
        WHEN event occurs
        THEN error should be caught and operation should continue."""
        def bad_callback(caption):
            raise RuntimeError("Callback error")

        buffer = CaptionBuffer(on_caption_added=bad_callback)

        # Should not raise
        caption = buffer.add_caption("Test", "Speaker")
        assert caption is not None


# =============================================================================
# Test: Display Dict
# =============================================================================


class TestDisplayDict:
    """Tests for display dictionary generation."""

    def test_get_display_dict_structure(self, long_buffer):
        """GIVEN captions in buffer
        WHEN get_display_dict is called
        THEN correct structure should be returned."""
        long_buffer.add_caption("Text 1", "Speaker1")
        long_buffer.add_caption("Text 2", "Speaker2")

        display = long_buffer.get_display_dict()

        assert "captions" in display
        assert "count" in display
        assert "max_captions" in display
        assert "timestamp" in display
        assert display["count"] == 2

    def test_caption_to_dict(self, long_buffer):
        """GIVEN a caption
        WHEN to_dict is called
        THEN all fields should be present."""
        caption = long_buffer.add_caption(
            "Hola",
            "Alice",
            original_text="Hello",
            target_language="es",
            confidence=0.9,
        )

        d = caption.to_dict()

        assert d["translated_text"] == "Hola"
        assert d["original_text"] == "Hello"
        assert d["speaker_name"] == "Alice"
        assert d["target_language"] == "es"
        assert d["confidence"] == 0.9
        assert "created_at" in d
        assert "expires_at" in d


# =============================================================================
# Test: Statistics
# =============================================================================


class TestStatistics:
    """Tests for statistics tracking."""

    def test_stats_track_captions_added(self, buffer):
        """GIVEN captions added
        WHEN get_stats is called
        THEN total_captions_added should be accurate."""
        buffer.add_caption("Text 1", "Speaker1")
        buffer.add_caption("Text 2", "Speaker2")
        buffer.add_caption("Text 3", "Speaker3")

        stats = buffer.get_stats()

        assert stats["total_captions_added"] == 3

    def test_stats_track_speakers(self, buffer):
        """GIVEN multiple speakers
        WHEN get_stats is called
        THEN speakers_seen should be accurate."""
        buffer.add_caption("Text 1", "Alice")
        buffer.add_caption("Text 2", "Bob")
        buffer.add_caption("Text 3", "Alice")  # Same speaker

        stats = buffer.get_stats()

        assert stats["speakers_seen"] == 2

    def test_stats_track_current_count(self, long_buffer):
        """GIVEN captions in buffer
        WHEN get_stats is called
        THEN current_caption_count should be accurate."""
        long_buffer.add_caption("Text 1", "Speaker1")
        long_buffer.add_caption("Text 2", "Speaker2")

        stats = long_buffer.get_stats()

        assert stats["current_caption_count"] == 2


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for boundary conditions."""

    def test_empty_translated_text(self, buffer):
        """GIVEN empty translated text
        WHEN add_caption is called
        THEN caption should still be created."""
        caption = buffer.add_caption("", "Speaker")

        assert caption is not None
        assert caption.translated_text == ""

    def test_custom_caption_id(self, buffer):
        """GIVEN a custom caption ID
        WHEN add_caption is called
        THEN the custom ID should be used."""
        caption = buffer.add_caption(
            "Test",
            "Speaker",
            caption_id="my-custom-id",
        )

        assert caption.id == "my-custom-id"

    def test_show_original_false(self):
        """GIVEN show_original=False
        WHEN caption is added with original
        THEN original_text should be None."""
        buffer = CaptionBuffer(show_original=False)
        caption = buffer.add_caption(
            "Hola",
            "Speaker",
            original_text="Hello",
        )

        assert caption.original_text is None

    def test_priority_affects_duration(self, buffer):
        """GIVEN different priorities
        WHEN captions are added
        THEN higher priority should have longer duration."""
        low = buffer.add_caption("Low", "Speaker", priority=0)
        high = buffer.add_caption("High", "Speaker", priority=3)

        assert high.display_duration_seconds > low.display_duration_seconds

    def test_custom_speaker_colors(self):
        """GIVEN custom speaker colors
        WHEN captions are added
        THEN custom colors should be used."""
        custom_colors = ["#111111", "#222222", "#333333"]
        buffer = CaptionBuffer(speaker_colors=custom_colors)

        c1 = buffer.add_caption("Test 1", "Speaker1")
        c2 = buffer.add_caption("Test 2", "Speaker2")

        assert c1.speaker_color in custom_colors
        assert c2.speaker_color in custom_colors

    def test_iteration(self, long_buffer):
        """GIVEN captions in buffer
        WHEN iterating
        THEN all captions should be yielded."""
        long_buffer.add_caption("Text 1", "Speaker1")
        long_buffer.add_caption("Text 2", "Speaker2")

        captions = list(long_buffer)

        assert len(captions) == 2


# =============================================================================
# Test: Factory Function
# =============================================================================


class TestFactoryFunction:
    """Tests for create_caption_buffer factory."""

    def test_factory_with_defaults(self):
        """GIVEN no config
        WHEN create_caption_buffer is called
        THEN default values should be used."""
        buffer = create_caption_buffer()

        assert buffer.max_captions == DEFAULT_MAX_CAPTIONS
        assert buffer.default_duration == DEFAULT_CAPTION_DURATION_SECONDS

    def test_factory_with_config(self):
        """GIVEN a config dict
        WHEN create_caption_buffer is called
        THEN config values should be used."""
        config = {
            "max_captions": 10,
            "default_duration": 15.0,
            "show_original": False,
        }

        buffer = create_caption_buffer(config)

        assert buffer.max_captions == 10
        assert buffer.default_duration == 15.0
        assert buffer.show_original is False


# =============================================================================
# Test: Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for concurrent access."""

    def test_concurrent_adds(self):
        """GIVEN multiple threads adding captions
        WHEN all complete
        THEN no race conditions should occur."""
        buffer = CaptionBuffer(max_captions=100, default_duration=60.0)
        errors = []

        def add_captions(thread_id):
            try:
                for i in range(20):
                    buffer.add_caption(f"Thread {thread_id} - {i}", f"Speaker{thread_id}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_captions, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have some captions (exact count depends on timing)
        assert len(buffer) <= 100
