"""MeetingSessionConfig — single control plane for meeting subtitle behavior.

Plain class with explicit update() for thread-safe mutation.
NOT a Pydantic BaseModel — uses threading.Lock and subscriber callbacks.

Composes existing configs:
- Language state (source_lang, target_lang)
- Display state (mode, theme, font)
- Source routing (caption_source) — genuinely new
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from livetranslate_common.logging import get_logger

logger = get_logger()

ConfigSubscriber = Callable[[set[str]], None]


class MeetingSessionConfig:
    """Thread-safe, observable config for meeting subtitle sessions."""

    __slots__ = (
        "session_id",
        "bot_id",
        "caption_source",
        "source_lang",
        "target_lang",
        "translation_enabled",
        "display_mode",
        "theme",
        "font_size",
        "show_speakers",
        "show_original",
        "_lock",
        "_subscribers",
    )

    def __init__(
        self,
        session_id: str,
        bot_id: str | None = None,
        caption_source: str = "bot_audio",
        source_lang: str = "auto",
        target_lang: str = "en",
        translation_enabled: bool = True,
        display_mode: str = "subtitle",
        theme: str = "dark",
        font_size: int = 24,
        show_speakers: bool = True,
        show_original: bool = False,
    ):
        self.session_id = session_id
        self.bot_id = bot_id
        self.caption_source = caption_source
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translation_enabled = translation_enabled
        self.display_mode = display_mode
        self.theme = theme
        self.font_size = font_size
        self.show_speakers = show_speakers
        self.show_original = show_original
        self._lock = threading.Lock()
        self._subscribers: list[ConfigSubscriber] = []

    def update(self, **changes: Any) -> set[str]:
        """Apply changes atomically. Returns set of changed field names.
        Thread-safe. Fires subscriber notification once per batch.
        Ignores unknown fields and private fields silently."""
        changed: set[str] = set()
        with self._lock:
            for field, value in changes.items():
                if field.startswith("_") or not hasattr(self, field):
                    continue
                if getattr(self, field) != value:
                    object.__setattr__(self, field, value)
                    changed.add(field)
        if changed:
            self._notify_subscribers(changed)
        return changed

    def snapshot(self) -> dict[str, Any]:
        """Return a frozen copy of all config values. For per-frame rendering."""
        with self._lock:
            return {
                slot: getattr(self, slot)
                for slot in self.__slots__
                if not slot.startswith("_")
            }

    def subscribe(self, callback: ConfigSubscriber) -> None:
        """Add a subscriber. Called with set of changed field names."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: ConfigSubscriber) -> None:
        """Remove a subscriber."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def _notify_subscribers(self, changed: set[str]) -> None:
        """Notify all subscribers of changes."""
        for sub in self._subscribers:
            try:
                sub(changed)
            except Exception as e:
                logger.warning("config_subscriber_error", error=str(e))
