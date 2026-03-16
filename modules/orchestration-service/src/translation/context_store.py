"""Directional context store for translation quality.

Wraps RollingContextWindow with per-(source_lang, target_lang) keys.
Interpreter mode uses two directions simultaneously (zh->en, en->zh).
Standard mode uses one. Direction flip is a no-op — separate keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from livetranslate_common.models import TranslationContext

from translation.context import RollingContextWindow


@dataclass
class DirectionalContextStore:
    """Per-(source_lang, target_lang) rolling context windows."""

    max_entries: int = 5
    max_tokens: int = 800
    cross_direction_max_tokens: int = 200
    _windows: dict[tuple[str, str], RollingContextWindow] = field(
        default_factory=dict, repr=False,
    )

    def _key(self, source: str, target: str) -> tuple[str, str]:
        return (source.lower(), target.lower())

    def get(self, source: str, target: str) -> list[TranslationContext]:
        key = self._key(source, target)
        if key not in self._windows:
            return []
        return self._windows[key].get_context()

    def get_cross_direction(self, source: str, target: str) -> list[TranslationContext]:
        """Get 1-2 entries from the opposite direction for referent tracking."""
        opposite = self._key(target, source)
        if opposite not in self._windows:
            return []
        entries = self._windows[opposite].get_context()
        # Return last 2 entries (token-level truncation deferred to Phase 5)
        return entries[-2:] if entries else []

    def add(
        self, source: str, target: str, source_text: str, translation: str,
    ) -> None:
        key = self._key(source, target)
        if key not in self._windows:
            self._windows[key] = RollingContextWindow(
                max_entries=self.max_entries,
                max_tokens=self.max_tokens,
            )
        self._windows[key].add(source_text, translation)

    def clear_direction(self, source: str, target: str) -> None:
        key = self._key(source, target)
        if key in self._windows:
            self._windows[key].clear()

    def clear_all(self) -> None:
        self._windows.clear()
