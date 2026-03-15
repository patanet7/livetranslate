"""Rolling context window for translation quality.

Maintains a ring buffer of recent (source, translation) pairs.
Each translation request includes the last N sentences as context,
giving the LLM continuity for pronouns, terminology, and tone.

Eviction: by count (max_entries) AND by token estimate (max_tokens),
whichever limit is hit first.
"""
from __future__ import annotations

from collections import deque

from livetranslate_common.models import TranslationContext


class RollingContextWindow:
    def __init__(self, max_entries: int = 5, max_tokens: int = 500):
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self._entries: deque[TranslationContext] = deque(maxlen=max_entries)

    def add(self, source_text: str, translation: str) -> None:
        """Add a successful translation pair to the context window."""
        self._entries.append(TranslationContext(text=source_text, translation=translation))
        self._evict_by_tokens()

    def get_context(self) -> list[TranslationContext]:
        """Return the current context window as a list."""
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def _evict_by_tokens(self) -> None:
        """Remove oldest entries until total tokens <= max_tokens."""
        while self._entries and self._total_tokens() > self.max_tokens:
            self._entries.popleft()

    def _total_tokens(self) -> int:
        return sum(
            self._estimate_tokens(e.text + e.translation) for e in self._entries
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English, ~2 for CJK."""
        # Simple heuristic — good enough for context window sizing
        return max(1, len(text) // 3)
