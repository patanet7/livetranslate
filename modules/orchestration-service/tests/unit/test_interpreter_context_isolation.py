"""I2.3: Unit tests for interpreter mode context isolation.

Tests that context windows are correctly managed across translation direction
changes. Covers both current behavior (clear all on flip) and the
DirectionalContextStore design (per-direction independent windows).
"""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from translation.service import TranslationService
from translation.config import TranslationConfig


def _make_service(context_window_size: int = 5) -> TranslationService:
    """Create a TranslationService with a non-reachable LLM endpoint.

    Used for context management tests that don't need actual LLM calls.
    """
    config = TranslationConfig(
        base_url="http://localhost:1",
        model="test-model",
        context_window_size=context_window_size,
        max_context_tokens=500,
        max_queue_depth=5,
        timeout_s=1,
    )
    return TranslationService(config)


class TestInterpreterContextIsolationCurrent:
    """Tests for the current clear_context() behavior on direction flip."""

    def test_context_cleared_before_direction_flip(self) -> None:
        """Add ZH->EN context entries, call clear_context(), verify empty."""
        service = _make_service()

        # Populate context as if we translated 3 zh->en segments
        service.context_store.add("zh", "en", "你好", "Hello")
        service.context_store.add("zh", "en", "今天天气很好", "The weather is nice today")

        ctx_before = service.get_context("zh", "en")
        assert len(ctx_before) == 2, f"Expected 2 entries before clear, got {len(ctx_before)}"

        # Simulate direction flip: clear context (current behavior)
        service.clear_context()

        ctx_after = service.get_context("zh", "en")
        assert len(ctx_after) == 0, (
            f"Context should be empty after clear, got {len(ctx_after)} entries"
        )

    def test_context_accumulates_in_new_direction_after_clear(self) -> None:
        """After clearing ZH->EN context, add EN->ZH entries and verify fresh accumulation."""
        service = _make_service()

        # Pre-populate zh->en context
        service.context_store.add("zh", "en", "你好", "Hello")

        # Direction flip: clear all
        service.clear_context()
        assert len(service.get_context("zh", "en")) == 0

        # Now accumulate EN->ZH context (reversed direction)
        service.context_store.add("en", "zh", "Hello", "你好")
        service.context_store.add("en", "zh", "Good morning", "早上好")

        ctx = service.get_context("en", "zh")
        assert len(ctx) == 2, f"Expected 2 EN->ZH entries, got {len(ctx)}"
        assert ctx[0].text == "Hello"
        assert ctx[0].translation == "你好"
        assert ctx[1].text == "Good morning"
        assert ctx[1].translation == "早上好"

    def test_clear_context_with_no_args_clears_all_windows(self) -> None:
        """Multiple per-direction context windows: clear() with no args removes all of them."""
        service = _make_service()

        # Add context for two directions
        service.context_store.add("zh", "en", "你好", "Hello")
        service.context_store.add("en", "zh", "Goodbye", "再见")

        # Verify both directions have entries before clear
        assert len(service.get_context("zh", "en")) == 1, "zh->en should have 1 entry"
        assert len(service.get_context("en", "zh")) == 1, "en->zh should have 1 entry"
        assert len(service.context_store._windows) == 2, "Should have 2 direction windows"

        # clear_context() with no args must drop all internal context windows
        service.clear_context()

        # After clear, the internal dict must be empty
        assert len(service.context_store._windows) == 0, (
            f"All context windows should be removed after clear_context(), "
            f"but found {len(service.context_store._windows)}"
        )

        # Accessing a direction's context returns empty list (no re-creation of window)
        assert len(service.get_context("zh", "en")) == 0, "zh->en should be empty after clear"
        assert len(service.get_context("en", "zh")) == 0, "en->zh should be empty after clear"


class TestDirectionalContextStoreTarget:
    """Tests for DirectionalContextStore per-direction independent windows.

    DirectionalContextStore is implemented: directions are keyed by
    (source_lang, target_lang) and survive each other's direction flips.
    """

    def test_zh_en_context_independent_from_en_zh(self) -> None:
        """zh->en and en->zh context windows must be fully independent."""
        service = _make_service()

        # Populate zh->en direction
        service.context_store.add("zh", "en", "你好", "Hello")
        service.context_store.add("zh", "en", "谢谢", "Thank you")

        # Populate en->zh direction
        service.context_store.add("en", "zh", "Good morning", "早上好")

        zh_en_ctx = service.get_context("zh", "en")
        en_zh_ctx = service.get_context("en", "zh")

        assert len(zh_en_ctx) == 2, f"zh->en should have 2 entries, got {len(zh_en_ctx)}"
        assert zh_en_ctx[0].text == "你好"
        assert zh_en_ctx[1].text == "谢谢"

        assert len(en_zh_ctx) == 1, f"en->zh should have 1 entry, got {len(en_zh_ctx)}"
        assert en_zh_ctx[0].text == "Good morning"

    def test_direction_flip_does_not_clear_prior_direction(self) -> None:
        """Adding en->zh context must NOT disturb the zh->en window.

        The DirectionalContextStore uses separate keys per direction, so a
        direction flip (adding to the opposite direction) leaves the original
        direction's history intact. No explicit clear is needed on flip.
        """
        service = _make_service()

        # Add two zh->en entries
        service.context_store.add("zh", "en", "你好", "Hello")
        service.context_store.add("zh", "en", "谢谢", "Thank you")

        # Simulate a direction flip: add one en->zh entry
        service.context_store.add("en", "zh", "Good morning", "早上好")

        # Add one more zh->en entry after the flip
        service.context_store.add("zh", "en", "再见", "Goodbye")

        zh_en_ctx = service.get_context("zh", "en")
        assert len(zh_en_ctx) == 3, (
            f"zh->en should have 3 entries after interleaved en->zh add, got {len(zh_en_ctx)}"
        )
        # The en->zh add in the middle must NOT have cleared zh->en history
        assert zh_en_ctx[0].text == "你好"
        assert zh_en_ctx[1].text == "谢谢"
        assert zh_en_ctx[2].text == "再见"
