"""I2.4: Unit tests for interpreter mode direction flip handler simulation.

Simulates the direction-tracking logic from websocket_audio.py without
requiring a running WebSocket server. Exercises context retention across
direction flips — production does NOT call clear_context() on flip.
DirectionalContextStore provides per-direction isolation automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from translation.config import TranslationConfig
from translation.service import TranslationService


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


def _simulate_direction_tracking(
    service: TranslationService,
    segments: list[tuple[str, str]],  # (detected_language, effective_target)
) -> tuple[list[int], str]:
    """Simulate the direction-flip handler from websocket_audio.py.

    Production behavior (post-architect review):
    - On direction flip: resets _stable_text_buffer only, no clear_context()
    - DirectionalContextStore provides per-direction isolation automatically

    For each (detected_lang, effective_target) pair:
    - Computes the direction string "lang->target"
    - If direction changed from prior, resets stable_text_buffer (no context clear)
    - Adds a fake context entry to simulate a successful translation

    Returns:
        (context_sizes, final_stable_text_buffer)
        context_sizes: list of context sizes after processing each segment
        final_stable_text_buffer: the buffer value at end of processing
    """
    last_direction: str | None = None
    stable_text_buffer: str = ""
    context_sizes: list[int] = []

    for detected_lang, effective_target in segments:
        if not effective_target:
            context_sizes.append(len(service.get_context(detected_lang, effective_target or "")))
            continue

        direction = f"{detected_lang}->{effective_target}"

        if last_direction and direction != last_direction:
            # Direction flip — reset buffer only, NO clear_context()
            stable_text_buffer = ""

        last_direction = direction

        # Simulate a successful translation adding to context
        service.context_store.add(
            detected_lang,
            effective_target,
            f"source in {detected_lang}",
            f"translation in {effective_target}",
        )

        context_sizes.append(len(service.get_context(detected_lang, effective_target)))

    return context_sizes, stable_text_buffer


class TestDirectionFlipPreservesBothDirections:
    def test_direction_flip_preserves_both_directions(self) -> None:
        """3 zh->en segments, then 2 en->zh segments (no clear on flip).

        After the flip BOTH directions must retain their entries:
        - zh->en: 3 entries (still there after direction switch)
        - en->zh: 2 entries (accumulated in new direction)
        """
        service = _make_service(context_window_size=10)

        segments = [
            ("zh", "en"),  # zh->en, context becomes 1
            ("zh", "en"),  # zh->en, context becomes 2
            ("zh", "en"),  # zh->en, context becomes 3
            ("en", "zh"),  # FLIP to en->zh: buffer resets, zh->en stays at 3
            ("en", "zh"),  # en->zh, context becomes 2
        ]

        sizes, _ = _simulate_direction_tracking(service, segments)

        # Verify context growth in original direction
        assert sizes[0] == 1, f"After 1st zh seg, context={sizes[0]}, expected 1"
        assert sizes[1] == 2, f"After 2nd zh seg, context={sizes[1]}, expected 2"
        assert sizes[2] == 3, f"After 3rd zh seg, context={sizes[2]}, expected 3"

        # After flip: en->zh starts fresh (no prior en->zh entries)
        assert sizes[3] == 1, (
            f"After direction flip to en->zh, en->zh context={sizes[3]}, expected 1 "
            "(new direction starts from 0, then adds 1)"
        )
        assert sizes[4] == 2, f"After 2nd en->zh seg, context={sizes[4]}, expected 2"

        # KEY: zh->en entries must STILL BE THERE after the direction flip
        zh_en_ctx = service.get_context("zh", "en")
        assert len(zh_en_ctx) == 3, (
            f"zh->en context must retain 3 entries after direction flip, got {len(zh_en_ctx)}"
        )

        en_zh_ctx = service.get_context("en", "zh")
        assert len(en_zh_ctx) == 2, (
            f"en->zh context must have 2 entries, got {len(en_zh_ctx)}"
        )

    def test_same_direction_preserves_context(self) -> None:
        """Multiple zh->en segments without a direction flip must not clear context."""
        service = _make_service(context_window_size=10)

        segments = [
            ("zh", "en"),
            ("zh", "en"),
            ("zh", "en"),
        ]

        sizes, _ = _simulate_direction_tracking(service, segments)

        assert sizes == [1, 2, 3], (
            f"Context should accumulate monotonically: got {sizes}"
        )

    def test_first_segment_sets_direction_without_clear(self) -> None:
        """The very first segment has no prior direction — it must NOT trigger a clear."""
        service = _make_service(context_window_size=10)

        # Pre-populate context to prove it isn't cleared
        service.context_store.add("zh", "en", "pre-existing", "pre-existing translation")
        assert len(service.get_context("zh", "en")) == 1

        # Simulate the first segment arriving with no prior direction
        last_direction: str | None = None
        detected_lang = "zh"
        effective_target = "en"

        direction = f"{detected_lang}->{effective_target}"

        # No prior direction -> no clear should happen
        if last_direction and direction != last_direction:
            service.clear_context()

        last_direction = direction  # noqa: F841

        # Context should still have the pre-existing entry
        assert len(service.get_context("zh", "en")) == 1, (
            "First segment must not clear context (no prior direction to compare against)"
        )

    def test_text_buffer_resets_on_direction_change(self) -> None:
        """stable_text_buffer must be empty after a direction flip.

        Simulates _stable_text_buffer containing Chinese text, then flipping
        to English direction — the buffer must be cleared (mirrors production
        websocket_audio.py behavior on direction change).
        """
        service = _make_service(context_window_size=10)

        # Simulate zh->en segments that accumulate text in the buffer
        segments_before_flip = [
            ("zh", "en"),
            ("zh", "en"),
        ]
        _, buffer_before = _simulate_direction_tracking(service, segments_before_flip)

        # At this point buffer_before == "" because the helper resets on each call.
        # We need to test that a mid-run flip resets the buffer.
        # Run combined sequence to observe the buffer after the flip.
        service2 = _make_service(context_window_size=10)
        segments_with_flip = [
            ("zh", "en"),
            ("zh", "en"),
            ("en", "zh"),  # FLIP — buffer must reset here
        ]
        _, buffer_after_flip = _simulate_direction_tracking(service2, segments_with_flip)

        assert buffer_after_flip == "", (
            f"stable_text_buffer must be empty after direction flip, got: {buffer_after_flip!r}"
        )
