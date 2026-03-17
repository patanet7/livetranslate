"""I2.4: Unit tests for interpreter mode direction flip handler simulation.

Simulates the direction-tracking logic from websocket_audio.py without
requiring a running WebSocket server. Exercises the context-clearing path
that fires when the detected language changes translation direction.
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
) -> list[int]:
    """Simulate the direction-flip handler from websocket_audio.py.

    For each (detected_lang, effective_target) pair:
    - Computes the direction string "lang->target"
    - If direction changed from the prior, calls service.clear_context()
      (current behavior: clears ALL directions)
    - Adds a fake context entry to simulate a successful translation

    Note: With DirectionalContextStore each direction has its own window,
    so the clear_context() on flip is no longer strictly necessary — the
    new direction's window starts fresh regardless. The clear is kept here
    to mirror current websocket_audio.py behavior.

    Returns: list of context sizes after processing each segment.
    """
    last_direction: str | None = None
    context_sizes: list[int] = []

    for detected_lang, effective_target in segments:
        if not effective_target:
            context_sizes.append(len(service.get_context(detected_lang, effective_target)))
            continue

        direction = f"{detected_lang}->{effective_target}"

        if last_direction and direction != last_direction:
            # Direction flip — clear all context (mirrors websocket_audio.py behavior)
            service.clear_context()

        last_direction = direction

        # Simulate a successful translation adding to context
        service.context_store.add(
            detected_lang,
            effective_target,
            f"source in {detected_lang}",
            f"translation in {effective_target}",
        )

        context_sizes.append(len(service.get_context(detected_lang, effective_target)))

    return context_sizes


class TestDirectionFlipClearsContext:
    def test_direction_flip_clears_context(self) -> None:
        """3 zh segments (context grows to 3), then 1 en segment (context cleared to 0,
        then grows to 1), then 1 more en (context grows to 2).

        Validates the websocket_audio.py direction-flip path.
        """
        service = _make_service(context_window_size=10)

        # 3 zh->en segments, then 1 en->zh segment, then 1 more en->zh
        segments = [
            ("zh", "en"),   # zh->en direction, context becomes 1
            ("zh", "en"),   # zh->en direction, context becomes 2
            ("zh", "en"),   # zh->en direction, context becomes 3
            ("en", "zh"),   # FLIP to en->zh: clear all (context -> 0) then add -> 1
            ("en", "zh"),   # en->zh direction, context becomes 2
        ]

        sizes = _simulate_direction_tracking(service, segments)

        # Verify context growth before flip
        assert sizes[0] == 1, f"After 1st zh seg, context={sizes[0]}, expected 1"
        assert sizes[1] == 2, f"After 2nd zh seg, context={sizes[1]}, expected 2"
        assert sizes[2] == 3, f"After 3rd zh seg, context={sizes[2]}, expected 3"

        # After flip: context was cleared then 1 entry added
        assert sizes[3] == 1, (
            f"After direction flip to en->zh, context={sizes[3]}, expected 1 "
            "(cleared then one entry added)"
        )

        # One more en->zh segment grows context again
        assert sizes[4] == 2, f"After 2nd en->zh seg, context={sizes[4]}, expected 2"

    def test_same_direction_preserves_context(self) -> None:
        """Multiple zh->en segments without a direction flip must not clear context."""
        service = _make_service(context_window_size=10)

        segments = [
            ("zh", "en"),
            ("zh", "en"),
            ("zh", "en"),
        ]

        sizes = _simulate_direction_tracking(service, segments)

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
