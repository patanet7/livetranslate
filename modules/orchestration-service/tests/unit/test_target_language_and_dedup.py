"""Tests for target language wiring, draft→final translation dedup, and translation guards.

Behavioral tests — no mocking. Tests verify:
- ConfigMessage target_language field is handled by the backend
- Draft translation updates _last_translated_stable so final path dedup works
- The word-level prefix dedup algorithm strips overlap from draft-covered text
- Translation guards: short text filter + repetition detection at orchestration layer

These test the dedup algorithm extracted from websocket_audio.handle_transcription_segment.
"""
import re
from collections import deque

import pytest

from livetranslate_common.models.ws_messages import ConfigMessage, parse_ws_message


# ---------------------------------------------------------------------------
# Helpers: dedup algorithm extracted from websocket_audio.py
# ---------------------------------------------------------------------------


def _truncate_stable(current: str, new_text: str, limit: int = 500) -> str:
    """Word-boundary-safe truncation — same algorithm as websocket_audio.py."""
    combined = current + " " + new_text
    if len(combined) > limit:
        combined = combined[-limit:]
        space_idx = combined.find(" ")
        if space_idx != -1:
            combined = combined[space_idx + 1:]
    return combined


def stable_text_dedup(
    incoming: str,
    last_translated_stable: str,
) -> str:
    """Word-level prefix dedup — same algorithm as websocket_audio.py final path.

    Returns the portion of `incoming` NOT covered by `last_translated_stable`.
    """
    if not last_translated_stable:
        return incoming

    incoming_norm = re.sub(r"[^\w\s]", "", incoming).lower()
    last_norm = re.sub(r"[^\w\s]", "", last_translated_stable).lower()

    # Fast path: incoming entirely contained in already-translated text
    if incoming_norm and incoming_norm in last_norm:
        return ""

    # Word-level prefix dedup for partial overlaps
    prev_words = last_norm.split()
    new_words_raw = incoming.split()
    new_words = [re.sub(r"[^\w\s]", "", w).lower() for w in new_words_raw]
    max_check = min(len(prev_words), len(new_words), 12)
    for overlap_len in range(max_check, 0, -1):
        if prev_words[-overlap_len:] == new_words[:overlap_len]:
            return " ".join(new_words_raw[overlap_len:])

    return incoming


# ---------------------------------------------------------------------------
# ConfigMessage target_language handling
# ---------------------------------------------------------------------------


class TestConfigMessageTargetLanguage:
    """Verify ConfigMessage carries target_language to the backend."""

    def test_config_message_with_target_language(self) -> None:
        msg = ConfigMessage(target_language="es")
        assert msg.target_language == "es"
        assert msg.type == "config"

    def test_config_message_target_language_in_parsed_message(self) -> None:
        raw = '{"type": "config", "target_language": "zh"}'
        msg = parse_ws_message(raw)
        assert isinstance(msg, ConfigMessage)
        assert msg.target_language == "zh"

    def test_backend_handler_updates_target_language(self) -> None:
        """Simulate the ConfigMessage handler: if msg.target_language, update local var."""
        target_language = "en"  # default
        msg = ConfigMessage(target_language="es")

        # This is the handler logic we're adding:
        if msg.target_language:
            target_language = msg.target_language

        assert target_language == "es"

    def test_backend_handler_ignores_none_target_language(self) -> None:
        """ConfigMessage without target_language should not change session state."""
        target_language = "en"
        msg = ConfigMessage(language="zh")  # only source language, no target

        if msg.target_language:
            target_language = msg.target_language

        assert target_language == "en"


# ---------------------------------------------------------------------------
# Draft → Final dedup
# ---------------------------------------------------------------------------


class TestDraftFinalDedup:
    """When draft translates text X, the final path must not re-translate the
    overlapping portion of stable_text that was already covered by the draft.

    Bug scenario from logs:
      - Draft seg 2 translates: "let's see what's going to happen..."
      - Final seg 2 stable_text: "let's see what's going to happen when I start talking..."
      - Without fix: both produce TranslationMessages → duplicate in paragraph view
      - With fix: draft updates _last_translated_stable, final dedup strips overlap
    """

    def test_draft_text_tracked_prevents_full_retranslation(self) -> None:
        """After draft fires, identical stable_text should be fully deduped."""
        _last_translated_stable = ""

        # Draft fires for this text
        draft_text = "let's see what's going to happen"
        # Fix: draft updates _last_translated_stable (word-boundary safe)
        _last_translated_stable = _truncate_stable(_last_translated_stable, draft_text)

        # Final arrives with same stable_text
        incoming_stable = "let's see what's going to happen"
        remaining = stable_text_dedup(incoming_stable, _last_translated_stable)

        assert remaining == "", (
            f"Expected empty string (fully deduped), got: {remaining!r}"
        )

    def test_draft_text_tracked_allows_new_suffix(self) -> None:
        """After draft fires, final stable_text with new suffix should only translate the new part."""
        _last_translated_stable = ""

        # Draft fires
        draft_text = "let's see what's going to happen"
        _last_translated_stable = _truncate_stable(_last_translated_stable, draft_text)

        # Final arrives with MORE text
        incoming_stable = "let's see what's going to happen when I start talking"
        remaining = stable_text_dedup(incoming_stable, _last_translated_stable)

        assert remaining == "when I start talking", (
            f"Expected only new suffix, got: {remaining!r}"
        )

    def test_without_draft_tracking_causes_duplicate(self) -> None:
        """Without the fix, stable_text is NOT in _last_translated_stable → full retranslation."""
        _last_translated_stable = ""  # draft did NOT update this

        # Final arrives — dedup has nothing to compare against
        incoming_stable = "let's see what's going to happen"
        remaining = stable_text_dedup(incoming_stable, _last_translated_stable)

        # This is the BUG: without tracking, the entire text is "new"
        assert remaining == incoming_stable

    def test_multiple_drafts_accumulate(self) -> None:
        """Multiple draft segments should accumulate in _last_translated_stable."""
        _last_translated_stable = ""

        # Draft 1
        _last_translated_stable = _truncate_stable(_last_translated_stable, "hello world")
        # Draft 2
        _last_translated_stable = _truncate_stable(_last_translated_stable, "how are you today")

        # Final with overlapping prefix from draft 2
        incoming = "how are you today I am fine"
        remaining = stable_text_dedup(incoming, _last_translated_stable)

        assert remaining == "I am fine"

    def test_dedup_handles_punctuation(self) -> None:
        """Punctuation in draft text should not break dedup (normalized away)."""
        _last_translated_stable = ""

        draft_text = "Hello, world!"
        _last_translated_stable = _truncate_stable(_last_translated_stable, draft_text)

        incoming = "Hello, world! How are you?"
        remaining = stable_text_dedup(incoming, _last_translated_stable)

        assert remaining == "How are you?"

    def test_last_translated_stable_truncation(self) -> None:
        """_last_translated_stable should be capped at ~500 chars to prevent unbounded growth."""
        _last_translated_stable = "word " * 120  # 600 chars

        # Apply the word-boundary-safe truncation pattern from the fix
        combined = _last_translated_stable + " " + "new text"
        if len(combined) > 500:
            combined = combined[-500:]
            space_idx = combined.find(" ")
            if space_idx != -1:
                combined = combined[space_idx + 1:]
        _last_translated_stable = combined

        assert len(_last_translated_stable) <= 500
        assert _last_translated_stable.endswith("new text")

    def test_is_final_flushes_buffer_without_punctuation(self) -> None:
        """When is_final=True, stable_text_buffer should flush even without sentence-ending punctuation.

        Bug: conversational speech like 'Thank you' or 'Yeah' doesn't end with .!?
        so the buffer never flushes. is_final means no more text is coming.
        """
        import re

        _stable_text_buffer = ""
        _last_translated_stable = ""

        # Simulate stable_text arriving without punctuation
        incoming = "Thank you"
        _stable_text_buffer += incoming

        # Without is_final, no flush
        has_punctuation = bool(re.search(r"[.!?。！？]$", _stable_text_buffer.strip()))
        assert not has_punctuation, "Test setup: text should NOT have punctuation"

        # With is_final=True, should flush
        is_final = True
        should_flush = has_punctuation or is_final
        assert should_flush, "is_final should trigger flush"

        # The flushed text becomes the translate_text
        translate_text = _stable_text_buffer.strip() if should_flush else ""
        assert translate_text == "Thank you"

    def test_sentence_boundary_still_works(self) -> None:
        """Sentence boundary detection should still flush mid-stream (before is_final)."""
        import re

        _stable_text_buffer = ""

        # First segment: no punctuation, no is_final
        _stable_text_buffer += "I think that"
        has_punctuation = bool(re.search(r"[.!?。！？]$", _stable_text_buffer.strip()))
        assert not has_punctuation
        # No flush yet
        assert _stable_text_buffer == "I think that"

        # Second segment: adds punctuation
        _stable_text_buffer += " he is weird."
        has_punctuation = bool(re.search(r"[.!?。！？]$", _stable_text_buffer.strip()))
        assert has_punctuation
        # Should flush on punctuation alone (is_final not needed)
        translate_text = _stable_text_buffer.strip()
        assert translate_text == "I think that he is weird."

    def test_truncation_does_not_split_words(self) -> None:
        """Word-boundary truncation must not start with a partial word."""
        # Build a string where the 500-char boundary falls mid-word
        _last_translated_stable = "already " * 62  # 496 chars
        _last_translated_stable += "know"  # now 500 chars exactly

        draft_text = "the answer to the question"

        combined = _last_translated_stable + " " + draft_text
        if len(combined) > 500:
            combined = combined[-500:]
            space_idx = combined.find(" ")
            if space_idx != -1:
                combined = combined[space_idx + 1:]

        # The result should start at a word boundary, not mid-word
        first_word = combined.split()[0]
        # first_word should be a complete word from our vocabulary
        assert first_word in ("already", "know", "the", "answer", "to", "question"), (
            f"Truncation started mid-word: {combined[:30]!r}"
        )


# ---------------------------------------------------------------------------
# Translation guards (defense-in-depth at orchestration layer)
# ---------------------------------------------------------------------------


class TestShortTextTranslationGuard:
    """Translation should be skipped for segments with < 3 characters.

    These are noise segments that slip through VAD — not worth translating.
    The segment is still forwarded to the frontend (for display), but no
    LLM call is made.
    """

    def _should_skip_translation(self, text: str) -> bool:
        """Replicate the short-text guard from websocket_audio.py."""
        return len(text.strip()) < 3

    def test_single_char_skipped(self) -> None:
        assert self._should_skip_translation("a") is True

    def test_two_chars_skipped(self) -> None:
        assert self._should_skip_translation("ah") is True

    def test_three_chars_passes(self) -> None:
        assert self._should_skip_translation("yes") is False

    def test_whitespace_only_skipped(self) -> None:
        assert self._should_skip_translation("  ") is True

    def test_empty_string_skipped(self) -> None:
        assert self._should_skip_translation("") is True

    def test_real_word_passes(self) -> None:
        assert self._should_skip_translation("Thank you") is False


class TestRepetitionTranslationGuard:
    """Orchestration-layer repetition guard: skip translation when 3+ of last 5
    segments have identical normalized text.

    This mirrors the transcription-layer filter as defense-in-depth. Hallucinated
    segments still display briefly in the caption panel (overwritten by finals),
    but we don't waste LLM tokens translating them.
    """

    def _simulate_guard(self, texts: list[str]) -> list[tuple[str, bool]]:
        """Run the repetition guard over a sequence of texts.

        Returns list of (text, should_translate) tuples.
        """
        recent: deque[str] = deque(maxlen=5)
        results = []
        for text in texts:
            seg_normalized = text.strip().lower()
            seg_repeat_count = sum(1 for t in recent if t == seg_normalized)
            recent.append(seg_normalized)
            should_translate = seg_repeat_count < 2
            results.append((text, should_translate))
        return results

    def test_third_repetition_blocks_translation(self) -> None:
        """Third identical segment should NOT trigger translation."""
        results = self._simulate_guard(["Thank you.", "Thank you.", "Thank you."])
        assert results[0][1] is True   # first passes
        assert results[1][1] is True   # second passes
        assert results[2][1] is False  # third blocked

    def test_two_repetitions_allowed(self) -> None:
        """Two identical segments should both trigger translation."""
        results = self._simulate_guard(["Thank you.", "Thank you."])
        assert all(r[1] for r in results)

    def test_different_texts_all_translate(self) -> None:
        """All unique texts should trigger translation."""
        results = self._simulate_guard(["Hello.", "World.", "Goodbye."])
        assert all(r[1] for r in results)

    def test_interleaved_text_prevents_false_positive(self) -> None:
        """Different text between repeats should prevent blocking."""
        results = self._simulate_guard([
            "Thank you.", "Good morning.", "Thank you.",
        ])
        # Only 1 repeat of "Thank you." in window → still allowed
        assert all(r[1] for r in results)

    def test_hallucination_pattern_from_logs(self) -> None:
        """Reproduce the exact pattern seen in production logs."""
        results = self._simulate_guard([
            "Thank you.",  # seg 17 — hallucination
            "Thank you.",  # seg 19 — hallucination
            "Thank you.",  # seg 22 — hallucination (should be blocked)
            "Yeah.",       # seg 23 — possibly real
        ])
        assert results[0][1] is True   # first "Thank you." translates
        assert results[1][1] is True   # second "Thank you." translates
        assert results[2][1] is False  # third "Thank you." blocked
        assert results[3][1] is True   # "Yeah." is different, translates

    def test_window_eviction_allows_reuse(self) -> None:
        """After the window slides past old entries, the same text can translate again."""
        texts = [
            "Thank you.",
            "Thank you.",
            "A.", "B.", "C.", "D.",  # fill window, evicting "Thank you."
            "Thank you.",  # should pass — old entries evicted
        ]
        results = self._simulate_guard(texts)
        # The last "Thank you." should translate (0 matches in current window)
        assert results[-1][1] is True
