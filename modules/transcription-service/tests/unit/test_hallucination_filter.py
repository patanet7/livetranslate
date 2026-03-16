"""Tests for consolidated HallucinationFilter class.

All tests construct TranscriptionResult objects directly — no Whisper needed.
RED-GREEN TDD: these tests are written first, then the implementation.
"""
from __future__ import annotations

import pytest
from livetranslate_common.models.transcription import TranscriptionResult

from transcription.hallucination_filter import HallucinationFilter, trim_trailing_repetition


def _result(
    text: str = "hello world",
    confidence: float = 0.9,
    compression_ratio: float | None = None,
    no_speech_prob: float | None = None,
) -> TranscriptionResult:
    """Helper to build a minimal TranscriptionResult."""
    return TranscriptionResult(
        text=text,
        language="en",
        confidence=confidence,
        compression_ratio=compression_ratio,
        no_speech_prob=no_speech_prob,
    )


class TestCompressionRatioGate:
    """Gate 1: compression_ratio > 2.4 → suppress."""

    def test_high_compression_ratio_suppressed(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(compression_ratio=3.0))
        assert suppressed is True
        assert reason == "compression_ratio"

    def test_normal_compression_ratio_passes(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(compression_ratio=1.5))
        assert suppressed is False
        assert reason == ""

    def test_compression_ratio_none_passes(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(compression_ratio=None))
        assert suppressed is False

    def test_compression_ratio_at_threshold_passes(self):
        """2.4 is not > 2.4, so it should pass."""
        f = HallucinationFilter()
        suppressed, _ = f.should_suppress(_result(compression_ratio=2.4))
        assert suppressed is False

    def test_compression_ratio_just_above_threshold_suppressed(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(compression_ratio=2.41))
        assert suppressed is True
        assert reason == "compression_ratio"


class TestBohPhraseGate:
    """Gate 2: Bag-of-Hallucinations exact phrase matching."""

    def test_thank_you_suppressed(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(text="Thank you."))
        assert suppressed is True
        assert reason == "boh_phrase"

    def test_thank_you_no_period_suppressed(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(text="Thank you"))
        assert suppressed is True
        assert reason == "boh_phrase"

    def test_please_subscribe_suppressed(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(text="Please subscribe"))
        assert suppressed is True
        assert reason == "boh_phrase"

    def test_thank_you_for_help_passes(self):
        """Longer phrases containing 'thank you' must NOT match (exact match only)."""
        f = HallucinationFilter()
        suppressed, _ = f.should_suppress(_result(text="Thank you for your help"))
        assert suppressed is False

    def test_subtitles_by_suppressed(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(text="Subtitles by"))
        assert suppressed is True
        assert reason == "boh_phrase"

    def test_case_insensitive(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(text="THANK YOU"))
        assert suppressed is True
        assert reason == "boh_phrase"


class TestConfidenceGates:
    """Gate 3: Two-tier confidence filtering."""

    def test_short_low_confidence_suppressed(self):
        """1-word segment with confidence 0.4 → suppressed (< 0.55)."""
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(_result(text="Thank", confidence=0.4))
        assert suppressed is True
        assert reason == "low_confidence_short"

    def test_short_at_threshold_passes(self):
        """2-word segment with confidence 0.55 → passes."""
        f = HallucinationFilter()
        suppressed, _ = f.should_suppress(_result(text="Good morning", confidence=0.55))
        assert suppressed is False

    def test_any_very_low_confidence_suppressed(self):
        """5-word segment with confidence 0.2 → suppressed (< 0.3)."""
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(
            _result(text="This is a long sentence", confidence=0.2)
        )
        assert suppressed is True
        assert reason == "low_confidence"

    def test_moderate_confidence_long_passes(self):
        """5-word segment with confidence 0.4 → passes."""
        f = HallucinationFilter()
        suppressed, _ = f.should_suppress(
            _result(text="This is a long sentence", confidence=0.4)
        )
        assert suppressed is False


class TestIntraWordRepetition:
    """Gate 4: Intra-word repetition (>5 words, <20% unique)."""

    def test_highly_repetitive_suppressed(self):
        f = HallucinationFilter()
        suppressed, reason = f.should_suppress(
            _result(text="hello hello hello hello hello hello")
        )
        assert suppressed is True
        assert reason == "intra_word_repetition"

    def test_short_repetitive_passes(self):
        """≤5 words — intra-word gate should not fire."""
        f = HallucinationFilter()
        suppressed, _ = f.should_suppress(_result(text="hello hello hello"))
        assert suppressed is False

    def test_diverse_words_passes(self):
        f = HallucinationFilter()
        suppressed, _ = f.should_suppress(
            _result(text="the quick brown fox jumps over the lazy dog")
        )
        assert suppressed is False


class TestCrossSegmentRepetition:
    """Gate 5: Cross-segment repetition (deque of last 5, threshold 2)."""

    def test_third_identical_segment_suppressed(self):
        f = HallucinationFilter()
        r = _result(text="Good morning")
        f.should_suppress(r)  # 1st — passes
        f.should_suppress(r)  # 2nd — passes
        suppressed, reason = f.should_suppress(r)  # 3rd — suppressed
        assert suppressed is True
        assert reason == "cross_segment_repetition"

    def test_two_identical_segments_pass(self):
        f = HallucinationFilter()
        r = _result(text="Good morning")
        s1, _ = f.should_suppress(r)
        s2, _ = f.should_suppress(r)
        assert s1 is False
        assert s2 is False

    def test_diverse_segments_all_pass(self):
        f = HallucinationFilter()
        texts = ["Hello.", "World.", "How are you?", "I am fine.", "Goodbye."]
        for text in texts:
            suppressed, _ = f.should_suppress(_result(text=text))
            assert suppressed is False


class TestGateOrdering:
    """Verify gate ordering: suppressed results don't pollute the repetition deque."""

    def test_high_cr_does_not_pollute_deque(self):
        """Results suppressed by compression_ratio should NOT enter the deque."""
        f = HallucinationFilter()
        bad = _result(text="Thank you.", compression_ratio=3.0)
        # Feed 3 high-CR results — all suppressed by gate 1
        for _ in range(3):
            suppressed, reason = f.should_suppress(bad)
            assert suppressed is True
            assert reason == "compression_ratio"
        # Now a normal "Thank you." without high CR — should not be suppressed
        # by cross-segment repetition (deque should be empty)
        # Note: it WILL be caught by BoH gate instead
        normal = _result(text="Normal speech here")
        suppressed, _ = f.should_suppress(normal)
        assert suppressed is False

    def test_boh_suppressed_does_not_pollute_deque(self):
        """Results suppressed by BoH should NOT enter the deque."""
        f = HallucinationFilter()
        for _ in range(5):
            f.should_suppress(_result(text="Thank you."))  # BoH catches it
        # Deque should be empty — unique text should pass
        suppressed, _ = f.should_suppress(_result(text="Unique sentence here"))
        assert suppressed is False


class TestTrimTrailingRepetition:
    """trim_trailing_repetition: collapse degeneration tails, keep real prefix."""

    def test_collapses_long_trailing_repeat(self):
        """Classic Whisper degeneration: real prefix + 'Yeah.' x20."""
        text = (
            "That's true. Yeah. So like the proportion of the graph is different. "
            + "Yeah. " * 20
        ).strip()
        result = trim_trailing_repetition(text)
        assert result == "That's true. Yeah. So like the proportion of the graph is different. Yeah."

    def test_no_repetition_unchanged(self):
        """Normal text with no trailing repetition passes through."""
        text = "The quick brown fox jumps over the lazy dog."
        assert trim_trailing_repetition(text) == text

    def test_short_natural_repetition_kept(self):
        """'yeah yeah yeah' (3 repeats) is natural speech, not degeneration."""
        text = "I agree yeah yeah yeah"
        assert trim_trailing_repetition(text) == text

    def test_exactly_at_threshold_trimmed(self):
        """5 identical trailing words should trigger trimming."""
        text = "Hello world. Ok. Ok. Ok. Ok. Ok."
        result = trim_trailing_repetition(text)
        assert result == "Hello world. Ok."

    def test_entire_text_is_repetition(self):
        """If the whole text is one word repeated, keep just one."""
        text = "Yeah. " * 15
        result = trim_trailing_repetition(text.strip())
        assert result == "Yeah."

    def test_two_word_repeating_phrase(self):
        """Multi-word repeating unit: 'Thank you. Thank you. Thank you...'"""
        text = "That was great. " + "Thank you. " * 10
        result = trim_trailing_repetition(text.strip())
        assert result == "That was great. Thank you."

    def test_empty_string(self):
        assert trim_trailing_repetition("") == ""

    def test_single_word(self):
        assert trim_trailing_repetition("Hello") == "Hello"

    def test_preserves_cjk_trailing_repetition(self):
        """CJK degeneration: real Chinese prefix + repeated suffix."""
        text = "这是一个很好的观点。对。对。对。对。对。对。对。"
        result = trim_trailing_repetition(text)
        # Should keep prefix + one instance of repeated token
        assert "对。对。对。对。" not in result
        assert result.startswith("这是一个很好的观点。")


class TestReset:
    """Verify reset() clears repetition state."""

    def test_reset_clears_state(self):
        f = HallucinationFilter()
        r = _result(text="Good morning")
        f.should_suppress(r)
        f.should_suppress(r)
        # 2 in deque — next would be suppressed
        f.reset()
        # After reset, deque is empty — should pass
        suppressed, _ = f.should_suppress(r)
        assert suppressed is False
