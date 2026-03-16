"""Consolidated hallucination filter for Whisper transcription.

Gates (applied in order — cheapest first):
1. compression_ratio > 2.4 (OpenAI's default threshold)
2. BoH phrase matching (exact match for short phrases)
3. Two-tier confidence (≤2 words: 0.55, all: 0.3)
4. Intra-word repetition (>5 words, <20% unique)
5. Cross-segment repetition (deque of last 5, threshold 2)

Gate ordering matters: only text that passes ALL prior gates is appended to
the cross-segment repetition deque. This prevents hallucinated text from
polluting the repetition window.
"""
from __future__ import annotations

from collections import deque

from livetranslate_common.models.transcription import TranscriptionResult

_HALLUCINATION_PHRASES: frozenset[str] = frozenset({
    "thank you",
    "thank you.",
    "thank you for watching",
    "thank you for watching.",
    "thanks for watching",
    "thanks for watching.",
    "please subscribe",
    "please subscribe.",
    "please like and subscribe",
    "please like and subscribe.",
    "subtitles by",
    "amara.org",
    "www.movieweb.com",
    "mbc 뉴스",
    "김정진입니다",
    "thanks for watching our channel",
    "thanks for watching our channel.",
})


def _has_cjk(text: str) -> bool:
    """Return True if text contains CJK characters (Chinese/Japanese/Korean)."""
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF      # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF    # CJK Extension A
            or 0x3040 <= cp <= 0x309F    # Hiragana
            or 0x30A0 <= cp <= 0x30FF    # Katakana
            or 0xAC00 <= cp <= 0xD7AF    # Hangul Syllables
        ):
            return True
    return False


class HallucinationFilter:
    """Stateful filter consolidating all hallucination detection gates.

    Args:
        compression_ratio_threshold: Max compression ratio before suppression.
        short_confidence_threshold: Min confidence for ≤2-word segments.
        min_confidence: Min confidence for any segment.
        repetition_window: Size of cross-segment repetition deque.
        repetition_threshold: How many matches in the window trigger suppression.
    """

    def __init__(
        self,
        *,
        compression_ratio_threshold: float = 2.4,
        short_confidence_threshold: float = 0.55,
        min_confidence: float = 0.3,
        repetition_window: int = 5,
        repetition_threshold: int = 2,
    ) -> None:
        self._cr_threshold = compression_ratio_threshold
        self._short_conf_threshold = short_confidence_threshold
        self._min_conf = min_confidence
        self._rep_window = repetition_window
        self._rep_threshold = repetition_threshold
        self._recent_texts: deque[str] = deque(maxlen=repetition_window)

    def should_suppress(self, result: TranscriptionResult) -> tuple[bool, str]:
        """Return (should_suppress, reason_key) for logging.

        Only text that passes all gates is appended to the repetition deque.
        """
        # Gate 1: compression_ratio
        if result.compression_ratio is not None and result.compression_ratio > self._cr_threshold:
            return True, "compression_ratio"

        # Gate 2: Bag-of-Hallucinations phrase matching (exact, normalized)
        normalized = result.text.strip().lower()
        if normalized in _HALLUCINATION_PHRASES:
            return True, "boh_phrase"

        # Gate 3: Two-tier confidence
        # CJK languages don't use spaces, so .split() returns 1-2 tokens for
        # entire sentences. Use character count for CJK (1 char ≈ 1 word).
        word_count = len(result.text.split())
        if _has_cjk(result.text):
            word_count = max(word_count, len(result.text.strip()))
        if word_count <= 2 and result.confidence < self._short_conf_threshold:
            return True, "low_confidence_short"
        if result.confidence < self._min_conf:
            return True, "low_confidence"

        # Gate 4: Intra-word repetition (>5 words, <20% unique)
        # For CJK: use character-level bigrams instead of space-split words,
        # since CJK has no spaces and single characters repeat naturally.
        if _has_cjk(result.text):
            chars = list(result.text.strip())
            if len(chars) > 10:
                bigrams = [result.text[i:i+2] for i in range(len(chars) - 1)]
                unique_ratio = len(set(bigrams)) / len(bigrams)
                if unique_ratio < 0.2:
                    return True, "intra_word_repetition"
        elif word_count > 5:
            words = result.text.lower().split()
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.2:
                return True, "intra_word_repetition"

        # Gate 5: Cross-segment repetition (side-effecting — runs last)
        repeat_count = sum(1 for t in self._recent_texts if t == normalized)
        if repeat_count >= self._rep_threshold:
            return True, "cross_segment_repetition"
        # Only append text that passed ALL gates
        self._recent_texts.append(normalized)

        return False, ""

    def reset(self) -> None:
        """Clear repetition state (call on session start/end)."""
        self._recent_texts.clear()
