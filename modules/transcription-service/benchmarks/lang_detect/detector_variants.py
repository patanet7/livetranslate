"""Wrappers around WhisperLanguageDetector that enable proposed fixes.

These exist so the sweep can test 'production-as-is' vs 'production + fix(es)'
without modifying the production code path. If the data shows wins, the
corresponding logic gets promoted into language_detection.py in a separate PR.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure transcription-service/src is importable when running from anywhere.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from language_detection import WhisperLanguageDetector  # noqa: E402

from .types import DetectorParams, FrameTrace


# Script ranges that strongly imply a specific language.
# (Unicode ranges chosen to match the script_tiebreaker design — Han for zh/ja,
# Hangul for ko, Arabic, Cyrillic.)
_SCRIPT_TO_LANG = [
    ("zh", (0x4E00, 0x9FFF)),   # CJK Unified Ideographs
    ("zh", (0x3400, 0x4DBF)),   # CJK Extension A
    ("ja", (0x3040, 0x309F)),   # Hiragana
    ("ja", (0x30A0, 0x30FF)),   # Katakana
    ("ko", (0xAC00, 0xD7AF)),   # Hangul Syllables
    ("ar", (0x0600, 0x06FF)),   # Arabic
    ("ru", (0x0400, 0x04FF)),   # Cyrillic
]


def script_implied_language(text: str, min_ratio: float) -> str | None:
    """Return the language whose script dominates ``text``, if any clears ``min_ratio``.

    The ratio is over non-whitespace characters. ASCII-only text returns None
    (no opinion). For zh/ja ambiguity, Hiragana/Katakana presence wins for ja;
    otherwise Han characters imply zh.
    """
    if not text:
        return None

    counts: dict[str, int] = {}
    non_ws = 0
    for ch in text:
        if ch.isspace():
            continue
        non_ws += 1
        cp = ord(ch)
        for lang, (lo, hi) in _SCRIPT_TO_LANG:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break

    if non_ws == 0:
        return None

    # ja wins over zh if any kana present.
    if counts.get("ja", 0) > 0 and (counts["ja"] / non_ws) >= min_ratio:
        return "ja"
    # otherwise pick the dominant script.
    best_lang, best_count = None, 0
    for lang, c in counts.items():
        if lang == "ja":
            continue
        if c > best_count:
            best_lang, best_count = lang, c

    if best_lang and (best_count / non_ws) >= min_ratio:
        return best_lang
    return None


class TunableDetector:
    """WhisperLanguageDetector + opt-in proposed fixes, driven by DetectorParams.

    Wraps the production detector so we test exactly what runs in prod, with
    bypasses/overrides applied at the same call sites as api.py:_run_inference.
    """

    def __init__(self, params: DetectorParams):
        self.params = params
        self._inner = WhisperLanguageDetector(
            confidence_margin=params.confidence_margin,
            min_dwell_frames=params.min_dwell_frames,
            min_dwell_ms=params.min_dwell_ms,
        )

    @property
    def current_language(self) -> str | None:
        return self._inner.current_language

    def ingest(self, frame: FrameTrace) -> tuple[str | None, bool]:
        """Feed one frame; return (current_language_after, switched_this_frame).

        Mirrors the branch structure of api.py:400-423:
          - If current_language is None: detect_initial (with optional gate)
          - Else: update (with optional script tiebreaker)
        """
        # Apply script tiebreaker to the effective (lang, conf) the detector sees.
        eff_lang, eff_conf = self._tiebreak(frame)

        if self._inner.current_language is None:
            # Stage 1 gate: skip detect_initial if confidence too low.
            if eff_conf < self.params.initial_confidence_threshold:
                return None, False
            self._inner.detect_initial(eff_lang, eff_conf)
            return self._inner.current_language, True

        switched = self._inner.update(eff_lang, frame.chunk_dur_s, eff_conf)
        return self._inner.current_language, switched is not None

    def _tiebreak(self, frame: FrameTrace) -> tuple[str, float]:
        """If script tiebreaker is on and applicable, override the LID hint."""
        if not self.params.script_tiebreaker_enabled:
            return frame.language, frame.confidence
        if frame.confidence > self.params.script_tiebreaker_max_confidence:
            return frame.language, frame.confidence
        implied = script_implied_language(frame.text, self.params.script_tiebreaker_min_ratio)
        if implied and implied != frame.language:
            # Boost confidence on the script-implied language; the dwell guards
            # still apply, so this can't cause instant flapping.
            return implied, max(frame.confidence, 0.8)
        return frame.language, frame.confidence
