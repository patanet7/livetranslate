"""Tests for authoritative language detection and normalization.

Covers:
- Language code normalization (regional variants, CJK dialects)
- Legacy LanguageDetector (basic threshold-based)
- WhisperLanguageDetector (SustainedLanguageDetector adapter with hysteresis)
  - Real flapping sequences from production logs
  - Hallucinated language rejection
  - Sustained real switches
"""
import pytest
from language_detection import (
    LanguageDetector,
    WhisperLanguageDetector,
    normalize_language_code,
)


class TestLanguageCodeNormalization:
    def test_simple_codes(self):
        assert normalize_language_code("en") == "en"
        assert normalize_language_code("zh") == "zh"

    def test_regional_variants(self):
        assert normalize_language_code("zh-CN") == "zh"
        assert normalize_language_code("zh-TW") == "zh"
        assert normalize_language_code("en-US") == "en"
        assert normalize_language_code("pt-BR") == "pt"

    def test_cantonese_and_mandarin_variants(self):
        assert normalize_language_code("yue") == "zh"
        assert normalize_language_code("cmn") == "zh"

    def test_case_insensitive(self):
        assert normalize_language_code("EN") == "en"
        assert normalize_language_code("ZH-cn") == "zh"


class TestLanguageDetector:
    """Legacy detector — kept for backwards compatibility."""

    def test_initial_detection(self):
        detector = LanguageDetector()
        lang = detector.detect_initial("en", 0.95)
        assert lang == "en"
        assert detector.current_language == "en"

    def test_no_switch_on_same_language(self):
        detector = LanguageDetector()
        detector.detect_initial("en", 0.95)
        result = detector.update("en", chunk_duration_s=1.0)
        assert result is None

    def test_switch_after_sustained_detection(self):
        detector = LanguageDetector(switch_threshold_s=3.0)
        detector.detect_initial("en", 0.95)
        # 3 chunks of Chinese, 1.5s each = 4.5s > 3.0s threshold
        assert detector.update("zh-CN", 1.5) is None
        assert detector.update("zh-CN", 1.5) is None  # 3.0s = threshold
        result = detector.update("zh-CN", 1.5)  # 4.5s > threshold
        assert result == "zh"
        assert detector.current_language == "zh"

    def test_switch_resets_on_different_candidate(self):
        detector = LanguageDetector(switch_threshold_s=3.0)
        detector.detect_initial("en", 0.95)
        detector.update("zh", 2.0)  # 2s of zh
        detector.update("ja", 1.0)  # switch to ja — resets zh counter
        result = detector.update("ja", 1.0)  # only 2s of ja
        assert result is None


class TestWhisperLanguageDetector:
    """Tests for the SustainedLanguageDetector adapter.

    Uses realistic parameters and patterns from production logs.
    """

    def test_detect_initial_sets_language(self):
        """First Whisper chunk sets the initial language."""
        detector = WhisperLanguageDetector()
        result = detector.detect_initial("en", 0.92)
        assert result == "en"
        assert detector.current_language == "en"

    def test_detect_initial_normalizes(self):
        """Initial detection normalizes language codes."""
        detector = WhisperLanguageDetector()
        result = detector.detect_initial("zh-CN", 0.85)
        assert result == "zh"
        assert detector.current_language == "zh"

    def test_no_switch_on_same_language(self):
        """Same language detections don't trigger a switch."""
        detector = WhisperLanguageDetector()
        detector.detect_initial("en", 0.92)
        for _ in range(20):
            result = detector.update("en", chunk_duration_s=3.0, confidence=0.8)
            assert result is None

    def test_real_flapping_sequence_no_switch(self):
        """Replay actual flapping from 2026-03-17 transcription log.

        Production saw 80+ switches in 40min with this pattern.
        The sustained detector must reject ALL of these.

        Real sequence: en→ru(3s)→en(3s)→zh(3s)→nn(3s)→en(3s)→nl(3s)→en(3s)
                      →pt(3s)→en(3s)→zh(3s)→en(3s)→ko(3s)→zh(3s)→en(3s)
        """
        detector = WhisperLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=4, min_dwell_ms=10000
        )
        detector.detect_initial("en", 0.85)

        # The real flapping sequence — each language lasts ~3s with moderate confidence
        flapping = [
            ("ru", 3.0, 0.4),
            ("en", 3.0, 0.7),
            ("zh", 3.0, 0.45),
            ("nn", 3.0, 0.3),
            ("en", 3.0, 0.65),
            ("nl", 3.0, 0.35),
            ("en", 3.0, 0.7),
            ("pt", 3.0, 0.4),
            ("en", 3.0, 0.75),
            ("zh", 3.0, 0.5),
            ("en", 3.0, 0.7),
            ("ko", 3.0, 0.35),
            ("zh", 3.0, 0.4),
            ("en", 3.0, 0.7),
        ]

        for lang, duration, confidence in flapping:
            result = detector.update(lang, duration, confidence)
            assert result is None, (
                f"Flapping: detector should NOT have switched to {lang} "
                f"(confidence={confidence})"
            )
        assert detector.current_language == "en"

    def test_hallucinated_languages_rejected(self):
        """Whisper hallucinates nn, cy, pl, ko, it, fr with ~0.3-0.5 confidence.

        Interleaved with en at ~0.7 confidence. None should trigger a switch.
        """
        detector = WhisperLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=4, min_dwell_ms=10000
        )
        detector.detect_initial("en", 0.9)

        hallucinations = [
            ("nn", 0.3), ("cy", 0.35), ("pl", 0.4), ("ko", 0.32),
            ("it", 0.45), ("fr", 0.38), ("ru", 0.42), ("pt", 0.3),
        ]
        for lang, confidence in hallucinations:
            # Each hallucinated detection for 3s
            result = detector.update(lang, chunk_duration_s=3.0, confidence=confidence)
            assert result is None
            # Followed by en re-detection
            result = detector.update("en", chunk_duration_s=3.0, confidence=0.7)
            assert result is None

        assert detector.current_language == "en"

    def test_real_interpreter_switch_en_to_zh(self):
        """Simulate real meeting: English for 30s, then sustained Chinese for 15s.

        This represents a real interpreter scenario where the speaker
        switches from English to Chinese. The detector SHOULD switch.
        """
        detector = WhisperLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=4, min_dwell_ms=10000
        )
        detector.detect_initial("en", 0.9)

        # English for 30s (10 chunks of 3s)
        for _ in range(10):
            result = detector.update("en", chunk_duration_s=3.0, confidence=0.8)
            assert result is None

        # Chinese with high confidence for 15s (5 chunks of 3s)
        switched = False
        for _ in range(5):
            result = detector.update("zh", chunk_duration_s=3.0, confidence=0.85)
            if result == "zh":
                switched = True

        assert switched, "Detector should have switched to zh after 15s of sustained zh"
        assert detector.current_language == "zh"

    def test_brief_chinese_does_not_switch(self):
        """Short bursts of Chinese (< min_dwell_ms) should not switch."""
        detector = WhisperLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=4, min_dwell_ms=10000
        )
        detector.detect_initial("en", 0.9)

        # Brief Chinese — 3 chunks of 3s = 9s < 10s min_dwell
        for _ in range(3):
            result = detector.update("zh", chunk_duration_s=3.0, confidence=0.8)
            assert result is None

        assert detector.current_language == "en"

    def test_low_confidence_does_not_switch(self):
        """Even sustained detection with low confidence shouldn't switch.

        Confidence margin test: if zh confidence is 0.55 and en is 0.45,
        margin = 0.1 < 0.2 threshold.
        """
        detector = WhisperLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=4, min_dwell_ms=10000
        )
        detector.detect_initial("en", 0.9)

        # Long Chinese but barely above 50% — low margin
        for _ in range(10):
            result = detector.update("zh", chunk_duration_s=3.0, confidence=0.55)
            assert result is None

        assert detector.current_language == "en"

    def test_update_returns_none_before_initial(self):
        """Calling update before detect_initial should work gracefully."""
        detector = WhisperLanguageDetector()
        # The inner SustainedLanguageDetector will set language on first update
        result = detector.update("en", chunk_duration_s=3.0, confidence=0.8)
        assert result is None

    def test_normalize_in_update(self):
        """update() normalizes language codes before passing to inner detector."""
        detector = WhisperLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=4, min_dwell_ms=10000
        )
        detector.detect_initial("en", 0.9)

        # Use regional variant zh-CN — should be normalized to zh
        switched = False
        for _ in range(8):
            result = detector.update("zh-CN", chunk_duration_s=3.0, confidence=0.85)
            if result:
                switched = True
                assert result == "zh"

        assert switched
