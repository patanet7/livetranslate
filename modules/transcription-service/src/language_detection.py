"""Authoritative language detection and code normalization.

Uses faster-whisper's built-in LID on first chunk for initial detection,
then SlidingLIDDetector for ongoing monitoring. Normalizes regional
variants (zh-CN, zh-TW, yue) to registry key space (zh, en, ja).

Two detector implementations:
- LanguageDetector: Legacy threshold-based (switches after 3s of any language)
- WhisperLanguageDetector: Production adapter wrapping SustainedLanguageDetector
  with hysteresis (confidence margin + frame count + dwell time)
"""
from __future__ import annotations

from language_id.sustained_detector import SustainedLanguageDetector
from livetranslate_common.logging import get_logger

logger = get_logger()

# Regional variant → base language code
_NORMALIZATION_MAP: dict[str, str] = {
    "yue": "zh",
    "cmn": "zh",
    "wuu": "zh",
    "nan": "zh",
}


def normalize_language_code(code: str) -> str:
    """Normalize a language code to the registry key space.

    'zh-CN' → 'zh', 'en-US' → 'en', 'yue' → 'zh', etc.
    """
    code = code.lower().strip()

    # Check special mappings first
    if code in _NORMALIZATION_MAP:
        return _NORMALIZATION_MAP[code]

    # Strip regional suffix: 'zh-CN' → 'zh'
    base = code.split("-")[0]

    # Check again after stripping
    if base in _NORMALIZATION_MAP:
        return _NORMALIZATION_MAP[base]

    return base


class LanguageDetector:
    """Authoritative language detector for registry routing.

    1. First chunk: faster-whisper LID (high confidence)
    2. Ongoing: SlidingLIDDetector monitors for sustained language switches
    3. Language codes normalized before registry lookup
    """

    def __init__(self, switch_threshold_s: float = 3.0):
        self._current_language: str | None = None
        self._switch_threshold_s = switch_threshold_s
        self._sustained_count = 0
        self._candidate_language: str | None = None

    @property
    def current_language(self) -> str | None:
        return self._current_language

    def detect_initial(self, language: str, confidence: float) -> str:
        """Set initial language from faster-whisper's first-chunk LID."""
        normalized = normalize_language_code(language)
        self._current_language = normalized
        logger.info("language_detected_initial", language=normalized, confidence=confidence)
        return normalized

    def update(self, detected_language: str, chunk_duration_s: float) -> str | None:
        """Update with ongoing detection. Returns new language if switch detected."""
        normalized = normalize_language_code(detected_language)

        if normalized == self._current_language:
            self._candidate_language = None
            self._sustained_count = 0
            return None

        if normalized == self._candidate_language:
            self._sustained_count += chunk_duration_s
        else:
            self._candidate_language = normalized
            self._sustained_count = chunk_duration_s

        if self._sustained_count > self._switch_threshold_s:
            old = self._current_language
            self._current_language = normalized
            self._candidate_language = None
            self._sustained_count = 0
            logger.info("language_switched", old=old, new=normalized)
            return normalized

        return None


class WhisperLanguageDetector:
    """Adapter: converts Whisper's (lang, confidence) to SustainedLanguageDetector's lid_probs API.

    In interpreter mode, the detector drives translation direction (critical path).
    In split/transcript mode, the detector is informational only (source is user-set).

    Default parameters are tuned to prevent the flapping observed in production:
    - confidence_margin=0.2: new language must beat current by 20%
    - min_dwell_frames=4: at least 4 consecutive detections
    - min_dwell_ms=10000: at least 10s of sustained detection
    """

    def __init__(
        self,
        confidence_margin: float = 0.2,
        min_dwell_frames: int = 4,
        min_dwell_ms: float = 10000,
    ):
        self._inner = SustainedLanguageDetector(
            confidence_margin=confidence_margin,
            min_dwell_frames=min_dwell_frames,
            min_dwell_ms=min_dwell_ms,
        )
        self._cumulative_time = 0.0

    @property
    def current_language(self) -> str | None:
        return self._inner.current_language

    def detect_initial(self, language: str, confidence: float) -> str:
        """Set initial language from faster-whisper's first-chunk LID."""
        normalized = normalize_language_code(language)
        self._inner.force_language(normalized)
        logger.info("language_detected_initial", language=normalized, confidence=confidence)
        return normalized

    def update(self, detected_language: str, chunk_duration_s: float, confidence: float = 0.5) -> str | None:
        """Update with ongoing detection. Returns new language if switch detected.

        Converts Whisper's single (language, confidence) into a probability
        distribution for the SustainedLanguageDetector.
        """
        normalized = normalize_language_code(detected_language)
        self._cumulative_time += chunk_duration_s
        current = self._inner.current_language or "en"

        # Build probability distribution: detected language vs current
        lid_probs = {normalized: confidence, current: 1.0 - confidence}

        event = self._inner.update(lid_probs, self._cumulative_time)
        if event:
            logger.info(
                "language_switched",
                old=event.from_language,
                new=event.to_language,
                margin=event.confidence_margin,
                dwell_ms=event.dwell_duration_ms,
            )
            return event.to_language
        return None
