"""Authoritative language detection and code normalization.

Uses faster-whisper's built-in LID on first chunk for initial detection,
then SlidingLIDDetector for ongoing monitoring. Normalizes regional
variants (zh-CN, zh-TW, yue) to registry key space (zh, en, ja).
"""
from __future__ import annotations

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
