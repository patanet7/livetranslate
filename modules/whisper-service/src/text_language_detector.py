#!/usr/bin/env python3
"""
Text-based Language Detector

Detects language from actual transcribed text using character analysis.
This is more reliable than audio-based detection when dealing with mixed-language chunks.

Strategy:
1. Audio-based detection (Whisper's lang_id): Analyzes audio features → language of AUDIO
2. Text-based detection (this module): Analyzes characters → language of TEXT
3. For code-switching: Use TEXT language for labeling, audio language for SOT reset logic
"""

import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TextLanguageDetector:
    """Detect language from transcribed text using character analysis"""

    # Character ranges for different scripts
    CJK_RANGES = [
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0x3400, 0x4DBF),    # CJK Extension A
        (0x20000, 0x2A6DF),  # CJK Extension B
        (0x2A700, 0x2B73F),  # CJK Extension C
        (0x2B740, 0x2B81F),  # CJK Extension D
        (0x2B820, 0x2CEAF),  # CJK Extension E
        (0x2CEB0, 0x2EBEF),  # CJK Extension F
        (0x3000, 0x303F),    # CJK Symbols and Punctuation
        (0xFF00, 0xFFEF),    # Halfwidth and Fullwidth Forms
    ]

    HIRAGANA_KATAKANA_RANGES = [
        (0x3040, 0x309F),    # Hiragana
        (0x30A0, 0x30FF),    # Katakana
    ]

    HANGUL_RANGES = [
        (0xAC00, 0xD7AF),    # Hangul Syllables
        (0x1100, 0x11FF),    # Hangul Jamo
        (0x3130, 0x318F),    # Hangul Compatibility Jamo
    ]

    ARABIC_RANGES = [
        (0x0600, 0x06FF),    # Arabic
        (0x0750, 0x077F),    # Arabic Supplement
        (0x08A0, 0x08FF),    # Arabic Extended-A
    ]

    CYRILLIC_RANGES = [
        (0x0400, 0x04FF),    # Cyrillic
        (0x0500, 0x052F),    # Cyrillic Supplement
    ]

    def __init__(self):
        """Initialize text language detector"""
        pass

    def _count_chars_in_ranges(self, text: str, ranges: list) -> int:
        """Count characters that fall within specified Unicode ranges"""
        count = 0
        for char in text:
            code_point = ord(char)
            for start, end in ranges:
                if start <= code_point <= end:
                    count += 1
                    break
        return count

    def _count_latin_chars(self, text: str) -> int:
        """Count Latin alphabet characters (A-Z, a-z)"""
        return len(re.findall(r'[A-Za-z]', text))

    def detect(self, text: str, audio_detected_language: Optional[str] = None) -> str:
        """
        Detect language from transcribed text

        Args:
            text: Transcribed text to analyze
            audio_detected_language: Language detected from audio (fallback)

        Returns:
            Language code (en, zh, ja, ko, ar, ru, etc.)
        """
        if not text or not text.strip():
            return audio_detected_language or 'en'

        # Remove whitespace for character counting
        text_no_space = text.replace(' ', '').replace('\n', '')

        if not text_no_space:
            return audio_detected_language or 'en'

        # Count characters in different scripts
        cjk_count = self._count_chars_in_ranges(text_no_space, self.CJK_RANGES)
        hiragana_katakana_count = self._count_chars_in_ranges(text_no_space, self.HIRAGANA_KATAKANA_RANGES)
        hangul_count = self._count_chars_in_ranges(text_no_space, self.HANGUL_RANGES)
        arabic_count = self._count_chars_in_ranges(text_no_space, self.ARABIC_RANGES)
        cyrillic_count = self._count_chars_in_ranges(text_no_space, self.CYRILLIC_RANGES)
        latin_count = self._count_latin_chars(text_no_space)

        total_chars = len(text_no_space)

        # Determine language based on character counts
        # Priority: Specific scripts > Latin

        # Chinese/Japanese detection (CJK + Hiragana/Katakana)
        if cjk_count > 0 or hiragana_katakana_count > 0:
            cjk_ratio = cjk_count / total_chars if total_chars > 0 else 0
            jp_ratio = hiragana_katakana_count / total_chars if total_chars > 0 else 0

            # Japanese has Hiragana/Katakana mixed with CJK
            if jp_ratio > 0.1 or (hiragana_katakana_count > 0 and cjk_ratio < 0.5):
                logger.debug(f"[TEXT_LID] Detected Japanese: CJK={cjk_count}, HK={hiragana_katakana_count}, ratio={jp_ratio:.2f}")
                return 'ja'

            # Pure Chinese (mostly CJK, no Hiragana/Katakana)
            if cjk_ratio > 0.2:
                logger.debug(f"[TEXT_LID] Detected Chinese: CJK={cjk_count}, ratio={cjk_ratio:.2f}")
                return 'zh'

        # Korean detection
        if hangul_count > 0:
            hangul_ratio = hangul_count / total_chars if total_chars > 0 else 0
            if hangul_ratio > 0.2:
                logger.debug(f"[TEXT_LID] Detected Korean: Hangul={hangul_count}, ratio={hangul_ratio:.2f}")
                return 'ko'

        # Arabic detection
        if arabic_count > 0:
            arabic_ratio = arabic_count / total_chars if total_chars > 0 else 0
            if arabic_ratio > 0.2:
                logger.debug(f"[TEXT_LID] Detected Arabic: Arabic={arabic_count}, ratio={arabic_ratio:.2f}")
                return 'ar'

        # Cyrillic detection (Russian, Ukrainian, etc.)
        if cyrillic_count > 0:
            cyrillic_ratio = cyrillic_count / total_chars if total_chars > 0 else 0
            if cyrillic_ratio > 0.2:
                logger.debug(f"[TEXT_LID] Detected Cyrillic (Russian): Cyrillic={cyrillic_count}, ratio={cyrillic_ratio:.2f}")
                return 'ru'

        # Latin-based languages (English, Spanish, French, German, etc.)
        # If mostly Latin characters, default to audio detection or English
        if latin_count > 0:
            latin_ratio = latin_count / total_chars if total_chars > 0 else 0
            if latin_ratio > 0.5:
                # Can't distinguish between Latin-based languages without more sophisticated analysis
                # Use audio detection as hint if available
                if audio_detected_language in ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'tr']:
                    logger.debug(f"[TEXT_LID] Detected Latin script, using audio hint: {audio_detected_language}")
                    return audio_detected_language
                else:
                    logger.debug(f"[TEXT_LID] Detected Latin script, defaulting to English: Latin={latin_count}, ratio={latin_ratio:.2f}")
                    return 'en'

        # Fallback: Use audio detection or default to English
        if audio_detected_language:
            logger.debug(f"[TEXT_LID] No clear script detected, using audio detection: {audio_detected_language}")
            return audio_detected_language

        logger.debug(f"[TEXT_LID] No clear script detected, defaulting to English")
        return 'en'

    def get_language_confidence(self, text: str, language: str) -> float:
        """
        Get confidence score for detected language (0.0 - 1.0)

        Args:
            text: Transcribed text
            language: Detected language code

        Returns:
            Confidence score (0.0 = low, 1.0 = high)
        """
        if not text or not text.strip():
            return 0.0

        text_no_space = text.replace(' ', '').replace('\n', '')
        total_chars = len(text_no_space)

        if total_chars == 0:
            return 0.0

        # Calculate confidence based on script character ratio
        if language == 'zh':
            cjk_count = self._count_chars_in_ranges(text_no_space, self.CJK_RANGES)
            return min(1.0, cjk_count / total_chars)

        elif language == 'ja':
            jp_count = self._count_chars_in_ranges(text_no_space, self.HIRAGANA_KATAKANA_RANGES)
            cjk_count = self._count_chars_in_ranges(text_no_space, self.CJK_RANGES)
            return min(1.0, (jp_count + cjk_count * 0.5) / total_chars)

        elif language == 'ko':
            hangul_count = self._count_chars_in_ranges(text_no_space, self.HANGUL_RANGES)
            return min(1.0, hangul_count / total_chars)

        elif language == 'ar':
            arabic_count = self._count_chars_in_ranges(text_no_space, self.ARABIC_RANGES)
            return min(1.0, arabic_count / total_chars)

        elif language == 'ru':
            cyrillic_count = self._count_chars_in_ranges(text_no_space, self.CYRILLIC_RANGES)
            return min(1.0, cyrillic_count / total_chars)

        else:  # Latin-based languages (en, es, fr, de, etc.)
            latin_count = self._count_latin_chars(text_no_space)
            return min(1.0, latin_count / total_chars)
