"""Tests for authoritative language detection and normalization."""
from language_detection import normalize_language_code, LanguageDetector


class TestLanguageCodeNormalization:
    def test_simple_codes(self):
        assert normalize_language_code("en") == "en"
        assert normalize_language_code("zh") == "zh"

    def test_regional_variants(self):
        assert normalize_language_code("zh-CN") == "zh"
        assert normalize_language_code("zh-TW") == "zh"
        assert normalize_language_code("en-US") == "en"
        assert normalize_language_code("pt-BR") == "pt"

    def test_cantonese(self):
        assert normalize_language_code("yue") == "zh"

    def test_case_insensitive(self):
        assert normalize_language_code("EN") == "en"
        assert normalize_language_code("ZH-cn") == "zh"


class TestLanguageDetector:
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
