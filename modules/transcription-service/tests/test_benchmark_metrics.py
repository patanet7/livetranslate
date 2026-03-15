from benchmarks.metrics import (
    _parse_zh_number,
    character_error_rate,
    normalize_numbers,
    word_error_rate,
)


class TestWER:
    def test_identical(self):
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_one_substitution(self):
        wer = word_error_rate("hello world", "hello earth")
        assert wer == 0.5

    def test_empty_reference(self):
        assert word_error_rate("", "something") == 1.0

    def test_empty_both(self):
        assert word_error_rate("", "") == 0.0


class TestCER:
    def test_identical_chinese(self):
        assert character_error_rate("你好世界", "你好世界") == 0.0

    def test_one_char_error(self):
        cer = character_error_rate("你好世界", "你好时间")
        assert cer == 0.5


class TestParseZhNumber:
    """Regression tests for _parse_zh_number."""

    def test_zero(self):
        """零 must parse to 0."""
        assert _parse_zh_number("零") == 0

    def test_ten(self):
        """十 (unit alone, no leading digit) must parse to 10."""
        assert _parse_zh_number("十") == 10

    def test_single_digit_one(self):
        """一 must parse to 1."""
        assert _parse_zh_number("一") == 1

    def test_compound_two_thousand_five_hundred(self):
        """两千五百 must parse to 2500."""
        assert _parse_zh_number("两千五百") == 2500

    def test_large_number_with_wan(self):
        """两千五百万 must parse to 25_000_000."""
        assert _parse_zh_number("两千五百万") == 25_000_000

    def test_none_on_non_number_text(self):
        """Non-number text must return None."""
        assert _parse_zh_number("hello") is None

    def test_none_on_empty_string(self):
        """Empty string must return None."""
        assert _parse_zh_number("") is None


class TestNormalizeNumbers:
    """Tests for normalize_numbers text normalisation."""

    def test_zero_normalised_to_digit(self):
        """'零' in text must be converted to '0'."""
        assert normalize_numbers("零") == "0"

    def test_ten_normalised_to_digit(self):
        """'十' in text must be converted to '10'."""
        assert normalize_numbers("十") == "10"

    def test_compound_number_in_sentence(self):
        """Chinese number embedded in text must be normalised."""
        result = normalize_numbers("共两千五百人参加了会议")
        assert "2500" in result
        assert "两千五百" not in result

    def test_non_number_text_unchanged(self):
        """Text without Chinese numbers must pass through unchanged."""
        text = "今天天气很好"
        assert normalize_numbers(text) == text

    def test_pure_latin_digits_unchanged(self):
        """Text containing only ASCII digits and no CJK number characters must not be altered."""
        assert normalize_numbers("2500") == "2500"
        assert normalize_numbers("The year 2024 was notable") == "The year 2024 was notable"
