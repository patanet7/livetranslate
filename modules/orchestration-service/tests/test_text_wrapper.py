"""Tests for multi-script text wrapping (Latin word-break, CJK char-break, mixed)."""

import pytest
from bot.text_wrapper import wrap_text


class TestWrapText:
    def test_short_text_no_wrap(self):
        result = wrap_text("Hello world", max_chars=40)
        assert result == ["Hello world"]

    def test_english_word_wrap(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = wrap_text(text, max_chars=20)
        assert len(result) >= 2
        for line in result:
            assert len(line) <= 20

    def test_english_preserves_words(self):
        text = "Hello wonderful world"
        result = wrap_text(text, max_chars=12)
        assert all(" " not in line.strip() or len(line) <= 12 for line in result)

    def test_cjk_char_wrap(self):
        text = "这是一个很长的中文句子需要换行显示"
        result = wrap_text(text, max_chars=8)
        assert len(result) >= 2
        for line in result:
            assert len(line) <= 8

    def test_mixed_script_wrap(self):
        text = "Hello 你好世界 World 再见"
        result = wrap_text(text, max_chars=10)
        assert len(result) >= 2

    def test_empty_string(self):
        result = wrap_text("", max_chars=40)
        assert result == [""]

    def test_max_lines_truncation(self):
        text = "Line one. Line two. Line three. Line four. Line five."
        result = wrap_text(text, max_chars=12, max_lines=3)
        assert len(result) <= 3

    def test_single_long_word_forced_break(self):
        text = "Supercalifragilisticexpialidocious"
        result = wrap_text(text, max_chars=10)
        assert len(result) >= 3
        for line in result:
            assert len(line) <= 10

    def test_japanese_wrap(self):
        text = "これはテストの文章です"
        result = wrap_text(text, max_chars=5)
        assert len(result) >= 2
        for line in result:
            assert len(line) <= 5

    def test_korean_wrap(self):
        text = "안녕하세요반갑습니다"
        result = wrap_text(text, max_chars=5)
        assert len(result) >= 2
        for line in result:
            assert len(line) <= 5
