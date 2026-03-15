"""Tests for language-universal sentence segmenter."""
from sentence_segmenter import SentenceSegmenter


class TestLatinPunctuation:
    def test_period(self):
        seg = SentenceSegmenter()
        result = seg.segment("Hello world. How are you")
        assert result.sentences == ["Hello world."]
        assert result.remainder == "How are you"

    def test_exclamation(self):
        seg = SentenceSegmenter()
        result = seg.segment("Wow! Amazing")
        assert result.sentences == ["Wow!"]
        assert result.remainder == "Amazing"

    def test_question_mark(self):
        seg = SentenceSegmenter()
        result = seg.segment("What is this? I wonder")
        assert result.sentences == ["What is this?"]
        assert result.remainder == "I wonder"

    def test_no_punctuation(self):
        seg = SentenceSegmenter()
        result = seg.segment("Hello world how are you")
        assert result.sentences == []
        assert result.remainder == "Hello world how are you"


class TestCJKPunctuation:
    def test_chinese_period(self):
        seg = SentenceSegmenter()
        result = seg.segment("你好世界。今天")
        assert result.sentences == ["你好世界。"]
        assert result.remainder == "今天"

    def test_chinese_exclamation(self):
        seg = SentenceSegmenter()
        result = seg.segment("太好了！是的")
        assert result.sentences == ["太好了！"]
        assert result.remainder == "是的"

    def test_japanese_period(self):
        seg = SentenceSegmenter()
        result = seg.segment("こんにちは。元気ですか")
        assert result.sentences == ["こんにちは。"]
        assert result.remainder == "元気ですか"


class TestMultipleSentences:
    def test_multiple_latin(self):
        seg = SentenceSegmenter()
        result = seg.segment("First. Second. Third")
        assert result.sentences == ["First.", "Second."]
        assert result.remainder == "Third"

    def test_mixed_latin_cjk(self):
        seg = SentenceSegmenter()
        result = seg.segment("Hello. 你好。World")
        assert result.sentences == ["Hello.", "你好。"]
        assert result.remainder == "World"
