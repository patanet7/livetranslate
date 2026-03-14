from benchmarks.metrics import word_error_rate, character_error_rate


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
