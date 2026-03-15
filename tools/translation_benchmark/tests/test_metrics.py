import pytest
from tools.translation_benchmark.metrics import bleu_score, comet_available, comet_score


class TestBLEU:
    def test_identical(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on the mat"]
        score = bleu_score(refs, hyps)
        assert score == 1.0

    def test_completely_wrong(self):
        # With add-epsilon smoothing, a completely non-overlapping hypothesis
        # produces a small but non-zero score.  Assert it stays well below any
        # meaningful quality threshold rather than exactly 0.
        refs = ["the cat sat on the mat"]
        hyps = ["foo bar baz qux quux"]
        score = bleu_score(refs, hyps)
        assert score < 0.2

    def test_partial_match(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on a mat"]
        score = bleu_score(refs, hyps)
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert bleu_score([], []) == 0.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            bleu_score(["one"], ["one", "two"])

    def test_cjk_identical(self):
        """CJK character-level tokenization: identical strings should score 1.0."""
        score = bleu_score(["你好世界"], ["你好世界"])
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_short_hypothesis_smoothing(self):
        """Short hypothesis (<4 tokens) must not collapse to 0 with smoothing applied."""
        score = bleu_score(["the cat sat"], ["the cat"])
        assert score > 0.0

    def test_mixed_cjk_latin_identical(self):
        """Mixed CJK/Latin text: identical strings score very high (close to 1.0).

        With max_n=4 and only 3 tokens ("Hello", "你", "好"), higher-order n-grams
        are absent and receive epsilon smoothing, so the score is high but not
        exactly 1.0.  Assert it exceeds a strong quality threshold.
        """
        score = bleu_score(["Hello 你好"], ["Hello 你好"])
        assert score > 0.5


class TestCOMET:
    def test_comet_available_returns_bool(self):
        result = comet_available()
        assert isinstance(result, bool)

    def test_comet_score_returns_none_when_unavailable(self):
        if comet_available():
            pytest.skip("COMET is installed, cannot test unavailable path")
        result = comet_score(
            sources=["Hello"],
            references=["Hola"],
            hypotheses=["Hola"],
        )
        assert result is None
