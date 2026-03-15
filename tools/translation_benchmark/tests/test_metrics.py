import pytest
from tools.translation_benchmark.metrics import bleu_score, comet_available, comet_score


class TestBLEU:
    def test_identical(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on the mat"]
        score = bleu_score(refs, hyps)
        assert score == 1.0

    def test_completely_wrong(self):
        refs = ["the cat sat on the mat"]
        hyps = ["foo bar baz qux quux"]
        score = bleu_score(refs, hyps)
        assert score == 0.0

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
