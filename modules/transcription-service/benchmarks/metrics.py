"""Metrics computation for transcription benchmarks.

WER for alphabetic languages, CER for CJK.
"""
from __future__ import annotations


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (Levenshtein on words)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    d = _levenshtein(ref_words, hyp_words)
    return d / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (for CJK languages)."""
    ref_chars = list(reference.strip().replace(" ", ""))
    hyp_chars = list(hypothesis.strip().replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    d = _levenshtein(ref_chars, hyp_chars)
    return d / len(ref_chars)


def _levenshtein(ref: list, hyp: list) -> int:
    """Dynamic programming Levenshtein distance."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]
