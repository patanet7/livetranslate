"""Metrics computation for transcription and translation benchmarks.

WER for alphabetic languages, CER for CJK.
BLEU for translation quality (corpus-level n-gram precision).
Number normalization handles mixed written/digit representations
(e.g. "两千五百万" == "2500万") before scoring.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

CJK_LANGUAGES = {"zh", "ja", "ko"}

# Chinese number word → value mapping (simplified + traditional subset)
_ZH_DIGIT = {
    "零": 0, "〇": 0,
    "一": 1, "壹": 1,
    "二": 2, "两": 2, "贰": 2, "兩": 2, "貳": 2,
    "三": 3, "叁": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陆": 6, "陸": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
}
_ZH_UNIT = {
    "十": 10, "拾": 10,
    "百": 100, "佰": 100,
    "千": 1000, "仟": 1000,
    "万": 10_000, "萬": 10_000,
    "亿": 100_000_000, "億": 100_000_000,
}


def _parse_zh_number(text: str) -> int | None:
    """Parse a Chinese number string into an integer.

    Handles patterns like 两千五百万 → 25_000_000.
    Returns None if ``text`` cannot be fully parsed as a number.
    """
    if not text:
        return None

    result = 0
    unit_stack: list[int] = []
    current = 0

    for ch in text:
        if ch in _ZH_DIGIT:
            current = _ZH_DIGIT[ch]
        elif ch in _ZH_UNIT:
            unit = _ZH_UNIT[ch]
            if unit >= 10_000:
                # 万 / 亿 flush current + stack into a section value
                section = current
                for u in unit_stack:
                    section += u
                result += (section if section else 1) * unit
                unit_stack = []
                current = 0
            else:
                # 十/百/千 multiply current digit
                unit_stack.append((current if current else 1) * unit)
                current = 0
        else:
            return None  # unexpected character

    # Flush remaining
    for u in unit_stack:
        result += u
    result += current
    return result if (result > 0 or current == 0 and any(ch in _ZH_DIGIT for ch in text)) else None


_ZH_NUMBER_PATTERN = re.compile(
    r"[零〇一壹二两贰兩貳三叁四肆五伍六陆陸七柒八捌九玖十拾百佰千仟万萬亿億]+"
)
_DIGIT_WITH_UNIT = re.compile(r"(\d[\d,]*)\s*([万萬亿億千百十]?)")


def normalize_numbers(text: str) -> str:
    """Normalize number representations to a canonical form.

    Converts Chinese number words to Arabic digits so that
    "两千五百万" and "2500万" both become "2500万" (or a pure integer),
    preventing false CER hits due to representation differences.

    The canonical form is: Arabic digits followed by any large-unit suffix.
    """
    def _replace_zh(m: re.Match) -> str:
        span = m.group(0)
        # Split off any trailing unit suffix that should remain
        for suffix in ("亿", "億", "万", "萬"):
            if span.endswith(suffix) and len(span) > 1:
                head = span[:-1]
                val = _parse_zh_number(head)
                if val is not None:
                    return str(val) + suffix
        val = _parse_zh_number(span)
        if val is not None:
            return str(val)
        return span

    return _ZH_NUMBER_PATTERN.sub(_replace_zh, text)


def normalize_text(text: str, language: str = "zh") -> str:
    """Full text normalization pipeline before metric computation.

    Steps applied in order:
    1. Unicode NFKC normalization (full-width → half-width, etc.)
    2. Number normalization (Chinese words → Arabic digits)
    3. Collapse whitespace
    4. Strip punctuation from CJK text (punctuation is not scored)
    """
    text = unicodedata.normalize("NFKC", text)
    text = normalize_numbers(text)

    if language in CJK_LANGUAGES:
        # Remove punctuation; CER only counts character-level content errors
        text = re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    else:
        # Lowercase + collapse whitespace for Latin scripts
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Core edit-distance
# ---------------------------------------------------------------------------

def _levenshtein(ref: list, hyp: list) -> int:
    """Space-optimised O(min(n,m)) Levenshtein distance."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------

def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (Levenshtein on whitespace-split words)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (Levenshtein on characters, spaces removed).

    Spaces are stripped because CJK text is not space-delimited; this also
    makes the metric insensitive to tokenisation differences.
    """
    ref_chars = list(reference.strip().replace(" ", ""))
    hyp_chars = list(hypothesis.strip().replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def error_rate(reference: str, hypothesis: str, language: str) -> tuple[float, str]:
    """Return the appropriate error rate and metric name for the language.

    Returns
    -------
    (rate, metric_name) — e.g. (0.22, "cer") or (0.15, "wer")
    """
    ref_n = normalize_text(reference, language)
    hyp_n = normalize_text(hypothesis, language)
    if language in CJK_LANGUAGES:
        return character_error_rate(ref_n, hyp_n), "cer"
    return word_error_rate(ref_n, hyp_n), "wer"


# ---------------------------------------------------------------------------
# Alignment: many hypothesis segments → single reference string
# ---------------------------------------------------------------------------

def align_hypothesis_to_reference(
    hypothesis_segments: list[str],
    reference: str,
    language: str,
) -> str:
    """Concatenate hypothesis segments into one string aligned with reference.

    Online VAC output is N segments that do not correspond 1:1 with reference
    turns.  This simply joins them (removing any overlap-boundary duplicates)
    for corpus-level metric computation.

    Deduplication strategy: if the end of the accumulated text already
    contains the beginning of the next segment (within a 10-token window),
    the overlapping part is dropped before appending.
    """
    if not hypothesis_segments:
        return ""

    is_cjk = language in CJK_LANGUAGES
    sep = "" if is_cjk else " "
    joined = hypothesis_segments[0]

    for seg in hypothesis_segments[1:]:
        seg = seg.strip()
        if not seg:
            continue
        # Check overlap: last N chars/words of accumulated vs first N of seg
        window = 15 if is_cjk else 8
        tail = joined[-window * 3:] if is_cjk else " ".join(joined.split()[-window:])
        head = seg[:window * 3] if is_cjk else " ".join(seg.split()[:window])
        # Find longest suffix of tail that is a prefix of head
        overlap_len = 0
        for k in range(min(len(tail), len(head)), 0, -1):
            if tail.endswith(head[:k]):
                overlap_len = k
                break
        seg = seg[overlap_len:].lstrip() if overlap_len else seg
        if seg:
            joined = joined + sep + seg

    return joined.strip()


# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------

def latency_percentiles(
    latencies_s: list[float],
) -> dict[str, float]:
    """Compute p50/p90/p95/p99 latency percentiles from a list of seconds."""
    if not latencies_s:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}

    import statistics

    sorted_l = sorted(latencies_s)
    n = len(sorted_l)

    def _pct(p: float) -> float:
        idx = max(0, int(n * p / 100) - 1)
        return sorted_l[min(idx, n - 1)]

    return {
        "p50": round(_pct(50), 3),
        "p90": round(_pct(90), 3),
        "p95": round(_pct(95), 3),
        "p99": round(_pct(99), 3),
        "mean": round(statistics.mean(sorted_l), 3),
        "max": round(sorted_l[-1], 3),
        "samples": n,
    }


# ---------------------------------------------------------------------------
# BLEU score (corpus-level n-gram precision)
# ---------------------------------------------------------------------------

_CJK_BLEU_RANGES = (
    ("\u4e00", "\u9fff"),   # CJK Unified Ideographs
    ("\u3040", "\u30ff"),   # Hiragana + Katakana
)

_BLEU_SMOOTHING_EPSILON = 0.1


def _is_cjk_char(char: str) -> bool:
    return any(lo <= char <= hi for lo, hi in _CJK_BLEU_RANGES)


def _tokenize_for_bleu(text: str) -> list[str]:
    """Tokenize text for BLEU. CJK chars become individual tokens."""
    tokens: list[str] = []
    for word in text.strip().split():
        if any(_is_cjk_char(c) for c in word):
            buf = ""
            for c in word:
                if _is_cjk_char(c):
                    if buf:
                        tokens.append(buf)
                        buf = ""
                    tokens.append(c)
                else:
                    buf += c
            if buf:
                tokens.append(buf)
        else:
            tokens.append(word)
    return tokens


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    references: list[str],
    hypotheses: list[str],
    max_n: int = 4,
) -> float:
    """Compute corpus-level BLEU with Chen & Cherry add-epsilon smoothing.

    CJK scripts are tokenized at character level.
    For publication-grade results, use sacrebleu.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have same length")
    if not references:
        return 0.0

    precisions = []
    bp_r = 0
    bp_c = 0

    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = _tokenize_for_bleu(ref)
            hyp_tokens = _tokenize_for_bleu(hyp)
            if n == 1:
                bp_r += len(ref_tokens)
                bp_c += len(hyp_tokens)
            ref_ngrams = _get_ngrams(ref_tokens, n)
            hyp_ngrams = _get_ngrams(hyp_tokens, n)
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            total += sum(hyp_ngrams.values())

        if total > 0:
            precision = matches / total
        else:
            precision = 0.0
        if precision == 0.0:
            precision = _BLEU_SMOOTHING_EPSILON / (total + 1) if total == 0 else _BLEU_SMOOTHING_EPSILON
        precisions.append(precision)

    log_avg = sum(math.log(p) for p in precisions) / max_n
    if log_avg == float("-inf"):
        return 0.0

    if bp_c == 0:
        bp = 0.0
    elif bp_c >= bp_r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - bp_r / bp_c)

    return bp * math.exp(log_avg)
