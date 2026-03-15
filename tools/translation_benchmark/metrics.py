"""Translation quality metrics for benchmarking.

BLEU: corpus-level n-gram precision (standard MT metric)
COMET: learned metric using pretrained models (optional, requires comet-ml)
"""
from __future__ import annotations

import math
from collections import Counter


def bleu_score(
    references: list[str],
    hypotheses: list[str],
    max_n: int = 4,
) -> float:
    """Compute corpus-level BLEU score.

    Simplified implementation for benchmarking — for publication-grade
    results, use sacrebleu.
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
            ref_tokens = ref.strip().split()
            hyp_tokens = hyp.strip().split()

            if n == 1:
                bp_r += len(ref_tokens)
                bp_c += len(hyp_tokens)

            ref_ngrams = _get_ngrams(ref_tokens, n)
            hyp_ngrams = _get_ngrams(hyp_tokens, n)

            # Clipped counts
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            total += sum(hyp_ngrams.values())

        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)

    # Geometric mean of precisions
    log_avg = sum(math.log(p) if p > 0 else float("-inf") for p in precisions) / max_n
    if log_avg == float("-inf"):
        return 0.0

    # Brevity penalty
    if bp_c == 0:
        bp = 0.0
    elif bp_c >= bp_r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - bp_r / bp_c)

    return bp * math.exp(log_avg)


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# --- COMET (optional, requires unbabel-comet) ---

try:
    from comet import download_model, load_from_checkpoint

    _COMET_AVAILABLE = True
except ImportError:
    _COMET_AVAILABLE = False


def comet_available() -> bool:
    """Check whether COMET scoring is available."""
    return _COMET_AVAILABLE


def comet_score(
    sources: list[str],
    references: list[str],
    hypotheses: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
) -> float | None:
    """Compute COMET score. Returns None if comet is not installed.

    Install with: uv add unbabel-comet --optional benchmark
    """
    if not _COMET_AVAILABLE:
        return None

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = model.predict(data, batch_size=32, gpus=0)
    return float(output.system_score)
