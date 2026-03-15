"""VAC parameter sweep benchmark.

Feeds a real audio file (with ground-truth transcript) through
VACOnlineProcessor under a grid of (prebuffer_s, stride_s, overlap_s)
configurations, collecting:
  - CER / WER against the ground-truth reference
  - Time-to-first-text (TTFT) — wall-clock seconds from audio start
  - Segment count
  - Hallucination flag — last segment score heuristic

The sweep is intentionally backend-agnostic: it takes a coroutine
``transcribe_fn(audio_array, language) → str`` so it can be driven by
any Whisper backend (vllm-mlx, faster-whisper, etc.).

Usage (programmatic)::

    from benchmarks.vac_sweep import run_vac_sweep
    results = asyncio.run(run_vac_sweep(
        audio=audio_np,
        reference=reference_text,
        language="zh",
        transcribe_fn=my_backend.transcribe_text,
    ))

Usage (CLI)::

    uv run python -m benchmarks.vac_sweep \\
        --audio  modules/transcription-service/tests/fixtures/audio/meeting_zh.wav \\
        --ref    modules/transcription-service/tests/fixtures/audio/meeting_zh.txt \\
        --lang   zh
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

from benchmarks.metrics import (
    align_hypothesis_to_reference,
    error_rate,
    latency_percentiles,
)

# ---------------------------------------------------------------------------
# Config grid
# ---------------------------------------------------------------------------

# Each dimension represents the range of values to sweep.
# Modify these lists to widen or narrow the search space.
# Default grid for real-time streaming (caption delay ≤7s).
# For offline/post-meeting, just run Whisper on the whole file — no VAC needed.
PREBUFFER_VALUES: list[float] = [0.5, 1.0, 1.5, 2.0, 3.0]
STRIDE_VALUES: list[float] = [1.5, 2.5, 3.5, 4.5, 6.0]
OVERLAP_VALUES: list[float] = [0.3, 0.5, 1.0, 1.5]

# Hallucination heuristic: if the last segment's character length
# relative to any prior segment is unusually large AND its CER against
# surrounding context is high, flag it.
_HALLUCINATION_LENGTH_RATIO = 2.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VACConfig:
    prebuffer_s: float
    stride_s: float
    overlap_s: float

    def as_label(self) -> str:
        return f"pre={self.prebuffer_s}s stride={self.stride_s}s ovlp={self.overlap_s}s"


@dataclass
class SegmentRecord:
    index: int
    text: str
    wall_time_s: float          # wall-clock seconds from first feed() call
    inference_latency_s: float  # time the transcribe_fn took for this chunk


@dataclass
class SweepResult:
    config: VACConfig
    language: str
    reference: str
    hypothesis: str
    error_metric: str           # "cer" or "wer"
    error_rate: float
    segment_count: int
    ttft_s: float | None        # wall-clock seconds to first non-empty text
    segments: list[SegmentRecord]
    inference_latencies: list[float]
    latency_stats: dict
    hallucination_flag: bool
    error: str | None = None    # populated if run raised an exception

    def summary_row(self) -> dict:
        return {
            "prebuffer_s": self.config.prebuffer_s,
            "stride_s": self.config.stride_s,
            "overlap_s": self.config.overlap_s,
            self.error_metric: round(self.error_rate, 4),
            "segment_count": self.segment_count,
            "ttft_s": round(self.ttft_s, 2) if self.ttft_s else None,
            "p50_latency_s": self.latency_stats.get("p50"),
            "p95_latency_s": self.latency_stats.get("p95"),
            "hallucination": self.hallucination_flag,
        }


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------

TranscribeFn = Callable[[np.ndarray, str], Awaitable[str]]


async def _run_single_config(
    audio: np.ndarray,
    reference: str,
    language: str,
    config: VACConfig,
    transcribe_fn: TranscribeFn,
    chunk_size: int = 1600,   # 100 ms at 16 kHz
    sample_rate: int = 16_000,
) -> SweepResult:
    """Run one VACOnlineProcessor configuration against a full audio file."""
    # Import here so the module is importable even without the full service env
    from vac_online_processor import VACOnlineProcessor

    proc = VACOnlineProcessor(
        prebuffer_s=config.prebuffer_s,
        overlap_s=config.overlap_s,
        stride_s=config.stride_s,
        sampling_rate=sample_rate,
    )

    segments: list[SegmentRecord] = []
    inference_latencies: list[float] = []
    ttft_s: float | None = None
    run_start = time.perf_counter()

    try:
        # Stream audio in 100 ms chunks, exactly as the live pipeline does
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i: i + chunk_size]
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
            await proc.feed_audio(chunk)

            if proc.ready_for_inference():
                inference_audio = proc.get_inference_audio()
                inf_start = time.perf_counter()
                text = await transcribe_fn(inference_audio, language)
                inf_elapsed = time.perf_counter() - inf_start
                inference_latencies.append(inf_elapsed)

                text = text.strip()
                if text:
                    wall_t = time.perf_counter() - run_start
                    if ttft_s is None:
                        ttft_s = wall_t
                    segments.append(SegmentRecord(
                        index=len(segments),
                        text=text,
                        wall_time_s=wall_t,
                        inference_latency_s=inf_elapsed,
                    ))

        # Flush any remaining buffered audio after stream ends
        if proc._buffer_samples > 0:
            remaining = proc.get_inference_audio()
            if len(remaining) > 0:
                inf_start = time.perf_counter()
                text = await transcribe_fn(remaining, language)
                inf_elapsed = time.perf_counter() - inf_start
                inference_latencies.append(inf_elapsed)
                text = text.strip()
                if text:
                    segments.append(SegmentRecord(
                        index=len(segments),
                        text=text,
                        wall_time_s=time.perf_counter() - run_start,
                        inference_latency_s=inf_elapsed,
                    ))

    except Exception as exc:
        return SweepResult(
            config=config,
            language=language,
            reference=reference,
            hypothesis="",
            error_metric="cer" if language in {"zh", "ja", "ko"} else "wer",
            error_rate=1.0,
            segment_count=0,
            ttft_s=None,
            segments=[],
            inference_latencies=[],
            latency_stats={},
            hallucination_flag=False,
            error=str(exc),
        )

    # Assemble hypothesis by joining segments with dedup at boundaries
    hypothesis = align_hypothesis_to_reference(
        [s.text for s in segments], reference, language
    )

    er, metric_name = error_rate(reference, hypothesis, language)
    lat_stats = latency_percentiles(inference_latencies)

    # Hallucination heuristic: last segment much longer than avg of preceding
    hallucination = False
    if len(segments) >= 3:
        preceding_lengths = [len(s.text) for s in segments[:-1]]
        avg_len = sum(preceding_lengths) / len(preceding_lengths)
        last_len = len(segments[-1].text)
        if avg_len > 0 and last_len / avg_len > _HALLUCINATION_LENGTH_RATIO:
            hallucination = True

    return SweepResult(
        config=config,
        language=language,
        reference=reference,
        hypothesis=hypothesis,
        error_metric=metric_name,
        error_rate=er,
        segment_count=len(segments),
        ttft_s=ttft_s,
        segments=segments,
        inference_latencies=inference_latencies,
        latency_stats=lat_stats,
        hallucination_flag=hallucination,
    )


async def run_vac_sweep(
    audio: np.ndarray,
    reference: str,
    language: str,
    transcribe_fn: TranscribeFn,
    prebuffer_values: list[float] | None = None,
    stride_values: list[float] | None = None,
    overlap_values: list[float] | None = None,
    sample_rate: int = 16_000,
) -> list[SweepResult]:
    """Run the full parameter sweep and return results sorted by error_rate."""
    pb = prebuffer_values or PREBUFFER_VALUES
    st = stride_values or STRIDE_VALUES
    ov = overlap_values or OVERLAP_VALUES

    configs = [
        VACConfig(prebuffer_s=pre, stride_s=stride, overlap_s=ovlp)
        for pre, stride, ovlp in itertools.product(pb, st, ov)
        # Constraint: overlap must be < stride (otherwise buffer never clears)
        if ovlp < stride
    ]

    results = []
    for cfg in configs:
        result = await _run_single_config(
            audio=audio,
            reference=reference,
            language=language,
            config=cfg,
            transcribe_fn=transcribe_fn,
            sample_rate=sample_rate,
        )
        results.append(result)

    # Primary sort: error rate ascending; secondary: TTFT ascending
    results.sort(key=lambda r: (r.error_rate, r.ttft_s or 999))
    return results


def print_sweep_table(results: list[SweepResult]) -> None:
    """Print a ranked summary table to stdout."""
    if not results:
        print("No results.")
        return

    metric = results[0].error_metric.upper()
    print(f"\n{'Rank':<5} {'Prebuf':>7} {'Stride':>7} {'Overlap':>8} "
          f"{metric:>7} {'Segs':>5} {'TTFT':>7} {'p95ms':>7} {'Hall':>5}")
    print("-" * 70)
    for rank, r in enumerate(results, 1):
        ttft = f"{r.ttft_s:.1f}s" if r.ttft_s else " —"
        p95 = f"{r.latency_stats.get('p95', 0) * 1000:.0f}" if r.latency_stats else "—"
        hall = "YES" if r.hallucination_flag else "no"
        print(
            f"{rank:<5} {r.config.prebuffer_s:>7.1f} {r.config.stride_s:>7.1f} "
            f"{r.config.overlap_s:>8.1f} {r.error_rate:>7.3f} "
            f"{r.segment_count:>5} {ttft:>7} {p95:>7} {hall:>5}"
        )
    print()
    best = results[0]
    print(f"Best config: {best.config.as_label()}")
    ttft_str = f"{best.ttft_s:.1f}s" if best.ttft_s is not None else "—"
    print(f"  {metric}: {best.error_rate:.3f}  |  TTFT: {ttft_str}")
    if best.hallucination_flag:
        print("  WARNING: hallucination flag set on best config — check last segment")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _make_stub_transcribe_fn(language: str) -> TranscribeFn:
    """Return a stub transcribe function for dry-run / unit testing.

    Simulates ~1.2 s inference latency and returns a plausible placeholder.
    Replace with a real backend in production use.
    """
    async def _stub(audio: np.ndarray, lang: str) -> str:
        await asyncio.sleep(0.05)   # non-blocking stub — fast for dry-run
        dur = len(audio) / 16_000
        return f"[stub {dur:.1f}s {lang}]"
    return _stub


def main() -> None:
    import argparse
    import soundfile as sf
    from livetranslate_common.logging import setup_logging, get_logger

    setup_logging(service_name="vac-sweep")
    log = get_logger()

    parser = argparse.ArgumentParser(description="VAC parameter sweep benchmark")
    parser.add_argument("--audio", type=Path, required=True,
                        help="Path to 16 kHz mono WAV audio file")
    parser.add_argument("--ref", type=Path, required=True,
                        help="Path to ground-truth transcript (.txt, one or more lines)")
    parser.add_argument("--lang", default="zh",
                        help="Language code (zh/en/ja/es, default: zh)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmarks/results/vac_sweep"))
    parser.add_argument("--prebuffer", nargs="+", type=float, default=None,
                        help="Override prebuffer_s values (e.g. 0.5 1.0 2.0)")
    parser.add_argument("--stride", nargs="+", type=float, default=None,
                        help="Override stride_s values")
    parser.add_argument("--overlap", nargs="+", type=float, default=None,
                        help="Override overlap_s values")
    parser.add_argument("--stub", action="store_true",
                        help="Use stub transcriber (no model needed, for CI/testing)")
    parser.add_argument("--backend", default="vllm",
                        choices=["vllm", "faster-whisper", "mlx"],
                        help="Whisper backend to use (default: vllm)")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vllm-mlx API base URL")
    parser.add_argument("--model", default="large-v3-turbo",
                        help="Whisper model name for non-vllm backends")
    args = parser.parse_args()

    # Load audio
    audio_data, sr = sf.read(str(args.audio))
    if sr != 16_000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16_000)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.float32)

    # Load reference
    reference = args.ref.read_text(encoding="utf-8").strip()
    # Join multi-line references into one string for corpus scoring
    reference = " ".join(reference.splitlines())

    # For autodetect mode, infer the metric language from the reference text
    metric_lang = args.lang
    if metric_lang == "auto":
        from benchmarks.metrics import CJK_LANGUAGES
        cjk_chars = sum(1 for c in reference if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
        if cjk_chars > len(reference) * 0.3:
            metric_lang = "zh"  # CJK-dominant → use CER
        else:
            metric_lang = "en"  # default to WER

    log.info("sweep_starting",
             audio=args.audio.name,
             language=args.lang,
             metric_lang=metric_lang,
             duration_s=round(len(audio_data) / 16_000, 1),
             stub=args.stub)

    # Build transcribe_fn
    if args.stub:
        transcribe_fn = _make_stub_transcribe_fn(args.lang)
    else:
        transcribe_fn = _build_real_transcribe_fn(args)

    results = asyncio.run(run_vac_sweep(
        audio=audio_data,
        reference=reference,
        language=metric_lang,
        transcribe_fn=transcribe_fn,
        prebuffer_values=args.prebuffer,
        stride_values=args.stride,
        overlap_values=args.overlap,
    ))

    print_sweep_table(results)

    # Persist results + human-readable table
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save table as TSV for easy viewing
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    table_path = args.output_dir / f"sweep_{args.lang}_{ts}.tsv"
    if results:
        metric = results[0].error_metric.upper()
        lines = [f"Rank\tPrebuf\tStride\tOverlap\t{metric}\tSegs\tTTFT\tp95ms\tHall"]
        for rank, r in enumerate(results, 1):
            ttft = f"{r.ttft_s:.1f}" if r.ttft_s else ""
            p95 = f"{r.latency_stats.get('p95', 0) * 1000:.0f}" if r.latency_stats else ""
            lines.append(
                f"{rank}\t{r.config.prebuffer_s}\t{r.config.stride_s}\t"
                f"{r.config.overlap_s}\t{r.error_rate:.4f}\t{r.segment_count}\t"
                f"{ttft}\t{p95}\t{'YES' if r.hallucination_flag else 'no'}"
            )
        table_path.write_text("\n".join(lines) + "\n")
        log.info("sweep_table_saved", path=str(table_path))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"sweep_{args.lang}_{ts}.json"
    payload = {
        "audio": str(args.audio),
        "language": args.lang,
        "reference": reference,
        "timestamp": ts,
        "model": args.model,
        "backend": args.backend,
        "results": [
            {**asdict(r), "config": asdict(r.config)}
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info("sweep_saved", path=str(out_path), configs=len(results))

    # Append best result to JSONL index for cross-run comparison
    index_path = args.output_dir / "transcription_index.jsonl"
    if results:
        best = results[0]
        index_entry = {
            "timestamp": ts,
            "model": args.model,
            "backend": args.backend,
            "language": args.lang,
            "audio": args.audio.name,
            **best.summary_row(),
        }
        with open(index_path, "a") as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
        log.info("index_appended", path=str(index_path))


def _build_real_transcribe_fn(args) -> TranscribeFn:
    """Construct a real transcribe function from CLI args."""
    import sys
    from pathlib import Path as _Path
    src = _Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src))

    if args.backend == "vllm":
        import httpx
        vllm_model = args.model if args.model != "large-v3-turbo" else "openai/whisper-large-v3-turbo"

        async def _vllm_transcribe(audio: np.ndarray, language: str) -> str:
            import io
            import soundfile as _sf
            buf = io.BytesIO()
            _sf.write(buf, audio, 16_000, format="WAV", subtype="FLOAT")
            buf.seek(0)
            data = {"model": vllm_model}
            # Support autodetect: --lang auto omits language hint
            if language and language != "auto":
                data["language"] = language
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{args.vllm_url}/audio/transcriptions",
                    files={"file": ("audio.wav", buf, "audio/wav")},
                    data=data,
                )
                resp.raise_for_status()
                return resp.json().get("text", "")

        return _vllm_transcribe

    elif args.backend == "faster-whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="auto", compute_type="float16")

        async def _fw_transcribe(audio: np.ndarray, language: str) -> str:
            segments, _ = model.transcribe(
                audio, language=language, beam_size=5,
                temperature=(0.0,), no_speech_threshold=0.6,
            )
            return " ".join(s.text for s in segments).strip()

        return _fw_transcribe

    elif args.backend == "mlx":
        from backends.mlx_whisper import MLXWhisperBackend
        backend = MLXWhisperBackend(model_name=args.model)

        async def _mlx_transcribe(audio: np.ndarray, language: str) -> str:
            result = await backend.transcribe(audio, language=language)
            return result.text

        return _mlx_transcribe

    else:
        raise ValueError(f"Unsupported backend: {args.backend}")


if __name__ == "__main__":
    main()
