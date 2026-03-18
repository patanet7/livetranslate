"""End-to-end pipeline benchmark: transcription + translation.

Measures the full latency budget from audio-stream-start to translated
caption appearing, across a configurable set of meeting audio files and
target languages.

Pipeline stages timed independently:
  Stage 1  VAC chunking (VACOnlineProcessor)          — sub-millisecond
  Stage 2  Whisper inference                           — ~1.2 s per chunk
  Stage 3  Overlap dedup                               — sub-millisecond
  Stage 4  Translation LLM (Ollama / vllm-mlx)        — ~1.0 s per segment
  Stage 5  Context window accumulation (rolling)

Outputs
-------
  benchmarks/results/pipeline/<timestamp>.json  — full per-segment trace
  benchmarks/results/pipeline/<timestamp>.txt   — human-readable summary

Context-window impact is measured by running each file at context_sizes
[0, 3, 5] and reporting BLEU delta between 0 and 5.

Usage::

    uv run python -m benchmarks.pipeline_benchmark \\
        --lang zh \\
        --target-lang en \\
        --vllm-url http://localhost:8000/v1 \\
        --translation-url http://localhost:3000

Or via just::

    just benchmark
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from translation.llm_client import extract_translation_text
except ImportError:  # pragma: no cover - benchmark still runs without orchestration package import
    def extract_translation_text(response: str) -> str:
        return response.strip()

_FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "audio"
_RESULTS = Path(__file__).parent / "results" / "pipeline"

# Default meeting files per language
_MEETING_FILES: dict[str, tuple[str, str]] = {
    "zh": ("meeting_zh.wav", "meeting_zh.txt"),
    "en": ("meeting_en.wav", "meeting_en.txt"),
    "ja": ("meeting_ja.wav", "meeting_ja.txt"),
    "es": ("meeting_es.wav", "meeting_es.txt"),
}

# Translation reference files keyed by (source_lang, target_lang)
_TRANSLATION_REFS: dict[tuple[str, str], str] = {
    ("zh", "en"): "meeting_zh_en.reference.txt",
    ("ja", "en"): "meeting_ja_en.reference.txt",
    ("es", "en"): "meeting_es_en.reference.txt",
    ("en", "zh"): "meeting_en_zh.reference.txt",
    ("en", "en"): "",  # no translation needed
}


@dataclass
class SegmentTrace:
    index: int
    audio_end_wall_s: float     # when the audio chunk that triggered inference ended
    transcription: str
    transcription_latency_s: float
    translation: str
    translation_latency_s: float
    e2e_latency_s: float        # wall-clock from segment audio-start to translated caption


@dataclass
class PipelineResult:
    language: str
    target_language: str
    context_size: int
    audio_file: str
    transcription_model: str
    reference_transcription: str
    reference_translation: str
    # Metrics
    transcription_error_rate: float
    transcription_metric: str    # "cer" or "wer"
    translation_bleu: float
    # Latency
    ttft_s: float | None         # time-to-first-transcription
    ttc_s: float | None          # time-to-first-caption (including translation)
    transcription_latency_stats: dict
    translation_latency_stats: dict
    e2e_latency_stats: dict
    # Segment detail
    segments: list[SegmentTrace]
    # Summary
    segment_count: int
    hallucination_flag: bool
    # VAC config (for sweep tracking)
    vac_stride_s: float = 4.0
    vac_overlap_s: float = 1.0
    # Translation config (for sweep)
    llm_temperature: float = 0.3
    max_context_tokens: int = 500


# ---------------------------------------------------------------------------
# Translation client (calls orchestration service or direct Ollama)
# ---------------------------------------------------------------------------

async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    context: list[str],
    translation_url: str,
    model: str,
) -> tuple[str, float]:
    """Call translation endpoint, return (translated_text, latency_s)."""
    import httpx

    payload = {
        "text": text,
        "source_language": source_lang,
        "target_language": target_lang,
        "context": context,
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{translation_url.rstrip('/')}/api/translate",
            json=payload,
        )
        resp.raise_for_status()
        translated = extract_translation_text(resp.json().get("translated_text", ""))
    latency = time.perf_counter() - t0
    return translated, latency


# ---------------------------------------------------------------------------
# Direct Ollama translation (bypasses orchestration service)
# ---------------------------------------------------------------------------

async def translate_via_ollama(
    text: str,
    source_lang: str,
    target_lang: str,
    context: list[str],
    ollama_url: str,
    model: str,
    temperature: float = 0.3,
    max_context_tokens: int = 500,
) -> tuple[str, float]:
    """Translate directly via Ollama/vllm-mlx OpenAI-compatible API."""
    import httpx

    _LANG_NAMES = {
        "zh": "Chinese", "en": "English", "ja": "Japanese",
        "es": "Spanish", "fr": "French", "de": "German", "ko": "Korean",
    }
    src_name = _LANG_NAMES.get(source_lang, source_lang)
    tgt_name = _LANG_NAMES.get(target_lang, target_lang)

    # Truncate context to max_context_tokens (rough: 1 CJK char ≈ 1 token, 4 Latin chars ≈ 1 token)
    ctx_block = ""
    if context:
        token_count = 0
        kept = []
        for c in reversed(context[-10:]):
            cjk = sum(1 for ch in c if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff')
            est = cjk + (len(c) - cjk) // 4
            if token_count + est > max_context_tokens:
                break
            kept.append(c)
            token_count += est
        if kept:
            kept.reverse()
            ctx_block = "Previous translations for context:\n" + "\n".join(
                f"  {i+1}. {c}" for i, c in enumerate(kept)
            ) + "\n\n"

    prompt = (
        f"{ctx_block}"
        f"Translate the following {src_name} text to {tgt_name}. "
        f"Output only the translation, no explanation.\n\n"
        f"{text}"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{ollama_url.rstrip('/')}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        translated = extract_translation_text(resp.json()["choices"][0]["message"]["content"])
    latency = time.perf_counter() - t0
    return translated, latency


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline(
    audio: np.ndarray,
    reference_transcription: str,
    reference_translation: str,
    language: str,
    target_language: str,
    context_size: int,
    transcribe_fn,
    translate_fn,
    vac_prebuffer_s: float = 1.0,
    vac_stride_s: float = 2.5,
    vac_overlap_s: float = 0.5,
    sample_rate: int = 16_000,
    chunk_size: int = 1600,
    audio_file: str = "",
    transcription_model: str = "",
    llm_temperature: float = 0.3,
    max_context_tokens: int = 500,
) -> PipelineResult:
    """Run full transcription+translation pipeline on an audio array."""
    from vac_online_processor import VACOnlineProcessor
    from benchmarks.metrics import (
        align_hypothesis_to_reference, error_rate, latency_percentiles,
        bleu_score,
    )

    proc = VACOnlineProcessor(
        prebuffer_s=vac_prebuffer_s,
        overlap_s=vac_overlap_s,
        stride_s=vac_stride_s,
        sampling_rate=sample_rate,
    )

    segments: list[SegmentTrace] = []
    rolling_context: list[str] = []
    ttft_s: float | None = None
    ttc_s: float | None = None
    stream_start = time.perf_counter()
    hallucination = False

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i: i + chunk_size].astype(np.float32)
        await proc.feed_audio(chunk)

        if proc.ready_for_inference():
            audio_end_wall = time.perf_counter() - stream_start
            inference_audio = proc.get_inference_audio()

            # Stage 2: Whisper
            t_asr_start = time.perf_counter()
            transcription = await transcribe_fn(inference_audio, language)
            asr_latency = time.perf_counter() - t_asr_start
            transcription = transcription.strip()

            if not transcription:
                continue

            if ttft_s is None:
                ttft_s = time.perf_counter() - stream_start

            # Stage 4: Translation
            if target_language and target_language != language:
                ctx = rolling_context[-context_size:] if context_size > 0 else []
                translation, tl_latency = await translate_fn(
                    transcription, language, target_language, ctx
                )
                if context_size > 0:
                    rolling_context.append(translation)
            else:
                translation = transcription
                tl_latency = 0.0

            if ttc_s is None and translation:
                ttc_s = time.perf_counter() - stream_start

            e2e = asr_latency + tl_latency
            segments.append(SegmentTrace(
                index=len(segments),
                audio_end_wall_s=audio_end_wall,
                transcription=transcription,
                transcription_latency_s=asr_latency,
                translation=translation,
                translation_latency_s=tl_latency,
                e2e_latency_s=e2e,
            ))

    # Flush remaining audio
    if proc._buffer_samples > 0:
        remaining = proc.get_inference_audio()
        if len(remaining) > 0:
            t_start = time.perf_counter()
            text = (await transcribe_fn(remaining, language)).strip()
            asr_lat = time.perf_counter() - t_start
            if text:
                ctx = rolling_context[-context_size:] if context_size > 0 else []
                if target_language and target_language != language:
                    trans, tl_lat = await translate_fn(text, language, target_language, ctx)
                else:
                    trans, tl_lat = text, 0.0
                segments.append(SegmentTrace(
                    index=len(segments),
                    audio_end_wall_s=time.perf_counter() - stream_start,
                    transcription=text,
                    transcription_latency_s=asr_lat,
                    translation=trans,
                    translation_latency_s=tl_lat,
                    e2e_latency_s=asr_lat + tl_lat,
                ))

    # Hallucination check
    if len(segments) >= 3:
        lens = [len(s.transcription) for s in segments[:-1]]
        avg = sum(lens) / len(lens)
        if avg > 0 and len(segments[-1].transcription) / avg > 2.5:
            hallucination = True

    # Compute metrics
    hyp_transcriptions = [s.transcription for s in segments]
    hypothesis = align_hypothesis_to_reference(
        hyp_transcriptions, reference_transcription, language
    )
    er, metric_name = error_rate(reference_transcription, hypothesis, language)

    # BLEU for translation
    bleu = 0.0
    if reference_translation and segments:
        hyp_translations = [s.translation for s in segments]
        hyp_joined = " ".join(hyp_translations)
        bleu = bleu_score([reference_translation], [hyp_joined])

    return PipelineResult(
        language=language,
        target_language=target_language,
        context_size=context_size,
        audio_file=audio_file,
        transcription_model=transcription_model,
        reference_transcription=reference_transcription,
        reference_translation=reference_translation,
        transcription_error_rate=er,
        transcription_metric=metric_name,
        translation_bleu=round(bleu, 4),
        ttft_s=ttft_s,
        ttc_s=ttc_s,
        transcription_latency_stats=latency_percentiles(
            [s.transcription_latency_s for s in segments]
        ),
        translation_latency_stats=latency_percentiles(
            [s.translation_latency_s for s in segments]
        ),
        e2e_latency_stats=latency_percentiles(
            [s.e2e_latency_s for s in segments]
        ),
        segments=segments,
        segment_count=len(segments),
        hallucination_flag=hallucination,
        vac_stride_s=vac_stride_s,
        vac_overlap_s=vac_overlap_s,
        llm_temperature=llm_temperature,
        max_context_tokens=max_context_tokens,
    )


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

# Latency budget thresholds — fail CI if exceeded
MAX_TTFT_S = 5.0   # time to first transcription
MAX_TTC_S = 7.0    # time to first translated caption
MAX_E2E_P95_S = 4.0  # 95th percentile ASR + translation


def check_latency_budgets(results: list[PipelineResult]) -> list[str]:
    """Check results against latency budgets. Returns list of failures."""
    failures = []
    for r in results:
        label = f"{r.language}→{r.target_language} ctx={r.context_size}"
        if r.ttft_s is not None and r.ttft_s > MAX_TTFT_S:
            failures.append(f"TTFT {r.ttft_s:.2f}s > {MAX_TTFT_S}s [{label}]")
        if r.ttc_s is not None and r.ttc_s > MAX_TTC_S:
            failures.append(f"TTC {r.ttc_s:.2f}s > {MAX_TTC_S}s [{label}]")
        e2e_p95 = r.e2e_latency_stats.get("p95", 0)
        if e2e_p95 > MAX_E2E_P95_S:
            failures.append(f"E2E p95 {e2e_p95:.2f}s > {MAX_E2E_P95_S}s [{label}]")
    return failures


def print_pipeline_report(results: list[PipelineResult]) -> None:
    """Print a formatted report for all pipeline runs."""
    print("\n" + "=" * 90)
    print("PIPELINE BENCHMARK RESULTS")
    print("=" * 90)

    for r in results:
        metric = r.transcription_metric.upper()
        print(f"\n  File: {r.audio_file}  Lang: {r.language}→{r.target_language}"
              f"  Ctx={r.context_size}")
        print(f"    ASR model:      {r.transcription_model or 'n/a'}")
        print(f"    {metric}:          {r.transcription_error_rate:.3f}"
              f"  ({100*(1-r.transcription_error_rate):.1f}% accuracy)")
        print(f"    BLEU:          {r.translation_bleu:.4f}")
        print(f"    Segments:      {r.segment_count}")
        print(f"    TTFT:          {r.ttft_s:.2f}s" if r.ttft_s else "    TTFT:          —")
        print(f"    TTC:           {r.ttc_s:.2f}s" if r.ttc_s else "    TTC:           —")
        asr = r.transcription_latency_stats
        tl = r.translation_latency_stats
        e2e = r.e2e_latency_stats
        print(f"    ASR latency:   p50={asr.get('p50',0):.2f}s  "
              f"p95={asr.get('p95',0):.2f}s")
        print(f"    Trans latency: p50={tl.get('p50',0):.2f}s  "
              f"p95={tl.get('p95',0):.2f}s")
        print(f"    E2E latency:   p50={e2e.get('p50',0):.2f}s  "
              f"p95={e2e.get('p95',0):.2f}s")
        if r.hallucination_flag:
            print("    WARNING: hallucination flag set on last segment")

    # Context window impact table (group by file+lang, vary context_size)
    from itertools import groupby
    keyfn = lambda r: (r.audio_file, r.language, r.target_language)
    sorted_r = sorted(results, key=keyfn)
    print("\n  Context Window Impact on Translation BLEU:")
    print(f"  {'File':<20} {'Lang':>8}  {'Ctx=0':>8} {'Ctx=3':>8} {'Ctx=5':>8}  Delta")
    print("  " + "-" * 60)
    for key, grp in groupby(sorted_r, key=keyfn):
        grp = list(grp)
        ctx_map = {r.context_size: r.translation_bleu for r in grp}
        b0 = ctx_map.get(0, None)
        b3 = ctx_map.get(3, None)
        b5 = ctx_map.get(5, None)
        delta = f"+{b5-b0:.4f}" if (b5 is not None and b0 is not None) else "—"
        fname = Path(key[0]).name[:20]
        lang_pair = f"{key[1]}→{key[2]}"
        b0s = f"{b0:.4f}" if b0 is not None else "—"
        b3s = f"{b3:.4f}" if b3 is not None else "—"
        b5s = f"{b5:.4f}" if b5 is not None else "—"
        print(f"  {fname:<20} {lang_pair:>8}  {b0s:>8} {b3s:>8} {b5s:>8}  {delta}")
    print()

    # Latency budget check
    failures = check_latency_budgets(results)
    print("  Latency Budget Check:")
    print(f"    TTFT max:    {MAX_TTFT_S}s")
    print(f"    TTC max:     {MAX_TTC_S}s")
    print(f"    E2E p95 max: {MAX_E2E_P95_S}s")
    if failures:
        print("    FAILURES:")
        for f in failures:
            print(f"      FAIL: {f}")
    else:
        print("    All budgets PASS")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import soundfile as sf
    from livetranslate_common.logging import setup_logging, get_logger

    setup_logging(service_name="pipeline-benchmark")
    log = get_logger()

    parser = argparse.ArgumentParser(description="End-to-end pipeline benchmark")
    parser.add_argument("--lang", default="zh",
                        help="Source language (zh/en/ja/es)")
    parser.add_argument("--target-lang", default="en",
                        help="Target translation language (default: en)")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vllm-mlx URL for Whisper transcription")
    parser.add_argument("--translation-url", default="",
                        help="Orchestration service URL for translation "
                             "(empty = use --ollama-url directly)")
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1",
                        help="Ollama URL for direct translation (when --translation-url empty)")
    parser.add_argument("--model", default="qwen3.5:7b",
                        help="Translation LLM model name")
    parser.add_argument(
        "--transcription-models",
        nargs="+",
        default=["openai/whisper-large-v3-turbo"],
        help="Transcription model IDs to sweep (default: openai/whisper-large-v3-turbo)",
    )
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[0, 3, 5],
                        help="Context window sizes to test (default: 0 3 5)")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.3],
                        help="LLM temperatures to sweep (default: 0.3)")
    parser.add_argument("--max-context-tokens", nargs="+", type=int, default=[500],
                        help="Max context token budgets to sweep (default: 500)")
    parser.add_argument("--prebuffer", type=float, default=0.5)
    parser.add_argument("--strides", nargs="+", type=float, default=[6.0],
                        help="VAC stride values to sweep (default: 6.0)")
    parser.add_argument("--overlaps", nargs="+", type=float, default=[1.5],
                        help="VAC overlap values to sweep (default: 1.5)")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Override audio file path")
    parser.add_argument("--ref", type=Path, default=None,
                        help="Override reference transcription path")
    parser.add_argument("--output-dir", type=Path, default=_RESULTS)
    parser.add_argument("--stub", action="store_true",
                        help="Use stub backend (no services required)")
    args = parser.parse_args()

    # Resolve audio + ref
    if args.audio:
        audio_path = args.audio
    else:
        wav, _ = _MEETING_FILES.get(args.lang, ("meeting_en.wav", "meeting_en.txt"))
        audio_path = _FIXTURES / wav

    if args.ref:
        ref_path = args.ref
    else:
        _, txt = _MEETING_FILES.get(args.lang, ("meeting_en.wav", "meeting_en.txt"))
        ref_path = _FIXTURES / txt

    ref_trans_path = _FIXTURES / _TRANSLATION_REFS.get((args.lang, args.target_lang), "")

    # Load audio
    audio_data, sr = sf.read(str(audio_path))
    if sr != 16_000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16_000)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.float32)

    reference_transcription = " ".join(ref_path.read_text(encoding="utf-8").splitlines())
    reference_translation = (
        " ".join(ref_trans_path.read_text(encoding="utf-8").splitlines())
        if ref_trans_path.exists() and ref_trans_path.name
        else ""
    )

    log.info("pipeline_benchmark_starting",
             lang=args.lang, target=args.target_lang,
             audio=audio_path.name,
             duration_s=round(len(audio_data) / 16_000, 1),
             context_sizes=args.context_sizes)

    # Build functions
    if args.stub:
        async def _transcribe(audio: np.ndarray, lang: str) -> str:
            await asyncio.sleep(0.05)
            return f"stub transcription {len(audio)/16000:.1f}s"

        def _make_translate_fn(temperature: float, max_ctx_tokens: int):
            async def _translate(text: str, src: str, tgt: str, ctx: list[str]) -> tuple[str, float]:
                await asyncio.sleep(0.03)
                return f"stub translation of: {text[:40]}", 0.03
            return _translate
    else:
        import httpx

        def _make_transcribe_fn(transcription_model: str):
            async def _transcribe(audio: np.ndarray, lang: str) -> str:
                import io
                import soundfile as _sf
                buf = io.BytesIO()
                _sf.write(buf, audio, 16_000, format="WAV", subtype="FLOAT")
                buf.seek(0)
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{args.vllm_url.rstrip('/v1')}/v1/audio/transcriptions",
                        files={"file": ("audio.wav", buf, "audio/wav")},
                        data={"model": transcription_model, "language": lang},
                    )
                    resp.raise_for_status()
                    return resp.json().get("text", "")

            return _transcribe

        def _make_translate_fn(temperature: float, max_ctx_tokens: int):
            if args.translation_url:
                async def _translate(text: str, src: str, tgt: str, ctx: list[str]) -> tuple[str, float]:
                    return await translate_text(text, src, tgt, ctx, args.translation_url, args.model)
            else:
                async def _translate(text: str, src: str, tgt: str, ctx: list[str]) -> tuple[str, float]:
                    return await translate_via_ollama(
                        text, src, tgt, ctx, args.ollama_url, args.model,
                        temperature=temperature, max_context_tokens=max_ctx_tokens,
                    )
            return _translate

    # Cross-product sweep: transcription_model × strides × overlaps × context_sizes × temperatures × max_context_tokens
    import itertools
    # Filter invalid combos: overlap must be < stride
    vac_combos = [(s, o) for s, o in itertools.product(args.strides, args.overlaps) if o < s]
    trans_combos = list(
        itertools.product(
            args.transcription_models,
            args.context_sizes,
            args.temperatures,
            args.max_context_tokens,
        )
    )
    total_combos = len(vac_combos) * len(trans_combos)

    log.info("sweep_matrix", total_combos=total_combos,
             vac_combos=len(vac_combos),
             trans_combos=len(trans_combos),
             transcription_models=args.transcription_models,
             strides=args.strides, overlaps=args.overlaps,
             context_sizes=args.context_sizes,
             temperatures=args.temperatures,
             max_context_tokens=args.max_context_tokens)

    all_results = []
    for stride, overlap in vac_combos:
        for transcription_model, ctx_size, temp, max_ctx_tok in trans_combos:
            log.info("running_combo",
                     transcription_model=transcription_model,
                     stride=stride, overlap=overlap,
                     ctx_size=ctx_size, temperature=temp,
                     max_context_tokens=max_ctx_tok)
            translate_fn = _make_translate_fn(temp, max_ctx_tok)
            transcribe_fn = _transcribe if args.stub else _make_transcribe_fn(transcription_model)
            result = asyncio.run(run_pipeline(
                audio=audio_data,
                reference_transcription=reference_transcription,
                reference_translation=reference_translation,
                language=args.lang,
                target_language=args.target_lang,
                context_size=ctx_size,
                transcribe_fn=transcribe_fn,
                translate_fn=translate_fn,
                vac_prebuffer_s=args.prebuffer,
                vac_stride_s=stride,
                vac_overlap_s=overlap,
                audio_file=str(audio_path),
                transcription_model="" if args.stub else transcription_model,
                llm_temperature=temp,
                max_context_tokens=max_ctx_tok,
            ))
            all_results.append(result)

    print_pipeline_report(all_results)

    # Save JSON
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"pipeline_{args.lang}_{ts}.json"

    def _serialize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return str(obj)

    payload = {
        "language": args.lang,
        "target_language": args.target_lang,
        "audio": str(audio_path),
        "timestamp": ts,
        "translation_model": args.model,
        "transcription_models": args.transcription_models,
        "sweep_config": {
            "prebuffer_s": args.prebuffer,
            "strides": args.strides,
            "overlaps": args.overlaps,
            "transcription_models": args.transcription_models,
            "context_sizes": args.context_sizes,
            "temperatures": args.temperatures,
            "max_context_tokens": args.max_context_tokens,
        },
        "results": [asdict(r) for r in all_results],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    log.info("pipeline_benchmark_saved", path=str(out_path))

    # Save full sweep table as TSV
    table_path = args.output_dir / f"pipeline_{args.lang}_{ts}.tsv"
    metric_name = all_results[0].transcription_metric.upper() if all_results else "CER"
    header = (
        f"TranscriptionModel\tStride\tOverlap\tCtx\tTemp\tMaxTok\t"
        f"{metric_name}\tBLEU\tSegs\tTTFT\tTTC\tE2E_p95"
    )
    rows = [header]
    for r in sorted(all_results, key=lambda x: -x.translation_bleu):
        ttft = f"{r.ttft_s:.1f}" if r.ttft_s else ""
        ttc = f"{r.ttc_s:.1f}" if r.ttc_s else ""
        e2e_p95 = f"{r.e2e_latency_stats.get('p95', 0):.1f}"
        rows.append(
            f"{r.transcription_model}\t{r.vac_stride_s}\t{r.vac_overlap_s}\t"
            f"{r.context_size}\t{r.llm_temperature}\t{r.max_context_tokens}\t"
            f"{r.transcription_error_rate:.4f}\t{r.translation_bleu:.4f}\t{r.segment_count}\t"
            f"{ttft}\t{ttc}\t{e2e_p95}"
        )
    table_path.write_text("\n".join(rows) + "\n")
    log.info("pipeline_table_saved", path=str(table_path))

    # Append summary to JSONL index
    index_path = args.output_dir / "pipeline_index.jsonl"
    for r in all_results:
        index_entry = {
            "timestamp": ts,
            "translation_model": args.model,
            "transcription_model": r.transcription_model,
            "language": r.language,
            "target_language": r.target_language,
            "vac_stride_s": r.vac_stride_s,
            "vac_overlap_s": r.vac_overlap_s,
            "context_size": r.context_size,
            "llm_temperature": r.llm_temperature,
            "max_context_tokens": r.max_context_tokens,
            r.transcription_metric: round(r.transcription_error_rate, 4),
            "bleu": r.translation_bleu,
            "ttft_s": round(r.ttft_s, 2) if r.ttft_s else None,
            "ttc_s": round(r.ttc_s, 2) if r.ttc_s else None,
            "e2e_p95_s": r.e2e_latency_stats.get("p95"),
            "segments": r.segment_count,
            "hallucination": r.hallucination_flag,
        }
        with open(index_path, "a") as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
    log.info("pipeline_index_appended", path=str(index_path))


if __name__ == "__main__":
    main()
