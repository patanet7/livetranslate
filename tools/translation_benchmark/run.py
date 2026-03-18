"""CLI benchmark runner for translation models.

Usage: uv run python -m tools.translation_benchmark.run --model qwen3.5:7b --lang-pair zh-en
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from livetranslate_common.logging import setup_logging, get_logger

from tools.translation_benchmark.metrics import bleu_score, comet_available, comet_score

logger = get_logger()


def get_system_info() -> dict:
    """Collect system info for reproducibility."""
    import subprocess

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": {},
        "packages": {},
    }

    # GPU info via nvidia-smi
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            parts = smi.stdout.strip().split(", ", 1)
            info["gpu"]["model"] = parts[0] if parts else "unknown"
            info["gpu"]["driver_version"] = parts[1] if len(parts) > 1 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpu"]["model"] = "N/A (nvidia-smi not found)"

    # CUDA version
    try:
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if nvcc.returncode == 0:
            for line in nvcc.stdout.splitlines():
                if "release" in line.lower():
                    info["gpu"]["cuda_version"] = line.strip().split("release")[-1].strip().rstrip(",")
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpu"]["cuda_version"] = "N/A"

    # Key Python package versions
    try:
        import importlib.metadata
        for pkg in ["httpx", "pydantic", "livetranslate-common", "unbabel-comet"]:
            try:
                info["packages"][pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                pass
    except ImportError:
        pass

    return info


def get_model_checksum(ollama_url: str, model: str) -> str | None:
    """Get SHA256 digest of the Ollama model for reproducibility."""
    import httpx as _httpx
    try:
        resp = _httpx.post(
            f"{ollama_url.rstrip('/v1')}/api/show",
            json={"name": model},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("digest", None)
    except Exception:
        pass
    return None


async def run_single_model_benchmark(
    model: str,
    lang_pair: str,
    sources: list[str],
    references: list[str],
    ollama_url: str,
    context_sizes: list[int],
    concurrency: int = 1,
) -> dict:
    """Run benchmark for a single model. Returns result dict."""
    from translation.config import TranslationConfig
    from translation.service import TranslationService
    from livetranslate_common.models import TranslationRequest

    src_lang, tgt_lang = lang_pair.split("-")
    model_result = {
        "model": model,
        "model_checksum": get_model_checksum(ollama_url, model),
        "runs": [],
    }

    for ctx_size in context_sizes:
        config = TranslationConfig(
            base_url=ollama_url,
            model=model,
            context_window_size=ctx_size,
        )
        service = TranslationService(config)

        hypotheses = []
        latencies = []

        for i, source in enumerate(sources):
            try:
                request = TranslationRequest(
                    text=source,
                    source_language=src_lang,
                    target_language=tgt_lang,
                    context=service.get_context(),
                )
                response = await service.translate(request)
                hypotheses.append(response.translated_text)
                latencies.append(response.latency_ms)
            except Exception as e:
                hypotheses.append("")
                logger.warning("translation_failed", index=i, error=str(e))

        bleu = bleu_score(references, hypotheses)

        run_result: dict = {
            "context_window_size": ctx_size,
            "bleu": round(bleu, 4),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
            "p95_latency_ms": round(
                sorted(latencies)[min(math.ceil(len(latencies) * 0.95) - 1, len(latencies) - 1)] if latencies else 0, 1
            ),
            "samples": len(sources),
            "failures": sum(1 for h in hypotheses if not h),
        }

        # COMET score (optional)
        if comet_available():
            comet = comet_score(sources, references, hypotheses)
            run_result["comet"] = round(comet, 4) if comet is not None else None
            logger.info("comet_computed", score=run_result["comet"])
        else:
            run_result["comet"] = None
            logger.info("comet_skipped", reason="unbabel-comet not installed")

        model_result["runs"].append(run_result)

        logger.info(
            "benchmark_run_complete",
            model=model,
            context_size=ctx_size,
            bleu=run_result["bleu"],
            avg_latency=run_result["avg_latency_ms"],
        )

        await service.close()

    # Concurrent throughput measurement
    if concurrency > 1:
        config = TranslationConfig(
            base_url=ollama_url,
            model=model,
            context_window_size=0,
        )
        service = TranslationService(config)

        sample_texts = (sources * ((concurrency // len(sources)) + 1))[:concurrency]

        async def single_request(text: str) -> float:
            start = time.monotonic()
            request = TranslationRequest(
                text=text,
                source_language=src_lang,
                target_language=tgt_lang,
                context=[],
            )
            await service.translate(request)
            return (time.monotonic() - start) * 1000

        batch_start = time.monotonic()
        concurrent_latencies = await asyncio.gather(
            *[single_request(t) for t in sample_texts],
            return_exceptions=True,
        )
        batch_elapsed_s = time.monotonic() - batch_start

        successful = [lat for lat in concurrent_latencies if isinstance(lat, float)]
        failed = len(concurrent_latencies) - len(successful)

        model_result["concurrent_throughput"] = {
            "concurrency": concurrency,
            "requests_per_second": round(len(successful) / batch_elapsed_s, 2) if batch_elapsed_s > 0 else 0,
            "total_requests": len(concurrent_latencies),
            "successful": len(successful),
            "failed": failed,
            "avg_latency_ms": round(sum(successful) / max(len(successful), 1), 1),
            "wall_clock_s": round(batch_elapsed_s, 2),
        }
        logger.info("concurrent_throughput_complete", **model_result["concurrent_throughput"])
        await service.close()

    return model_result


async def run_benchmark(
    models: list[str],
    lang_pair: str,
    data_dir: Path,
    output_dir: Path,
    ollama_url: str = "http://thomas-pc:11434/v1",
    context_sizes: list[int] | None = None,
    concurrency: int = 1,
) -> None:
    """Run translation benchmark with one or more models and a language pair."""
    setup_logging(service_name="translation-benchmark", log_format="dev")

    context_sizes = context_sizes or [0, 3, 5]

    # Load test data
    source_file = data_dir / f"{lang_pair}.source"
    reference_file = data_dir / f"{lang_pair}.reference"

    if not source_file.exists() or not reference_file.exists():
        logger.error("no_test_data", data_dir=str(data_dir), lang_pair=lang_pair)
        raise FileNotFoundError(
            f"Missing benchmark data for {lang_pair}: "
            f"{source_file} and/or {reference_file}"
        )

    sources = source_file.read_text().strip().split("\n")
    references = reference_file.read_text().strip().split("\n")

    if len(sources) != len(references):
        logger.error("data_mismatch", sources=len(sources), references=len(references))
        raise ValueError(
            f"Benchmark data mismatch for {lang_pair}: {len(sources)} sources vs {len(references)} references"
        )

    results = {
        "lang_pair": lang_pair,
        "ollama_url": ollama_url,
        "system_info": get_system_info(),
        "models": [],
    }

    for model in models:
        logger.info("benchmark_starting", model=model, samples=len(sources))
        model_result = await run_single_model_benchmark(
            model, lang_pair, sources, references,
            ollama_url, context_sizes, concurrency,
        )
        results["models"].append(model_result)

    # Print comparison table if multiple models
    if len(models) > 1:
        print("\n=== Model Comparison ===")
        print(f"{'Model':<25} {'Ctx':>4} {'BLEU':>8} {'COMET':>8} {'Avg ms':>8} {'P95 ms':>8}")
        print("-" * 70)
        for m in results["models"]:
            for run in m["runs"]:
                comet_str = f"{run['comet']:.4f}" if run.get("comet") is not None else "N/A"
                print(
                    f"{m['model']:<25} {run['context_window_size']:>4} "
                    f"{run['bleu']:>8.4f} {comet_str:>8} "
                    f"{run['avg_latency_ms']:>8.1f} {run['p95_latency_ms']:>8.1f}"
                )
        print()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_slug = "_vs_".join(m.replace(":", "_") for m in models)
    out_file = output_dir / f"{model_slug}_{lang_pair}_{ts}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("benchmark_complete", output=str(out_file))


def main():
    parser = argparse.ArgumentParser(description="Translation Benchmark")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Single Ollama model name (e.g. qwen3.5:7b)")
    model_group.add_argument(
        "--models",
        help="Comma-separated model names for comparison (e.g. qwen3.5:7b,llama3.1:8b)",
    )
    parser.add_argument("--lang-pair", required=True, help="Language pair (e.g. zh-en)")
    parser.add_argument("--data-dir", type=Path, default=Path("tools/translation_benchmark/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("tools/translation_benchmark/results"))
    parser.add_argument("--ollama-url", default="http://thomas-pc:11434/v1")
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[0, 3, 5])
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent requests for throughput measurement (default: 1, no concurrency test)",
    )
    args = parser.parse_args()

    models = args.models.split(",") if args.models else [args.model]

    asyncio.run(run_benchmark(
        models, args.lang_pair, args.data_dir, args.output_dir,
        args.ollama_url, args.context_sizes, args.concurrency,
    ))


if __name__ == "__main__":
    main()
