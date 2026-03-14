"""CLI benchmark runner for transcription backends.

Usage: uv run python -m benchmarks.run --backend whisper --language en
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from livetranslate_common.logging import setup_logging, get_logger

from benchmarks.metrics import word_error_rate, character_error_rate

logger = get_logger()

CJK_LANGUAGES = {"zh", "ja", "ko"}


def get_system_info() -> dict:
    """Collect system info for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["cuda_available"] = False

    for pkg in ["faster_whisper", "ctranslate2", "torch", "numpy"]:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return info


def measure_peak_vram() -> int:
    """Return peak VRAM usage in MB since last reset."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() // (1024 * 1024)
    except ImportError:
        pass
    return 0


def reset_vram_tracking():
    """Reset CUDA peak memory tracking."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


async def load_backend(backend_name: str, model_name: str, compute_type: str, device: str):
    """Load a transcription backend for benchmarking."""
    if backend_name == "whisper":
        from backends.whisper import WhisperBackend
        backend = WhisperBackend(
            model_name=model_name, compute_type=compute_type, device=device,
        )
        await backend.load_model(model_name, device)
        await backend.warmup()
        return backend
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def run_benchmark(backend_name: str, language: str, data_dir: Path, output_dir: Path):
    """Run benchmark on test data pairs (audio.wav + reference.txt)."""
    import asyncio
    import soundfile as sf

    setup_logging(service_name="benchmark")

    system_info = get_system_info()

    results = {
        "backend": backend_name,
        "language": language,
        "system_info": system_info,
        "samples": [],
        "aggregate": {},
    }

    test_files = sorted(data_dir.glob("*.wav"))
    if not test_files:
        logger.warning("no_test_data", data_dir=str(data_dir))
        return

    model_name = "large-v3-turbo"
    compute_type = "float16"
    device = "cuda"

    loop = asyncio.new_event_loop()
    backend = loop.run_until_complete(load_backend(backend_name, model_name, compute_type, device))

    error_metric = character_error_rate if language in CJK_LANGUAGES else word_error_rate
    error_metric_name = "cer" if language in CJK_LANGUAGES else "wer"

    all_errors = []
    all_latencies = []
    all_ttft = []

    for audio_path in test_files:
        ref_path = audio_path.with_suffix(".txt")
        if not ref_path.exists():
            continue

        reference = ref_path.read_text().strip()
        audio_data, sr = sf.read(str(audio_path))

        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)

        logger.info("benchmarking", file=audio_path.name, duration_s=len(audio_data) / 16000)

        reset_vram_tracking()

        t0 = time.perf_counter()
        result = loop.run_until_complete(
            backend.transcribe(audio_data, language=language, beam_size=5, batch_profile="batch")
        )
        t1 = time.perf_counter()

        inference_time_s = t1 - t0
        hypothesis = result.text.strip()
        error_rate = error_metric(reference, hypothesis)
        peak_vram = measure_peak_vram()

        reset_vram_tracking()
        t0_stream = time.perf_counter()
        first_token_time = None
        async def _stream_measure():
            nonlocal first_token_time
            async for partial in backend.transcribe_stream(audio_data, language=language):
                if partial.text.strip() and first_token_time is None:
                    first_token_time = time.perf_counter() - t0_stream
                    break
        loop.run_until_complete(_stream_measure())

        sample_result = {
            "file": audio_path.name,
            "reference": reference,
            "hypothesis": hypothesis,
            error_metric_name: round(error_rate, 4),
            "inference_time_s": round(inference_time_s, 4),
            "time_to_first_token_s": round(first_token_time, 4) if first_token_time else None,
            "peak_vram_mb": peak_vram,
            "audio_duration_s": round(len(audio_data) / 16000, 2),
            "rtf": round(inference_time_s / (len(audio_data) / 16000), 4),
        }
        results["samples"].append(sample_result)
        all_errors.append(error_rate)
        all_latencies.append(inference_time_s)
        if first_token_time:
            all_ttft.append(first_token_time)

    if all_errors:
        results["aggregate"] = {
            f"mean_{error_metric_name}": round(sum(all_errors) / len(all_errors), 4),
            "mean_inference_time_s": round(sum(all_latencies) / len(all_latencies), 4),
            "mean_ttft_s": round(sum(all_ttft) / len(all_ttft), 4) if all_ttft else None,
            "total_samples": len(all_errors),
        }

    loop.run_until_complete(backend.unload_model())
    loop.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"{backend_name}_{language}_{ts}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("benchmark_complete", output=str(out_file), samples=len(results["samples"]))


def main():
    parser = argparse.ArgumentParser(description="Transcription Benchmark")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    run_benchmark(args.backend, args.language, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
