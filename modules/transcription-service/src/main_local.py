"""Local macOS transcription service entry point.

Uses MLX Whisper on Apple Silicon (Metal GPU) instead of CUDA/CTranslate2.
Auto-detects the best available backend:
  1. MLX (Apple Silicon) — fastest on Mac
  2. faster-whisper CPU — fallback

Usage:
  uv run python modules/transcription-service/src/main_local.py
  uv run python modules/transcription-service/src/main_local.py --model medium --port 5001
"""
import argparse
import platform
import sys
from pathlib import Path

import uvicorn
from livetranslate_common.logging import setup_logging


def _detect_backend() -> str:
    """Detect the best available backend for this machine."""
    system = platform.system()
    machine = platform.machine()

    # Check if vllm-mlx server is running
    import os
    vllm_url = os.getenv("VLLM_MLX_URL")
    if vllm_url:
        try:
            import httpx
            resp = httpx.get(f"{vllm_url}/health", timeout=2)
            if resp.status_code == 200:
                return "vllm"
        except Exception:
            pass

    # Apple Silicon
    if system == "Darwin" and machine == "arm64":
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except ImportError:
            pass

    # CUDA GPU
    try:
        import torch
        if torch.cuda.is_available():
            return "whisper"
    except ImportError:
        pass

    # CPU fallback (faster-whisper with CTranslate2)
    try:
        import faster_whisper  # noqa: F401
        return "whisper"
    except ImportError:
        pass

    print("ERROR: No transcription backend available.")
    print("Install one of: mlx-whisper (Mac), faster-whisper (CPU/CUDA)")
    sys.exit(1)


def _generate_registry(backend: str, model: str, compute_type: str) -> dict:
    """Generate a registry config dict for the detected backend."""
    backends = {}
    if backend == "mlx":
        backends["mlx"] = {
            "module": "backends.mlx_whisper",
            "class": "MLXWhisperBackend",
        }
    elif backend == "vllm":
        backends["vllm"] = {
            "module": "backends.vllm_whisper",
            "class": "VLLMWhisperBackend",
        }
    else:
        backends["whisper"] = {
            "module": "backends.whisper",
            "class": "WhisperBackend",
        }

    return {
        "version": 1,
        "backends": backends,
        "vram_budget_mb": 16000,
        "language_routing": {
            "en": {
                "backend": backend,
                "model": model,
                "compute_type": compute_type,
                "chunk_duration_s": 5.0,
                "stride_s": 4.5,
                "overlap_s": 0.5,
                "vad_threshold": 0.5,
                "beam_size": 1,
                "prebuffer_s": 3.0 if backend in ("mlx", "vllm") else 0.3,
                "batch_profile": "realtime",
            },
            "*": {
                "backend": backend,
                "model": model,
                "compute_type": compute_type,
                "chunk_duration_s": 5.0,
                "stride_s": 4.5,
                "overlap_s": 0.5,
                "vad_threshold": 0.5,
                "beam_size": 1,
                "prebuffer_s": 3.0 if backend in ("mlx", "vllm") else 0.3,
                "batch_profile": "realtime",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Transcription Service (Local Mac)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--model", default="medium", help="Whisper model: tiny, base, small, medium")
    parser.add_argument("--backend", default=None, help="Force backend: mlx, whisper (auto-detect if omitted)")
    parser.add_argument("--compute-type", default=None, help="Compute type: float16, int8, float32")
    parser.add_argument("--log-format", default="dev", choices=["dev", "json"])
    args = parser.parse_args()

    setup_logging(service_name="transcription", log_format=args.log_format)

    backend = args.backend or _detect_backend()
    compute_type = args.compute_type or ("float16" if backend == "mlx" else "int8")

    print(f"Backend: {backend} | Model: {args.model} | Compute: {compute_type}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    # Write temp registry and start
    import tempfile
    import yaml

    registry_data = _generate_registry(backend, args.model, compute_type)
    registry_file = Path(tempfile.mktemp(suffix=".yaml", prefix="registry_local_"))
    registry_file.write_text(yaml.dump(registry_data))

    from api import create_app

    app = create_app(registry_path=registry_file)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        registry_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
