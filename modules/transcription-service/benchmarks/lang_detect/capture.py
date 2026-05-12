"""Stage 1 — capture a Whisper LID trace from a WAV file.

Reads a 48 kHz mono WAV (the loopback-page sample rate), resamples to 16 kHz,
chunks it with a sliding window that approximates production VAC stride/overlap,
sends each chunk to vllm-mlx /v1/audio/transcriptions, and writes a JSONL trace
suitable for Stage 2 replay.

Run once per fixture and commit the JSONL — replay is then fully offline.

CLI:
    uv run python -m benchmarks.lang_detect.capture \\
        --wav modules/dashboard-service/tests/fixtures/lang_detect_zh_short_48k.wav \\
        --out modules/transcription-service/benchmarks/lang_detect/fixtures/zh_short_60s.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly  # type: ignore

from .types import FrameTrace


DEFAULT_VLLM_URL = "http://localhost:8005"
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
TARGET_SR = 16_000

# Sliding window — close to production VAC defaults (stride=4.5s, overlap=0.5s).
# Made configurable so users can capture under whatever chunking they want to
# sweep. For LID purposes, frame cadence matters less than frame *content*.
DEFAULT_STRIDE_S = 3.0
DEFAULT_OVERLAP_S = 0.5


def load_and_resample(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV (any rate), return (mono float32 @ 16 kHz, original rate)."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        # resample_poly with rational up/down keeps phase clean for speech.
        g = math.gcd(int(sr), TARGET_SR)
        up, down = TARGET_SR // g, int(sr) // g
        audio = resample_poly(audio, up, down).astype(np.float32)
    return audio, sr


def chunks(audio: np.ndarray, stride_s: float, overlap_s: float) -> list[tuple[float, np.ndarray]]:
    """Yield (t_ms_start, chunk_samples) sliding by stride with overlap retention."""
    stride = int(stride_s * TARGET_SR)
    overlap = int(overlap_s * TARGET_SR)
    window = stride + overlap
    out: list[tuple[float, np.ndarray]] = []
    pos = 0
    while pos < len(audio):
        end = min(pos + window, len(audio))
        chunk = audio[pos:end]
        if len(chunk) < TARGET_SR * 0.5:  # skip <0.5s tail
            break
        out.append((pos / TARGET_SR * 1000.0, chunk))
        pos += stride
    return out


def chunk_to_wav_bytes(chunk: np.ndarray) -> bytes:
    """Encode a float32 16 kHz chunk to WAV in memory."""
    buf = io.BytesIO()
    sf.write(buf, chunk, TARGET_SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


async def transcribe_chunk(
    client: httpx.AsyncClient,
    chunk: np.ndarray,
    model: str,
) -> dict:
    """POST one chunk to vllm-mlx; return the verbose_json response."""
    wav_bytes = chunk_to_wav_bytes(chunk)
    files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
    data = {"model": model, "response_format": "verbose_json"}
    r = await client.post("/v1/audio/transcriptions", files=files, data=data)
    r.raise_for_status()
    return r.json()


def to_frame_trace(t_ms: float, chunk: np.ndarray, response: dict) -> FrameTrace:
    """Convert one Whisper response into a FrameTrace."""
    segs = response.get("segments", [])
    no_speech = max((s.get("no_speech_prob", 0.0) for s in segs), default=None)
    # Whisper API returns avg_logprob; convert to a 0..1 confidence the same
    # way VLLMWhisperBackend does in production.
    if segs:
        confs = [max(0.0, min(1.0, math.exp(s.get("avg_logprob", -1.0)))) for s in segs]
        confidence = float(np.mean(confs))
    else:
        confidence = 0.5
    rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0

    return FrameTrace(
        t_ms=t_ms,
        chunk_dur_s=len(chunk) / TARGET_SR,
        language=response.get("language", "en"),
        confidence=confidence,
        text=response.get("text", "").strip(),
        no_speech_prob=no_speech,
        audio_rms=rms,
    )


async def capture(wav_path: Path, out_path: Path, vllm_url: str, model: str,
                  stride_s: float, overlap_s: float) -> int:
    """Run capture end-to-end. Returns number of frames written."""
    audio, _ = load_and_resample(wav_path)
    chs = chunks(audio, stride_s, overlap_s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    async with httpx.AsyncClient(base_url=vllm_url, timeout=60.0) as client:
        with out_path.open("w") as fout:
            for i, (t_ms, chunk) in enumerate(chs):
                resp = await transcribe_chunk(client, chunk, model)
                frame = to_frame_trace(t_ms, chunk, resp)
                fout.write(json.dumps(asdict(frame), ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                if (i + 1) % 5 == 0 or i == len(chs) - 1:
                    print(f"  [{i+1}/{len(chs)}] t={t_ms/1000:.1f}s "
                          f"lang={frame.language} conf={frame.confidence:.2f} "
                          f"text={frame.text[:40]!r}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)
    ap.add_argument("--model", default=DEFAULT_WHISPER_MODEL)
    ap.add_argument("--stride-s", type=float, default=DEFAULT_STRIDE_S)
    ap.add_argument("--overlap-s", type=float, default=DEFAULT_OVERLAP_S)
    args = ap.parse_args()

    if not args.wav.exists():
        print(f"WAV not found: {args.wav}")
        return 1

    print(f"Capturing {args.wav} → {args.out}")
    print(f"  vllm: {args.vllm_url} model: {args.model}")
    print(f"  stride: {args.stride_s}s overlap: {args.overlap_s}s")

    n = asyncio.run(capture(
        args.wav, args.out, args.vllm_url, args.model, args.stride_s, args.overlap_s
    ))
    print(f"Wrote {n} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
