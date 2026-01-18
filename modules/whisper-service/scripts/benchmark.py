#!/usr/bin/env python3
"""
Whisper Service Benchmarking Tool

Measures performance characteristics of key components.

Usage:
    python scripts/benchmark.py --component vad
    python scripts/benchmark.py --component lid
    python scripts/benchmark.py --component all
    python scripts/benchmark.py --audio test_audio.wav
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from service_config import VADConfig
from vad_detector import SileroVAD


def generate_test_audio(duration_seconds: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic test audio"""
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples)

    # Generate speech-like signal (sum of sine waves with noise)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.random.randn(samples)
    )

    return audio.astype(np.float32)


def benchmark_vad(iterations: int = 100, chunk_size: int = 8000):
    """Benchmark VAD performance"""
    print(f"\n=== Benchmarking VAD (iterations={iterations}) ===")

    # Initialize VAD
    config = VADConfig.from_env()
    vad = SileroVAD(
        threshold=config.threshold,
        sampling_rate=config.sampling_rate,
        min_silence_duration_ms=config.min_silence_duration_ms,
    )

    # Generate test audio
    audio = generate_test_audio(duration_seconds=0.5)

    # Warmup
    for _ in range(10):
        vad.check_speech(audio[:chunk_size])

    # Benchmark
    timings = []
    for _ in range(iterations):
        start = time.time()
        vad.check_speech(audio[:chunk_size])
        elapsed = (time.time() - start) * 1000  # ms
        timings.append(elapsed)

    # Statistics
    avg = sum(timings) / len(timings)
    min_time = min(timings)
    max_time = max(timings)
    p95 = sorted(timings)[int(len(timings) * 0.95)]

    print("✅ VAD Performance:")
    print(f"   - Average: {avg:.2f}ms")
    print(f"   - Min: {min_time:.2f}ms")
    print(f"   - Max: {max_time:.2f}ms")
    print(f"   - P95: {p95:.2f}ms")
    print(f"   - Chunk size: {chunk_size} samples ({chunk_size/16000*1000:.1f}ms)")

    # Check if performance is acceptable
    target_latency = 50  # ms
    if avg < target_latency:
        print(f"   ✅ Performance target met (< {target_latency}ms)")
    else:
        print(f"   ⚠️  Performance below target (target: < {target_latency}ms)")

    return timings


def benchmark_audio_processing(iterations: int = 100):
    """Benchmark basic audio processing operations"""
    print(f"\n=== Benchmarking Audio Processing (iterations={iterations}) ===")

    audio = generate_test_audio(duration_seconds=1.0)

    operations = {
        "RMS Calculation": lambda x: np.sqrt(np.mean(x**2)),
        "Max Amplitude": lambda x: np.max(np.abs(x)),
        "Normalization": lambda x: x / np.max(np.abs(x)),
        "Concatenation": lambda x: np.concatenate([x, x]),
    }

    for op_name, op_func in operations.items():
        timings = []

        # Warmup
        for _ in range(10):
            op_func(audio)

        # Benchmark
        for _ in range(iterations):
            start = time.time()
            op_func(audio)
            elapsed = (time.time() - start) * 1000  # ms
            timings.append(elapsed)

        avg = sum(timings) / len(timings)
        print(f"   {op_name}: {avg:.3f}ms")


def benchmark_buffer_operations(iterations: int = 100):
    """Benchmark buffer operations"""
    print(f"\n=== Benchmarking Buffer Operations (iterations={iterations}) ===")

    chunk_size = 8000
    buffer = np.array([], dtype=np.float32)


    # Benchmark append
    timings_append = []
    for _ in range(iterations):
        chunk = np.random.randn(chunk_size).astype(np.float32)
        start = time.time()
        buffer = np.concatenate([buffer, chunk])
        elapsed = (time.time() - start) * 1000
        timings_append.append(elapsed)

    avg_append = sum(timings_append) / len(timings_append)
    print(f"   Buffer Append: {avg_append:.3f}ms")

    # Benchmark clear
    timings_clear = []
    for _ in range(iterations):
        start = time.time()
        buffer = np.array([], dtype=np.float32)
        elapsed = (time.time() - start) * 1000
        timings_clear.append(elapsed)

    avg_clear = sum(timings_clear) / len(timings_clear)
    print(f"   Buffer Clear: {avg_clear:.3f}ms")


def load_audio_file(file_path: str) -> np.ndarray:
    """Load audio file for testing"""
    try:
        import soundfile as sf

        audio, sr = sf.read(file_path)

        # Resample if needed
        if sr != 16000:
            print(f"⚠️  Resampling from {sr}Hz to 16000Hz")
            from scipy.signal import resample

            audio = resample(audio, int(len(audio) * 16000 / sr))

        return audio.astype(np.float32)
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return None


def benchmark_real_audio(audio_file: str):
    """Benchmark with real audio file"""
    print(f"\n=== Benchmarking with Real Audio: {audio_file} ===")

    audio = load_audio_file(audio_file)
    if audio is None:
        return

    duration = len(audio) / 16000
    print(f"Audio: {len(audio)} samples, {duration:.2f}s")

    # Initialize VAD
    config = VADConfig.from_env()
    vad = SileroVAD(
        threshold=config.threshold,
        sampling_rate=config.sampling_rate,
        min_silence_duration_ms=config.min_silence_duration_ms,
    )

    # Process in chunks
    chunk_size = 8000  # 500ms chunks
    total_chunks = len(audio) // chunk_size

    print(f"Processing {total_chunks} chunks...")

    start_time = time.time()
    speech_events = []

    for i in range(total_chunks):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        result = vad.check_speech(chunk)

        if result:
            speech_events.append(result)

    elapsed = time.time() - start_time

    print("✅ Processing Complete:")
    print(f"   - Total Time: {elapsed:.2f}s")
    print(f"   - Real-time Factor: {duration/elapsed:.2f}x")
    print(f"   - Speech Events: {len(speech_events)}")

    if duration / elapsed > 1.0:
        print("   ✅ Faster than real-time")
    else:
        print("   ⚠️  Slower than real-time")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper Service Components")
    parser.add_argument(
        "--component",
        type=str,
        default="all",
        choices=["vad", "audio", "buffer", "all"],
        help="Component to benchmark",
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations for benchmarks"
    )
    parser.add_argument("--audio", type=str, help="Audio file to benchmark with")

    args = parser.parse_args()

    print("=" * 60)
    print("Whisper Service Benchmark Tool")
    print("=" * 60)

    if args.audio:
        benchmark_real_audio(args.audio)
    else:
        if args.component in ["vad", "all"]:
            benchmark_vad(iterations=args.iterations)

        if args.component in ["audio", "all"]:
            benchmark_audio_processing(iterations=args.iterations)

        if args.component in ["buffer", "all"]:
            benchmark_buffer_operations(iterations=args.iterations)

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
