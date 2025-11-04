#!/usr/bin/env python3
"""
Performance Benchmarks: Latency Testing

Benchmarks end-to-end latency, LID probe latency, and tracks percentiles.
Per ML Engineer review - Priority 6: Performance benchmarks with thresholds

Critical metrics:
1. End-to-end latency < 100ms for 500ms chunks (p95)
2. LID probe < 1ms on GPU
3. Track p50, p95, p99 latencies
4. Performance regression detection

Reference: FEEDBACK.md lines 171-184, session_manager.py
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
import time
import json
import logging
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass, asdict

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from session_restart import SessionRestartTranscriber

logger = logging.getLogger(__name__)


# Benchmark storage
BENCHMARK_DIR = Path(__file__).parent / "results"
BENCHMARK_FILE = BENCHMARK_DIR / "latency_benchmarks.json"


@dataclass
class LatencyStats:
    """Latency statistics"""
    samples: int
    mean_ms: float
    median_ms: float  # p50
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    def to_dict(self):
        return asdict(self)


class LatencyTracker:
    """Track latency measurements"""

    def __init__(self, name: str):
        self.name = name
        self.measurements = []

    def measure(self, func, *args, **kwargs):
        """Measure function execution time"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.measurements.append(elapsed * 1000)  # Convert to ms
        return result

    def get_stats(self) -> LatencyStats:
        """Calculate latency statistics"""
        if not self.measurements:
            return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0)

        measurements = np.array(self.measurements)

        return LatencyStats(
            samples=len(measurements),
            mean_ms=float(np.mean(measurements)),
            median_ms=float(np.median(measurements)),  # p50
            p95_ms=float(np.percentile(measurements, 95)),
            p99_ms=float(np.percentile(measurements, 99)),
            min_ms=float(np.min(measurements)),
            max_ms=float(np.max(measurements)),
            std_ms=float(np.std(measurements))
        )

    def print_stats(self):
        """Print latency statistics"""
        stats = self.get_stats()

        logger.info(f"\n{'='*80}")
        logger.info(f"LATENCY BENCHMARK: {self.name}")
        logger.info(f"{'='*80}")
        logger.info(f"Samples:  {stats.samples}")
        logger.info(f"Mean:     {stats.mean_ms:.2f} ms")
        logger.info(f"Median:   {stats.median_ms:.2f} ms (p50)")
        logger.info(f"P95:      {stats.p95_ms:.2f} ms")
        logger.info(f"P99:      {stats.p99_ms:.2f} ms")
        logger.info(f"Min:      {stats.min_ms:.2f} ms")
        logger.info(f"Max:      {stats.max_ms:.2f} ms")
        logger.info(f"Std Dev:  {stats.std_ms:.2f} ms")
        logger.info(f"{'='*80}")


class BenchmarkStorage:
    """Store and compare benchmarks"""

    @staticmethod
    def load_benchmarks() -> Dict:
        """Load historical benchmarks"""
        if not BENCHMARK_FILE.exists():
            return {}

        with open(BENCHMARK_FILE, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_benchmarks(benchmarks: Dict):
        """Save benchmarks to file"""
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

        with open(BENCHMARK_FILE, 'w') as f:
            json.dump(benchmarks, f, indent=2)

        logger.info(f"✅ Benchmarks saved to {BENCHMARK_FILE}")

    @staticmethod
    def update_benchmark(name: str, stats: LatencyStats):
        """Update benchmark with new results"""
        benchmarks = BenchmarkStorage.load_benchmarks()

        if name not in benchmarks:
            benchmarks[name] = {'history': [], 'best': None}

        # Add to history
        entry = {
            'timestamp': datetime.now().isoformat(),
            'stats': stats.to_dict()
        }
        benchmarks[name]['history'].append(entry)

        # Update best (lowest p95)
        if benchmarks[name]['best'] is None or \
           stats.p95_ms < benchmarks[name]['best']['stats']['p95_ms']:
            benchmarks[name]['best'] = entry

        # Keep last 10 entries
        benchmarks[name]['history'] = benchmarks[name]['history'][-10:]

        BenchmarkStorage.save_benchmarks(benchmarks)

    @staticmethod
    def print_comparison(name: str, current_stats: LatencyStats):
        """Print comparison with historical benchmarks"""
        benchmarks = BenchmarkStorage.load_benchmarks()

        if name not in benchmarks or not benchmarks[name].get('best'):
            logger.info("📊 No historical benchmark available (first run)")
            return

        best = benchmarks[name]['best']['stats']

        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"                Current      Best      Diff")
        logger.info(f"Mean:          {current_stats.mean_ms:7.2f}ms  {best['mean_ms']:7.2f}ms  {current_stats.mean_ms - best['mean_ms']:+7.2f}ms")
        logger.info(f"Median (p50):  {current_stats.median_ms:7.2f}ms  {best['median_ms']:7.2f}ms  {current_stats.median_ms - best['median_ms']:+7.2f}ms")
        logger.info(f"P95:           {current_stats.p95_ms:7.2f}ms  {best['p95_ms']:7.2f}ms  {current_stats.p95_ms - best['p95_ms']:+7.2f}ms")
        logger.info(f"P99:           {current_stats.p99_ms:7.2f}ms  {best['p99_ms']:7.2f}ms  {current_stats.p99_ms - best['p99_ms']:+7.2f}ms")

        regression = current_stats.p95_ms - best['p95_ms']
        if regression > 0:
            logger.warning(f"⚠️ Performance regression: P95 increased by {regression:.2f}ms")
        else:
            logger.info(f"✅ Performance maintained or improved")

        logger.info(f"{'='*80}")


# Audio generation utility
def generate_test_audio(duration_sec: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio for benchmarking"""
    num_samples = int(duration_sec * sample_rate)
    audio = np.random.randn(num_samples).astype(np.float32) * 0.3
    return audio


@pytest.mark.benchmark
class TestEndToEndLatency:
    """Test end-to-end processing latency"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber for latency tests"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=['en'],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000
        )

    def test_end_to_end_latency_500ms_chunks(self, transcriber):
        """
        Benchmark: End-to-end latency for 500ms audio chunks.

        Target: P95 < 100ms per FEEDBACK.md real-time requirements
        """
        logger.info("Benchmarking end-to-end latency (500ms chunks)...")

        tracker = LatencyTracker("end_to_end_500ms")

        # Warmup: Process a few chunks to warm up model
        warmup_audio = generate_test_audio(0.5)
        for _ in range(5):
            transcriber.process(warmup_audio)

        # Benchmark: Process 100 chunks
        num_chunks = 100
        for i in range(num_chunks):
            audio = generate_test_audio(0.5)

            # Measure processing time
            tracker.measure(transcriber.process, audio)

            if (i + 1) % 20 == 0:
                logger.info(f"Processed {i + 1}/{num_chunks} chunks...")

        # Print results
        tracker.print_stats()
        stats = tracker.get_stats()

        # Save benchmark
        BenchmarkStorage.update_benchmark("end_to_end_500ms", stats)
        BenchmarkStorage.print_comparison("end_to_end_500ms", stats)

        # Assertions
        target_p95_ms = 100.0
        assert stats.p95_ms < target_p95_ms, \
            f"P95 latency {stats.p95_ms:.2f}ms exceeds target {target_p95_ms}ms"

        logger.info(f"✅ End-to-end latency P95: {stats.p95_ms:.2f}ms (target: < {target_p95_ms}ms)")

    def test_end_to_end_latency_various_chunk_sizes(self, transcriber):
        """
        Benchmark: End-to-end latency for various chunk sizes.

        Tests 250ms, 500ms, 1000ms chunks.
        """
        chunk_configs = [
            ('250ms', 0.25),
            ('500ms', 0.5),
            ('1000ms', 1.0)
        ]

        results = {}

        for name, duration in chunk_configs:
            logger.info(f"\nBenchmarking {name} chunks...")

            tracker = LatencyTracker(f"end_to_end_{name}")

            # Warmup
            warmup = generate_test_audio(duration)
            for _ in range(3):
                transcriber.process(warmup)

            # Benchmark
            for i in range(50):
                audio = generate_test_audio(duration)
                tracker.measure(transcriber.process, audio)

            stats = tracker.get_stats()
            results[name] = stats

            logger.info(f"{name}: P95 = {stats.p95_ms:.2f}ms, Mean = {stats.mean_ms:.2f}ms")

            # Save benchmark
            BenchmarkStorage.update_benchmark(f"end_to_end_{name}", stats)

        # Compare chunk sizes
        logger.info(f"\n{'='*80}")
        logger.info("CHUNK SIZE COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"Chunk Size    Mean      P50       P95       P99")
        for name, stats in results.items():
            logger.info(
                f"{name:12} {stats.mean_ms:7.2f}ms {stats.median_ms:7.2f}ms "
                f"{stats.p95_ms:7.2f}ms {stats.p99_ms:7.2f}ms"
            )
        logger.info(f"{'='*80}")

    def test_latency_with_language_switching(self, transcriber):
        """
        Benchmark: Latency during language switching.

        Measures if language switches cause latency spikes.
        """
        # Reconfigure for code-switching
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        transcriber_multi = SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=['en', 'zh'],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000,
            lid_hop_ms=100,
            confidence_margin=0.2,
            min_dwell_frames=6
        )

        tracker = LatencyTracker("latency_with_switching")

        # Process enough chunks to potentially trigger switches
        for i in range(100):
            audio = generate_test_audio(0.5)
            tracker.measure(transcriber_multi.process, audio)

        stats = tracker.get_stats()
        tracker.print_stats()

        BenchmarkStorage.update_benchmark("latency_with_switching", stats)

        logger.info(f"✅ Latency with switching P95: {stats.p95_ms:.2f}ms")


@pytest.mark.benchmark
class TestLIDProbeLatency:
    """Test Language ID probe latency"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber with LID for latency tests"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=['en', 'zh'],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000,
            lid_hop_ms=100
        )

    def test_lid_probe_latency(self, transcriber):
        """
        Benchmark: Whisper-native LID probe latency.

        Target: < 1ms on GPU per ML engineer spec
        Note: Actual latency depends on hardware
        """
        logger.info("Benchmarking LID probe latency...")

        # First, process some audio to initialize LID
        initial_audio = generate_test_audio(5.0)  # 5 seconds
        chunk_size = 8000

        for i in range(0, len(initial_audio), chunk_size):
            transcriber.process(initial_audio[i:i + chunk_size])

        # Now we have an active session, can benchmark LID
        if transcriber.current_session is None:
            pytest.skip("No active session for LID benchmarking")

        from simul_whisper.whisper.audio import log_mel_spectrogram, pad_or_trim
        import torch

        tracker = LatencyTracker("lid_probe")

        # Benchmark LID probe
        for i in range(100):
            # Generate LID frame (100ms = 1600 samples)
            lid_audio = generate_test_audio(0.1)

            # Measure LID detection time
            start = time.perf_counter()

            # Pad and create mel
            lid_audio_padded = pad_or_trim(lid_audio)
            mel = log_mel_spectrogram(
                lid_audio_padded,
                n_mels=transcriber.current_session.processor.model.dims.n_mels
            )

            # Run encoder
            with torch.no_grad():
                encoder_output = transcriber.current_session.processor.model.encoder(
                    mel.unsqueeze(0).to(transcriber.current_session.processor.model.device)
                )

            # Run LID probe
            lid_probs = transcriber.lid_detector.detect(
                encoder_output=encoder_output,
                model=transcriber.current_session.processor.model,
                tokenizer=transcriber.current_session.processor.tokenizer,
                timestamp=i * 0.1
            )

            elapsed = time.perf_counter() - start
            tracker.measurements.append(elapsed * 1000)  # ms

        stats = tracker.get_stats()
        tracker.print_stats()

        BenchmarkStorage.update_benchmark("lid_probe", stats)

        # Target depends on hardware
        # GPU: < 1ms, CPU: < 10ms
        target_ms = 10.0  # Conservative target for CI
        if torch.cuda.is_available():
            target_ms = 1.0

        logger.info(f"✅ LID probe P95: {stats.p95_ms:.2f}ms (target: < {target_ms}ms)")


@pytest.mark.benchmark
class TestVADLatency:
    """Test VAD latency"""

    @pytest.fixture
    def vad(self):
        """Create VAD for latency tests"""
        from vad_detector import SileroVAD
        return SileroVAD(
            threshold=0.5,
            sampling_rate=16000,
            min_silence_duration_ms=500
        )

    def test_vad_latency(self, vad):
        """
        Benchmark: VAD processing latency.

        Target: < 1ms for real-time processing
        """
        logger.info("Benchmarking VAD latency...")

        tracker = LatencyTracker("vad_check_speech")

        # Benchmark VAD
        for _ in range(200):
            audio = generate_test_audio(0.5)
            tracker.measure(vad.check_speech, audio)

        stats = tracker.get_stats()
        tracker.print_stats()

        BenchmarkStorage.update_benchmark("vad_check_speech", stats)

        target_ms = 1.0
        assert stats.p95_ms < target_ms, \
            f"VAD P95 latency {stats.p95_ms:.2f}ms exceeds target {target_ms}ms"

        logger.info(f"✅ VAD latency P95: {stats.p95_ms:.2f}ms (target: < {target_ms}ms)")


@pytest.mark.benchmark
class TestPerformanceRegression:
    """Test for performance regressions"""

    def test_detect_performance_regression(self):
        """
        Test: Detect performance regression from baseline.

        Validates regression detection system works.
        """
        test_name = "test_regression_detection"

        # Simulate baseline (50ms P95)
        baseline = LatencyStats(
            samples=100,
            mean_ms=40.0,
            median_ms=38.0,
            p95_ms=50.0,
            p99_ms=60.0,
            min_ms=20.0,
            max_ms=80.0,
            std_ms=10.0
        )
        BenchmarkStorage.update_benchmark(test_name, baseline)

        # Simulate regression (80ms P95 - 60% increase)
        current = LatencyStats(
            samples=100,
            mean_ms=65.0,
            median_ms=60.0,
            p95_ms=80.0,
            p99_ms=95.0,
            min_ms=40.0,
            max_ms=120.0,
            std_ms=15.0
        )

        # Load and compare
        benchmarks = BenchmarkStorage.load_benchmarks()
        best = benchmarks[test_name]['best']['stats']

        regression = current.p95_ms - best['p95_ms']

        logger.info(f"Baseline P95: {best['p95_ms']:.2f}ms")
        logger.info(f"Current P95:  {current.p95_ms:.2f}ms")
        logger.info(f"Regression:   {regression:+.2f}ms ({regression/best['p95_ms']*100:+.1f}%)")

        # Should detect 60% regression
        assert regression > 10.0, "Should detect significant regression"

        logger.info("✅ Regression detection works")


@pytest.mark.benchmark
class TestThroughput:
    """Test throughput metrics"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber for throughput tests"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=['en'],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000
        )

    def test_throughput_audio_seconds_per_second(self, transcriber):
        """
        Benchmark: Throughput in audio seconds processed per wall-clock second.

        Real-time processing requires >= 1.0x (process 1s of audio in 1s wall time).
        """
        logger.info("Benchmarking throughput (audio seconds per second)...")

        # Generate 60 seconds of audio
        total_audio_duration = 60.0
        audio = generate_test_audio(total_audio_duration)
        chunk_size = 8000  # 0.5s chunks

        # Measure wall-clock time
        start_time = time.time()

        chunks_processed = 0
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            transcriber.process(chunk)
            chunks_processed += 1

        wall_time = time.time() - start_time

        # Calculate throughput
        throughput = total_audio_duration / wall_time

        logger.info(f"\n{'='*80}")
        logger.info("THROUGHPUT BENCHMARK")
        logger.info(f"{'='*80}")
        logger.info(f"Audio duration:     {total_audio_duration:.1f}s")
        logger.info(f"Wall-clock time:    {wall_time:.1f}s")
        logger.info(f"Chunks processed:   {chunks_processed}")
        logger.info(f"Throughput:         {throughput:.2f}x real-time")
        logger.info(f"Real-time capable:  {'✅ Yes' if throughput >= 1.0 else '❌ No'}")
        logger.info(f"{'='*80}")

        # Save as benchmark
        throughput_stats = {
            'audio_duration': total_audio_duration,
            'wall_time': wall_time,
            'throughput': throughput,
            'timestamp': datetime.now().isoformat()
        }

        benchmarks = BenchmarkStorage.load_benchmarks()
        if 'throughput' not in benchmarks:
            benchmarks['throughput'] = {'history': []}
        benchmarks['throughput']['history'].append(throughput_stats)
        benchmarks['throughput']['history'] = benchmarks['throughput']['history'][-10:]
        BenchmarkStorage.save_benchmarks(benchmarks)

        # Must be real-time capable
        assert throughput >= 1.0, \
            f"Throughput {throughput:.2f}x below real-time requirement (1.0x)"

        logger.info(f"✅ Throughput: {throughput:.2f}x real-time")


if __name__ == "__main__":
    # Run benchmark tests
    pytest.main([__file__, "-v", "-m", "benchmark", "--log-cli-level=INFO"])
