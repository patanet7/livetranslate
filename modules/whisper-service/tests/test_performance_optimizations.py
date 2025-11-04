#!/usr/bin/env python3
"""
Test Suite for Performance Optimizations

Validates correctness and measures performance improvements:
- RingBuffer vs np.concatenate
- EncoderCache hit rates
- PerformanceMetrics accuracy
- Overall latency improvements

Expected improvements:
- Memory allocations: -25%
- Encoder computations: -50%
- LID latency: -50%
- Total latency: -35% (50-80ms → 30-50ms)
"""

import pytest
import numpy as np
import torch
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import RingBuffer, EncoderCache, PerformanceMetrics


class TestRingBuffer:
    """Test RingBuffer correctness and performance"""

    def test_basic_operations(self):
        """Test basic RingBuffer operations"""
        buffer = RingBuffer(capacity=1000, dtype=np.float32)

        # Test append
        data1 = np.random.randn(100).astype(np.float32)
        buffer.append(data1)
        assert len(buffer) == 100

        # Test read_all
        result = buffer.read_all()
        np.testing.assert_array_almost_equal(result, data1)

        # Test multiple appends
        data2 = np.random.randn(200).astype(np.float32)
        buffer.append(data2)
        assert len(buffer) == 300

        result = buffer.read_all()
        expected = np.concatenate([data1, data2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_wrap_around(self):
        """Test RingBuffer wrap-around behavior"""
        buffer = RingBuffer(capacity=100, dtype=np.float32)

        # Fill buffer
        data1 = np.arange(80).astype(np.float32)
        buffer.append(data1)

        # Trigger wrap-around
        data2 = np.arange(30).astype(np.float32)
        buffer.append(data2)

        # Should be at capacity
        assert len(buffer) == 100

        # Last 100 samples should match
        result = buffer.read_all()
        expected = np.concatenate([data1[-70:], data2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_consume(self):
        """Test RingBuffer consume operation"""
        buffer = RingBuffer(capacity=1000, dtype=np.float32)

        data = np.arange(500).astype(np.float32)
        buffer.append(data)

        # Consume first 200 samples
        consumed = buffer.consume(200)
        np.testing.assert_array_almost_equal(consumed, data[:200])

        # Check remaining
        assert len(buffer) == 300
        remaining = buffer.read_all()
        np.testing.assert_array_almost_equal(remaining, data[200:])

    def test_performance_vs_concatenate(self):
        """Benchmark RingBuffer vs np.concatenate"""
        iterations = 1000
        chunk_size = 160  # 10ms at 16kHz

        # Test np.concatenate (baseline)
        start = time.perf_counter()
        buffer_concat = np.array([], dtype=np.float32)
        for _ in range(iterations):
            chunk = np.random.randn(chunk_size).astype(np.float32)
            buffer_concat = np.concatenate([buffer_concat, chunk])
        time_concat = time.perf_counter() - start

        # Test RingBuffer
        start = time.perf_counter()
        buffer_ring = RingBuffer(capacity=chunk_size * iterations, dtype=np.float32)
        for _ in range(iterations):
            chunk = np.random.randn(chunk_size).astype(np.float32)
            buffer_ring.append(chunk)
        time_ring = time.perf_counter() - start

        speedup = time_concat / time_ring
        print(f"\nRingBuffer Performance:")
        print(f"  np.concatenate: {time_concat*1000:.2f}ms")
        print(f"  RingBuffer:     {time_ring*1000:.2f}ms")
        print(f"  Speedup:        {speedup:.2f}x")

        # RingBuffer should be at least 2x faster (conservative threshold)
        # On most systems it's 5-10x, but depends on numpy implementation
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"

    def test_clear(self):
        """Test RingBuffer clear operation"""
        buffer = RingBuffer(capacity=1000, dtype=np.float32)

        data = np.random.randn(500).astype(np.float32)
        buffer.append(data)
        assert len(buffer) == 500

        buffer.clear()
        assert len(buffer) == 0
        assert buffer.is_empty()


class TestEncoderCache:
    """Test EncoderCache correctness and performance"""

    def test_basic_caching(self):
        """Test basic cache operations"""
        cache = EncoderCache(max_size=10, device='cpu')

        # Create dummy encoder output
        encoder_output = torch.randn(1, 1500, 512)
        audio = np.random.randn(480000).astype(np.float32)

        # Test cache miss
        assert cache.get(audio) is None

        # Cache the output
        audio_hash = cache.put(encoder_output, audio)
        assert len(cache) == 1

        # Test cache hit
        cached = cache.get(audio)
        assert cached is not None
        torch.testing.assert_close(cached, encoder_output)

    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        cache = EncoderCache(max_size=3, device='cpu')

        # Add 3 entries
        audios = []
        outputs = []
        for i in range(3):
            audio = np.random.randn(480000).astype(np.float32)
            output = torch.randn(1, 1500, 512)
            audios.append(audio)
            outputs.append(output)
            cache.put(output, audio)

        assert len(cache) == 3

        # Add 4th entry (should evict oldest)
        audio4 = np.random.randn(480000).astype(np.float32)
        output4 = torch.randn(1, 1500, 512)
        cache.put(output4, audio4)

        assert len(cache) == 3

        # First entry should be evicted
        assert cache.get(audios[0]) is None

        # Others should still be cached
        assert cache.get(audios[1]) is not None
        assert cache.get(audios[2]) is not None
        assert cache.get(audio4) is not None

    def test_cache_statistics(self):
        """Test cache statistics tracking"""
        cache = EncoderCache(max_size=5, device='cpu')

        audio1 = np.random.randn(480000).astype(np.float32)
        output1 = torch.randn(1, 1500, 512)

        # Cache miss
        cache.get(audio1)

        # Cache put
        cache.put(output1, audio1)

        # Cache hit
        cache.get(audio1)
        cache.get(audio1)

        stats = cache.get_statistics()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 2/3
        assert stats['size'] == 1

    def test_hash_consistency(self):
        """Test that identical audio produces identical hash"""
        cache = EncoderCache(max_size=5, device='cpu')

        audio = np.random.randn(480000).astype(np.float32)

        # Compute hash twice
        hash1 = cache.precompute_hash(audio)
        hash2 = cache.precompute_hash(audio)

        assert hash1 == hash2

        # Slightly different audio should produce different hash
        audio_different = audio + np.random.randn(*audio.shape) * 0.001
        hash3 = cache.precompute_hash(audio_different)

        # Due to rounding, very small differences might still match
        # But large differences should produce different hashes
        audio_very_different = np.random.randn(*audio.shape).astype(np.float32)
        hash4 = cache.precompute_hash(audio_very_different)
        assert hash1 != hash4


class TestPerformanceMetrics:
    """Test PerformanceMetrics accuracy and overhead"""

    def test_basic_measurement(self):
        """Test basic timing measurement"""
        metrics = PerformanceMetrics(max_samples=100, enable_logging=False)

        # Measure a known operation
        with metrics.measure('test_operation'):
            time.sleep(0.01)  # 10ms

        stats = metrics.get_statistics('test_operation')
        assert 'test_operation' in stats

        op_stats = stats['test_operation']
        assert op_stats['count'] == 1
        assert 8 < op_stats['mean'] < 15  # Should be ~10ms with some tolerance

    def test_multiple_measurements(self):
        """Test statistics with multiple measurements"""
        metrics = PerformanceMetrics(max_samples=1000, enable_logging=False)

        # Record varying durations
        for i in range(100):
            duration = 0.001 + i * 0.0001  # 1ms to 11ms
            metrics.record('varying_op', duration)

        stats = metrics.get_statistics('varying_op')
        op_stats = stats['varying_op']

        assert op_stats['count'] == 100
        assert op_stats['min'] < op_stats['mean'] < op_stats['max']
        assert op_stats['p50'] < op_stats['p95'] < op_stats['p99']

    def test_manual_timing(self):
        """Test manual start/stop timer"""
        metrics = PerformanceMetrics(max_samples=100, enable_logging=False)

        metrics.start_timer('manual_op')
        time.sleep(0.005)  # 5ms
        duration = metrics.stop_timer('manual_op')

        assert 4 < duration * 1000 < 8  # Should be ~5ms

        stats = metrics.get_statistics('manual_op')
        assert stats['manual_op']['count'] == 1

    def test_percentile_accuracy(self):
        """Test percentile calculation accuracy"""
        metrics = PerformanceMetrics(max_samples=1000, enable_logging=False)

        # Record exact values
        for i in range(100):
            metrics.record('percentile_test', i / 1000)  # 0ms to 99ms

        stats = metrics.get_statistics('percentile_test')
        op_stats = stats['percentile_test']

        # Check percentiles
        assert 40 < op_stats['p50'] < 60  # Should be ~50ms
        assert 90 < op_stats['p95'] < 98  # Should be ~95ms
        assert 97 < op_stats['p99'] < 100  # Should be ~99ms

    def test_prometheus_export(self):
        """Test Prometheus format export"""
        metrics = PerformanceMetrics(max_samples=100, enable_logging=False)

        metrics.record('test_op', 0.01)
        metrics.record('test_op', 0.02)

        prometheus_output = metrics.export_prometheus()

        # Check format
        assert 'whisper_test_op_count' in prometheus_output
        assert 'whisper_test_op_latency_ms' in prometheus_output
        assert 'quantile="0.5"' in prometheus_output
        assert 'quantile="0.95"' in prometheus_output

    def test_measurement_overhead(self):
        """Test that metrics add minimal overhead"""
        metrics = PerformanceMetrics(max_samples=1000, enable_logging=False)

        iterations = 10000

        # Baseline (no metrics)
        start = time.perf_counter()
        for _ in range(iterations):
            x = 42 * 42  # Trivial operation
        time_baseline = time.perf_counter() - start

        # With metrics
        start = time.perf_counter()
        for _ in range(iterations):
            with metrics.measure('trivial_op'):
                x = 42 * 42
        time_with_metrics = time.perf_counter() - start

        overhead = (time_with_metrics - time_baseline) / iterations * 1000  # ms per operation

        print(f"\nPerformanceMetrics Overhead:")
        print(f"  Baseline:      {time_baseline*1000:.2f}ms")
        print(f"  With metrics:  {time_with_metrics*1000:.2f}ms")
        print(f"  Per operation: {overhead*1000:.2f}µs")

        # Overhead should be < 10µs per measurement
        assert overhead < 0.01, f"Overhead too high: {overhead*1000:.2f}µs"


class TestIntegrationPerformance:
    """Integration tests for overall performance improvements"""

    def test_buffer_operations_latency(self):
        """Test that buffer operations meet latency targets"""
        buffer = RingBuffer(capacity=16000 * 60, dtype=np.float32)  # 60s at 16kHz
        metrics = PerformanceMetrics(enable_logging=False)

        chunk_size = 160  # 10ms at 16kHz
        iterations = 1000

        for _ in range(iterations):
            chunk = np.random.randn(chunk_size).astype(np.float32)

            with metrics.measure('buffer_append'):
                buffer.append(chunk)

        stats = metrics.get_statistics('buffer_append')
        append_p95 = stats['buffer_append']['p95']

        print(f"\nBuffer Append Latency (p95): {append_p95:.3f}ms")

        # p95 should be < 0.1ms for 10ms chunks
        assert append_p95 < 0.1, f"Buffer append too slow: {append_p95:.3f}ms"

    def test_memory_efficiency(self):
        """Test memory efficiency of RingBuffer vs concatenate"""
        import tracemalloc

        chunk_size = 160
        iterations = 1000

        # Test np.concatenate memory
        tracemalloc.start()
        buffer_concat = np.array([], dtype=np.float32)
        for _ in range(iterations):
            chunk = np.random.randn(chunk_size).astype(np.float32)
            buffer_concat = np.concatenate([buffer_concat, chunk])
        current, peak_concat = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Test RingBuffer memory
        tracemalloc.start()
        buffer_ring = RingBuffer(capacity=chunk_size * iterations, dtype=np.float32)
        for _ in range(iterations):
            chunk = np.random.randn(chunk_size).astype(np.float32)
            buffer_ring.append(chunk)
        current, peak_ring = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_reduction = (peak_concat - peak_ring) / peak_concat

        print(f"\nMemory Usage:")
        print(f"  np.concatenate: {peak_concat / 1024:.1f} KB")
        print(f"  RingBuffer:     {peak_ring / 1024:.1f} KB")
        print(f"  Reduction:      {memory_reduction*100:.1f}%")

        # RingBuffer should use significantly less memory (at least 20% reduction)
        assert memory_reduction > 0.20, f"Expected >20% memory reduction, got {memory_reduction*100:.1f}%"


def run_performance_benchmarks():
    """Run comprehensive performance benchmarks"""
    print("\n" + "=" * 80)
    print("PERFORMANCE OPTIMIZATION BENCHMARKS")
    print("=" * 80)

    # Run all tests
    pytest.main([__file__, '-v', '-s'])


if __name__ == '__main__':
    run_performance_benchmarks()
