#!/usr/bin/env python3
"""
Performance Optimization Utilities

Provides high-performance data structures and monitoring tools for
real-time audio processing and Whisper transcription.

Components:
- RingBuffer: O(1) preallocated circular buffer (replaces np.concatenate)
- EncoderCache: LRU cache for Whisper encoder outputs
- PerformanceMetrics: Latency tracking with percentile statistics

Expected Performance Improvements:
- Memory allocations: -25% (RingBuffer)
- Encoder computations: -50% (EncoderCache)
- LID latency: -50% (batch processing + caching)
- Total latency: -35% (50-80ms → 30-50ms)
"""

from .ring_buffer import RingBuffer
from .encoder_cache import EncoderCache, BatchEncoderCache
from .performance_metrics import PerformanceMetrics, OperationStats, TimingContext

__all__ = [
    'RingBuffer',
    'EncoderCache',
    'BatchEncoderCache',
    'PerformanceMetrics',
    'OperationStats',
    'TimingContext',
]