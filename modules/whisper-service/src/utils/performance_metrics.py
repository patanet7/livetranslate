#!/usr/bin/env python3
"""
PerformanceMetrics - Latency Tracking with Percentiles

Tracks operation latencies and computes percentile statistics.
Exports metrics in Prometheus format for monitoring.

Performance:
- Sub-microsecond overhead per measurement
- O(n log n) percentile computation
- Memory-efficient circular buffer

Usage:
    metrics = PerformanceMetrics(max_samples=1000)

    # Context manager for automatic timing
    with metrics.measure('vad_processing'):
        vad_result = process_vad(audio)

    # Manual timing
    metrics.start_timer('lid_detection')
    lid_result = detect_language(audio)
    metrics.stop_timer('lid_detection')

    # Get statistics
    stats = metrics.get_statistics()
    print(f"VAD p95: {stats['vad_processing']['p95']:.2f}ms")
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OperationStats:
    """Statistics for a single operation type."""

    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    samples: list[float] = field(default_factory=list)

    def add_sample(self, duration: float, max_samples: int = 1000) -> None:
        """Add timing sample and update statistics."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)

        # Maintain circular buffer of samples for percentile calculation
        self.samples.append(duration)
        if len(self.samples) > max_samples:
            self.samples.pop(0)

    def get_mean(self) -> float:
        """Get mean latency in milliseconds."""
        if self.count == 0:
            return 0.0
        return (self.total_time / self.count) * 1000

    def get_percentile(self, p: float) -> float:
        """
        Get percentile latency in milliseconds.

        Args:
            p: Percentile (0-100)

        Returns:
            Latency at percentile in ms
        """
        if not self.samples:
            return 0.0

        # Convert to ms and compute percentile
        samples_ms = np.array(self.samples) * 1000
        return float(np.percentile(samples_ms, p))

    def get_statistics(self) -> dict[str, float]:
        """Get comprehensive statistics."""
        if self.count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        return {
            "count": self.count,
            "mean": self.get_mean(),
            "min": self.min_time * 1000,
            "max": self.max_time * 1000,
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
        }


class PerformanceMetrics:
    """
    Performance metrics tracker with percentile statistics.

    Tracks latencies for different operation types (VAD, LID, encoder, decoder)
    and computes percentile statistics (p50, p95, p99).

    Args:
        max_samples: Maximum samples to keep per operation (default 1000)
        enable_logging: Log metrics periodically (default True)
        log_interval_seconds: Seconds between log outputs (default 60)

    Attributes:
        operations: Dict of operation name to OperationStats
        active_timers: Dict of operation name to start time
    """

    __slots__ = (
        "_last_log_time",
        "active_timers",
        "enable_logging",
        "log_interval_seconds",
        "max_samples",
        "operations",
    )

    def __init__(
        self,
        max_samples: int = 1000,
        enable_logging: bool = True,
        log_interval_seconds: float = 60.0,
    ):
        """Initialize performance metrics tracker."""
        self.max_samples = max_samples
        self.enable_logging = enable_logging
        self.log_interval_seconds = log_interval_seconds

        # Operation statistics
        self.operations: dict[str, OperationStats] = defaultdict(lambda: OperationStats(name=""))

        # Active timers (for manual timing)
        self.active_timers: dict[str, float] = {}

        # Logging state
        self._last_log_time = time.time()

        logger.info(
            f"PerformanceMetrics initialized: max_samples={max_samples}, "
            f"logging={'enabled' if enable_logging else 'disabled'}"
        )

    @contextmanager
    def measure(self, operation_name: str):
        """
        Context manager for automatic timing.

        Usage:
            with metrics.measure('vad_processing'):
                result = process_vad(audio)

        Args:
            operation_name: Name of operation being measured
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record(operation_name, duration)

    def start_timer(self, operation_name: str) -> None:
        """
        Start manual timer for operation.

        Args:
            operation_name: Name of operation
        """
        self.active_timers[operation_name] = time.perf_counter()

    def stop_timer(self, operation_name: str) -> float:
        """
        Stop manual timer and record duration.

        Args:
            operation_name: Name of operation

        Returns:
            Duration in seconds
        """
        if operation_name not in self.active_timers:
            logger.warning(
                f"Timer '{operation_name}' was never started, " f"cannot stop. Recording 0.0s."
            )
            return 0.0

        start_time = self.active_timers.pop(operation_name)
        duration = time.perf_counter() - start_time
        self.record(operation_name, duration)

        return duration

    def record(self, operation_name: str, duration: float) -> None:
        """
        Record operation duration.

        Args:
            operation_name: Name of operation
            duration: Duration in seconds
        """
        # Get or create operation stats
        if operation_name not in self.operations:
            self.operations[operation_name] = OperationStats(name=operation_name)

        # Add sample
        self.operations[operation_name].add_sample(duration, self.max_samples)

        # Periodic logging
        if self.enable_logging:
            current_time = time.time()
            if current_time - self._last_log_time >= self.log_interval_seconds:
                self._log_statistics()
                self._last_log_time = current_time

    def get_statistics(self, operation_name: str | None = None) -> dict[str, dict[str, float]]:
        """
        Get performance statistics.

        Args:
            operation_name: Specific operation (None for all operations)

        Returns:
            Dict of operation name to statistics dict
        """
        if operation_name is not None:
            if operation_name not in self.operations:
                return {}
            return {operation_name: self.operations[operation_name].get_statistics()}

        # Return all operations
        return {name: stats.get_statistics() for name, stats in self.operations.items()}

    def get_summary(self) -> str:
        """
        Get human-readable summary of all metrics.

        Returns:
            Formatted string with statistics
        """
        lines = ["Performance Metrics Summary:", "=" * 80]

        for name, stats in self.operations.items():
            stats_dict = stats.get_statistics()
            lines.append(
                f"{name:20s}: count={stats_dict['count']:5d}, "
                f"mean={stats_dict['mean']:6.2f}ms, "
                f"p50={stats_dict['p50']:6.2f}ms, "
                f"p95={stats_dict['p95']:6.2f}ms, "
                f"p99={stats_dict['p99']:6.2f}ms"
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    def _log_statistics(self) -> None:
        """Log current statistics."""
        logger.info("\n" + self.get_summary())

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        for name, stats in self.operations.items():
            stats_dict = stats.get_statistics()
            metric_name = f"whisper_{name.replace('.', '_')}"

            # Export count
            lines.append(f"# HELP {metric_name}_count Total number of operations")
            lines.append(f"# TYPE {metric_name}_count counter")
            lines.append(f"{metric_name}_count {stats_dict['count']}")

            # Export latencies
            lines.append(f"# HELP {metric_name}_latency_ms Operation latency in milliseconds")
            lines.append(f"# TYPE {metric_name}_latency_ms summary")
            lines.append(f'{metric_name}_latency_ms{{quantile="0.5"}} {stats_dict["p50"]}')
            lines.append(f'{metric_name}_latency_ms{{quantile="0.95"}} {stats_dict["p95"]}')
            lines.append(f'{metric_name}_latency_ms{{quantile="0.99"}} {stats_dict["p99"]}')
            lines.append(f'{metric_name}_latency_ms_sum {stats_dict["mean"] * stats_dict["count"]}')
            lines.append(f'{metric_name}_latency_ms_count {stats_dict["count"]}')

        return "\n".join(lines)

    def reset(self, operation_name: str | None = None) -> None:
        """
        Reset metrics.

        Args:
            operation_name: Specific operation to reset (None for all)
        """
        if operation_name is not None:
            if operation_name in self.operations:
                self.operations[operation_name] = OperationStats(name=operation_name)
                logger.debug(f"Reset metrics for '{operation_name}'")
        else:
            self.operations.clear()
            self.active_timers.clear()
            logger.info("Reset all metrics")

    def get_total_time(self, operation_name: str) -> float:
        """
        Get total accumulated time for operation.

        Args:
            operation_name: Name of operation

        Returns:
            Total time in seconds
        """
        if operation_name not in self.operations:
            return 0.0
        return self.operations[operation_name].total_time

    def __repr__(self) -> str:
        op_count = len(self.operations)
        total_measurements = sum(op.count for op in self.operations.values())
        return (
            f"PerformanceMetrics(operations={op_count}, "
            f"total_measurements={total_measurements})"
        )


class TimingContext:
    """
    Lightweight timing context for nested operations.

    Supports hierarchical timing with parent/child relationships.

    Usage:
        metrics = PerformanceMetrics()

        with TimingContext(metrics, 'process_chunk'):
            # Main operation
            with TimingContext(metrics, 'process_chunk.vad'):
                vad_result = process_vad()

            with TimingContext(metrics, 'process_chunk.lid'):
                lid_result = process_lid()
    """

    __slots__ = ("metrics", "operation_name", "start_time")

    def __init__(self, metrics: PerformanceMetrics, operation_name: str):
        """Initialize timing context."""
        self.metrics = metrics
        self.operation_name = operation_name
        self.start_time = 0.0

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        duration = time.perf_counter() - self.start_time
        self.metrics.record(self.operation_name, duration)
        return False
