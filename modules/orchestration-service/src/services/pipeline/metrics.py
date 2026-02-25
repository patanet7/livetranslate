"""
Pipeline Metrics -- Per-chunk timing and aggregated counters.

Tracks end-to-end latency through each pipeline stage:
  receive -> dedup -> aggregate -> translate -> display

Uses an in-memory ring buffer (no DB writes per chunk) with
percentile aggregation and structlog-compatible output.
"""

import statistics
from collections import deque
from dataclasses import dataclass

from livetranslate_common.logging import get_logger

logger = get_logger()


@dataclass
class ChunkTimeline:
    """Timing record for a single chunk through the pipeline."""

    chunk_id: str
    speaker_name: str
    source: str  # "fireflies", "google_meet", etc.

    # Stage timestamps (time.monotonic() for precision)
    received_at: float
    dedup_decided_at: float | None = None
    dedup_result: str = ""  # "forwarded" | "duplicate_skipped" | "shrink_suppressed"
    aggregated_at: float | None = None
    translate_started_at: float | None = None
    translate_completed_at: float | None = None
    display_emitted_at: float | None = None

    # Metadata (filled as chunk moves through pipeline)
    text_length: int = 0
    word_count: int = 0
    boundary_type: str = ""

    def latencies_ms(self) -> dict[str, float]:
        """Compute per-stage latencies in milliseconds."""
        result: dict[str, float] = {}
        if self.dedup_decided_at is not None:
            result["dedup_ms"] = (self.dedup_decided_at - self.received_at) * 1000
        if self.aggregated_at is not None and self.dedup_decided_at is not None:
            result["aggregation_ms"] = (self.aggregated_at - self.dedup_decided_at) * 1000
        if self.translate_completed_at is not None and self.translate_started_at is not None:
            result["translation_ms"] = (
                self.translate_completed_at - self.translate_started_at
            ) * 1000
        if self.display_emitted_at is not None:
            result["end_to_end_ms"] = (self.display_emitted_at - self.received_at) * 1000
        return result


class PipelineMetricsCollector:
    """In-memory ring buffer of ChunkTimelines with counter aggregation.

    All methods must be called from the same asyncio event loop / thread.
    For background snapshot logging, use emit_snapshot() from the pipeline's
    own event loop.
    """

    def __init__(self, session_id: str, max_buffer: int = 1000):
        self.session_id = session_id
        self._timelines: deque[ChunkTimeline] = deque(maxlen=max_buffer)
        self._counters: dict[str, int] = {
            "raw_messages_received": 0,
            "duplicates_skipped": 0,
            "chunks_forwarded": 0,
            "shrinks_suppressed": 0,
            "interim_grow_events": 0,
            "interim_correction_events": 0,
            "sentences_produced": 0,
            "translations_completed": 0,
            "translations_failed": 0,
            "too_short_filtered": 0,
            "timelines_recorded": 0,
        }

    def record(self, timeline: ChunkTimeline) -> None:
        """Record a completed chunk timeline."""
        self._timelines.append(timeline)
        self._counters["timelines_recorded"] += 1

    def increment(self, counter_name: str, amount: int = 1) -> None:
        """Increment a named counter."""
        if counter_name not in self._counters:
            self._counters[counter_name] = 0
        self._counters[counter_name] += amount

    def get_counters(self) -> dict[str, int]:
        """Get all counter values."""
        return dict(self._counters)

    def get_latency_percentiles(self) -> dict[str, dict[str, float | int]]:
        """Compute p50/p95/p99 latencies per stage from buffered timelines."""
        if not self._timelines:
            return {}

        # Snapshot the deque so iteration is safe even if another
        # coroutine appends while we compute.
        snapshot = list(self._timelines)

        # Collect latencies by stage
        stage_latencies: dict[str, list[float]] = {}
        for tl in snapshot:
            for stage, ms in tl.latencies_ms().items():
                if stage not in stage_latencies:
                    stage_latencies[stage] = []
                stage_latencies[stage].append(ms)

        result: dict[str, dict[str, float | int]] = {}
        for stage, values in stage_latencies.items():
            if len(values) < 2:
                result[stage] = {
                    "p50": values[0],
                    "p95": values[0],
                    "p99": values[0],
                    "mean": values[0],
                    "count": len(values),
                }
                continue
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            result[stage] = {
                "p50": sorted_vals[n // 2],
                "p95": sorted_vals[int(n * 0.95)],
                "p99": sorted_vals[int(n * 0.99)],
                "mean": statistics.mean(sorted_vals),
                "count": n,
            }
        return result

    def to_structured_log(self) -> dict:
        """Return dict suitable for structlog emission."""
        return {
            "session_id": self.session_id,
            "counters": self.get_counters(),
            "latency_percentiles": self.get_latency_percentiles(),
            "buffer_size": len(self._timelines),
        }

    def emit_snapshot(self) -> None:
        """Emit a periodic metrics snapshot via structlog."""
        logger.info("pipeline_metrics_snapshot", **self.to_structured_log())
