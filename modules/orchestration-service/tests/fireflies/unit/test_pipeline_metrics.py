#!/usr/bin/env python3
"""Tests for ChunkTimeline and PipelineMetricsCollector."""

import os
import sys
import time
from pathlib import Path

import pytest

os.environ["SKIP_MAIN_FASTAPI_IMPORT"] = "1"

orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

import importlib.util

_metrics_spec = importlib.util.spec_from_file_location(
    "metrics", src_path / "services" / "pipeline" / "metrics.py"
)
_metrics_module = importlib.util.module_from_spec(_metrics_spec)
_metrics_spec.loader.exec_module(_metrics_module)

ChunkTimeline = _metrics_module.ChunkTimeline
PipelineMetricsCollector = _metrics_module.PipelineMetricsCollector


# === ChunkTimeline Tests ===


class TestChunkTimeline:
    def test_latencies_ms_full_pipeline(self):
        """All stages populated produces all latency values."""
        t0 = time.monotonic()
        tl = ChunkTimeline(
            chunk_id="1", speaker_name="Alice", source="fireflies",
            received_at=t0,
            dedup_decided_at=t0 + 0.001,
            dedup_result="forwarded",
            aggregated_at=t0 + 0.010,
            translate_started_at=t0 + 0.011,
            translate_completed_at=t0 + 0.111,
            display_emitted_at=t0 + 0.112,
        )
        latencies = tl.latencies_ms()
        assert "dedup_ms" in latencies
        assert "aggregation_ms" in latencies
        assert "translation_ms" in latencies
        assert "end_to_end_ms" in latencies
        assert latencies["translation_ms"] == pytest.approx(100.0, abs=5.0)
        assert latencies["end_to_end_ms"] == pytest.approx(112.0, abs=5.0)

    def test_latencies_ms_partial(self):
        """Skipped stages produce partial latency dict."""
        t0 = time.monotonic()
        tl = ChunkTimeline(
            chunk_id="1", speaker_name="Alice", source="fireflies",
            received_at=t0,
            dedup_decided_at=t0 + 0.001,
            dedup_result="duplicate_skipped",
            aggregated_at=None,
            translate_started_at=None,
            translate_completed_at=None,
            display_emitted_at=None,
        )
        latencies = tl.latencies_ms()
        assert "dedup_ms" in latencies
        assert "translation_ms" not in latencies
        assert "end_to_end_ms" not in latencies


# === PipelineMetricsCollector Tests ===


class TestPipelineMetricsCollector:
    def test_record_and_counters(self):
        """Recording timelines updates counters."""
        collector = PipelineMetricsCollector(session_id="s1", max_buffer=100)
        t0 = time.monotonic()
        collector.record(ChunkTimeline(
            chunk_id="1", speaker_name="Alice", source="fireflies",
            received_at=t0, dedup_decided_at=t0 + 0.001, dedup_result="forwarded",
            aggregated_at=t0 + 0.01, translate_started_at=t0 + 0.01,
            translate_completed_at=t0 + 0.1, display_emitted_at=t0 + 0.11,
        ))
        counters = collector.get_counters()
        assert counters["timelines_recorded"] == 1

    def test_ring_buffer_overflow(self):
        """Buffer does not exceed max_buffer."""
        collector = PipelineMetricsCollector(session_id="s1", max_buffer=5)
        t0 = time.monotonic()
        for i in range(10):
            collector.record(ChunkTimeline(
                chunk_id=str(i), speaker_name="Alice", source="fireflies",
                received_at=t0 + i * 0.1, dedup_decided_at=t0 + i * 0.1 + 0.001,
                dedup_result="forwarded", aggregated_at=None,
                translate_started_at=None, translate_completed_at=None,
                display_emitted_at=None,
            ))
        assert len(collector._timelines) == 5

    def test_latency_percentiles(self):
        """Percentile computation works with recorded timelines."""
        collector = PipelineMetricsCollector(session_id="s1", max_buffer=100)
        t0 = time.monotonic()
        for i in range(20):
            collector.record(ChunkTimeline(
                chunk_id=str(i), speaker_name="Alice", source="fireflies",
                received_at=t0, dedup_decided_at=t0 + 0.001, dedup_result="forwarded",
                aggregated_at=t0 + 0.01,
                translate_started_at=t0 + 0.01,
                translate_completed_at=t0 + 0.05 + i * 0.005,
                display_emitted_at=t0 + 0.06 + i * 0.005,
            ))
        percentiles = collector.get_latency_percentiles()
        assert "translation_ms" in percentiles
        assert "p50" in percentiles["translation_ms"]
        assert "p95" in percentiles["translation_ms"]
        assert "p99" in percentiles["translation_ms"]

    def test_increment_counter(self):
        """Direct counter increments work."""
        collector = PipelineMetricsCollector(session_id="s1")
        collector.increment("duplicates_skipped")
        collector.increment("duplicates_skipped")
        collector.increment("shrinks_suppressed")
        counters = collector.get_counters()
        assert counters["duplicates_skipped"] == 2
        assert counters["shrinks_suppressed"] == 1

    def test_to_structured_log(self):
        """Structured log output has required keys."""
        collector = PipelineMetricsCollector(session_id="s1")
        log_data = collector.to_structured_log()
        assert log_data["session_id"] == "s1"
        assert "counters" in log_data
        assert "latency_percentiles" in log_data
