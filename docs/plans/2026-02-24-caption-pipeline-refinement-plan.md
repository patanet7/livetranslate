# Caption Pipeline Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix interim caption jitter (grow-only filter), add per-chunk pipeline timing metrics, tune sentence boundaries using real captured data, and add speaker identity to translation prompts.

**Architecture:** Three additive changes to the existing Layered Pipeline: (1) grow filter in LiveCaptionManager, (2) ChunkTimeline + PipelineMetricsCollector in new metrics module, (3) threshold tuning + replay validation using captured JSONL. See `docs/plans/2026-02-24-caption-pipeline-refinement-design.md` for full design.

**Tech Stack:** Python/FastAPI, structlog, pytest, JSONL replay, time.monotonic()

**Services required:** None — all changes are in-process. Tests run standalone.

**Test commands:** `uv run pytest tests/fireflies/unit/ -v` from `modules/orchestration-service/`

---

## Phase 1: Pipeline Metrics Foundation

### Task 1: ChunkTimeline + PipelineMetricsCollector

**Files:**
- Create: `src/services/pipeline/metrics.py`
- Test: `tests/fireflies/unit/test_pipeline_metrics.py`

**Step 1: Write the failing tests**

Create `tests/fireflies/unit/test_pipeline_metrics.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/fireflies/unit/test_pipeline_metrics.py -v`
Expected: FAIL — `metrics.py` doesn't exist yet

**Step 3: Write the implementation**

Create `src/services/pipeline/metrics.py`:

```python
"""
Pipeline Metrics — Per-chunk timing and aggregated counters.

Tracks end-to-end latency through each pipeline stage:
  receive → dedup → aggregate → translate → display

Uses an in-memory ring buffer (no DB writes per chunk) with
percentile aggregation and structlog-compatible output.
"""

import statistics
import time
from collections import deque
from dataclasses import dataclass, field

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

    Thread-safe for single-writer (pipeline) use. Counters are atomic increments.
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

    def get_latency_percentiles(self) -> dict[str, dict[str, float]]:
        """Compute p50/p95/p99 latencies per stage from buffered timelines."""
        if not self._timelines:
            return {}

        # Collect latencies by stage
        stage_latencies: dict[str, list[float]] = {}
        for tl in self._timelines:
            for stage, ms in tl.latencies_ms().items():
                if stage not in stage_latencies:
                    stage_latencies[stage] = []
                stage_latencies[stage].append(ms)

        result: dict[str, dict[str, float]] = {}
        for stage, values in stage_latencies.items():
            if len(values) < 2:
                result[stage] = {"p50": values[0], "p95": values[0], "p99": values[0], "count": len(values)}
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/fireflies/unit/test_pipeline_metrics.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/services/pipeline/metrics.py tests/fireflies/unit/test_pipeline_metrics.py
git commit -m "feat: add ChunkTimeline and PipelineMetricsCollector for pipeline observability"
```

---

## Phase 2: Grow Filter

### Task 2: Server-Side Grow Filter in LiveCaptionManager

**Files:**
- Modify: `src/services/pipeline/live_caption_manager.py` (full file — 139 lines)
- Test: `tests/fireflies/unit/test_live_caption_manager.py` (add new test class)

**Step 1: Write the failing tests**

Add to the existing test file (or create `tests/fireflies/unit/test_grow_filter.py`):

```python
#!/usr/bin/env python3
"""Tests for LiveCaptionManager grow filter."""

import os
import sys
from pathlib import Path
from dataclasses import dataclass

import pytest

os.environ["SKIP_MAIN_FASTAPI_IMPORT"] = "1"

orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

import importlib.util

_lcm_spec = importlib.util.spec_from_file_location(
    "live_caption_manager", src_path / "services" / "pipeline" / "live_caption_manager.py"
)
_lcm_module = importlib.util.module_from_spec(_lcm_spec)
_lcm_spec.loader.exec_module(_lcm_module)

LiveCaptionManager = _lcm_module.LiveCaptionManager

# Minimal stubs
@dataclass
class FakeChunk:
    chunk_id: str = "c1"
    text: str = "hello"
    speaker_name: str = "Alice"

@dataclass
class FakeConfig:
    session_id: str = "s1"
    display_mode: str = "both"
    enable_interim_captions: bool = True


class TestGrowFilter:
    """Test the grow-only interim caption filter."""

    @pytest.fixture
    def broadcasts(self):
        return []

    @pytest.fixture
    def manager(self, broadcasts):
        async def fake_broadcast(session_id, msg):
            broadcasts.append(msg)
        config = FakeConfig()
        return LiveCaptionManager(config=config, broadcast=fake_broadcast, session_id="s1")

    @pytest.mark.asyncio
    async def test_first_text_always_broadcast(self, manager, broadcasts):
        """First interim for a chunk_id is always sent."""
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        assert len(broadcasts) == 1
        assert broadcasts[0]["text"] == "hello"
        assert broadcasts[0]["type"] == "grow"

    @pytest.mark.asyncio
    async def test_grow_appends_broadcast(self, manager, broadcasts):
        """Text that grows (startswith previous) is broadcast."""
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        assert len(broadcasts) == 2
        assert broadcasts[1]["type"] == "grow"

    @pytest.mark.asyncio
    async def test_shrink_suppressed(self, manager, broadcasts):
        """Text that shrinks is NOT broadcast."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        assert len(broadcasts) == 1  # Only the first one

    @pytest.mark.asyncio
    async def test_correction_longer_broadcast(self, manager, broadcasts):
        """Text that is rewritten but longer is broadcast as correction."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="Hello World! And more"), is_final=False)
        assert len(broadcasts) == 2
        assert broadcasts[1]["type"] == "correction"

    @pytest.mark.asyncio
    async def test_duplicate_suppressed(self, manager, broadcasts):
        """Exact same text is suppressed."""
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)
        assert len(broadcasts) == 1

    @pytest.mark.asyncio
    async def test_final_always_broadcast(self, manager, broadcasts):
        """is_final=True always broadcasts regardless of text length."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=True)
        assert len(broadcasts) == 2
        assert broadcasts[1]["type"] == "final"
        assert broadcasts[1]["is_final"] is True

    @pytest.mark.asyncio
    async def test_final_cleans_displayed_text(self, manager, broadcasts):
        """After final, a new text for same chunk_id starts fresh."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="done"), is_final=True)
        # New text for same chunk_id should be treated as first
        await manager.handle_interim_update(FakeChunk(text="new"), is_final=False)
        assert len(broadcasts) == 3
        assert broadcasts[2]["type"] == "grow"

    @pytest.mark.asyncio
    async def test_stats_track_suppressed(self, manager, broadcasts):
        """Stats counters track suppressed shrinks."""
        await manager.handle_interim_update(FakeChunk(text="hello world"), is_final=False)
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)  # shrink
        await manager.handle_interim_update(FakeChunk(text="hello"), is_final=False)  # dup
        stats = manager.stats
        assert stats["interim_updates_sent"] == 1
        assert stats["interim_shrinks_suppressed"] >= 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/fireflies/unit/test_grow_filter.py -v`
Expected: FAIL — `LiveCaptionManager` doesn't have `_displayed_text` state or `type` field yet

**Step 3: Implement the grow filter**

Modify `src/services/pipeline/live_caption_manager.py`. Replace the current `handle_interim_update` method (lines 72-105) and add state:

In `__init__` (after line 52), add:
```python
        # Grow filter state: chunk_id -> last text broadcast
        self._displayed_text: dict[str, str] = {}
        self._interim_shrinks_suppressed: int = 0
        self._interim_duplicates_suppressed: int = 0
```

Update `stats` property (lines 65-70) to include new counters:
```python
    @property
    def stats(self) -> dict[str, int]:
        return {
            "interim_updates_sent": self._interim_updates_sent,
            "interim_updates_filtered": self._interim_updates_filtered,
            "interim_shrinks_suppressed": self._interim_shrinks_suppressed,
            "interim_duplicates_suppressed": self._interim_duplicates_suppressed,
            "captions_sent": self._captions_sent,
            "displayed_text_entries": len(self._displayed_text),
        }
```

Replace `handle_interim_update` (lines 72-105) with:
```python
    async def handle_interim_update(self, chunk, is_final: bool) -> None:
        """Handle interim caption with grow-only filter.

        Only broadcasts when:
        - is_final=True (always)
        - Text is new (first time for this chunk_id)
        - Text grew (starts with previous text)
        - Text was corrected but is longer

        Suppresses:
        - Pure duplicates (same text)
        - Shrinks (text got shorter without being a grow/correction)
        """
        # Display mode gate (unchanged)
        if self.display_mode == "translated" and not is_final:
            self._interim_updates_filtered += 1
            return

        # Interim enabled gate (unchanged)
        if not self.interim_enabled and not is_final:
            self._interim_updates_filtered += 1
            return

        chunk_id = chunk.chunk_id
        new_text = chunk.text

        # Final always broadcasts and cleans up
        if is_final:
            self._displayed_text.pop(chunk_id, None)
            await self._broadcast(
                self._session_id,
                {
                    "event": "interim_caption",
                    "chunk_id": chunk_id,
                    "text": new_text,
                    "speaker_name": chunk.speaker_name,
                    "speaker_color": None,
                    "is_final": True,
                    "type": "final",
                },
            )
            self._interim_updates_sent += 1
            return

        last_text = self._displayed_text.get(chunk_id, "")

        # Pure duplicate — skip
        if new_text == last_text:
            self._interim_duplicates_suppressed += 1
            return

        # Determine update type
        if not last_text:
            update_type = "grow"  # First text for this chunk
        elif new_text.startswith(last_text):
            update_type = "grow"  # Pure append
        elif len(new_text) > len(last_text):
            update_type = "correction"  # Rewritten but longer
        else:
            # Shrink — suppress
            self._interim_shrinks_suppressed += 1
            return

        self._displayed_text[chunk_id] = new_text
        await self._broadcast(
            self._session_id,
            {
                "event": "interim_caption",
                "chunk_id": chunk_id,
                "text": new_text,
                "speaker_name": chunk.speaker_name,
                "speaker_color": None,
                "is_final": False,
                "type": update_type,
            },
        )
        self._interim_updates_sent += 1
```

Add a cleanup method at the end of the class:
```python
    def cleanup_stale_displayed_text(self, max_age_seconds: float = 30.0) -> int:
        """Remove displayed text entries older than max_age. Call periodically."""
        # In practice, entries are cleaned on finalization. This is a safety net.
        # For simplicity, just clear all if called. The real staleness would
        # need timestamps per entry; for now this serves as a manual reset.
        count = len(self._displayed_text)
        self._displayed_text.clear()
        return count
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/fireflies/unit/test_grow_filter.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/services/pipeline/live_caption_manager.py tests/fireflies/unit/test_grow_filter.py
git commit -m "feat: add grow-only filter to LiveCaptionManager, suppress interim caption jitter"
```

---

## Phase 3: Wire Metrics Into Pipeline

### Task 3: Stamp Timestamps at Each Pipeline Stage

**Files:**
- Modify: `src/clients/fireflies_client.py:777-844` — add `received_at` to chunks
- Modify: `src/services/pipeline/coordinator.py:210-265` — stamp aggregated_at, wire collector
- Modify: `src/services/rolling_window_translator.py:198-284` — expose translate timestamps
- Modify: `src/services/caption_buffer.py:262-409` — stamp display_emitted_at
- Modify: `src/models/fireflies.py` — add `received_at` field to `FirefliesChunk`

**Step 1: Add `received_at` to FirefliesChunk**

In `src/models/fireflies.py`, find the `FirefliesChunk` dataclass and add:
```python
    received_at: float = 0.0  # time.monotonic() when message arrived at client
```

**Step 2: Stamp received_at in fireflies_client.py**

In `_handle_transcript()` (line 787), add immediately after `self._raw_messages_received += 1`:
```python
            _received_at = time.monotonic()
```

And after creating the `FirefliesChunk` (around line 824), add:
```python
            chunk.received_at = _received_at
```

Also add `import time` at the top of the file (line 18 area).

**Step 3: Wire PipelineMetricsCollector into coordinator**

In `src/services/pipeline/coordinator.py`:
- Import: `from .metrics import ChunkTimeline, PipelineMetricsCollector`
- In `__init__` (around line 133): add `self._metrics: PipelineMetricsCollector | None = None`
- In `initialize()` (around line 196): add `self._metrics = PipelineMetricsCollector(session_id=self.config.session_id)`
- In `process_raw_chunk()` after adapt (line 231-241): stamp `aggregated_at` when sentence produced
- In `_handle_sentence_ready()` (line 287-356): stamp translate start/end and display
- In `get_stats()` (line 651-667): merge `self._metrics.to_structured_log()` into stats dict

**Step 4: Expose in session stats**

In `src/routers/fireflies.py`, in `get_session_status()`, include `coordinator.get_stats()["metrics"]` if present.

**Step 5: Run existing tests**

Run: `uv run pytest tests/fireflies/unit/ -v`
Expected: All existing tests still PASS (additive changes only)

**Step 6: Commit**

```bash
git add src/models/fireflies.py src/clients/fireflies_client.py src/services/pipeline/coordinator.py src/services/pipeline/metrics.py src/routers/fireflies.py
git commit -m "feat: wire pipeline timing metrics through all stages (receive→dedup→aggregate→translate→display)"
```

---

## Phase 4: Sentence Boundary Tuning

### Task 4: Replay Script + Baseline Measurement

**Files:**
- Create: `scripts/replay_captured_data.py`

**Step 1: Write the replay script**

Create `scripts/replay_captured_data.py`:

```python
#!/usr/bin/env python3
"""
Replay captured Fireflies JSONL through SentenceAggregator.

Feeds finalized chunks (deduped to last version per chunk_id) through the
aggregator and reports sentence statistics: count, avg length, boundary
types, fragments, etc.

Usage:
    uv run python scripts/replay_captured_data.py [--pause-ms 600] [--max-words 25]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Setup import path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from models.fireflies import FirefliesChunk, FirefliesSessionConfig, TranslationUnit
from services.sentence_aggregator import SentenceAggregator

DEFAULT_JSONL = root / "captured_data" / "20260224_190411_fireflies_raw_capture.jsonl"


def load_finalized_chunks(jsonl_path: Path) -> list[dict]:
    """Load JSONL and deduplicate to final version per chunk_id."""
    events = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Filter to transcript events, deduplicate by chunk_id (last wins)
    final_by_id: dict[str, dict] = {}
    for e in events:
        if e["event"] not in ("transcription.broadcast", "transcript"):
            continue
        data = e.get("data", {})
        payload = data.get("payload", data.get("data", data))
        if isinstance(payload, dict) and payload.get("chunk_id"):
            final_by_id[payload["chunk_id"]] = payload

    # Sort by start_time
    return sorted(final_by_id.values(), key=lambda c: c.get("start_time", 0))


def replay(chunks: list[dict], config: FirefliesSessionConfig) -> list[TranslationUnit]:
    """Feed chunks through SentenceAggregator, collect produced sentences."""
    sentences: list[TranslationUnit] = []

    def on_sentence(unit: TranslationUnit):
        sentences.append(unit)

    aggregator = SentenceAggregator(
        session_id="replay",
        transcript_id="replay",
        config=config,
        on_sentence_ready=on_sentence,
    )

    for c in chunks:
        chunk = FirefliesChunk(
            transcript_id="replay",
            chunk_id=str(c["chunk_id"]),
            text=c.get("text", ""),
            speaker_name=c.get("speaker_name", "Unknown"),
            start_time=float(c.get("start_time", 0)),
            end_time=float(c.get("end_time", 0)),
        )
        aggregator.process_chunk(chunk)

    # Flush remaining
    remaining = aggregator.flush_all()
    sentences.extend(remaining)

    return sentences


def report(sentences: list[TranslationUnit], label: str):
    """Print stats report for a set of sentences."""
    if not sentences:
        print(f"\n=== {label}: No sentences produced ===")
        return

    lengths = [len(s.text.split()) for s in sentences]
    boundaries = Counter(s.boundary_type for s in sentences)
    speakers = Counter(s.speaker_name for s in sentences)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Sentences produced:   {len(sentences)}")
    print(f"  Avg words/sentence:   {sum(lengths) / len(lengths):.1f}")
    print(f"  Min words:            {min(lengths)}")
    print(f"  Max words:            {max(lengths)}")
    print(f"  Boundary types:       {dict(boundaries)}")
    print(f"  Speakers:             {dict(speakers)}")
    print(f"  Short (<3 words):     {sum(1 for l in lengths if l < 3)}")
    print(f"  Long (>20 words):     {sum(1 for l in lengths if l > 20)}")
    print()
    for i, s in enumerate(sentences):
        words = len(s.text.split())
        print(f"  [{i+1:3}] ({s.boundary_type:15}) [{s.speaker_name:>20}] ({words:2}w) \"{s.text}\"")


def main():
    parser = argparse.ArgumentParser(description="Replay Fireflies data through SentenceAggregator")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL, help="JSONL file to replay")
    parser.add_argument("--pause-ms", type=float, default=None, help="Override pause_threshold_ms")
    parser.add_argument("--max-words", type=int, default=None, help="Override max_buffer_words")
    parser.add_argument("--max-seconds", type=float, default=None, help="Override max_buffer_seconds")
    parser.add_argument("--min-words", type=int, default=None, help="Override min_words_for_translation")
    parser.add_argument("--compare", action="store_true", help="Run both current and proposed thresholds")
    args = parser.parse_args()

    chunks = load_finalized_chunks(args.jsonl)
    print(f"Loaded {len(chunks)} finalized chunks from {args.jsonl}")

    if args.compare:
        # Current thresholds
        current_config = FirefliesSessionConfig(
            api_key="", transcript_id="replay",
            pause_threshold_ms=800, max_buffer_words=30,
            max_buffer_seconds=5.0, min_words_for_translation=3,
        )
        current_sentences = replay(chunks, current_config)
        report(current_sentences, "CURRENT THRESHOLDS (800ms / 30w / 5s / min3)")

        # Proposed thresholds
        proposed_config = FirefliesSessionConfig(
            api_key="", transcript_id="replay",
            pause_threshold_ms=600, max_buffer_words=25,
            max_buffer_seconds=4.0, min_words_for_translation=2,
        )
        proposed_sentences = replay(chunks, proposed_config)
        report(proposed_sentences, "PROPOSED THRESHOLDS (600ms / 25w / 4s / min2)")
    else:
        config = FirefliesSessionConfig(
            api_key="", transcript_id="replay",
            pause_threshold_ms=args.pause_ms or 800,
            max_buffer_words=args.max_words or 30,
            max_buffer_seconds=args.max_seconds or 5.0,
            min_words_for_translation=args.min_words or 3,
        )
        sentences = replay(chunks, config)
        report(sentences, f"Thresholds: {config.pause_threshold_ms}ms / {config.max_buffer_words}w / {config.max_buffer_seconds}s / min{config.min_words_for_translation}")


if __name__ == "__main__":
    main()
```

**Step 2: Run baseline comparison**

Run: `uv run python scripts/replay_captured_data.py --compare`
Expected: Two reports showing sentence distributions with current vs proposed thresholds. Review output to confirm proposed thresholds produce better sentence boundaries.

**Step 3: Commit**

```bash
git add scripts/replay_captured_data.py
git commit -m "feat: add JSONL replay script for data-driven sentence boundary validation"
```

---

### Task 5: Apply Tuned Thresholds

**Files:**
- Modify: `src/services/pipeline/config.py:59-62` — update defaults

**Step 1: Update PipelineConfig defaults**

In `src/services/pipeline/config.py`, change lines 59-62:

```python
    # Before:
    pause_threshold_ms: float = 800.0
    max_words_per_sentence: int = 30
    max_time_per_sentence_ms: float = 5000.0
    min_words_for_translation: int = 3

    # After:
    pause_threshold_ms: float = 600.0
    max_words_per_sentence: int = 25
    max_time_per_sentence_ms: float = 4000.0
    min_words_for_translation: int = 2
```

**Step 2: Run all tests**

Run: `uv run pytest tests/fireflies/unit/ -v`
Expected: All tests PASS (threshold changes are backward-compatible — they're just defaults)

**Step 3: Commit**

```bash
git add src/services/pipeline/config.py
git commit -m "feat: tune sentence boundaries from real data (800→600ms pause, 30→25 words, 5→4s buffer, 3→2 min words)"
```

---

## Phase 5: Speaker-Aware Translation

### Task 6: Add Speaker Name to Translation Prompt

**Files:**
- Modify: `src/services/translation_prompt_builder.py:28-42` — add speaker to template
- Modify: `src/services/translation_prompt_builder.py:171-194` — format speaker in prompt
- Test: `tests/fireflies/unit/test_rolling_window_translator.py` — add speaker prompt test

**Step 1: Write the failing test**

Add to `tests/fireflies/unit/test_rolling_window_translator.py`:

```python
def test_prompt_includes_speaker_name():
    """Translation prompt should include speaker identity for context."""
    from services.translation_prompt_builder import TranslationPromptBuilder, PromptContext

    builder = TranslationPromptBuilder()
    result = builder.build(PromptContext(
        current_sentence="We need to fix the diarization pipeline",
        target_language="zh",
        speaker_name="Thomas Patane",
        previous_sentences=["The audio is working now"],
    ))
    assert "Thomas Patane" in result.prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/fireflies/unit/test_rolling_window_translator.py::test_prompt_includes_speaker_name -v`
Expected: FAIL — speaker_name not included in prompt output

**Step 3: Update the prompt template and builder**

In `src/services/translation_prompt_builder.py`, update `TRANSLATION_PROMPT_TEMPLATE` (line 28):

```python
TRANSLATION_PROMPT_TEMPLATE = """You are a professional real-time translator.

Target Language: {target_language}
{speaker_section}
{glossary_section}

Previous context (DO NOT translate, only use for understanding references):
{context_window}

---

Translate ONLY the following sentence to {target_language}:
{current_sentence}

Translation:"""
```

In `_build_full_prompt()` (line 171), add speaker section formatting:

```python
    def _build_full_prompt(
        self,
        context: PromptContext,
        previous_sentences: list[str],
        glossary_terms: dict[str, str],
    ) -> str:
        """Build prompt with context, glossary, and speaker identity."""
        # Format speaker section
        speaker_section = ""
        if context.speaker_name:
            speaker_section = f"Current Speaker: {context.speaker_name}"

        # Format glossary section
        glossary_section = ""
        if glossary_terms:
            glossary_lines = [f"- {source} = {target}" for source, target in glossary_terms.items()]
            glossary_section = "Glossary (use these exact translations):\n" + "\n".join(
                glossary_lines
            )

        # Format context window
        context_window = self._format_context_window(previous_sentences)

        return self.full_template.format(
            target_language=context.target_language,
            speaker_section=speaker_section,
            glossary_section=glossary_section,
            context_window=context_window,
            current_sentence=context.current_sentence,
        )
```

Also update the `SIMPLE_TRANSLATION_PROMPT` and `_build_simple_prompt` to not require speaker_section (they don't use it — no change needed since `str.format()` only fills named placeholders that exist in the template).

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/fireflies/unit/test_rolling_window_translator.py -v`
Expected: All tests PASS including new speaker name test

**Step 5: Commit**

```bash
git add src/services/translation_prompt_builder.py tests/fireflies/unit/test_rolling_window_translator.py
git commit -m "feat: include speaker identity in translation prompt for per-speaker consistency"
```

---

## Phase 6: Integration + Update plan.md

### Task 7: Run Full Test Suite + Update plan.md

**Files:**
- Modify: `plan.md` — add "Caption Pipeline Refinement" section

**Step 1: Run full test suite**

Run: `uv run pytest tests/fireflies/ -v`
Expected: All tests PASS

**Step 2: Run replay validation**

Run: `uv run python scripts/replay_captured_data.py --compare`
Expected: Report shows proposed thresholds produce better sentence boundaries

**Step 3: Update plan.md**

Add new section to `modules/orchestration-service/plan.md` under "Completed" sections documenting what was built.

**Step 4: Final commit**

```bash
git add plan.md
git commit -m "docs: update plan.md with caption pipeline refinement completion"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | ChunkTimeline + PipelineMetricsCollector | Create `metrics.py` | 7 tests |
| 2 | Grow filter | Modify `live_caption_manager.py` | 8 tests |
| 3 | Wire metrics into pipeline stages | Modify 5 files | Existing tests |
| 4 | Replay script + baseline | Create `replay_captured_data.py` | Manual validation |
| 5 | Apply tuned thresholds | Modify `config.py` | Existing tests |
| 6 | Speaker-aware translation | Modify `translation_prompt_builder.py` | 1 test |
| 7 | Integration + docs | Update `plan.md` | Full suite |

**Total: 7 tasks, ~16 new tests, 7 commits**
