# Caption Pipeline Refinement Design

**Date:** 2026-02-24
**Status:** Approved
**Builds on:** `2026-02-20-fireflies-realtime-enhancement-design.md` (Layered Pipeline)

## Problem Statement

After capturing real Fireflies data from a live meeting (987 events, 86 unique chunks, 2 speakers), three issues emerged:

1. **Interim caption jitter** — `LiveCaptionManager.handle_interim_update()` broadcasts every ASR refinement (~11.4 updates per chunk). Fireflies doesn't just append — it corrects, shortens, re-capitalizes. Text visually flickers instead of growing naturally.
2. **No pipeline observability** — No way to prove dedup is working, measure end-to-end latency, or identify bottlenecks. Existing `PipelineStats` tracks counts but not per-chunk timing.
3. **Sentence boundaries not tuned for real data** — Thresholds (800ms pause, 30 words max, 5s buffer) were set speculatively. Real captured data shows different conversational patterns.

## Captured Data Reference

File: `modules/orchestration-service/captured_data/20260224_190411_fireflies_raw_capture.jsonl`

Key observations from real data:
- **Payload structure**: 5 fields only — `chunk_id`, `text`, `speaker_name`, `start_time`, `end_time`
- **ASR refinement**: 2-44 updates per chunk (avg 11.4). Text grows, shrinks, re-capitalizes
- **Speaker overlap**: Concurrent chunk_ids for different speakers with overlapping time ranges
- **No meeting-end signal**: Connection just stops receiving chunks
- **Delivery is not chronological**: Older chunks arrive after newer ones (finalization backfill)

## Design

### Component 1: Server-Side Grow Filter

**Location:** `src/services/pipeline/live_caption_manager.py`

**New state per session:**
```python
_displayed_text: dict[str, str]    # chunk_id -> last text broadcast to clients
```

**Logic in `handle_interim_update()`:**

```
on_live_update(chunk, is_final):

  if is_final:
      broadcast interim_caption(is_final=True)
      cleanup _displayed_text[chunk_id]
      return

  last = _displayed_text.get(chunk_id, "")
  new = chunk.text

  if new == last:
      skip (duplicate)

  elif new.startswith(last):
      GROW: broadcast with type="grow"
      update _displayed_text

  elif len(new) > len(last):
      CORRECTION+GROW: broadcast with type="correction"
      update _displayed_text

  else:
      SHRINK: suppress, don't broadcast
      increment _interim_corrections_suppressed
```

**Rationale:** Text only flows forward (grows or corrects-to-longer). Shrinks are suppressed because Fireflies' ASR frequently shortens text mid-correction then grows it back. Suppressing shrinks eliminates visual jitter while still showing corrections that end up longer.

**Cleanup:** Remove `_displayed_text` entries when chunk is finalized or after 30s staleness.

**Events broadcast to frontend:**
```json
{"event": "interim_caption", "chunk_id": "65704", "text": "of speakers in the Diarization. So that's kind of", "speaker_name": "Thomas Patane", "is_final": false, "type": "grow"}
{"event": "interim_caption", "chunk_id": "65704", "text": "of speakers in the Diarization. So that's kind of a little annoying but the", "speaker_name": "Thomas Patane", "is_final": true, "type": "final"}
```

**UI impact:** All three display surfaces (captions.html, React CaptionOverlay, OBS overlay) receive the same cleaned events via the shared WebSocket feed. No frontend changes required for basic functionality — the `type` field is informational.

### Component 2: Pipeline Timing Metrics

**New file:** `src/services/pipeline/metrics.py`

**ChunkTimeline dataclass:**
```python
@dataclass
class ChunkTimeline:
    chunk_id: str
    speaker_name: str
    source: str                         # "fireflies", "google_meet", etc.

    received_at: float                  # time.monotonic() when raw message arrived
    dedup_decided_at: float | None      # when dedup forwarded/skipped
    dedup_result: str                   # "forwarded" | "duplicate_skipped" | "shrink_suppressed"
    aggregated_at: float | None         # when aggregator produced a sentence unit
    translate_started_at: float | None
    translate_completed_at: float | None
    display_emitted_at: float | None    # when caption was broadcast

    text_length: int = 0
    word_count: int = 0
    boundary_type: str = ""             # "punctuation" | "pause" | "nlp" | "forced" | "speaker_change"

    def latencies_ms(self) -> dict[str, float]:
        """Compute per-stage latencies in milliseconds."""
```

**PipelineMetricsCollector:**
- In-memory ring buffer (max 1000 ChunkTimelines per session)
- Running counters: `raw_messages_received`, `duplicates_skipped`, `chunks_forwarded`, `shrinks_suppressed`, `sentences_produced`, `translations_completed`, `translations_failed`, `too_short_filtered`
- Aggregation: `get_latency_percentiles()` → p50/p95/p99 per stage
- Structured log: every 30s emit `pipeline_metrics_snapshot` via structlog

**Integration points (where timestamps are stamped):**

| Stage | File | Method | Timestamp |
|-------|------|--------|-----------|
| Receive | `fireflies_client.py` | `_handle_transcript()` | `received_at` |
| Dedup | `live_caption_manager.py` | `handle_interim_update()` | `dedup_decided_at` |
| Aggregate | `sentence_aggregator.py` | `process_chunk()` | `aggregated_at` |
| Translate start | `rolling_window_translator.py` | `translate()` | `translate_started_at` |
| Translate end | `rolling_window_translator.py` | `translate()` | `translate_completed_at` |
| Display | `caption_buffer.py` | `add_caption()` | `display_emitted_at` |

**Exposure:**
- Merged into `GET /fireflies/sessions/{id}` stats response
- Structured log `pipeline_metrics_snapshot` every 30s
- Format compatible with future Prometheus histogram export

### Component 3: Sentence Boundary Tuning

**Threshold changes (validated via replay of captured data):**

| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| `pause_threshold_ms` | 800 | 600 | Real data shows natural pauses at 500-700ms between chunks |
| `max_buffer_seconds` | 5.0 | 4.0 | Conversational speech rarely has 5s continuous segments |
| `min_words_for_translation` | 3 | 2 | 1-2 word responses ("Yeah.", "Okay.") are valid; 2-word min prevents noise |
| `max_buffer_words` | 30 | 25 | Tighter sentences feel more natural for live captions |

**No filler filtering:** All text goes to translation regardless of content.

**Speaker-aware translation context:**
The `RollingWindowTranslator` prompt should include speaker identity (`speaker_name`) as signal. Different speakers use different terminology and patterns — the LLM can maintain per-speaker consistency if told who's speaking. The rolling window already tracks per-speaker sentence history; the enhancement is to weight the current speaker's prior sentences more heavily in the context window and include the speaker name in the translation prompt. This is a lightweight change to `_build_translation_prompt()`.

**Validation approach:**
1. Build a replay script that feeds captured JSONL through `SentenceAggregator`
2. Run with current thresholds → baseline report
3. Run with new thresholds → comparison report
4. Compare: sentence count, avg length, boundary distribution, fragments produced
5. Replay script becomes a permanent test fixture

## Files Modified

### Modify
- `src/services/pipeline/live_caption_manager.py` — grow filter logic, new state tracking, metrics counters
- `src/services/pipeline/config.py` — updated default thresholds, new metrics fields
- `src/services/pipeline/coordinator.py` — wire metrics collector, stamp timestamps at each stage
- `src/services/sentence_aggregator.py` — stamp `aggregated_at` timestamp, updated default thresholds
- `src/services/rolling_window_translator.py` — stamp translate start/end timestamps, add speaker name to translation prompt for per-speaker consistency
- `src/services/caption_buffer.py` — stamp `display_emitted_at` timestamp
- `src/clients/fireflies_client.py` — stamp `received_at` timestamp, expose dedup counters
- `src/routers/fireflies.py` — expose metrics in session stats response
- `src/models/fireflies.py` — add metrics fields to session model

### Create
- `src/services/pipeline/metrics.py` — `ChunkTimeline`, `PipelineMetricsCollector`
- `scripts/replay_captured_data.py` — Replay tool for boundary tuning validation
- Test files for new components

## Integration with Existing Plans

This design is an **additive refinement** to the Feb 20 Fireflies Real-Time Enhancement. All 18 tasks from that plan are already implemented and working. This design addresses gaps discovered during live data capture:

- **Feb 20 Feature Checklist items 1-2** (dedup, word-by-word captions): Working but need quality refinement (this design)
- **Feb 20 items 3-16**: Already complete and not affected by this design

The orchestration service `plan.md` should be updated to reflect this work under a new "Caption Pipeline Refinement" section.

## Success Criteria

1. **Interim captions grow smoothly** — text only gets longer or shows a clean correction. No visual jitter/shrinking.
2. **Metrics prove dedup works** — session stats show `duplicates_skipped` >> `chunks_forwarded`
3. **End-to-end latency is measurable** — p50/p95/p99 latencies per pipeline stage available via API
4. **Sentence boundaries validated** — replay of captured data produces natural-length sentences
5. **All UIs work** — captions.html, React CaptionOverlay, OBS overlay all show clean captions from the same backend events
