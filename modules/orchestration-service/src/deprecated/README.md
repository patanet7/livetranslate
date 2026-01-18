# Deprecated Modules

These modules have been deprecated as part of the DRY Pipeline Audit (2026-01-17).

## Why Deprecated

All functionality has been consolidated into the unified pipeline:
- `TranscriptionPipelineCoordinator` - Central processing coordinator
- `SentenceAggregator` - Handles sentence boundary detection and speaker grouping
- Source-specific adapters (`FirefliesChunkAdapter`, `GoogleMeetChunkAdapter`, `AudioUploadChunkAdapter`)

## Deprecated Files

### `streaming_coordinator.py`
**Reason**: Replaced by `TranscriptionPipelineCoordinator` in `src/services/pipeline/`

The streaming coordinator was a monolithic processor that combined deduplication,
speaker grouping, and coordination. The new pipeline architecture separates concerns:
- Adapters handle source-specific conversion
- SentenceAggregator handles text aggregation with speaker awareness
- Coordinator orchestrates the flow

### `speaker_grouper.py`
**Reason**: Functionality merged into `SentenceAggregator`

The `SentenceAggregator` now handles:
- Per-speaker buffer management
- Speaker change detection (triggers flush)
- Consecutive segment merging within same speaker

### `segment_deduplicator.py`
**Reason**: Deduplication handled at adapter level

Each adapter (`FirefliesChunkAdapter`, etc.) assigns unique `chunk_id` values.
The pipeline tracks processed chunks to avoid duplicates.

## Migration Guide

If you were using these modules, migrate to:

```python
from services.pipeline import (
    TranscriptionPipelineCoordinator,
    PipelineConfig,
    FirefliesChunkAdapter,  # or appropriate adapter
)
from services.sentence_aggregator import SentenceAggregator
```

See `src/services/pipeline/` for the new unified architecture.

## Tests

Tests for deprecated modules moved to `tests/deprecated/`.
