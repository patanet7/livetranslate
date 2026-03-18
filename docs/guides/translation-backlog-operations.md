# Translation Backlog Operations Runbook

This runbook covers the operational path for asynchronous meeting translation recovery.

It applies to both:

- loopback meetings that persist `meeting_chunks`
- Fireflies imports and synced transcripts that persist `meeting_sentences`

Both sources now converge on the same recovery pipeline and the same `meeting_translations` table.

## Operating Model

The current design is intentionally write-first:

- meeting transcript data is stored immediately
- translation may complete inline for some live paths
- any missing translations are recovered asynchronously through the shared `TranslationService`

This is the preferred shape for Fireflies ingestion because it keeps import latency low and avoids blocking transcript persistence on model availability.

## Why This Exists

Use the recovery path to handle:

- model or endpoint outages during ingest
- process restarts between transcript persistence and translation persistence
- backlog spikes after maintenance windows
- Fireflies transcript imports that landed sentences before translation caught up

## Primary Signals

Use these in order.

### UI

- Dashboard: compact `Translation Recovery` widget
- System Analytics: `Translation Recovery` tab

### API

- `GET /api/meetings/backlog`
- `GET /api/meetings/{meeting_id}/translation-status`
- `POST /api/meetings/{meeting_id}/translations/recover`
- `GET /api/system/metrics`

## Data Contract

Backlog accounting is split by transcript unit type:

- `pending_chunk_translation_count`
- `pending_sentence_translation_count`
- `pending_translation_count`
- `translation_count`

Interpretation:

- chunk backlog usually indicates loopback/live meeting final chunks
- sentence backlog usually indicates Fireflies-backed or imported transcript rows

## Normal State

Healthy steady state looks like this:

- `pending_translation_count` trends back to zero after ingest bursts
- `total_failed` in recovery counters remains zero or near zero
- per-meeting backlog clears without repeated manual replay

## Triage Workflow

1. Open the dashboard widget or `System Analytics -> Translation Recovery`.
2. Check fleet totals:
   - pending units
   - pending meetings
   - final chunk backlog
   - sentence backlog
3. Check recovery counters:
   - last run completed time
   - last run scope
   - total recovered
   - total failed
4. If backlog is concentrated in a few meetings, inspect those first.
5. If backlog is broad across many meetings, validate translation service health before replaying work.

## Manual Replay

Use per-meeting replay first. Do not treat fleet-wide backlog as a reason to mass-replay blindly.

### From UI

In `System Analytics -> Translation Recovery`:

- select the meeting row
- review chunk vs sentence backlog
- run `Recover Selected Meeting`

### From API

```bash
curl -X POST "http://localhost:3000/api/meetings/<meeting_id>/translations/recover?limit=500"
```

Expected response shape:

```json
{
  "meeting_id": "...",
  "before": {
    "pending_chunk_translation_count": 0,
    "pending_sentence_translation_count": 12,
    "pending_translation_count": 12
  },
  "recovery": {
    "scanned": 12,
    "recovered": 12,
    "failed": 0
  },
  "after": {
    "pending_chunk_translation_count": 0,
    "pending_sentence_translation_count": 0,
    "pending_translation_count": 0
  }
}
```

## Safety Rules

- Prefer per-meeting replay over broad replay.
- Validate translation endpoint health before repeated retries.
- Treat persistent `failed > 0` as a service/config issue, not an operator-click issue.
- Do not reload model-heavy services repeatedly while debugging. The runtime and test fixtures were hardened specifically to avoid duplicate model loads and memory crashes.

## Model Lifecycle / Memory Guardrails

When validating recovery in development or CI:

- reuse long-lived service processes where possible
- avoid starting multiple transcription or translation backends in the same test process
- prefer explicit teardown and cache cleanup over ad hoc restart loops
- if a test needs live models, skip cleanly when the service is unavailable rather than trying to boot multiple copies

This is especially important for local benchmark and integration workflows where repeated model initialization previously caused process instability.

## Fireflies Notes

Fireflies transcript data arrives in a different source format, but operationally it should be treated as first-class input to the same translation system.

Current behavior:

- Fireflies sync/import stores canonical meeting sentence rows first
- target languages are persisted on the meeting row
- missing translations are later recovered through the same shared recovery path used by loopback meetings

This means:

- backlog visibility is unified
- replay tooling is unified
- translation persistence is unified

## Escalation Conditions

Escalate beyond routine replay if any of the following are true:

- backlog grows across multiple refresh cycles while recovery runs continue
- `total_failed` or `last_run_failed` is non-zero and increasing
- only Fireflies meetings fail while loopback meetings recover
- only loopback chunk backlog grows while Fireflies sentence backlog remains clear
- recovery runs are stale and no longer updating timestamps

## Quick Verification Commands

```bash
curl http://localhost:3000/api/system/metrics | jq '.translation_backlog'
curl http://localhost:3000/api/meetings/backlog | jq '.summary'
curl http://localhost:3000/api/meetings/<meeting_id>/translation-status | jq '.translation_status'
```

## Related Guides

- [Translation Testing Guide](./translation-testing.md)
- [Quick Start Guide](./quick-start.md)
- [Database Setup Guide](./database-setup.md)
