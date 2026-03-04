# Offline Diarization with VibeVoice-ASR — Design Document

**Date:** 2026-03-04
**Status:** Approved
**Author:** Thomas Patane + Claude

## Problem

Fireflies handles real-time transcription well, but its speaker attribution is based on calendar participant mapping rather than acoustic diarization. For meetings that need accurate "who said what" (e.g., dev weekly syncs, 1:1s with Eric), we need a higher-quality offline diarization system.

Additionally, some weekly sync meetings are being summarized in Chinese due to Fireflies auto-detecting the dominant spoken language. While the dashboard language setting can fix this going forward, we want the ability to re-process meetings locally with full control.

## Solution Overview

Add Microsoft VibeVoice-ASR as a **thin inference service** on a local GPU box, with all diarization logic (speaker mapping, merge, transcript enrichment, auto-rules) living in the **orchestration service** — matching the existing pattern where ML models are stateless inference endpoints and business logic stays centralized.

```
┌─────────────────────────────────────────────────────┐
│  Orchestration Service (existing)                    │
│                                                      │
│  ├── routers/diarization.py     (API endpoints)      │
│  ├── services/diarization/                           │
│  │   ├── pipeline.py            (job queue + worker) │
│  │   ├── speaker_mapper.py      (name assignment)    │
│  │   ├── speaker_merge.py       (dedup/merge)        │
│  │   ├── transcript_merge.py    (best-of selection)  │
│  │   └── rules.py               (auto-trigger rules) │
│  └── clients/vibevoice_client.py (HTTP client)       │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (LAN)
┌──────────────────────▼──────────────────────────────┐
│  VibeVoice vLLM Service (GPU box)                    │
│                                                      │
│  vLLM + VibeVoice plugin (Docker)                    │
│  POST /v1/chat/completions  (OpenAI-compatible)      │
│  GET  /v1/health                                     │
│                                                      │
│  No business logic. Stateless inference only.        │
└─────────────────────────────────────────────────────┘
```

## VibeVoice-ASR Model Details

- **Architecture:** 7B params, Qwen2 decoder + dual acoustic/semantic encoders
- **Capabilities:** Joint ASR + speaker diarization + timestamping in a single pass
- **Context window:** 64K tokens — up to 60 minutes of audio, no chunking
- **Languages:** 50+ with code-switching support (handles Chinese↔English in same meeting)
- **Hotwords:** Custom domain-specific terms for better recognition
- **License:** MIT
- **Serving:** vLLM plugin with OpenAI-compatible API, continuous batching
- **DER:** 4.28% on English (MLC-Challenge) — significantly better than pyannote 3.1 (~11-19%)

### GPU Requirements

| Quantization | Model Size | VRAM Needed | Quality |
|-------------|-----------|------------|---------|
| BF16 (full) | 18 GB | ~18-24 GB | Best |
| 8-bit | 9.9 GB | ~11 GB | Excellent |
| 4-bit (nf4) | 6.2 GB | ~7 GB | Very Good |

### Deployment (Docker)

```bash
docker run -d --gpus all --name vibevoice \
  --ipc=host -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py"
```

### Output Format

```json
{
  "segments": [
    {
      "speaker": 0,
      "start": 0.52,
      "end": 3.21,
      "text": "So about the deployment..."
    }
  ]
}
```

Options: `return_format="parsed"` (list of dicts) or `return_format="transcription_only"` (plain text).

### References

- HuggingFace: https://huggingface.co/microsoft/VibeVoice-ASR
- GitHub: https://github.com/microsoft/VibeVoice
- vLLM Integration: https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr.md
- Technical Report: https://arxiv.org/pdf/2601.18184
- Low-VRAM Quantized: https://huggingface.co/DevParker/VibeVoice7b-low-vram

## Data Flow

### Job Lifecycle

```
Trigger (auto-rule match after Fireflies sync, or manual from dashboard)
  │
  ▼
Create diarization_jobs record (status: "queued")
  │
  ▼
Download audio from Fireflies audio_url
  │  (signed URL with expiration — download promptly)
  │  Store in temp storage, record audio_size_bytes
  │  Status → "downloading"
  ▼
POST /v1/chat/completions to VibeVoice vLLM service
  │  Send audio + optional params (hotwords, language hint, max_speakers)
  │  Status → "processing"
  │  Long-running: ~5-15 min for 1hr meeting
  ▼
VibeVoice returns segments: [{speaker, start, end, text}, ...]
  │  Store raw_segments in job record
  ▼
Speaker Mapping Pipeline (status → "mapping")
  │  1. Cross-ref Fireflies participant list (timestamp overlap)
  │  2. Match against enrolled voice profiles (embedding similarity)
  │  3. Flag unresolved speakers for manual assignment
  ▼
Speaker Merge
  │  Detect over-segmented speakers (same voice, different IDs)
  │  Auto-merge high-confidence matches, flag ambiguous ones
  ▼
Transcript Merge ("best of")
  │  Align VibeVoice segments with Fireflies sentences by timestamp
  │  Keep Fireflies text, replace speaker labels with VibeVoice diarization
  │  Store VibeVoice raw output as separate source
  ▼
Update meeting record
  │  Enriched transcript → meeting_sentences (updated speaker labels)
  │  Raw VibeVoice output → meeting_data_insights (type: "diarization")
  │  Job status → "completed"
  ▼
Completion hook (future: webhook, notification, downstream automation)
```

### Auto-Trigger Rules

After Fireflies `_persist_transcript` completes, evaluate rules:

```json
{
  "diarization_rules": {
    "enabled": true,
    "participant_patterns": ["eric@*", "thomas@*"],
    "title_patterns": ["dev weekly*", "1:1*"],
    "min_duration_minutes": 5,
    "exclude_empty": true
  }
}
```

Meeting matches any rule → auto-queue diarization job.

## Orchestration Service — New Components

### Router: `src/routers/diarization.py`

```
POST /diarization/jobs                    # Manual trigger: submit meeting for diarization
GET  /diarization/jobs                    # List jobs (with status filter)
GET  /diarization/jobs/{id}               # Job detail + progress
POST /diarization/jobs/{id}/cancel        # Cancel queued/running job

GET  /diarization/rules                   # Get auto-trigger rules
PUT  /diarization/rules                   # Update rules

GET  /diarization/speakers                # List known speaker profiles
POST /diarization/speakers                # Enroll new speaker (voice sample + name)
PUT  /diarization/speakers/{id}           # Update speaker name/profile
POST /diarization/speakers/merge          # Merge two speaker profiles
DELETE /diarization/speakers/{id}         # Remove speaker profile

GET  /diarization/meetings/{id}/compare   # Side-by-side: Fireflies vs VibeVoice
POST /diarization/meetings/{id}/apply     # Apply diarization results to meeting transcript
```

### Client: `src/clients/vibevoice_client.py`

OpenAI-compatible client via vLLM:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://gpu-box:8000/v1")
```

### Pipeline: `src/services/diarization/`

| File | Purpose |
|------|---------|
| `pipeline.py` | Async job queue, worker loop, status tracking |
| `speaker_mapper.py` | SPEAKER_0 → "Eric" (three-strategy layered mapping) |
| `speaker_merge.py` | Detect & merge over-segmented speakers |
| `transcript_merge.py` | Align and merge Fireflies + VibeVoice transcripts |
| `rules.py` | Auto-trigger rule evaluation after Fireflies sync |

### Speaker Mapping — Three Strategies (layered)

1. **Fireflies cross-reference:** Align VibeVoice segments with Fireflies sentences by timestamp overlap. If Fireflies attributed "Eric" to a segment overlapping SPEAKER_0, assign that name.

2. **Voice profile matching:** Extract speaker embeddings from VibeVoice segments, compare against enrolled profiles using cosine similarity. High-confidence matches auto-assign.

3. **Manual assignment:** Dashboard UI for labeling unknowns. On save, optionally enrolls the voice profile for future auto-matching.

### Transcript Merge ("Best Of")

- Align by timestamp (VibeVoice start/end ↔ Fireflies start_time/end_time)
- Keep Fireflies text (unless VibeVoice text is significantly different — flag for review)
- Replace speaker labels with VibeVoice diarization
- Store both versions: enriched transcript in `meeting_sentences`, raw VibeVoice in `meeting_data_insights`
- Output format matches Fireflies sentence structure exactly (same display in dashboard)

## Database Schema

New migration: `010_diarization_jobs`

```sql
-- Diarization job tracking
CREATE TABLE diarization_jobs (
    id SERIAL PRIMARY KEY,
    meeting_id INTEGER REFERENCES meetings(id),
    status VARCHAR(20) DEFAULT 'queued',
    -- statuses: queued, downloading, processing, mapping, completed, failed
    triggered_by VARCHAR(20),
    -- 'auto_rule' or 'manual'
    rule_matched JSONB,
    audio_url TEXT,
    audio_size_bytes BIGINT,

    -- VibeVoice results
    raw_segments JSONB,
    detected_language VARCHAR(10),
    num_speakers_detected INTEGER,
    processing_time_seconds FLOAT,

    -- Speaker mapping state
    speaker_map JSONB,
    -- e.g. {0: {name: "Eric", confidence: 0.92, method: "voice_profile"}}
    unmapped_speakers INTEGER[],

    -- Merge state
    merge_applied BOOLEAN DEFAULT FALSE,
    merge_applied_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE INDEX idx_diarization_jobs_meeting ON diarization_jobs(meeting_id);
CREATE INDEX idx_diarization_jobs_status ON diarization_jobs(status);

-- Voice enrollment profiles
CREATE TABLE speaker_profiles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    embedding JSONB,
    -- Speaker embedding vector (stored as JSON array)
    -- Alternative: use pgvector VECTOR(256) if extension is available
    enrollment_source VARCHAR(50),
    -- 'manual', 'auto_from_diarization', 'fireflies'
    sample_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_speaker_profiles_email ON speaker_profiles(email);
```

## Configuration

Stored in `system_config` table, editable from dashboard:

```json
{
  "diarization": {
    "enabled": true,
    "vibevoice_url": "http://192.168.1.x:8000/v1",
    "hotwords": ["LiveTranslate", "Fireflies", "sprint"],
    "max_concurrent_jobs": 1,
    "auto_apply_threshold": 0.85,
    "rules": {
      "enabled": true,
      "participant_patterns": ["eric@*", "thomas@*"],
      "title_patterns": ["dev weekly*", "1:1*"],
      "min_duration_minutes": 5,
      "exclude_empty": true
    },
    "speaker_mapping": {
      "auto_enroll": true,
      "min_confidence_auto_assign": 0.80,
      "fireflies_crossref_enabled": true
    }
  }
}
```

## Dashboard Frontend

### Meeting Detail Enhancement (`/meetings/[id]`)

- **"Diarize" button** — triggers manual diarization job
- **Job status indicator** — progress bar (downloading → processing → mapping → done)
- **Speaker assignment panel** — inline UI for unmapped speakers after job completes
- **Source toggle** — switch between "Fireflies" and "Diarized" transcript view
- **Side-by-side compare** — both versions aligned by timestamp

### Diarization Hub Page (`/diarization`)

Four tabs:

**Active Jobs:** List of queued/running/recent diarization jobs with progress indicators.

**History:** Completed jobs with summary stats (speakers detected, processing time, merge status).

**Speakers:** Known speaker profiles with match counts, enrollment management, merge UI.
- Enroll new speakers (upload voice sample + name)
- Merge over-segmented speakers (drag-and-drop or checkbox)
- Edit speaker names/emails

**Rules:** Auto-trigger rule editor.
- Participant patterns (glob-style)
- Title patterns (glob-style)
- Duration threshold
- Enable/disable toggle

### Transcript Compare View

Side-by-side on meeting detail:

```
┌──── Fireflies ──────────┬──── VibeVoice (Diarized) ────┐
│ Speaker 1: "So about..." │ Eric: "So about the..."      │
│ Speaker 1: "I think we"  │ Eric: "I think we should..."  │
│ Speaker 2: "Yeah agreed"  │ Thomas: "Yeah agreed on..."   │
│                           │                               │
│              [Apply Diarized Version]                     │
└───────────────────────────┴───────────────────────────────┘
```

Diarized version uses identical sentence card format as Fireflies transcript — same speaker badges, timestamps, and layout. Seamless visual consistency.

## Architectural Notes

- **Whisper service is legacy** — with Fireflies handling real-time transcription, VibeVoice replaces the need for Whisper+pyannote diarization. Whisper service remains available as fallback but is no longer the primary path.
- **Sidecar model pattern** — ML models are stateless inference endpoints, business logic stays in orchestration. VibeVoice can be swapped for a better model later without touching orchestration code.
- **Voice profiles in Postgres** (not SQLite on GPU box) — accessible from dashboard, backed up with everything else.
- **vLLM serving** — continuous batching, OpenAI-compatible API, significantly faster inference than raw PyTorch.
