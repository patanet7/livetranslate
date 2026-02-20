# Fireflies Real-Time Enhancement Design

**Date:** 2026-02-20
**Status:** Approved
**Architecture:** Layered Pipeline (Approach A)

## Problem Statement

The current Fireflies integration has critical issues:
1. **16x chunk duplication** — Fireflies sends ~16 interim updates per chunk_id, all treated as new chunks
2. **No real-time word-by-word captions** — captions only appear after full sentence aggregation + translation
3. **No persistence** — meetings not saved to database, insights lost
4. **No auto-connect** — must manually connect to each meeting
5. **Dashboard UX broken** — connect requires double-click, no inline caption preview
6. **Translation not configurable at runtime** — backend locked to env vars

## Architecture: Layered Pipeline

Three new layers added to the existing pipeline without restructuring:

```
Layer 1: DEDUP LAYER (in FirefliesRealtimeClient._handle_transcript)
  Fireflies Socket.IO → chunk_id tracking → emit interim/final events

Layer 2: DISPLAY LAYER (LiveCaptionManager)
  interim chunks → grow-in-place WebSocket broadcast (word-by-word)
  final chunks → SentenceAggregator → translation → caption display
  command detection → command router

Layer 3: PERSISTENCE LAYER (MeetingStore)
  real-time chunks → DB storage during meeting
  post-meeting webhook → full Fireflies download (transcript + insights)
  → PostgreSQL with full-text search
```

## Layer 1: Chunk Deduplication

### Location
`modules/orchestration-service/src/clients/fireflies_client.py` — `_handle_transcript()`

### Mechanism
- Track `_pending_chunks: dict[str, FirefliesChunk]` — latest version per chunk_id
- Track `_pending_text: dict[str, str]` — last text per chunk_id for pure-duplicate detection

### Flow
```
Message arrives (chunk_id + text)
  │
  ├─ Same chunk_id, same text → SKIP (pure duplicate)
  ├─ Same chunk_id, different text → UPDATE pending, emit on_live_update(chunk)
  └─ New chunk_id → emit on_live_update(new), FINALIZE all other pending chunks via on_transcript()
```

### Finalization
- **Trigger:** New chunk_id arrival finalizes all other pending chunks (no timer delay)
- **Disconnect:** Flush all remaining pending chunks as final
- **Stats:** Track both `raw_messages_received` and `chunks_received` (unique chunk_ids)

### New Callback
```python
# Added to FirefliesRealtimeClient
on_live_update: Callable[[FirefliesChunk, bool], Awaitable[None]]
# Second param is_final: True when chunk is being finalized
```

## Layer 2: Display (LiveCaptionManager)

### Two Display Tracks

**Track 1 — Interim captions (word-by-word, grow in place):**
- Every `on_live_update` call broadcasts to WebSocket:
  ```json
  {"event": "interim_caption", "chunk_id": "65198", "text": "This is how...", "speaker_name": "Thomas", "speaker_color": "#4CAF50"}
  ```
- `captions.html` replaces DOM element with matching `chunk_id` (text grows in place)

**Track 2 — Final captions + translation:**
- Finalized chunks → SentenceAggregator → RollingWindowTranslator → CaptionBuffer
- Broadcasts via existing `caption_added`/`caption_updated` events
- Final caption replaces corresponding interim caption(s)

### Display Modes (switchable at runtime)
| Mode | Shows |
|------|-------|
| `english` | Only interim captions (Track 1) |
| `translated` | Only final translated captions (Track 2), hide interim |
| `both` | Interim growing text + translation below once available (default) |

### Mode Switching
- Dashboard toggle button
- WebSocket message: `{"event": "set_mode", "mode": "both"}`
- Voice command (experimental): "LiveTranslate switch to Chinese"

### Command Interception
Dashboard UI provides buttons/inputs for common commands:
- Change target language
- Switch display mode
- Pause/resume translation

**Voice commands (experimental, opt-in):**
- Detect spoken phrases like "LiveTranslate switch to Chinese" in transcript stream
- Must match exact prefix to avoid false triggers
- Disabled by default (`VOICE_COMMANDS_ENABLED=false`)

## Layer 3: Persistence (MeetingStore)

### Database Schema

```sql
-- Core meeting record
CREATE TABLE meetings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fireflies_transcript_id TEXT,
    title TEXT,
    meeting_link TEXT,
    organizer_email TEXT,
    participants JSONB DEFAULT '[]',
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration INTEGER, -- seconds
    source TEXT NOT NULL DEFAULT 'fireflies', -- 'fireflies', 'upload', 'whisper'
    status TEXT NOT NULL DEFAULT 'live', -- 'live', 'completed', 'processing'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Raw transcript chunks (deduplicated)
CREATE TABLE meeting_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    text TEXT NOT NULL,
    speaker_name TEXT,
    start_time REAL,
    end_time REAL,
    is_command BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(meeting_id, chunk_id)
);

-- Aggregated sentences
CREATE TABLE meeting_sentences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    speaker_name TEXT,
    start_time REAL,
    end_time REAL,
    boundary_type TEXT, -- 'punctuation', 'pause', 'speaker_change', 'nlp', 'forced'
    chunk_ids JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Translations of sentences
CREATE TABLE meeting_translations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sentence_id UUID REFERENCES meeting_sentences(id) ON DELETE CASCADE,
    translated_text TEXT NOT NULL,
    target_language TEXT NOT NULL,
    source_language TEXT DEFAULT 'en',
    confidence REAL DEFAULT 1.0,
    translation_time_ms REAL DEFAULT 0.0,
    model_used TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- AI-generated insights (extensible JSONB)
CREATE TABLE meeting_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    insight_type TEXT NOT NULL,
    -- Types: 'summary', 'action_items', 'keywords', 'topics', 'sentiment',
    --        'speaker_analytics', 'ai_filters', 'attendance', 'media',
    --        'outline', 'questions', 'decisions', 'custom'
    content JSONB NOT NULL,
    source TEXT NOT NULL DEFAULT 'fireflies', -- 'fireflies', 'ollama', 'manual'
    model_used TEXT,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Speaker metadata per meeting
CREATE TABLE meeting_speakers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    speaker_name TEXT NOT NULL,
    email TEXT,
    talk_time_seconds REAL DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    sentiment_score REAL,
    analytics JSONB, -- filler_words, wpm, longest_monologue, etc.
    UNIQUE(meeting_id, speaker_name)
);

-- Indexes
CREATE INDEX idx_meetings_ff_id ON meetings(fireflies_transcript_id);
CREATE INDEX idx_meetings_status ON meetings(status);
CREATE INDEX idx_meetings_source ON meetings(source);
CREATE INDEX idx_chunks_meeting ON meeting_chunks(meeting_id);
CREATE INDEX idx_sentences_meeting ON meeting_sentences(meeting_id);
CREATE INDEX idx_translations_sentence ON meeting_translations(sentence_id);
CREATE INDEX idx_insights_meeting ON meeting_insights(meeting_id);
CREATE INDEX idx_insights_type ON meeting_insights(insight_type);
CREATE INDEX idx_speakers_meeting ON meeting_speakers(meeting_id);
-- Full-text search
CREATE INDEX idx_chunks_text_search ON meeting_chunks USING gin(to_tsvector('english', text));
CREATE INDEX idx_sentences_text_search ON meeting_sentences USING gin(to_tsvector('english', text));
```

### Data Flow

**During live meeting:**
1. Deduplicated final chunks → `meeting_chunks`
2. Aggregated sentences → `meeting_sentences`
3. Translations → `meeting_translations`

**Post-meeting (webhook or polling):**
1. Fireflies webhook fires OR polling detects meeting ended
2. Call expanded `transcript(id)` GraphQL to fetch ALL Fireflies data:
   - summary (overview, action_items, keywords, outline, shorthand_bullet, bullet_gist, gist, short_summary, short_overview, meeting_type, topics_discussed, transcript_chapters)
   - analytics (sentiments, categories, speakers)
   - sentences with ai_filters (task, pricing, metric, question, date_and_time, sentiment)
   - meeting_attendees, meeting_attendance
   - transcript_url, audio_url, video_url
3. Store each category in `meeting_insights` with appropriate `insight_type`
4. Optionally run Ollama insights on transcript (configurable)

**Upload flow:**
1. Upload audio/transcript via dashboard or API
2. Store in same schema with `source='upload'`
3. Run Ollama insights if configured

## Auto-Connect

### Startup Flow
```
Orchestration service starts
  → Poll active_meetings GraphQL every FIREFLIES_POLL_INTERVAL seconds (default 30)
  → New meeting detected → auto-create session → connect Socket.IO
  → Meeting gone from active list → finalize session → download full transcript
```

### Manual Connect Options
1. **Paste meeting link:** POST with meeting URL → `addToLiveMeeting` mutation → Fireflies bot joins → poll for transcript_id → auto-connect
2. **Connect to existing:** POST with transcript_id → direct Socket.IO connect

### Configuration
```bash
FIREFLIES_AUTO_CONNECT=true
FIREFLIES_POLL_INTERVAL=30
```

## Runtime Translation Backend Config

### API Endpoint
`PUT /fireflies/config/translation`

```json
{
  "backend": "ollama",
  "model": "qwen2.5:3b",
  "base_url": "http://localhost:11434/v1",
  "target_language": "zh",
  "temperature": 0.3,
  "max_tokens": 2048,
  "quality_threshold": 0.7
}
```

### Fallback Chain (configurable)
```
Primary: Ollama (qwen2.5:3b, local)
  → Fallback 1: vLLM (if GPU available)
  → Fallback 2: Groq API (if API key set)
  → Fallback 3: OpenAI (if API key set)
```

### Hot-Reload
- Settings stored in memory + persisted to DB
- Translation service has hot-reload endpoint: `POST /api/config/update`
- No restart required to switch backends

## Dashboard UX Enhancements

1. **Single-click connect** — save config AND connect in one action
2. **Paste meeting link** — input field to paste Google Meet URL, auto-invites Fireflies bot
3. **Live status bar** — connection status, chunks/sec, speakers detected
4. **Inline caption preview** — see captions directly in dashboard
5. **Mode toggle** — buttons for English / Translated / Both
6. **Translation config panel** — backend selector, model picker, language selector
7. **Meeting history** — list of past meetings with view/download/insights
8. **Upload area** — drag-and-drop for transcript/audio upload

## Expanded Fireflies GraphQL Queries

The existing `TRANSCRIPT_DETAIL_QUERY` must be expanded to capture all Fireflies data:

```graphql
query TranscriptFull($id: String!) {
  transcript(id: $id) {
    id
    title
    date
    duration
    transcript_url
    audio_url
    video_url
    meeting_link
    host_email
    organizer_email
    participants

    sentences {
      index
      text
      raw_text
      start_time
      end_time
      speaker_name
      ai_filters {
        task
        pricing
        metric
        question
        date_and_time
        text_cleanup
        sentiment
      }
    }

    summary {
      overview
      action_items
      outline
      shorthand_bullet
      keywords
      bullet_gist
      gist
      short_summary
      short_overview
      meeting_type
      topics_discussed
      transcript_chapters
    }

    meeting_attendees {
      displayName
      email
      phoneNumber
      location
    }

    meeting_attendance {
      name
      join_time
      leave_time
    }

    analytics {
      sentiments {
        negative_pct
        neutral_pct
        positive_pct
      }
      categories {
        questions
        date_times
        metrics
        tasks
      }
      speakers {
        duration
        word_count
        longest_monologue
        filler_words
        questions
        words_per_minute
      }
    }
  }
}
```

## Configuration Summary

### Env Vars (orchestration .env)
```bash
# Fireflies
FIREFLIES_API_KEY=<key>
FIREFLIES_AUTO_CONNECT=true
FIREFLIES_POLL_INTERVAL=30
FIREFLIES_WEBHOOK_ENABLED=true
FIREFLIES_WEBHOOK_URL=http://localhost:3000/fireflies/webhook

# Display
DEFAULT_DISPLAY_MODE=both
DEFAULT_TARGET_LANGUAGE=zh

# Persistence
MEETING_AUTO_SAVE=true
MEETING_DOWNLOAD_ON_COMPLETE=true
MEETING_DOWNLOAD_INSIGHTS=true

# Voice Commands (experimental)
VOICE_COMMANDS_ENABLED=false
VOICE_COMMAND_PREFIX=LiveTranslate
```

### PipelineConfig additions
```python
display_mode: str = "both"
enable_interim_captions: bool = True
enable_persistence: bool = True
voice_commands_enabled: bool = False
voice_command_prefix: str = "LiveTranslate"
```

## Files to Modify/Create

### Modify
- `modules/orchestration-service/src/clients/fireflies_client.py` — dedup layer, expanded GraphQL queries, remove temp debug code
- `modules/orchestration-service/src/routers/fireflies.py` — auto-connect, webhook, config API, dashboard UX callbacks
- `modules/orchestration-service/src/services/pipeline/config.py` — new config fields
- `modules/orchestration-service/src/services/pipeline/coordinator.py` — interim caption support
- `modules/orchestration-service/static/fireflies-dashboard.html` — UX overhaul
- `modules/orchestration-service/static/captions.html` — interim caption support
- `modules/orchestration-service/.env` — new config vars

### Create
- `modules/orchestration-service/src/services/live_caption_manager.py` — LiveCaptionManager
- `modules/orchestration-service/src/services/meeting_store.py` — MeetingStore persistence
- `modules/orchestration-service/src/services/command_interceptor.py` — command detection/execution
- `scripts/meeting-schema.sql` — new database schema
- `modules/orchestration-service/src/routers/meetings.py` — meeting history/upload API

## Feature Checklist

1. [ ] Chunk deduplication (fix 16x duplication)
2. [ ] Word-by-word captions (grow-in-place, no delay)
3. [ ] Translation display (Chinese via Ollama)
4. [ ] Switchable display modes (English/Chinese/Both)
5. [ ] Dashboard commands (language, mode, pause)
6. [ ] Voice commands (experimental, opt-in)
7. [ ] DB persistence for meetings
8. [ ] Post-meeting webhook/polling for full download
9. [ ] Download and save all Fireflies AI insights
10. [ ] Auto-connect to active meetings on startup
11. [ ] Dashboard UX fixes (single-click connect, inline captions)
12. [ ] Paste meeting link to invite Fireflies bot
13. [ ] Upload transcripts from other meetings
14. [ ] Runtime-configurable translation backend
15. [ ] Expanded GraphQL queries for full Fireflies data
16. [ ] Meeting history with search
