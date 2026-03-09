# System Completion & Business Insights Chat — Design Document

**Date:** 2026-03-09
**Status:** Approved
**Scope:** Fix all stubs, complete diarization, add data export, build business insights chat, resolve tech debt + infrastructure issues

---

## Overview

This design covers 5 work streams to bring LiveTranslate from prototype (~50% complete) to a fully functional system with no stubs, complete data export, and a new general-purpose business insights chat with independent LLM model selection.

| # | Work Stream | Size | Dependencies |
|---|-------------|------|-------------|
| WS1 | Diarization Backend Completion | Medium | WS4/WS5 first |
| WS2 | Data Export / Download | Medium | WS4/WS5 first |
| WS3 | Business Insights Chat | Large | WS4/WS5 first |
| WS4 | Tech Debt Remediation | Small | None |
| WS5 | Critical Infrastructure Fixes | Small | None |

**Execution order:** WS4 + WS5 first (clean foundation), then WS1 + WS2 + WS3 in parallel.

---

## WS1: Diarization Backend Completion

### Current State
- 8 stub endpoints in `routers/diarization.py` with `# TODO` comments returning hardcoded/empty data
- `diarization_jobs` and `speaker_profiles` tables exist (Alembic migration 010)
- `DiarizationPipeline` manages in-memory state but has no worker loop
- Supporting modules exist: `speaker_mapper.py`, `speaker_merge.py`, `transcript_merge.py`, `auto_trigger.py`, `rules.py`
- Design doc: `docs/plans/2026-03-04-offline-diarization-vibevoice-design.md`

### Changes

#### 1. Speaker Profile CRUD
Wire all speaker endpoints to real SQLAlchemy queries against `speaker_profiles` table:

```
GET    /speakers           → SELECT with filters (name, meeting_id, confidence threshold)
POST   /speakers           → INSERT into speaker_profiles (name, embedding, enrollment_source)
PUT    /speakers/{id}      → UPDATE speaker_profiles (name, metadata, merge target)
POST   /speakers/merge     → Merge profiles: update embedding (weighted avg), update diarization_jobs.speaker_maps, soft-delete source profile
DELETE /speakers/{id}      → Soft-delete from speaker_profiles (set is_active=false)
```

#### 2. Rules Persistence
Store diarization rules in `system_config` table with key `diarization_rules`:

```
GET    /rules              → Read from system_config WHERE key='diarization_rules'
PUT    /rules              → Upsert to system_config with JSON-serialized DiarizationRules
```

Uses existing `DiarizationRules` Pydantic model from `models/diarization.py`.

#### 3. Comparison Endpoint
Wire to existing `transcript_merge.py`:

```
POST   /compare            → Fetch Fireflies sentences + VibeVoice segments from DB
                            → Run TranscriptMerger.align()
                            → Return aligned comparison with confidence scores
```

#### 4. Pipeline Worker
Add asyncio background task registered in orchestration startup:

```python
async def diarization_worker():
    """Background worker that processes queued diarization jobs."""
    while True:
        # Query diarization_jobs WHERE status='queued' ORDER BY created_at LIMIT 1
        job = await get_next_queued_job(db)
        if job:
            await process_job(job)  # download → vibevoice → speaker_map → persist
        else:
            await asyncio.sleep(5)
```

Worker state machine:
```
queued → downloading (fetch audio URL) → processing (send to VibeVoice-ASR)
→ mapping (run speaker_mapper) → completed | failed
```

Each state transition updates the `diarization_jobs` row with status + timestamps.

#### 5. Pipeline ↔ Database
Replace in-memory `active_jobs: dict` in `DiarizationPipeline` with actual database queries. The pipeline becomes a thin orchestrator over DB state, not a state container.

### Files Modified
- `modules/orchestration-service/src/routers/diarization.py` — replace all 8 stubs
- `modules/orchestration-service/src/services/diarization/pipeline.py` — add worker, wire to DB
- `modules/orchestration-service/src/services/diarization/db.py` — new: SQLAlchemy query helpers
- `modules/orchestration-service/src/main_fastapi.py` — register background worker on startup

---

## WS2: Data Export / Download

### Export Formats

| Format | MIME Type | Use Case |
|--------|-----------|----------|
| TXT | text/plain | Speaker-attributed transcript with timestamps |
| SRT | application/x-subrip | Standard SubRip subtitles (sequence + timecodes + text) |
| VTT | text/vtt | WebVTT subtitles with optional speaker cues |
| JSON | application/json | Full structured data (segments, speakers, translations, confidence) |
| PDF | application/pdf | Formatted meeting report via `reportlab` |

### API Endpoints

New router: `routers/export.py` mounted at `/api/export`

```
GET /api/export/meetings/{meeting_id}/transcript?format=srt|vtt|txt|json|pdf
    → StreamingResponse with Content-Disposition: attachment

GET /api/export/meetings/{meeting_id}/translations?format=srt|vtt|txt|json&lang=es
    → Translation export in specified language, same format options

GET /api/export/meetings/{meeting_id}/audio
    → Stream audio file (from stored path in audio_files table)
    → Content-Type based on file extension

GET /api/export/meetings/{meeting_id}/archive
    → ZIP bundle containing: transcript.txt, transcript.srt, translations/<lang>.srt, audio.<ext>, metadata.json
```

All endpoints return `StreamingResponse` with appropriate `Content-Disposition` headers for browser download.

### Export Service

New service: `services/export_service.py`

Conversion functions:
- `to_srt(segments: list[TranscriptSegment]) -> str`
- `to_vtt(segments: list[TranscriptSegment]) -> str`
- `to_txt(segments: list[TranscriptSegment], include_timestamps: bool) -> str`
- `to_json(meeting: Meeting, segments, translations) -> str`
- `to_pdf(meeting: Meeting, segments, translations) -> bytes`
- `to_zip(meeting_id: str) -> AsyncIterator[bytes]`

SRT format:
```
1
00:00:01,000 --> 00:00:04,500
[Speaker A] Hello everyone, let's begin.

2
00:00:05,200 --> 00:00:08,100
[Speaker B] Thanks for joining.
```

### Dashboard Integration
- Add download dropdown to meeting detail page (`/(app)/meetings/[id]`)
- Format picker: SRT, VTT, TXT, PDF, JSON
- Language selector for translation exports
- "Download All" ZIP button

### Files
- New: `modules/orchestration-service/src/routers/export.py`
- New: `modules/orchestration-service/src/services/export_service.py`
- New dependency: `reportlab` in orchestration service `pyproject.toml`
- Modified: `modules/dashboard-service/src/routes/(app)/meetings/[id]/+page.svelte` — download UI
- New: `modules/dashboard-service/src/lib/api/export.ts` — TypeScript client

---

## WS3: Business Insights Chat

### Architecture Overview

```
Dashboard Chat UI
       │
       │  REST + SSE
       v
Orchestration Service
  ├── routers/chat.py          (API endpoints)
  ├── services/llm/            (provider-agnostic adapter layer)
  │   ├── adapter.py           (abstract interface)
  │   ├── registry.py          (provider registry + model discovery)
  │   ├── tool_executor.py     (tool-calling engine)
  │   └── providers/           (one per backend)
  │       ├── ollama.py
  │       ├── vllm.py
  │       ├── groq.py
  │       ├── openai.py
  │       ├── anthropic.py
  │       └── openai_compat.py
  └── services/chat_tools.py   (business insight tool definitions)
       │
       │  SQL queries via SQLAlchemy
       v
  PostgreSQL (meetings, transcripts, translations, speakers, etc.)
```

### LLM Adapter Layer

Completely independent from the Translation Service. Each provider implements:

```python
class LLMAdapter(ABC):
    """Provider-agnostic LLM interface."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """Send messages, optionally with tool definitions. Returns response + any tool calls."""

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming variant. Yields text chunks and tool call chunks."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Discover available models from this provider."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is reachable."""
```

**Provider registry** (`registry.py`):
- Maintains configured providers with their connection info
- Auto-discovers available models on demand
- Settings stored in `system_config` table (API keys encrypted via Fernet with SECRET_KEY)

### Tool-Calling Engine

Pre-built tools the LLM can invoke:

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `query_meetings` | Search/filter/count meetings | `date_from`, `date_to`, `participant`, `language`, `limit` |
| `get_meeting_details` | Get full details for a specific meeting | `meeting_id` |
| `get_meeting_summary` | Get or generate transcript summary | `meeting_id` |
| `get_translation_stats` | Translation volume, quality scores, language pairs | `date_from`, `date_to`, `language_pair` |
| `get_speaker_analytics` | Speaker participation, talk time, frequency | `date_from`, `date_to`, `speaker_name` |
| `get_language_distribution` | Language pair usage over time | `date_from`, `date_to`, `group_by` |
| `get_diarization_stats` | Job status counts, speaker profile summary | None |
| `get_system_health` | Service status, uptime, error rates | None |
| `get_usage_trends` | Time-series: meetings/day, minutes transcribed, translations/day | `date_from`, `date_to`, `metric`, `interval` |
| `search_transcripts` | Full-text search across all meeting transcripts | `query`, `date_from`, `date_to`, `limit` |

Each tool is defined with a JSON Schema for parameters (compatible with OpenAI/Anthropic tool-calling format).

**Tool execution flow:**
1. User sends message
2. Backend prepends system prompt + conversation history + tool definitions
3. LLM responds with either text or tool calls
4. If tool calls: execute against DB, append results, send back to LLM
5. LLM synthesizes final natural language response
6. Max 3 tool-call rounds per message to prevent infinite loops

**Fallback for models without native tool support:**
Tools are injected into the system prompt as structured descriptions. The LLM's response is parsed for `<tool_call>` blocks. This works with basic Ollama models.

### Chat Settings (persisted in `system_config`)

```python
class ChatSettings(BaseModel):
    """Chat LLM configuration — independent from translation settings."""
    provider: Literal['ollama', 'vllm', 'groq', 'openai', 'anthropic', 'custom'] = 'ollama'
    model: str = 'llama3.1:8b'
    api_key: str | None = None          # encrypted at rest
    base_url: str | None = None         # for custom/self-hosted endpoints
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str | None = None    # custom override (default prompt included)
```

### API Endpoints

New router: `routers/chat.py` mounted at `/api/chat`

```
# Provider/Model Management
GET    /api/chat/providers                        → List configured providers with health status
GET    /api/chat/providers/{provider}/models       → List available models (calls adapter.list_models())
GET    /api/chat/settings                          → Get current chat settings
PUT    /api/chat/settings                          → Save chat settings

# Conversations (general business insights, not tied to a single session)
POST   /api/chat/conversations                     → Start new conversation
GET    /api/chat/conversations                     → List conversations (paginated)
GET    /api/chat/conversations/{id}                → Get conversation with full message history
DELETE /api/chat/conversations/{id}                → Delete conversation

# Messages
POST   /api/chat/conversations/{id}/messages       → Send message, get response (sync)
POST   /api/chat/conversations/{id}/messages/stream → Send message, get SSE streaming response
GET    /api/chat/conversations/{id}/suggestions     → Get contextual suggested questions
```

### Database

New Alembic migration `011_chat_conversations`:

```sql
CREATE TABLE chat_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT,
    model_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    system_prompt TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES chat_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_calls JSONB,        -- tool invocations made by the LLM
    tool_results JSONB,      -- results returned by tool execution
    model_name TEXT,          -- which model generated this (for assistant messages)
    tokens_used INTEGER,
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_chat_messages_conversation ON chat_messages(conversation_id, created_at);
CREATE INDEX idx_chat_conversations_updated ON chat_conversations(updated_at DESC);
```

### Dashboard UI

New page: `/(app)/chat/+page.svelte`

Layout:
```
┌─────────────────────────────────────────────────────────────┐
│  Business Insights Chat                        ⚙️ Settings  │
├───────────────┬─────────────────────────────────────────────┤
│               │                                             │
│ Conversations │  Model: claude-sonnet-4-20250514 (Anthropic)           │
│ sidebar       │                                             │
│               │  [Message history with tool call indicators]│
│ - Meeting     │                                             │
│   Analytics   │  🔧 Queried meetings (found 12 this week)  │
│               │                                             │
│ - Usage       │  📊 You had 12 meetings this week, up 20%  │
│   Trends      │     from last week. Most common language    │
│               │     pair was EN→ES (45%). Average           │
│ - Speaker     │     translation quality: 0.87/1.0           │
│   Insights    │                                             │
│               │  ┌─────────────────────────────────────┐    │
│ [+ New Chat]  │  │ Ask about your business data...     │ ➤  │
│               │  └─────────────────────────────────────┘    │
└───────────────┴─────────────────────────────────────────────┘
```

Settings drawer (slide-out panel):
- **Provider** dropdown: Ollama, vLLM, Groq, OpenAI, Anthropic, Custom
- **Model** picker: dynamically loaded from `/api/chat/providers/{provider}/models`
- **API Key** input: password field, for cloud providers
- **Base URL**: for custom/self-hosted endpoints
- **Temperature** slider: 0.0 – 2.0
- **Max Tokens**: number input
- **System Prompt**: textarea with default shown as placeholder

Components:
- `ChatMessage.svelte` — renders user/assistant/tool messages with markdown
- `ChatInput.svelte` — textarea with send button, suggestion chips
- `SettingsDrawer.svelte` — slide-out settings panel
- `ConversationList.svelte` — sidebar with conversation history
- `ToolCallIndicator.svelte` — shows when the LLM is querying data

### Files
- New: `modules/orchestration-service/src/services/llm/__init__.py`
- New: `modules/orchestration-service/src/services/llm/adapter.py`
- New: `modules/orchestration-service/src/services/llm/registry.py`
- New: `modules/orchestration-service/src/services/llm/tool_executor.py`
- New: `modules/orchestration-service/src/services/llm/providers/ollama.py`
- New: `modules/orchestration-service/src/services/llm/providers/vllm.py`
- New: `modules/orchestration-service/src/services/llm/providers/groq.py`
- New: `modules/orchestration-service/src/services/llm/providers/openai.py`
- New: `modules/orchestration-service/src/services/llm/providers/anthropic.py`
- New: `modules/orchestration-service/src/services/llm/providers/openai_compat.py`
- New: `modules/orchestration-service/src/services/chat_tools.py`
- New: `modules/orchestration-service/src/routers/chat.py`
- New: `modules/orchestration-service/src/models/chat.py` (Pydantic models)
- New: `modules/orchestration-service/alembic/versions/011_chat_conversations.py`
- New: `modules/dashboard-service/src/routes/(app)/chat/+page.svelte`
- New: `modules/dashboard-service/src/routes/(app)/chat/+page.server.ts`
- New: `modules/dashboard-service/src/lib/api/chat.ts`
- New: `modules/dashboard-service/src/lib/components/chat/ChatMessage.svelte`
- New: `modules/dashboard-service/src/lib/components/chat/ChatInput.svelte`
- New: `modules/dashboard-service/src/lib/components/chat/SettingsDrawer.svelte`
- New: `modules/dashboard-service/src/lib/components/chat/ConversationList.svelte`
- New: `modules/dashboard-service/src/lib/components/chat/ToolCallIndicator.svelte`
- Modified: `modules/dashboard-service/src/lib/components/layout/Sidebar.svelte` — add Chat nav item
- New dependencies: `anthropic`, `openai` in orchestration `pyproject.toml`

---

## WS4: Tech Debt Remediation

Execute all 8 tasks from `docs/plans/2026-03-02-tech-debt-remediation-plan.md`:

1. **Fix `nativeName` → `native` type mismatch** in dashboard config types
2. **Remove dead API methods** (`batchTranslate`, `detectLanguage`) from dashboard
3. **Delete 6 orphaned files** (~1,567 lines) — see Appendix A for preservation
4. **Fix dashboard health store** response shape + WS_BASE fallback
5. **Standardize service URL env vars** (AUDIO_SERVICE_URL → WHISPER_SERVICE_URL)
6. **Clean up TODO stubs** in bot/webcam routers (→ 501 Not Implemented)
7. **Make auth bypass explicit** with AUTH_ENABLED flag
8. **Run full verification suite**

---

## WS5: Critical Infrastructure Fixes

### 9. Fix Dockerfile Python Version
All Dockerfiles using `python:3.14-slim` → `python:3.12-slim`:
- `modules/orchestration-service/Dockerfile`
- `modules/translation-service/Dockerfile`
- `modules/whisper-service/Dockerfile` (verify)

### 10. Standardize on UV
- `modules/translation-service/Dockerfile`: Remove PDM, use UV
- `compose.local.yml`: Remove `POETRY_INSTALL_ARGS` build arg references

### 11. Add Circuit Breaker to TranslationServiceClient
Port `CircuitBreaker` + `RetryManager` pattern from `AudioServiceClient`:
- File: `modules/orchestration-service/src/clients/translation_service_client.py`
- Add: `CircuitBreaker(failure_threshold=5, recovery_timeout=30, success_threshold=2)`
- Add: `RetryManager(max_retries=3, base_delay=0.5, max_delay=10, backoff_factor=2)`

### 12. Remove Phantom Speaker Service
Remove all references to `speaker-service` (port 5002) from:
- `docker-compose.comprehensive.yml`
- `docker-compose.dev.yml`
- Any other compose files that reference it

### 13. Fix WebSocket Message Queue Drain
File: `modules/frontend-service/src/store/slices/websocketSlice.ts`
The `processMessageQueue` reducer currently clears the queue without re-sending messages.
Fix: dispatch a thunk that iterates the queue and calls `websocketRef.send()` for each message.

---

## Appendix A: Orphaned Code Preservation

Before deleting orphaned files (Task 3 of WS4), their functionality is documented here for future reference.

### 1. `src/routers/seamless.py` (~140 lines)
**Purpose:** SeamLess M4T proxy router — forwarded requests to a Facebook SeamlessM4T translation model service.
**Endpoints:**
- `POST /api/seamless/translate` — text translation via SeamlessM4T
- `POST /api/seamless/speech-to-text` — S2T via SeamlessM4T
- `GET /api/seamless/languages` — supported language list
**Why orphaned:** Never registered in `main_fastapi.py`. SeamlessM4T integration was abandoned in favor of the vLLM/Ollama multi-backend approach in the Translation Service.
**Recovery:** If SeamlessM4T support is desired, re-implement as a provider in the Translation Service's `model_manager.py` rather than a separate router.

### 2. `src/gateway/api_gateway.py` (~622 lines)
**Purpose:** Standalone API gateway class with request routing, rate limiting, load balancing, circuit breaker, and request/response transformation.
**Key classes:** `APIGateway`, `RouteConfig`, `LoadBalancer`, `RateLimiter`
**Why orphaned:** Never imported. The gateway functionality was partially absorbed into FastAPI middleware (rate limiting) and service clients (circuit breaker). A proper API gateway (nginx/Traefik) is recommended for production instead.
**Recovery:** Useful as reference for implementing an API gateway layer. The `LoadBalancer` class has round-robin and least-connections algorithms.

### 3. `src/dashboard/real_time_dashboard.py` (~421 lines)
**Purpose:** Standalone Flask-based real-time dashboard with SocketIO for live metrics display.
**Features:** Live transcription feed, translation quality graphs, service health panels, WebSocket connection counter.
**Why orphaned:** Replaced by the SvelteKit dashboard service (`modules/dashboard-service/`). Never imported from the FastAPI app.
**Recovery:** The real-time metric rendering logic could inform Grafana dashboard panel design.

### 4. `src/utils/dependency_check.py` (~170 lines)
**Purpose:** Runtime dependency checker — verifies Python packages, system tools (ffmpeg, etc.), and service connectivity at startup.
**Features:** `check_python_packages()`, `check_system_tools()`, `check_service_connectivity()`, colored console output.
**Why orphaned:** Never imported. Startup checks are now handled by Docker health checks and the `HealthMonitor`.
**Recovery:** Could be useful as a CLI diagnostic tool (`python -m dependency_check`).

### 5. `src/main.py` (~104 lines)
**Purpose:** Legacy Flask entry point for the orchestration service.
**Why orphaned:** Replaced by `main_fastapi.py`. Flask was migrated to FastAPI.
**Recovery:** Not needed. FastAPI is the canonical framework.

### 6. `src/routers/audio.py` (~110 lines)
**Purpose:** Legacy monolithic audio router before the `routers/audio/` package split.
**Endpoints:** `POST /api/audio/upload`, `POST /api/audio/transcribe`, `GET /api/audio/status/{id}`
**Why orphaned:** Replaced by `routers/audio/__init__.py` + `routers/audio/audio_core.py` + `routers/audio/websocket_audio.py`.
**Recovery:** Not needed. The package version is more complete.

---

## Appendix B: Relationship to Existing Plans

| Existing Plan | Relationship |
|--------------|-------------|
| `2026-03-04-offline-diarization-vibevoice-design.md` | WS1 implements the remaining backend work from this design |
| `2026-03-04-offline-diarization-vibevoice-plan.md` | WS1 covers remaining tasks from this plan |
| `2026-03-02-tech-debt-remediation-plan.md` | WS4 executes this plan verbatim |
| `2026-02-25-sveltekit-dashboard-design.md` | WS3 chat UI follows this dashboard's patterns |

---

## Appendix C: New Dependencies

| Package | Service | Purpose |
|---------|---------|---------|
| `reportlab` | orchestration | PDF generation for transcript export |
| `anthropic` | orchestration | Anthropic API adapter for chat |
| `openai` | orchestration | OpenAI API adapter for chat (also used for OpenAI-compatible endpoints) |
| `cryptography` | orchestration | Fernet encryption for API keys at rest (may already be transitive dep) |
