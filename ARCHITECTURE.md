# LiveTranslate Architecture

## System Overview

LiveTranslate is a real-time speech transcription and translation system for live meetings.
Audio flows from browser → orchestration → transcription → translation → display.

## Service Topology

```
┌─────────────────────────────────────────────────────────┐
│  MacBook (local)                                        │
│  ┌──────────────┐    ┌───────────────────────────────┐  │
│  │  Dashboard    │◄──►│  Orchestration Service       │  │
│  │  (SvelteKit)  │    │  (FastAPI)                   │  │
│  │  :5173        │    │  :3000                       │  │
│  └──────────────┘    │  - WebSocket hub              │  │
│                       │  - Meeting pipeline           │  │
│                       │  - Translation (via Ollama)   │  │
│                       │  - FLAC recording             │  │
│                       │  - Audio downsampling         │  │
│                       └─────────────┬─────────────────┘  │
└─────────────────────────────────────┼────────────────────┘
                                      │ Tailscale (16kHz mono)
┌─────────────────────────────────────┼────────────────────┐
│  thomas-pc (RTX 4090)               │                    │
│  ┌──────────────────────────────────▼─────────────────┐  │
│  │  Transcription Service (faster-whisper)            │  │
│  │  :5001                                             │  │
│  │  - Pluggable backends (BackendManager)             │  │
│  │  - VRAM budget (10GB of 24GB)                      │  │
│  │  - ModelRegistry (YAML)                            │  │
│  │  - Authoritative LID                               │  │
│  │  - Silero VAD                                      │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Ollama (LLM inference)                            │  │
│  │  :11434                                            │  │
│  │  - qwen3.5:7b (translation)                        │  │
│  │  - OpenAI-compatible API                           │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Audio Flow

1. **Browser** captures mic audio via AudioWorklet at native sample rate (48kHz+)
2. **Binary WebSocket** sends Float32Array frames to orchestration (no base64)
3. **Orchestration** forks: native quality → FLAC disk, 16kHz mono → transcription
4. **Transcription** runs VAD → LID → backend inference → segments back as text frames
5. **Translation** receives draft/final segments, applies rolling context + glossary, calls Ollama
6. **Display** renders in split/subtitle/transcript modes via Svelte 5 runes

## Draft/Final Protocol

Each transcription segment passes through two phases before context is committed.

### Lifecycle

1. **Audio accumulates** in the VAC (voice-activity chunk) buffer.
2. **At stride/2** — non-destructive snapshot → fast inference → `SegmentMessage(segment_id=N, is_draft=True)`.
3. **At full stride** — destructive consume → final inference → `SegmentMessage(segment_id=N, is_draft=False)` with Whisper's punctuation-based `is_final` flag.

### Translation Routing

| Phase | Behaviour |
|-------|-----------|
| Draft (`is_draft=True`) | Non-streaming translation, no context write, provisional context read only. Dropped if a previous draft translation is still in-flight (`_draft_lock`). Timeout: `LLM_DRAFT_TIMEOUT_S` (default 4 s). Max tokens: `LLM_DRAFT_MAX_TOKENS` (default 256). |
| Non-final final (`is_draft=False, is_final=False`) | Text accumulated into `SegmentStore._pending_sentence`. No translation issued yet. |
| Sentence-boundary final (`is_draft=False, is_final=True`) | `_pending_sentence` flushed + this segment's text → streaming translation with full context. Context window updated with `(source_text, translation)` after completion. |

### Frontend: Draft-in-Place Replacement

The frontend matches incoming translation messages by `segment_id`. A draft translation sets `translationState = 'draft'`; the corresponding final translation overwrites it (`translationState = 'complete'`). The caption text is replaced in-place — no flicker or reorder.

`TranslationState` lifecycle: `pending` → `draft` → `streaming` → `complete`.

### Segment Eviction

`SegmentStore.evict_old(keep_last=50)` is called after every `on_draft_received` and `on_final_received` in `websocket_audio.py`. Records referenced by `_pending_segment_ids` (sentence accumulation buffer) are protected from eviction even when over the limit.

## Per-Direction Context

### DirectionalContextStore

`DirectionalContextStore` (`modules/orchestration-service/src/translation/context_store.py`) maintains independent `RollingContextWindow` instances keyed by `(source_lang, target_lang)`.

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_MAX_CONTEXT_TOKENS` | 800 | Token budget for same-direction context entries |
| `LLM_CROSS_DIRECTION_MAX_TOKENS` | 200 | Token budget for cross-direction referent entries |
| `LLM_CONTEXT_WINDOW_SIZE` | 5 | Maximum rolling entries per direction |

### Interpreter Mode

In interpreter mode two directions run simultaneously (e.g., `zh→en` and `en→zh`). Each direction's window is independent — a direction flip does not pollute the other side's context.

Cross-direction context (`get_cross_direction`) returns the last 1–2 entries from the opposite direction. This lets the LLM resolve referents that were established in the other language without inflating the prompt budget.

## Shared Contracts

All services share Pydantic models from `livetranslate-common` (`modules/shared/`):
- `TranscriptionResult`, `Segment`, `ModelInfo` — transcription output
- `AudioChunk`, `MeetingAudioStream` — audio pipeline types
- `TranslationRequest/Response/Context` — translation with rolling context + glossary
- `BackendConfig` — model registry entries
- WebSocket message schemas — typed protocol with versioning

TypeScript equivalents live in `modules/dashboard-service/src/lib/types/`.

## Meeting Pipeline

Sessions start **ephemeral** (stream-through, no persistence). "Start Meeting" promotes
to **active** (recording + DB persistence). Crash safety via flush-on-write FLAC chunks,
row-by-row DB persistence, manifest tracking, and 120s heartbeat orphan detection.

## Glossary System

Domain-specific terms stored in PostgreSQL via `GlossaryService`:
- **Translation**: glossary terms injected into LLM prompt for consistent terminology
- **Transcription**: glossary terms fed as Whisper's `initial_prompt` to bias recognition

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Dashboard | SvelteKit (Svelte 5 runes), TypeScript |
| Orchestration | FastAPI, SQLAlchemy, Alembic, FLAC (soundfile) |
| Transcription | faster-whisper (CTranslate2), Silero VAD |
| Translation | httpx → Ollama OpenAI-compatible API |
| Shared | Pydantic v2, UV workspace monorepo |
| Database | PostgreSQL |
| IPC | Tailscale VPN |
