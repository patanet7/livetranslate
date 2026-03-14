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
5. **Translation** receives final segments, applies rolling context + glossary, calls Ollama
6. **Display** renders in split/subtitle/transcript modes via Svelte 5 runes

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
