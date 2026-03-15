# Shared Library (`livetranslate-common`)

Cross-service Pydantic v2 contracts, structured logging, error handling, and middleware.

## Package Structure

```
modules/shared/src/livetranslate_common/
├── models/          # Pydantic v2 shared contracts
│   ├── audio.py         # AudioChunk, MeetingAudioStream Protocol
│   ├── transcription.py # Segment, TranscriptionResult, ModelInfo
│   ├── translation.py   # TranslationContext, TranslationRequest/Response
│   ├── registry.py      # BackendConfig (VRAM budget, VAD, chunking)
│   └── ws_messages.py   # 15 WebSocket message types, PROTOCOL_VERSION=1
├── logging/         # structlog setup, performance logging
├── errors/          # Exception hierarchy, error handlers
├── middleware/      # Request ID injection, logging middleware
├── config/          # Settings management
└── health/          # Health check endpoints
```

## Commands

```bash
uv run pytest modules/shared/tests/ -v
```

## Key Contracts

- **BackendConfig**: `model_validator` enforces `stride_s == chunk_duration_s - overlap_s`. VAD-bounded mode: `overlap_s=0, stride_s=chunk_duration_s`.
- **Segment**: `model_validator` enforces `end_ms >= start_ms`. Has `duration_ms` property.
- **TranslationRequest**: `Field(ge=0)` on `context_window_size` and `max_context_tokens`.
- **ws_messages**: TypeScript mirror at `modules/dashboard-service/src/lib/types/ws-messages.ts`.

## Gotchas

- All models are **Pydantic v2 BaseModel** — use `model_dump()`, never `asdict()`.
- `vad_min_silence_ms` default is 2000ms (Silero default) — too conservative for real-time; override to 300-600ms.
- `chunk_duration_s` has `le=30.0` upper bound (Whisper positional encoding limit).
