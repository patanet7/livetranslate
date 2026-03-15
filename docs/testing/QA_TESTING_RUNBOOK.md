# QA Testing Runbook — Plan 5: End-to-End Wiring Verification

## Overview

This runbook documents all tests created by the QA E2E plan, how to run them, what infrastructure they need, and the deployment sequence for thomas-pc.

---

## Test Inventory

### Phase 1: Unit Tests (no external deps)

| Test File | Count | What it covers | Run from |
|-----------|-------|----------------|----------|
| `orchestration-service/tests/unit/test_transcription_client.py` | 25 | WebSocketTranscriptionClient: connect/close lifecycle, binary audio format, config JSON, callback dispatch, reconnect with backoff, error isolation | `modules/orchestration-service/` |

```bash
cd modules/orchestration-service
uv run pytest tests/unit/test_transcription_client.py -v --timeout=30
```

### Phase 2: Contract Tests (no external deps)

| Test File | Count | What it covers | Run from |
|-----------|-------|----------------|----------|
| `shared/tests/test_cross_service_contracts.py` | 13 | TranscriptionResult→SegmentMessage roundtrip, TranslationResponse→TranslationMessage mapping, float32 binary roundtrip, edge cases | project root |
| `shared/tests/test_ts_python_alignment.py` | 6 | Python↔TypeScript field alignment (regex-parsed), PROTOCOL_VERSION match | project root |
| `transcription-service/tests/test_registry.py` (Tasks 2.3+3.4) | 13 | Registry load, wildcard fallback, invalid stride rejected at load, chunk>30s rejected, VAD-bounded mode, hot-reload with invalid/empty/deleted YAML preserves old config | project root |

```bash
uv run pytest modules/shared/tests/ -v --timeout=30
uv run pytest modules/transcription-service/tests/test_registry.py -v --timeout=30
```

### Phase 3: Failure Mode Tests (testcontainers, no GPU)

| Test File | Count | What it covers | Run from |
|-----------|-------|----------------|----------|
| `orchestration-service/tests/integration/test_failure_modes.py` | 14 | Transcription unreachable (error frame, not crash), Ollama unreachable (graceful degradation), service status reporting, audio before/after failed start_session, FLAC disk-full resilience, malformed WS messages, promote without DB | `modules/orchestration-service/` |
| `orchestration-service/tests/test_recorder.py` | 8 | FLAC recorder lifecycle, manifest tracking, disk-full absorption, idempotent stop, monotonic sequences | `modules/orchestration-service/` |
| `transcription-service/tests/test_backpressure.py` | 5 | Queue drops frames at maxsize=16, remains functional after backpressure, producer uses put_nowait | project root |

```bash
cd modules/orchestration-service
uv run pytest tests/integration/test_failure_modes.py tests/test_recorder.py -v --timeout=30

# From project root:
uv run pytest modules/transcription-service/tests/test_backpressure.py -v --timeout=30
```

### Phase 4: E2E Tests (need services running)

| Test File | Needs | Count | What it covers |
|-----------|-------|-------|----------------|
| `orchestration-service/tests/e2e/test_meeting_flow.py` | Testcontainers PostgreSQL | 6+1 | Ephemeral→promote→end lifecycle, FLAC chunks written, 48k→16k downsampling, DB status transitions |
| `orchestration-service/tests/e2e/test_meeting_flow.py::TestTranslationOnFinalSegments` | Local Ollama | 2 | Translation on is_final=True, no translation on is_final=False |
| `transcription-service/tests/smoke/test_ws_stream_smoke.py` | **GPU (thomas-pc)** | 2 | Real speech → /api/stream → language_detected before segment, non-empty text |
| `orchestration-service/tests/e2e/test_full_pipeline.py` | **GPU + orchestration** | 1 | 48kHz browser audio → orchestration → transcription → validated SegmentMessage |

```bash
# Meeting flow (testcontainers — runs locally):
cd modules/orchestration-service
uv run pytest tests/e2e/test_meeting_flow.py -v --timeout=60

# Translation (needs local Ollama with qwen3.5:4b):
uv run pytest tests/e2e/test_meeting_flow.py::TestTranslationOnFinalSegments -v --timeout=60

# GPU smoke tests (needs thomas-pc):
uv run pytest modules/transcription-service/tests/smoke/test_ws_stream_smoke.py -v --timeout=60 -m "e2e and gpu"
cd modules/orchestration-service
uv run pytest tests/e2e/test_full_pipeline.py -v --timeout=60 -m "e2e and gpu"
```

### Phase 5: Visual Regression (Playwright — needs all services)

| Test File | Needs | What it covers |
|-----------|-------|----------------|
| `dashboard-service/tests/e2e/loopback-playback.spec.ts` | Dashboard + Orchestration + Transcription (GPU) + Ollama (optional) | Full JFK speech playback → captions → translation → screenshots |

```bash
cd modules/dashboard-service
npx playwright test tests/e2e/loopback-playback.spec.ts
```

**Screenshots saved to:** `tests/output/loopback-screenshots/`
- `01-page-loaded.png` — initial state
- `02-capture-started.png` — after Start Capture clicked
- `03-captions-visible.png` — transcription results displayed
- `04-with-translations.png` — translation overlay (if Ollama running)
- `05-capture-stopped.png` — after Stop Capture
- `06-default-display-mode.png` — display mode check
- `07-connection-error.png` — error state on unreachable WS

---

## Audio Test Fixtures

| File | Location | Format | Duration | Content |
|------|----------|--------|----------|---------|
| `jfk.wav` | `transcription-service/tests/fixtures/audio/` | 16kHz mono float32 | 10.4s | JFK inaugural address (real speech) |
| `jfk_48k.wav` | `dashboard-service/tests/fixtures/` | 48kHz mono float32 | 10.4s | Same, upsampled for browser playback |
| `hello_world.wav` | `transcription-service/tests/fixtures/audio/` | 16kHz mono | 3.0s | Synthetic formant audio |
| `long_speech.wav` | `transcription-service/tests/fixtures/audio/` | 16kHz mono | 5.0s | Synthetic formant audio |
| `silence.wav` | `transcription-service/tests/fixtures/audio/` | 16kHz mono | 2.0s | Pure silence |
| `noisy.wav` | `transcription-service/tests/fixtures/audio/` | 16kHz mono | 3.0s | Synth speech + 10dB SNR noise |

**Missing (never committed):** `OSR_cn_000_0072_8k.wav` through `0075` (Chinese Mandarin from Open Speech Repository). Only `.txt` transcripts exist.

---

## Quick Run: Everything Without GPU

Runs all 182 tests locally in ~15 seconds:

```bash
# From project root:
uv run pytest modules/shared/tests/ -v --timeout=30

uv run pytest modules/transcription-service/tests/test_registry.py \
              modules/transcription-service/tests/test_backpressure.py -v --timeout=30

# From orchestration-service directory:
cd modules/orchestration-service
uv run pytest tests/unit/test_transcription_client.py \
              tests/integration/test_failure_modes.py \
              tests/test_recorder.py \
              tests/e2e/test_meeting_flow.py -v --timeout=60
```

---

## Deployment to thomas-pc

### Prerequisites on thomas-pc

- Python 3.12+ with `uv`
- NVIDIA GPU with CUDA (for faster-whisper)
- PostgreSQL (or Docker for testcontainers)
- Ollama with `qwen3.5:4b` model loaded
- Node.js 20+ with npm (for dashboard)
- Chromium (for Playwright: `npx playwright install chromium`)

### Step 1: Sync code

```bash
# On thomas-pc:
cd ~/livetranslate
git pull origin main
uv sync --all-packages --group dev
```

### Step 2: Start services

```bash
# Terminal 1 — Transcription Service (GPU):
uv run python modules/transcription-service/src/main.py

# Terminal 2 — Orchestration Service:
uv run python modules/orchestration-service/src/main_fastapi.py

# Terminal 3 — Dashboard:
cd modules/dashboard-service && npm install && npm run dev

# Terminal 4 — Ollama (if not already running):
ollama serve
ollama pull qwen3.5:4b
```

### Step 3: Run GPU tests

```bash
# Transcription smoke test (Task 4.1):
uv run pytest modules/transcription-service/tests/smoke/test_ws_stream_smoke.py -v --timeout=60

# Full pipeline e2e (Task 4.2):
cd modules/orchestration-service
uv run pytest tests/e2e/test_full_pipeline.py -v --timeout=60
```

### Step 4: Run Playwright visual test

```bash
cd modules/dashboard-service
npx playwright install chromium  # first time only
npx playwright test tests/e2e/loopback-playback.spec.ts --headed  # --headed to watch
```

### Step 5: Review screenshots

```bash
open modules/dashboard-service/tests/output/loopback-screenshots/
# Or view the Playwright HTML report:
npx playwright show-report tests/output/playwright-report
```

---

## Wiring Fixes Applied (Phase 1)

| Gap | Fix | Files |
|-----|-----|-------|
| SocketIOWhisperClient ↔ plain WebSocket mismatch | New `WebSocketTranscriptionClient` with binary frames + JSON control | `orchestration-service/src/clients/transcription_client.py` (created) |
| MeetingPipeline orphaned (not connected to any endpoint) | Wired into `websocket_audio_stream` handler with optional DB | `orchestration-service/src/routers/audio/websocket_audio.py` (rewritten) |
| Transcription sends 5 fields, dashboard expects 11 | Expanded `model_dump()` to include `stable_text`, `unstable_text`, `speaker_id`, `start_ms`, `end_ms` | `transcription-service/src/api.py` (modified) |
| BackendSwitchedMessage schema ≠ wire format | Changed transcription service to emit `backend`, `model`, `language` | `transcription-service/src/api.py` (modified) |
| Dashboard connects `/ws/loopback`, endpoint at `/api/audio/stream` | Added `/ws/loopback` alias in `main_fastapi.py` | `orchestration-service/src/main_fastapi.py` (modified) |
| No validation at WebSocket boundaries | All messages validated through `parse_ws_message()` + Pydantic | `websocket_audio.py` uses `isinstance()` checks |

## WebSocket Review Fixes Applied

| Finding | Severity | Fix |
|---------|----------|-----|
| `source_language` updated after send — stale on failure | High | Moved language tracking before `send_text` |
| Translation tasks in module-level set, not cancelled on disconnect | High | Session-local `session_tasks` set, cancelled in `finally` |
| Reconnect spawns dual receive loop | Medium | Receive loop owns reconnect inline (no new task) |
| Double `start_session` leaks previous client | Medium | Tear down old client/pipeline before creating new |
| Per-connection DB engine with NullPool | Medium | Uses shared `DatabaseManager` singleton |
| Ping handler unreachable | Low | Moved to `msg is None` branch (raw JSON check) |

## Production Fixes Applied

| Issue | Fix | File |
|-------|-----|------|
| FLAC recorder crashes on disk-full | `sf.write` wrapped in try/except, records `samples: 0` in manifest | `meeting/recorder.py` |
| LLM cold-start takes 10-15s | Fire-and-forget warm-up call on `start_session` | `websocket_audio.py` |
| Sample rate hardcoded to 48000 | Configurable via `DEFAULT_SAMPLE_RATE` env var | `websocket_audio.py` |

---

## Test Counts Summary

| Suite | Passed | Infrastructure |
|-------|--------|---------------|
| Shared contracts | 117 | None |
| Transcription registry + backpressure | 18 | None |
| Orchestration unit (transcription client) | 25 | None |
| Orchestration integration (failure modes) | 14 | Testcontainers |
| Orchestration recorder | 8 | Testcontainers |
| Meeting flow e2e | 7 | Testcontainers + Ollama |
| **Subtotal (runs locally)** | **189** | |
| Transcription smoke (GPU) | 2 | GPU |
| Full pipeline e2e (GPU) | 1 | GPU + orchestration |
| Playwright visual regression | 3 | All services |
| **Grand total** | **195** | |
