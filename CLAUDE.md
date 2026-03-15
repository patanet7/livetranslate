# CLAUDE.md

## Project Overview

LiveTranslate is a real-time speech-to-text transcription and translation system. Microservices architecture with WebSocket infrastructure for real-time communication.

See `ARCHITECTURE.md` for full system topology.

## Hard Rules

- **NEVER USE MOCK DATA** — All tests must be behavioral/integration with real system behavior
- **ALWAYS USE UV** for dependency management — not pip, PDM, or Poetry
- **Pydantic v2 BaseModel** for all shared contracts — use `model_dump()`, not `asdict()`
- **structlog only** — never `import logging` / `logging.getLogger()` (exception: silencing third-party loggers)

## Service Architecture

### 3 Core Services + External LLM

1. **Transcription Service** (`modules/transcription-service/`) — GPU, faster-whisper, Silero VAD, pluggable backends, VRAM budgeting
2. **Orchestration Service** (`modules/orchestration-service/`) — CPU, WebSocket hub, FLAC recording, Ollama translation, Google Meet bot
3. **Dashboard Service** (`modules/dashboard-service/`) — SvelteKit + Svelte 5 runes, real-time WebSocket streaming
4. **External: Ollama** (`:11434`) — Translation inference via OpenAI-compatible API (qwen3.5:7b)

**Legacy (not active):** `modules/frontend-service/` (React), `modules/translation-service/`, `modules/whisper-service/` — being replaced by services above.

**Tools:** `tools/translation_benchmark/` — Standalone CLI for benchmarking translation models (BLEU/COMET metrics, multi-model comparison, concurrency throughput). Run: `uv run python -m tools.translation_benchmark --model qwen3.5:7b --lang-pair zh-en`

### Dashboard Loopback Page (`/loopback`)

Real-time audio capture → transcription → translation page in the SvelteKit dashboard.

| Module | Path | Purpose |
|--------|------|---------|
| AudioWorklet | `static/audio-worklet-processor.js` | Ring buffer capture, RMS silence detection, zero-GC |
| AudioCapture | `src/lib/audio/capture.ts` | mic/system/both via getUserMedia, DynamicsCompressor mixing |
| LoopbackWebSocket | `src/lib/audio/websocket.ts` | Binary audio frames + JSON control, auto-reconnect with onReconnect |
| Loopback Store | `src/lib/stores/loopback.svelte.ts` | Svelte 5 `$state` factory, captions, translations, meeting state |
| SplitView | `src/lib/components/loopback/SplitView.svelte` | Side-by-side original/translation panels |
| SubtitleView | `src/lib/components/loopback/SubtitleView.svelte` | Bottom-anchored overlay with pop-out |
| TranscriptView | `src/lib/components/loopback/TranscriptView.svelte` | Timestamped scrolling transcript |
| Toolbar | `src/lib/components/loopback/Toolbar.svelte` | Device/language/model/mode selectors, meeting controls |
| Page | `src/routes/(app)/loopback/+page.svelte` | Wires capture → WS → store → display |

**Key:** Page uses `ssr = false` (browser-only APIs). `.env` must have `PUBLIC_WS_URL` and `PUBLIC_APP_NAME`.

### Key Technical Components

#### Google Meet Bot Management System ✅
- **Location**: `modules/orchestration-service/src/bot/`
- **GoogleMeetBotManager**: Central bot lifecycle management (`src/bot/bot_manager.py`)
- **Google Meet Browser Automation**: Headless Chrome integration (`src/bot/google_meet_automation.py`)
- **Browser Audio Capture**: Specialized Google Meet audio extraction (`src/bot/browser_audio_capture.py`)
- **Virtual Webcam System**: Real-time translation overlay generation (`src/bot/virtual_webcam.py`)
- **Time Correlation Engine**: Advanced timeline matching (`src/bot/time_correlation.py`)
- **Bot Integration Pipeline**: Complete orchestration flow (`src/bot/bot_integration.py`)
- **Database Integration**: PostgreSQL persistence (`src/database/bot_session_manager.py`)
- **Schema**: `scripts/bot-sessions-schema.sql` - Comprehensive PostgreSQL schema

#### Configuration Synchronization System ✅
- **ConfigurationSyncManager**: `modules/orchestration-service/src/audio/config_sync.py`
- **Dashboard Settings**: `modules/dashboard-service/src/pages/Settings/`
- **API Endpoints**: `modules/orchestration-service/src/routers/settings.py`
- **Transcription Integration**: `modules/transcription-service/src/api_server.py` (orchestration mode)

## Service Ports

- **Dashboard**: 5173 (development), 3000 (production)
- **Orchestration**: 3000
- **Transcription**: 5001
- **Ollama**: 11434
- **Monitoring**: 3001
- **Prometheus**: 9090

## Development Commands

### Install Dependencies

```bash
# CRITICAL: Must use --all-packages to install workspace packages (livetranslate-common, etc.)
uv sync --all-packages --group dev
```

> **Python Version**: >=3.12,<3.14. Python 3.14+ not supported (grpcio, onnxruntime).
> **Worktrees**: After creating a git worktree, run `uv sync --all-packages --group dev` before running tests.

### Run Services

```bash
uv run python modules/transcription-service/src/main.py   # Transcription (GPU)
uv run python modules/orchestration-service/src/main_fastapi.py  # Orchestration

cd modules/dashboard-service && npm install && npm run dev  # Dashboard (SvelteKit)
```

### Run Tests

```bash
uv run pytest modules/shared/tests/ -v               # Shared contracts
uv run pytest modules/orchestration-service/tests/ -v  # Orchestration
uv run pytest modules/transcription-service/tests/ -v  # Transcription

# Just commands
just test-orchestration
just test-transcription
just coverage-backend
```

### Test Markers

- `@pytest.mark.behavioral` — No mocks
- `@pytest.mark.integration` — Integration tests
- `@pytest.mark.e2e` — End-to-end
- `@pytest.mark.slow` — Skip with `-m "not slow"`
- `@pytest.mark.gpu` — Requires GPU

## Shared Library (`livetranslate-common`)

Package: `modules/shared/src/livetranslate_common/`

| Module | Purpose |
|--------|---------|
| `models/` | Pydantic v2 contracts: audio, transcription, translation, registry, ws_messages |
| `logging/` | structlog setup, performance logging, processors |
| `errors/` | Exception hierarchy, error handlers |
| `middleware/` | Request ID injection, logging middleware |
| `config/` | Settings management |
| `health/` | Health check endpoints |

### Logging Pattern

```python
from livetranslate_common.logging import setup_logging, get_logger

setup_logging(service_name="orchestration")  # entry point only
logger = get_logger()
logger.info("event_name", key="value")
```

## Benchmark Framework

### Running Benchmarks

```bash
# VAC transcription sweep (finds optimal prebuffer/stride/overlap per language)
just benchmark-vac lang=zh model=large-v3-turbo backend=vllm

# Full pipeline sweep (VAC × translation: stride × overlap × context × temp × tokens)
just benchmark-pipeline lang=zh model=qwen3.5:7b

# Quick pipeline (context sweep only, fixed VAC config)
just benchmark-pipeline-quick lang=zh

# All languages in stub mode (CI dry-run)
just benchmark-all-stub
```

### Optimal Configs (from benchmark sweep)

| Language | Prebuffer | Stride | Overlap | CER/WER | Caption freq | Use case |
|----------|-----------|--------|---------|---------|--------------|----------|
| zh (best accuracy) | 0.5s | 6.0s | 1.5s | 9.5% CER | every 6s | Transcripts, offline |
| zh (balanced) | 0.5s | 4.5s | 0.5s | 15.6% CER | every 4.5s | Live captions |
| zh (fastest) | 0.5s | 1.5s | 0.5s | 20.4% CER | every 1.5s | Real-time subtitles |
| en | 1.0s | 4.0s | 1.0s | 28.3% WER | every 4s | General |

### Benchmark Output

- **JSON**: Full per-segment trace with timing (`benchmarks/results/`)
- **TSV**: Human-readable ranked table
- **JSONL**: Append-only index for cross-run regression tracking
- **Playback fixtures**: Saved transcription output for translation-only tuning

### Translation Config

`TranslationConfig` (pydantic-settings, env_prefix=`LLM_`):
- `LLM_BASE_URL` — LLM API endpoint (default: `http://localhost:11434/v1`)
- `LLM_MODEL` — model name (default: `qwen3.5:7b`)
- `LLM_TEMPERATURE` — generation temperature (default: `0.3`)
- `LLM_CONTEXT_WINDOW_SIZE` — rolling context entries (default: `5`)
- `LLM_MAX_CONTEXT_TOKENS` — token budget for context (default: `500`)

## Gotchas

- **`uv sync --group dev`** alone does NOT install workspace packages — always use `--all-packages`
- **Browser audio processing** (echoCancellation, noiseSuppression) must be DISABLED for loopback audio capture
- **Audio resampling**: Browser sends 48kHz, services need 16kHz — librosa fallback handles conversion
- **Alembic revision IDs** must be ≤32 chars — see `modules/orchestration-service/CLAUDE.md`
- **`is_final` flag** in Whisper segments means "ends with punctuation", NOT "final transcription" — collect ALL segments

## Code Quality

- **Ruff** for linting/formatting
- **mypy** for type checking
- **Pre-commit hooks**: Run `pre-commit install` after cloning
