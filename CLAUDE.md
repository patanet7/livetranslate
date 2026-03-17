# CLAUDE.md

## Project Overview

LiveTranslate is a real-time speech-to-text transcription and translation system. Microservices architecture with WebSocket infrastructure for real-time communication.

See `ARCHITECTURE.md` for full system topology.

## Hard Rules

- **NEVER USE MOCK DATA** — All tests must be behavioral/integration with real system behavior
- **ALWAYS USE UV** for dependency management — not pip, PDM, or Poetry
- **Pydantic v2 BaseModel** for all shared contracts — use `model_dump()`, not `asdict()`
- **structlog only** — never `import logging` / `logging.getLogger()` (exception: silencing third-party loggers)
- **Test fixtures loading ML models MUST use `yield` + teardown** with `gc.collect()` — never bare `return`. Apple Silicon unified memory means leaked models OOM the entire system

## Service Architecture

### 3 Core Services + External LLM

1. **Transcription Service** (`modules/transcription-service/`) — Pluggable backends (vLLM-MLX on Apple Silicon, faster-whisper on GPU), Silero VAD, VRAM budgeting
2. **Orchestration Service** (`modules/orchestration-service/`) — CPU, WebSocket hub, FLAC recording, LLM translation, Google Meet bot
3. **Dashboard Service** (`modules/dashboard-service/`) — SvelteKit + Svelte 5 runes, real-time WebSocket streaming
4. **External LLM** — Translation inference via OpenAI-compatible API. Local: vLLM-MLX (`:8006`, `mlx-community/Qwen3-4B-4bit`). Remote: Ollama (`:11434`, `qwen3.5:7b`)

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
- **Canonical bot runtime**: `modules/meeting-bot-service/`
- **Canonical orchestration control path**: `modules/orchestration-service/src/bot/docker_bot_manager.py`
- **Canonical orchestration routes**: `modules/orchestration-service/src/routers/bot/bot_docker_management.py`
- **Legacy bot manager**: `modules/orchestration-service/src/bot/bot_manager.py` is not the default place for new work
- **Virtual Webcam System**: `modules/orchestration-service/src/bot/virtual_webcam.py`
- **Time Correlation Engine**: `modules/orchestration-service/src/bot/time_correlation.py`
- **Database Integration**: `modules/orchestration-service/src/database/bot_session_manager.py`
- **Schema**: `scripts/bot-sessions-schema.sql` - Comprehensive PostgreSQL schema

#### Translation Pipeline ✅
- **Location**: `modules/orchestration-service/src/translation/`
- **DirectionalContextStore**: Per-`(source_lang, target_lang)` rolling context windows (`src/translation/context_store.py`)
- **SegmentStore**: Draft/final lifecycle tracker, sentence accumulation, eviction (`src/translation/segment_store.py`)
- **SegmentRecord / SegmentPhase**: Single-segment state dataclass + phase enum (`src/translation/segment_record.py`)
- **TranslationConfig**: pydantic-settings with `LLM_` prefix, draft/final token budgets (`src/translation/config.py`)
- **Eviction**: `SegmentStore.evict_old(keep_last=50)` called after every draft/final received in `websocket_audio.py`

#### Language Detection & Session Restart ✅
- **WhisperLanguageDetector**: `modules/transcription-service/src/language_detection.py` — wraps `SustainedLanguageDetector` with hysteresis (margin=0.2, dwell=4 frames/10s)
- **Session Restart**: On sustained switch, flushes VAC buffer, resets hallucination filter and dedup context (`api.py` `if switched:` block)
- **Mode Guard**: Detection + restart only fires when `lock_language=False` (interpreter mode). Split mode forces whisper hint, skips detection entirely
- **SessionConfig**: `modules/orchestration-service/src/routers/audio/websocket_audio.py` — manages interpreter↔split mode transitions, language save/restore

#### Configuration Synchronization System ✅
- **ConfigurationSyncManager**: `modules/orchestration-service/src/audio/config_sync.py`
- **Dashboard Settings**: `modules/dashboard-service/src/pages/Settings/`
- **API Endpoints**: `modules/orchestration-service/src/routers/settings.py`
- **Transcription Integration**: `modules/transcription-service/src/api.py` (orchestration mode)

## Service Ports

- **Dashboard**: 5173 (development), 3000 (production)
- **Orchestration**: 3000
- **Transcription**: 5001
- **vLLM-MLX STT**: 8005 (Whisper transcription inference, Apple Silicon)
- **vLLM-MLX LLM**: 8006 (Qwen3-4B-4bit translation inference, Apple Silicon)
- **Ollama**: 11434 (alternative LLM backend, remote GPU)
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
- `@pytest.mark.e2e` — End-to-end (require running services)
- `@pytest.mark.slow` — Skip with `-m "not slow"`
- `@pytest.mark.stress` — 60+ minute stress tests
- `@pytest.mark.accuracy` — Accuracy regression baselines (loads large models)
- `@pytest.mark.benchmark` — Performance benchmarks
- `@pytest.mark.gpu` — Requires GPU
- `@pytest.mark.npu` — Requires Intel NPU

### E2E / Playwright Testing

All E2E tests assume `just dev` is already running (split vLLM-MLX inference).

```bash
just create-e2e-fixtures   # One-time: upsample 16kHz → 48kHz WAVs
just test-playwright        # Playwright: browser → orchestration → vLLM-MLX → DOM
just test-e2e-playback      # Backend: translation-only replay via vLLM-MLX LLM on :8006
```

**Fixture recording**: Set `LIVETRANSLATE_RECORD_FIXTURES=1` when running `just dev` to capture live sessions as WAV + JSON sidecar pairs in `/tmp/livetranslate/fixture-recordings/`.

#### Language Detection Replay Tests

```bash
just create-lang-detect-fixtures  # Convert FLAC recordings → 48kHz WAV fixtures
just test-lang-detect             # Playwright: replay real meeting audio, verify stable detection
```

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
| zh (best) | 0.5s | 6.0s | 1.5s | **9.5% CER** | every 6s | Best accuracy |
| zh (balanced) | 0.5s | 4.5s | 0.5s | 15.6% CER | every 4.5s | Live captions |
| zh (fastest) | 0.5s | 1.5s | 0.5s | 20.4% CER | every 1.5s | Real-time subtitles |
| en (best) | 0.5s | 6.0s | 0.5s | **19.1% WER** | every 6s | Best accuracy |
| en (balanced) | 0.5s | 4.5s | 1.0s | 21.7% WER | every 4.5s | Live captions |
| en (fastest) | 0.5s | 1.5s | 0.5s | 32.2% WER | every 1.5s | Real-time subtitles |

**Key finding:** English needs overlap=0.5s (word boundaries dedup well). Chinese needs overlap=1.5s (CJK characters need more context).

### Benchmark Output

- **JSON**: Full per-segment trace with timing (`benchmarks/results/`)
- **TSV**: Human-readable ranked table
- **JSONL**: Append-only index for cross-run regression tracking
- **Playback fixtures**: Saved transcription output for translation-only tuning

### Translation Config

`TranslationConfig` (pydantic-settings, env_prefix=`LLM_`):
- `LLM_BASE_URL` — LLM API endpoint (default: `http://localhost:8006/v1` for vLLM-MLX)
- `LLM_MODEL` — model name (default: `mlx-community/Qwen3-4B-4bit`)
- `LLM_TEMPERATURE` — generation temperature (default: `0.7`)
- `LLM_CONTEXT_WINDOW_SIZE` — rolling context entries per direction (default: `5`)
- `LLM_MAX_CONTEXT_TOKENS` — token budget for same-direction context (default: `800`)
- `LLM_CROSS_DIRECTION_MAX_TOKENS` — token budget for cross-direction referent context (default: `200`)
- `LLM_MAX_TOKENS` — max output tokens for final translation (default: `512`)
- `LLM_DRAFT_MAX_TOKENS` — max output tokens for draft translation (default: `256`)
- `LLM_DRAFT_TIMEOUT_S` — wall-clock timeout for draft translations (default: `4`)

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
