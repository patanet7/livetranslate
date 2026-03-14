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

### Ports

| Service | Port |
|---------|------|
| Dashboard | 5173 (dev) / 3000 (prod) |
| Orchestration | 3000 |
| Transcription | 5001 |
| Ollama | 11434 |

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
