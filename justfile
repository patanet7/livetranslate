# ==============================================================================
# LiveTranslate Justfile
# ==============================================================================
#
# Usage:
#   just              Show this help
#   just dev           Start all services locally
#   just test          Run all tests (no GPU)
#   just test-stream   Run streaming e2e tests (needs transcription service)
#
# Install just: https://github.com/casey/just

set shell := ['bash', '-eu', '-o', 'pipefail', '-c']

project_root := justfile_directory()
orchestration_dir := project_root / "modules/orchestration-service"
transcription_dir := project_root / "modules/transcription-service"
dashboard_dir := project_root / "modules/dashboard-service"
meeting_bot_dir := project_root / "modules/meeting-bot-service"

default: help

# ==============================================================================
# Help
# ==============================================================================

# Show available recipes
help:
    @printf "LiveTranslate Just Commands\n"
    @printf "===========================\n\n"
    @just --list --unsorted

# ==============================================================================
# Setup
# ==============================================================================

# Install all dependencies (Python + Node + Swift CLI)
install:
    uv sync --all-packages --group dev
    cd {{dashboard_dir}} && npm install
    cd {{meeting_bot_dir}} && npm install
    @just build-screencapture

# Build ScreenCaptureKit CLI (macOS only)
build-screencapture:
    #!/usr/bin/env bash
    set -eu
    if [[ "$(uname)" != "Darwin" ]]; then
        echo "Skipping screencapture build (macOS only)"
        exit 0
    fi
    echo "Building livetranslate-capture..."
    cd {{project_root}}/tools/screencapture && swift build -c release
    mkdir -p {{project_root}}/bin
    cp .build/release/livetranslate-capture {{project_root}}/bin/
    echo "Installed to bin/livetranslate-capture"

# ==============================================================================
# Development — Start Services
# ==============================================================================

# Split inference: two vllm-mlx processes to avoid Metal GPU crash from concurrent requests
# Whisper instance handles /v1/audio/transcriptions only
# LLM instance handles /v1/chat/completions only
whisper_port := "8005"
llm_port := "8006"
llm_model := env("LLM_MODEL", "mlx-community/Qwen3-4B-4bit")

# Log level for Python services (INFO, DEBUG, WARNING)
log_level := env("LOG_LEVEL", "INFO")

# Log directory for dev services
log_dir := "/tmp/livetranslate/logs"
# Timestamp for log file names (preserves logs across restarts)
log_ts := `date +%Y%m%d_%H%M%S`

# Start all services locally (two vllm-mlx: Whisper on :8005, LLM on :8006)
dev:
    @mkdir -p {{log_dir}}
    @echo "Starting all services (split inference)..."
    @echo "  vllm-mlx STT:    http://localhost:{{whisper_port}} (Whisper)"
    @echo "  vllm-mlx LLM:    http://localhost:{{llm_port}} ({{llm_model}})"
    @echo "  Transcription:   http://localhost:5001 (vllm backend → :{{whisper_port}})"
    @echo "  Orchestration:   http://localhost:3000 (LLM → :{{llm_port}})"
    @echo "  Dashboard:       http://localhost:5173"
    @echo ""
    @echo "Logs: {{log_dir}}/ (session: {{log_ts}})"
    @echo "  tail -f {{log_dir}}/vllm-mlx-stt.log"
    @echo "  tail -f {{log_dir}}/vllm-mlx-llm.log"
    @echo "  tail -f {{log_dir}}/transcription.log"
    @echo "  tail -f {{log_dir}}/orchestration.log"
    @echo "  tail -f {{log_dir}}/dashboard.log"
    @echo ""
    @echo "Use Ctrl+C to stop all."
    @trap 'kill 0' EXIT; \
        ln -sf vllm-mlx-stt_{{log_ts}}.log {{log_dir}}/vllm-mlx-stt.log && \
        ln -sf vllm-mlx-llm_{{log_ts}}.log {{log_dir}}/vllm-mlx-llm.log && \
        ln -sf transcription_{{log_ts}}.log {{log_dir}}/transcription.log && \
        ln -sf orchestration_{{log_ts}}.log {{log_dir}}/orchestration.log && \
        ln -sf dashboard_{{log_ts}}.log {{log_dir}}/dashboard.log && \
        PYTHONUNBUFFERED=1 uv run vllm-mlx serve {{llm_model}} --port {{whisper_port}} 2>&1 | tee {{log_dir}}/vllm-mlx-stt_{{log_ts}}.log | awk '{print "[vllm-stt] " $0; fflush()}' & \
        PYTHONUNBUFFERED=1 uv run vllm-mlx serve {{llm_model}} --port {{llm_port}} 2>&1 | tee {{log_dir}}/vllm-mlx-llm_{{log_ts}}.log | awk '{print "[vllm-llm] " $0; fflush()}' & \
        sleep 5 && \
        VLLM_MLX_URL=http://localhost:{{whisper_port}} LOG_LEVEL={{log_level}} PYTHONUNBUFFERED=1 FORCE_COLOR=1 \
        uv run python -u {{transcription_dir}}/src/main_local.py --model large-v3-turbo --backend vllm 2>&1 | tee {{log_dir}}/transcription_{{log_ts}}.log | awk '{print "[transcription] " $0; fflush()}' & \
        sleep 3 && \
        TRANSCRIPTION_HOST=localhost TRANSCRIPTION_PORT=5001 \
        LLM_BASE_URL=http://localhost:{{llm_port}}/v1 LLM_MODEL={{llm_model}} \
        LLM_TIMEOUT_S=30 DEFAULT_TARGET_LANGUAGE=zh LOG_LEVEL={{log_level}} PYTHONUNBUFFERED=1 FORCE_COLOR=1 \
        uv run python -u {{orchestration_dir}}/src/main_fastapi.py 2>&1 | tee {{log_dir}}/orchestration_{{log_ts}}.log | awk '{print "[orchestration] " $0; fflush()}' & \
        sleep 2 && cd {{dashboard_dir}} && npm run dev 2>&1 | tee {{log_dir}}/dashboard_{{log_ts}}.log | awk '{print "[dashboard] " $0; fflush()}' & \
        wait

# Start all services with DEBUG logging (shows segment lifecycle, dedup, draft→final)
dev-debug:
    LOG_LEVEL=DEBUG just dev

# Start all services with Ollama for LLM (no vllm-mlx LLM instance needed)
dev-ollama:
    @mkdir -p {{log_dir}}
    @echo "Starting services (Ollama for LLM, no vllm-mlx LLM)..."
    @echo "Logs: {{log_dir}}/ (session: {{log_ts}})"
    @trap 'kill 0' EXIT; \
        ln -sf transcription_{{log_ts}}.log {{log_dir}}/transcription.log && \
        ln -sf orchestration_{{log_ts}}.log {{log_dir}}/orchestration.log && \
        ln -sf dashboard_{{log_ts}}.log {{log_dir}}/dashboard.log && \
        PYTHONUNBUFFERED=1 FORCE_COLOR=1 uv run python -u {{transcription_dir}}/src/main_local.py --model large-v3-turbo 2>&1 | tee {{log_dir}}/transcription_{{log_ts}}.log | awk '{print "[transcription] " $0; fflush()}' & \
        sleep 5 && \
        TRANSCRIPTION_HOST=localhost TRANSCRIPTION_PORT=5001 \
        LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3.5:4b \
        LLM_TIMEOUT_S=30 DEFAULT_TARGET_LANGUAGE=zh PYTHONUNBUFFERED=1 FORCE_COLOR=1 \
        uv run python -u {{orchestration_dir}}/src/main_fastapi.py 2>&1 | tee {{log_dir}}/orchestration_{{log_ts}}.log | awk '{print "[orchestration] " $0; fflush()}' & \
        sleep 2 && cd {{dashboard_dir}} && npm run dev 2>&1 | tee {{log_dir}}/dashboard_{{log_ts}}.log | awk '{print "[dashboard] " $0; fflush()}' & \
        wait

# Start vllm-mlx LLM server standalone (Apple Silicon)
dev-llm model=llm_model:
    uv run vllm-mlx serve {{model}} --port {{llm_port}}

# Start vllm-mlx Whisper server standalone (Apple Silicon)
dev-stt:
    uv run vllm-mlx serve {{llm_model}} --port {{whisper_port}}

# Start transcription service (Mac: auto-detect MLX/CPU)
dev-transcription model='medium':
    uv run python {{transcription_dir}}/src/main_local.py --model {{model}}

# Start transcription service (GPU — thomas-pc)
dev-transcription-gpu:
    uv run python {{transcription_dir}}/src/main.py

# Start orchestration service
dev-orchestration:
    TRANSCRIPTION_HOST=localhost TRANSCRIPTION_PORT=5001 \
    LLM_BASE_URL=http://localhost:{{llm_port}}/v1 LLM_MODEL={{llm_model}} \
    LLM_TIMEOUT_S=15 DEFAULT_TARGET_LANGUAGE=es \
    uv run python {{orchestration_dir}}/src/main_fastapi.py

# Start dashboard (SvelteKit dev server)
dev-dashboard:
    cd {{dashboard_dir}} && npm run dev

# Start the canonical meeting bot runtime locally
dev-meeting-bot:
    cd {{meeting_bot_dir}} && npm run api

# Start the canonical meeting bot runtime with Docker
dev-meeting-bot-docker:
    cd {{meeting_bot_dir}} && docker compose up --build

# ==============================================================================
# Testing — Unit + Integration (no GPU needed)
# ==============================================================================

# Run all tests that don't need GPU or running services
test:
    @echo "=== Shared contracts ==="
    uv run pytest modules/shared/tests/ -v --timeout=30
    @echo ""
    @echo "=== Transcription registry + backpressure ==="
    uv run pytest {{transcription_dir}}/tests/test_registry.py {{transcription_dir}}/tests/test_backpressure.py -v --timeout=30
    @echo ""
    @echo "=== Orchestration unit + integration ==="
    cd {{orchestration_dir}} && uv run pytest tests/unit/ tests/integration/ tests/test_recorder.py -v --timeout=30 -m "not integration"

# Run shared contract + TS alignment tests
test-contracts:
    uv run pytest modules/shared/tests/ -v --timeout=30

# Run orchestration tests (skip integration tests that need running services)
test-orchestration:
    cd {{orchestration_dir}} && uv run pytest tests/unit/ tests/integration/ tests/test_recorder.py -v --timeout=30 -m "not integration"

# Run orchestration integration tests (needs `just dev` running)
test-orchestration-integration:
    cd {{orchestration_dir}} && uv run pytest tests/unit/test_llm_client.py -v --timeout=30 -m integration

# Run transcription registry + config tests
test-transcription:
    uv run pytest {{transcription_dir}}/tests/test_registry.py {{transcription_dir}}/tests/test_backpressure.py -v --timeout=30

# ==============================================================================
# Testing — E2E (needs running services)
# ==============================================================================

# Run streaming e2e tests (needs transcription service on :5001)
test-stream:
    uv run pytest {{transcription_dir}}/tests/integration/test_streaming_e2e.py -v -s --timeout=120

# Run meeting flow e2e tests (testcontainers PostgreSQL)
test-meeting:
    cd {{orchestration_dir}} && uv run pytest tests/e2e/test_meeting_flow.py -v --timeout=60

# Run translation smoke test (needs local Ollama)
test-translation:
    cd {{orchestration_dir}} && uv run pytest tests/e2e/test_meeting_flow.py::TestTranslationOnFinalSegments -v --timeout=60

# Run Playwright visual regression (needs all services running)
test-playwright:
    cd {{dashboard_dir}} && npx playwright test tests/e2e/loopback-playback.spec.ts --headed

# Run ALL tests (unit + integration + e2e + streaming)
test-all: test test-stream test-meeting

# ==============================================================================
# Testing — Coverage
# ==============================================================================

# Generate coverage for orchestration service
coverage-orchestration:
    cd {{orchestration_dir}} && uv run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v --timeout=60

# Generate coverage for transcription service
coverage-transcription:
    uv run pytest {{transcription_dir}}/tests/ --cov={{transcription_dir}}/src --cov-report=html --cov-report=term-missing -v --timeout=60

# Generate coverage for all backend services
coverage-backend: coverage-orchestration coverage-transcription

# ==============================================================================
# Database
# ==============================================================================

# Start PostgreSQL + Redis containers
db-up:
    docker compose -f docker-compose.database.yml up -d postgres redis
    @echo "PostgreSQL: localhost:5432 | Redis: localhost:6379"

# Stop database containers
db-down:
    docker compose -f docker-compose.database.yml down

# Run Alembic migrations
db-migrate:
    cd {{orchestration_dir}} && DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run alembic upgrade head

# Open PostgreSQL shell
db-shell:
    docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate

# ==============================================================================
# Code Quality
# ==============================================================================

# Lint all Python code
lint:
    uv run ruff check modules/

# Format all Python code
fmt:
    uv run ruff format modules/

# Type check
typecheck:
    uv run mypy modules/orchestration-service/src/ modules/shared/src/

# Run pre-commit hooks
pre-commit:
    pre-commit run --all-files

# Full CI check (lint + type + test)
ci: lint typecheck test

# ==============================================================================
# Docker
# ==============================================================================

# Build the optional compatibility compose stack
docker-build:
    docker compose -f docker/optional/compose.local.yml build

# Start the optional compatibility compose stack (not the default local-dev path)
compose-up profiles="core,inference":
    COMPOSE_PROFILES={{profiles}} docker compose -f docker/optional/compose.local.yml up --build

# Stop the optional compatibility compose stack
compose-down:
    docker compose -f docker/optional/compose.local.yml down

# ==============================================================================
# Cleanup
# ==============================================================================

# Clean Python caches + build artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name ".DS_Store" -delete 2>/dev/null || true
    @echo "Clean!"

# ==============================================================================
# Utilities
# ==============================================================================

# Show environment info
info:
    @echo "LiveTranslate Environment"
    @echo "========================="
    @uv --version
    @python3 --version
    @node --version 2>/dev/null || echo "Node: not found"
    @echo "Platform: $(uname -sm)"
    @echo ""
    @echo "Services:"
    @curl -s http://localhost:5001/health 2>/dev/null && echo "  Transcription: UP" || echo "  Transcription: DOWN"
    @curl -s http://localhost:3000/health 2>/dev/null && echo "  Orchestration: UP" || echo "  Orchestration: DOWN"
    @curl -s http://localhost:5173 2>/dev/null > /dev/null && echo "  Dashboard: UP" || echo "  Dashboard: DOWN"
    @curl -s http://localhost:11434/api/tags 2>/dev/null > /dev/null && echo "  Ollama: UP" || echo "  Ollama: DOWN"

# ==============================================================================
# Benchmarks
# ==============================================================================

# VAC parameter sweep — finds optimal prebuffer/stride/overlap for a language
# Usage: just benchmark-vac [lang=zh] [stub=true] [model=large-v3-turbo] [backend=vllm]
#   lang:     source language code (zh/en/ja/es)
#   stub:     set to "true" to use stub transcriber (no GPU/service required)
#   model:    Whisper model name (default: large-v3-turbo)
#   backend:  whisper backend — vllm, faster-whisper, mlx (default: vllm)
benchmark-vac lang="zh" stub="false" model="large-v3-turbo" backend="vllm":
    #!/usr/bin/env bash
    set -euo pipefail
    STUB_FLAG=""
    if [ "{{stub}}" = "true" ]; then STUB_FLAG="--stub"; fi
    cd {{transcription_dir}} && uv run python -m benchmarks.vac_sweep \
        --audio tests/fixtures/audio/meeting_{{lang}}.wav \
        --ref   tests/fixtures/audio/meeting_{{lang}}.txt \
        --lang  {{lang}} \
        --model {{model}} \
        --backend {{backend}} \
        --output-dir benchmarks/results/vac_sweep \
        $STUB_FLAG

# Full pipeline benchmark — VAC × translation sweep (real-time configs only)
# Usage: just benchmark-pipeline [lang=zh] [target=en] [stub=false] [model=qwen3.5:7b]
#   Sweeps: strides × overlaps × context_sizes × temperatures × max_context_tokens
#   Strides capped at 6.0s — anything higher is not real-time (see benchmark-offline)
benchmark-pipeline lang="zh" target="en" stub="false" model="qwen3.5:7b":
    #!/usr/bin/env bash
    set -euo pipefail
    STUB_FLAG=""
    if [ "{{stub}}" = "true" ]; then STUB_FLAG="--stub"; fi
    cd {{transcription_dir}} && uv run python -m benchmarks.pipeline_benchmark \
        --lang        {{lang}} \
        --target-lang {{target}} \
        --model       {{model}} \
        --strides 3.5 4.5 6.0 \
        --overlaps 0.5 1.0 1.5 \
        --context-sizes 0 3 5 \
        --temperatures 0.1 0.3 0.7 \
        --max-context-tokens 200 500 \
        --prebuffer 0.5 \
        $STUB_FLAG


# Quick pipeline benchmark — context sweep only (no temperature/token sweep)
benchmark-pipeline-quick lang="zh" target="en" stub="false" model="qwen3.5:7b":
    #!/usr/bin/env bash
    set -euo pipefail
    STUB_FLAG=""
    if [ "{{stub}}" = "true" ]; then STUB_FLAG="--stub"; fi
    cd {{transcription_dir}} && uv run python -m benchmarks.pipeline_benchmark \
        --lang        {{lang}} \
        --target-lang {{target}} \
        --model       {{model}} \
        --context-sizes 0 3 5 \
        --prebuffer 0.5 \
        --stride 6.0 \
        --overlap 1.5 \
        $STUB_FLAG

# Run benchmark pytest suite (no GPU — stub backend only)
benchmark-tests:
    uv run pytest {{transcription_dir}}/tests/benchmarks/test_vac_sweep.py \
                  {{transcription_dir}}/tests/benchmarks/test_pipeline_benchmark.py \
                  -v -s --timeout=120 -m benchmark

# ==============================================================================
# E2E Testing — Fixtures & Playback
# ==============================================================================

# Generate 48kHz audio fixtures for Playwright (one-time, from 16kHz sources)
create-e2e-fixtures:
    uv run python tools/create_e2e_fixtures.py

# Run backend translation playback tests (needs LLM on :8006, run standalone — not with `just dev`)
# On Apple Silicon: `just dev-llm` in one terminal, `just test-e2e-playback` in another
test-e2e-playback:
    cd {{orchestration_dir}} && \
    LLM_BASE_URL=http://localhost:{{llm_port}}/v1 LLM_MODEL={{llm_model}} \
    uv run pytest tests/e2e/test_translation_playback.py tests/e2e/test_mixed_language_playback.py -v -m e2e --timeout=120

# Create fixtures for language detection replay tests (from FLAC recordings)
create-lang-detect-fixtures:
    uv run python tools/create_flac_replay_fixtures.py

# Run language detection Playwright replay tests (needs `just dev` running)
test-lang-detect:
    cd {{dashboard_dir}} && npx playwright test tests/e2e/language-detection-replay.spec.ts --headed

# ==============================================================================
# Bot — Build, Launch, Join Meetings
# ==============================================================================

bot_container_dir := project_root / "modules/bot-container"

# Build the bot Docker image
bot-build:
    docker build -t livetranslate-bot:latest {{bot_container_dir}}
    @echo "✓ Built livetranslate-bot:latest"

# Start infrastructure for bot (Postgres + Redis + migrations)
bot-infra:
    #!/usr/bin/env bash
    set -euo pipefail
    just db-up
    echo "Waiting for Postgres..."
    for i in $(seq 1 30); do
        docker exec livetranslate-postgres pg_isready -U postgres 2>/dev/null && break
        sleep 1
    done
    sleep 2
    just db-migrate
    echo "✓ Infrastructure ready (Postgres + Redis + migrations)"

# Start a bot to join a Google Meet (requires: just bot-infra, just dev or orchestration running)
# Usage: just bot-join "https://meet.google.com/xxx-yyyy-zzz"
bot-join meeting_url:
    #!/usr/bin/env bash
    set -euo pipefail
    # Check bot image exists
    if ! docker image inspect livetranslate-bot:latest >/dev/null 2>&1; then
        echo "Bot image not found. Building..."
        just bot-build
    fi
    # Check orchestration is running
    if ! curl -sf http://localhost:3000/ >/dev/null 2>&1; then
        echo "ERROR: Orchestration not running on :3000. Start it first:"
        echo "  just dev  OR  uv run python modules/orchestration-service/src/main_fastapi.py"
        exit 1
    fi
    # Check Redis is running
    if ! docker exec livetranslate-redis redis-cli ping >/dev/null 2>&1; then
        echo "ERROR: Redis not running. Start with: just db-up"
        exit 1
    fi
    echo "Starting bot for: {{meeting_url}}"
    RESPONSE=$(curl -s -X POST http://localhost:3000/api/bot/start \
        -H "Content-Type: application/json" \
        -d "{\"meeting_url\": \"{{meeting_url}}\", \"user_token\": \"dev-token\", \"user_id\": \"dev-user\", \"enable_virtual_webcam\": false}")
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    CONNECTION_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('connection_id',''))" 2>/dev/null)
    if [ -n "$CONNECTION_ID" ]; then
        echo ""
        echo "✓ Bot started: $CONNECTION_ID"
        echo "  Status:  curl http://localhost:3000/api/bot/status/$CONNECTION_ID"
        echo "  Stop:    just bot-stop $CONNECTION_ID"
        echo "  Overlay: http://localhost:5173/captions?session=$CONNECTION_ID"
    fi

# Stop a running bot
bot-stop connection_id:
    curl -s -X POST "http://localhost:3000/api/bot/stop/{{connection_id}}" \
        -H "Content-Type: application/json" \
        -d '{"timeout": 30}' | python3 -m json.tool

# List all bots
bot-list:
    curl -s "http://localhost:3000/api/bot/list" | python3 -m json.tool

# Run demo mode (no real meeting needed — replays captured Fireflies data)
# Requires: just bot-full running in another terminal.
# Usage: just bot-demo
bot-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! lsof -i :3000 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Orchestration not running. Start it first:"
        echo "  just bot-full    (in another terminal)"
        exit 1
    fi
    echo "Starting demo replay..."
    uv run python tools/bot_demo.py

# Adversarial QA for caption pipeline (needs just bot-full running)
caption-qa test="all":
    uv run python tools/caption_qa.py --test {{test}}

# Full bot pipeline: start infra + orchestration + dashboard + demo
bot-full:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT
    echo "Starting full pipeline..."
    just bot-infra
    echo ""
    echo "Starting orchestration..."
    TRANSCRIPTION_HOST=localhost TRANSCRIPTION_PORT=5001 \
    LLM_BASE_URL=http://localhost:{{llm_port}}/v1 LLM_MODEL={{llm_model}} \
    PYTHONUNBUFFERED=1 uv run python -u modules/orchestration-service/src/main_fastapi.py &
    sleep 3
    echo ""
    echo "Starting dashboard..."
    cd {{dashboard_dir}} && PUBLIC_WS_URL=ws://localhost:3000 PUBLIC_APP_NAME=LiveTranslate npm run dev -- --port 5173 &
    sleep 3
    echo ""
    echo "========================================="
    echo "  LiveTranslate Bot Pipeline Ready"
    echo "========================================="
    echo "  Dashboard:     http://localhost:5173"
    echo "  Orchestration: http://localhost:3000"
    echo ""
    echo "  Join meeting:  just bot-join 'https://meet.google.com/xxx-yyyy-zzz'"
    echo "  Run demo:      just bot-demo"
    echo "========================================="
    echo ""
    echo "Press Ctrl+C to stop all."
    wait

# ==============================================================================
# Benchmarks
# ==============================================================================

# Full benchmark: VAC sweep + pipeline benchmark across all meeting languages (stub)
# Use this for CI dry-run validation. Set stub=false to use real backend.
benchmark lang="zh" stub="true":
    @echo "=== VAC Sweep ({{lang}}, stub={{stub}}) ==="
    just benchmark-vac lang={{lang}} stub={{stub}}
    @echo ""
    @echo "=== Pipeline Benchmark ({{lang}}→en, stub={{stub}}) ==="
    just benchmark-pipeline lang={{lang}} target=en stub={{stub}}
    @echo ""
    @echo "Benchmark complete. Results in modules/transcription-service/benchmarks/results/"

# Benchmark all languages sequentially (stub mode for CI)
benchmark-all-stub:
    @echo "=== Benchmarking all languages (stub mode) ==="
    just benchmark lang=zh stub=true
    just benchmark lang=en stub=true
    just benchmark lang=ja stub=true
    just benchmark lang=es stub=true
