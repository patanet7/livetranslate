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

# Install all dependencies (Python + Node)
install:
    uv sync --all-packages --group dev
    cd {{dashboard_dir}} && npm install

# ==============================================================================
# Development — Start Services
# ==============================================================================

# LLM model for translation (vllm-mlx on Mac, Ollama on thomas-pc)
llm_model := env("LLM_MODEL", "mlx-community/Qwen3.5-4B-4bit")
llm_port := "8000"

# Start all services locally (vllm-mlx handles both Whisper + LLM)
dev:
    @echo "Starting all services..."
    @echo "  vllm-mlx:        http://localhost:{{llm_port}} ({{llm_model}} + Whisper)"
    @echo "  Transcription:   http://localhost:5001 (vllm backend → :{{llm_port}})"
    @echo "  Orchestration:   http://localhost:3000"
    @echo "  Dashboard:       http://localhost:5173"
    @echo ""
    @echo "Use Ctrl+C to stop all."
    @trap 'kill 0' EXIT; \
        uv run vllm-mlx serve {{llm_model}} --port {{llm_port}} 2>&1 | sed 's/^/[vllm-mlx] /' & \
        sleep 5 && \
        VLLM_MLX_URL=http://localhost:{{llm_port}} \
        uv run python {{transcription_dir}}/src/main_local.py --model large-v3-turbo --backend vllm 2>&1 | sed 's/^/[transcription] /' & \
        sleep 3 && \
        TRANSCRIPTION_HOST=localhost TRANSCRIPTION_PORT=5001 \
        LLM_BASE_URL=http://localhost:{{llm_port}}/v1 LLM_MODEL={{llm_model}} \
        LLM_TIMEOUT_S=15 DEFAULT_TARGET_LANGUAGE=es \
        uv run python {{orchestration_dir}}/src/main_fastapi.py 2>&1 | sed 's/^/[orchestration] /' & \
        sleep 2 && cd {{dashboard_dir}} && npm run dev 2>&1 | sed 's/^/[dashboard] /' & \
        wait

# Start all services with Ollama instead of vllm-mlx
dev-ollama:
    @trap 'kill 0' EXIT; \
        uv run python {{transcription_dir}}/src/main_local.py --model medium 2>&1 | sed 's/^/[transcription] /' & \
        sleep 5 && \
        TRANSCRIPTION_HOST=localhost TRANSCRIPTION_PORT=5001 \
        LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:3b \
        LLM_TIMEOUT_S=15 DEFAULT_TARGET_LANGUAGE=es \
        uv run python {{orchestration_dir}}/src/main_fastapi.py 2>&1 | sed 's/^/[orchestration] /' & \
        sleep 2 && cd {{dashboard_dir}} && npm run dev 2>&1 | sed 's/^/[dashboard] /' & \
        wait

# Start translation LLM server (vllm-mlx on Apple Silicon)
dev-llm model=llm_model:
    uv run vllm-mlx serve {{model}} --port {{llm_port}}

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
    cd {{orchestration_dir}} && uv run pytest tests/unit/ tests/integration/ tests/test_recorder.py -v --timeout=30

# Run shared contract + TS alignment tests
test-contracts:
    uv run pytest modules/shared/tests/ -v --timeout=30

# Run orchestration tests
test-orchestration:
    cd {{orchestration_dir}} && uv run pytest tests/unit/ tests/integration/ tests/test_recorder.py -v --timeout=30

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
    cd {{orchestration_dir}} && uv run alembic upgrade head

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

# Build all Docker images
docker-build:
    docker compose -f compose.local.yml build

# Start compose (core + inference)
compose-up:
    COMPOSE_PROFILES=core,inference docker compose -f compose.local.yml up --build

# Stop compose
compose-down:
    docker compose -f compose.local.yml down

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
