# ==============================================================================
# LiveTranslate Justfile
# ==============================================================================
#
# Enhanced just commands for the LiveTranslate project.
# Install just: https://github.com/casey/just
#
# Usage:
#   just [recipe]
#   just --list
#

set shell := ['bash', '-eu', '-o', 'pipefail', '-c']

# Project directories
project_root := justfile_directory()
orchestration_dir := project_root / "modules/orchestration-service"
whisper_dir := project_root / "modules/whisper-service"
translation_dir := project_root / "modules/translation-service"
frontend_dir := project_root / "modules/frontend-service"

# Default recipe
default: help

# ==============================================================================
# Help
# ==============================================================================

# Show available recipes
help:
    @printf "LiveTranslate Just Commands\n"
    @printf "===========================\n\n"
    @just --list

# ==============================================================================
# Environment Setup
# ==============================================================================

# Bootstrap development environment
bootstrap-env:
    @cp -n env.template .env.local || true
    @echo ".env.local ready (edit as needed)."

# Install all dependencies
install-all:
    @echo "Installing all dependencies..."
    @cd {{orchestration_dir}} && (pdm install --no-self 2>/dev/null || pdm install || poetry install --no-root)
    @cd {{whisper_dir}} && (pdm install --no-self 2>/dev/null || pdm install || poetry install --no-root)
    @cd {{translation_dir}} && (pdm install --no-self 2>/dev/null || pdm install || poetry install --no-root)
    @cd {{frontend_dir}} && pnpm install
    @echo "All dependencies installed!"

# ==============================================================================
# Development
# ==============================================================================

# Start full development environment
dev:
    @./start-development.sh

# Start frontend only
dev-frontend:
    @cd {{frontend_dir}} && ./start-frontend.sh

# Start backend only
dev-backend:
    @cd {{orchestration_dir}} && ./start-backend.sh

# ==============================================================================
# Docker Compose
# ==============================================================================

# Start compose with profiles (default: core,inference)
compose-up profiles='core,inference':
    @echo "Starting compose profiles: {{profiles}}"
    @COMPOSE_PROFILES={{profiles}} docker compose -f compose.local.yml up --build

# Stop compose services
compose-down:
    @docker compose -f compose.local.yml down

# Show compose logs
compose-logs:
    @docker compose -f compose.local.yml logs -f

# Build deployment images
deploy-build:
    @docker compose -f compose.local.yml build orchestration frontend

# ==============================================================================
# Database
# ==============================================================================

# Start PostgreSQL and Redis containers
db-up:
    @echo "Starting database services..."
    @docker compose -f docker-compose.database.yml up -d postgres redis
    @echo "Waiting for PostgreSQL to be ready..."
    @sleep 5
    @docker exec livetranslate-postgres pg_isready -U livetranslate -d livetranslate || echo "PostgreSQL starting..."
    @echo "Database services started!"
    @echo "  PostgreSQL: localhost:5432"
    @echo "  Redis: localhost:6379"

# Stop PostgreSQL and Redis containers
db-down:
    @echo "Stopping database services..."
    @docker compose -f docker-compose.database.yml down
    @echo "Database services stopped."

# Run database migrations
db-migrate:
    @echo "Running database migrations..."
    @cd {{orchestration_dir}} && (pdm run alembic upgrade head 2>/dev/null || poetry run alembic upgrade head)
    @echo "Migrations complete!"

# Open PostgreSQL shell
db-shell:
    @docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate

# Reset database (WARNING: destroys all data)
db-reset:
    @echo "WARNING: This will destroy all database data!"
    @read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
    @docker compose -f docker-compose.database.yml down -v
    @docker compose -f docker-compose.database.yml up -d postgres redis
    @echo "Database reset complete!"

# ==============================================================================
# Testing
# ==============================================================================

# Run orchestration service tests
test-orchestration mark='':
    @echo "Running orchestration service tests..."
    @mkdir -p {{orchestration_dir}}/tests/output
    @cd {{orchestration_dir}} && \
        (pdm run pytest tests/ {{mark}} -v 2>&1 || poetry run pytest tests/ {{mark}} -v 2>&1) | \
        tee tests/output/$(date +%Y%m%d_%H%M%S)_test_orchestration_results.log
    @echo "Test results saved to tests/output/"

# Run whisper service tests
test-whisper mark='':
    @echo "Running whisper service tests..."
    @mkdir -p {{whisper_dir}}/tests/output
    @cd {{whisper_dir}} && \
        (pdm run pytest tests/ {{mark}} -v 2>&1 || poetry run pytest tests/ {{mark}} -v 2>&1) | \
        tee tests/output/$(date +%Y%m%d_%H%M%S)_test_whisper_results.log
    @echo "Test results saved to tests/output/"

# Run translation service tests
test-translation mark='':
    @echo "Running translation service tests..."
    @mkdir -p {{translation_dir}}/tests/output
    @cd {{translation_dir}} && \
        (pdm run pytest tests/ {{mark}} -v 2>&1 || poetry run pytest tests/ {{mark}} -v 2>&1) | \
        tee tests/output/$(date +%Y%m%d_%H%M%S)_test_translation_results.log
    @echo "Test results saved to tests/output/"

# Run all backend tests
test-backend mark='':
    @just test-orchestration "{{mark}}"
    @just test-whisper "{{mark}}"
    @just test-translation "{{mark}}"

# Run frontend tests
test-frontend:
    @cd {{frontend_dir}} && pnpm test

# Run all tests
test-all:
    @just test-backend
    @just test-frontend

# ==============================================================================
# Coverage
# ==============================================================================

# Generate coverage report for orchestration service
coverage-orchestration:
    @echo "Generating orchestration service coverage..."
    @mkdir -p {{orchestration_dir}}/tests/output
    @cd {{orchestration_dir}} && \
        (pdm run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1 || \
         poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1) | \
        tee tests/output/$(date +%Y%m%d_%H%M%S)_coverage_orchestration.log
    @echo "Coverage report: {{orchestration_dir}}/htmlcov/index.html"

# Generate coverage report for whisper service
coverage-whisper:
    @echo "Generating whisper service coverage..."
    @mkdir -p {{whisper_dir}}/tests/output
    @cd {{whisper_dir}} && \
        (pdm run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1 || \
         poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1) | \
        tee tests/output/$(date +%Y%m%d_%H%M%S)_coverage_whisper.log
    @echo "Coverage report: {{whisper_dir}}/htmlcov/index.html"

# Generate coverage report for translation service
coverage-translation:
    @echo "Generating translation service coverage..."
    @mkdir -p {{translation_dir}}/tests/output
    @cd {{translation_dir}} && \
        (pdm run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1 || \
         poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1) | \
        tee tests/output/$(date +%Y%m%d_%H%M%S)_coverage_translation.log
    @echo "Coverage report: {{translation_dir}}/htmlcov/index.html"

# Generate coverage reports for all backend services
coverage-backend:
    @just coverage-orchestration
    @just coverage-whisper
    @just coverage-translation

# Generate frontend coverage report
coverage-frontend:
    @echo "Generating frontend coverage..."
    @cd {{frontend_dir}} && (pnpm test:coverage || pnpm test -- --coverage)

# ==============================================================================
# Code Formatting
# ==============================================================================

# Format backend code (orchestration)
fmt-backend:
    @cd {{orchestration_dir}} && (pdm run black src && pdm run isort src 2>/dev/null || poetry run black src && poetry run isort src)

# Format whisper service code
fmt-whisper:
    @cd {{whisper_dir}} && (pdm run black src && pdm run isort src 2>/dev/null || poetry run black src && poetry run isort src)

# Format translation service code
fmt-translation:
    @cd {{translation_dir}} && (pdm run black src && pdm run isort src 2>/dev/null || poetry run black src && poetry run isort src)

# Format all backend code
fmt-all-backend:
    @just fmt-backend
    @just fmt-whisper
    @just fmt-translation

# Format frontend code
fmt-frontend:
    @cd {{frontend_dir}} && pnpm format

# Format all code
fmt-all:
    @just fmt-all-backend
    @just fmt-frontend

# ==============================================================================
# Linting
# ==============================================================================

# Lint orchestration service
lint-backend:
    @cd {{orchestration_dir}} && (pdm run flake8 src 2>/dev/null || poetry run flake8 src)

# Lint whisper service
lint-whisper:
    @cd {{whisper_dir}} && (pdm run flake8 src 2>/dev/null || poetry run flake8 src)

# Lint translation service
lint-translation:
    @cd {{translation_dir}} && (pdm run flake8 src 2>/dev/null || poetry run flake8 src)

# Lint all backend code
lint-all-backend:
    @just lint-backend
    @just lint-whisper
    @just lint-translation

# Lint frontend code
lint-frontend:
    @cd {{frontend_dir}} && pnpm lint

# Lint all code
lint-all:
    @just lint-all-backend
    @just lint-frontend

# ==============================================================================
# Type Checking
# ==============================================================================

# Run mypy on orchestration service
mypy:
    @cd {{orchestration_dir}} && (pdm run mypy src 2>/dev/null || poetry run mypy src)

# Run mypy on all backend services
mypy-all:
    @cd {{orchestration_dir}} && (pdm run mypy src 2>/dev/null || poetry run mypy src) || true
    @cd {{whisper_dir}} && (pdm run mypy src 2>/dev/null || poetry run mypy src) || true
    @cd {{translation_dir}} && (pdm run mypy src 2>/dev/null || poetry run mypy src) || true

# ==============================================================================
# Docker Build
# ==============================================================================

# Build specific service Docker image
docker-build service:
    @echo "Building {{service}} Docker image..."
    @case "{{service}}" in \
        orchestration) cd {{orchestration_dir}} && docker build -t livetranslate-orchestration:latest . ;; \
        frontend) cd {{frontend_dir}} && docker build -t livetranslate-frontend:latest . ;; \
        whisper) cd {{whisper_dir}} && docker build -t livetranslate-whisper:latest . ;; \
        translation) cd {{translation_dir}} && docker build -t livetranslate-translation:latest . ;; \
        *) echo "Unknown service: {{service}}"; exit 1 ;; \
    esac
    @echo "Docker image built: livetranslate-{{service}}:latest"

# Build all service Docker images
docker-build-all:
    @echo "Building all Docker images..."
    @just docker-build orchestration
    @just docker-build frontend
    @just docker-build whisper || echo "Whisper Dockerfile not found, skipping..."
    @just docker-build translation || echo "Translation Dockerfile not found, skipping..."
    @echo "All Docker images built!"

# ==============================================================================
# Cleanup
# ==============================================================================

# Clean all build artifacts, caches, and temporary files
clean:
    @echo "Cleaning build artifacts and caches..."
    @# Python artifacts
    @find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    @find . -type f -name "*.pyc" -delete 2>/dev/null || true
    @find . -type f -name "*.pyo" -delete 2>/dev/null || true
    @find . -type f -name ".coverage" -delete 2>/dev/null || true
    @find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
    @# Node artifacts
    @find . -path "*/node_modules" -prune -type d -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name "dist" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name "build" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    @# OS artifacts
    @find . -type f -name ".DS_Store" -delete 2>/dev/null || true
    @find . -type f -name "Thumbs.db" -delete 2>/dev/null || true
    @echo "Cleanup complete!"

# Clean Docker resources
clean-docker:
    @echo "Cleaning Docker resources..."
    @docker compose -f docker-compose.database.yml down -v --remove-orphans 2>/dev/null || true
    @docker system prune -f
    @echo "Docker cleanup complete!"

# ==============================================================================
# Pre-commit
# ==============================================================================

# Install pre-commit hooks
pre-commit-install:
    @pre-commit install

# Run pre-commit on all files
pre-commit-run:
    @pre-commit run --all-files

# ==============================================================================
# CI/CD
# ==============================================================================

# Run full CI check (format, lint, type check, test)
ci-check:
    @just fmt-backend
    @just lint-backend
    @just mypy
    @just fmt-frontend
    @just lint-frontend
    @just test-backend
    @just test-frontend

# Quick check (lint only, no tests)
quick-check:
    @just lint-all-backend
    @just lint-frontend
    @just mypy-all

# ==============================================================================
# Utilities
# ==============================================================================

# Check Docker environment
check-docker:
    @./scripts/check_docker_env.sh

# Show version info
version:
    @echo "LiveTranslate Development Environment"
    @echo "====================================="
    @echo ""
    @echo "Tools:"
    @node --version 2>/dev/null && echo "  Node.js: $(node --version)" || echo "  Node.js: not found"
    @pnpm --version 2>/dev/null && echo "  pnpm: $(pnpm --version)" || echo "  pnpm: not found"
    @python3 --version 2>/dev/null && echo "  Python: $(python3 --version 2>&1)" || echo "  Python: not found"
    @pdm --version 2>/dev/null && echo "  PDM: $(pdm --version 2>&1)" || echo "  PDM: not found"
    @poetry --version 2>/dev/null && echo "  Poetry: $(poetry --version 2>&1)" || echo "  Poetry: not found"
    @docker --version 2>/dev/null && echo "  Docker: $(docker --version)" || echo "  Docker: not found"
    @just --version 2>/dev/null && echo "  Just: $(just --version)" || echo "  Just: not found"
