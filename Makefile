# ==============================================================================
# LiveTranslate Makefile
# ==============================================================================
#
# Comprehensive build automation for the LiveTranslate project.
#
# Usage:
#   make [target]
#
# Run 'make help' to see all available targets.
#

.PHONY: help install dev test lint format docker-build docker-up docker-down clean \
        install-frontend install-backend install-whisper install-translation \
        test-frontend test-backend test-whisper test-translation \
        lint-frontend lint-backend format-frontend format-backend \
        db-up db-down db-migrate docker-build-all docker-logs \
        coverage coverage-backend coverage-frontend

# Default shell
SHELL := /bin/bash

# Project directories
PROJECT_ROOT := $(shell pwd)
FRONTEND_DIR := $(PROJECT_ROOT)/modules/frontend-service
ORCHESTRATION_DIR := $(PROJECT_ROOT)/modules/orchestration-service
WHISPER_DIR := $(PROJECT_ROOT)/modules/whisper-service
TRANSLATION_DIR := $(PROJECT_ROOT)/modules/translation-service

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# ==============================================================================
# Help
# ==============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(CYAN)LiveTranslate Makefile$(NC)"
	@echo "========================"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC)"
	@echo "  make [target]"
	@echo ""
	@echo "$(YELLOW)Available Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ==============================================================================
# Installation
# ==============================================================================

install: install-frontend install-backend ## Install all dependencies
	@echo "$(GREEN)All dependencies installed$(NC)"

install-frontend: ## Install frontend dependencies
	@echo "$(CYAN)Installing frontend dependencies...$(NC)"
	@cd $(FRONTEND_DIR) && pnpm install
	@echo "$(GREEN)Frontend dependencies installed$(NC)"

install-backend: install-orchestration install-whisper install-translation ## Install all backend dependencies
	@echo "$(GREEN)All backend dependencies installed$(NC)"

install-orchestration: ## Install orchestration service dependencies
	@echo "$(CYAN)Installing orchestration service dependencies...$(NC)"
	@cd $(ORCHESTRATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm install --no-self 2>/dev/null || pdm install; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry install --no-root; \
		else \
			pip install -r requirements.txt; \
		fi
	@echo "$(GREEN)Orchestration service dependencies installed$(NC)"

install-whisper: ## Install whisper service dependencies
	@echo "$(CYAN)Installing whisper service dependencies...$(NC)"
	@cd $(WHISPER_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm install --no-self 2>/dev/null || pdm install; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry install --no-root; \
		else \
			pip install -r requirements.txt; \
		fi
	@echo "$(GREEN)Whisper service dependencies installed$(NC)"

install-translation: ## Install translation service dependencies
	@echo "$(CYAN)Installing translation service dependencies...$(NC)"
	@cd $(TRANSLATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm install --no-self 2>/dev/null || pdm install; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry install --no-root; \
		else \
			pip install -r requirements.txt; \
		fi
	@echo "$(GREEN)Translation service dependencies installed$(NC)"

install-all: install ## Alias for install (install all dependencies)

# ==============================================================================
# Development
# ==============================================================================

dev: ## Start development environment
	@echo "$(CYAN)Starting development environment...$(NC)"
	@./start-development.sh

dev-frontend: ## Start only frontend service
	@echo "$(CYAN)Starting frontend service...$(NC)"
	@cd $(FRONTEND_DIR) && ./start-frontend.sh

dev-backend: ## Start only backend service
	@echo "$(CYAN)Starting backend service...$(NC)"
	@cd $(ORCHESTRATION_DIR) && ./start-backend.sh

# ==============================================================================
# Testing
# ==============================================================================

test: test-backend test-frontend ## Run all tests
	@echo "$(GREEN)All tests completed$(NC)"

test-frontend: ## Run frontend tests
	@echo "$(CYAN)Running frontend tests...$(NC)"
	@cd $(FRONTEND_DIR) && pnpm test

test-backend: test-orchestration test-whisper test-translation ## Run all backend tests
	@echo "$(GREEN)All backend tests completed$(NC)"

test-orchestration: ## Run orchestration service tests
	@echo "$(CYAN)Running orchestration service tests...$(NC)"
	@mkdir -p $(ORCHESTRATION_DIR)/tests/output
	@cd $(ORCHESTRATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run pytest tests/ -v 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_test_orchestration_results.log; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run pytest tests/ -v 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_test_orchestration_results.log; \
		else \
			python -m pytest tests/ -v 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_test_orchestration_results.log; \
		fi

test-whisper: ## Run whisper service tests
	@echo "$(CYAN)Running whisper service tests...$(NC)"
	@mkdir -p $(WHISPER_DIR)/tests/output
	@cd $(WHISPER_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run pytest tests/ -v 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_test_whisper_results.log; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run pytest tests/ -v 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_test_whisper_results.log; \
		else \
			python -m pytest tests/ -v 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_test_whisper_results.log; \
		fi

test-translation: ## Run translation service tests
	@echo "$(CYAN)Running translation service tests...$(NC)"
	@mkdir -p $(TRANSLATION_DIR)/tests/output
	@cd $(TRANSLATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run pytest tests/ -v 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_test_translation_results.log; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run pytest tests/ -v 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_test_translation_results.log; \
		else \
			python -m pytest tests/ -v 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_test_translation_results.log; \
		fi

# ==============================================================================
# Coverage
# ==============================================================================

coverage: coverage-backend coverage-frontend ## Generate all coverage reports
	@echo "$(GREEN)All coverage reports generated$(NC)"

coverage-backend: ## Generate coverage reports for all Python services
	@echo "$(CYAN)Generating backend coverage reports...$(NC)"
	@mkdir -p $(ORCHESTRATION_DIR)/tests/output
	@cd $(ORCHESTRATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_coverage_orchestration.log; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_coverage_orchestration.log; \
		fi
	@mkdir -p $(WHISPER_DIR)/tests/output
	@cd $(WHISPER_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_coverage_whisper.log; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_coverage_whisper.log; \
		fi
	@mkdir -p $(TRANSLATION_DIR)/tests/output
	@cd $(TRANSLATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_coverage_translation.log; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing 2>&1 | tee tests/output/$$(date +%Y%m%d_%H%M%S)_coverage_translation.log; \
		fi
	@echo "$(GREEN)Backend coverage reports generated$(NC)"

coverage-frontend: ## Generate frontend coverage report
	@echo "$(CYAN)Generating frontend coverage report...$(NC)"
	@cd $(FRONTEND_DIR) && pnpm test:coverage || pnpm test -- --coverage
	@echo "$(GREEN)Frontend coverage report generated$(NC)"

# ==============================================================================
# Linting
# ==============================================================================

lint: lint-backend lint-frontend ## Run all linting
	@echo "$(GREEN)All linting completed$(NC)"

lint-frontend: ## Run frontend linting
	@echo "$(CYAN)Linting frontend...$(NC)"
	@cd $(FRONTEND_DIR) && pnpm lint

lint-backend: ## Run backend linting
	@echo "$(CYAN)Linting backend services...$(NC)"
	@cd $(ORCHESTRATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run flake8 src || true; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run flake8 src || true; \
		fi
	@cd $(WHISPER_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run flake8 src || true; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run flake8 src || true; \
		fi
	@cd $(TRANSLATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run flake8 src || true; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run flake8 src || true; \
		fi

# ==============================================================================
# Formatting
# ==============================================================================

format: format-backend format-frontend ## Run all formatters
	@echo "$(GREEN)All formatting completed$(NC)"

format-frontend: ## Format frontend code
	@echo "$(CYAN)Formatting frontend...$(NC)"
	@cd $(FRONTEND_DIR) && pnpm format || pnpm prettier --write "src/**/*.{ts,tsx,js,jsx}"

format-backend: ## Format backend code
	@echo "$(CYAN)Formatting backend services...$(NC)"
	@cd $(ORCHESTRATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run black src && pdm run isort src; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run black src && poetry run isort src; \
		fi
	@cd $(WHISPER_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run black src && pdm run isort src; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run black src && poetry run isort src; \
		fi
	@cd $(TRANSLATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run black src && pdm run isort src; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run black src && poetry run isort src; \
		fi

# ==============================================================================
# Docker
# ==============================================================================

docker-build: docker-build-all ## Build all Docker images

docker-build-all: ## Build all Docker images
	@echo "$(CYAN)Building all Docker images...$(NC)"
	@docker compose -f docker-compose.dev.yml build 2>/dev/null || \
		docker compose -f compose.local.yml build 2>/dev/null || \
		echo "$(YELLOW)No compose file found for building$(NC)"
	@echo "$(GREEN)Docker images built$(NC)"

docker-build-orchestration: ## Build orchestration service Docker image
	@echo "$(CYAN)Building orchestration service Docker image...$(NC)"
	@cd $(ORCHESTRATION_DIR) && docker build -t livetranslate-orchestration:latest .

docker-build-frontend: ## Build frontend Docker image
	@echo "$(CYAN)Building frontend Docker image...$(NC)"
	@cd $(FRONTEND_DIR) && docker build -t livetranslate-frontend:latest .

docker-build-whisper: ## Build whisper service Docker image
	@echo "$(CYAN)Building whisper service Docker image...$(NC)"
	@cd $(WHISPER_DIR) && docker build -t livetranslate-whisper:latest .

docker-build-translation: ## Build translation service Docker image
	@echo "$(CYAN)Building translation service Docker image...$(NC)"
	@cd $(TRANSLATION_DIR) && docker build -t livetranslate-translation:latest .

docker-up: ## Start Docker Compose services
	@echo "$(CYAN)Starting Docker Compose services...$(NC)"
	@docker compose -f docker-compose.database.yml up -d
	@echo "$(GREEN)Docker services started$(NC)"

docker-down: ## Stop Docker Compose services
	@echo "$(CYAN)Stopping Docker Compose services...$(NC)"
	@docker compose -f docker-compose.database.yml down
	@echo "$(GREEN)Docker services stopped$(NC)"

docker-logs: ## Show Docker Compose logs
	@docker compose -f docker-compose.database.yml logs -f

docker-status: ## Show Docker container status
	@docker compose -f docker-compose.database.yml ps

# ==============================================================================
# Database
# ==============================================================================

db-up: ## Start PostgreSQL container
	@echo "$(CYAN)Starting PostgreSQL container...$(NC)"
	@docker compose -f docker-compose.database.yml up -d postgres redis
	@echo "$(GREEN)Database services started$(NC)"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Redis: localhost:6379"

db-down: ## Stop PostgreSQL container
	@echo "$(CYAN)Stopping database containers...$(NC)"
	@docker compose -f docker-compose.database.yml down
	@echo "$(GREEN)Database services stopped$(NC)"

db-migrate: ## Run database migrations
	@echo "$(CYAN)Running database migrations...$(NC)"
	@cd $(ORCHESTRATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run alembic upgrade head; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run alembic upgrade head; \
		else \
			alembic upgrade head; \
		fi
	@echo "$(GREEN)Database migrations completed$(NC)"

db-shell: ## Open PostgreSQL shell
	@docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate

db-reset: ## Reset database (WARNING: destroys data)
	@echo "$(RED)WARNING: This will destroy all database data!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	@docker compose -f docker-compose.database.yml down -v
	@docker compose -f docker-compose.database.yml up -d postgres redis
	@echo "$(GREEN)Database reset completed$(NC)"

# ==============================================================================
# Cleanup
# ==============================================================================

clean: ## Clean build artifacts and caches
	@echo "$(CYAN)Cleaning build artifacts and caches...$(NC)"
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
	@find . -type d -name ".coverage.*" -exec rm -rf {} + 2>/dev/null || true
	@# Node artifacts
	@find . -type d -name "node_modules" -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".next" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	@# IDE and editor artifacts
	@find . -type d -name ".idea" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.swp" -delete 2>/dev/null || true
	@find . -type f -name "*.swo" -delete 2>/dev/null || true
	@find . -type f -name "*~" -delete 2>/dev/null || true
	@# OS artifacts
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@find . -type f -name "Thumbs.db" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed$(NC)"

clean-docker: ## Clean Docker resources
	@echo "$(CYAN)Cleaning Docker resources...$(NC)"
	@docker compose -f docker-compose.database.yml down -v --remove-orphans 2>/dev/null || true
	@docker system prune -f
	@echo "$(GREEN)Docker cleanup completed$(NC)"

clean-all: clean clean-docker ## Clean everything
	@echo "$(GREEN)Full cleanup completed$(NC)"

# ==============================================================================
# Type Checking
# ==============================================================================

typecheck: ## Run type checking
	@echo "$(CYAN)Running type checks...$(NC)"
	@cd $(ORCHESTRATION_DIR) && \
		if command -v pdm >/dev/null 2>&1; then \
			pdm run mypy src || true; \
		elif command -v poetry >/dev/null 2>&1; then \
			poetry run mypy src || true; \
		fi
	@cd $(FRONTEND_DIR) && pnpm type-check || pnpm tsc --noEmit || true
	@echo "$(GREEN)Type checking completed$(NC)"

# ==============================================================================
# CI/CD
# ==============================================================================

ci: lint typecheck test ## Run full CI pipeline
	@echo "$(GREEN)CI pipeline completed$(NC)"

pre-commit: ## Run pre-commit hooks
	@echo "$(CYAN)Running pre-commit hooks...$(NC)"
	@pre-commit run --all-files || true

# ==============================================================================
# Version Info
# ==============================================================================

version: ## Show version information
	@echo "$(CYAN)LiveTranslate Version Information$(NC)"
	@echo "=================================="
	@echo "Project: LiveTranslate"
	@echo ""
	@echo "$(YELLOW)Tools:$(NC)"
	@node --version 2>/dev/null && echo "  Node.js: $$(node --version)" || echo "  Node.js: not found"
	@pnpm --version 2>/dev/null && echo "  pnpm: $$(pnpm --version)" || echo "  pnpm: not found"
	@python3 --version 2>/dev/null && echo "  Python: $$(python3 --version 2>&1)" || echo "  Python: not found"
	@pdm --version 2>/dev/null && echo "  PDM: $$(pdm --version 2>&1)" || echo "  PDM: not found"
	@poetry --version 2>/dev/null && echo "  Poetry: $$(poetry --version 2>&1)" || echo "  Poetry: not found"
	@docker --version 2>/dev/null && echo "  Docker: $$(docker --version)" || echo "  Docker: not found"
