set shell := ['bash', '-eu', '-o', 'pipefail', '-c']

default: help

help:
    @printf "Available tasks:\n"
    @just --list

bootstrap-env:
    @cp -n env.template .env.local || true
    @echo ".env.local ready (edit as needed)."

compose-up profiles='core,inference':
    @echo "Starting compose profiles: ${profiles}"
    @COMPOSE_PROFILES=${profiles} docker compose -f compose.local.yml up --build

compose-down:
    @docker compose -f compose.local.yml down

compose-logs:
    @docker compose -f compose.local.yml logs -f

deploy-build:
    @docker compose -f compose.local.yml build orchestration frontend

fmt-backend:
    @cd modules/orchestration-service && poetry run black src && poetry run isort src

lint-backend:
    @cd modules/orchestration-service && poetry run flake8 src

mypy:
    @cd modules/orchestration-service && poetry run mypy src

test-backend mark='':
    @cd modules/orchestration-service && poetry run pytest ${mark:+-m ${mark}}

fmt-frontend:
    @cd modules/frontend-service && pnpm format

lint-frontend:
    @cd modules/frontend-service && pnpm lint

test-frontend:
    @cd modules/frontend-service && pnpm test

pre-commit-install:
    @pre-commit install

pre-commit-run:
    @pre-commit run --all-files

ci-check:
    @just fmt-backend
    @just lint-backend
    @just mypy
    @just fmt-frontend
    @just lint-frontend
    @just test-backend
    @just test-frontend
