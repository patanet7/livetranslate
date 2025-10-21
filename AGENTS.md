# Repository Guidelines

## Project Structure & Module Organization
- Core microservices live in `modules/`: `orchestration-service` (FastAPI backend via Poetry), `frontend-service` (React + Vite), `whisper-service` / `translation-service` (model hosts), and `shared` utilities.  
- Root-level `docker-compose.*.yml` files define common deployment profiles; `scripts/` holds helper automation (secrets, database, security hardening).  
- Tests mirror services: Python coverage under `modules/orchestration-service/tests/`, frontend specs in `modules/frontend-service/src/__tests__`, and cross-service smoke checks at the repo root (`test_*.py`).

## Build, Test, and Development Commands
- Backend: `cd modules/orchestration-service && poetry install && poetry run uvicorn src.main:app --reload` for local FastAPI work.  
- Backend tests: `poetry run pytest` (use markers such as `-m "not slow"`; default timeout 300s).  
- Frontend: `cd modules/frontend-service && pnpm install && pnpm dev` for Vite, `pnpm test` for unit specs, and `pnpm lint` before PRs.  
- Full stack: `docker-compose -f docker-compose.comprehensive.yml up -d` to boot all services; `./start-development.ps1` offers a Windows-friendly shortcut.

## Coding Style & Naming Conventions
- Python code follows Black (88-char lines), isort’s Black profile, and mypy with `disallow_untyped_defs`; keep module names snake_case and FastAPI routers in `src/routers/`.  
- TypeScript uses ESLint + Prettier with 2-space indentation; React components are PascalCase, hooks camelCase, Redux slices stored under `modules/frontend-service/src/store/`.  
- Configuration files (`*.env`, YAML) should remain lowercase with hyphenated keys for clarity.

## Testing Guidelines
- Tag new Python tests with existing pytest markers (`integration`, `audio_pipeline`, etc.) to control suite scope; prefer fixtures from `tests/fixtures`.  
- Maintain parity between frontend components and Vitest coverage; colocate snapshots under `__snapshots__`.  
- When adding microservice integrations, extend the root smoke tests to keep cross-service assertions current.

## Commit & Pull Request Guidelines
- Follow the repository history convention: concise, present-tense summaries (e.g., `Improve websocket retry logic`).  
- Reference related issues in the PR body, list affected services, and include reproduction or validation commands.  
- Attach screenshots or CLI output when UI or monitoring dashboards change, and call out required config/env updates explicitly.

## Configuration & Secrets
- Duplicate `env.template` into `.env` for shared vars; use `scripts/load-secrets.sh` or PowerShell equivalents instead of committing secrets.  
- Database bootstrapping scripts (`scripts/start-database.sh`, `bot-sessions-schema.sql`) should run inside the provided Docker network to match production settings.
