# Quick Start Guide

This guide uses current repository workflows and avoids legacy commands.

## Prerequisites

- Docker + Docker Compose
- Python 3.12+
- Node.js 20+ and `pnpm`
- `just` (recommended) or direct `docker compose` commands
- `pdm` (primary Python workflow in this repo) or Poetry fallback

## Option A: Local Compose Profiles (Recommended)

```bash
cp -n env.template .env.local
just bootstrap-env
just compose-up profiles="core,inference,ui,infra"
```

No `just` installed:

```bash
cp -n env.template .env.local
COMPOSE_PROFILES="core,inference,ui,infra" docker compose -f compose.local.yml up --build
```

## Option B: Frontend + Orchestration Local Scripts

```bash
./start-development.sh
```

On Windows:

```powershell
./start-development.ps1
```

If you use this mode, run Whisper and Translation separately.

## Option C: Service-by-Service (Manual)

Terminal 1, orchestration:

```bash
cd modules/orchestration-service
(pdm install --no-self || poetry install --no-root)
(pdm run uvicorn src.main:app --host 0.0.0.0 --port 3000 --reload || poetry run uvicorn src.main:app --host 0.0.0.0 --port 3000 --reload)
```

Terminal 2, whisper:

```bash
cd modules/whisper-service
(pdm install --no-self || poetry install --no-root)
(pdm run python src/main.py || poetry run python src/main.py)
```

Terminal 3, translation:

```bash
cd modules/translation-service
(pdm install --no-self || poetry install --no-root)
(pdm run python src/api_server_fastapi.py || poetry run python src/api_server_fastapi.py)
```

Terminal 4, frontend:

```bash
cd modules/frontend-service
pnpm install
pnpm dev --host 0.0.0.0 --port 5173
```

## Health Checks

```bash
curl http://localhost:3000/api/health
curl http://localhost:5001/health
curl http://localhost:5003/api/health
```

## URLs

- Frontend: `http://localhost:5173`
- Orchestration API: `http://localhost:3000`
- API Docs: `http://localhost:3000/docs`

## Next Guides

- [Database Setup](./database-setup.md)
- [Translation Testing](./translation-testing.md)
