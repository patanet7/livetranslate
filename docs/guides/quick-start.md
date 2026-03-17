# Quick Start Guide

This guide uses current repository workflows and avoids legacy commands.

## Prerequisites

- Docker + Docker Compose
- Python 3.12+
- Node.js 20+ and `pnpm`
- `just` (recommended) or direct `docker compose` commands
- `uv` (Python package manager)

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
uv sync --all-packages --group dev
uv run uvicorn src.main_fastapi:app --host 0.0.0.0 --port 3000 --reload
```

Terminal 2, transcription:

```bash
cd modules/transcription-service
uv sync --all-packages --group dev
uv run python src/main.py
```

Terminal 3, frontend:

```bash
cd modules/frontend-service
pnpm install
pnpm dev --host 0.0.0.0 --port 5173
```

## Health Checks

```bash
curl http://localhost:3000/health
curl http://localhost:5001/health
```

## URLs

- Frontend: `http://localhost:5173`
- Orchestration API: `http://localhost:3000`
- API Docs: `http://localhost:3000/docs`

## Next Guides

- [Database Setup](./database-setup.md)
- [Translation Testing](./translation-testing.md)
