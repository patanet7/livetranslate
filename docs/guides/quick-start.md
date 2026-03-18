# Quick Start Guide

This guide uses current repository workflows and avoids legacy commands.

## Prerequisites

- Python 3.12+
- Node.js 20+ and `pnpm`
- Docker + Docker Compose
- `just` (recommended) or direct service commands
- `uv` (Python package manager)

## Option A: Canonical Local Development (Recommended)

```bash
cp -n env.template .env.local
just install
just db-up
just dev
```

Use `just db-down` to stop PostgreSQL, Redis, and pgAdmin when you are finished.

If you do not want the full `just dev` stack, you can run the services individually:

```bash
just dev-orchestration
just dev-dashboard
just dev-transcription
just dev-meeting-bot
```

## Option B: Service-by-Service (Manual)

```bash
cp -n env.template .env.local
docker compose -f docker-compose.database.yml up -d
```

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

Terminal 3, dashboard:

```bash
cd modules/dashboard-service
npm install
npm run dev
```

Terminal 4, meeting bot runtime:

```bash
cd modules/meeting-bot-service
npm install
npm run api
```

## Option C: Optional Compatibility Compose

This repo still keeps one non-default compose stack for historical compatibility and troubleshooting. It is not the recommended day-to-day workflow.

```bash
cp -n env.template .env.local
just compose-up
```

Direct Docker command:

```bash
COMPOSE_PROFILES="core,inference" docker compose -f docker/optional/compose.local.yml up --build
```

## Health Checks

```bash
curl http://localhost:3000/api/health
curl http://localhost:5001/health
curl http://localhost:5005/api/health
```

## URLs

- Frontend: `http://localhost:5173`
- Orchestration API: `http://localhost:3000`
- API Docs: `http://localhost:3000/docs`
- Meeting Bot Service: `http://localhost:5005`

## Bot Notes

The canonical meeting bot runtime lives in `modules/meeting-bot-service`.

The canonical orchestration bot-control routes are exposed by the orchestration service at:

- `POST /api/start`
- `GET /api/status/{connection_id}`
- `POST /api/stop/{connection_id}`

The old root startup scripts and root-level full-stack compose files have been archived and are no longer part of the active quick-start path.

## Next Guides

- [Database Setup](./database-setup.md)
- [Translation Testing](./translation-testing.md)
