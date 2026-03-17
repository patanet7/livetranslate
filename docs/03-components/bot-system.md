# Bot System

This document defines the active bot architecture after the bot-system consolidation pass.

## Canonical Runtime

The active meeting bot runtime is [`modules/meeting-bot-service`](/Users/thomaspatane/GitHub/personal/livetranslate/modules/meeting-bot-service).

Reasons this is the canonical runtime:

- it contains the HTTP API used by orchestration in [`src/api_server.ts`](/Users/thomaspatane/GitHub/personal/livetranslate/modules/meeting-bot-service/src/api_server.ts)
- it contains the audio streaming integration in [`src/audio_streaming.ts`](/Users/thomaspatane/GitHub/personal/livetranslate/modules/meeting-bot-service/src/audio_streaming.ts)
- it contains the platform-specific Chrome path handling in [`src/config.ts`](/Users/thomaspatane/GitHub/personal/livetranslate/modules/meeting-bot-service/src/config.ts)
- orchestration integration tests already target this service on port `5005`

## Canonical Control Plane

The active orchestration-side bot control path is:

- [`modules/orchestration-service/src/bot/docker_bot_manager.py`](/Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service/src/bot/docker_bot_manager.py)
- [`modules/orchestration-service/src/routers/bot/bot_docker_management.py`](/Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service/src/routers/bot/bot_docker_management.py)
- [`modules/orchestration-service/src/clients/meeting_bot_service_client.py`](/Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service/src/clients/meeting_bot_service_client.py)

This path owns container lifecycle, bot status tracking, callbacks, and the dashboard-facing API.

## Legacy / Removed Paths

### Removed duplicate

- `meeting-bot/`

This was a near-duplicate of `modules/meeting-bot-service` but lacked the API server and audio streaming additions that the active system uses.

### Legacy, not canonical

- [`modules/orchestration-service/src/bot/bot_manager.py`](/Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service/src/bot/bot_manager.py)

This Python-native manager remains in the repo as legacy functionality, but it is not the canonical runtime path for current bot orchestration.

## Feature Matrix

| Capability | `modules/meeting-bot-service` | Docker manager path | Legacy Python bot manager |
|---|---|---|---|
| HTTP bot join/status/leave API | Canonical | Consumes via client/proxy | No |
| Google Meet browser automation | Canonical | No | Legacy/alternate |
| Audio streaming to orchestration | Canonical | No | Partial/legacy |
| Docker/container lifecycle | No | Canonical | No |
| Bot status and callback tracking | Service-local only | Canonical | Legacy/alternate |
| Dashboard start/status flow | Via orchestration | Canonical | No |
| Real local integration target on port `5005` | Canonical | Uses it | No |
| Multi-step health/state management | Minimal | Canonical | Legacy/alternate |
| Virtual webcam / advanced analytics internals | Partial | Control only | Legacy-heavy |

## Canonical Commands

### Start the bot runtime locally

```bash
cd modules/meeting-bot-service
npm install
npm run api
```

Health check:

```bash
curl http://localhost:5005/api/health
```

### Start the bot runtime with Docker

```bash
cd modules/meeting-bot-service
docker compose up --build
```

### Start orchestration locally

```bash
cd modules/orchestration-service
uv sync --all-packages --group dev
uv run python src/main_fastapi.py
```

### Orchestration bot routes

Mounted by FastAPI as:

- `POST /api/start`
- `POST /api/stop/{connection_id}`
- `GET /api/status/{connection_id}`
- `GET /api/list`
- `POST /api/command/{connection_id}`

Dashboard proxy routes:

- `POST /api/bot/start`
- `GET /api/bot/{connection_id}/status`

## Removal Rules

- Do not recreate a second active bot runtime at repository root.
- New runtime features belong in `modules/meeting-bot-service`.
- New orchestration-side lifecycle features belong in the Docker manager path.
- Legacy Python bot manager code should be treated as migration source material, not the default place for new work.
