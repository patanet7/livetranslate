# Level 2: Container Architecture

LiveTranslate runs as a multi-service system with clear runtime responsibilities.

## Core Runtime Services

| Service | Port | Path | Primary Responsibility |
|---|---:|---|---|
| Frontend | 5173 | `modules/frontend-service` | Browser UI, dashboards, operator workflows |
| Orchestration | 3000 | `modules/orchestration-service` | API gateway, workflow coordination, session management |
| Whisper | 5001 | `modules/whisper-service` | Transcription and speech-related processing |
| Translation | 5003 | `modules/translation-service` | Text translation and backend routing |
| Redis | 6379 | compose-managed | cache/event bus support |
| PostgreSQL | 5432 | compose-managed | persistence |

## Communication Paths

- Frontend -> Orchestration (HTTP/WebSocket).
- Orchestration -> Whisper (HTTP).
- Orchestration -> Translation (HTTP).
- Orchestration -> PostgreSQL/Redis.

## Local Runtime Profiles

- `compose.local.yml` is the active compose file for local development profiles.
- Default helper command:

```bash
just compose-up profiles="core,inference,ui,infra"
```

## Related Container Docs

- [Service Overview](./service-overview.md)
- [Communication Patterns](./communication-patterns.md)
- [Data Flow](./data-flow.md)
- [Deployment Architecture](./deployment-architecture.md)
- [Hardware Optimization](./hardware-optimization.md)

## Next Step

Continue to [Level 3: Component Details](../03-components/README.md).
