# Level 2: Container Architecture

LiveTranslate runs as a multi-service system with clear runtime responsibilities.

## Core Runtime Services

| Service | Port | Path | Primary Responsibility |
|---|---:|---|---|
| Dashboard | 5173 (dev) / 3000 (prod) | `modules/dashboard-service` | Real-time UI, loopback page, settings, meeting controls |
| Orchestration | 3000 | `modules/orchestration-service` | API gateway, WebSocket hub, translation orchestration |
| Transcription | 5001 | `modules/transcription-service` | Speech-to-text with VAD (pluggable backends) |
| vLLM-MLX STT | 8005 | Inference service | Whisper transcription inference (Apple Silicon) |
| vLLM-MLX LLM | 8006 | Inference service | Qwen3-4B-4bit translation inference (Apple Silicon) |
| Ollama | 11434 | Inference service | Alternative LLM API for translation |
| Redis | 6379 | compose-managed | cache/event bus support |
| PostgreSQL | 5432 | compose-managed | persistence |

## Communication Paths

- Dashboard -> Orchestration (HTTP/WebSocket).
- Orchestration -> Transcription (HTTP).
- Orchestration -> vLLM-MLX :8006 or Ollama (LLM API for translation).
- Transcription -> vLLM-MLX :8005 (Whisper inference).
- Orchestration -> PostgreSQL/Redis.

## Local Runtime Profiles

- Canonical local development uses `just dev` or the service-by-service commands in the quick start guide.
- Default Docker usage is supporting infrastructure only:

```bash
just db-up
```

- The old multi-service Docker stack is preserved only as an optional compatibility workflow:

```bash
just compose-up
```

- That optional stack now lives at `docker/optional/compose.local.yml` and is not the recommended path for day-to-day development.

## Related Container Docs

- [Service Overview](./service-overview.md)
- [Communication Patterns](./communication-patterns.md)
- [Data Flow](./data-flow.md)
- [Deployment Architecture](./deployment-architecture.md)
- [Hardware Optimization](./hardware-optimization.md)

## Next Step

Continue to [Level 3: Component Details](../03-components/README.md).
