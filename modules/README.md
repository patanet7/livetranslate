# LiveTranslate Modules

This directory contains all service modules and shared packages.

## Core Runtime Modules

| Module | Path | Default Port | Responsibility |
|---|---|---:|---|
| Orchestration | `modules/orchestration-service` | 3000 | API gateway, coordination, persistence |
| Whisper | `modules/whisper-service` | 5001 | Transcription and speech processing |
| Translation | `modules/translation-service` | 5003 | Translation backend and model routing |
| Frontend | `modules/frontend-service` | 5173 | Web UI and operator workflows |
| Shared | `modules/shared` | n/a | Cross-service shared code |

## Supporting/Adjacent Modules

| Module | Path | Purpose |
|---|---|---|
| Bot Container | `modules/bot-container` | Bot runtime assets and integration docs |
| Meeting Bot Service | `modules/meeting-bot-service` | Meeting bot service variant and deployment docs |
| Seamless | `modules/seamless` | Seamless model experimentation/docs |
| FunASR | `modules/funasr` | FunASR experimentation/docs |

## Module Documentation

- `modules/orchestration-service/README.md`
- `modules/whisper-service/README.md`
- `modules/translation-service/README.md`
- `modules/frontend-service/README.md`
- `modules/shared/README.md`

## Development Entry Points

- Repository quick start: `docs/guides/quick-start.md`
- Service architecture docs: `docs/02-containers/README.md`
- Component maps: `docs/03-components/README.md`
