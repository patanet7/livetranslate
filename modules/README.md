# LiveTranslate Modules

This directory contains all service modules and shared packages.

## Core Runtime Modules

| Module | Path | Default Port | Responsibility |
|---|---|---:|---|
| Orchestration | `modules/orchestration-service` | 3000 | API gateway, coordination, persistence |
| Transcription | `modules/transcription-service` | 5001 | Speech processing and streaming transcription |
| Dashboard | `modules/dashboard-service` | 5173 | SvelteKit UI and operator workflows |
| Shared | `modules/shared` | n/a | Cross-service shared code |

## Supporting/Adjacent Modules

| Module | Path | Purpose |
|---|---|---|
| Meeting Bot Service | `modules/meeting-bot-service` | Canonical meeting bot runtime service |
| Bot Container | `modules/bot-container` | Supporting bot runtime assets and integration docs |
| Seamless | `modules/seamless` | Seamless model experimentation/docs |
| FunASR | `modules/funasr` | FunASR experimentation/docs |

## Module Documentation

- `modules/orchestration-service/README.md`
- `modules/transcription-service/README.md`
- `modules/dashboard-service/README.md`
- `modules/meeting-bot-service/README.md`
- `modules/shared/README.md`

## Development Entry Points

- Repository quick start: `docs/guides/quick-start.md`
- Service architecture docs: `docs/02-containers/README.md`
- Component maps: `docs/03-components/README.md`
