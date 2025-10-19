# Update Plan

## Objectives
- Stabilize orchestration service boot by fixing dependency/DI inconsistencies and pruning legacy Python modules.
- Establish a consistent environment workflow covering local dev, automated testing, and production packaging.
- Replace ad-hoc scripts with reproducible tooling (Poetry, pnpm, Just/Make) and coherent Docker/Compose topology.
- Prepare the codebase for discrete Kubernetes deployments with clear separation of stateful vs stateless components.
- Harden observability, configuration, and secret management so the system is production ready.

## Immediate Fixes
- ✅ Align `AudioServiceClient`/`TranslationServiceClient` constructors with `dependencies.py` factories; add regression tests.
- ✅ Remove or archive duplicate/backup routers (`audio_core_temp.py`, `audio_original_backup.py`, etc.) to shrink boot surface.
- ✅ Normalize logging: drop forced DEBUG setup in `main_fastapi.py`, avoid hard-coded `/tmp` log paths.
- Audit environment variables: ensure orchestration reads `AUDIO_SERVICE_URL` / `TRANSLATION_SERVICE_URL`, update `env.template`.

## Architectural Adjustments
- Carve the orchestration service into a lean API layer plus background workers (config sync, Google Meet bots, audio coordination) using a shared queue.
- Decide and document the diarization boundary (remain in Whisper, move to a dedicated speaker service, or fold into orchestration) and delete stale references.
- Introduce an event pipeline for audio/transcription messages so translation and analytics consumers can scale independently.
- Add contract/integration tests that exercise orchestration ↔ whisper/translation APIs to protect independent release cycles.

## Dev Environment & Dependencies
- Standardize on Poetry for backend installs (`poetry install --with dev,audio`) and pnpm for frontend.
- 🔄 Add canonical orchestration upstream variables (`AUDIO_SERVICE_URL`, `TRANSLATION_SERVICE_URL`) to `.env` templates.
- ✅ Provide production-ready frontend Docker image (Vite → nginx) for Compose/K8s workflows.
- Introduce a `justfile` (or Makefile) with tasks: `up-dev`, `down`, `fmt`, `lint`, `test-backend`, `test-frontend`, `compose` profiles.
- Generate `.env.local` from `env.template` via `just bootstrap-env`, scoped per service (`modules/*/.env`).
- Provide lightweight mock services (FastAPI stubs) for Whisper/Translation so local CPUs can run the full workflow.
- Enforce pre-commit hooks that run Black, isort, mypy, ESLint, Prettier before push.

## Docker & Compose Restructure
- Author new `compose.local.yml` using profiles (`orchestration`, `frontend`, `whisper`, `translation`, `redis`, `postgres`, `monitoring`).
- Remove dependency on pre-created external networks/volumes; create them on demand with sensible defaults.
- Split dev vs prod images: dev mounts source, prod images copy built artefacts (frontend served via nginx, backend via gunicorn/uvicorn workers).
- ✅ Update orchestration Dockerfile to build FastAPI service with Poetry-managed dependencies.
- Document workflow in `README.md` (start with `just up-dev`, how to toggle GPU/NPU profiles, running tests).

## CI/CD & Testing
- Update CI pipeline to run backend unit + integration tests, frontend lint/test/build, and Docker image builds per SHA.
- Refine integration tests to use real encoders or stub responses rather than header-based fixtures.
- Add smoke tests that validate service contracts (orchestration ↔ whisper/translation) using mocks.

## Kubernetes Rollout
- Publish container images to a registry; version with `main` SHA.
- Create Helm chart/Kustomize base: Deployments, Services, ConfigMaps, Secrets, HPA for stateless services.
- Use managed Redis/Postgres where possible; mount PVCs only for model caches.
- Configure Ingress routing (`/api`, `/translate`, `/ws`), optional service mesh for retries/mTLS.

## Observability
- Expose `/metrics` endpoints; deploy Prometheus Operator + Grafana dashboards from repo (clean up missing files).
- Ship logs via Loki or OpenTelemetry Collector; tie alerts to Alertmanager/Slack.
- Replace ad-hoc `scripts/load-secrets.sh` with a managed secrets workflow when production deployment is imminent.

## Tracking & Next Steps
- Track progress in this document; update sections as tasks complete.
- Group related changes into logical commits; run `git add` per module and commit with descriptive messages (e.g., “Cleanup orchestration DI and routers”).
- Revisit remaining legacy assets after immediate fixes and environment overhaul are merged.
