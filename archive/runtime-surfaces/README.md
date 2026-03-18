# Runtime Surface Archive

These files were moved out of the repository root to stop them competing with the active local-development workflow.

## Archived Here

- `docker-compose.dev.yml`
- `docker-compose.minimal.yml`
- `docker-compose.comprehensive.yml`
- `start-development.sh`
- `start-development.ps1`
- `deploy.py`

## Active Replacements

- Canonical local development: `just install`, `just db-up`, `just dev`
- Service-by-service startup: `docs/guides/quick-start.md`
- Database/supporting infra only: `docker-compose.database.yml`
- Optional compatibility compose stack: `docker/optional/compose.local.yml`

These archived files are preserved for historical reference and troubleshooting only.
