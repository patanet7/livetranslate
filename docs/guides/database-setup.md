# Database Setup Guide

Use this guide to run PostgreSQL + Redis for local development.

## Start Database Services

Linux/macOS:

```bash
./scripts/start-database.sh
```

Windows PowerShell:

```powershell
./scripts/start-database.ps1
```

Both scripts wrap `docker-compose.database.yml`.

## Verify Services

```bash
docker compose -f docker-compose.database.yml ps
curl -s http://localhost:8080 > /dev/null && echo "pgAdmin reachable"
docker exec livetranslate-postgres pg_isready -U livetranslate -d livetranslate
docker exec livetranslate-redis redis-cli ping
```

Expected:

- PostgreSQL responds as ready.
- Redis returns `PONG`.
- pgAdmin is reachable at `http://localhost:8080`.

## Connection Strings

```bash
DATABASE_URL=postgresql://livetranslate:livetranslate_dev_password@localhost:5432/livetranslate
REDIS_URL=redis://localhost:6379/0
```

## Helpful Commands

```bash
# open psql shell
docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate

# stop services
docker compose -f docker-compose.database.yml down

# stop and remove volumes (destructive)
docker compose -f docker-compose.database.yml down -v
```

## Related Docs

- [Root Database README](../../README-DATABASE.md)
- [Quick Start](./quick-start.md)
