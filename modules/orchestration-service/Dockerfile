# syntax=docker/dockerfile:1.4
#
# Build context MUST be the repository root (workspace root):
#     docker build -f modules/orchestration-service/Dockerfile .
#
# The orchestration service depends on the workspace package
# `livetranslate-common` (modules/shared/), which uv can only resolve when
# invoked from a directory containing the workspace's `[tool.uv.workspace]`
# section — i.e. the repo root.

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Workspace manifests — every member must be present so uv can parse the
# workspace before resolving the requested package's dependencies.
COPY pyproject.toml uv.lock ./
COPY modules/shared/pyproject.toml modules/shared/pyproject.toml
COPY modules/orchestration-service/pyproject.toml modules/orchestration-service/pyproject.toml
COPY modules/orchestration-service/README.md modules/orchestration-service/README.md
COPY modules/transcription-service/pyproject.toml modules/transcription-service/pyproject.toml
COPY tools/translation_benchmark/pyproject.toml tools/translation_benchmark/pyproject.toml

# Source for packages we actually build wheels for in this image.
COPY modules/shared/src/ modules/shared/src/
COPY modules/orchestration-service/src/ modules/orchestration-service/src/

# `--no-editable` produces real wheels in .venv (no .pth files pointing at
# /build), so the runtime image is fully self-contained.
RUN uv sync --no-dev --no-editable --package livetranslate-orchestration-service

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Orchestration Service" \
      org.opencontainers.image.description="Backend API coordination and bot management" \
      org.opencontainers.image.vendor="LiveTranslate" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=3000

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# Service application files — paths are relative to the repo-root build context.
COPY --chown=appuser:appuser modules/orchestration-service/src ./src
COPY --chown=appuser:appuser modules/orchestration-service/config ./config
COPY --chown=appuser:appuser modules/orchestration-service/static ./static
COPY --chown=appuser:appuser modules/orchestration-service/templates ./templates

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${UVICORN_PORT}/api/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
