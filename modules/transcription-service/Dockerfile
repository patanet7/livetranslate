# syntax=docker/dockerfile:1.4
#
# Build context MUST be the repository root (workspace root):
#     docker build -f modules/transcription-service/Dockerfile .
#
# The transcription service depends on the workspace package
# `livetranslate-common` (modules/shared/), which uv can only resolve when
# invoked from a directory containing the workspace's `[tool.uv.workspace]`
# section — i.e. the repo root.

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Workspace manifests — every member must be present so uv can parse the
# workspace before resolving the requested package's dependencies.
COPY pyproject.toml uv.lock ./
COPY modules/shared/pyproject.toml modules/shared/pyproject.toml
COPY modules/orchestration-service/pyproject.toml modules/orchestration-service/pyproject.toml
COPY modules/transcription-service/pyproject.toml modules/transcription-service/pyproject.toml
COPY modules/transcription-service/README.md modules/transcription-service/README.md
COPY tools/translation_benchmark/pyproject.toml tools/translation_benchmark/pyproject.toml

# Source for packages we actually build wheels for in this image.
COPY modules/shared/src/ modules/shared/src/
COPY modules/transcription-service/src/ modules/transcription-service/src/

# `--no-editable` produces real wheels in .venv (no .pth files pointing at
# /build), so the runtime image is fully self-contained.
RUN uv sync --no-dev --no-editable --package transcription-service

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Whisper Service" \
      org.opencontainers.image.description="NPU-optimized Whisper transcription service" \
      org.opencontainers.image.vendor="LiveTranslate" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app:/app/src \
    WHISPER_CACHE_DIR=/app/.whisper

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# Service application files — paths are relative to the repo-root build context.
COPY --chown=appuser:appuser modules/transcription-service/src ./src

RUN mkdir -p /app/.whisper/models /app/sessions /app/logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

CMD ["python", "src/api_server.py", "--host", "0.0.0.0", "--port", "5001"]
