# syntax=docker/dockerfile:1.4

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

ARG POETRY_INSTALL_ARGS="--without dev,test"
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi ${POETRY_INSTALL_ARGS}

FROM base as runtime

WORKDIR /app

ENV PYTHONPATH=/app:/app/src

COPY src ./src
COPY config ./config
COPY static ./static
COPY templates ./templates

ENV UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=3000

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${UVICORN_PORT}/api/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
