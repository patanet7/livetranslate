# syntax=docker/dockerfile:1.4

# =============================================================================
# Stage 1: Builder
# Install dependencies and prepare the application
# =============================================================================
FROM python:3.14-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PDM_CHECK_UPDATE=false

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install --no-cache-dir pdm

# Set working directory
WORKDIR /app

# Copy only dependency files first (layer caching)
COPY pyproject.toml pdm.lock* ./

# Install dependencies (production only, no lock file generation)
ARG PDM_INSTALL_ARGS="--prod --no-lock --no-editable"
RUN pdm install ${PDM_INSTALL_ARGS}

# =============================================================================
# Stage 2: Runtime
# Minimal runtime image with only necessary components
# =============================================================================
FROM python:3.14-slim AS runtime

# Labels for metadata
LABEL org.opencontainers.image.title="Orchestration Service" \
      org.opencontainers.image.description="Backend API coordination and bot management" \
      org.opencontainers.image.vendor="LiveTranslate" \
      org.opencontainers.image.version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=3000

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -d /app -s /sbin/nologin appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Update PATH to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy application files with proper ownership
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser static ./static
COPY --chown=appuser:appuser templates ./templates

# Ensure proper ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${UVICORN_PORT}/api/health || exit 1

# Default command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
