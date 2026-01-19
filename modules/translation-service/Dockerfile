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
# Use build arg to allow switching between CPU and GPU dependencies
ARG PDM_INSTALL_ARGS="--prod --no-lock --no-editable"
RUN pdm install ${PDM_INSTALL_ARGS}

# =============================================================================
# Stage 2: Runtime
# Minimal runtime image with only necessary components
# =============================================================================
FROM python:3.14-slim AS runtime

# Labels for metadata
LABEL org.opencontainers.image.title="Translation Service" \
      org.opencontainers.image.description="GPU-optimized multi-language translation service" \
      org.opencontainers.image.vendor="LiveTranslate" \
      org.opencontainers.image.version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src

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

# Copy source code with proper ownership
COPY --chown=appuser:appuser src/ ./src/

# Ensure proper ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5003/health || exit 1

# Default command - run the Flask API server
CMD ["python", "src/api_server.py"]
