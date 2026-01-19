# syntax=docker/dockerfile:1.4

# =============================================================================
# Stage 1: Builder
# Install dependencies and prepare the application
# =============================================================================
FROM python:3.14-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PDM_CHECK_UPDATE=false

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install --no-cache-dir pdm

# Set working directory
WORKDIR /app

# Copy only dependency files first (layer caching)
COPY pyproject.toml pdm.lock* ./

# Install dependencies (production only, no lock file generation)
RUN pdm install --prod --no-lock --no-editable

# =============================================================================
# Stage 2: Runtime
# Minimal runtime image with only necessary components
# =============================================================================
FROM python:3.14-slim AS runtime

# Labels for metadata
LABEL org.opencontainers.image.title="Whisper Service" \
      org.opencontainers.image.description="NPU-optimized Whisper transcription service" \
      org.opencontainers.image.vendor="LiveTranslate" \
      org.opencontainers.image.version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app:/app/src \
    WHISPER_CACHE_DIR=/app/.whisper

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    portaudio19-dev \
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

# Copy source code
COPY --chown=appuser:appuser src ./src

# Create necessary directories with proper permissions
RUN mkdir -p /app/.whisper/models /app/sessions /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Default command
CMD ["python", "src/api_server.py", "--host", "0.0.0.0", "--port", "5001"]
