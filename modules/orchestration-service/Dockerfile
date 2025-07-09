# Dockerfile for Current Flask-based Orchestration Service
# NOTE: This will be replaced by Dockerfile.react during the React migration
# See REACT_MIGRATION_PLAN.md for details on the upcoming React + FastAPI architecture

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY static/ static/
COPY templates/ templates/
COPY config/ config/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Start command
CMD ["python", "src/orchestration_service.py", "--host", "0.0.0.0", "--port", "3000"]