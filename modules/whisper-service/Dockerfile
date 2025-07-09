# Use Python base image for better compatibility
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create directories
RUN mkdir -p /root/.whisper/models /app/sessions /app/logs

# Set permissions
RUN chmod -R 777 /root/.whisper
RUN chmod -R 777 /app/sessions
RUN chmod -R 777 /app/logs

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Default command
CMD ["python", "src/api_server.py", "--host", "0.0.0.0", "--port", "5001"]
