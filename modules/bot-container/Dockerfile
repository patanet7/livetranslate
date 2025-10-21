# LiveTranslate Bot Container
# Phase 3.3: Simplified Bot Architecture
#
# This bot is a "headless frontend" that:
# 1. Joins Google Meet via browser automation
# 2. Captures audio
# 3. Streams to orchestration (SAME protocol as frontend!)
# 4. Receives segments back

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - For browser automation: chromium, chromedriver
# - For audio processing: ffmpeg, pulseaudio
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    ffmpeg \
    pulseaudio \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src

# Environment variables (set by bot manager at runtime)
# - MEETING_URL: Google Meet URL to join
# - CONNECTION_ID: Unique bot connection ID
# - USER_TOKEN: User API token for orchestration
# - ORCHESTRATION_WS_URL: Orchestration WebSocket URL
# - REDIS_URL: Redis URL for commands
# - BOT_MANAGER_URL: Bot manager URL for callbacks
# - LANGUAGE: Transcription language (default: en)
# - TASK: Transcription task (default: transcribe)

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run bot
CMD ["python", "-u", "src/bot_main.py"]
