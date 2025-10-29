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
# - For Playwright: Dependencies for Chromium
# - For audio processing: ffmpeg, pulseaudio
# - For virtual webcam: v4l2loopback
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    fonts-unifont \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    ffmpeg \
    pulseaudio \
    xvfb \
    v4l2loopback-dkms \
    v4l2loopback-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium only for Google Meet)
# Note: System dependencies already installed above, skipping install-deps
RUN playwright install chromium

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
