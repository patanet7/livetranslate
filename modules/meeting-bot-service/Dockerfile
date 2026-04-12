# Production Dockerfile for LiveTranslate Meeting Bot
FROM node:22-slim

WORKDIR /usr/src/app

# Install system dependencies for Playwright/Chrome
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  ffmpeg \
  libnss3 \
  libxss1 \
  libasound2 \
  libatk1.0-0 \
  libatk-bridge2.0-0 \
  libcups2 \
  libxkbcommon-x11-0 \
  libgbm-dev \
  libgl1-mesa-dri \
  libgl1-mesa-glx \
  mesa-utils \
  xvfb \
  wget \
  gnupg \
  ca-certificates \
  fonts-liberation \
  && rm -rf /var/lib/apt/lists/*

# Install Chromium (works on both amd64 and arm64)
RUN apt-get update && \
  apt-get install -y chromium && \
  rm -rf /var/lib/apt/lists/* && \
  ln -sf /usr/bin/chromium /usr/bin/google-chrome

# Copy Chrome policy for protocol launching
RUN mkdir -p /etc/opt/chrome/policies/managed
COPY auto_launch_protocols.json /etc/opt/chrome/policies/managed/

# Install ALL dependencies (need devDeps for build + playwright)
COPY package*.json ./
RUN npm ci

# Install Playwright browsers
RUN npx playwright install chromium --with-deps

# Copy source and build
COPY . .
RUN npm run build

# Prune dev dependencies
RUN npm prune --production

# Run as non-root
RUN useradd -m -u 1001 botuser && chown -R botuser:botuser /usr/src/app
USER botuser

EXPOSE 5005

# Run the API server
CMD ["node", "dist/api_server.js"]
