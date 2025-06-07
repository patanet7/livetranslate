# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install additional dependencies
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    sounddevice>=0.4.6 \
    soundfile>=0.12.1 \
    websockets>=10.4 \
    python-dotenv>=1.0.0 \
    asyncio>=3.4.3 \
    requests>=2.31.0 \
    scipy>=1.11.0 \
    vllm \
    transformers \
    torch \
    accelerate

# Create app directory
WORKDIR /app

# Copy application code
COPY python/ /app/python/
COPY tests/ /app/tests/

# Create models directory
RUN mkdir -p /root/.whisper/models
RUN mkdir -p /root/.cache/huggingface

# Set cache permissions
RUN chmod -R 777 /root/.cache

# Expose ports
EXPOSE 5000 8765 8010

# Default command
CMD ["python3", "-m", "python.server", "--help"] 