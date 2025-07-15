# Translation Service - Multi-Backend Translation with Llama 3.1

**Hardware Target**: GPU (primary), CPU (fallback) - **PRODUCTION READY** ✅

## 🆕 Latest Enhancements

### ✅ **Device Information API** - NEW!
- **Device Status Endpoint**: `/api/device-info` for GPU/CPU status monitoring with CUDA detection
- **Backend Detection**: Intelligent device reporting based on vLLM/Triton/Ollama backends
- **CUDA Integration**: GPU availability detection with detailed device information
- **Service Integration**: Complete orchestration service compatibility for dynamic model loading

### ✅ **Enhanced Service Integration** - COMPLETED!
- **Orchestration Compatibility**: Full integration with orchestration service for device status reporting
- **Dynamic Backend Selection**: Automatic detection of optimal backend based on available hardware
- **Real-time Device Monitoring**: Live GPU/CPU status reporting with acceleration details
- **Graceful Fallback**: Intelligent fallback when GPU acceleration is unavailable

## Service Overview

The Translation Service is a comprehensive microservice that provides:
- **🚀 Llama 3.1-8B-Instruct**: Primary translation model with direct transformers integration
- **🔄 Multi-Backend Support**: Intelligent fallback chain - Llama → NLLB-200 → vLLM → Ollama → External APIs
- **⚡ Real-time Translation**: Sub-200ms latency with WebSocket streaming support
- **📊 Quality Scoring**: Confidence metrics, backend performance tracking, and session management
- **🔧 Configuration Sync**: Real-time config synchronization with orchestration service
- **🌍 50+ Languages**: Auto-detection and comprehensive language support
- **📱 Device Monitoring**: Real-time GPU/CPU status with acceleration details

## 🚀 Quick Start (Standalone)

Get the Translation service running independently in under 5 minutes:

```bash
# Navigate to translation service
cd modules/translation-service

# Option 1: Direct Llama 3.1 with transformers (Recommended)
./start-local.sh

# Option 2: Manual startup with Llama 3.1
conda activate vllm-cuda
export TRANSLATION_MODEL="./models/Llama-3.1-8B-Instruct"
python src/api_server.py

# Option 3: Docker with vLLM (GPU recommended)
docker-compose -f docker-compose-gpu.yml up --build -d

# Option 4: Docker simple setup (CPU fallback)
docker-compose -f docker-compose-simple.yml up --build -d

# Option 5: Direct Python with backend selection
python src/api_server.py --backend=vllm  # or ollama, triton

# Test the service
curl http://localhost:5003/api/health
curl -X POST http://localhost:5003/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_language": "Spanish"}'
```

**Service will be available at:**
- **REST API**: http://localhost:5003
- **Health Check**: http://localhost:5003/api/health
- **Device Info**: http://localhost:5003/api/device-info (NEW - GPU/CPU status with CUDA details)
- **Service Status**: http://localhost:5003/api/status (backend information and device details)
- **WebSocket**: ws://localhost:5003/translate/stream
- **Test Interface**: http://localhost:5003/test (development mode)

## 📋 Prerequisites

### System Requirements
- **CPU**: Modern x86_64 processor with AVX support
- **RAM**: 8-32GB depending on model size (16GB+ recommended for GPU models)
- **Storage**: 10GB for models and cache
- **GPU**: NVIDIA GPU with 8GB+ VRAM for optimal performance (GTX 1080 Ti / RTX 3060 or better)
- **CUDA**: CUDA 11.8+ with cuDNN for GPU acceleration
- **Networking**: Gigabit Ethernet for high-throughput translation

### Software Dependencies
```bash
# Python 3.9+
python --version

# Docker Desktop (for containerized deployment)
docker --version
docker-compose --version

# For local LLM inference (choose one)
# Option A: Ollama (CPU/GPU support, easiest setup)
curl -fsSL https://ollama.ai/install.sh | sh

# Option B: vLLM (GPU acceleration, high performance)
pip install vllm

# Option C: Triton (enterprise inference server)
# docker pull nvcr.io/nvidia/tritonserver:latest

# For GPU acceleration
# NVIDIA CUDA Toolkit 11.8+
# PyTorch with CUDA support
```

## 🛠️ Installation & Setup

### Method 1: Docker with vLLM (Recommended for GPU)

```bash
# 1. Navigate to translation service directory
cd modules/translation-service

# 2. Copy environment template
cp .env.example .env

# 3. Configure GPU settings
# Edit .env file:
# TRANSLATION_BACKEND=vllm
# VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
# CUDA_VISIBLE_DEVICES=0

# 4. Start with GPU acceleration
docker-compose -f docker-compose-gpu.yml up --build -d

# 5. Verify it's running
curl http://localhost:5003/api/health
```

### Method 2: Docker with Ollama (CPU-friendly)

```bash
# 1. Navigate to translation service directory
cd modules/translation-service

# 2. Simple setup with Ollama
docker-compose -f docker-compose-simple.yml up --build -d

# 3. Download a model (first time)
docker exec translation-service ollama pull llama3.1:8b

# 4. Test translation
curl -X POST http://localhost:5003/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "target_language": "French", "use_local": true}'
```

### Method 3: Local Development Setup

```bash
# 1. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install minimal dependencies for quick start
pip install -r requirements-minimal.txt

# 3. Install Ollama separately
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b

# 4. Set environment variables
export TRANSLATION_BACKEND=ollama
export OLLAMA_MODEL=llama3.1:8b
export PORT=5003

# 5. Start the service
python src/main.py
```

### Method 4: Full Local Setup with vLLM

```bash
# 1. Install all dependencies (requires GPU)
pip install -r requirements.txt

# 2. Set up vLLM environment
export TRANSLATION_BACKEND=vllm
export VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export CUDA_VISIBLE_DEVICES=0

# 3. Download model (first run)
python -c "from vllm import LLM; LLM('meta-llama/Llama-3.1-8B-Instruct')"

# 4. Start the service
python src/translation_service.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the translation-service directory:

```bash
# Service Configuration
PORT=5003
HOST=0.0.0.0
WORKERS=1

# Backend Selection
TRANSLATION_BACKEND=auto  # auto, ollama, vllm, openai
ENABLE_FALLBACK=true
FALLBACK_TIMEOUT=30

# Local Models (Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_AUTO_PULL=true

# Local Models (vLLM)
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.8

# External APIs (fallback)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key

# Performance Settings
MAX_TOKENS=1024
TEMPERATURE=0.3
BATCH_SIZE=4
MAX_CONCURRENT_REQUESTS=10

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Integration (optional)
REDIS_URL=redis://localhost:6379
```

### Backend Selection Priority

The service automatically selects the best available backend:

```bash
# Priority order (when TRANSLATION_BACKEND=auto):
1. vLLM (if GPU available and model loaded)
2. Ollama (if service running and model available)
3. External APIs (OpenAI/Google) as fallback
```

## 📊 Complete REST API Reference

### Health & Status
```bash
# Basic health check
GET /health

# API health check (orchestration compatible)
GET /api/health

# Device information (NEW - for orchestration integration)
GET /api/device-info

# Service status with comprehensive backend and device information
GET /api/status
```

### Core Translation API
```bash
# Translate text
POST /translate
Content-Type: application/json
Body: {
  "text": "Hello world",
  "target_language": "Spanish",
  "source_language": "English",  # optional, auto-detected if omitted
  "use_local": true,             # optional, prefer local models
  "quality_threshold": 0.8,      # optional, minimum quality score
  "context": "casual conversation" # optional, translation context
}

# Streaming translation (real-time)
POST /translate/stream
Content-Type: application/json
Body: {
  "session_id": "stream-123",
  "text": "Hello",
  "target_language": "es",
  "chunk_id": "chunk-001"  # optional, for ordering
}

# Language detection
POST /detect_language
Content-Type: application/json
Body: {"text": "Bonjour le monde"}

# Get supported languages
GET /languages
```

### Session Management
```bash
# Create translation session
POST /sessions
Content-Type: application/json
Body: {
  "source_lang": "en",
  "target_lang": "es",
  "backend": "ollama",        # optional, specific backend
  "quality_threshold": 0.8,   # optional
  "context": "technical docs" # optional
}

# Get session details
GET /sessions/{session_id}

# Update session configuration
PUT /sessions/{session_id}
Content-Type: application/json
Body: {"target_lang": "fr", "context": "medical"}

# Close session
DELETE /sessions/{session_id}
```

### WebSocket API (ws://localhost:5003/translate/stream)

#### Connection & Session Management
```javascript
// Connect to translation stream
const socket = new WebSocket('ws://localhost:5003/translate/stream');

// Join translation session
socket.send(JSON.stringify({
  type: 'join_session',
  session_id: 'translation-123',
  source_lang: 'en',
  target_lang: 'es'
}));

// Leave session
socket.send(JSON.stringify({
  type: 'leave_session',
  session_id: 'translation-123'
}));
```

#### Real-time Translation
```javascript
// Send text for translation
socket.send(JSON.stringify({
  type: 'translate_stream',
  session_id: 'translation-123',
  text: 'Hello world',
  chunk_id: 'chunk-001',     // optional
  priority: 'high'           // optional: high, normal, low
}));

// Receive translation result
socket.onmessage = function(event) {
  const result = JSON.parse(event.data);
  if (result.type === 'translation_result') {
    console.log('Translation:', result.translated_text);
    console.log('Quality Score:', result.quality_score);
    console.log('Backend Used:', result.backend);
  }
};
```

### Integration APIs

#### Whisper Integration
```bash
# Start Whisper integration
POST /api/whisper/start
Content-Type: application/json
Body: {
  "whisper_url": "http://localhost:5001",
  "session_id": "integrated-session",
  "target_language": "Spanish"
}

# Stop Whisper integration
POST /api/whisper/stop/{session_id}

# Process transcription for translation
POST /api/whisper/process
Content-Type: application/json
Body: {
  "session_id": "integrated-session",
  "transcription": "Hello world",
  "confidence": 0.95
}

# Get integration status
GET /api/whisper/status/{session_id}
```

### Advanced Configuration

#### Backend Priority (when TRANSLATION_BACKEND=auto)
1. vLLM (if GPU available and model loaded)
2. Ollama (if service running and model available)
3. External APIs (if API keys configured)
4. Error (no backends available)
```

## 🧪 Testing & Usage

### Health Check
```bash
# Basic health check
curl http://localhost:5003/api/health

# Device information (NEW)
curl http://localhost:5003/api/device-info

# Comprehensive status
curl http://localhost:5003/api/status
```

Expected responses:
```json
// Health Check
{
  "status": "healthy",
  "service": "translation",
  "backend": "vllm",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}

// Device Info (NEW)
{
  "device": "gpu",
  "device_type": "gpu",
  "acceleration": "cuda",
  "status": "healthy",
  "details": {
    "backend": "vllm",
    "gpu_available": true,
    "cuda_available": true,
    "cuda_version": "12.1",
    "device_count": 2,
    "current_device": 0,
    "device_name": "NVIDIA RTX 4090"
  },
  "service_info": {
    "version": "1.0.0",
    "backend": "vllm",
    "active_sessions": 3
  }
}

// Status (Enhanced)
{
  "status": "ok",
  "service": "translation",
  "backend": "vllm",
  "backends": {
    "vllm": {"status": "healthy", "gpu_memory": "6.2GB/24.0GB"},
    "ollama": {"status": "unavailable"},
    "triton": {"status": "unavailable"}
  },
  "active_sessions": 3,
  "device_info": {
    "device": "gpu",
    "acceleration": "cuda",
    "utilization": "65%"
  }
}
```

### Basic Translation
```bash
# Simple translation
curl -X POST http://localhost:5003/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "target_language": "Spanish",
    "use_local": true
  }'

# Response:
{
  "translated_text": "Hola, ¿cómo estás?",
  "source_language": "English",
  "target_language": "Spanish",
  "backend": "ollama",
  "confidence": 0.95,
  "processing_time": 1.23
}
```

### Batch Translation
```bash
curl -X POST http://localhost:5003/translate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "Goodbye", "Thank you"],
    "target_language": "French",
    "use_local": true
  }'
```

### WebSocket Streaming
```python
import asyncio
import websockets
import json

async def stream_translation():
    uri = "ws://localhost:5003/translate/stream"
    async with websockets.connect(uri) as websocket:
        # Send translation request
        await websocket.send(json.dumps({
            "text": "This is a long text that will be translated in real-time...",
            "target_language": "Spanish",
            "stream": True
        }))
        
        # Receive streaming translation
        async for message in websocket:
            result = json.loads(message)
            if result["type"] == "translation_chunk":
                print(result["partial_text"], end="", flush=True)
            elif result["type"] == "translation_complete":
                print(f"\nFinal: {result['final_text']}")
                break

asyncio.run(stream_translation())
```

### Language Detection
```bash
curl -X POST http://localhost:5003/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, comment allez-vous?"}'

# Response:
{
  "detected_language": "French",
  "confidence": 0.98,
  "alternatives": [
    {"language": "French", "confidence": 0.98},
    {"language": "Italian", "confidence": 0.02}
  ]
}
```

## 🔧 Development

### Running Tests
```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests (requires running service)
python -m pytest tests/integration/ -v

# Test specific backend
python -m pytest tests/integration/ -k test_ollama
python -m pytest tests/integration/ -k test_vllm

# Load tests
python tests/load/load_test.py --concurrent=10 --duration=60
```

### Development with Hot Reload
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Start with hot reload
python src/main.py --reload --debug

# Or with Docker development setup
docker-compose -f docker-compose.dev.yml up --build
```

### Testing Different Backends
```bash
# Test Ollama backend
TRANSLATION_BACKEND=ollama python src/main.py

# Test vLLM backend
TRANSLATION_BACKEND=vllm python src/main.py

# Test with fallback
TRANSLATION_BACKEND=auto ENABLE_FALLBACK=true python src/main.py
```

## 📊 Monitoring

### Metrics Endpoint
```bash
# Prometheus metrics
curl http://localhost:5003/metrics
```

Key metrics:
- `translation_requests_total` - Total requests by backend and language pair
- `translation_duration_seconds` - Processing latency by device type (GPU/CPU)
- `translation_errors_total` - Error rates by category and backend
- `translation_backend_switches_total` - Fallback frequency
- `translation_model_load_duration_seconds` - Model loading performance
- `translation_gpu_utilization` - GPU utilization percentage
- `translation_gpu_memory_usage` - GPU memory consumption
- `translation_quality_score` - Translation quality metrics
- `translation_tokens_per_second` - Throughput by device type

### Backend Status with Device Information
```bash
# Check backend availability with device details
curl http://localhost:5003/api/backends

# Enhanced response with device information:
{
  "backends": {
    "vllm": {
      "status": "available",
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "device": "gpu",
      "gpu_memory": "6.2GB/24.0GB",
      "acceleration": "cuda"
    },
    "ollama": {
      "status": "available",
      "model": "llama3.1:8b",
      "device": "cpu",
      "url": "http://localhost:11434"
    },
    "triton": {
      "status": "unavailable",
      "error": "Service not running"
    },
    "openai": {
      "status": "available",
      "model": "gpt-4",
      "device": "external"
    }
  },
  "current_backend": "vllm",
  "device_summary": {
    "primary_device": "gpu",
    "acceleration": "cuda",
    "fallback_available": true
  }
}
```

## 🚀 Deployment Options

### Standalone Production (GPU)
```bash
# Build production image with vLLM
docker build -f Dockerfile.vllm -t my-translation-service .

# Run with GPU
docker run -d \
  --name translation-service \
  --gpus all \
  -p 5003:5003 \
  -v $(pwd)/models:/app/models \
  -e VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  my-translation-service
```

### Standalone Production (CPU)
```bash
# Build lightweight image with Ollama
docker build -t my-translation-service-cpu .

# Run with Ollama
docker run -d \
  --name translation-service \
  -p 5003:5003 \
  -e TRANSLATION_BACKEND=ollama \
  -e OLLAMA_MODEL=llama3.1:8b \
  my-translation-service-cpu
```

### Integration with LiveTranslate
```bash
# As part of full system
cd ../../  # Back to project root
docker-compose -f docker-compose.comprehensive.yml up translation-service
```

## 🔍 Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check port availability
netstat -tulpn | grep 5003

# Check Docker logs
docker-compose logs translation-service

# Verify backend availability
curl http://localhost:11434/api/version  # Ollama
nvidia-smi  # GPU for vLLM
```

**Model loading errors:**
```bash
# Check disk space
df -h

# Check GPU memory and status
nvidia-smi
curl http://localhost:5003/api/device-info

# Download model manually
ollama pull llama3.1:8b  # For Ollama
python -c "from vllm import LLM; LLM('meta-llama/Llama-3.1-8B-Instruct')"  # For vLLM

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

**Poor translation quality:**
```bash
# Use larger model
OLLAMA_MODEL=llama3.1:70b
VLLM_MODEL=meta-llama/Llama-3.1-70B-Instruct

# Adjust temperature
TEMPERATURE=0.1  # More deterministic

# Enable context
curl -X POST http://localhost:5003/translate \
  -d '{"text": "...", "context": "This is a technical document about..."}'
```

## 🤝 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Service health check |
| GET | `/api/device-info` | GPU/CPU status with CUDA details |
| GET | `/api/status` | Detailed service status with device information |
| GET | `/api/backends` | Available backends with device details |
| POST | `/translate` | Single text translation |
| POST | `/translate/batch` | Batch translation |
| POST | `/detect-language` | Language detection |
| GET | `/languages` | Supported languages |
| POST | `/models/load` | Load specific model |

### WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `translate` | Client → Server | Start translation |
| `translation_chunk` | Server → Client | Partial translation |
| `translation_complete` | Server → Client | Final translation |
| `error` | Server → Client | Error notification |

## Overview

This module provides a unified translation interface that:
- **Preserves existing functionality** from `python/translation_server.py`
- **Adds local inference** via vLLM and Ollama
- **Provides automatic fallback** between local and external APIs
- **Supports independent testing** and deployment

## Architecture

```
Translation Service Module
├── Legacy Wrapper          # Existing translation_server.py functionality
├── Local Inference         # vLLM/Ollama integration via shared module
├── Unified Service         # Combines both with intelligent routing
└── REST API Interface      # Standard HTTP/WebSocket endpoints
```

## Features

### Existing Features (Preserved)
- ✅ External API integration (OpenAI, Google, etc.)
- ✅ Multiple language support
- ✅ Real-time translation
- ✅ WebSocket streaming
- ✅ Translation quality metrics

### New Features (Added)
- 🆕 Local vLLM inference (high-performance GPU)
- 🆕 Local Ollama inference (easy model management)
- 🆕 Automatic backend detection and fallback
- 🆕 Model-specific language optimization
- 🆕 Performance monitoring and metrics
- 🆕 Standalone module testing

## Usage

### Basic Usage
```python
from modules.translation_service import TranslationService

# Initialize with auto-detection
service = TranslationService()
await service.initialize()

# Translate text
result = await service.translate(
    text="Hello, world!",
    target_language="Spanish",
    source_language="English"  # Optional
)

print(result.translated_text)  # "¡Hola, mundo!"
print(result.backend)          # "ollama" or "vllm" or "openai"
print(result.confidence)       # 0.95
```

### Advanced Configuration
```python
# Force specific backend
service = TranslationService(
    prefer_local=True,
    fallback_to_external=True,
    local_backend="vllm",  # or "ollama"
    external_backend="openai"
)

# Streaming translation
async for chunk in service.translate_stream(text, target_lang):
    print(chunk.partial_text, end="", flush=True)
```

### REST API
```bash
# Start standalone service
cd modules/translation-service
python src/main.py --port 8010

# Translate via HTTP
curl -X POST http://localhost:8010/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "target_language": "Spanish",
    "use_local": true
  }'

# WebSocket streaming
wscat -c ws://localhost:8010/translate/stream
```

## Configuration

### Environment Variables
```bash
# Local inference settings
TRANSLATION_BACKEND=auto              # auto, local, external
TRANSLATION_LOCAL_BACKEND=ollama      # vllm, ollama
TRANSLATION_EXTERNAL_BACKEND=openai   # openai, google, azure

# Local model settings
TRANSLATION_VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
TRANSLATION_OLLAMA_MODEL=llama3.1:8b

# Fallback settings
TRANSLATION_ENABLE_FALLBACK=true
TRANSLATION_FALLBACK_TIMEOUT=30s

# Performance settings
TRANSLATION_MAX_TOKENS=1024
TRANSLATION_TEMPERATURE=0.3          # Lower for more consistent translation
TRANSLATION_BATCH_SIZE=4
```

### Model Configuration
```yaml
# config/models.yml
models:
  translation:
    local:
      vllm:
        model: "meta-llama/Llama-3.1-8B-Instruct"
        languages: ["en", "es", "fr", "de", "zh", "ja"]
        specialization: "multilingual"
      ollama:
        model: "llama3.1:8b"
        languages: ["en", "es", "fr", "de"]
        auto_pull: true
    external:
      openai:
        model: "gpt-4"
        api_key_env: "OPENAI_API_KEY"
      google:
        model: "gemini-pro"
        api_key_env: "GOOGLE_API_KEY"
```

## Integration with Existing Code

### Legacy Compatibility
```python
# This continues to work unchanged
from python.translation_server import TranslationServer
from python.translation_client import TranslationClient

# But now you can also use the modular version
from modules.translation_service import TranslationService
```

### Gradual Migration
1. **Phase 1**: Use legacy wrapper (no changes needed)
2. **Phase 2**: Enable local inference alongside existing
3. **Phase 3**: Gradually shift traffic to local inference
4. **Phase 4**: Optional - deprecate external APIs

## Testing

### Unit Tests
```bash
cd modules/translation-service
python -m pytest tests/unit/
```

### Integration Tests
```bash
# Test with different backends
python -m pytest tests/integration/ -k test_vllm
python -m pytest tests/integration/ -k test_ollama
python -m pytest tests/integration/ -k test_fallback
```

### Performance Tests
```bash
# Load testing
python tests/performance/load_test.py --concurrent=10 --duration=60s

# Language accuracy testing
python tests/accuracy/test_languages.py --languages=es,fr,de,zh
```

### Standalone Testing
```bash
# Test module independently
docker-compose up --build
python tests/standalone/test_service.py
```

## Deployment Options

### Standalone Deployment
```bash
# Deploy as independent service
cd modules/translation-service
docker build -t livetranslate/translation-service .
docker run -p 8010:8010 livetranslate/translation-service
```

### Integrated Deployment
```bash
# Deploy with other modules
python scripts/deploy.py --modules translation-service,whisper-service
```

### Development Mode
```bash
# Hot reload for development
python src/main.py --dev --watch
```

## API Reference

### REST Endpoints

#### POST /translate
Translate text using the best available backend.

**Request:**
```json
{
  "text": "Hello world",
  "target_language": "Spanish",
  "source_language": "English",
  "use_local": true,
  "stream": false
}
```

**Response:**
```json
{
  "translated_text": "Hola mundo",
  "source_language": "English",
  "target_language": "Spanish", 
  "backend": "ollama",
  "model": "llama3.1:8b",
  "confidence": 0.95,
  "latency_ms": 145,
  "tokens": {
    "input": 2,
    "output": 2
  }
}
```

#### WebSocket /translate/stream
Real-time streaming translation.

#### GET /health
Service health check.

#### GET /models
List available translation models.

#### GET /metrics
Prometheus-compatible metrics.

## Performance Characteristics

### Local Inference Benefits
- **Privacy**: No data sent to external APIs
- **Speed**: Reduced latency (no network calls)
- **Cost**: No per-token charges
- **Reliability**: No API rate limits or outages

### Benchmarks (Approximate)
| Backend | Latency | Throughput | Cost | Privacy |
|---------|---------|------------|------|---------|
| vLLM    | 50-150ms | 1000+ tok/s | Free | 100% |
| Ollama  | 100-300ms | 100+ tok/s | Free | 100% |
| OpenAI  | 200-500ms | Variable | $0.03/1k | No |
| Google  | 150-400ms | Variable | $0.02/1k | No |

## Troubleshooting

### Common Issues

**Local models not loading:**
```bash
# Check model availability
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8000/v1/models  # vLLM

# Pull missing models
ollama pull llama3.1:8b
```

**Poor translation quality:**
```bash
# Try different models or adjust temperature
export TRANSLATION_TEMPERATURE=0.1  # More deterministic
export TRANSLATION_OLLAMA_MODEL=llama3.1:70b  # Larger model
```

**Fallback not working:**
```bash
# Check external API keys
echo $OPENAI_API_KEY
python tests/debug/test_fallback.py
```

## Monitoring

### Key Metrics
- Translation latency by backend
- Translation quality scores
- Backend availability and fallback rates
- Token usage and costs
- Error rates by language pair

### Dashboards
- Grafana dashboard: `dashboards/translation-service.json`
- Log analysis: Available in centralized logging

This module provides a smooth transition from external APIs to local inference while preserving all existing functionality! 