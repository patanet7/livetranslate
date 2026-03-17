# LiveTranslate

**Real-time Speech-to-Text Transcription and Translation System with AI Acceleration**

LiveTranslate is a comprehensive, production-ready system for real-time audio transcription and translation. Built as a microservices architecture with enterprise-grade WebSocket infrastructure, it provides NPU/GPU acceleration, speaker diarization, multi-language translation, distributed deployment capabilities, and Google Meet bot integration.

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![AI Acceleration](https://img.shields.io/badge/AI-NPU%2FGPU%20Ready-green)
![Architecture](https://img.shields.io/badge/Architecture-Microservices-purple)
![WebSocket](https://img.shields.io/badge/WebSocket-Enterprise%20Grade-orange)

## 🏗️ Architecture Overview

### Hardware-Optimized Service Architecture

The system is organized into **3 core services + external LLM** optimized for specific hardware acceleration:

1. **Transcription Service** (Port 5001) - Pluggable backends ✅
   - Real-time speech-to-text transcription (vLLM-MLX on Apple Silicon, faster-whisper on GPU)
   - Silero Voice Activity Detection
   - VRAM budgeting and model management
   - Multi-format audio processing (WAV, MP3, WebM, OGG, MP4)
   - Enterprise WebSocket infrastructure with connection pooling

2. **Orchestration Service** (Port 3000) - **[CPU OPTIMIZED]** ✅
   - FastAPI backend with async/await patterns
   - Enterprise WebSocket connection management (10,000+ concurrent)
   - Service health monitoring and auto-recovery
   - LLM-based translation via external service (vLLM-MLX or Ollama)
   - **Google Meet Bot Management**: Complete bot lifecycle, official API integration, PostgreSQL persistence
   - **Virtual Webcam System**: Professional translation overlays with speaker attribution
   - **Browser Audio Capture**: Specialized Google Meet audio extraction with multiple fallback methods
   - **Time Correlation Engine**: Advanced timeline matching between Google Meet captions and internal transcriptions
   - **Configuration Synchronization**: Real-time config sync between all services

3. **Dashboard Service** (Port 5173) - **[BROWSER OPTIMIZED]** ✅
   - **SvelteKit + Svelte 5 runes**: Modern reactive UI framework
   - **Loopback Page** (`/loopback`): Real-time audio capture, transcription, and translation
   - **Real-time Audio Testing**: Audio capture (microphone/system/both), visualization, live streaming
   - **Translation Testing Interface**: Live translation testing with multiple target languages
   - **Bot Management Dashboard**: Complete Google Meet bot lifecycle management
   - **Configuration Management**: Real-time settings sync with all backend services
   - **Analytics Dashboard**: Session statistics, performance metrics, and service monitoring

4. **External LLM** - Translation inference ✅
   - Local: vLLM-MLX (`:8006`, `mlx-community/Qwen3-4B-4bit`)
   - Remote: Ollama (`:11434`, `qwen3.5:7b`)

### Supporting Infrastructure
- **Google Meet Bot System**: Complete bot integration with official Google Meet API, virtual webcam generation
- **Virtual Webcam System**: Professional translation overlays with speaker attribution and real-time display
- **Browser Audio Capture**: Specialized audio extraction from Google Meet sessions with multiple fallback methods
- **Time Correlation Engine**: Advanced timeline matching between external captions and internal transcriptions
- **Database Schema**: Comprehensive PostgreSQL schema for bot sessions, audio files, and correlation data
- **Configuration Management**: Unified settings with real-time synchronization across all services
- **Monitoring Stack**: Prometheus + Grafana + AlertManager + Loki for enterprise observability

## 📚 Documentation

Documentation is organized around current operational usage and C4 architecture levels:

- **[Documentation Hub](./docs/README.md)** - Canonical navigation for active docs
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Full system topology and design
- **[Quick Start Guide](./docs/guides/quick-start.md)** - Local startup workflows
- **[Database Setup Guide](./docs/guides/database-setup.md)** - PostgreSQL/Redis bootstrap
- **[C4 Level 1: Context](./docs/01-context/README.md)**
- **[C4 Level 2: Containers](./docs/02-containers/README.md)**
- **[C4 Level 3: Components](./docs/03-components/README.md)**

**Note on Legacy Services**: `modules/frontend-service/`, `modules/translation-service/`, and `modules/speaker-service/` have been archived. See [Service Architecture](#hardware-optimized-service-architecture) for active services.

## 🚀 Quick Start

### Prerequisites
- **Python 3.12-3.13** (3.14+ not supported)
- **UV** (dependency management, installed with Python)
- **Node.js 18+** (frontend tooling)
- **8GB RAM minimum** (16GB+ recommended for model inference)
- **10GB storage** for models and data
- **Optional**: NVIDIA GPU (CUDA) or Apple Silicon for acceleration
- **Optional**: PostgreSQL database for bot session persistence

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd livetranslate

# Install dependencies (CRITICAL: --all-packages required for workspace packages)
uv sync --all-packages --group dev
```

### 2. Run Services
```bash
# Transcription Service (GPU/Apple Silicon)
uv run python modules/transcription-service/src/main.py

# Orchestration Service
uv run python modules/orchestration-service/src/main_fastapi.py

# Dashboard Service (SvelteKit)
cd modules/dashboard-service && npm install && npm run dev

# External LLM (choose one):
# Local: vLLM-MLX (Apple Silicon)
python -m vllm.entrypoints.openai.api_server --model mlx-community/Qwen3-4B-4bit --port 8006

# Or remote: Ollama (GPU)
ollama run qwen3.5:7b
```

Access services:
- **Dashboard**: http://localhost:5173 (dev) or http://localhost:3000 (prod)
- **Orchestration API**: http://localhost:3000
- **Loopback**: http://localhost:5173/loopback

## 🛠️ Developer Workflow

- Install Git hooks once:
  ```bash
  pre-commit install
  ```
  Run linting/formatting on demand with `ruff check --fix .` and `ruff format .`

- Run tests:
  ```bash
  just test-orchestration
  just test-transcription
  just coverage-backend
  ```

- Run Playwright E2E tests (requires `just dev` running):
  ```bash
  just test-playwright
  just test-e2e-playback
  ```

### 3. First Usage
1. Open dashboard at http://localhost:5173
2. Navigate to **Loopback** page for live transcription/translation
3. Select audio source (microphone, system, or both)
4. Choose transcription model and target languages
5. View real-time captions and translations
6. Use **Bot Management** dashboard for Google Meet integration

## 📡 Service Endpoints & Ports

| Service | Port | Purpose | Health Check |
|---------|------|---------|--------------|
| **Dashboard** | 5173 (dev) / 3000 (prod) | SvelteKit web interface | N/A |
| **Orchestration** | 3000 | Backend API, WebSocket hub | `/api/health` |
| **Transcription** | 5001 | Speech-to-text service | `/health` |
| **vLLM-MLX (STT)** | 8005 | Whisper inference (Apple Silicon) | `/v1/models` |
| **vLLM-MLX (LLM)** | 8006 | Qwen3-4B translation inference | `/v1/models` |
| **Ollama** | 11434 | Alternative LLM backend | `/api/tags` |
| **Monitoring** | 3001 | Grafana dashboards | `/api/health` |
| **Prometheus** | 9090 | Metrics collection | `/-/healthy` |

### WebSocket Endpoints
| Service | WebSocket URL | Purpose |
|---------|---------------|---------|
| **Dashboard** | ws://localhost:5173/ws | Audio streaming, real-time updates |
| **Orchestration** | ws://localhost:3000/ws | System coordination & updates |

## 🔧 Starting Individual Services

Each service can be started independently for distributed deployment or development:

### Transcription Service (Port 5001)
```bash
uv run python modules/transcription-service/src/main.py

# Test the service
curl http://localhost:5001/health
```

### Orchestration Service (Port 3000)
```bash
uv run python modules/orchestration-service/src/main_fastapi.py

# Test the service
curl http://localhost:3000/api/health
```

### Dashboard Service
```bash
cd modules/dashboard-service

# Install and run
npm install
npm run dev

# Access at http://localhost:5173
```

### External LLM Services

**vLLM-MLX (Apple Silicon)**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model mlx-community/Qwen3-4B-4bit \
  --port 8006

# Test: curl http://localhost:8006/v1/models
```

**Ollama (GPU)**
```bash
ollama run qwen3.5:7b

# Or specify port: OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Legacy Services (Archived)
The following services have been archived and are no longer active:
- `modules/frontend-service/` (React) — replaced by dashboard-service
- `modules/translation-service/` — translation now in orchestration via LLM
- `modules/speaker-service/` — diarization functionality integrated elsewhere
- `modules/whisper-service/` — renamed to transcription-service

## 📊 Key API Endpoints

See `/docs` on running services for interactive API documentation.

### Orchestration Service (Port 3000)
```bash
# Health check
GET /api/health

# Shared contracts (pydantic v2)
GET /api/models/{model_name}

# WebSocket audio streaming
WS /ws
```

### Transcription Service (Port 5001)
```bash
# Health check
GET /health

# List available models
GET /models

# Transcribe audio file
POST /transcribe
Content-Type: multipart/form-data
Body: file=audio.wav
```

### External LLM (OpenAI-compatible API)
```bash
# List models
GET /v1/models

# Create chat completion (translation)
POST /v1/chat/completions
Content-Type: application/json
Body: {
  "model": "qwen3.5:7b",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.7
}
```

## 🤖 Google Meet Bot Management System

### Enterprise-Grade Bot Management
- **10+ concurrent bots** with automatic queuing and health monitoring
- **Official Google Meet API Integration** with OAuth 2.0 authentication
- **Comprehensive Database Integration** with PostgreSQL for session persistence
- **Real-time Bot Lifecycle Management** with automatic recovery strategies
- **Time-coded Data Storage** for audio, transcripts, translations, and correlations

### Bot Features
- **Audio Capture**: Real-time audio streaming from Google Meet sessions
- **Caption Processing**: Google Meet caption extraction and processing
- **Time Correlation**: Advanced timing correlation between external and internal sources
- **Speaker Attribution**: Complete speaker tracking throughout meeting timeline
- **Analytics Dashboard**: Session statistics, quality metrics, and performance tracking

### Usage
```bash
# Setup database schema (PostgreSQL required)
psql -U postgres -d livetranslate -f scripts/bot-sessions-schema.sql

# Configure Google Meet API (optional - create credentials.json from Google Cloud Console)
export GOOGLE_MEET_CREDENTIALS_PATH="/path/to/credentials.json"

# Start orchestration service with bot management
cd modules/orchestration-service && ./start-backend.ps1

# Access bot management dashboard at http://localhost:3000
```

## ⚙️ Configuration Synchronization System

### Real-time Configuration Management
- **Bidirectional Synchronization** between Frontend ↔ Orchestration ↔ Whisper service
- **Live Updates** with hot-reloadable configuration changes (no service restarts)
- **Compatibility Validation** with automatic detection of configuration differences
- **Configuration Presets** for different deployment scenarios (performance, accuracy, real-time)
- **Professional UI** with 7 comprehensive settings tabs

### Configuration Features
- **Audio Processing Settings**: VAD, enhancement, chunking parameters
- **Speaker Correlation**: Time correlation and speaker attribution settings
- **Translation Settings**: Model selection, quality thresholds, language preferences
- **Bot Management**: Meeting bot lifecycle and API configuration
- **System Settings**: Service coordination and monitoring parameters

### Presets Available
- **Exact Whisper Match**: Match current whisper service settings exactly
- **Optimized Performance**: Better performance with minimal overlap
- **High Accuracy**: Higher accuracy with more overlap and longer chunks
- **Real-time Optimized**: Minimal latency for real-time processing

## 🎯 Key Features & Architecture

### ✅ Production-Ready Components
- **✅ WebSocket Infrastructure** - Enterprise-grade real-time communication (90% complete)
  - Connection management with pooling (1000 object capacity)
  - Advanced error handling (20+ error categories)
  - Heartbeat monitoring with RTT tracking
  - Session persistence and reconnection (30-min timeout)
  - Message routing with pub-sub capabilities
  - Performance optimization with async processing
- **✅ Docker Infrastructure** - Complete containerized deployment
- **✅ Web Interface** - Modern, responsive UI with performance dashboard
- **✅ Audio Processing** - Rolling buffer with Voice Activity Detection
  - Fixed audio resampling (48kHz → 16kHz) with librosa fallback
  - Voice-specific processing with tunable parameters
  - Step-by-step debugging with pause capability
- **✅ Monitoring Stack** - Comprehensive metrics and dashboards

### 🚀 AI Acceleration Support
- **NPU Acceleration**: Intel NPU support via OpenVINO for Whisper
- **GPU Acceleration**: NVIDIA GPU support with CUDA for translation
- **CPU Fallback**: Automatic fallback for systems without acceleration
- **Model Optimization**: Quantized models for faster inference

### 🎙️ Audio Processing Pipeline
- **Real-time Capture**: System audio, microphone input
- **Voice Activity Detection**: Smart silence detection
- **Audio Enhancement**: Noise reduction, normalization
- **Multiple Formats**: WAV, MP3, FLAC, OGG support
- **Streaming**: Chunked processing with overlap

### 🧠 AI-Powered Features
- **Speech Recognition**: OpenAI Whisper with multiple model sizes
- **Speaker Diarization**: Multi-speaker identification and tracking with enhanced attribution
- **Translation**: Local LLM-based translation (vLLM/Ollama)
- **Language Detection**: Automatic source language identification
- **🆕 Virtual Webcam Generation**: Real-time translation overlays with professional speaker attribution
- **🆕 Time Correlation**: Advanced timeline matching between Google Meet captions and internal transcriptions

### 🎥 Google Meet Bot Integration
- **Complete Bot Lifecycle**: Spawn, monitor, and terminate Google Meet bots
- **Browser Audio Capture**: Specialized audio extraction from Google Meet sessions
- **Virtual Webcam System**: Professional translation overlays with speaker attribution
  - **Dual Content Display**: Original transcriptions (🎤) with actual Whisper confidence and translations (🌐) with real confidence scores
  - **Speaker Attribution**: Enhanced display with diarization info (e.g., "John Doe (SPEAKER_00)")
  - **Professional Layout**: Honest confidence indicators, language direction, timestamps
  - **Multiple Themes**: Dark, light, high contrast, minimal, corporate
  - **Real-time Streaming**: 30fps frame generation with configurable duration
- **Time Correlation Engine**: Match Google Meet captions with high-quality internal transcriptions
- **Database Integration**: Complete session tracking with PostgreSQL persistence

### 🌐 Enterprise Features
- **WebSocket Infrastructure**: Enterprise-grade real-time communication
  - Connection pooling and management
  - Automatic reconnection and session recovery
  - Message buffering and retry logic
  - Performance monitoring and optimization
- **Distributed Deployment**: Services can run on separate machines
- **Health Monitoring**: Comprehensive service health checks
- **Session Management**: Persistent transcription sessions
- **Export Capabilities**: CSV, JSON, SRT format exports
- **Configuration Management**: Dynamic service configuration

## 🏗️ Advanced WebSocket Architecture

### Enterprise-Grade Features
```
┌─────────────────────────────────────────────────────────────┐
│                WebSocket Infrastructure                      │
├─────────────────────────────────────────────────────────────┤
│  Connection Management                                      │
│  ├─ Connection Pool (1000 capacity)                        │
│  ├─ Weak Reference Tracking                                │
│  ├─ Automatic Cleanup                                      │
│  └─ Pool Efficiency Monitoring                             │
├─────────────────────────────────────────────────────────────┤
│  Error Handling Framework                                   │
│  ├─ 20+ Error Categories                                   │
│  ├─ Automatic Recovery Strategies                          │
│  ├─ Error Rate Limiting                                    │
│  └─ Comprehensive Error Statistics                         │
├─────────────────────────────────────────────────────────────┤
│  Heartbeat System                                           │
│  ├─ RTT Tracking                                           │
│  ├─ Connection Quality Monitoring                          │
│  ├─ Adaptive Intervals                                     │
│  └─ Health State Management                                │
├─────────────────────────────────────────────────────────────┤
│  Session Persistence                                        │
│  ├─ 30-minute Session Timeout                              │
│  ├─ Message Buffering (100 messages max)                   │
│  ├─ Priority-based Queuing                                 │
│  ├─ Automatic State Recovery                               │
│  └─ Zero-message-loss Design                               │
├─────────────────────────────────────────────────────────────┤
│  Performance Optimization                                   │
│  ├─ Async Audio Processing                                 │
│  ├─ Message Queue Batching                                 │
│  ├─ Thread Pool Management                                 │
│  └─ Performance Metrics Dashboard                          │
└─────────────────────────────────────────────────────────────┘
```

### Performance Monitoring
Access real-time performance metrics at:
- **REST API**: `GET /performance` on any service
- **Web Dashboard**: http://localhost:3000 (Performance/Monitoring tab)

**Available Metrics:**
- Audio processing times
- Transcription latency
- WebSocket message throughput
- Queue sizes and efficiency
- Memory and CPU usage
- Connection pool statistics

## 🔧 Troubleshooting

### Audio Issues

#### Low Audio Levels / Poor Transcription Quality
If you're experiencing poor transcription results or low audio levels (especially with loopback devices):

**Problem**: Browser audio processing features can severely attenuate loopback audio
**Solution**: Disable audio processing in MediaRecorder constraints:

```javascript
// In audio capture configuration
const constraints = {
    audio: {
        echoCancellation: false,    // Was causing loopback attenuation
        noiseSuppression: false,    // Was treating loopback as noise
        autoGainControl: false,     // Was reducing gain levels
        // ... other settings
    }
};
```

**Fixed in**: `modules/orchestration-service/static/js/audio.js`

#### WebM/Opus Audio Processing
- Noise reduction can remove loopback audio content completely
- Disable aggressive noise reduction for loopback sources
- Backend RMS threshold adjusted from 0.001 to 0.0001 for better sensitivity

#### NPU/GPU Detection Issues
- Check device drivers and OpenVINO installation
- Verify `OPENVINO_DEVICE` environment variable
- Use CPU fallback if hardware acceleration fails

### Common Issues ✅ **RECENTLY RESOLVED**

#### ✅ **Fixed: 422 Validation Errors**
**Problem**: Frontend Meeting Test Dashboard experiencing 422 errors on audio upload
**Root Cause**: Improper FastAPI dependency injection in orchestration service
**Solution**: Fixed `upload_audio_file()` endpoint with proper `Depends()` wrapper
**Status**: ✅ **RESOLVED** - Audio streaming now fully operational

#### ✅ **Fixed: Model Selection Issues**  
**Problem**: Frontend model dropdown showing "base" but services expecting "whisper-base"
**Root Cause**: Inconsistent model naming in fallback mechanisms
**Solution**: Standardized all model names to use "whisper-" prefix
**Status**: ✅ **RESOLVED** - Dynamic model loading working correctly

#### Remaining Common Issues
- **Port conflicts**: Ensure ports 3000, 5001, 8005, 8006, 11434 are available
- **Python version**: Ensure Python >=3.12,<3.14 (3.14+ not supported due to grpcio/onnxruntime)
- **Model downloads**: First run may take time to download models
- **GPU memory**: Allocate sufficient VRAM for transcription and translation models

#### Dependency Management
- **Always use UV**: `uv sync --all-packages --group dev` (not pip, PDM, or Poetry)
- **--all-packages required**: Installs workspace packages (livetranslate-common, etc.)
- **Pytest**: `uv run pytest ...` (not `python -m pytest`)
- **Code formatting**: `ruff check --fix .` and `ruff format .` (not Black/isort/flake8)

## 🚀 Planning

Current implementation planning should be tracked in active issues/PRs and the docs hub.
Historical planning snapshots remain in:

- [`docs/archive/planning/`](docs/archive/planning/)

## 📝 License

This project is licensed under the terms of the included license file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run relevant checks: `just test-all` (or module-specific test commands)
5. Submit a pull request

## 🆘 Support

### Documentation
- [Documentation Hub](docs/README.md)
- [Modules Index](modules/README.md)
- [Quick Start](docs/guides/quick-start.md)

### Troubleshooting
- [Debugging Guide](docs/debugging.md)
- [Translation Testing](docs/guides/translation-testing.md)
- [Database Setup](docs/guides/database-setup.md)

### Community
- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Documentation: Project Wiki

---

**LiveTranslate** - Breaking down language barriers with AI-powered real-time translation. 
