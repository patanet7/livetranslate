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

The system is organized into **4 core services** optimized for specific hardware acceleration:

1. **Whisper Service** (Port 5001) - **[NPU OPTIMIZED]** ✅
   - Real-time speech-to-text transcription (OpenVINO optimized Whisper)
   - Advanced speaker diarization with multiple embedding methods
   - Multi-format audio processing (WAV, MP3, WebM, OGG, MP4)
   - Voice Activity Detection (WebRTC + Silero)
   - Enterprise WebSocket infrastructure with connection pooling

2. **Translation Service** (Port 5003) - **[GPU OPTIMIZED]** ✅
   - **Llama 3.1-8B-Instruct**: Primary translation model with direct transformers integration
   - **Multi-backend Support**: vLLM (with compatibility fallback), NLLB-200, Ollama, Triton
   - **Real-time Translation Streaming**: WebSocket-based streaming with sub-200ms latency
   - **Quality Scoring**: Confidence metrics and backend performance tracking
   - **Intelligent Fallback Chain**: Llama → NLLB → Ollama → External APIs

3. **Orchestration Service** (Port 3000) - **[CPU OPTIMIZED]** ✅
   - FastAPI backend with async/await patterns
   - Enterprise WebSocket connection management (10,000+ concurrent)
   - Service health monitoring and auto-recovery
   - API Gateway with load balancing and circuit breaking
   - **Google Meet Bot Management**: Complete bot lifecycle, official API integration, PostgreSQL persistence
   - **Virtual Webcam System**: Professional translation overlays with speaker attribution
   - **Browser Audio Capture**: Specialized Google Meet audio extraction with multiple fallback methods
   - **Time Correlation Engine**: Advanced timeline matching between Google Meet captions and internal transcriptions
   - **Configuration Synchronization**: Real-time config sync between all services

4. **Frontend Service** (Port 5173) - **[BROWSER OPTIMIZED]** ✅
   - **Modern React 18**: TypeScript + Material-UI + Vite with Redux Toolkit state management
   - **Real-time Audio Testing**: Comprehensive audio capture, processing, and visualization
   - **Translation Testing Interface**: Live translation testing with multiple target languages
   - **Bot Management Dashboard**: Complete Google Meet bot lifecycle management
   - **Configuration Management**: Real-time settings sync with all backend services
   - **Analytics Dashboard**: Session statistics, performance metrics, and service monitoring

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
- **[Quick Start Guide](./docs/guides/quick-start.md)** - Local startup workflows
- **[Database Setup Guide](./docs/guides/database-setup.md)** - PostgreSQL/Redis bootstrap
- **[Translation Testing Guide](./docs/guides/translation-testing.md)** - End-to-end and integration checks
- **[C4 Level 1: Context](./docs/01-context/README.md)**
- **[C4 Level 2: Containers](./docs/02-containers/README.md)**
- **[C4 Level 3: Components](./docs/03-components/README.md)**
- **[Documentation Maintenance Standards](./docs/MAINTENANCE.md)** - Keep docs current and clean

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** (Windows/Mac/Linux) - Version 20.10+ recommended
- **Poetry 1.8+** (backend dependency management)
- **Node.js 18+ with pnpm 8+** (frontend tooling; run `corepack enable` once)
- **8GB RAM minimum** (16GB+ recommended for GPU acceleration)
- **10GB storage** for models and data
- **Optional**: NVIDIA GPU (CUDA 11.8+) or Intel NPU for acceleration
- **Optional**: PostgreSQL database for bot session persistence

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd livetranslate

# Copy environment template (recommended)
cp env.template .env
# Edit .env file to customize configuration
```

### 2. Choose Your Deployment Method

#### Option A: Docker Compose (Hot Reload + Local Inference)
```bash
# Bootstrap environment variables (creates .env.local if missing)
just bootstrap-env

# Start orchestration, frontend, redis, whisper, and translation services
just compose-up

# Tail logs or stop services
just compose-logs
just compose-down
```

By default this launches the real Whisper and Translation containers on CPU (no GPU required) alongside orchestration and the frontend dev server. To include Postgres or other profiles:

```bash
just compose-up profiles="core,inference,infra"
# include workers for config sync processing
# just compose-up profiles="core,inference,infra,workers"

```

Need quick mocks instead of the full inference stack? Start the mock profile and point orchestration at it by updating `.env.local`:

```bash
COMPOSE_PROFILES="core,mock" just compose-up
# .env.local
AUDIO_SERVICE_URL=http://whisper-mock:5001
TRANSLATION_SERVICE_URL=http://translation-mock:5003
```

#### Option B: Development Environment (Recommended)
> Requires Poetry and Node.js 18+. The script installs backend deps via `poetry install --with dev,audio` and ensures pnpm is available.
```bash
# Start complete development environment
./start-development.ps1

# Access services:
# Frontend: http://localhost:5173
# Backend:  http://localhost:3000
# API Docs: http://localhost:3000/docs
```

#### Option C: Individual Service Development
```bash
# Backend Service (FastAPI + Poetry)
cd modules/orchestration-service
poetry install --with dev,audio
poetry run uvicorn src.main:app --host 0.0.0.0 --port 3000 --reload

# Frontend Service (React + pnpm)
cd modules/frontend-service
pnpm install
pnpm dev --host 0.0.0.0 --port 5173

# Optional: spin up inference services individually
cd modules/whisper-service && docker compose up --build
cd modules/translation-service && docker compose -f docker-compose-simple.yml up --build
```

#### Option D: Full System (All Services)
```bash
# Start all services with comprehensive setup
docker-compose -f docker-compose.comprehensive.yml up -d

# Check service health
docker-compose -f docker-compose.comprehensive.yml ps
```

#### Option E: Core Services Only (Lightweight)
```bash
# Start essential services only
docker-compose -f docker-compose.comprehensive.yml up -d frontend whisper translation whisper-redis translation-redis

# Verify services are running
curl http://localhost:3000/api/health
curl http://localhost:5001/health
curl http://localhost:5003/api/health
```

## 🛠️ Developer Workflow

- Install Git hooks once:
  ```bash
  pre-commit install
  ```
  Run linting/formatting on demand with `just pre-commit-run`.
- Queue publishing is enabled by default; override via `.env.local`:
  ```bash
  EVENT_BUS_ENABLED=false
  EVENT_BUS_REDIS_URL=redis://localhost:6379/0
  ```
- Switch configuration sync to worker mode (default `api`):
  ```bash
  CONFIG_SYNC_MODE=worker
  ```


### 3. Access the System
- **Web Interface**: http://localhost:5173 (Development) or http://localhost:3000 (Production)
- **Backend API**: http://localhost:3000
- **API Documentation**: http://localhost:3000/docs
- **Performance Dashboard**: Built into web interface (Performance/Monitoring tab)

### 4. First Usage ✅ **NOW FULLY OPERATIONAL**
1. Open the web interface in your browser
2. **Meeting Test Dashboard**: Navigate to real-time streaming interface ✅ **WORKING**
3. **Start Streaming**: Click "Start Streaming" with configurable chunk sizes ✅ **FIXED 422 ERRORS**
4. **Model Selection**: Choose from dynamically loaded Whisper models ✅ **FIXED NAMING**
5. **Multi-language Translation**: Select target languages for real-time translation
6. **View Results**: See real-time transcription and translation results as they arrive
7. **Device Monitoring**: Monitor NPU/GPU/CPU status across all services  
8. **Bot Management**: Use Bot Management dashboard for Google Meet integration

## 📡 Service Endpoints & Ports

| Service | Port | Primary Endpoint | Purpose | Health Check |
|---------|------|------------------|---------|--------------|
| **Frontend** | 5173 | http://localhost:5173 | Modern React web interface | N/A |
| **Orchestration** | 3000 | http://localhost:3000 | Backend API & bot management | `/api/health` |
| **Whisper** | 5001 | http://localhost:5001 | Speech-to-text transcription | `/health` |
| **Translation** | 5003 | http://localhost:5003 | Multi-language translation | `/api/health` |
| **Monitoring** | 3001 | http://localhost:3001 | Grafana dashboards | `/api/health` |
| **Prometheus** | 9090 | http://localhost:9090 | Metrics collection | `/-/healthy` |

### WebSocket Endpoints
| Service | WebSocket URL | Purpose |
|---------|---------------|---------|
| **Whisper** | ws://localhost:5001/ws | Real-time transcription streaming |
| **Translation** | ws://localhost:5003/translate/stream | Real-time translation streaming |
| **Orchestration** | ws://localhost:3000/ws | System coordination & updates |

## 🔧 Starting Individual Services

Each service can be started independently for distributed deployment or development:

### Whisper Service (Speech-to-Text)
```bash
cd modules/whisper-service

# Docker deployment
docker-compose up -d

# Local development
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/api_server.py

# NPU acceleration (Intel systems)
docker-compose -f docker-compose.npu.yml up -d

# Test the service
curl http://localhost:5001/health
curl -X POST -F "file=@test.wav" http://localhost:5001/transcribe/whisper-small.en
```

### Translation Service
```bash
cd modules/translation-service

# Direct Llama 3.1 with transformers (Recommended)
./start-local.sh

# Docker with GPU acceleration
docker-compose -f docker-compose-gpu.yml up -d

# Docker simple setup (CPU)
docker-compose -f docker-compose-simple.yml up -d

# Local development with conda environment
conda activate vllm-cuda
export TRANSLATION_MODEL="./models/Llama-3.1-8B-Instruct"
python src/api_server.py

# Test the service
curl http://localhost:5003/api/health
curl -X POST http://localhost:5003/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_language": "Spanish"}'
```

### Speaker Service (Diarization)
```bash
cd modules/speaker-service

# Docker deployment
docker-compose up -d

# Local development
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/api_server.py

# Test the service
curl http://localhost:5002/health
curl -X POST -F "file=@audio.wav" http://localhost:5002/diarize
```

### Frontend Service
```bash
cd modules/frontend-service

# Docker deployment
docker-compose up -d

# Local development
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/frontend_service.py

# Access web interface
# Open http://localhost:3000
```

### Monitoring Stack
```bash
cd modules/monitoring-service

# Start full monitoring stack
docker-compose up -d

# Access dashboards
# Grafana: http://localhost:3001 (admin/admin)
# Prometheus: http://localhost:9090
```

## 📊 API Endpoints

### Whisper Service API (Port 5001)

#### Core Transcription
```bash
# Health check
GET /health

# List available models
GET /models

# Basic transcription
POST /transcribe
Content-Type: multipart/form-data
Body: file=audio.wav

# Model-specific transcription
POST /transcribe/{model_name}
Content-Type: multipart/form-data
Body: file=audio.wav, language=en, task=transcribe

# Enhanced transcription with audio processing
POST /transcribe/enhanced/{model_name}
Content-Type: multipart/form-data
Body: file=audio.wav, enhance_audio=true, noise_reduction=true
```

#### Real-time Streaming
```bash
# Configure streaming parameters
POST /stream/configure
Content-Type: application/json
Body: {"chunk_size": 1024, "overlap": 0.1, "vad_enabled": true}

# Start streaming session
POST /stream/start
Content-Type: application/json
Body: {"model_name": "whisper-small.en", "language": "auto"}

# Send audio chunk
POST /stream/audio
Content-Type: multipart/form-data
Body: audio_chunk=<binary_data>, session_id=<id>

# Get rolling transcriptions
GET /stream/transcriptions?session_id=<id>&limit=10

# Stop streaming
POST /stream/stop
Content-Type: application/json
Body: {"session_id": "<id>"}
```

#### Session Management
```bash
# Create session
POST /sessions
Content-Type: application/json
Body: {"model_name": "whisper-small.en", "language": "auto"}

# Get session info
GET /sessions/{session_id}

# Close session
DELETE /sessions/{session_id}
```

#### WebSocket Infrastructure
```bash
# System status and monitoring
GET /status                    # Service status and metrics
GET /connections              # Active WebSocket connections
GET /errors                   # Error statistics
GET /heartbeat               # Heartbeat monitoring stats
GET /router                  # Message routing information
GET /performance             # Performance metrics and optimization stats

# Authentication
POST /auth/login             # User authentication
POST /auth/guest            # Guest token creation
POST /auth/validate         # Token validation
POST /auth/logout           # Logout
GET /auth/stats             # Authentication statistics

# Session Persistence & Reconnection
GET /reconnection                        # Reconnection statistics
GET /sessions/{session_id}/info         # Detailed session information
GET /sessions/{session_id}/messages     # Buffered messages for session
```

#### WebSocket Events (ws://localhost:5001/ws)
```bash
# Connection Management
connect                      # Establish WebSocket connection
disconnect                   # Close WebSocket connection
join_session                 # Join transcription session
leave_session               # Leave transcription session

# Real-time Transcription
transcribe_stream           # Stream audio for real-time transcription

# Connection Health
ping                        # Send ping
pong                        # Respond to ping
heartbeat                   # Heartbeat monitoring

# Advanced Features
authenticate                # WebSocket authentication
route_message              # Message routing
subscribe_events           # Subscribe to event notifications
unsubscribe_events         # Unsubscribe from events
reconnect_session          # Reconnect to existing session
get_session_info           # Get session information
buffer_message             # Buffer message for session
```

### Translation Service API (Port 5003)

#### Core Translation
```bash
# Health check
GET /health
GET /api/health

# Service status
GET /api/status

# Translate text
POST /translate
Content-Type: application/json
Body: {
  "text": "Hello world",
  "target_language": "Spanish",
  "source_language": "English", # optional
  "use_local": true,             # optional
  "quality_threshold": 0.8       # optional
}

# Streaming translation
POST /translate/stream
Content-Type: application/json
Body: {"session_id": "<id>", "text": "Hello", "target_language": "es"}

# Language detection
POST /detect_language
Content-Type: application/json
Body: {"text": "Hello world"}

# Get supported languages
GET /languages
```

#### Session Management
```bash
# Create translation session
POST /sessions
Content-Type: application/json
Body: {"source_lang": "en", "target_lang": "es", "quality_threshold": 0.8}

# Get session
GET /sessions/{session_id}

# Close session
DELETE /sessions/{session_id}
```

#### WebSocket Events (ws://localhost:5003/translate/stream)
```bash
connect                     # Connect to translation stream
disconnect                  # Disconnect from stream
join_session               # Join translation session
leave_session              # Leave translation session
translate_stream           # Stream text for real-time translation
```

### Speaker Service API (Port 5002)

#### Core Diarization
```bash
# Health check
GET /health

# Diarize audio file
POST /diarize
Content-Type: multipart/form-data
Body: file=audio.wav, min_speakers=1, max_speakers=10

# Configure streaming
POST /stream/configure
Content-Type: application/json
Body: {"min_speakers": 1, "max_speakers": 5, "chunk_duration": 2.0}

# Start streaming diarization
POST /stream/start
Content-Type: application/json
Body: {"session_id": "<optional>"}

# Send audio chunk
POST /stream/audio
Content-Type: multipart/form-data
Body: audio_chunk=<binary>, session_id=<id>

# Get speaker segments
GET /stream/segments?session_id=<id>&limit=10

# Stop streaming
POST /stream/stop
Content-Type: application/json
Body: {"session_id": "<id>"}
```

#### Speaker Management
```bash
# Configure known speakers
POST /config/speakers
Content-Type: application/json
Body: {"speakers": [{"id": "speaker1", "name": "John", "samples": ["..."]}]}

# Clear speaker configuration
POST /config/clear

# Align transcription with speakers
POST /align
Content-Type: application/json
Body: {"transcription": "...", "speaker_segments": [...]}

# Service status
GET /status
```

#### WebSocket Events (ws://localhost:5002/stream)
```bash
connect                    # Connect to diarization stream
disconnect                 # Disconnect
join_session              # Join diarization session
leave_session             # Leave session
diarize_stream            # Stream audio for real-time diarization
```

### Frontend Service API (Port 3000)

#### Web Interface
```bash
# Main interface
GET /                      # Main web application
GET /settings             # Settings page

# Health check
GET /api/health
```

#### Pipeline Integration
```bash
# Start complete pipeline
POST /api/pipeline/start
Content-Type: application/json
Body: {"audio_source": "microphone", "target_language": "es"}

# Stop pipeline
POST /api/pipeline/stop/{session_id}

# Process audio through pipeline
POST /api/pipeline/process
Content-Type: multipart/form-data
Body: file=audio.wav, session_id=<id>

# Get pipeline sessions
GET /api/pipeline/sessions

# Get specific session
GET /api/pipeline/sessions/{session_id}
```

#### WebSocket Events (ws://localhost:3000/ws)
```bash
pipeline_start            # Start complete processing pipeline
pipeline_audio            # Send audio to pipeline
pipeline_stop             # Stop pipeline processing
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
- **Port conflicts**: Ensure ports 3000, 5001, 5003 are available
- **Docker memory**: Increase Docker memory limit to 8GB+ for model loading
- **Model downloads**: First run may take time to download Whisper models

#### ✅ **Audio Flow Now Fully Operational**
**Complete Pipeline**: Frontend → Orchestration → Whisper → Translation → Response
**Features Working**: Real-time streaming, dynamic models, hardware acceleration, error recovery

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
