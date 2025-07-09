# LiveTranslate

**Real-time Speech-to-Text Transcription and Translation System with AI Acceleration**

LiveTranslate is a comprehensive, production-ready system for real-time audio transcription and translation. It provides enterprise-grade features including NPU/GPU acceleration, speaker diarization, multi-language translation, distributed deployment capabilities, and advanced WebSocket infrastructure with performance optimization.

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![AI Acceleration](https://img.shields.io/badge/AI-NPU%2FGPU%20Ready-green)
![Architecture](https://img.shields.io/badge/Architecture-Microservices-purple)
![WebSocket](https://img.shields.io/badge/WebSocket-Enterprise%20Grade-orange)

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** (Windows/Mac/Linux) - Version 20.10+ recommended
- **8GB RAM minimum** (16GB+ recommended for GPU acceleration)
- **10GB storage** for models and data
- **Optional**: NVIDIA GPU (CUDA 11.8+) or Intel NPU for acceleration

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

#### Option A: Full System (All Services)
```bash
# Start all services with comprehensive setup
docker-compose -f docker-compose.comprehensive.yml up -d

# Check service health
docker-compose -f docker-compose.comprehensive.yml ps
```

#### Option B: Core Services Only (Lightweight)
```bash
# Start essential services only
docker-compose -f docker-compose.comprehensive.yml up -d frontend whisper translation whisper-redis translation-redis

# Verify services are running
curl http://localhost:3000/api/health
curl http://localhost:5001/health
curl http://localhost:5003/api/health
```

#### Option C: Development Setup (Live Code Editing)
```bash
# Start with volume mounting for development
docker-compose -f docker-compose.dev.yml up -d

# Code changes will be reflected immediately without rebuilds
# Perfect for development and testing
```

### 3. Access the System
- **Web Interface**: http://localhost:3000
- **Performance Dashboard**: http://localhost:3000 (Performance/Monitoring tab)
- **API Documentation**: See [API Endpoints](#api-endpoints) section below

### 4. First Usage
1. Open http://localhost:3000 in your browser
2. Click "Start Recording" or upload an audio file
3. Select your target language for translation
4. View real-time transcription and translation results
5. Access performance metrics via the monitoring dashboard

## 📡 Service Endpoints & Ports

| Service | Port | Primary Endpoint | Purpose | Health Check |
|---------|------|------------------|---------|--------------|
| **Frontend** | 3000 | http://localhost:3000 | Web interface & API gateway | `/api/health` |
| **Whisper** | 5001 | http://localhost:5001 | Speech-to-text transcription | `/health` |
| **Translation** | 5003 | http://localhost:5003 | Multi-language translation | `/api/health` |
| **Speaker** | 5002 | http://localhost:5002 | Speaker diarization | `/health` |
| **Monitoring** | 3001 | http://localhost:3001 | Grafana dashboards | `/api/health` |
| **Prometheus** | 9090 | http://localhost:9090 | Metrics collection | `/-/healthy` |

### WebSocket Endpoints
| Service | WebSocket URL | Purpose |
|---------|---------------|---------|
| **Whisper** | ws://localhost:5001/ws | Real-time transcription streaming |
| **Translation** | ws://localhost:5003/translate/stream | Real-time translation streaming |
| **Speaker** | ws://localhost:5002/stream | Real-time speaker diarization |
| **Frontend** | ws://localhost:3000/ws | System coordination & updates |

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

# Docker with GPU acceleration
docker-compose -f docker-compose-gpu.yml up -d

# Docker simple setup (CPU)
docker-compose -f docker-compose-simple.yml up -d

# Local development with Ollama
pip install -r requirements-minimal.txt
python src/api_server.py --backend ollama

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
- **Speaker Diarization**: Multi-speaker identification and tracking
- **Translation**: Local LLM-based translation (vLLM/Ollama)
- **Language Detection**: Automatic source language identification

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

### Common Issues
- **Port conflicts**: Ensure ports 3000, 5001, 5003 are available
- **Docker memory**: Increase Docker memory limit to 8GB+ for model loading
- **Model downloads**: First run may take time to download Whisper models

## 🚀 Future Roadmap

### Phase 1: Core Functionality (Q1 2024)
- ✅ Docker infrastructure
- 🔄 Audio processing pipeline
- 🔄 Whisper transcription
- 📋 Speaker diarization

### Phase 2: AI Enhancement (Q2 2024)
- 📋 NPU acceleration optimization
- 📋 Multi-model support
- 📋 Translation pipeline
- 📋 Performance optimization

### Phase 3: Enterprise Features (Q3 2024)
- 📋 Meeting bot framework
- 📋 Google Meet integration
- 📋 Advanced security
- 📋 Scalability features

### Phase 4: Advanced AI (Q4 2024)
- 📋 Advanced analytics
- 📋 Custom model training
- 📋 Enterprise deployment
- 📋 API ecosystem

## 📝 License

This project is licensed under the terms of the included license file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `docker-compose -f docker-compose.test.yml up`
5. Submit a pull request

## 🆘 Support

### Documentation
- [Service Documentation](modules/README.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api.md)

### Troubleshooting
- [Common Issues](docs/troubleshooting.md)
- [Performance Tuning](docs/performance.md)
- [Development Setup](docs/development.md)

### Community
- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Documentation: Project Wiki

---

**LiveTranslate** - Breaking down language barriers with AI-powered real-time translation. 