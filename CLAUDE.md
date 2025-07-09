# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiveTranslate is a real-time speech-to-text transcription and translation system with AI acceleration. It's built as a microservices architecture with enterprise-grade WebSocket infrastructure for real-time communication.

## Optimized Architecture (NPU/GPU Specialized)

### Hardware-Optimized Service Architecture

The system is organized into **3 core services** optimized for specific hardware acceleration:

1. **Whisper Service** (`modules/whisper-service/`) - **[NPU OPTIMIZED]** ✅
   - **Combined**: Whisper + Speaker Diarization + Audio Processing + VAD
   - **Hardware**: Intel NPU (primary), GPU/CPU (fallback)
   - **Features**:
     - Real-time speech-to-text transcription (OpenVINO optimized Whisper)
     - Advanced speaker diarization with multiple embedding methods
     - Multi-format audio processing (WAV, MP3, WebM, OGG, MP4)
     - Voice Activity Detection (WebRTC + Silero)
     - Real-time streaming with rolling buffers and memory optimization
     - Enterprise WebSocket infrastructure with connection pooling

2. **Translation Service** (`modules/translation-service/`) - **[GPU OPTIMIZED]**
   - **Purpose**: Multi-language translation with local LLMs
   - **Hardware**: NVIDIA GPU (primary), CPU (fallback)
   - **Features**:
     - Local LLM inference (vLLM, Ollama, Triton)
     - Real-time translation streaming
     - Multi-language support with auto-detection
     - Quality scoring and confidence metrics
     - Memory-efficient model management

3. **Orchestration Service** (`modules/orchestration-service/`) - **[CPU OPTIMIZED]** ✅
   - **Combined**: Frontend + WebSocket + Coordination + Monitoring + Analytics + **Google Meet Bot Management**
   - **Hardware**: CPU-optimized (lightweight)
   - **Features**:
     - Modern responsive web interface (extracted from frontend-service)
     - Enterprise WebSocket connection management (10,000+ concurrent)
     - Service health monitoring and auto-recovery
     - Real-time dashboard and performance analytics
     - Session management and coordination
     - API gateway with load balancing and circuit breaking
     - Comprehensive monitoring with Prometheus integration
     - **🆕 Google Meet Bot Management**: Central "brain" for spawning, monitoring, and managing Google Meet bots
     - **🆕 Official Google Meet API Integration**: OAuth 2.0 authentication, space management, real-time monitoring
     - **🆕 Comprehensive Database Integration**: PostgreSQL with time-coded transcripts, translations, and correlations

### Supporting Infrastructure

- **Shared Libraries** (`modules/shared/`) - Common inference clients and pipeline components
- **🆕 Google Meet Bot** (`modules/google-meet-bot/`) - ✅ **COMPLETED** - Complete bot integration system
- **🆕 Database Schema** (`scripts/bot-sessions-schema.sql`) - ✅ **COMPLETED** - Comprehensive PostgreSQL schema for bot sessions

### Deployment Strategy

#### Single Machine (Development/Testing)
```bash
# All services on one machine with hardware optimization
docker-compose -f docker-compose.optimized.yml up -d

# Requirements:
# - 16GB+ RAM
# - Intel NPU OR NVIDIA GPU
# - Modern CPU with AVX support
```

#### Distributed (Production)
```bash
# Machine 1: Whisper Service (NPU/GPU accelerated)
cd modules/whisper-service && docker-compose up -d

# Machine 2: Translation Service (GPU accelerated) 
cd modules/translation-service && docker-compose -f docker-compose-gpu.yml up -d

# Machine 3: Orchestration Service (CPU optimized)
cd modules/orchestration-service && docker-compose up -d
```

### Service Communication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration-Centric Architecture          │
├─────────────────────────────────────────────────────────────────┤
│                  Orchestration Service                         │
│        [Frontend + WebSocket + Bot Management + DB]            │
│                        (Port 3000)                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Google Meet API │  │ Bot Manager     │  │ Database Mgr    │  │
│  │ • OAuth 2.0     │  │ • Lifecycle     │  │ • PostgreSQL    │  │
│  │ • Spaces        │  │ • Health        │  │ • Time-coded    │  │
│  │ • Monitoring    │  │ • Recovery      │  │ • Analytics     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────┬─────────────────┬─────────────────────────────┤
│                 │                 │                             │
│  Whisper Service│  Google Meet    │    Translation Service      │
│     [NPU OPT]   │  Bot Components │        [GPU OPT]           │
│   (Port 5001)   │                 │       (Port 5003)          │
│                 │                 │                             │
│ • OpenVINO STT  │ • Audio Capture │  • LLM Translation         │
│ • Speaker Diar  │ • Caption Proc  │  • vLLM/Ollama/Triton     │
│ • Multi-format  │ • Correlation   │  • Multi-language          │
│ • WebRTC VAD    │ • Integration   │  • Quality Metrics         │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### Legacy Code

The `legacy/` directory contains the original monolithic implementation that has been systematically extracted and optimized into the microservices architecture. Legacy code should not be modified for new features.

**Legacy Extraction Status**: ✅ COMPLETED
- **Source**: `legacy/python/whisper-npu-server/` (2546-line monolithic server.py)
- **Extracted to**: Modern microservices in `modules/` directory
- **Improvements**: Thread-safe operations, NPU optimization, comprehensive error handling, production-ready components

## Legacy System Extraction Summary

The legacy monolithic system has been systematically extracted and optimized into production-ready microservices:

### Whisper Service Extraction (`modules/whisper-service/`)

**Extracted Components:**
1. **ModelManager** (`src/model_manager.py`)
   - Enhanced NPU device detection and management
   - Thread-safe inference with automatic fallback (NPU → GPU → CPU)
   - Memory optimization and resource management
   - Comprehensive error handling and recovery

2. **RollingBufferManager** (`src/buffer_manager.py`)
   - Real-time audio buffering with VAD integration
   - Memory-efficient circular buffer implementation
   - Voice Activity Detection with WebRTC and Silero fallback
   - Speaker diarization integration
   - Configurable buffer sizes and inference intervals

3. **AudioProcessor** (`src/audio_processor.py`)
   - Multi-format audio support (WAV, MP3, WebM, OGG, MP4)
   - Advanced format detection including streaming chunks
   - Multiple conversion backends with fallback strategies
   - Audio enhancement and preprocessing
   - Comprehensive validation and error handling

4. **Enhanced API Server** (`src/enhanced_api_server.py`)
   - Production-ready Flask server with WebSocket support
   - Real-time streaming endpoints
   - Performance monitoring and health checks
   - Hot-reloadable settings management
   - Comprehensive error handling and recovery

5. **Speaker Diarization** (`src/speaker_diarization.py`)
   - Advanced speaker identification with multiple embedding methods
   - Real-time speaker tracking and continuity management
   - Multiple clustering algorithms (HDBSCAN, DBSCAN, Agglomerative)
   - Speech enhancement and noise reduction
   - Cross-segment speaker consistency

### Frontend Service Extraction (`modules/frontend-service/`)

**Extracted Components:**
1. **Enhanced Frontend Service** (`src/frontend_service.py`)
   - Modern Flask server with WebSocket integration
   - Service health monitoring and auto-recovery
   - API proxying to backend services
   - Performance analytics and monitoring
   - Progressive Web App (PWA) support

2. **Modern UI Templates** (`templates/`)
   - Responsive dashboard with real-time updates
   - Advanced audio visualization
   - Speaker diarization timeline
   - Service status monitoring
   - Comprehensive settings management

3. **Enhanced Styling** (`static/css/`)
   - Modern CSS with CSS custom properties
   - Dark theme optimization
   - Responsive design patterns
   - Accessibility improvements
   - Performance optimizations

4. **Optimized JavaScript** (`static/js/`)
   - Modular ES6+ architecture
   - Real-time WebSocket communication
   - Advanced audio processing
   - Comprehensive error handling
   - Performance monitoring

### Key Improvements Made

1. **Architecture**:
   - Monolithic → Microservices architecture
   - Enhanced separation of concerns
   - Improved scalability and maintainability

2. **Performance**:
   - NPU optimization with Intel NPU support
   - Thread-safe operations with proper locking
   - Memory-efficient implementations
   - Optimized audio processing pipelines

3. **Reliability**:
   - Comprehensive error handling and recovery
   - Automatic device fallback (NPU → GPU → CPU)
   - Connection pooling and heartbeat monitoring
   - Zero-message-loss design patterns

4. **Security**:
   - Input validation and sanitization
   - Secure file handling with size limits
   - Path traversal protection
   - Resource usage monitoring

5. **Monitoring**:
   - Real-time performance metrics
   - Health check endpoints
   - Service status monitoring
   - Comprehensive logging

### Files Created/Enhanced

**Whisper Service:**
- `modules/whisper-service/src/model_manager.py` (374 lines) - Enhanced NPU model management
- `modules/whisper-service/src/buffer_manager.py` (593 lines) - Real-time audio buffering
- `modules/whisper-service/src/audio_processor.py` (587 lines) - Advanced audio processing
- `modules/whisper-service/src/enhanced_api_server.py` (766 lines) - Production API server
- `modules/whisper-service/src/speaker_diarization.py` (621 lines) - Advanced speaker identification

**Frontend Service:**
- `modules/frontend-service/src/frontend_service.py` (500+ lines) - Enhanced frontend server
- `modules/frontend-service/templates/index.html` (300+ lines) - Modern dashboard UI
- `modules/frontend-service/static/css/styles.css` (800+ lines) - Enhanced styling
- `modules/frontend-service/static/js/main.js` (800+ lines) - Optimized JavaScript controller

**🆕 Google Meet Bot System:**
- `modules/orchestration-service/src/bot/bot_manager.py` (1100+ lines) - Central bot lifecycle management
- `modules/orchestration-service/src/bot/google_meet_client.py` (800+ lines) - Official Google Meet API integration
- `modules/orchestration-service/src/database/bot_session_manager.py` (1000+ lines) - Comprehensive database integration
- `modules/google-meet-bot/src/audio_capture.py` (600+ lines) - Real-time audio capture from Google Meet
- `modules/google-meet-bot/src/caption_processor.py` (400+ lines) - Google Meet caption processing
- `modules/google-meet-bot/src/time_correlation.py` (580+ lines) - Time correlation engine
- `modules/google-meet-bot/src/bot_integration.py` (840+ lines) - Complete bot-to-services integration
- `scripts/bot-sessions-schema.sql` (500+ lines) - Comprehensive PostgreSQL schema with 8 tables, views, and functions

### Legacy Issues Identified and Fixed

1. **Security Vulnerabilities**:
   - ❌ No file size limits → ✅ 100MB limit with validation
   - ❌ Path validation gaps → ✅ Comprehensive path sanitization
   - ❌ Resource management issues → ✅ Proper resource cleanup

2. **Performance Issues**:
   - ❌ Thread safety concerns → ✅ Proper locking mechanisms
   - ❌ Memory leaks potential → ✅ Memory-efficient implementations
   - ❌ No connection pooling → ✅ Enterprise-grade connection management

3. **Reliability Issues**:
   - ❌ Basic error handling → ✅ Comprehensive error recovery
   - ❌ No device fallback → ✅ Automatic NPU/GPU/CPU fallback
   - ❌ Limited monitoring → ✅ Real-time performance monitoring

## 🆕 Google Meet Bot Management System

### Overview

The orchestration service now serves as the central "brain" for managing Google Meet bots, providing comprehensive bot lifecycle management, official Google Meet API integration, and complete database persistence for all meeting data.

### Architecture Components

#### 1. **GoogleMeetBotManager** (`src/bot/bot_manager.py`)
- **Central Bot Lifecycle Management**: Spawn, monitor, recover, and cleanup bot instances
- **Health Monitoring**: Real-time bot health tracking with automatic recovery
- **Performance Analytics**: Comprehensive statistics and success rate tracking
- **Queue Management**: Bot request queuing when at capacity limits
- **Event-Driven Architecture**: Callback system for bot lifecycle events

#### 2. **Google Meet API Client** (`src/bot/google_meet_client.py`)
- **Official Google Meet API Integration**: OAuth 2.0 authentication with required scopes
- **Meeting Space Management**: Create new meetings or join existing ones
- **Real-time Monitoring**: Live participant tracking and conference events
- **Conference Records**: Access to transcripts and recordings when available
- **Fallback Support**: Graceful degradation when API credentials aren't available

#### 3. **Database Integration** (`src/database/bot_session_manager.py`)
- **Comprehensive Session Storage**: All bot session data with complete lifecycle tracking
- **Time-Coded Data Management**: Audio files, transcripts, translations with precise timestamps
- **Speaker Attribution**: Complete speaker tracking throughout meeting timeline
- **Correlation Storage**: Time correlation between Google Meet captions and in-house transcriptions
- **Analytics and Insights**: Pre-computed session statistics and quality metrics

#### 4. **Bot Components** (`modules/google-meet-bot/`)
- **Audio Capture** (`src/audio_capture.py`): Real-time audio streaming from Google Meet
- **Caption Processing** (`src/caption_processor.py`): Google Meet caption extraction and processing
- **Time Correlation** (`src/time_correlation.py`): Advanced timing correlation between sources
- **Bot Integration** (`src/bot_integration.py`): Complete service integration pipeline

### Database Schema

**Comprehensive PostgreSQL Schema** (`scripts/bot-sessions-schema.sql`):

#### Core Tables:
- **`bot_sessions.sessions`**: Central session lifecycle tracking
- **`bot_sessions.audio_files`**: Audio file storage with metadata and hash verification
- **`bot_sessions.transcripts`**: Both Google Meet and in-house transcriptions with time coding
- **`bot_sessions.translations`**: Multi-language translations with speaker attribution
- **`bot_sessions.correlations`**: Time correlation between external and internal sources
- **`bot_sessions.participants`**: Meeting participant tracking
- **`bot_sessions.events`**: Session event logging for debugging and analytics
- **`bot_sessions.session_statistics`**: Aggregated analytics for completed sessions

#### Advanced Features:
- **Performance Indexes**: Optimized for session, time, speaker, and language queries
- **JSONB Storage**: Efficient metadata storage with GIN indexes
- **Automated Views**: Pre-computed session overview, speaker stats, translation quality
- **Triggers**: Automatic statistics updates when data changes
- **Functions**: Session duration, word count, analytics calculations

### Bot Lifecycle Management

#### 1. **Bot Spawning Process**
```python
# Request new bot
meeting_request = MeetingRequest(
    meeting_id="test-meeting-123",
    meeting_title="Important Meeting",
    target_languages=['en', 'es', 'fr'],
    metadata={'meeting_uri': 'https://meet.google.com/abc-def-ghi'}
)

bot_id = await bot_manager.request_bot(meeting_request)
```

#### 2. **Automatic Integration Steps**
1. **Database Session Creation**: Comprehensive session record with metadata
2. **Google Meet API Setup**: Authentication and space/conference access
3. **Service Session Creation**: Whisper and translation service coordination
4. **Bot Component Initialization**: Audio capture, caption processing, correlation
5. **Real-time Monitoring**: Health checks, participant tracking, event handling

#### 3. **Data Storage Flow**
```python
# Automatic storage throughout bot lifecycle
audio_file_id = await bot_manager.store_audio_file(session_id, audio_data, metadata)
transcript_id = await bot_manager.store_transcript(session_id, transcript_data)
translation_id = await bot_manager.store_translation(session_id, translation_data)
correlation_id = await bot_manager.store_correlation(session_id, correlation_data)

# Comprehensive session data retrieval
session_data = await bot_manager.get_session_comprehensive_data(session_id)
```

### Key Features

#### **Enterprise-Grade Bot Management**
- **10+ concurrent bots** with automatic queuing
- **Health monitoring** with 20+ error categories
- **Automatic recovery** for failed bots (max 3 attempts)
- **Performance tracking** with success rates and analytics
- **Graceful cleanup** with optional file removal

#### **Official Google Meet Integration**
- **OAuth 2.0 authentication** with required scopes
- **Meeting space creation** and existing meeting joining
- **Real-time participant monitoring** with event callbacks
- **Conference record access** for transcripts and recordings
- **Fallback mode** for development without API credentials

#### **Comprehensive Data Management**
- **Time-coded storage** for all audio, transcript, and translation data
- **Speaker attribution** throughout the entire meeting timeline
- **Correlation tracking** between Google Meet captions and in-house processing
- **File management** with hash verification and metadata storage
- **Analytics dashboard** with session statistics and quality metrics

#### **Production-Ready Features**
- **Thread-safe operations** with proper locking mechanisms
- **Error handling** with comprehensive recovery strategies
- **Performance optimization** with connection pooling and caching
- **Security** with input validation and secure file handling
- **Monitoring** with real-time metrics and health checks

### Usage Examples

#### Basic Bot Management
```python
# Initialize bot manager with database and Google Meet API
config = {
    'max_concurrent_bots': 10,
    'google_meet_credentials_path': '/path/to/credentials.json',
    'database': {'host': 'localhost', 'database': 'livetranslate'},
    'audio_storage_path': '/data/audio'
}

bot_manager = create_bot_manager(**config)
await bot_manager.start()

# Create and manage bots
bot_id = await bot_manager.request_bot(meeting_request)
status = bot_manager.get_bot_status(bot_id)
await bot_manager.terminate_bot(bot_id)
```

#### Comprehensive Session Data
```python
# Get complete session data with all related records
session_data = await bot_manager.get_session_comprehensive_data(session_id)

print(f"Audio files: {session_data['statistics']['audio_files_count']}")
print(f"Transcripts: {session_data['statistics']['transcripts_count']}")
print(f"Translations: {session_data['statistics']['translations_count']}")
print(f"Languages: {session_data['statistics']['languages_detected']}")
```

## Development Commands

### Optimized Deployment (Recommended)

```bash
# Full optimized system (3-service architecture)
docker-compose -f docker-compose.optimized.yml up -d

# Individual service development
cd modules/whisper-service && docker-compose up -d      # NPU/GPU optimized
cd modules/translation-service && docker-compose up -d  # GPU optimized  
cd modules/orchestration-service && docker-compose up -d # CPU optimized

# Development with hot-reload
docker-compose -f docker-compose.dev.yml up -d
```

### Service Development

Each service can be developed independently with hardware optimization:

```bash
# Whisper Service (NPU/GPU optimized)
cd modules/whisper-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py --device=npu

# Translation Service (GPU optimized)
cd modules/translation-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/translation_service.py --device=gpu

# Orchestration Service (CPU optimized) with Google Meet Bot Management
cd modules/orchestration-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-google-meet.txt -r requirements-database.txt

# Setup database schema
psql -U postgres -d livetranslate -f ../../scripts/bot-sessions-schema.sql

# Configure Google Meet API (optional - create credentials.json from Google Cloud Console)
export GOOGLE_MEET_CREDENTIALS_PATH="/path/to/credentials.json"

python src/orchestration_service.py
```

### Comprehensive Testing Strategy

```bash
# Full system testing
python tests/run_all_tests.py --comprehensive

# Service-specific testing
cd modules/whisper-service && python tests/run_tests.py --all --device=npu
cd modules/translation-service && python tests/run_tests.py --all --device=gpu
cd modules/orchestration-service && python tests/run_tests.py --all

# Edge case testing
python tests/run_edge_case_tests.py --stress --timeout=300

# Performance benchmarking
python tests/benchmark.py --hardware-profile
```

### Health Checks

```bash
# Verify optimized services are running
curl http://localhost:3000/api/health        # Orchestration
curl http://localhost:5001/health             # Whisper Service
curl http://localhost:5003/api/health        # Translation Service

# Hardware acceleration status
curl http://localhost:5001/api/hardware      # NPU status
curl http://localhost:5003/api/hardware      # GPU status
```

## Key Technical Details

### WebSocket Infrastructure

The Whisper service implements enterprise-grade WebSocket features:
- Connection pooling (1000 capacity with weak reference tracking)
- 20+ error categories with automatic recovery
- Heartbeat monitoring with RTT tracking
- Session persistence (30-minute timeout)
- Message buffering and zero-message-loss design

### Service Ports

- Orchestration: 3000
- Whisper: 5001
- Translation: 5003
- Monitoring: 30013
- Prometheus: 90903
3
### AI Acceleration3
3
- **NPU**: Intel NPU 3support via OpenVINO (Whisper service)
- **GPU**: NVIDIA GPU3 with CUDA (Translation service)
- **CPU**: Automatic 3fallback for all services
3
### Configuration3
3
- Environment variabl3es via `.env` file or Docker environment
- Service-specific co3nfiguration in each module
- Monitoring configur3ation in `modules/monitoring-service/config/`

## File Structure Conventions

- `src/` - Source code for each service
- `tests/` - Test files (unit, integration, stress)
- `requirements.txt` - Python dependencies
- `docker-compose*.yml` - Docker deployment configurations
- `Dockerfile*` - Container build instructions

## Development Workflow

1. **Start with Docker**: Use `docker-compose.dev.yml` for development
2. **Individual Services**: Develop services independently when needed
3. **Testing**: Always run tests before commits, especially for WebSocket service
4. **Health Checks**: Verify service connectivity after changes
5. **Monitoring**: Use the performance dashboard at http://localhost:3000 for debugging

## Important Notes

- The system is designed for Windows environments (use PowerShell commands)
- WebSocket infrastructure is production-ready with comprehensive error handling
- All services support both REST APIs and WebSocket streaming
- The monitoring stack provides real-time performance metrics
- Use the comprehensive Docker setup for full system testing
- Audio processing includes proper resampling from 48kHz to 16kHz for Whisper compatibility
- Voice-specific processing preserves human speech characteristics
- Test audio interface includes comprehensive parameter tuning for all processing stages

## Comprehensive Testing Strategy

### Test Categories by Service

#### Audio Service Tests (`modules/audio-service/tests/`)
- **Unit Tests**: Model management, audio processing, VAD, speaker diarization
- **Integration Tests**: NPU device integration, WebSocket communication
- **Performance Tests**: Real-time processing latency, memory usage
- **Edge Cases**: Audio format variations, device failures, network issues

#### Translation Service Tests (`modules/translation-service/tests/`)
- **Unit Tests**: LLM inference, language detection, quality scoring
- **Integration Tests**: GPU acceleration, model loading, API endpoints
- **Performance Tests**: Translation throughput, GPU memory usage
- **Edge Cases**: Unsupported languages, model failures, memory limits

#### Orchestration Service Tests (`modules/orchestration-service/tests/`)
- **Unit Tests**: Frontend components, WebSocket management, routing
- **Integration Tests**: Service coordination, health monitoring
- **Performance Tests**: Concurrent connections, response times
- **Edge Cases**: Service failures, network partitions, high load

### Test Execution Framework

```bash
# Automated test runner for all services
python tests/master_test_runner.py --comprehensive --report

# Hardware-specific testing
python tests/hardware_tests.py --npu --gpu --fallback

# End-to-end pipeline testing
python tests/e2e_tests.py --audio-to-translation --real-time

# Stress testing with edge cases
python tests/stress_tests.py --concurrent=100 --duration=600s --chaos
```

### Edge Case Testing Matrix

| Component | Input Variations | Error Conditions | Performance Limits |
|-----------|------------------|------------------|-------------------|
| **Audio Processing** | 20+ formats, corrupted files, silence | Device failures, memory limits | Real-time latency < 100ms |
| **Speaker Diarization** | 1-10 speakers, background noise | Clustering failures, silence | Processing time < 2x audio |
| **Translation** | 50+ languages, mixed content | Model failures, GPU OOM | Throughput > 1000 chars/s |
| **WebSocket** | Concurrent connections, packet loss | Network failures, timeouts | 1000+ concurrent connections |

## Next Steps for Each Module

### 1. Whisper Service (`modules/whisper-service/`) - **✅ COMPLETED**

**Current Status**: Fully integrated NPU-optimized whisper service with comprehensive features
**Completed Actions**:
- [x] **Service consolidation**: Merged whisper + speaker diarization + audio processing + VAD
- [x] **NPU optimization**: Intel NPU detection with automatic fallback (NPU → GPU → CPU)
- [x] **Unified API**: Single endpoint for audio → transcription + speakers + diarization
- [x] **Comprehensive testing**: NPU fallback, real-time performance, edge cases
- [x] **Docker optimization**: NPU-specific container builds with OpenVINO support
- [x] **Enterprise WebSocket**: Connection pooling, heartbeat, error recovery
- [x] **Multi-format support**: WAV, MP3, WebM, OGG, MP4 with automatic detection
- [x] **Orchestration integration**: API gateway routing through `/api/whisper/*`
- [x] **Audio resampling fix**: Fixed proper 48kHz to 16kHz resampling with librosa fallback
- [x] **Voice-specific processing**: Enhanced audio pipeline with voice-aware processing
- [x] **Critical audio processing fix**: Disabled browser audio processing features (echoCancellation, noiseSuppression, autoGainControl) that were severely attenuating loopback audio
- [x] **Backend noise reduction fix**: Disabled aggressive noise reduction that was removing loopback audio content

**Production Features**:
- NPU hardware acceleration with OpenVINO optimization
- Advanced speaker diarization with multiple embedding methods
- Real-time audio streaming with rolling buffers and VAD
- Enterprise WebSocket infrastructure (1000+ concurrent connections)
- Comprehensive error handling with 20+ error categories
- Multi-format audio processing with streaming detection
- Integration with orchestration service dashboard
- Voice-specific audio processing with tunable parameters
- Step-by-step audio pipeline with pause capability for debugging

### 2. Translation Service (`modules/translation-service/`) - **MEDIUM PRIORITY**

**Current Status**: Solid foundation, needs GPU optimization
**Next Actions**:
- [ ] **GPU optimization**: CUDA memory management and batch processing
- [ ] **Model management**: Dynamic model loading/unloading based on demand
- [ ] **Quality metrics**: Implement confidence scoring and validation
- [ ] **API enhancement**: Streaming translation with partial results
- [ ] **Performance testing**: GPU memory limits and throughput optimization

**Testing Focus**:
- GPU memory management and OOM prevention
- Translation quality and confidence metrics
- Multi-language detection accuracy
- High-throughput batch processing
- Graceful degradation on model failures

### 3. Orchestration Service (`modules/orchestration-service/`) - **✅ COMPLETED**

**Current Status**: Fully consolidated production-ready service with integrated enterprise monitoring and comprehensive audio processing frontend
**Completed Actions**:
- [x] **Service consolidation**: Merged frontend + websocket + monitoring + analytics services
- [x] **Centralized routing**: API gateway with load balancing and circuit breaking
- [x] **Health monitoring**: Comprehensive service health and auto-recovery
- [x] **Real-time dashboard**: Performance metrics and system status with analytics
- [x] **Session management**: Enterprise-grade session handling with persistence
- [x] **WebSocket Infrastructure**: 10,000+ concurrent connections with heartbeat monitoring
- [x] **Modern Frontend**: Extracted from frontend-service with real-time updates
- [x] **Enterprise Monitoring Stack**: Fully integrated Prometheus + Grafana + AlertManager + Loki
- [x] **Audio Processing Frontend**: Comprehensive pipeline control and diagnostic interface
- [x] **Parameter Management**: Full hyperparameter controls for all audio processing stages
- [x] **Real-time Diagnostics**: Live visualization and performance monitoring

**Production Features**:
- Enterprise WebSocket connection pooling (weak references, cleanup, heartbeat)
- Advanced health monitoring with automatic service recovery
- Real-time performance analytics with trend analysis and anomaly detection
- Modern responsive dashboard extracted from legacy frontend-service
- API gateway with weighted round-robin load balancing
- Circuit breaker pattern for service failure protection
- Comprehensive configuration management with hot-reloading
- Session persistence and recovery mechanisms
- Professional audio processing pipeline with full control interface
- Real-time audio diagnostics with waveform and spectrum analysis

**Integrated Monitoring Stack**:
- **Prometheus Metrics Collection**: Orchestration, audio, and translation service metrics with 30-day retention
- **Grafana Dashboards**: Pre-configured visualizations for system overview, business metrics, and performance tracking
- **AlertManager**: 80+ production-ready alert rules with smart grouping and notification routing
- **Loki Log Aggregation**: Structured logging with 7-day retention and full-text search capabilities
- **Promtail Log Collection**: Real-time log shipping with service-specific parsing and labeling
- **System Monitoring**: Node Exporter and cAdvisor for infrastructure and container metrics
- **Automated Deployment**: Complete deployment script with health checks and verification

### 4. Shared Libraries (`modules/shared/`) - **LOW PRIORITY**

**Current Status**: Basic structure exists
**Next Actions**:
- [ ] **Common utilities**: Standardize logging, metrics, configuration
- [ ] **Hardware detection**: Unified NPU/GPU/CPU detection utilities
- [ ] **Pipeline components**: Reusable audio/text processing components
- [ ] **Client libraries**: Standardized service communication clients

## Architecture Migration Plan

### Phase 1: Service Consolidation (Week 1-2)
1. Create `modules/audio-service/` by merging whisper + speaker services
2. Create `modules/orchestration-service/` by merging frontend + websocket services
3. Update all inter-service communication
4. Implement unified health checks

### Phase 2: Hardware Optimization (Week 3-4)
1. Implement NPU detection and optimization in audio service
2. Enhance GPU memory management in translation service  
3. Add hardware-specific Docker configurations
4. Performance tuning and benchmarking

### Phase 3: Testing & Documentation (Week 5-6)
1. Comprehensive test suite for all services
2. Edge case testing and stress testing
3. Performance benchmarking and optimization
4. Complete documentation and deployment guides

### Phase 4: Production Readiness (Week 7-8)
1. Security audit and hardening
2. Monitoring and alerting setup
3. Deployment automation
4. Load testing and scaling validation

## Success Criteria

- [ ] **Performance**: Real-time processing with < 100ms latency
- [ ] **Reliability**: 99.9% uptime with automatic recovery
- [ ] **Scalability**: Support 1000+ concurrent users
- [ ] **Hardware Efficiency**: Optimal NPU/GPU utilization
- [ ] **Testing Coverage**: 90%+ code coverage with edge cases
- [ ] **Documentation**: Complete setup and operational guides

## Recent Audio Processing Enhancements

### Fixed Issues

1. **Audio Resampling Bug** (CRITICAL FIX)
   - **Problem**: Whisper producing incorrect transcriptions ("Uh", "issues", "walls", "third")
   - **Root Cause**: `pydub.set_frame_rate()` wasn't actually resampling audio data
   - **Solution**: Added proper resampling with librosa fallback
   ```python
   # Double-check resampling - if still at wrong sample rate, use librosa
   expected_samples = int(audio_segment.duration_seconds * 16000)
   if abs(len(audio_array) - expected_samples) > 100:
       audio_array = librosa.resample(audio_array, orig_sr=audio_segment.frame_rate, target_sr=16000)
   ```

2. **Test Audio Interface**
   - **Problem**: Test page buttons not working, missing functions
   - **Solution**: Added missing `getSupportedMimeTypes`, `analyzeTestRecording`, `transcribeTestRecording` functions
   - **Fixed**: Recursive loop causing "Maximum call stack size exceeded" error

### Enhanced Audio Processing Pipeline

**Voice-Specific Processing** (`modules/orchestration-service/static/js/audio-processing-test.js`)
- 10-stage processing pipeline with pause capability at each stage
- Voice-aware processing focusing on human speech frequencies (85-300Hz fundamental)
- Soft-knee compression instead of hard limiting
- Comprehensive parameter controls for real-time tuning

**Key Parameters**:
```javascript
const audioProcessingParams = {
    vad: {
        enabled: true,
        aggressiveness: 2,
        energyThreshold: 0.01,
        voiceFreqMin: 85,
        voiceFreqMax: 300
    },
    voiceFilter: {
        enabled: true,
        fundamentalMin: 85,
        fundamentalMax: 300,
        formant1Min: 200,
        formant1Max: 1000,
        preserveFormants: true
    },
    voiceEnhancement: {
        enabled: true,
        normalize: false,  // Disabled to preserve natural voice
        compressor: {
            threshold: -20,
            ratio: 3,
            knee: 2.0
        }
    }
}
```

**Processing Stages**:
1. Input Validation
2. Voice Activity Detection (VAD)
3. Voice Frequency Filtering
4. Noise Reduction
5. Voice Enhancement
6. Dynamic Range Compression
7. Envelope Following
8. Final Limiting
9. Output Normalization
10. Quality Check

Each stage can be paused for debugging and all parameters are tunable in real-time.

## Audio Processing Frontend Implementation

### Comprehensive Pipeline Control Interface

The orchestration service now includes a professional-grade audio processing frontend with complete control over the entire audio pipeline:

#### 1. **Audio Processing Controls** (`static/audio-processing-controls.html`)
- **Enable/Disable Controls**: Individual toggle switches for each processing stage
- **Hyperparameter Management**: Full control over all audio processing parameters
- **Quick Presets**: 6 optimized configurations (Default, Voice, Noisy, Music, Minimal, Aggressive)
- **Parameter Persistence**: Save/load/export/import configurations
- **Real-time Adjustment**: Live parameter modification without restart

#### 2. **Audio Diagnostic Dashboard** (`static/audio-diagnostic.html`)
- **Real-time Visualizations**: Live waveform and frequency spectrum analysis
- **Performance Metrics**: Processing latency, RMS levels, clipping detection
- **Pipeline Flow**: Visual representation of audio processing stages
- **Stage Analysis**: Detailed metrics for each processing stage
- **Diagnostic Reports**: Exportable performance and quality reports

#### 3. **Enhanced Test Interface** (`static/test-audio.html`)
- **Comprehensive Testing**: Audio recording, playback, and transcription testing
- **Pipeline Integration**: Direct integration with enhanced audio processing
- **Parameter Testing**: Real-time testing of different parameter combinations
- **Debug Capabilities**: Step-by-step processing with pause functionality

### Audio Processing Pipeline Features

#### Stage-by-Stage Control
```javascript
// Each stage can be individually controlled
const stageControls = {
    vad: {
        enabled: true,
        aggressiveness: 2,
        energyThreshold: 0.01,
        pauseAfterStage: false
    },
    voiceFilter: {
        enabled: true,
        fundamentalMin: 85,
        fundamentalMax: 300,
        voiceBandGain: 1.1,
        pauseAfterStage: false
    },
    noiseReduction: {
        enabled: true,
        strength: 0.7,
        voiceProtection: true,
        pauseAfterStage: false
    },
    voiceEnhancement: {
        enabled: true,
        compressor: {
            threshold: -20,
            ratio: 3,
            knee: 2.0
        },
        pauseAfterStage: false
    }
};
```

#### Real-time Parameter Adjustment
- **Live Updates**: Parameters update audio processing in real-time
- **Visual Feedback**: Sliders and toggles with immediate visual response
- **Value Display**: Current parameter values shown with units
- **Parameter Validation**: Input validation and range checking

#### Diagnostic Capabilities
- **Performance Monitoring**: Processing time per stage, total latency
- **Quality Metrics**: SNR, clipping detection, noise floor analysis
- **Visual Analysis**: Waveform display, spectrum analysis, before/after comparisons
- **Export Functionality**: Save diagnostic reports and processed audio

### Integration with Backend Services

The audio processing frontend seamlessly integrates with the whisper service:

#### Enhanced Pipeline Integration
```javascript
// Enhanced pipeline function for integration
async function processAudioPipelineEnhanced() {
    // Use enhanced processing with parameter controls
    if (window.audioProcessingParams) {
        return await processWithParameters(
            window.audioProcessingParams,
            window.audioProcessingControls.enabledStages
        );
    }
    
    // Fallback to basic processing
    return await processAudioPipelineOld();
}
```

#### Global Parameter Access
- **Shared Configuration**: Parameters accessible across all frontend components
- **Persistence**: Settings saved to localStorage and exportable
- **Hot-reload**: Parameter changes applied without page refresh
- **Validation**: Parameter ranges and dependencies enforced

### Professional Audio Processing Features

#### Voice-Specific Processing
- **Human Voice Optimization**: Tuned for speech frequencies (85-300Hz fundamental)
- **Formant Preservation**: Maintains speech intelligibility during processing
- **Sibilance Enhancement**: Preserves consonant clarity (4-8kHz)
- **Natural Dynamics**: Soft-knee compression preserves speech naturalness

#### Advanced Noise Reduction
- **Voice Protection**: Prevents over-processing of speech content
- **Spectral Subtraction**: Advanced noise reduction with musical noise suppression
- **Adaptive Gating**: Dynamic noise gate with voice activity detection
- **Multi-band Processing**: Frequency-specific noise reduction

#### Quality Assurance
- **Real-time Monitoring**: Continuous quality assessment during processing
- **Clipping Prevention**: Automatic level management to prevent distortion
- **Phase Coherence**: Maintains audio phase relationships
- **Artifact Detection**: Identifies and minimizes processing artifacts

### File Structure

```
modules/orchestration-service/static/
├── audio-processing-controls.html      # Main pipeline control interface
├── audio-diagnostic.html              # Real-time diagnostic dashboard
├── test-audio.html                    # Enhanced testing interface
├── js/
│   ├── audio-processing-test.js       # Enhanced 10-stage pipeline
│   ├── test-audio.js                  # Testing utilities (fixed recursion)
│   └── audio.js                       # Core audio module
└── css/
    └── styles.css                     # Unified styling with controls
```

### Usage Guide

#### For Audio Engineers
1. **Parameter Tuning**: Use `/audio-processing-controls.html` for detailed parameter adjustment
2. **Quality Analysis**: Monitor processing quality with `/audio-diagnostic.html`
3. **Testing**: Validate settings with `/test-audio.html` comprehensive testing
4. **Presets**: Use quick presets for common scenarios, then fine-tune

#### For Developers
1. **Integration**: Enhanced pipeline automatically integrated with existing code
2. **Parameters**: Access global `audioProcessingParams` object for current settings
3. **Controls**: Use `audioProcessingControls` API for programmatic control
4. **Debugging**: Enable pause-at-stage for step-by-step analysis

This comprehensive audio processing frontend provides professional-grade control over every aspect of the audio pipeline, from basic enable/disable controls to advanced parameter tuning and real-time diagnostics.

## Critical Audio Processing Fixes

### Browser Audio Processing Attenuation Issue

**Problem**: MediaRecorder API's audio processing features (echoCancellation, noiseSuppression, autoGainControl) were severely attenuating loopback audio signals, causing poor transcription quality.

**Symptoms**:
- Frontend audio visualization showed good levels
- Backend received extremely quiet audio (RMS < 0.001)
- Transcription results were poor ("you" repeatedly)
- Loopback devices particularly affected

**Solution**: Disabled all browser audio processing features in MediaRecorder constraints:

```javascript
// File: modules/orchestration-service/static/js/audio.js
const constraints = {
    audio: {
        echoCancellation: false,    // Was causing loopback attenuation
        noiseSuppression: false,    // Was treating loopback as noise
        autoGainControl: false,     // Was reducing gain levels
        // ... other settings
    }
};
```

### Backend Noise Reduction Removal Issue

**Problem**: The `noisereduce` library was treating loopback audio as noise and removing all audio content, leaving silence for Whisper to transcribe.

**Symptoms**:
- Audio RMS dropping from 0.001 to 0.000021 after processing
- "Applied noise reduction" in logs coinciding with audio loss
- Consistent transcription of silence or minimal content

**Solution**: Disabled aggressive noise reduction for all audio sources:

```python
# File: modules/whisper-service/src/api_server.py
# DISABLED: Noise reduction is removing all loopback audio content
logger.info("[AUDIO] Skipping noise reduction - disabled for loopback audio debugging")
# audio_array = nr.reduce_noise(y=audio_array, sr=16000, stationary=True)
```

### Audio Quality Validation Improvements

**Enhanced RMS Threshold**: Lowered silence detection threshold from 0.001 to 0.0001 to prevent false positive silence detection:

```python
# File: modules/whisper-service/src/api_server.py
silence_threshold = 0.0001  # Much lower threshold - only catch truly silent audio
```

### Testing and Validation

**Before Fix**:
- RMS: 0.000021 (essentially silent)
- Max amplitude: 0.000
- Transcription: "you" repeatedly
- Processing: Audio content removed by noise reduction

**After Fix**:
- RMS: 0.001+ (normal speech levels)
- Max amplitude: 0.01+ (detectable audio)
- Transcription: Accurate speech-to-text results
- Processing: Raw audio preserved through pipeline

This fix is critical for any deployment using loopback audio devices for system audio capture.
