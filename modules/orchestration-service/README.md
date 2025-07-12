# Orchestration Service - Backend API & Service Coordination

**Hardware Target**: CPU (optimized for high I/O and concurrent connections)

## 🚀 Latest Enhancements - Centralized Audio Chunking System

### ✅ **Centralized Audio Processing Pipeline** - NEW!
- **AudioCoordinator**: Central coordination for all audio processing with bot_sessions database integration
- **ChunkManager**: Advanced audio chunking with overlap handling and database persistence  
- **Configuration System**: Hot-reloadable audio processing parameters with chunking, correlation, database settings
- **Audio Pipeline Integration**: Complete VAD, noise reduction, voice enhancement, compression with persistent settings
- **Comprehensive Testing**: Unit, integration, E2E, performance tests with I/O validation across all components

### ✅ **Advanced Speaker & Timing Coordination** - NEW!
- **SpeakerCorrelationManager**: Links whisper speakers with Google Meet speakers, stores correlations in bot_sessions.correlations table
- **TimingCoordinator**: Timestamp alignment and database correlation across all bot_sessions tables for precise time synchronization
- **Database Integration**: Specialized CRUD operations for audio processing tables with connection pooling
- **Enhanced Data Models**: Pydantic models for AudioChunkMetadata, SpeakerCorrelation, ProcessingResult with comprehensive validation

### ✅ **Enterprise Audio Architecture** - COMPLETED!
- **Centralized Chunking**: x seconds with overlap handled by orchestration service (no more scattered chunking logic)
- **Bot Audio Streaming**: Real-time WebSocket or n-second chunks streamed to orchestration coordinator
- **Database Persistence**: All audio data tagged and tracked through existing bot_sessions PostgreSQL schema
- **Speaker-labeled Transcripts**: Complete correlation tracking between whisper and Google Meet speakers
- **Quality Monitoring**: Real-time audio quality assessment and processing pipeline optimization

## Service Overview

The Orchestration Service is a CPU-optimized backend microservice that provides:
- **FastAPI Backend**: Modern async/await API with automatic OpenAPI documentation
- **🆕 Centralized Audio Processing**: AudioCoordinator with chunking, speaker correlation, and timing alignment
- **🆕 Advanced Database Integration**: Specialized bot_sessions operations with connection pooling and persistence
- **WebSocket Management**: Enterprise-grade real-time communication with connection pooling
- **Service Coordination**: Health monitoring, auto-recovery, and intelligent routing
- **Session Management**: Multi-client session handling with persistence and recovery
- **API Gateway**: Load balancing and request routing with circuit breaking
- **Monitoring Dashboard**: Real-time performance analytics and system health monitoring
- **Enterprise Monitoring Stack**: Integrated Prometheus, Grafana, AlertManager, Loki
- **🆕 Google Meet Bot Management**: Complete bot lifecycle with audio capture and virtual webcam generation

**Note**: Frontend UI has been moved to `modules/frontend-service/` for clean separation of concerns

## 🚀 Quick Start

### Method 1: Complete Development Environment (Recommended)

```bash
# Start backend service
cd modules/orchestration-service
./start-backend.ps1  # FastAPI backend (port 3000)

# Start frontend service (separate module)
cd modules/frontend-service
./start-frontend.ps1  # React frontend (port 5173)

# Or use Docker
docker-compose up -d

# Check services
curl http://localhost:3000/api/health     # Orchestration backend
curl http://localhost:3000/docs           # API documentation
curl http://localhost:3000/api/audio/models  # Dynamic models API (NEW)
open http://localhost:5173                # Frontend service
```

### Method 2: Legacy Flask (Maintenance Mode)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export WHISPER_SERVICE_URL=http://localhost:5001
export TRANSLATION_SERVICE_URL=http://localhost:5003

# Start legacy service
python src/orchestration_service.py --legacy

# Check health
curl http://localhost:3000/api/health
```

### Method 3: Production Deployment

```bash
# Build production images
docker build -f Dockerfile.react -t orchestration-service:latest .

# Deploy with monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Run with production settings
docker run -d \
  --name orchestration-service \
  -p 3000:3000 \
  -e SECRET_KEY=your-production-secret \
  -e LOG_LEVEL=INFO \
  orchestration-service:latest
```

## 📊 Service Integration Dashboard

**Frontend Service**: http://localhost:5173 (Meeting Test Dashboard, Audio Testing)
**Backend API**: http://localhost:3000 (Dynamic Models, Device Information)
**API Documentation**: http://localhost:3000/docs (Complete API reference)

### 🎯 Core Features

#### **🎤 Transcription Panel**
- Real-time audio transcription with Whisper models
- Speaker diarization with timeline visualization
- Multiple audio format support (WAV, MP3, WebM, etc.)
- Voice activity detection and audio level monitoring

#### **🌍 Translation Panel**
- Live translation in 10+ languages (English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese)
- Automatic transcription-to-translation pipeline
- Confidence scoring and quality metrics
- Language swapping and translation history

#### **🔍 Service Health Monitor**
- Real-time status of all backend services (Audio, Translation)
- Response time monitoring and uptime tracking
- Service failure detection and recovery alerts
- System-wide health summaries

#### **🔗 WebSocket Connection Manager**
- Live connection pool monitoring (supports 10,000+ concurrent connections)
- Connection details with IP addresses and user agents
- Message rate tracking and connection statistics
- Peak connection monitoring

#### **🌐 API Gateway Dashboard**
- Request rate monitoring and response time analytics
- Success rate tracking and error monitoring
- Circuit breaker status and failure protection
- Service routing and load balancing metrics

#### **📋 Activity Logs**
- Real-time system activity logging
- Service operation tracking
- Error reporting and debugging information
- Performance insights and alerts

### 🎨 Frontend Architecture

The dashboard is built with:
- **Modern JavaScript**: Modular ES6+ architecture with real-time WebSocket integration
- **Responsive CSS**: Beautiful dark theme with orchestration-focused design
- **Component-based**: Translation, health monitoring, connection management, and analytics modules
- **Real-time Updates**: Auto-refreshing health monitoring and live connection statistics
- **API Integration**: All requests routed through orchestration service gateway

## 🎵 Centralized Audio Processing Architecture

### Core Audio Processing Modules

#### **AudioCoordinator** (`src/audio/audio_coordinator.py`)
Central coordination class that integrates all audio processing components:
- **Session Management**: Per-session audio processing configuration and state
- **Database Integration**: Persistent storage of audio chunks, metadata, and processing results
- **Service Communication**: Coordination with whisper and translation services
- **Quality Monitoring**: Real-time audio quality assessment and optimization
- **Configuration Management**: Hot-reloadable processing parameters

#### **ChunkManager** (`src/audio/chunk_manager.py`) 
Intelligent audio chunking with overlap management and database persistence:
- **AudioBuffer**: Rolling buffer with configurable duration and overlap
- **ChunkFileManager**: Efficient file storage with compression and cleanup
- **AudioQualityAnalyzer**: Real-time quality assessment and validation
- **Overlap Processing**: Intelligent overlap handling to prevent audio loss

#### **SpeakerCorrelationManager** (`src/audio/speaker_correlator.py`)
Links whisper-detected speakers with external sources (Google Meet, manual input):
- **Manual Mapping Support**: For testing scenarios with loopback audio
- **Google Meet API Integration**: Automatic speaker correlation via official API
- **Fallback Mechanisms**: Graceful handling when external data unavailable
- **Database Persistence**: Store correlations in bot_sessions.correlations table

#### **TimingCoordinator** (`src/audio/timing_coordinator.py`)
Timestamp alignment and correlation across all bot_sessions tables:
- **Cross-table Correlation**: Links audio chunks with transcripts, translations, speaker data
- **Time Drift Detection**: Automatic detection and correction of timing inconsistencies
- **Synchronization Quality**: Comprehensive quality scoring and monitoring
- **Database Correlation**: Precise timing correlation tracking

#### **DatabaseAdapter** (`src/audio/database_adapter.py`)
Specialized database operations for audio processing pipeline:
- **Connection Pooling**: Efficient database connection management
- **Optimized CRUD**: High-performance operations for audio processing tables
- **Batch Operations**: Efficient bulk data processing
- **Transaction Management**: ACID compliance for audio data operations

#### **Configuration System** (`src/audio/config.py`)
Hot-reloadable configuration management:
- **YAML Persistence**: Configuration stored in human-readable format
- **Built-in Presets**: Optimized configurations for different scenarios
- **Parameter Validation**: Comprehensive validation with dependency checking
- **Frontend Integration**: Configuration API for UI customization

### Enhanced Data Models (`src/audio/models.py`)

Comprehensive Pydantic models with validation:
- **AudioChunkMetadata**: Complete chunk information with database mapping
- **SpeakerCorrelation**: Speaker mapping between whisper and external sources
- **ProcessingResult**: Unified result format for all processing stages
- **AudioChunkingConfig**: Configuration parameters for chunking system
- **QualityMetrics**: Audio quality assessment metrics

## 🔌 API Endpoints

### Health and Status
```http
GET /api/health              # Orchestration service health check
GET /api/services            # Backend service status monitoring
GET /api/metrics             # Performance metrics collection
GET /api/dashboard           # Dashboard data aggregation
```

### 🆕 Centralized Audio Processing API
```http
GET /api/audio/models        # ENHANCED - Aggregated models + device info from all services
POST /api/audio/upload       # Audio processing with dynamic configuration
POST /api/audio/chunk        # NEW - Centralized audio chunking with overlap handling
GET /api/audio/config        # NEW - Audio processing configuration management
POST /api/audio/config       # NEW - Update audio processing parameters
GET /api/audio/correlations  # NEW - Speaker correlation management
POST /api/audio/correlations # NEW - Create/update speaker correlations
GET /api/audio/timing        # NEW - Timing coordination and alignment
POST /api/audio/timing       # NEW - Timestamp alignment operations
```

### Configuration
```http
GET /api/config              # Frontend configuration
POST /api/config             # Update configuration
GET /api/settings            # Application settings
POST /api/settings           # Update settings
```

### Service Gateway
```http
GET|POST /api/whisper/*      # Proxy to Whisper service (replaced by /api/audio/*)
GET|POST /api/audio/*        # Proxy to Audio/Whisper service with device info
GET|POST /api/translation/*  # Proxy to Translation service with device monitoring
```

## 🔄 WebSocket API

Connect to real-time WebSocket at `ws://localhost:3000`

### Connection Events
```javascript
// Basic connection
socket.emit('connect');
socket.on('connected', (data) => console.log('Connected:', data));

// Subscribe to updates
socket.emit('subscribe', { service: 'whisper' });
socket.on('subscribed', (data) => console.log('Subscribed:', data));
```

### Service Communication
```javascript
// Service requests
socket.emit('service_message', {
  type: 'service_request',
  target_service: 'whisper',
  action: 'get_models'
});

// Real-time streams
socket.emit('transcription_stream', { audio_data: '...' });
socket.emit('speaker_stream', { audio_data: '...' });
socket.emit('translation_stream', { text: 'Hello world', target_lang: 'es' });
```

### Session Management
```javascript
// Join session
socket.emit('service_message', {
  type: 'join_session',
  session_id: 'my-session-123'
});

// Leave session
socket.emit('service_message', {
  type: 'leave_session'
});
```

## ⚙️ Configuration

### Environment Variables

```bash
# === Core Service Configuration ===
HOST=0.0.0.0                           # Bind host
PORT=3000                              # Service port
SECRET_KEY=change-me-in-production     # Flask secret key
LOG_LEVEL=INFO                         # Logging level

# === Backend Services ===
AUDIO_SERVICE_URL=http://localhost:5001      # Audio/Whisper service (NPU optimized)
TRANSLATION_SERVICE_URL=http://localhost:5003 # Translation service (GPU optimized)

# Service Discovery
SERVICE_DISCOVERY_ENABLED=true
SERVICE_TIMEOUT=30
SERVICE_RETRY_ATTEMPTS=3

# === WebSocket Configuration ===
WEBSOCKET_MAX_CONNECTIONS=10000        # Max concurrent connections
WEBSOCKET_TIMEOUT=1800                 # Connection timeout (30 min)

# === Monitoring ===
HEALTH_CHECK_INTERVAL=10               # Health check interval (seconds)
```

### Configuration File

Create `config/orchestration.yaml`:

```yaml
orchestration:
  service_name: "orchestration-service"
  version: "2.0.0"

frontend:
  host: "0.0.0.0"
  port: 3000
  workers: 4

websocket:
  max_connections: 10000
  connection_timeout: 1800
  heartbeat_interval: 30

gateway:
  timeout: 30
  retries: 3
  circuit_breaker_threshold: 5

monitoring:
  health_check_interval: 10
  auto_recovery: true

services:
  whisper:
    url: "http://localhost:5001"
    health_endpoint: "/health"
  translation:
    url: "http://localhost:5003"  
    health_endpoint: "/api/health"
```

## 🏗️ Architecture

### Service Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Service                       │
│                      [CPU OPTIMIZED]                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Frontend    │↔ │ WebSocket   │↔ │ API Gateway │↔ │Backend  │ │
│  │ Dashboard   │  │ Manager     │  │ Routing     │  │Services │ │
│  │ Static      │  │ Sessions    │  │ Balancing   │  │Health   │ │
│  │ Templates   │  │ Heartbeat   │  │ Fallback    │  │Monitor  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│                           ↓                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Monitor     │← │ Analytics   │← │ Session     │← │ Config  │ │
│  │ Health      │  │ Dashboard   │  │ Manager     │  │ Manager │ │
│  │ Alerts      │  │ Metrics     │  │ Recovery    │  │ Hot     │ │
│  │ Recovery    │  │ Trends      │  │ Persist     │  │ Reload  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Enterprise Features

- **Connection Pooling**: Up to 10,000 concurrent WebSocket connections
- **Circuit Breaker**: Automatic service failure protection  
- **Load Balancing**: Weighted round-robin with health-based routing
- **Session Persistence**: Redis-backed session recovery
- **Health Monitoring**: Automatic service health checks and recovery
- **Real-time Analytics**: Live performance metrics and dashboards
- **Auto-scaling**: Connection and request throttling

## 🔍 Monitoring & Observability

### Health Checks

```bash
# Service health
curl http://localhost:3000/api/health

# Backend services
curl http://localhost:3000/api/services

# Performance metrics  
curl http://localhost:3000/api/metrics
```

### Real-time Metrics

The dashboard provides real-time monitoring of:

- **System Health**: Overall service status and uptime
- **WebSocket Connections**: Active connections, sessions, message throughput
- **API Gateway**: Request rates, response times, error rates
- **Backend Services**: Health status, response times, availability
- **Performance Trends**: Historical metrics and performance analysis

### Alerts & Recovery

- **Automatic Health Monitoring**: Continuous health checks every 10 seconds
- **Alert Generation**: Warning and critical alerts for service issues
- **Auto-recovery**: Automatic service recovery attempts
- **Circuit Breaking**: Protect against cascading failures

## 🧪 Comprehensive Testing Framework

### Audio Processing Component Tests

#### **Unit Tests** (`tests/audio/unit/`)
Comprehensive unit testing for all audio processing components:

```bash
# Test AudioCoordinator functionality
python tests/audio/unit/test_audio_coordinator.py

# Test ChunkManager with overlap handling
python tests/audio/unit/test_chunk_manager.py

# Test SpeakerCorrelationManager (26 tests passing)
python tests/audio/unit/test_speaker_correlator.py

# Test TimingCoordinator (22 tests passing)
python tests/audio/unit/test_timing_coordinator.py

# Test DatabaseAdapter operations
python tests/audio/unit/test_database_adapter.py

# Test Configuration system
python tests/audio/unit/test_audio_config.py

# Test Audio Models validation
python tests/audio/unit/test_audio_models.py
```

#### **Integration Tests** (`tests/audio/integration/`)
End-to-end testing of audio processing pipeline:

```bash
# Test complete audio processing flow
python tests/audio/integration/test_audio_pipeline.py

# Test database integration with PostgreSQL
python tests/audio/integration/test_database_integration.py

# Test service communication
python tests/audio/integration/test_service_integration.py

# Test WebSocket audio streaming
python tests/audio/integration/test_websocket_audio.py
```

#### **Performance Tests** (`tests/audio/performance/`)
Performance validation and benchmarking:

```bash
# Test audio processing latency
python tests/audio/performance/test_processing_latency.py

# Test concurrent chunking performance
python tests/audio/performance/test_chunking_performance.py

# Test database operation performance
python tests/audio/performance/test_database_performance.py

# Test memory usage under load
python tests/audio/performance/test_memory_usage.py
```

#### **Edge Case Tests** (`tests/audio/edge_cases/`)
Comprehensive edge case validation:

```bash
# Test audio format variations
python tests/audio/edge_cases/test_audio_formats.py

# Test network failure scenarios
python tests/audio/edge_cases/test_network_failures.py

# Test database transaction failures
python tests/audio/edge_cases/test_transaction_failures.py

# Test configuration edge cases
python tests/audio/edge_cases/test_config_edge_cases.py
```

### Testing Results Summary

#### ✅ **SpeakerCorrelationManager** - All 26 Tests Passing
- Manual speaker mapping functionality
- Loopback audio configuration
- Google Meet API integration
- Fallback correlation mechanisms
- Database persistence and caching
- Statistics and monitoring
- Session management and cleanup

#### ✅ **TimingCoordinator** - All 22 Tests Passing  
- Time window overlap detection
- Audio chunk timestamp correlation
- Time drift detection and correction
- Timestamp alignment across tables
- Synchronized data retrieval
- Statistics and monitoring
- Session timing management

#### ✅ **Audio Models** - Comprehensive Validation
- Pydantic model validation
- Factory function testing
- Serialization/deserialization
- Configuration loading
- Error handling

### Service Testing

```bash
# Test all services
curl http://localhost:3000/api/services

# Test specific service via gateway
curl http://localhost:3000/api/whisper/health
curl http://localhost:3000/api/translation/api/health

# Test new audio processing endpoints
curl http://localhost:3000/api/audio/config
curl http://localhost:3000/api/audio/correlations
curl http://localhost:3000/api/audio/timing
```

### Audio Processing Pipeline Testing

#### **Manual Testing Interface**
Access comprehensive testing interfaces:

```bash
# Audio processing controls
open http://localhost:3000/audio-processing-controls.html

# Real-time diagnostics
open http://localhost:3000/audio-diagnostic.html

# Enhanced audio testing
open http://localhost:3000/test-audio.html
```

#### **Automated Test Suite Execution**
```bash
# Run all audio processing tests
cd modules/orchestration-service
python -m pytest tests/audio/ -v --tb=short

# Run specific test categories
python -m pytest tests/audio/unit/ -v
python -m pytest tests/audio/integration/ -v
python -m pytest tests/audio/performance/ -v

# Generate test coverage report
python -m pytest tests/audio/ --cov=src/audio --cov-report=html
```

#### **Performance Benchmarking**
```bash
# Audio processing latency benchmark
python tests/audio/performance/benchmark_latency.py --samples=1000

# Chunking performance under load
python tests/audio/performance/benchmark_chunking.py --concurrent=50

# Database operations benchmark
python tests/audio/performance/benchmark_database.py --operations=10000
```

### WebSocket Testing

Use the built-in WebSocket test page at `/websocket-test` or:

```javascript
const socket = io('ws://localhost:3000');
socket.on('connected', data => console.log('Connected:', data));
socket.emit('service_message', { type: 'heartbeat' });
```

### Load Testing

```bash
# Install dependencies
pip install websocket-client requests

# Run load test
python tests/load_test.py --connections 1000 --duration 300
```

## 🐛 Troubleshooting

### Common Issues

1. **Service not responding**
   - Check backend service URLs in environment variables
   - Verify services are running: `docker ps` or `curl service_url/health`

2. **WebSocket connection failures**  
   - Check firewall settings for port 3000
   - Verify CORS settings in browser console

3. **High memory usage**
   - Reduce `WEBSOCKET_MAX_CONNECTIONS`
   - Check for connection leaks in browser dev tools

4. **Dashboard not loading**
   - Verify static files are properly copied
   - Check browser console for JavaScript errors

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/orchestration_service.py --debug

# View detailed logs
tail -f logs/orchestration.log
```

### Performance Tuning

```bash
# Increase connection limits
export WEBSOCKET_MAX_CONNECTIONS=20000

# Reduce health check frequency  
export HEALTH_CHECK_INTERVAL=30

# Optimize for high throughput
export GATEWAY_TIMEOUT=60
export WEBSOCKET_TIMEOUT=3600
```

## 📈 Production Deployment

### Production Checklist

- [ ] Set unique `SECRET_KEY`
- [ ] Configure proper `LOG_LEVEL` (INFO or WARNING)
- [ ] Set up SSL/TLS termination
- [ ] Configure proper CORS origins
- [ ] Set up log aggregation
- [ ] Configure monitoring alerts
- [ ] Set resource limits in Docker
- [ ] Configure backup for session data

### Security Considerations

- Use strong secret keys for session management
- Configure proper CORS origins (not `*` in production)
- Set up rate limiting for API endpoints
- Use SSL/TLS for all connections
- Validate all input data
- Monitor for suspicious connection patterns

### Scaling

The orchestration service is designed to be CPU-optimized and can handle:

- **10,000+ concurrent WebSocket connections**
- **1000+ requests per second**
- **Multi-instance deployment** behind a load balancer
- **Horizontal scaling** with session persistence

For higher loads, consider:
- Load balancer with sticky sessions
- Redis cluster for session storage  
- Multiple orchestration service instances
- CDN for static assets