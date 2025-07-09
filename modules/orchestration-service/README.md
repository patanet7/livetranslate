# LiveTranslate Orchestration Service

**🌍 Complete Real-time Transcription and Translation Platform**

The fully integrated orchestration service providing a comprehensive dashboard for:

- **🎤 Real-time Transcription**: Advanced Whisper-based speech-to-text with speaker diarization
- **🌍 Live Translation**: Multi-language bidirectional translation with confidence scoring
- **🔍 Service Health Monitoring**: Real-time backend service status and health analytics
- **🔗 WebSocket Management**: Enterprise-grade connection pool monitoring (10,000+ connections)
- **🌐 API Gateway**: Intelligent request routing with circuit breaker protection
- **📊 Performance Analytics**: Live system metrics and connection statistics

## 🚀 Quick Start

### Method 1: Docker (Recommended)

```bash
# Start orchestration service
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
curl http://localhost:3000/api/health
```

### Method 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export WHISPER_SERVICE_URL=http://localhost:5001
export TRANSLATION_SERVICE_URL=http://localhost:5003

# Start service
python src/orchestration_service.py

# Or with debug mode
python src/orchestration_service.py --debug
```

### Method 3: Production Deployment

```bash
# Build production image
docker build -t orchestration-service:latest .

# Run with production settings
docker run -d \
  --name orchestration-service \
  -p 3000:3000 \
  -e SECRET_KEY=your-production-secret \
  -e LOG_LEVEL=INFO \
  orchestration-service:latest
```

## 📊 LiveTranslate Dashboard

Access the complete orchestration dashboard at: **http://localhost:3000**

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

## 🔌 API Endpoints

### Health and Status
```http
GET /api/health              # Service health check
GET /api/services            # Backend service status  
GET /api/metrics             # Performance metrics
GET /api/dashboard           # Dashboard data
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
GET|POST /api/whisper/*      # Proxy to Whisper service
GET|POST /api/speaker/*      # Proxy to Speaker service  
GET|POST /api/translation/*  # Proxy to Translation service
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
WHISPER_SERVICE_URL=http://localhost:5001    # Whisper/Audio service
SPEAKER_SERVICE_URL=http://localhost:5002    # Speaker service  
TRANSLATION_SERVICE_URL=http://localhost:5003 # Translation service

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

## 🧪 Testing

### Service Testing

```bash
# Test all services
curl http://localhost:3000/api/services

# Test specific service via gateway
curl http://localhost:3000/api/whisper/health
curl http://localhost:3000/api/translation/api/health
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