# Whisper Service - NPU Optimized Audio Processing

**Hardware Target**: NPU (Intel NPU), GPU, CPU fallback - **NPU OPTIMIZED**

## Service Overview

The Whisper Service is an NPU-optimized microservice that provides:
- **Real-time Speech-to-Text**: Advanced Whisper model inference with OpenVINO optimization
- **NPU Hardware Acceleration**: Intel NPU detection and automatic fallback (NPU → GPU → CPU)
- **Speaker Diarization**: Multi-speaker identification and timeline tracking
- **Voice Activity Detection**: Real-time VAD with WebRTC and Silero integration
- **Enterprise WebSocket Infrastructure**: Production-ready real-time streaming
- **Audio Processing**: Multi-format support with format detection and conversion

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Whisper Service                         │
│                      [NPU OPTIMIZED]                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ NPU Model   │↔ │ Audio Proc  │↔ │ Speaker     │↔ │ VAD     │ │
│  │ • OpenVINO  │  │ • Multi-fmt │  │ • Diarize   │  │ • WebRTC│ │
│  │ • Fallback  │  │ • Convert   │  │ • Track     │  │ • Silero│ │
│  │ • Auto      │  │ • Enhance   │  │ • Timeline  │  │ • Real  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│                           ↓                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Rolling     │← │ WebSocket   │← │ API Server  │← │ Health  │ │
│  │ Buffer      │  │ • Real-time │  │ • REST      │  │ Monitor │ │
│  │ • Stream    │  │ • Sessions  │  │ • Upload    │  │ • NPU   │ │
│  │ • VAD       │  │ • Events    │  │ • Stream    │  │ • Status│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                   Enterprise Infrastructure                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Connection  │↔ │ Error       │↔ │ Message     │↔ │ Auth    │ │
│  │ • Pool      │  │ • Recovery  │  │ • Route     │  │ • Simple│ │
│  │ • Heartbeat │  │ • Fallback  │  │ • Priority  │  │ • Roles │ │
│  │ • Weak Ref  │  │ • Cleanup   │  │ • Queue     │  │ • Users │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Current Status

### ✅ FULLY COMPLETED - Production Ready Whisper Service with NPU Optimization

#### Core Audio Processing
- **Model Manager** → ✅ **NPU Detection** with automatic hardware fallback (NPU → GPU → CPU)
- **Audio Processor** → ✅ **Multi-format Support** (WAV, MP3, WebM, OGG, MP4) with streaming detection
- **Rolling Buffer** → ✅ **Real-time Processing** with VAD integration and memory optimization
- **Speaker Diarization** → ✅ **Advanced Identification** with embedding methods and clustering

#### API Infrastructure
- **REST API Server** → ✅ **Flask-based** with comprehensive endpoint coverage
- **WebSocket Support** → ✅ **Enterprise Infrastructure** with connection pooling and heartbeat
- **Health Monitoring** → ✅ **NPU Status** detection and service health reporting
- **Error Handling** → ✅ **20+ Error Categories** with automatic recovery mechanisms

#### Integration & Deployment
- **Orchestration Integration** → ✅ **API Gateway** routing through `/api/whisper/*` endpoints
- **Frontend Integration** → ✅ **Dashboard Support** with real-time transcription display
- **Docker Support** → ✅ **NPU-optimized** containers with Intel NPU detection
- **Configuration Management** → ✅ **Environment Variables** and hot-reloadable settings

## Hardware Optimization

### NPU Detection and Fallback

```python
def _detect_best_device(self) -> str:
    """Detect the best available device for inference"""
    try:
        # Check environment variable first
        env_device = os.getenv("OPENVINO_DEVICE")
        if env_device:
            return env_device
        
        # Auto-detect available devices
        core = ov.Core()
        available_devices = core.available_devices
        
        # Prefer NPU, then GPU, then CPU
        if "NPU" in available_devices:
            logger.info("✓ NPU detected! Using NPU for inference.")
            return "NPU"
        elif "GPU" in available_devices:
            logger.info("⚠ NPU not found, using GPU fallback.")
            return "GPU"
        else:
            logger.info("⚠ NPU/GPU not found, using CPU fallback.")
            return "CPU"
    except Exception as e:
        logger.error(f"Error detecting devices: {e}")
        return "CPU"
```

### OpenVINO Model Loading

```python
def load_model(self, model_name: str, language: str = None) -> bool:
    """Load OpenVINO optimized Whisper model"""
    try:
        device = self._detect_best_device()
        model_path = self._get_model_path(model_name)
        
        # Load with device-specific optimization
        self.pipe = ov_genai.WhisperPipeline(model_path, device)
        
        # Configure for optimal performance
        generation_config = ov_genai.WhisperGenerationConfig()
        generation_config.language = language or "<|en|>"
        generation_config.task = "transcribe"
        generation_config.return_timestamps = True
        
        self.current_model = model_name
        self.device = device
        
        logger.info(f"✅ Model {model_name} loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return False
```

## Enterprise WebSocket Infrastructure

### Connection Management

The service implements enterprise-grade WebSocket features:

- **Connection Pooling**: 1000 capacity with weak reference tracking
- **20+ Error Categories**: Comprehensive error handling with automatic recovery
- **Heartbeat Monitoring**: RTT tracking and connection health
- **Session Persistence**: 30-minute timeout with recovery mechanisms
- **Message Buffering**: Zero-message-loss design with priority queuing

### Advanced Features

```python
class EnterpriseConnectionManager:
    """Enterprise-grade WebSocket connection management"""
    
    def __init__(self, max_connections: int = 1000):
        self.connection_pool = weakref.WeakValueDictionary()
        self.session_store = {}
        self.heartbeat_manager = HeartbeatManager()
        self.error_handler = ErrorHandler()
        self.message_router = MessageRouter()
        self.auth_manager = SimpleAuth()
        self.reconnection_manager = ReconnectionManager()
```

## Audio Processing Pipeline

### Multi-Format Support

```python
class AudioProcessor:
    """Advanced audio processing with multi-format support"""
    
    SUPPORTED_FORMATS = {
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg',
        'webm': 'audio/webm',
        'ogg': 'audio/ogg',
        'm4a': 'audio/mp4',
        'mp4': 'audio/mp4',
        'flac': 'audio/flac'
    }
    
    def process_audio(self, audio_data: bytes, format_hint: str = None) -> np.ndarray:
        """Process audio with automatic format detection"""
        # Detect format if not provided
        detected_format = self._detect_audio_format(audio_data)
        
        # Convert to standard format
        audio_array = self._convert_audio(audio_data, detected_format)
        
        # Apply preprocessing
        audio_array = self._preprocess_audio(audio_array)
        
        return audio_array
```

### Real-Time Streaming

```python
class RollingBufferManager:
    """Memory-efficient real-time audio buffering"""
    
    def __init__(self, buffer_duration: float = 30.0, chunk_duration: float = 1.0):
        self.buffer_duration = buffer_duration
        self.chunk_duration = chunk_duration
        self.audio_buffer = collections.deque(maxlen=int(buffer_duration / chunk_duration))
        self.vad = WebRTCVAD()
        
    def add_audio_chunk(self, audio_data: np.ndarray) -> bool:
        """Add audio chunk with VAD processing"""
        # Voice activity detection
        has_speech = self.vad.is_speech(audio_data)
        
        if has_speech:
            self.audio_buffer.append(audio_data)
            return True
        
        return False
```

## API Endpoints

### REST API

```http
# Health and status
GET /health                        # Service health check
GET /models                        # Available models
GET /device-info                   # NPU/GPU/CPU status

# Transcription
POST /transcribe                   # Transcribe audio (auto model)
POST /transcribe/{model_name}      # Transcribe with specific model
POST /transcribe/stream            # Start streaming session

# Management
POST /load-model                   # Load specific model
POST /clear-cache                  # Clear model cache
GET /settings                      # Current settings
POST /settings                     # Update settings
```

### WebSocket Events

```javascript
// Connection management
socket.emit('connect');
socket.emit('join_session', { session_id: 'session-123' });
socket.emit('leave_session');

// Real-time transcription
socket.emit('transcribe_stream', { audio_data: base64_audio });
socket.on('transcription_result', (data) => {
    console.log('Transcription:', data.text);
    console.log('Confidence:', data.confidence);
    console.log('Speakers:', data.speakers);
});

// Voice activity detection
socket.on('voice_activity', (data) => {
    console.log('Speech detected:', data.has_speech);
    console.log('Audio level:', data.level);
});
```

## Speaker Diarization

### Advanced Speaker Identification

```python
class SpeakerDiarization:
    """Advanced speaker identification and tracking"""
    
    def __init__(self):
        self.embedding_methods = ['speechbrain', 'pyannote', 'resemblyzer']
        self.clustering_algorithms = ['hdbscan', 'dbscan', 'agglomerative']
        self.speaker_tracker = SpeakerTracker()
        
    def identify_speakers(self, audio: np.ndarray, embeddings_method: str = 'speechbrain') -> Dict:
        """Identify speakers with multiple embedding methods"""
        
        # Extract speaker embeddings
        embeddings = self._extract_embeddings(audio, embeddings_method)
        
        # Perform clustering
        speaker_labels = self._cluster_speakers(embeddings)
        
        # Track speaker continuity
        tracked_speakers = self.speaker_tracker.track_speakers(speaker_labels)
        
        return {
            'speakers': tracked_speakers,
            'timeline': self._create_speaker_timeline(tracked_speakers),
            'confidence': self._calculate_confidence(embeddings, speaker_labels)
        }
```

## Deployment

### Docker Deployment

```bash
# NPU-optimized deployment
docker build -t whisper-service:npu .
docker run -d \
  --name whisper-service \
  --device=/dev/accel/accel0 \
  -p 5001:5001 \
  -e OPENVINO_DEVICE=NPU \
  -e WHISPER_DEFAULT_MODEL=whisper-base \
  whisper-service:npu
```

### Environment Configuration

```bash
# Hardware configuration
OPENVINO_DEVICE=AUTO              # AUTO, NPU, GPU, CPU
WHISPER_DEFAULT_MODEL=whisper-base
WHISPER_MODELS_DIR=/models

# Service configuration
HOST=0.0.0.0
PORT=5001
LOG_LEVEL=INFO
DEBUG=false
WORKERS=1

# Audio processing
ENABLE_VAD=true
BUFFER_DURATION=30.0
CHUNK_DURATION=1.0
SAMPLE_RATE=16000

# WebSocket configuration
MAX_CONNECTIONS=1000
CONNECTION_TIMEOUT=1800
HEARTBEAT_INTERVAL=30

# Performance optimization
AUDIO_QUEUE_SIZE=100
MODEL_CACHE_SIZE=3
THREAD_POOL_SIZE=4
```

### Integration with Orchestration Service

The whisper service integrates seamlessly with the orchestration service through:

1. **API Gateway Routing**: All `/api/whisper/*` requests are proxied to the whisper service
2. **Health Monitoring**: Continuous health checks and status reporting
3. **WebSocket Coordination**: Real-time transcription through orchestration WebSocket
4. **Service Discovery**: Automatic registration and health reporting

## Testing

### Unit Tests

```python
# Test NPU detection
def test_npu_detection():
    model_manager = ModelManager()
    device = model_manager._detect_best_device()
    assert device in ['NPU', 'GPU', 'CPU']

# Test audio processing
def test_audio_format_detection():
    processor = AudioProcessor()
    wav_data = create_test_wav()
    format_detected = processor._detect_audio_format(wav_data)
    assert format_detected == 'wav'

# Test real-time streaming
async def test_streaming_transcription():
    service = WhisperService()
    await service.initialize()
    
    # Simulate audio chunks
    for chunk in create_audio_chunks():
        result = await service.transcribe_chunk(chunk)
        assert result.text is not None
```

### Integration Tests

```python
# Test orchestration integration
async def test_orchestration_integration():
    # Start whisper service
    whisper_service = await start_whisper_service()
    
    # Test health check through orchestration
    response = await test_client.get('/api/whisper/health')
    assert response.status_code == 200
    
    # Test transcription through orchestration
    audio_file = create_test_audio()
    response = await test_client.post('/api/whisper/transcribe', files={'audio': audio_file})
    data = response.json()
    assert 'text' in data
```

### Performance Tests

```python
# Test NPU performance
def test_npu_transcription_performance():
    service = WhisperService()
    service.load_model('whisper-base')
    
    audio = create_test_audio(duration=10)  # 10 seconds
    
    start_time = time.time()
    result = service.transcribe(audio)
    processing_time = time.time() - start_time
    
    # Should process faster than real-time on NPU
    assert processing_time < 10
    assert result.text is not None
```

## Legacy Migration

### Extracted from Legacy System

The whisper service was systematically extracted from the legacy monolithic system:

#### Legacy Issues Fixed
- ❌ Thread safety concerns → ✅ Proper locking mechanisms
- ❌ Memory leaks potential → ✅ Memory-efficient implementations  
- ❌ Basic error handling → ✅ Comprehensive error recovery
- ❌ No device fallback → ✅ Automatic NPU/GPU/CPU fallback
- ❌ Limited monitoring → ✅ Real-time performance monitoring
- ❌ Security vulnerabilities → ✅ Input validation and resource limits

#### Key Improvements Made
1. **Architecture**: Monolithic → Microservice with clean interfaces
2. **Performance**: NPU optimization with Intel NPU support
3. **Reliability**: 20+ error categories with automatic recovery
4. **Security**: Input validation, resource limits, path protection
5. **Monitoring**: Real-time metrics and health reporting

## Future Enhancements

### Planned Features
1. **Advanced NPU Optimization**: Model quantization and optimization
2. **Multi-Model Support**: Concurrent model loading for different languages
3. **Advanced VAD**: Deep learning-based voice activity detection
4. **Real-time Translation Integration**: Direct integration with translation service
5. **Performance Analytics**: Detailed performance profiling and optimization

### Scaling Considerations
- **Horizontal Scaling**: Multiple whisper service instances
- **Model Sharing**: Shared model cache across instances  
- **Load Balancing**: Request distribution based on model availability
- **Resource Management**: Dynamic CPU/NPU/GPU allocation

This NPU-optimized Whisper Service provides enterprise-grade speech-to-text capabilities with automatic hardware acceleration and comprehensive real-time processing features.

## Recent Enhancements

### Critical Audio Processing Fixes (Latest)

#### Browser Audio Processing Attenuation Fix
**Problem**: MediaRecorder API audio processing features were severely attenuating loopback audio
**Solution**: Disabled echoCancellation, noiseSuppression, and autoGainControl in frontend audio capture
**Impact**: Restored proper audio levels for loopback devices and system audio capture

#### Backend Noise Reduction Removal
**Problem**: `noisereduce` library was treating loopback audio as noise and removing all content
**Solution**: Disabled aggressive noise reduction that was destroying loopback audio signals
**Result**: Audio now preserves original content through entire processing pipeline

#### Enhanced Audio Quality Validation
- Lowered silence detection threshold from 0.001 to 0.0001
- Added comprehensive debug logging for audio levels and processing steps
- Fixed false positive silence detection that was blocking legitimate audio

### Audio Resampling Fix
Fixed critical audio resampling issue where `pydub.set_frame_rate()` wasn't properly resampling audio data, causing Whisper to produce incorrect transcriptions. Now includes librosa fallback for proper resampling from any sample rate to 16kHz.

### Voice-Specific Processing
Enhanced audio processing pipeline with voice-aware features:
- Voice frequency filtering (85-300Hz fundamental)
- Soft-knee compression for natural voice preservation
- Multi-stage processing with pause capability
- Comprehensive parameter tuning interface
- Real-time adjustments without restarting

### Test Audio Interface
Fixed and enhanced test audio page with:
- Working recording and playback controls
- Real-time parameter adjustment UI
- Step-by-step processing visualization
- Debugging capabilities with pause at each stage