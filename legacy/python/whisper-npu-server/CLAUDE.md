# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance speech transcription server that leverages Intel NPU acceleration with OpenVINO-optimized Whisper models. The system provides real-time speech-to-text transcription with advanced features including speaker diarization, voice activity detection, and a modern web frontend.

## Architecture

### Core Components

1. **Backend Server** (`server.py`) - Flask-based API server
   - Real-time transcription using OpenVINO Whisper models
   - NPU/GPU/CPU acceleration support with automatic fallback
   - Advanced speaker diarization with multiple clustering algorithms
   - Voice Activity Detection (VAD) with WebRTC and Silero
   - Session persistence and audio buffer management
   - RESTful API with health monitoring

2. **Frontend Interface** (`frontend/`) - Modern web UI
   - Real-time transcription display with timestamps
   - Model selection and audio device management
   - Comprehensive settings management interface
   - Activity logging and performance monitoring
   - Mobile-responsive design

3. **Speaker Diarization** (`speaker_diarization.py`) - Advanced audio processing
   - Multiple embedding methods (Resemblyzer, SpeechBrain, PyAnnote)
   - Clustering algorithms (HDBSCAN, DBSCAN, UMAP)
   - Speech enhancement and noise reduction
   - Punctuation alignment with transcription

## Development Commands

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start server directly
python server.py

# Start server with debug mode
python start-debug.py

# Start with NPU optimization
python start-native.py
```

### Docker Development

```bash
# Build and start NPU server
docker-compose -f docker-compose.npu.yml up --build

# Start with CPU fallback
docker-compose -f docker-compose.npu.yml --profile cpu-fallback up

# Start with frontend
docker-compose -f docker-compose.npu.yml --profile frontend up

# Start all services
docker-compose -f docker-compose.npu.yml --profile cpu-fallback --profile frontend up
```

### Windows Deployment

```powershell
# Start NPU server
.\start-windows.ps1

# Start with frontend
.\start-windows.ps1 -Frontend

# Start CPU fallback mode
.\start-windows.ps1 -Mode cpu

# Show logs
.\start-windows.ps1 -Logs

# Stop all services
.\start-windows.ps1 -Stop
```

### Testing

```bash
# Test transcription with curl
curl --data-binary @audio.wav -X POST http://localhost:8009/transcribe

# Test specific model
curl --data-binary @audio.wav -X POST http://localhost:8009/transcribe/whisper-medium.en

# Health check
curl http://localhost:8009/health

# List available models
curl http://localhost:8009/models
```

## Key Technical Details

### Model Management

- **Storage Location**: `~/.whisper/models/` (automatically created)
- **Supported Models**: whisper-tiny.en through whisper-large-v3
- **Format**: OpenVINO IR format optimized for NPU inference
- **Default Model**: whisper-medium.en (balanced speed/accuracy)

### Hardware Acceleration

- **Primary**: Intel NPU via OpenVINO (fastest)
- **Secondary**: GPU acceleration (CUDA/OpenCL)
- **Fallback**: CPU inference (most compatible)
- **Device Selection**: Automatic with manual override via `OPENVINO_DEVICE`

### API Endpoints

#### Core Transcription
- `POST /transcribe` - Transcribe with default model
- `POST /transcribe/{model_name}` - Transcribe with specific model
- `GET /models` - List available models
- `GET /health` - Server health status

#### Settings Management
- `GET /settings` - Get current configuration
- `POST /settings` - Update configuration
- `POST /restart` - Restart server with new settings
- `POST /shutdown` - Gracefully shutdown server

#### Speaker Diarization
- `POST /diarize` - Perform speaker diarization
- `GET /speaker-stats` - Get speaker statistics
- `POST /enhance-audio` - Audio enhancement

#### Session Management
- Session persistence with automatic cleanup
- Transcription history with configurable limits
- Buffer management for real-time processing

### Configuration

#### Environment Variables
```bash
OPENVINO_DEVICE=NPU          # Device preference (NPU/GPU/CPU)
OPENVINO_LOG_LEVEL=1         # OpenVINO logging level
PYTHONUNBUFFERED=1           # Unbuffered Python output
DOCKER_HOST_OS=windows       # Docker host detection
```

#### Model Configuration
- Default model selection
- Inference intervals for NPU protection
- Buffer duration (3-30 seconds)
- Sample rate options (16kHz, 44.1kHz, 48kHz)

#### Audio Processing
- Voice Activity Detection (WebRTC/Silero)
- Noise reduction and enhancement
- Real-time buffer management
- Multi-format support (WAV, MP3, FLAC, OGG)

#### Speaker Diarization Settings
- Number of speakers (auto-detect or fixed)
- Embedding methods (Resemblyzer, SpeechBrain, PyAnnote)
- Clustering algorithms (HDBSCAN, DBSCAN, UMAP)
- Speech enhancement controls

## File Structure

```
whisper-npu-server/
├── server.py                 # Main Flask server
├── speaker_diarization.py    # Advanced speaker diarization
├── requirements.txt          # Python dependencies
├── start-windows.ps1         # Windows deployment script
├── start-debug.py           # Debug mode launcher
├── start-native.py          # Native mode launcher
├── docker-compose.yml       # Basic Docker setup
├── docker-compose.npu.yml   # Enhanced NPU Docker setup
├── Dockerfile               # Base container image
├── Dockerfile.npu           # NPU-optimized container
├── Dockerfile.frontend      # Frontend container
├── nginx.conf              # Production web server config
├── frontend/               # Web interface
│   ├── index.html          # Main transcription UI
│   ├── settings.html       # Settings management
│   ├── test-audio.html     # Audio testing
│   ├── js/                 # JavaScript modules
│   │   ├── main.js         # Core functionality
│   │   ├── settings.js     # Settings management
│   │   ├── audio.js        # Audio processing
│   │   └── api.js          # API communication
│   └── css/                # Stylesheets
│       ├── styles.css      # Main styles
│       └── settings.css    # Settings page styles
├── models/                 # Pre-downloaded models
│   ├── whisper-tiny.en/
│   ├── whisper-base.en/
│   ├── whisper-small.en/
│   ├── whisper-medium.en/
│   └── whisper-large-v3/
└── session_data/          # Session persistence
```

## Important Implementation Details

### NPU Optimization

The server is specifically optimized for Intel NPU acceleration:
- First-time model loading may take 30-60 seconds for NPU initialization
- Subsequent loads are significantly faster
- Minimum inference intervals protect NPU from overload
- Automatic fallback to CPU if NPU unavailable

### Error Handling

Comprehensive error handling with:
- Graceful degradation when hardware acceleration unavailable
- Automatic model fallbacks if preferred model fails
- Session recovery and state management
- Detailed logging for debugging

### Performance Considerations

- **Memory Management**: Configurable buffer sizes and queue limits
- **Concurrency**: Thread-safe audio processing with queue management
- **Real-time Processing**: Optimized for low-latency transcription
- **Resource Monitoring**: Built-in performance metrics and health checks

### Windows Compatibility

Special considerations for Windows development:
- PowerShell deployment scripts with proper error handling
- Docker Desktop integration with WSL2 support
- Path handling for Windows-specific directories
- NPU device passthrough limitations in Docker

## Development Workflow

1. **Environment Setup**: Use `pip install -r requirements.txt` for local development
2. **Model Preparation**: Ensure OpenVINO models are available in `~/.whisper/models/`
3. **Testing**: Use curl commands or the web interface for testing
4. **Debugging**: Use `start-debug.py` for verbose logging
5. **Docker Testing**: Use docker-compose for containerized testing
6. **Windows Deployment**: Use PowerShell scripts for production deployment

## Dependencies

### Core Requirements
- Flask (web server)
- OpenVINO + openvino-genai (NPU acceleration)
- librosa + soundfile (audio processing)
- numpy + scipy (numerical computing)

### Audio Processing
- webrtcvad (voice activity detection)
- pydub (audio format conversion)
- noisereduce (audio enhancement)

### Speaker Diarization (Optional)
- pyannote.audio (speaker embedding)
- torch + torchaudio (PyTorch backend)
- scikit-learn (clustering algorithms)
- resemblyzer (speaker verification)
- speechbrain (advanced audio processing)

### Advanced Features
- hdbscan + umap-learn (clustering)
- matplotlib + seaborn (visualization)
- pandas (data processing)

## Common Issues and Solutions

### NPU Not Detected
- Verify Intel NPU drivers are installed
- Check `OPENVINO_DEVICE=NPU` environment variable
- Use CPU fallback if NPU unavailable

### Model Loading Failures
- Ensure models are in OpenVINO IR format
- Check model directory permissions
- Verify model files are complete (not partial downloads)

### Audio Processing Issues
- Check audio format compatibility
- Verify sample rate settings
- Test with different audio devices

### Frontend Not Loading
- Ensure server is running on correct port
- Check CORS settings for cross-origin requests
- Verify static file serving is enabled

## Performance Tuning

- **Buffer Duration**: Adjust based on use case (3-30 seconds)
- **Inference Interval**: Set minimum intervals for NPU protection
- **Queue Sizes**: Configure based on memory constraints
- **Model Selection**: Choose appropriate model size for performance/accuracy trade-off
- **Device Selection**: Use NPU for best performance, GPU for compatibility, CPU for fallback

## Detailed Code Analysis

### Core Architecture Components

#### 1. ModelManager Class (`server.py:250-439`)
**Purpose**: Central model management with NPU optimization and thread safety

**Key Features**:
- **Device Detection**: Auto-detects NPU/GPU/CPU with environment override support
- **Thread Safety**: Uses `threading.Lock()` for NPU access protection
- **Queue Management**: `Queue(maxsize=10)` to limit concurrent requests
- **Minimum Inference Interval**: 200ms cooldown to prevent NPU overload
- **Automatic Fallback**: NPU → GPU → CPU fallback on device failures
- **Memory Management**: Automatic model cache clearing with garbage collection

**Critical Issues Found**:
1. **Path Hardcoding**: `self.models_dir = os.path.abspath("./models")` (line 254) - Should use configurable path
2. **Device State Mutation**: `self.device = "CPU"` (line 329) - Global device state changes during fallback could affect other models
3. **Missing Error Recovery**: NPU device loss clears individual model but doesn't handle system-wide NPU failure

#### 2. RollingBufferManager Class (`server.py:440-620`)
**Purpose**: Real-time audio buffer management with VAD and speaker diarization

**Key Features**:
- **Rolling Audio Buffer**: `deque(maxlen=self.max_samples)` for efficient memory usage
- **Voice Activity Detection**: WebRTC VAD with configurable aggressiveness (0-3)
- **Speaker Diarization**: Advanced speaker identification with multiple algorithms
- **Speech Enhancement**: Optional noise reduction and audio improvement
- **Overlap Management**: Prevents transcription duplication with timing controls

**Thread Safety Analysis**:
- ✅ **Good**: Uses `threading.Lock()` for buffer access (`self.buffer_lock`)
- ⚠️ **Concern**: Multiple deques accessed without consistent locking pattern

#### 3. Audio Processing Pipeline (`server.py:983-1200+`)
**Purpose**: Multi-format audio ingestion and preprocessing

**Sophisticated Format Detection**:
```python
# Detects: WAV, WebM, OGG, MP4, MP3, MP4 fragments
if audio_data.startswith(b'RIFF'): detected_format = "wav"
elif audio_data.startswith(b'\x1a\x45\xdf\xa3'): detected_format = "webm"
# ... extensive format detection logic
```

**Multi-Layer Fallback Strategy**:
1. Direct librosa loading
2. pydub AudioSegment conversion  
3. soundfile with BytesIO
4. Raw audio interpretation

**Critical Issues Found**:
1. **Incomplete Fragment Handling**: MP4 fragments detected but may fail processing (line 1066-1067)
2. **Memory Leaks**: Temporary files created but cleanup only in happy path (missing finally blocks)
3. **Resource Exhaustion**: No timeout on audio conversion operations

#### 4. Settings Management (`server.py:2242-2400`)
**Purpose**: Dynamic configuration with persistence

**Features**:
- **Runtime Introspection**: Real-time server state reporting
- **Type Validation**: Automatic type coercion for settings
- **Hot Reloading**: Buffer manager restart on streaming setting changes
- **Persistence**: JSON-based settings storage in `session_data/`

**Issues Found**:
1. **Race Conditions**: Settings updates during active transcription could cause inconsistencies
2. **Validation Gaps**: Limited validation for setting ranges and dependencies
3. **No Rollback**: Failed setting updates don't restore previous state

### Advanced Features Analysis

#### Speaker Diarization Integration
**Algorithms Supported**:
- **Embedding Methods**: Resemblyzer, SpeechBrain, PyAnnote
- **Clustering**: HDBSCAN, DBSCAN, UMAP, Agglomerative
- **Enhancement**: Real-time speech enhancement with noise reduction

**Implementation Quality**:
- ✅ **Graceful Degradation**: Stub classes when diarization unavailable
- ✅ **Optional Integration**: Can disable without affecting core functionality
- ⚠️ **Error Handling**: Limited error recovery in diarization pipeline

#### NPU-Specific Optimizations
**Intel NPU Protection**:
```python
# Minimum 200ms between inferences (line 263)
self.min_inference_interval = 0.2
# Thread-safe inference queue (line 261)
self.request_queue = Queue(maxsize=10)
# Specific error handling for NPU device states
if "ZE_RESULT_ERROR_DEVICE_LOST" in error_msg:
    logger.error("NPU device lost/hung - attempting recovery")
```

**Memory Management**:
- Automatic model cache clearing under memory pressure
- Garbage collection after model unloading
- Queue size limits to prevent memory exhaustion

### Critical Security and Reliability Issues

#### 1. Resource Management
- **Temporary File Cleanup**: Inconsistent cleanup patterns could lead to disk space exhaustion
- **Memory Leaks**: Audio buffers and model caches may not be fully cleaned up
- **File Handle Leaks**: Multiple file operations without consistent resource management

#### 2. Input Validation
- **Audio Size Limits**: No maximum file size limits - potential DoS vector
- **Format Validation**: Extensive format detection but limited validation
- **Path Traversal**: Settings file operations don't validate paths

#### 3. Thread Safety
- **Global State**: Multiple global variables accessed across threads
- **Lock Granularity**: Coarse-grained locks could impact performance
- **Deadlock Potential**: Multiple locks without consistent ordering

#### 4. Error Handling
- **Partial Failures**: NPU device failures handled inconsistently
- **Resource Cleanup**: Exception paths don't always clean up resources
- **Error Propagation**: Some errors masked instead of properly handled

### Performance Characteristics

#### Strengths
- **Intelligent Caching**: Model pipelines cached and reused efficiently
- **Adaptive Processing**: Automatic device fallback maintains availability
- **Optimized Buffering**: Rolling buffer prevents memory growth
- **Minimal Latency**: Direct audio processing without unnecessary conversions

#### Bottlenecks
- **NPU Serialization**: Thread locks force sequential NPU access
- **Audio Conversion**: Multiple fallback attempts can be slow
- **File I/O**: Temporary file operations add latency
- **JSON Processing**: Settings serialization on every update

### Recommended Improvements

#### High Priority
1. **Resource Management**: Implement proper context managers for temporary files
2. **Input Validation**: Add file size limits and format validation
3. **Error Recovery**: Improve NPU device failure recovery
4. **Path Security**: Validate all file paths in settings operations

#### Medium Priority
1. **Performance**: Implement async audio processing
2. **Monitoring**: Add detailed performance metrics collection
3. **Configuration**: Make model directory configurable
4. **Testing**: Add comprehensive unit tests for edge cases

#### Low Priority
1. **Code Organization**: Split large functions into smaller components
2. **Documentation**: Add inline documentation for complex algorithms
3. **Logging**: Implement structured logging with correlation IDs