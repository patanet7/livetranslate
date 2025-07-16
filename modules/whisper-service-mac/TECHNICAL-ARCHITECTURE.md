# whisper-service-mac Technical Architecture

## Architecture Overview

The whisper-service-mac is a native macOS implementation of the Whisper speech-to-text service, optimized for Apple Silicon hardware and providing full compatibility with the LiveTranslate orchestration service.

## Core Components

### 1. WhisperCppEngine (`src/core/whisper_cpp_engine.py`)
The heart of the service, providing native whisper.cpp integration.

**Key Features:**
- Direct whisper.cpp binary integration
- Apple Silicon capability detection (Metal, Core ML, ANE)
- GGML model management and loading
- Thread-safe inference with performance tracking
- Automatic fallback chains (ANE → Metal → CPU)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                WhisperCppEngine                             │
├─────────────────────────────────────────────────────────────┤
│ • Model Management (GGML format)                           │
│ • Capability Detection (Metal/Core ML/ANE)                 │
│ • Audio Processing (16kHz conversion)                      │
│ • Thread-safe Inference                                    │
│ • Performance Tracking                                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 whisper.cpp Binary                         │
├─────────────────────────────────────────────────────────────┤
│ • Native C++ Implementation                                │
│ • Metal GPU Acceleration                                   │
│ • Core ML + ANE Support                                    │
│ • GGML Model Loading                                       │
│ • Word-level Timestamps                                    │
└─────────────────────────────────────────────────────────────┘
```

### 2. API Server (`src/api/api_server.py`)
Flask-based REST API providing orchestration service compatibility.

**Endpoint Categories:**
- **Core Endpoints**: `/health`, `/models`, `/api/models`, `/api/device-info`
- **Transcription**: `/transcribe`, `/api/process-chunk`
- **macOS-Specific**: `/api/metal/status`, `/api/coreml/models`
- **Advanced**: `/api/word-timestamps`

**Request Flow:**
```
HTTP Request → Flask Router → Endpoint Handler → WhisperCppEngine → Response
```

### 3. Build System (`build-scripts/`)
Automated whisper.cpp compilation with Apple Silicon optimizations.

**Components:**
- `build-whisper-cpp.sh`: Main build script with Metal/Core ML support
- `download-models.sh`: GGML model management
- `generate-coreml-models.sh`: Core ML model conversion

## Apple Silicon Optimizations

### Hardware Acceleration Stack
```
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                        │
│              (whisper-service-mac)                         │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                 whisper.cpp Engine                         │
│        (Native C++ with Apple optimizations)              │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                Hardware Acceleration                       │
├─────────────────────────────────────────────────────────────┤
│ Apple Neural Engine (ANE)  │  Metal GPU   │  CPU (NEON)   │
│ via Core ML                │  via Metal   │  via Accelerate│
│ • Lowest Power             │  • High Perf │  • Universal   │
│ • Highest Efficiency       │  • Parallel  │  • Fallback    │
└─────────────────────────────────────────────────────────────┘
```

### Capability Detection
The service automatically detects and prioritizes hardware capabilities:

1. **Apple Neural Engine (ANE)**
   - Accessed via Core ML framework
   - Lowest power consumption
   - Highest efficiency for supported models
   - Limited to specific model sizes

2. **Metal GPU**
   - Direct Metal Performance Shaders access
   - High-performance parallel processing
   - Unified memory architecture benefits
   - Supports all model sizes

3. **CPU with NEON**
   - ARM64 NEON SIMD instructions
   - Accelerate framework optimizations
   - Universal compatibility
   - Final fallback option

## Model Management

### GGML Format Support
The service uses GGML (GPT-Generated Model Language) format for models:

**Model Types:**
- **Full Precision**: `ggml-{model}.bin`
- **Quantized**: `ggml-{model}-q{level}.bin`
- **Core ML**: `{model}.mlmodelc` (compiled for ANE)

**Model Conversion Pipeline:**
```
HuggingFace Model → GGML Converter → ggml-{model}.bin
                                        │
                                        ▼
                     Core ML Converter → {model}.mlmodelc
```

### Unified Model Directory
Models are stored in the ecosystem's unified structure:
```
../models/
├── ggml/                    # GGML models for whisper.cpp
│   ├── ggml-tiny.bin
│   ├── ggml-base.bin
│   ├── ggml-small.bin
│   ├── ggml-medium.bin
│   └── ggml-large-v3.bin
├── cache/coreml/            # Core ML compiled models
│   ├── tiny.mlmodelc
│   ├── base.mlmodelc
│   └── small.mlmodelc
└── base/                    # Source models (for conversion)
    ├── whisper-tiny/
    ├── whisper-base/
    └── whisper-small/
```

## Performance Characteristics

### Latency Targets
- **Apple Neural Engine**: <50ms (small models)
- **Metal GPU**: <100ms (M3 Pro+), <200ms (M1)
- **CPU**: <300ms (with NEON optimizations)

### Throughput
- **Real-time Factor**: 10-15x on Apple Silicon
- **Concurrent Requests**: Handled via thread-safe inference
- **Memory Usage**: 4-8GB unified memory (model dependent)

### Power Efficiency
- **ANE**: 5-10x more power efficient than CPU
- **Metal**: 2-3x more efficient than CPU
- **Thermal Management**: Automatic throttling prevention

## Integration Architecture

### Orchestration Service Compatibility
The service provides seamless integration with the existing orchestration service:

**API Compatibility:**
- Same endpoint URLs and HTTP methods
- Identical request/response formats
- Model name conversion (orchestration ↔ GGML)
- Error handling and status codes

**Service Discovery:**
```
┌─────────────────────────────────────────────────────────────┐
│                Orchestration Service                       │
├─────────────────────────────────────────────────────────────┤
│ • Service Registry                                          │
│ • Load Balancing                                           │
│ • Health Monitoring                                        │
│ • Request Routing                                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              whisper-service-mac                           │
├─────────────────────────────────────────────────────────────┤
│ • Health Endpoint: /health                                 │
│ • Capabilities: /api/device-info                           │
│ • Models: /api/models                                      │
│ • Processing: /api/process-chunk                           │
└─────────────────────────────────────────────────────────────┘
```

### Real-time Streaming
The `/api/process-chunk` endpoint supports real-time audio streaming:

**Streaming Architecture:**
```
Audio Stream → Chunks → Base64 Encoding → HTTP POST → Transcription → Response
    (2-5s)      (16kHz)        (JSON)       (REST)        (whisper.cpp)
```

**Session Management:**
- Chunk-based processing for continuous audio
- Session tracking across multiple chunks
- Automatic model loading and caching
- Context preservation for better accuracy

## Error Handling and Reliability

### Fallback Mechanisms
1. **Hardware Fallback**: ANE → Metal → CPU
2. **Model Fallback**: Large → Medium → Small → Tiny
3. **Service Fallback**: Orchestration handles service unavailability

### Error Categories
- **Initialization Errors**: Engine startup, model loading
- **Processing Errors**: Audio format, transcription failures
- **Resource Errors**: Memory, disk space, model availability
- **Network Errors**: HTTP timeouts, connection issues

### Monitoring and Logging
- **Performance Metrics**: Processing time, model usage
- **Error Tracking**: Categorized error logging
- **Health Monitoring**: Continuous capability checking
- **Resource Monitoring**: Memory and CPU usage tracking

## Security Considerations

### Input Validation
- Audio format validation
- File size limits
- Base64 encoding validation
- Model name sanitization

### Resource Protection
- Memory usage limits
- Processing timeouts
- Concurrent request limits
- Temporary file cleanup

### Service Isolation
- No external network access required
- Local model storage
- Sandboxed processing environment
- Minimal system dependencies

## Development and Testing

### Development Environment
```bash
# Setup local environment
./setup-env.sh

# Build whisper.cpp
./build-scripts/build-whisper-cpp.sh

# Download models
./scripts/download-models.sh

# Start development server
./start-mac-dev.sh
```

### Testing Framework
- **Unit Tests**: Core engine functionality
- **Integration Tests**: API endpoint validation
- **Performance Tests**: Latency and throughput
- **Compatibility Tests**: Orchestration service integration

### Continuous Integration
- **Build Verification**: whisper.cpp compilation
- **API Testing**: Endpoint functionality
- **Performance Benchmarking**: Regression detection
- **Compatibility Validation**: Orchestration integration

## Deployment Considerations

### System Requirements
- **OS**: macOS 11.0+ (Big Sur or later)
- **Hardware**: Apple Silicon recommended (M1/M2/M3)
- **Memory**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for models and cache

### Docker Support
While optimized for native macOS, Docker support is available:
```dockerfile
FROM --platform=linux/arm64 python:3.12-slim
# Note: Hardware acceleration not available in containers
```

### Production Deployment
- **Service Mesh**: Istio integration for routing
- **Load Balancing**: Multiple instance support
- **Health Checks**: Kubernetes readiness/liveness probes
- **Monitoring**: Prometheus metrics integration

## Future Enhancements

### Planned Features
1. **Dynamic Model Loading**: Hot-swap models without restart
2. **Batch Processing**: Multi-file transcription optimization
3. **Custom Models**: Fine-tuned model support
4. **Streaming Improvements**: Lower latency chunk processing
5. **Advanced ANE**: Direct Neural Engine API access

### Performance Optimizations
1. **Model Quantization**: INT8/INT4 model support
2. **Memory Management**: Advanced caching strategies
3. **Parallel Processing**: Multi-threaded inference
4. **Hardware Scheduling**: Intelligent workload distribution

This architecture provides a solid foundation for high-performance, Apple Silicon-optimized speech-to-text processing while maintaining full compatibility with the existing LiveTranslate ecosystem.