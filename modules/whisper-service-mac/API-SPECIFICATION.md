# whisper-service-mac API Specification

## Overview
Native macOS whisper.cpp service with Apple Silicon optimizations for the LiveTranslate ecosystem.

## Base URL
- Development: `http://localhost:5002`
- Production: `http://whisper-service-mac:5002`

## Authentication
No authentication required for internal service communication.

## Core Endpoints

### Health Check
**GET** `/health`

Returns service health status and capabilities.

**Response:**
```json
{
  "status": "healthy",
  "service": "whisper-service-mac",
  "version": "1.0.0-mac",
  "engine": "whisper.cpp",
  "uptime": 123.45,
  "timestamp": 1642782000.123,
  "capabilities": {
    "metal": true,
    "coreml": true,
    "ane": true,
    "unified_memory": true,
    "neon": true,
    "word_timestamps": true,
    "quantization": true
  },
  "current_model": "base",
  "available_models": 5,
  "apple_silicon": {
    "metal_enabled": true,
    "coreml_enabled": true,
    "ane_enabled": true
  }
}
```

### Models Management

#### List Models
**GET** `/models`

Returns detailed model information.

**Response:**
```json
{
  "models": [
    {
      "name": "tiny",
      "file_name": "ggml-tiny.bin",
      "size": "39MB",
      "quantization": null,
      "coreml_available": true,
      "format": "ggml"
    }
  ],
  "default_model": "base",
  "engine": "whisper.cpp"
}
```

#### API Models (Orchestration)
**GET** `/api/models`

Returns model names in orchestration-compatible format.

**Response:**
```json
{
  "available_models": ["whisper-tiny", "whisper-base", "whisper-small"],
  "current_model": "base"
}
```

### Device Information
**GET** `/api/device-info`

Returns hardware capabilities for orchestration routing.

**Response:**
```json
{
  "device": "Metal",
  "device_type": "Apple Silicon",
  "acceleration": {
    "metal": true,
    "coreml": true,
    "ane": true,
    "accelerate": true,
    "neon": true
  },
  "memory": "Unified Memory",
  "platform": "macOS-15.5-arm64-arm-64bit",
  "architecture": "arm64",
  "capabilities": {
    "metal": true,
    "coreml": true,
    "ane": true,
    "unified_memory": true,
    "neon": true,
    "word_timestamps": true,
    "quantization": true
  },
  "engine": "whisper.cpp",
  "service": "whisper-service-mac"
}
```

## Transcription Endpoints

### File Upload Transcription
**POST** `/transcribe`

Transcribes uploaded audio files.

**Request (multipart/form-data):**
- `file`: Audio file (WAV, MP3, etc.)
- `language`: Target language (optional, default: auto)
- `task`: "transcribe" or "translate" (optional, default: transcribe)
- `model`: Model name (optional, uses current model)

**Response:**
```json
{
  "text": "This is the transcribed text.",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "This is the transcribed text."
    }
  ],
  "model": "base",
  "processing_time": 0.85,
  "engine": "whisper.cpp",
  "service": "whisper-service-mac",
  "word_timestamps": [
    {
      "word": "This",
      "start": 0.0,
      "end": 0.3
    }
  ]
}
```

### Orchestration Chunk Processing
**POST** `/api/process-chunk`

Processes audio chunks for real-time streaming (CRITICAL for orchestration).

**Request:**
```json
{
  "audio_data": "base64-encoded-audio-data",
  "sample_rate": 16000,
  "model": "whisper-base",
  "chunk_id": "chunk-001",
  "session_id": "session-123",
  "language": "en"
}
```

**Response:**
```json
{
  "transcription": "Transcribed text from chunk",
  "chunk_id": "chunk-001",
  "session_id": "session-123",
  "processing_time": 0.15,
  "model_used": "base",
  "confidence": 0.95,
  "segments": [
    {
      "start": 0.0,
      "end": 1.0,
      "text": "Transcribed text from chunk"
    }
  ],
  "word_timestamps": [
    {
      "word": "Transcribed",
      "start": 0.0,
      "end": 0.4
    }
  ]
}
```

## macOS-Specific Endpoints

### Metal GPU Status
**GET** `/api/metal/status`

Returns Metal GPU acceleration status.

**Response:**
```json
{
  "metal_available": true,
  "metal_performance_shaders": true,
  "unified_memory": true,
  "metal_device": "Apple M3 Pro",
  "metal_family": "Apple7"
}
```

### Core ML Models
**GET** `/api/coreml/models`

Returns available Core ML models for Apple Neural Engine.

**Response:**
```json
{
  "coreml_models": [
    {
      "name": "whisper-base-coreml",
      "path": "/models/cache/coreml/base.mlmodelc",
      "ane_compatible": true
    }
  ],
  "ane_available": true
}
```

### Word-Level Timestamps
**POST** `/api/word-timestamps`

Get detailed word-level timestamps for transcribed text.

**Request:**
```json
{
  "audio_data": "base64-encoded-audio-data",
  "model": "base",
  "language": "en"
}
```

**Response:**
```json
{
  "text": "This is a test transcription.",
  "word_timestamps": [
    {
      "word": "This",
      "start": 0.0,
      "end": 0.3,
      "confidence": 0.98
    }
  ],
  "processing_time": 0.45
}
```

## Error Handling

### Error Response Format
```json
{
  "error": "Error description",
  "code": "ERROR_CODE",
  "service": "whisper-service-mac"
}
```

### Status Codes
- `200`: Success
- `400`: Bad Request (invalid audio data, missing parameters)
- `404`: Not Found (endpoint not found)
- `500`: Internal Server Error (transcription failed, engine error)
- `503`: Service Unavailable (engine not initialized)

## Model Name Conversion

The service automatically converts between orchestration and GGML model names:

- `whisper-tiny` → `tiny`
- `whisper-base` → `base`
- `whisper-small` → `small`
- `whisper-medium` → `medium`
- `whisper-large-v3` → `large-v3`

## Apple Silicon Optimizations

### Metal GPU Acceleration
- Automatic detection and usage
- Parallel processing for faster inference
- Memory optimization for unified memory architecture

### Core ML + Apple Neural Engine
- Automatic model conversion to Core ML format
- ANE acceleration for supported models
- Fallback to Metal/CPU when needed

### Performance Characteristics
- **Latency**: <100ms (M3 Pro+), <200ms (M1)
- **Throughput**: 10-15x real-time on Apple Silicon
- **Memory**: 4-8GB unified memory usage
- **Power**: Optimized for battery efficiency

## Integration Notes

### Orchestration Service Compatibility
This service provides drop-in compatibility with the original whisper-service:
- Same endpoint URLs and request/response formats
- Model name conversion for seamless integration
- Hardware-specific optimizations transparent to orchestration

### Real-time Streaming
The `/api/process-chunk` endpoint is optimized for real-time streaming:
- Low latency processing (<150ms)
- Chunk-based processing for continuous audio
- Session tracking for multi-chunk transcriptions
- Automatic model loading and caching