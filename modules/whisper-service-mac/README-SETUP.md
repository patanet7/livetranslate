# macOS Whisper Service

Native macOS implementation of the Whisper service using whisper.cpp with Apple Silicon optimizations.

## Quick Setup

1. **Setup local environment (recommended):**
   ```bash
   ./setup-env.sh
   ```
   This creates a local `venv/` directory with all Python dependencies.

2. **Install system dependencies:**
   ```bash
   brew install cmake git
   ```

3. **Build whisper.cpp:**
   ```bash
   ./build-scripts/build-whisper-cpp.sh
   ```

4. **Download models:**
   ```bash
   ./scripts/download-models.sh
   ```

5. **Start the service:**
   ```bash
   ./start-mac-dev.sh
   ```

6. **Run tests:**
   ```bash
   ./test.sh
   ```

## Local Environment

The setup creates a local virtual environment in `./venv/` that contains all Python dependencies. This keeps everything isolated to this project folder.

### Manual Environment Management

```bash
# Activate environment
source venv/bin/activate

# Deactivate when done
deactivate

# Install additional packages
pip install package-name
```

## Features

- **Apple Silicon Optimized**: Metal GPU + Core ML + Apple Neural Engine
- **Universal Compatibility**: Works on Intel and Apple Silicon Macs
- **Full API Compatibility**: Drop-in replacement for whisper-service
- **Local Dependencies**: Everything contained in project folder
- **Multilingual Models**: All Whisper models (tiny → large-v3)

## API Endpoints

Compatible with orchestration service:
- `GET /health` - Service health check
- `GET /api/models` - Available models
- `GET /api/device-info` - Hardware capabilities
- `POST /api/process-chunk` - Process audio chunk
- `POST /transcribe` - Transcribe audio file

## Directory Structure

```
whisper-service-mac/
├── venv/                     # Local virtual environment
├── src/                      # Python source code
├── whisper_cpp/              # whisper.cpp submodule
├── build-scripts/            # Build automation
├── scripts/                  # Model management
├── config/                   # Configuration files
└── requirements.txt          # Python dependencies
```

## Models Location

Models are stored in the unified models directory:
```
../models/ggml/              # GGML models for whisper.cpp
../models/cache/coreml/      # Core ML models (Apple Silicon)
```

This follows the ecosystem's unified model management system.

## Testing

The service includes comprehensive tests for all components:

### Quick Test Run
```bash
./test.sh
```

### Specific Test Types
```bash
# Unit tests only
./test.sh --type unit

# API endpoint tests
./test.sh --type api

# Orchestration compatibility tests
./test.sh --type compatibility

# With coverage report
./test.sh --coverage

# Setup test environment
./test.sh --setup
```

### Test Categories

- **Unit Tests**: Core engine functionality, model management, Apple Silicon optimizations
- **Integration Tests**: Full API endpoints, service startup, configuration loading
- **Compatibility Tests**: Orchestration service compatibility, response format validation
- **Performance Tests**: Response times, concurrent requests, resource usage

### Test Coverage

Tests cover:
- All API endpoints (`/health`, `/api/models`, `/api/device-info`, `/api/process-chunk`, `/transcribe`)
- WhisperCppEngine functionality (transcription, model loading, capability detection)
- Apple Silicon optimizations (Metal, Core ML, ANE)
- Model name conversion (orchestration ↔ GGML formats)
- Error handling and edge cases
- CORS headers and browser compatibility
- Service startup and configuration loading