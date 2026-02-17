# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiveTranslate is a real-time speech-to-text transcription and translation system with AI acceleration. It's built as a microservices architecture with enterprise-grade WebSocket infrastructure for real-time communication.

## ML Deployment Infrastructure Documentation

**NEW:** Comprehensive ML pipeline analysis and MLOps recommendations now available:

- **[ML Pipeline Executive Summary](docs/ML_PIPELINE_SUMMARY.md)** - Quick-reference guide with immediate action items
- **[ML Deployment Infrastructure (Full)](docs/ML_DEPLOYMENT_INFRASTRUCTURE.md)** - Complete 1,081-line analysis

**Key Findings:**
- NPU-optimized Whisper service with OpenVINO IR models
- GPU-optimized Translation service with Triton/vLLM serving
- Identified 10x throughput improvement opportunities via dynamic batching
- 97% cost reduction potential through auto-scaling and optimization
- 16-week MLOps implementation roadmap included

## Service Architecture
# NEVER USE MOCK DATA!!!! ALWAYS COMPREHENSIVE AND INTEGRATED!!!
# ALWAYS USE UV FOR DEPENDENCY MANAGEMENT - NOT PIP OR PDM!
### 4 Core Services

1. **Whisper Service** (`modules/whisper-service/`) - **[NPU OPTIMIZED]** ✅
   - **Purpose**: Combined Whisper + Speaker Diarization + Audio Processing + VAD
   - **Hardware**: Intel NPU (primary), GPU/CPU (fallback)
   - **Status**: Production-ready with comprehensive WebSocket infrastructure

2. **Translation Service** (`modules/translation-service/`) - **[GPU OPTIMIZED]**
   - **Purpose**: Multi-language translation with local LLMs (vLLM, Ollama, Triton)
   - **Hardware**: NVIDIA GPU (primary), CPU (fallback)
   - **Status**: Solid foundation, needs GPU optimization

3. **Orchestration Service** (`modules/orchestration-service/`) - **[CPU OPTIMIZED]** ✅
   - **Purpose**: Backend API coordination, bot management, configuration sync
   - **Hardware**: CPU-optimized (lightweight)
   - **Status**: Production-ready with integrated Google Meet bot management and config sync
   - **🆕 Audio Upload API**: Fixed 422 validation errors with proper FastAPI dependency injection
   - **🆕 Model Consistency**: Standardized "whisper-base" naming across all fallback mechanisms

4. **Frontend Service** (`modules/frontend-service/`) - **[BROWSER OPTIMIZED]** ✅
   - **Purpose**: Modern React user interface
   - **Technology**: React 18 + TypeScript + Material-UI + Vite
   - **Status**: Production-ready with comprehensive settings management
   - **🆕 Meeting Test Dashboard**: Fully operational real-time streaming without 422 errors
   - **🆕 Dynamic Model Loading**: Fixed model selection with proper "whisper-base" naming

### Key Technical Components

#### Google Meet Bot Management System ✅
- **Location**: `modules/orchestration-service/src/bot/`
- **GoogleMeetBotManager**: Central bot lifecycle management (`src/bot/bot_manager.py`)
- **Google Meet Browser Automation**: Headless Chrome integration (`src/bot/google_meet_automation.py`)
- **Browser Audio Capture**: Specialized Google Meet audio extraction (`src/bot/browser_audio_capture.py`)
- **Virtual Webcam System**: Real-time translation overlay generation (`src/bot/virtual_webcam.py`)
- **Time Correlation Engine**: Advanced timeline matching (`src/bot/time_correlation.py`)
- **Bot Integration Pipeline**: Complete orchestration flow (`src/bot/bot_integration.py`)
- **Database Integration**: PostgreSQL persistence (`src/database/bot_session_manager.py`)
- **Schema**: `scripts/bot-sessions-schema.sql` - Comprehensive PostgreSQL schema

#### Configuration Synchronization System ✅
- **ConfigurationSyncManager**: `modules/orchestration-service/src/audio/config_sync.py`
- **Frontend Settings**: `modules/frontend-service/src/pages/Settings/`
- **API Endpoints**: `modules/orchestration-service/src/routers/settings.py`
- **Whisper Integration**: `modules/whisper-service/src/api_server.py` (orchestration mode)

## Service Ports

- **Frontend**: 5173 (development), 3000 (production)
- **Orchestration**: 3000
- **Whisper**: 5001
- **Translation**: 5003
- **Monitoring**: 3001
- **Prometheus**: 9090

## Development Commands

### Quick Start
```bash
# Complete development environment (cross-platform)
./start-development.sh        # macOS/Linux
./start-development.ps1       # Windows

# Individual services
cd modules/frontend-service && ./start-frontend.sh
cd modules/orchestration-service && ./start-backend.sh

# Using just (recommended)
just compose-up               # Start all services with Docker
just dev                      # Start development environment
```

### Service-Specific Commands

> **IMPORTANT**: Use `uv sync --group dev` to install dev dependencies including pytest, pytest-timeout, and code quality tools.
> **Python Version**: Services require Python >=3.12,<3.14. Python 3.14+ is not supported due to ML package compatibility (grpcio, onnxruntime).
> **Workspace**: This is a UV workspace monorepo. Run `uv sync` from the root to install all packages including the shared `livetranslate-common` library.

```bash
# Install all dependencies (from repo root)
uv sync --group dev

# Whisper Service (NPU/GPU optimized)
uv run python modules/whisper-service/src/main.py --device=npu
uv run pytest modules/whisper-service/tests/ -v

# Translation Service (GPU optimized)
uv run python modules/translation-service/src/api_server.py
uv run pytest modules/translation-service/tests/ -v

# Orchestration Service with bot management
uv run python modules/orchestration-service/src/main_fastapi.py
uv run pytest modules/orchestration-service/tests/ -v

# Shared Library tests
uv run pytest modules/shared/tests/ -v

# Frontend Service (React/TypeScript)
cd modules/frontend-service
npm install
npm run dev  # Start development server
npm test     # Run tests
```

### Justfile Commands
```bash
just help              # Show all available commands
just test-orchestration    # Run orchestration tests
just test-whisper          # Run whisper tests
just test-translation      # Run translation tests
just coverage-backend      # Generate coverage reports
just docker-build-all      # Build all Docker images
just clean                 # Clean build artifacts
just install-all           # Install all dependencies
just db-up / db-down       # Database management
```

## File Structure Conventions

- `src/` - Source code for each service
- `tests/` - Test files (unit, integration, stress)
- `requirements*.txt` - Python dependencies (service-specific)
- `docker-compose*.yml` - Docker deployment configurations
- `static/` - Static web assets (orchestration service)

## Structured Logging (structlog)

All services use **structlog** via the shared `livetranslate-common` package. Do NOT use `import logging` / `logging.getLogger(__name__)`.

### Usage Pattern
```python
from livetranslate_common.logging import setup_logging, get_logger

# In service entry point (once):
setup_logging(service_name="orchestration")  # or "whisper", "translation"

# In all modules:
logger = get_logger()
logger.info("event_name", key="value", count=42)
```

### Key Points
- **JSON output** (production): `LOG_FORMAT=json` or default
- **Dev output** (colored): `LOG_FORMAT=dev` or `setup_logging(..., log_format="dev")`
- **Sensitive data** automatically redacted (passwords, tokens, API keys)
- **Request ID** middleware injects `request_id` into all log lines via contextvars
- **Performance logging**: `from livetranslate_common.logging.performance import log_performance`
- **stdlib bridging**: Third-party library logs (uvicorn, httpx, etc.) are captured and formatted through structlog
- **Exception**: `import logging` is OK only for `logging.getLogger("third_party").setLevel(...)` to control third-party log levels

### Shared Library Location
- Package: `modules/shared/src/livetranslate_common/`
- Modules: `logging/`, `errors/`, `middleware/`, `config/`, `health/`
- Tests: `modules/shared/tests/`

## Development Notes

### Audio Processing Pipeline
- **Critical Fix**: Browser audio processing features (echoCancellation, noiseSuppression, autoGainControl) disabled in `modules/orchestration-service/static/js/audio.js` to prevent loopback audio attenuation
- **Backend Fix**: Aggressive noise reduction disabled in `modules/whisper-service/src/api_server.py` to preserve loopback audio content
- **Voice-Specific Processing**: 10-stage pipeline with pause capability in `modules/orchestration-service/static/js/audio-processing-test.js`

### WebSocket Infrastructure
- **Enterprise-grade features**: Connection pooling (1000 capacity), 20+ error categories, heartbeat monitoring
- **Session persistence**: 30-minute timeout with message buffering
- **Zero-message-loss design**: Message routing with pub-sub capabilities

### Database Schema
- **PostgreSQL**: Comprehensive schema for bot sessions, audio files, transcripts, translations, correlations
- **Indexes**: Optimized for session, time, speaker, and language queries
- **Views**: Pre-computed session statistics and analytics

## Important Technical Details

### Configuration Flow
```
Frontend Settings ↔ Orchestration API ↔ Whisper Service Config
```
- **Bidirectional sync** with real-time updates
- **Compatibility validation** prevents breaking changes
- **Preset system** for common deployment scenarios

### Bot Lifecycle & Virtual Webcam Pipeline
```
Request → Database Session → Google Meet Browser → Audio Capture → Orchestration Service
    ↓
Whisper Service (NPU) → Speaker Diarization → Time Correlation → Translation Service
    ↓
Virtual Webcam Generation → Real-time Display → Speaker Attribution
```
- **Complete Audio Pipeline**: Google Meet browser audio → orchestration → whisper → translation → virtual webcam
- **Speaker Attribution**: Enhanced display with diarization info (e.g., "John Doe (SPEAKER_00)")
- **Real-time Translation Overlay**: Professional webcam output with speaker names and confidence scores
- **Thread-safe operations** with proper locking
- **Automatic recovery** for failed bots (max 3 attempts)
- **Performance tracking** with success rates

### Hardware Acceleration
- **NPU**: Intel NPU support via OpenVINO (Whisper service)
- **GPU**: NVIDIA GPU with CUDA (Translation service)
- **CPU**: Automatic fallback for all services

## Testing Strategy

### CRITICAL: Behavioral Tests with NO MOCKS

All tests in this repository **MUST** be behavioral/integration tests that test real system behavior:

1. **NO MOCKING** - Do not mock services, databases, or external dependencies
2. **Real Services** - Tests should spin up real service instances (use docker-compose)
3. **Real Data Flow** - Test actual data flowing through the system
4. **Test Outputs** - All test results go to `tests/output/` with timestamp format: `TIMESTAMP_test_XXX_results.log`

### Test Output Locations
- Backend tests: `modules/<service>/tests/output/`
- Frontend tests: `modules/frontend-service/tests/output/`

### Example Behavioral Test Pattern
```python
# CORRECT: Behavioral test with real services
@pytest.mark.integration
@pytest.mark.behavioral
async def test_audio_upload_and_transcription():
    """Test real audio upload through the entire pipeline."""
    output_path = get_test_output_path("audio_upload")

    # Start real services
    async with ServiceTestContext() as ctx:
        # Upload real audio file
        response = await ctx.client.post("/api/audio/upload", files={"file": audio_bytes})
        assert response.status_code == 200

        # Wait for real transcription
        result = await ctx.wait_for_transcription(response.json()["session_id"])
        assert result["text"] is not None
        assert len(result["text"]) > 0

        # Log results to output file
        with open(output_path, "w") as f:
            f.write(f"Test passed: {result}")

# INCORRECT: Mocked test (DO NOT USE)
# def test_audio_upload_mocked():
#     with patch("services.whisper") as mock_whisper:  # NO!
#         mock_whisper.transcribe.return_value = "fake"  # NO!
```

### Service-Specific Tests
- **Whisper**: NPU fallback, real-time performance, edge cases
- **Translation**: GPU memory management, quality metrics
- **Orchestration**: Service coordination, health monitoring
- **Frontend**: Component testing, E2E tests with Playwright

### Test Commands
```bash
# Full system testing
python tests/run_all_tests.py --comprehensive

# Service-specific (using UV)
uv run pytest modules/orchestration-service/tests/ -m "behavioral"
uv run pytest modules/whisper-service/tests/ -m "integration"
uv run pytest modules/translation-service/tests/ -m "behavioral"

# Shared library tests
uv run pytest modules/shared/tests/ -v

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Using just commands
just test-orchestration
just test-whisper
just test-translation
just coverage-backend
```

### Test Markers
- `@pytest.mark.behavioral` - Behavioral tests (no mocks)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow tests (skip with `-m "not slow"`)
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.npu` - Requires Intel NPU

## Important Notes

- **Cross-platform support**: Bash scripts (`.sh`) for macOS/Linux, PowerShell (`.ps1`) for Windows
- **Dependency management**: Use UV workspaces (not PDM, Poetry, or pip directly)
- **Audio resampling**: Fixed 48kHz to 16kHz conversion with librosa fallback
- **Production deployment**: Services can run on separate machines
- **Real-time processing**: < 100ms latency target
- **Concurrent support**: 1000+ WebSocket connections
- **Code quality**: Ruff for linting/formatting, mypy for type checking
- **Pre-commit hooks**: Run `pre-commit install` after cloning

## Latest Critical Fixes (Audio Flow Resolution)

### ✅ **422 Validation Error Resolution**
**Problem**: Frontend Meeting Test Dashboard failing with 422 errors on `/api/audio/upload`
**Root Cause**: FastAPI dependency injection not properly implemented in orchestration service
**Files Fixed**: `modules/orchestration-service/src/routers/audio.py`
- Added proper `audio_client=Depends(get_audio_service_client)` to function signature
- Fixed direct function call to use injected dependency parameter
- Resolved all HTTP 422 Unprocessable Content errors

### ✅ **Model Name Standardization**
**Problem**: Inconsistent model naming between frontend ("base") and services ("whisper-base")
**Root Cause**: Multiple fallback mechanisms using different naming conventions
**Files Fixed**: 
- `modules/orchestration-service/src/routers/audio.py` - Updated fallback model arrays
- `modules/orchestration-service/src/clients/audio_service_client.py` - Fixed client fallbacks
**Result**: Consistent "whisper-base" naming across all components and fallback scenarios

### ✅ **Complete Audio Flow Verification**
**Flow Validated**: Frontend → Orchestration → Whisper → Translation → Response
**Status**: ✅ **FULLY OPERATIONAL** 
**Features Confirmed**:
- Real-time streaming with configurable 2-5 second chunks
- Dynamic model loading with proper device status display  
- Hardware acceleration fallback (NPU → GPU → CPU)
- Comprehensive error handling and service recovery
- Session tracking and chunk management
- Multi-language translation with quality scoring

### ✅ **Complete Virtual Webcam Implementation**
**Problem**: Need virtual webcam display for Google Meet bot with speaker attribution
**Solution**: Comprehensive virtual webcam system with professional translation overlays
**Files Implemented**:
- `modules/orchestration-service/src/bot/virtual_webcam.py` - Complete webcam generation system
- `modules/orchestration-service/src/bot/bot_integration.py` - Enhanced pipeline integration
- `modules/orchestration-service/src/routers/bot.py` - Virtual webcam API endpoints
**Features Delivered**:
- **Speaker Attribution**: Enhanced display with both human names and diarization IDs
- **Dual Content Display**: Shows both original transcriptions (🎤) and translations (🌐)
- **Professional Layout**: Enhanced boxes with confidence scores, language indicators, timestamps
- **Real-time Updates**: 30fps frame generation with configurable content duration
- **API Integration**: Complete REST API for frame streaming and configuration

### 🎯 **Complete Google Meet Bot System - Production Ready**
All components of the Google Meet bot system are now fully operational:
- ✅ **Browser Automation**: Headless Chrome Google Meet integration
- ✅ **Audio Capture**: Specialized browser audio extraction with multiple fallback methods
- ✅ **Audio Pipeline**: Complete orchestration → whisper → translation flow
- ✅ **Virtual Webcam**: Professional translation overlay with speaker attribution
- ✅ **Time Correlation**: Advanced matching between Google Meet captions and internal transcriptions
- ✅ **Database Integration**: Complete session tracking and analytics
- ✅ **API Endpoints**: Full REST API for bot management and webcam control
