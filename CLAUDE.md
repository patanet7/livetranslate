# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiveTranslate is a real-time speech-to-text transcription and translation system with AI acceleration. It's built as a microservices architecture with enterprise-grade WebSocket infrastructure for real-time communication.

## Service Architecture

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
# Complete development environment
./start-development.ps1

# Individual services
cd modules/frontend-service && ./start-frontend.ps1
cd modules/orchestration-service && ./start-backend.ps1
```

### Service-Specific Commands
```bash
# Whisper Service (NPU/GPU optimized)
cd modules/whisper-service
python src/main.py --device=npu

# Translation Service (GPU optimized)
cd modules/translation-service
python src/translation_service.py --device=gpu

# Orchestration Service with bot management
cd modules/orchestration-service
pip install -r requirements.txt -r requirements-google-meet.txt -r requirements-database.txt
python src/orchestration_service.py
```

## File Structure Conventions

- `src/` - Source code for each service
- `tests/` - Test files (unit, integration, stress)
- `requirements*.txt` - Python dependencies (service-specific)
- `docker-compose*.yml` - Docker deployment configurations
- `static/` - Static web assets (orchestration service)

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

### Service-Specific Tests
- **Whisper**: NPU fallback, real-time performance, edge cases
- **Translation**: GPU memory management, quality metrics
- **Orchestration**: Service coordination, health monitoring
- **Frontend**: Component testing, integration tests

### Test Commands
```bash
# Full system testing
python tests/run_all_tests.py --comprehensive

# Service-specific
cd modules/whisper-service && python tests/run_tests.py --all --device=npu
cd modules/orchestration-service && python tests/run_tests.py --all
```

## Important Notes

- **Windows environment**: Use PowerShell commands (`.ps1` scripts)
- **Audio resampling**: Fixed 48kHz to 16kHz conversion with librosa fallback
- **Production deployment**: Services can run on separate machines
- **Real-time processing**: < 100ms latency target
- **Concurrent support**: 1000+ WebSocket connections

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