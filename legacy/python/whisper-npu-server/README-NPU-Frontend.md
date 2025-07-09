# 🧠 Whisper NPU Server Frontend

A modern web interface for the Whisper NPU server that provides real-time transcription with NPU acceleration support, comprehensive settings management, and server control capabilities.

## 🚀 Quick Start

### Windows Docker Setup (Recommended)

1. **Start the NPU Server with Docker Compose**
```powershell
# For NPU support (preferred)
docker-compose -f docker-compose.npu.yml up whisper-npu-server

# For CPU fallback
docker-compose -f docker-compose.npu.yml --profile cpu-fallback up whisper-npu-server-cpu

# With frontend service
docker-compose -f docker-compose.npu.yml --profile frontend up
```

2. **Access the Interface**
- **Main Interface**: http://localhost:8009 (direct server)
- **With Frontend Service**: http://localhost:8080 (nginx proxy)

### Native Python Setup

1. **Start the NPU Server**
```bash
python setup-npu.py start
```

2. **Start the Frontend**
```bash
python serve-frontend.py
```

3. **Open the Interface**
The frontend will automatically open in your browser at:
**http://localhost:8080/npu-frontend.html**

## 🎯 Features

### 📊 **Real-time Monitoring**
- **Server Status**: Live health monitoring of the NPU server
- **Device Detection**: Shows NPU/CPU device status
- **Model Count**: Displays available Whisper models
- **Session Status**: Tracks recording and processing states

### ⚙️ **Settings Management**
- **Hyperparameter Control**: Adjust all AI model parameters from the web UI
- **Audio Processing Settings**: Configure buffer duration, inference intervals, sample rates
- **Speaker Diarization**: Enable/disable and configure speaker identification
- **Performance Tuning**: Queue sizes, memory limits, processing intervals
- **Environment Variables**: Control OpenVINO device preferences and logging
- **Export/Import**: Save and load configuration files

### 🔧 **Server Management**
- **Restart Server**: Gracefully restart the backend with new settings
- **Start/Stop Control**: Manage server lifecycle from the frontend
- **Health Monitoring**: Real-time server status and performance metrics
- **Configuration Display**: View current settings and runtime information

### 🎤 **Audio Recording**
- **One-click Recording**: Simple start/stop recording interface
- **Device Selection**: Choose from available audio input devices
- **Sample Rate Control**: Configure audio quality (16kHz, 44.1kHz, 48kHz)
- **Model Selection**: Pick from available Whisper models

### 📝 **Live Transcription**
- **Real-time Display**: See transcriptions as they complete
- **Timestamped Results**: Each transcription includes timestamp
- **Session History**: Keep track of multiple recordings
- **Clear Interface**: Easy-to-read transcription results

### 🤖 **Model Management**
- **Model Browser**: View all available Whisper models
- **Quick Selection**: One-click model switching
- **Status Indicators**: See which models are ready to use
- **Auto-refresh**: Automatically updates model list

### 📋 **Activity Logs**
- **Real-time Logging**: See all system activities
- **Color-coded Entries**: Different colors for INFO, SUCCESS, WARNING, ERROR
- **Auto-scroll**: Always shows latest log entries
- **Clear Function**: Easy log management

## 🎨 **Interface Layout**

```
┌─────────────────────────────────────────────────────────┐
│ 🧠 Whisper NPU Server    Device: NPU | Models: 8 | ●   │
├─────────────────────┬───────────────────────────────────┤
│                     │ 🤖 Available Models              │
│ 🎤 Live             │ ┌─────────────────────────────┐   │
│ Transcription       │ │ whisper-medium.en   [Select]│   │
│                     │ │ whisper-small       [Select]│   │
│ [Transcription      │ │ whisper-base        [Select]│   │
│  results appear     │ └─────────────────────────────┘   │
│  here with          ├───────────────────────────────────┤
│  timestamps]        │ 📋 Activity Logs                  │
│                     │ ┌─────────────────────────────┐   │
├─────────────────────┤ │ [12:34:56] Server connected │   │
│ ⚡ Server Status    │ │ [12:35:01] Model loaded     │   │
│ ┌─────┬─────┬─────┐ │ │ [12:35:05] Recording start  │   │
│ │ ✓   │ NPU │  8  │ │ └─────────────────────────────┘   │
│ │Hlthy│ Dev │Mdls │ │                                   │
│ └─────┴─────┴─────┘ │                                   │
├─────────────────────┴───────────────────────────────────┤
│ 🎤 [Record] | Model: [Select] | Device: [Select] | Rate │
└─────────────────────────────────────────────────────────┘
```

## ⚙️ **Settings Page**

Access the comprehensive settings page at `/settings.html` to configure:

### **Server Management**
- Start/Stop/Restart server
- View server status and uptime
- Export current configuration

### **Model Configuration**
- Default model selection
- Device preference (NPU/GPU/CPU)
- Minimum inference interval for NPU protection

### **Audio Processing**
- Buffer duration (3-30 seconds)
- Inference interval (1-10 seconds)
- Sample rate selection
- Voice Activity Detection settings

### **Speaker Diarization**
- Enable/disable speaker identification
- Number of speakers (auto-detect or fixed)
- Embedding method (Resemblyzer, SpeechBrain)
- Clustering algorithm configuration
- Speech enhancement controls

### **Advanced Settings**
- Request queue limits
- Transcription history size
- Logging configuration
- OpenVINO environment variables

## 🐳 **Docker Configuration**

The project includes enhanced Docker support for Windows:

### **Health Checks**
All services include health monitoring:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### **Windows-Specific Settings**
```yaml
environment:
  - DOCKER_HOST_OS=windows
  - ENABLE_GPU_ACCELERATION=false
  - OPENVINO_DEVICE=NPU
```

### **Volume Mounting**
Properly configured for Windows paths:
```yaml
volumes:
  - ${USERPROFILE}/.whisper/models:/root/.whisper/models
```

## 🔧 **Configuration**

The frontend automatically detects and connects to:
- **NPU Server**: `http://localhost:8009`
- **Settings API**: `/settings`
- **Health Endpoint**: `/health`
- **Models Endpoint**: `/models`
- **Transcription**: `/transcribe` and `/transcribe/<model>`
- **Server Control**: `/restart`, `/shutdown`, `/start`

## 📱 **Mobile Support**

The interface is responsive and works on mobile devices:
- Touch-friendly controls
- Adaptive layout
- Mobile audio recording support

## 🛠 **API Integration**

The frontend integrates with these NPU server endpoints:

### Health Check
```http
GET /health
Response: {
  "status": "healthy",
  "device": "NPU",
  "models_available": 8
}
```

### Settings Management
```http
GET /settings
Response: {
  "default_model": "whisper-base",
  "device_preference": "NPU",
  "buffer_duration": 6.0,
  // ... all configuration parameters
}

POST /settings
Body: { "buffer_duration": 8.0, "device_preference": "CPU" }
```

### Server Control
```http
POST /restart
POST /shutdown  
POST /start
```

### List Models
```http
GET /models
Response: {
  "models": ["whisper-medium.en", "whisper-small", ...],
  "device": "NPU"
}
```

### Transcribe Audio
```http
POST /transcribe/whisper-medium.en
Body: [audio data]
Response: {
  "text": "transcribed text here"
}
```

## 🔧 **Troubleshooting**

### Server Not Responding
1. Check if NPU container is running: `docker ps`
2. Check container logs: `docker logs whisper-server-npu`
3. Restart NPU server: Use the settings page restart button or `docker-compose restart`

### No Models Available
1. Ensure models are mounted at `/root/.whisper/models`
2. Check if models directory contains model folders
3. Refresh models list in the interface

### Recording Issues
1. Check browser microphone permissions
2. Try different audio devices
3. Check browser console for errors
4. Ensure HTTPS for production use (required for microphone access)

### NPU Not Detected
On Windows with Docker Desktop, NPU device passthrough is limited. The server will automatically fall back to CPU with a warning message.

### Settings Not Saving
1. Check server logs for permission errors
2. Verify the server has write access to configuration files
3. Try restarting the server from the settings page

## 🪟 **Windows Docker Desktop Setup**

1. **Enable WSL2 Backend**
   - Docker Desktop → Settings → General → Use WSL 2 based engine

2. **Configure Resource Limits**
   - Docker Desktop → Settings → Resources
   - Allocate at least 4GB RAM for NPU processing

3. **Start Services**
```powershell
# Basic NPU server
docker-compose -f docker-compose.npu.yml up -d whisper-npu-server

# With frontend
docker-compose -f docker-compose.npu.yml --profile frontend up -d
```

4. **Access Points**
   - Server API: http://localhost:8009
   - Frontend: http://localhost:8080
   - Settings: http://localhost:8080/settings.html

## 🎯 **Next Steps**

1. **Model Conversion**: Convert Whisper models to OpenVINO IR format for better NPU support
2. **Linux Testing**: Test on native Linux for full NPU device access
3. **Performance Monitoring**: Add detailed performance metrics and bottleneck analysis
4. **Batch Processing**: Support for processing multiple files
5. **Real-time Streaming**: WebSocket-based continuous transcription

## 📄 **Files**

- `index.html` - Main transcription interface
- `test-audio.html` - Audio testing and debugging
- `settings.html` - **NEW** Comprehensive settings management
- `js/settings.js` - **NEW** Settings page functionality
- `css/settings.css` - **NEW** Settings page styling
- `serve-frontend.py` - Simple HTTP server for development
- `docker-compose.npu.yml` - **UPDATED** Enhanced Windows Docker support
- `Dockerfile.frontend` - **NEW** Dedicated frontend container
- `nginx.conf` - **NEW** Production-ready web server configuration

---

**Enjoy your NPU-accelerated transcription experience with full control!** 🚀🔧 