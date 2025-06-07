# 🪟 Whisper NPU Server - Windows Setup Guide

Complete guide for running Whisper NPU Server on Windows 11 with Docker Desktop, including the new settings management interface and server control capabilities.

## 🚀 Quick Start (Recommended)

### 1. **One-Command Setup**
```powershell
# Start NPU server with frontend
.\start-windows.ps1 -Frontend

# Access points:
# - Frontend: http://localhost:8080
# - Settings: http://localhost:8080/settings.html  
# - API: http://localhost:8009
```

### 2. **Alternative Modes**
```powershell
# CPU fallback mode
.\start-windows.ps1 -Mode cpu -Frontend

# Pre-built image fallback
.\start-windows.ps1 -Mode fallback -Frontend
```

## 📋 Prerequisites

### **Windows 11 Requirements**
- Windows 11 with WSL2 enabled
- Intel Core Ultra processor (for NPU support)
- At least 8GB RAM (16GB recommended)
- 10GB free disk space

### **Software Requirements**
1. **Docker Desktop for Windows**
   ```powershell
   # Download from: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
   # Ensure WSL2 backend is enabled
   ```

2. **PowerShell 5.1 or later** (included with Windows 11)

3. **Optional: Visual Studio Code** for development

## 🔧 Installation Steps

### **Step 1: Setup Docker Desktop**

1. **Install Docker Desktop**
   - Download and install Docker Desktop for Windows
   - Enable WSL2 integration during setup

2. **Configure Docker Settings**
   ```
   Docker Desktop → Settings → General
   ✅ Use WSL 2 based engine
   
   Docker Desktop → Settings → Resources
   Memory: 8GB+ (for NPU processing)
   CPU: 4+ cores
   ```

3. **Verify Installation**
   ```powershell
   docker --version
   docker-compose --version
   ```

### **Step 2: Clone and Setup**

1. **Navigate to Project Directory**
   ```powershell
   cd C:\Users\patan\Projects\livetranslate\livetranslate\external\whisper-npu-server
   ```

2. **Create Models Directory**
   ```powershell
   mkdir "$env:USERPROFILE\.whisper\models" -Force
   ```

3. **Make PowerShell Script Executable**
   ```powershell
   # If execution policy blocks the script
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### **Step 3: Launch Services**

1. **Start with Settings Interface**
   ```powershell
   .\start-windows.ps1 -Frontend
   ```

2. **Verify Services**
   ```powershell
   .\start-windows.ps1 -Status
   ```

## 🎯 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:8080 | Main web interface with navigation |
| **Settings** | http://localhost:8080/settings.html | **NEW** Comprehensive settings management |
| **API Direct** | http://localhost:8009 | Direct server API access |
| **Health Check** | http://localhost:8009/health | Server health status |

## ⚙️ Settings Management (NEW)

The new settings page provides comprehensive control over all server parameters:

### **🔧 Server Management**
- **Restart Server**: Gracefully restart backend with new settings
- **Start/Stop Control**: Manage server lifecycle
- **Status Monitoring**: Real-time health and performance metrics
- **Configuration Export**: Save settings to JSON file

### **🤖 Model Configuration**
- **Default Model Selection**: Choose from available Whisper models
- **Device Preference**: NPU/GPU/CPU priority
- **Inference Intervals**: Control processing timing for NPU protection

### **🎵 Audio Processing**
- **Buffer Duration**: 3-30 seconds of rolling audio buffer
- **Sample Rate**: 8kHz to 48kHz audio quality settings
- **Voice Activity Detection**: Automatic speech detection configuration

### **🎤 Speaker Diarization**
- **Enable/Disable**: Toggle speaker identification
- **Speaker Count**: Auto-detect or specify number of speakers
- **Algorithms**: Choose between Resemblyzer/SpeechBrain embeddings
- **Clustering Methods**: HDBSCAN for auto-detection, Agglomerative for fixed speakers

### **🔬 Advanced Settings**
- **Performance Tuning**: Queue sizes, memory limits
- **Logging Configuration**: Debug levels and file logging
- **Environment Variables**: OpenVINO device overrides

## 🐳 Docker Architecture

### **Service Overview**
```yaml
whisper-npu-server:     # Main NPU-accelerated server (Port 8009)
whisper-npu-server-cpu: # CPU fallback (Port 8010)  
whisper-frontend:       # Nginx frontend proxy (Port 8080)
```

### **Health Monitoring**
All services include automatic health checks:
- **Interval**: Every 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3 attempts
- **Startup Grace**: 40 seconds

### **Volume Mapping**
```
Host: %USERPROFILE%\.whisper\models
Container: /root/.whisper/models
```

## 🛠 Management Commands

### **Service Control**
```powershell
# Start services
.\start-windows.ps1                      # NPU mode
.\start-windows.ps1 -Mode cpu            # CPU mode  
.\start-windows.ps1 -Frontend            # With web interface

# Control services
.\start-windows.ps1 -Stop                # Stop all
.\start-windows.ps1 -Restart            # Restart all
.\start-windows.ps1 -Status             # Show status

# Debugging
.\start-windows.ps1 -Logs               # View logs
```

### **Docker Commands**
```powershell
# Manual service management
docker-compose -f docker-compose.npu.yml ps                    # Status
docker-compose -f docker-compose.npu.yml logs whisper-npu-server  # Logs
docker-compose -f docker-compose.npu.yml restart whisper-npu-server # Restart

# Cleanup
docker-compose -f docker-compose.npu.yml down --volumes        # Full cleanup
docker system prune -a                                         # Clean Docker cache
```

## 🔍 Troubleshooting

### **NPU Not Detected**
**Symptom**: Server falls back to CPU even with NPU hardware

**Solutions**:
1. **Check Processor Compatibility**
   ```powershell
   # Verify Intel Core Ultra processor
   Get-WmiObject -Class Win32_Processor | Select-Object Name
   ```

2. **Docker Device Passthrough Limitation**
   - Windows Docker Desktop has limited NPU device passthrough
   - This is expected behavior - CPU fallback is automatic
   - For native NPU support, consider Linux deployment

3. **Force NPU Mode** (for testing)
   ```powershell
   # Set environment variable
   $env:OPENVINO_DEVICE="NPU"
   .\start-windows.ps1
   ```

### **Container Startup Issues**

**Symptom**: Services fail to start or immediately exit

**Solutions**:
1. **Check Docker Resources**
   ```powershell
   # Increase memory allocation in Docker Desktop settings
   # Minimum: 8GB RAM, 4 CPU cores
   ```

2. **Verify Models Directory**
   ```powershell
   # Ensure directory exists and is accessible
   Test-Path "$env:USERPROFILE\.whisper\models"
   ```

3. **Check Port Conflicts**
   ```powershell
   # Find processes using ports 8009, 8080
   netstat -ano | findstr ":8009"
   netstat -ano | findstr ":8080"
   ```

### **Frontend Not Loading**

**Symptom**: Can't access http://localhost:8080

**Solutions**:
1. **Start with Frontend Profile**
   ```powershell
   .\start-windows.ps1 -Frontend
   ```

2. **Check Service Status**
   ```powershell
   .\start-windows.ps1 -Status
   ```

3. **Direct API Access**
   ```powershell
   # Test direct server connection
   curl http://localhost:8009/health
   ```

### **Settings Not Saving**

**Symptom**: Configuration changes don't persist

**Solutions**:
1. **Check Server Logs**
   ```powershell
   .\start-windows.ps1 -Logs
   ```

2. **Verify Container Permissions**
   ```powershell
   # Restart with fresh container
   .\start-windows.ps1 -Restart
   ```

3. **Use Settings Page Restart**
   - Go to http://localhost:8080/settings.html
   - Click "🔄 Restart Server" after making changes

### **Audio Recording Issues**

**Symptom**: Microphone not working in browser

**Solutions**:
1. **HTTPS Requirement**
   - Modern browsers require HTTPS for microphone access
   - Use localhost (HTTP allowed) for development
   - For production, set up proper SSL certificates

2. **Browser Permissions**
   - Grant microphone permissions when prompted
   - Check browser settings for site permissions

3. **Audio Device Selection**
   - Use the settings page to select proper input device
   - Test with different sample rates (16kHz recommended)

## 🎯 Performance Optimization

### **For NPU Mode**
- Use 16kHz sample rate for optimal NPU performance
- Keep buffer duration between 6-10 seconds
- Set minimum inference interval to 200ms+

### **For CPU Mode**
- Increase buffer duration to 8-12 seconds
- Lower sample rate if experiencing stutters
- Adjust inference interval based on CPU performance

### **Memory Management**
- Allocate 8GB+ RAM to Docker Desktop
- Monitor container memory usage in Docker Desktop
- Use settings page to adjust queue sizes and history limits

## 🔄 Update Process

### **Update Container Images**
```powershell
# Pull latest images
docker-compose -f docker-compose.npu.yml pull

# Restart with new images
.\start-windows.ps1 -Restart
```

### **Update Configuration**
1. Export current settings from settings page
2. Update server or Docker files
3. Import settings after restart
4. Verify functionality

## 🎨 Development Setup

### **Local Development**
```powershell
# Run server natively (if Python environment available)
python server.py

# Serve frontend only
cd frontend
python -m http.server 8080
```

### **Frontend Development**
```powershell
# Start backend in container
.\start-windows.ps1

# Serve frontend locally for live editing
cd frontend  
# Edit files and refresh browser at http://localhost:8080
```

## 📊 Monitoring and Logs

### **Real-time Monitoring**
- Settings page: Real-time server status
- Docker Desktop: Container resource usage
- Windows Task Manager: Overall system performance

### **Log Analysis**
```powershell
# View all logs
.\start-windows.ps1 -Logs

# Follow logs in real-time  
docker-compose -f docker-compose.npu.yml logs -f whisper-npu-server

# Export logs for analysis
docker-compose -f docker-compose.npu.yml logs > whisper-logs.txt
```

## 🌟 What's New

### **Settings Management Interface**
- Complete hyperparameter control from web UI
- Real-time configuration updates
- Export/import configuration files
- Server restart/control capabilities

### **Enhanced Windows Docker Support**
- Improved volume mounting for Windows paths
- Health checks for all services
- Better error handling and fallback options
- PowerShell deployment script

### **Performance Improvements**
- Optimized NPU memory usage
- Better audio processing pipelines
- Reduced latency for real-time transcription

---

## 📞 Support

For issues specific to Windows deployment:
1. Check this troubleshooting guide
2. Review container logs with `.\start-windows.ps1 -Logs`
3. Test with CPU fallback mode
4. Verify Docker Desktop configuration

**Enjoy your Windows NPU-accelerated transcription experience!** 🚀🪟 