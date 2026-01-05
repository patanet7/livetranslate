# LiveTranslate Examples

Demonstration scripts and example usage for LiveTranslate features.

## Available Examples

### Orchestration Service Examples

#### 1. Streaming Integration Demo (`demo_streaming_integration.py`)
**Purpose**: Demonstrates real-time audio streaming through the complete pipeline.

**Features:**
- Audio capture with configurable chunk sizes
- Orchestration API integration
- Real-time transcription display
- Multi-language translation output
- Session tracking

**Usage:**
```bash
# Start required services first
cd modules/orchestration-service && python src/main.py &
cd modules/whisper-service && python src/main.py &
cd modules/translation-service && python src/api_server_fastapi.py &

# Run demo
cd examples
python demo_streaming_integration.py
```

**Configuration:**
```python
CHUNK_DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz
TARGET_LANGUAGES = ["es", "fr", "de"]  # Spanish, French, German
```

#### 2. Virtual Webcam Live Demo (`demo_virtual_webcam_live.py`)
**Purpose**: Demonstrates virtual webcam with live translation overlays.

**Features:**
- Real-time webcam video feed
- Translation subtitle overlay
- Speaker attribution display
- Confidence scores and timestamps
- Professional layout with dual content (transcription + translation)

**Usage:**
```bash
# Requires virtual webcam device (OBS Virtual Camera, etc.)
cd examples
python demo_virtual_webcam_live.py
```

**Output:**
- 1280x720 video frames @ 30fps
- Translation overlays with speaker names
- Both original transcription and translations displayed
- Configurable content duration (default: 5 seconds)

## Example Categories

### Audio Processing
- **Loopback Audio Capture**: Examples using BlackHole or system audio
- **Microphone Input**: Real-time microphone processing
- **File Processing**: Batch audio file translation

### Translation Workflows
- **Chinese→English**: Specialized CN→EN translation examples
- **Multi-language**: Simultaneous translation to multiple target languages
- **Quality Testing**: Translation accuracy and confidence scoring

### Integration Patterns
- **REST API**: Direct API endpoint usage
- **WebSocket**: Real-time WebSocket streaming
- **Database**: Session persistence and retrieval

## Prerequisites

### System Requirements
- **Python**: 3.10+
- **Audio Device**: Microphone or BlackHole (for loopback)
- **Virtual Camera**: OBS Virtual Camera (optional, for webcam demos)

### Service Dependencies
All examples require running services:

```bash
# Terminal 1: Orchestration
cd modules/orchestration-service
python src/main.py

# Terminal 2: Whisper
cd modules/whisper-service
python src/main.py

# Terminal 3: Translation
cd modules/translation-service
python src/api_server_fastapi.py
```

### Python Dependencies
```bash
# Install from orchestration service requirements
cd modules/orchestration-service
pip install -r requirements.txt

# Additional for audio examples
pip install pyaudio librosa soundfile
```

## Running Examples

### Basic Workflow

1. **Start All Services:**
   ```bash
   ./start-development.ps1  # Windows
   # Or manually start each service in separate terminals
   ```

2. **Verify Services Ready:**
   ```bash
   curl http://localhost:3000/api/health  # Orchestration
   curl http://localhost:5001/health      # Whisper
   curl http://localhost:5003/api/health  # Translation
   ```

3. **Run Example:**
   ```bash
   cd examples
   python demo_streaming_integration.py
   ```

### Chinese→English Translation Example

```bash
# 1. Start services with Chinese language support
cd modules/whisper-service
python src/main.py --device=npu  # or gpu/cpu

# 2. Configure translation service for CN→EN
cd modules/translation-service
python src/api_server_fastapi.py

# 3. Run streaming demo with Chinese audio
cd examples
python demo_streaming_integration.py

# 4. Play Chinese audio on your system or speak into microphone
```

## Example Output

### Streaming Integration Demo Output:
```
================================================================================
  🎤 STREAMING INTEGRATION DEMO
  Loopback → Orchestration → Whisper → Translation
================================================================================

🔍 Checking services...
✅ Orchestration: READY
✅ Whisper: READY
✅ Translation: READY (backend: ollama)

🎙️  LISTENING... (Ctrl+C to stop)
   Session: stream_demo_1704448800
   Languages: es, fr, de
   Chunk size: 5s

================================================================================
🎵 CHUNK 0001 | 14:30:15
================================================================================
📤 Sending to orchestration: /api/audio/upload
✅ Transcribed (zh): "你好世界"

🌐 Translations:
   [es] "Hola mundo"
           (confidence: 0.95)
   [fr] "Bonjour le monde"
           (confidence: 0.93)
   [de] "Hallo Welt"
           (confidence: 0.94)

📊 Stats: 1 chunks, 3 translations, 5.2s
```

### Virtual Webcam Demo Output:
```
================================================================================
  📹 VIRTUAL WEBCAM LIVE DEMO
  Real-time Translation Overlay
================================================================================

✅ Virtual webcam initialized (1280x720 @ 30fps)
🎙️  Starting audio capture...

Frame 0001 | Speaker: John Doe (SPEAKER_00)
   🎤 Original: "Hello everyone, welcome to the meeting"
   🌐 Spanish: "Hola a todos, bienvenidos a la reunión"
   Confidence: 0.96 | Duration: 3.2s

Frame 0002 | Speaker: Jane Smith (SPEAKER_01)
   🎤 Original: "Thank you, let's begin"
   🌐 Spanish: "Gracias, comencemos"
   Confidence: 0.94 | Duration: 2.1s
```

## Customization

### Modify Audio Settings
```python
# In demo scripts
SAMPLE_RATE = 16000  # Whisper standard
CHUNK_DURATION = 5   # Adjust for responsiveness vs. accuracy
CHANNELS = 1         # Mono audio
```

### Change Target Languages
```python
# Multi-language
TARGET_LANGUAGES = ["es", "fr", "de", "ja", "ko"]

# Chinese→English only
TARGET_LANGUAGES = ["en"]
```

### Adjust Translation Quality
```python
# In API calls
data = {
    'target_languages': json.dumps(TARGET_LANGUAGES),
    'whisper_model': 'whisper-base',  # or whisper-small, whisper-medium
    'enable_diarization': 'true',     # Speaker identification
}
```

## Troubleshooting

### No Audio Detected
```bash
# List audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"

# Install BlackHole for loopback
brew install blackhole-2ch
```

### Service Connection Errors
```bash
# Check service ports
lsof -i :3000  # Orchestration
lsof -i :5001  # Whisper
lsof -i :5003  # Translation
```

### Translation Quality Issues
- Use larger Whisper models (`whisper-small` or `whisper-medium`)
- Enable speaker diarization for multi-speaker scenarios
- Increase chunk duration for better context (5-10 seconds)

## Creating New Examples

When creating new examples:

1. **File Naming**: Use `demo_<feature>_<variant>.py` format
2. **Documentation**: Include docstring with purpose, features, usage
3. **Service Checks**: Always verify services are running before proceeding
4. **Error Handling**: Graceful error messages with troubleshooting hints
5. **Output**: Clean, formatted output with emojis for readability

### Example Template
```python
#!/usr/bin/env python3
"""
EXAMPLE: <Feature Name>

Purpose: Brief description of what this example demonstrates

Features:
- Feature 1
- Feature 2

Usage:
    python demo_example.py
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("\\n" + "="*80)
    print("  🎯 EXAMPLE: <Feature Name>")
    print("="*80 + "\\n")

    # Check services
    # Run example logic
    # Display results

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

## Contributing

To add new examples:
1. Create demo script in `examples/` directory
2. Follow naming convention (`demo_*.py`)
3. Include comprehensive docstring
4. Update this README with example description
5. Test with all required services running
