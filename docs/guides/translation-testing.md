# LiveTranslate Translation Testing Guide

Quick guide to test the full translation pipeline with loopback audio.

## 🎯 What Gets Tested

The **complete real-time translation flow**:
```
Loopback Audio → Orchestration → Whisper → Translation → Display
```

## 🚀 Quick Start (3 Terminals)

### Terminal 1: Orchestration Service
```bash
cd modules/orchestration-service
python src/main.py
```
**Port:** 3000
**Role:** Coordinates all services, handles audio upload

### Terminal 2: Whisper Service
```bash
cd modules/whisper-service
python src/main.py
```
**Port:** 5001
**Role:** Speech-to-text transcription with automatic hardware detection (NPU/GPU/CPU)

### Terminal 3: Translation Service
```bash
cd modules/translation-service
# One-time install
pip install python-dotenv fastapi uvicorn httpx langdetect structlog

# Start service
python src/api_server_fastapi.py
```
**Port:** 5003
**Role:** Multi-language translation with Ollama/Groq backend

### Terminal 4: Run Test
```bash
# Install test dependencies (one-time)
pip install pyaudio httpx

# Run full-stack test
python test_loopback_fullstack.py
```

## 📋 Test Scripts

### `test_loopback_fullstack.py` (Recommended)
- **What:** Full orchestration pipeline test
- **How:** Sends via `/api/audio/upload` like the frontend
- **Languages:** Spanish, French, German
- **Audio:** 5-second chunks from loopback or mic

### `test_loopback_translation.py` (Direct Service Test)
- **What:** Direct service-to-service testing
- **How:** Bypasses orchestration, calls Whisper + Translation directly
- **Use:** Debug individual services

## 🎤 Audio Setup

### macOS - BlackHole (Recommended)
```bash
# Install
brew install blackhole-2ch

# Setup
System Settings → Sound → Input → BlackHole 2ch
```

Then play any audio/video on your system - it will be captured and translated!

### Without Loopback Device
The test will use your **default microphone**. Just speak into your mic.

## ✅ Expected Output

```
🔍 Checking services...
✅ Orchestration: READY
✅ Whisper: READY
✅ Translation: READY (backend: ollama)

🎙️  LISTENING... (Ctrl+C to stop)

================================================================================
🎵 CHUNK 1 | 19:45:23
================================================================================
📤 Sending to orchestration: /api/audio/upload
✅ Transcribed (en): "Hello everyone, welcome to the meeting."

🌐 Translations:
   [es] "Hola a todos, bienvenidos a la reunión."
           (confidence: 0.92)
   [fr] "Bonjour à tous, bienvenue à la réunion."
           (confidence: 0.90)
   [de] "Hallo zusammen, willkommen zum Meeting."
           (confidence: 0.88)

📊 Stats: 1 chunks, 3 translations, 5.2s
```

## 🔧 Configuration

Edit target languages in `test_loopback_fullstack.py`:
```python
TARGET_LANGUAGES = ["es", "fr", "de"]  # Spanish, French, German
```

Change chunk duration:
```python
CHUNK_DURATION = 5  # seconds (default)
```

## 🐛 Troubleshooting

### "Services not ready"
- Check each service is running in its terminal
- Visit health endpoints:
  - http://localhost:3000/api/health
  - http://localhost:5001/health
  - http://localhost:5003/api/health

### "Translation backend unavailable"
- Make sure Ollama is running on your network
- Check `.env` in `modules/translation-service`:
  ```
  OLLAMA_BASE_URL=http://192.168.1.239:11434/v1
  OLLAMA_MODEL=mistral:latest
  ```

### "No loopback device found"
- Install BlackHole: `brew install blackhole-2ch`
- Or use default mic (will work but capture mic input instead)

### "ModuleNotFoundError: No module named 'pyaudio'"
```bash
pip install pyaudio httpx
```

On macOS with Apple Silicon, if pyaudio fails:
```bash
brew install portaudio
pip install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio
```

## 📚 Next Steps

After testing translation:
- **Frontend Integration:** The frontend uses the same `/api/audio/upload` endpoint
- **Real-time Dashboard:** Visit http://localhost:3000 for the web UI
- **Database Queries:** Use `/api/data/query` for translation history
- **Virtual Webcam:** Test Google Meet bot with translation overlay

## 🎯 Architecture

```
┌──────────────┐
│ Loopback/Mic │
└──────┬───────┘
       │ Raw Audio (16kHz)
       ↓
┌─────────────────────┐
│ Orchestration :3000 │ ← POST /api/audio/upload
└──────┬──────────────┘
       │ Forward audio
       ↓
┌────────────────┐
│ Whisper :5001  │ ← Transcribe (NPU/GPU/CPU)
└──────┬─────────┘
       │ Text + Language
       ↓
┌──────────────────┐
│ Translation :5003│ ← Translate to multiple languages
└──────┬───────────┘
       │ Translations
       ↓
┌──────────────┐
│ Display       │ ← Print results
└───────────────┘
```

This is the **exact same flow** the frontend uses!
