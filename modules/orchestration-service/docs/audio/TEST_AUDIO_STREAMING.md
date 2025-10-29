# Audio Streaming Test Guide

This test script bypasses the frontend UI complexity and streams audio directly through the orchestration service, **following the exact same pattern as bot containers**.

## Architecture

```
┌─────────────┐
│   ffmpeg    │ ← Audio source (mic, file, stdin)
└──────┬──────┘
       │ Raw audio (16kHz, mono, S16LE)
       ↓
┌──────────────────────────┐
│  test_audio_streaming.py │ ← This test script
└──────┬───────────────────┘
       │ WebSocket + base64 audio chunks
       ↓
┌─────────────────────────┐
│ Orchestration Service   │ ← ws://localhost:3000/api/audio/stream
└──────┬──────────────────┘
       │ Socket.IO
       ↓
┌─────────────────────────┐
│   Whisper Service       │ ← Flask-SocketIO on port 5001
└──────┬──────────────────┘
       │ Transcription segments
       ↓
┌─────────────────────────┐
│   Console Output        │ ← Real-time transcription display
└─────────────────────────┘
```

**This is IDENTICAL to the bot pattern:**
```
Bot Container → Orchestration → Whisper
```

## Prerequisites

```bash
# Make sure services are running
cd modules/orchestration-service
poetry run python src/main_fastapi.py  # Port 3000

cd modules/whisper-service
PYTHONPATH=src poetry run python src/api_server.py --host 0.0.0.0 --port 5001
```

## Usage Examples

### 1. Test with Microphone (Real-time)

**macOS:**
```bash
cd modules/orchestration-service
python test_audio_streaming.py --mic
```

This will:
- Use ffmpeg to capture from your default microphone
- Convert to 16kHz mono
- Stream 100ms chunks to orchestration service
- Display real-time transcriptions

**Linux:**
```bash
python test_audio_streaming.py --mic
```

**Windows:**
```bash
python test_audio_streaming.py --mic
```

### 2. Test with Audio File

```bash
# Test with any audio format (mp3, wav, m4a, etc.)
python test_audio_streaming.py --file audio.mp3

# Test with specific model and language
python test_audio_streaming.py --file spanish.mp3 --model whisper-large-v3 --language es
```

### 3. Test with stdin (Advanced)

```bash
# Pipe from ffmpeg
ffmpeg -i input.mp3 -f s16le -ar 16000 -ac 1 - | python test_audio_streaming.py --stdin

# Pipe from arecord (Linux)
arecord -f S16_LE -c1 -r 16000 -t raw -D default | python test_audio_streaming.py --stdin

# Pipe from existing raw audio
cat raw_audio.raw | python test_audio_streaming.py --stdin
```

## Configuration Options

### Basic Options
```bash
--mic                    # Stream from microphone
--file FILE             # Stream from audio file
--stdin                 # Stream from stdin

--url URL               # Orchestration WebSocket URL (default: ws://localhost:3000/api/audio/stream)
--chunk-duration MS     # Chunk duration in milliseconds (default: 100)
--session-id ID         # Session identifier (auto-generated if not provided)
```

### Whisper Configuration
```bash
--model MODEL           # Whisper model (default: whisper-base)
                       # Options: whisper-tiny, whisper-base, whisper-small,
                       #          whisper-medium, whisper-large-v3

--language LANG         # Source language (default: en)
                       # Options: en, es, fr, de, it, pt, ru, ja, ko, zh, etc.
```

### Feature Flags
```bash
--no-vad               # Disable Voice Activity Detection
--no-diarization       # Disable speaker diarization
--no-cif               # Disable CIF (end-of-word detection)
```

### Debugging
```bash
--debug                # Enable debug logging
```

## Example Commands

### Quick Test (Microphone)
```bash
python test_audio_streaming.py --mic
```

### Test with Spanish Audio File
```bash
python test_audio_streaming.py --file spanish_meeting.mp3 --language es --model whisper-large-v3
```

### Test with Custom Chunk Size
```bash
# Use 200ms chunks instead of 100ms
python test_audio_streaming.py --mic --chunk-duration 200
```

### Test Without Speaker Diarization
```bash
python test_audio_streaming.py --file audio.wav --no-diarization
```

### Debug Mode
```bash
python test_audio_streaming.py --mic --debug
```

## Output Format

The script displays transcription segments in real-time:

```
================================================================================
✅ FINAL | 👤 SPEAKER_00 | 🌐 EN | 📊 95.3%
📝 Hello, this is a test of the audio streaming system.
================================================================================

================================================================================
⏳ PARTIAL | 👤 SPEAKER_01 | 🌐 EN | 📊 87.2%
📝 I can see that the
================================================================================

================================================================================
✅ FINAL | 👤 SPEAKER_01 | 🌐 EN | 📊 92.8%
📝 I can see that the transcription is working correctly.
================================================================================

================================================================================
📊 STREAMING STATISTICS
================================================================================
Duration:          30.2s
Chunks sent:       302
Segments received: 8
Chunk rate:        10.0 chunks/sec
================================================================================
```

## Troubleshooting

### "Connection refused"
- Make sure orchestration service is running on port 3000
- Check: `curl http://localhost:3000/health`

### "No audio input"
- **macOS**: Check System Preferences → Security & Privacy → Microphone
- **Linux**: Check `arecord -l` to list audio devices
- **Windows**: Check audio device in Sound settings

### "ffmpeg not found"
```bash
# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### "No transcriptions appearing"
- Check Whisper service is running: `curl http://localhost:5001/health`
- Enable debug mode: `--debug`
- Verify audio format is correct (16kHz, mono, S16LE)

### "Authentication failed"
- Check orchestration service logs
- Verify WebSocket endpoint is `/api/audio/stream`

## How It Works

1. **Audio Capture**: ffmpeg captures audio from mic/file and converts to raw S16LE format
2. **Chunking**: Script reads audio in 100ms chunks (configurable)
3. **Encoding**: Each chunk is base64-encoded (same as frontend/bots)
4. **WebSocket Protocol**:
   - Connect to `ws://localhost:3000/api/audio/stream`
   - Send `authenticate` message
   - Send `start_session` message with Whisper config
   - Send `audio_chunk` messages with base64 audio
   - Receive `segment` messages with transcriptions
   - Send `end_session` message when done
5. **Display**: Pretty-print segments with speaker, language, confidence

## Comparison to Frontend UI

| Feature | Frontend UI | Test Script |
|---------|-------------|-------------|
| Audio Source | Browser MediaRecorder | ffmpeg (any source) |
| WebSocket Protocol | ✅ Same | ✅ Same |
| Message Format | ✅ Same | ✅ Same |
| Audio Format | ✅ 16kHz mono | ✅ 16kHz mono |
| Chunk Duration | ✅ 100ms | ✅ 100ms (configurable) |
| Base64 Encoding | ✅ Yes | ✅ Yes |
| Orchestration Path | ✅ Yes | ✅ Yes |
| Bot Pattern | ✅ Same | ✅ Same |

**Result**: Test script follows the EXACT same pattern as bots and frontend, ensuring we're testing the real production code path.

## SimulStreaming Compatibility

This test script is inspired by [SimulStreaming](../../reference/SimulStreaming/), the state-of-the-art simultaneous translation system:

**SimulStreaming Pattern:**
```bash
arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 43001
```

**Our Pattern:**
```bash
# Direct equivalent
arecord -f S16_LE -c1 -r 16000 -t raw -D default | python test_audio_streaming.py --stdin

# Or simplified
python test_audio_streaming.py --mic
```

Both use:
- 16kHz sampling rate
- Mono channel (1 channel)
- S16LE format (signed 16-bit little endian)
- Real-time streaming with small chunks
- TCP/WebSocket streaming protocol

## Next Steps

After verifying the pipeline works with this test script:

1. **Verify end-to-end flow**: Mic → Orchestration → Whisper → Transcription
2. **Test different models**: tiny, base, small, medium, large-v3
3. **Test multiple languages**: en, es, fr, de, etc.
4. **Stress test**: Long audio files, multiple simultaneous streams
5. **Fix any issues**: Then return to frontend UI with confidence

This approach removes the complexity of the browser UI and lets us focus on the core audio processing pipeline!
