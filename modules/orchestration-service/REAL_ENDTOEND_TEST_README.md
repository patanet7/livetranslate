# REAL End-to-End Transcription → Virtual Webcam Test

## Overview

This is a **TRUE REAL** end-to-end test that uses **ACTUAL RUNNING SERVICES** - NO MOCKS!

## What It Does

### Flow
```
1. Prerequisites Check:
   ✓ Orchestration service running on port 3000?
   ✓ Whisper service running on port 5001?
   ✓ Database connection (optional)

2. Create Bot Session:
   → Uses GoogleMeetBotManager (REAL)
   → Creates test bot session (REAL)
   → Sets up virtual webcam for session (REAL)

3. Stream Real Audio:
   → Generate synthetic audio with speech content (REAL AUDIO)
   → Send chunks via HTTP POST to /api/audio/upload (REAL HTTP)

4. Real Service Processing:
   → Orchestration receives audio (REAL)
   → Orchestration → Whisper service (REAL HTTP call)
   → Whisper processes REAL audio
   → Whisper returns REAL transcription

5. Virtual Webcam Rendering:
   → Receives transcription data (REAL)
   → Renders subtitle frames (REAL)
   → Captures ALL frames (REAL)

6. Verify and Save:
   → Save frames to disk (REAL FILES)
   → Verify transcriptions appear
   → Provide ffmpeg command to create video
```

## Prerequisites

### Required Services

1. **Orchestration Service** (port 3000)
   ```bash
   cd modules/orchestration-service
   python src/main_fastapi.py
   ```

2. **Whisper Service** (port 5001)
   ```bash
   cd modules/whisper-service
   python src/main.py --device=cpu
   ```

### Optional
3. **PostgreSQL Database** (for persistence)
   - Test will work without it, but won't persist data

## Running the Test

### Quick Start
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/orchestration-service
python test_real_endtoend_transcription.py
```

### Expected Output
```
================================================================================
🎯 REAL END-TO-END TRANSCRIPTION TEST
================================================================================

📋 Prerequisites Check:
  ✅ Orchestration service: Running (http://localhost:3000)
  ✅ Whisper service: Running (http://localhost:5001)
  ℹ️  Database: Optional (test will work without it)

🎬 Creating bot session with virtual webcam...
  ✅ Session ID: bot_abc123_1234567890
  ✅ Virtual webcam initialized

🎤 Running Test Scenarios:
--------------------------------------------------------------------------------

🎤 Scenario 1: Single Transcription
  ▶ Generating audio: "Hello, this is a test transcription"
  ▶ Uploading to orchestration service...
  ✅ Upload successful
  ✅ Received transcription: "Hello, this is a test transcription"
  ✅ Frames captured: 90

🎤 Scenario 2: Continuous Stream (5 chunks)
  ▶ Chunk 1: "Welcome to the meeting"
  ▶ Chunk 2: "Let's discuss the quarterly results"
  ▶ Chunk 3: "Our revenue increased by thirty five percent"
  ▶ Chunk 4: "The team did an excellent job"
  ▶ Chunk 5: "Looking forward to next quarter"
  ✅ All 5 chunks uploaded
  ✅ Frames captured: 450

🎤 Scenario 3: Rapid Fire (3 chunks)
  ▶ Chunk 1: "First message"
  ▶ Chunk 2: "Second message"
  ▶ Chunk 3: "Third message"
  ✅ 3/3 chunks uploaded successfully
  ✅ Frames captured: 540

📊 Test Results:
--------------------------------------------------------------------------------
  Total duration: 45.2s
  Frames saved: 540
  Transcriptions verified: 6

  Output directory: /path/to/test_output/real_endtoend_test
  First frame: frame_0000.png
  Last frame: frame_0539.png

  Transcriptions received:
    [  3.12s] chunk_1: "Hello, this is a test transcription"
    [  8.45s] chunk_2_1: "Welcome to the meeting"
    [ 11.23s] chunk_2_2: "Let's discuss the quarterly results"
    ...

🎬 Create Video:
  cd /path/to/test_output/real_endtoend_test
  ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
         -c:v libx264 -pix_fmt yuv420p \
         output.mp4

================================================================================
✅ REAL END-TO-END TEST COMPLETE!
================================================================================
```

## Test Scenarios

### Scenario 1: Single Transcription
- Uploads 1 audio chunk with speech
- Waits for transcription
- Verifies it appears on webcam
- Saves frames (~90 frames for 3 seconds @ 30fps)

### Scenario 2: Continuous Stream
- Uploads 5 audio chunks sequentially
- Each with different speech content
- Verifies all transcriptions appear
- Verifies frame saving works continuously (~450 frames)

### Scenario 3: Rapid Fire
- Uploads 3 chunks quickly (concurrent)
- Simulates real meeting scenario
- Verifies webcam handles rapid updates
- Verifies no frames dropped (~90 frames)

## Output

### Frames
All frames are saved to:
```
test_output/real_endtoend_test/frame_0000.png
test_output/real_endtoend_test/frame_0001.png
...
test_output/real_endtoend_test/frame_0539.png
```

### Video Creation
Use ffmpeg to create a video from the frames:
```bash
cd test_output/real_endtoend_test
ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
       -c:v libx264 -pix_fmt yuv420p \
       output.mp4
```

## What Makes This REAL?

### ✅ Real Services
- **Orchestration Service**: Actual HTTP server on port 3000
- **Whisper Service**: Actual transcription service on port 5001
- No mocks, no simulations

### ✅ Real HTTP Calls
- `httpx.AsyncClient()` makes actual network requests
- Real multipart/form-data uploads
- Real JSON responses

### ✅ Real Audio
- Generated using numpy with speech-like characteristics
- Realistic waveforms with harmonics and formants
- Can be transcribed by actual Whisper model

### ✅ Real Transcriptions
- Whisper service processes actual audio
- Returns real transcription results
- No pre-recorded responses

### ✅ Real Virtual Webcam
- VirtualWebcamManager renders actual subtitle frames
- PIL/Pillow generates real images
- Frames saved as actual PNG files on disk

### ✅ Real Bot Session
- GoogleMeetBotManager creates real session
- Real session tracking and lifecycle management
- Real database integration (if available)

## Troubleshooting

### Services Not Running
If you see:
```
❌ Orchestration service: Not running (http://localhost:3000)
```

Start the service:
```bash
cd modules/orchestration-service
python src/main_fastapi.py
```

### Whisper Service Not Running
If you see:
```
❌ Whisper service: Not running (http://localhost:5001)
```

Start the service:
```bash
cd modules/whisper-service
python src/main.py --device=cpu
```

### No Frames Captured
If frames are not being captured:
- Check that virtual webcam initialized successfully
- Check logs for frame rendering errors
- Ensure output directory is writable

### Transcriptions Not Appearing
If transcriptions are not coming back:
- Check whisper service logs
- Verify audio is being generated correctly
- Check network connectivity between services

## Advanced Usage

### Custom Test Duration
Modify `TEST_DURATION` in the script:
```python
TEST_DURATION = 60  # Run for 60 seconds
```

### Custom Output Directory
Modify `OUTPUT_DIR` in the script:
```python
OUTPUT_DIR = Path("/custom/path/to/output")
```

### Different Whisper Model
Modify the upload parameters:
```python
data = {
    ...
    'whisper_model': 'whisper-large',  # Use large model
    ...
}
```

### Enable Translation
Modify the upload parameters:
```python
data = {
    ...
    'enable_translation': 'true',
    'target_languages': '["es", "fr"]',  # Spanish and French
    ...
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Test Orchestrator                         │
│  (test_real_endtoend_transcription.py)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ HTTP POST /api/audio/upload
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Orchestration Service (port 3000)               │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Audio Upload Endpoint                                │  │
│  │  (/api/audio/upload)                                  │  │
│  └─────────────┬────────────────────────────────────────┘  │
│                │                                             │
└────────────────┼─────────────────────────────────────────────┘
                 │ HTTP POST to Whisper
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Whisper Service (port 5001)                     │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Transcription Engine                                 │  │
│  │  (Real Whisper Model)                                 │  │
│  └─────────────┬────────────────────────────────────────┘  │
│                │                                             │
└────────────────┼─────────────────────────────────────────────┘
                 │ Returns transcription
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Bot Integration Pipeline                        │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Virtual Webcam Manager                               │  │
│  │  - Renders subtitle frames                            │  │
│  │  - Speaker attribution                                │  │
│  │  - Confidence scores                                  │  │
│  └─────────────┬────────────────────────────────────────┘  │
│                │                                             │
└────────────────┼─────────────────────────────────────────────┘
                 │ Frame callback
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Frame Capture & Storage                         │
│                                                               │
│  frame_0000.png, frame_0001.png, ... frame_0539.png         │
│                                                               │
│  → ffmpeg → output.mp4                                       │
└─────────────────────────────────────────────────────────────┘
```

## Success Criteria

✅ **Only runs if services are actually running**
✅ **Uses REAL HTTP calls** (no mocks)
✅ **Sends REAL audio** (generated or from file)
✅ **Gets REAL transcriptions** from whisper service
✅ **Displays REAL subtitles** on virtual webcam
✅ **Saves ALL frames** (not just first one)
✅ **Provides clear output** showing what happened
✅ **Generates video** from frames

## Related Files

- `src/bot/bot_manager.py` - Bot lifecycle management
- `src/bot/virtual_webcam.py` - Virtual webcam rendering
- `src/bot/bot_integration.py` - Complete pipeline integration
- `src/routers/audio/audio_core.py` - Audio upload endpoint

## License

Same as LiveTranslate project

## Author

Created for LiveTranslate real-world testing
