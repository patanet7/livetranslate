# Google Meet Bot System - Complete Integration

**Professional Google Meet bot integration with virtual webcam generation and speaker attribution**

This module provides a comprehensive Google Meet bot system that captures audio from Google Meet sessions, processes it through the LiveTranslate pipeline, and generates professional virtual webcam overlays with real-time transcription and translation display.

## 🏗️ Architecture Overview

### Complete Audio Pipeline (DRY Architecture)

**As of 2026-01-17, the Google Meet bot uses the unified `TranscriptionPipelineCoordinator`.**

```
Google Meet Browser → Browser Audio Capture → Orchestration Service
    ↓
Whisper Service (NPU) → Speaker Diarization
    ↓
GoogleMeetChunkAdapter → TranscriptChunk (unified format)
    ↓
TranscriptionPipelineCoordinator
    ├─→ SentenceAggregator (sentence boundary detection)
    ├─→ RollingWindowTranslator + GlossaryService
    ├─→ CaptionBuffer (real-time display)
    └─→ BotSessionDatabaseManager (persistence)
    ↓
Virtual Webcam Generation → Professional Display
```

This unified architecture ensures the Google Meet bot uses the exact same translation and storage logic as Fireflies and audio uploads.

### Key Components

#### 1. **Bot Integration Pipeline** (`bot_integration.py`)
- **Main orchestration class** that coordinates all bot functionality
- **Unified pipeline integration** - uses `TranscriptionPipelineCoordinator` with `GoogleMeetChunkAdapter`
- **Pipeline callbacks** - `_handle_pipeline_translation()` for virtual webcam updates
- **Service coordination** with whisper and translation services
- **Session management** with database persistence
- **Error handling** and automatic recovery

Key pipeline-related members added in 2026-01-17 refactoring:
- `pipeline_coordinator: TranscriptionPipelineCoordinator` - Main pipeline instance
- `caption_buffer: CaptionBuffer` - For real-time caption display
- `_handle_pipeline_translation()` - Callback for translation results
- `_handle_pipeline_error()` - Error handling callback

#### 2. **Google Meet Browser Automation** (`google_meet_automation.py`)
- **Headless Chrome integration** for Google Meet session management
- **Automatic meeting joining** with camera/microphone control
- **Live caption extraction** from Google Meet UI
- **Participant monitoring** with real-time updates
- **Meeting state management** (connecting, joined, error, leaving)

#### 3. **Browser Audio Capture** (`browser_audio_capture.py`)
- **Specialized audio capture** from Google Meet browser sessions
- **Multiple capture methods**: Virtual devices, loopback, system default
- **Audio quality validation** with RMS level checking and clipping detection
- **Direct streaming** to orchestration service via `/api/audio/upload`
- **Comprehensive error handling** with graceful fallbacks

#### 4. **Virtual Webcam System** (`virtual_webcam.py`)
- **Professional translation overlays** with broadcast-quality rendering
- **Speaker attribution** with enhanced diarization display
- **Multiple display modes**: Overlay, sidebar, bottom banner, floating, fullscreen
- **Theme support**: Dark, light, high contrast, minimal, corporate
- **Real-time frame generation** at 30fps with configurable duration

#### 5. **Time Correlation Engine** (`time_correlation.py`)
- **Advanced timeline matching** between Google Meet captions and internal transcriptions
- **Speaker state tracking** with transition detection
- **Confidence scoring** for correlation quality
- **Multiple correlation methods**: Direct, inferred, interpolated
- **Database integration** for persistent correlation storage

#### 6. **Caption Processor** (`caption_processor.py`)
- **Google Meet caption extraction** with speaker timeline building
- **Real-time processing** of live captions
- **Speaker event generation** for time correlation
- **Database persistence** of caption data

#### 7. **Audio Capture** (`audio_capture.py`)
- **System audio capture** with WebSocket streaming
- **Voice Activity Detection** integration
- **Transcription callback** handling
- **Legacy fallback** for non-browser audio sources

## 🎥 Virtual Webcam Features

### Speaker Attribution Display
- **Enhanced Speaker Names**: Shows both human-readable names and diarization IDs
- **Format Example**: "John Doe (SPEAKER_00)" for maximum clarity
- **Speaker Color Coding**: Unique colors for each participant (8-color palette)
- **Fallback Handling**: Graceful display for unknown speakers

### Dual Content Streaming
- **Original Transcriptions**: Immediate display with 🎤 indicator and **actual Whisper confidence scores**
- **Multi-language Translations**: Real-time translations with 🌐 indicator and actual translation confidence scores
- **Visual Distinction**: Clear separation between transcription and translation content
- **Language Indicators**: Source → target language display for translations

### Professional Visual Layout
- **Enhanced Box Design**: Professional typography and spacing
- **Confidence Indicators**: Color-coded confidence scores (📊 high/medium/low)
- **Language Direction**: Clear source → target language indicators (🔄)
- **Session Information**: Live header with session ID and participant count (📹)
- **Timestamp Display**: Optional timestamp overlay for each message
- **Word Wrapping**: Intelligent text formatting for longer messages

### Display Modes & Themes

#### Display Modes:
- **Overlay**: Floating translation boxes (default)
- **Sidebar**: Translations in side panel
- **Bottom Banner**: Scrolling translations at bottom
- **Floating**: Dynamic positioned bubbles
- **Fullscreen**: Large centered translations

#### Theme Options:
- **Dark**: Professional dark theme with blue accents
- **Light**: Clean light theme with high readability
- **High Contrast**: Maximum accessibility with bold colors
- **Minimal**: Clean, simple design
- **Corporate**: Business-appropriate styling

## 🔧 Configuration

### Bot Configuration (`BotConfig`)
```python
@dataclass
class BotConfig:
    bot_id: str
    bot_name: str = "LiveTranslate Bot"
    target_languages: List[str] = ["en", "es", "fr", "de", "zh"]
    audio_config: AudioConfig = None
    correlation_config: CorrelationConfig = None
    webcam_config: WebcamConfig = None
    service_endpoints: ServiceEndpoints = None
    auto_join_meetings: bool = True
    recording_enabled: bool = False
    real_time_translation: bool = True
    virtual_webcam_enabled: bool = True
```

### Virtual Webcam Configuration (`WebcamConfig`)
```python
@dataclass
class WebcamConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    format: str = "RGB24"
    device_name: str = "LiveTranslate Virtual Camera"
    display_mode: DisplayMode = DisplayMode.OVERLAY
    theme: Theme = Theme.DARK
    max_translations_displayed: int = 5
    translation_duration_seconds: float = 10.0
    font_size: int = 24
    background_opacity: float = 0.8
    show_speaker_names: bool = True
    show_confidence: bool = True
    show_timestamps: bool = False
```

### Browser Audio Configuration (`BrowserAudioConfig`)
```python
@dataclass
class BrowserAudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 2.0  # seconds
    audio_format: str = "float32"
    quality_threshold: float = 0.01
    browser_process_name: str = "chrome"
    virtual_audio_device: Optional[str] = None
    loopback_device: Optional[str] = None
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = True
    audio_gain: float = 1.0
```

## 🚀 API Endpoints

### Bot Management
```python
# Spawn a new bot
POST /api/bot/spawn
{
  "meeting_id": "abc-defg-hij",
  "meeting_title": "Team Meeting",
  "target_languages": ["es", "fr", "de"],
  "enable_virtual_webcam": true
}

# Get bot status
GET /api/bot/{bot_id}

# Terminate bot
POST /api/bot/{bot_id}/terminate
```

### Virtual Webcam API
```python
# Get current webcam frame
GET /api/bot/virtual-webcam/frame/{bot_id}
Returns: {
  "bot_id": "string",
  "frame_base64": "base64_encoded_image",
  "timestamp": 1640995200.0,
  "webcam_stats": {
    "is_streaming": true,
    "frames_generated": 1234,
    "average_fps": 29.8,
    "current_translations_count": 3,
    "speakers_count": 2
  }
}

# Get webcam configuration
GET /api/bot/virtual-webcam/config/{bot_id}

# Update webcam configuration
POST /api/bot/virtual-webcam/config/{bot_id}
{
  "display_mode": "overlay",
  "theme": "dark",
  "max_translations_displayed": 5,
  "translation_duration_seconds": 10.0,
  "show_speaker_names": true,
  "show_confidence": true,
  "show_timestamps": false
}
```

## 💡 Usage Examples

### Basic Bot Usage
```python
# Create bot integration
bot = create_bot_integration(
    bot_id="meeting-bot-001",
    bot_name="Team Meeting Bot",
    target_languages=["en", "es", "fr"],
    virtual_webcam_enabled=True,
)

# Set up callbacks
def on_transcription(data):
    print(f"Transcription: {data['speaker_name']} - {data['text']}")

def on_translation(data):
    print(f"Translation ({data['target_language']}): {data['translated_text']}")

bot.set_transcription_callback(on_transcription)
bot.set_translation_callback(on_translation)

# Join meeting
meeting = MeetingInfo(
    meeting_id="abc-defg-hij",
    meeting_title="Team Meeting",
    organizer_email="organizer@company.com",
    participant_count=5,
)

session_id = await bot.join_meeting(meeting)
if session_id:
    print(f"Bot joined meeting: {session_id}")
    
    # Meeting will run automatically
    # Virtual webcam will display translations
    
    # Leave when done
    await bot.leave_meeting(session_id)
```

### Virtual Webcam Customization
```python
# Create custom webcam configuration
webcam_config = WebcamConfig(
    display_mode=DisplayMode.SIDEBAR,
    theme=Theme.CORPORATE,
    max_translations_displayed=3,
    translation_duration_seconds=15.0,
    show_speaker_names=True,
    show_confidence=True,
    show_timestamps=True,
)

# Create bot with custom config
bot = GoogleMeetBotIntegration(BotConfig(
    bot_id="custom-bot",
    webcam_config=webcam_config,
    target_languages=["en", "es", "fr", "de"],
))
```

## 🔍 Database Schema

The bot system uses a comprehensive PostgreSQL schema for session tracking:

### Core Tables
- **`bot_sessions`**: Bot session metadata and lifecycle
- **`audio_files`**: Audio chunk storage and metadata
- **`transcripts`**: Transcription results with speaker attribution
- **`translations`**: Translation results with confidence scores
- **`correlations`**: Time correlation data between external and internal sources
- **`speaker_timelines`**: Speaker activity tracking

### Key Features
- **Comprehensive indexing** for performance
- **JSON metadata storage** for flexible data
- **Time-based partitioning** for large datasets
- **Session analytics views** for reporting

## 🧪 Testing

### Unit Tests
```bash
# Run bot system tests
cd modules/orchestration-service
python -m pytest tests/bot/ -v

# Run specific component tests
python -m pytest tests/bot/test_virtual_webcam.py -v
python -m pytest tests/bot/test_browser_audio_capture.py -v
```

### Integration Tests
```bash
# Run full integration tests
python tests/bot/test_bot_integration.py

# Test with actual Google Meet (requires setup)
python tests/bot/test_google_meet_automation.py --live
```

## 🚨 Production Considerations

### Performance
- **Memory Usage**: ~200MB per active bot
- **CPU Usage**: ~5-10% per bot (varies with NPU/GPU availability)
- **Network**: ~50KB/s per bot for audio streaming
- **Latency**: <100ms total pipeline latency target

### Scalability
- **Concurrent Bots**: Up to 50 bots per orchestration service instance
- **Load Balancing**: Distribute bots across multiple orchestration instances
- **Database**: PostgreSQL handles thousands of sessions efficiently

### Security
- **Headless Browser Security**: Sandboxed Chrome instances
- **Audio Privacy**: Local processing, no external API calls
- **Session Isolation**: Each bot runs in isolated context
- **Error Containment**: Bot failures don't affect other bots

### Monitoring
- **Session Tracking**: Complete audit trail in database
- **Performance Metrics**: Real-time monitoring via Prometheus
- **Error Logging**: Comprehensive error capture and alerting
- **Quality Metrics**: Translation confidence and correlation scoring

## 📚 Additional Resources

- **Main Project Documentation**: `/CLAUDE.md`
- **Orchestration Service Docs**: `/modules/orchestration-service/CLAUDE.md`
- **Database Schema**: `/scripts/bot-sessions-schema.sql`
- **API Documentation**: Auto-generated at `/docs` when service is running
- **Configuration Examples**: `/env.template`

This Google Meet bot system provides enterprise-grade integration with professional virtual webcam generation, making it perfect for real-time translation scenarios in Google Meet environments.
