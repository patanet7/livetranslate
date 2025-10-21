# LiveTranslate Bot Container

**Phase 3.3: Simplified Bot Architecture** - "Headless Frontend" Design

## Overview

The bot container is a **headless frontend** that runs as a Docker container. It joins Google Meet meetings, captures audio, and streams to the orchestration service using the **SAME WebSocket protocol as the browser frontend**.

### Key Benefits

✅ **Reuses Existing Infrastructure**: Uses same orchestration, same session management, same processing
✅ **Consistent Processing**: Deduplication, speaker grouping all handled by orchestration
✅ **Account Tracking**: User ID flows through orchestration naturally
✅ **Isolation**: Bot failures don't crash manager
✅ **Scalability**: Run multiple bots across machines easily
✅ **-60% Complexity**: Manager just orchestrates Docker containers

## Architecture

```
Bot Container (this)
    ↓ audio chunks (WebSocket)
Orchestration Service (websocket_frontend_handler.py)
    ↓ authenticate, track session
Streaming Coordinator (streaming_coordinator.py)
    ↓ deduplicate, group speakers
Whisper Client (websocket_whisper_client.py)
    ↓ transcribe
Whisper Service
    ↓ segments back (deduplicated, speaker-grouped)
Bot Container (receives processed segments)
```

## Components

### Core Components (Phase 3.3a - ✅ Complete)

- **`orchestration_client.py`**: WebSocket client to orchestration service
  - Authenticates with user token
  - Streams audio chunks (base64 encoded)
  - Receives transcription segments
  - Handles reconnection

- **`bot_main.py`**: Main entry point
  - Orchestrates all bot components
  - Handles lifecycle (startup, active, shutdown)
  - Sends HTTP callbacks to bot manager
  - Graceful shutdown handling

### To Be Implemented (Phase 3.3b)

- **`browser_automation.py`**: Google Meet browser control
- **`audio_capture.py`**: Audio extraction from browser
- **`redis_subscriber.py`**: Listen for commands from manager
- **`virtual_webcam.py`**: Optional display output

## Usage

### Building the Container

```bash
cd modules/bot-container
docker build -t livetranslate-bot:latest .
```

### Running a Bot

```bash
docker run -d \
  --name bot-meeting-123 \
  --network livetranslate_default \
  -e MEETING_URL="https://meet.google.com/abc-def-ghi" \
  -e CONNECTION_ID="bot-connection-456" \
  -e USER_TOKEN="user-api-token" \
  -e ORCHESTRATION_WS_URL="ws://orchestration:3000/ws" \
  -e REDIS_URL="redis://redis:6379" \
  -e BOT_MANAGER_URL="http://bot-manager:8080" \
  -e LANGUAGE="en" \
  -e TASK="transcribe" \
  livetranslate-bot:latest
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MEETING_URL` | ✅ | - | Google Meet URL to join |
| `CONNECTION_ID` | ✅ | - | Unique bot connection ID |
| `USER_TOKEN` | ✅ | - | User API token for orchestration auth |
| `ORCHESTRATION_WS_URL` | ❌ | `ws://orchestration:3000/ws` | Orchestration WebSocket URL |
| `REDIS_URL` | ❌ | - | Redis URL for commands |
| `BOT_MANAGER_URL` | ❌ | - | Bot manager URL for callbacks |
| `LANGUAGE` | ❌ | `en` | Transcription language |
| `TASK` | ❌ | `transcribe` | Task: `transcribe` or `translate` |

## Bot Lifecycle

### 1. Startup

```python
Bot Manager
    ↓ docker run (sets env vars)
Bot Container
    ↓ Read configuration from env
    ↓ Connect to orchestration (WebSocket)
    ↓ HTTP callback: POST /bots/internal/callback/started
Bot Manager (receives callback, updates DB)
```

### 2. Joining Meeting

```python
Bot Container
    ↓ HTTP callback: POST /bots/internal/callback/joining
    ↓ Join Google Meet (browser automation)
    ↓ HTTP callback: POST /bots/internal/callback/active
    ↓ Start audio streaming
Bot Manager (receives callbacks, tracks status)
```

### 3. Active Streaming

```python
Bot Container
    ↓ Capture audio from meeting
    ↓ WebSocket send: audio_chunk
Orchestration
    ↓ Process (deduplicate, speaker group)
    ↓ WebSocket send: segment
Bot Container
    ↓ Receive segment
    ↓ Display on virtual webcam (optional)
```

### 4. Shutdown

```python
Manager sends Redis command: {"action": "leave"}
    ↓
Bot Container
    ↓ Stop audio capture
    ↓ Leave Google Meet
    ↓ Disconnect from orchestration
    ↓ HTTP callback: POST /bots/internal/callback/completed
Bot Manager (receives callback, cleanup)
```

## Communication Protocols

### WebSocket (Bot ↔ Orchestration)

**Same protocol as frontend!**

#### Bot → Orchestration

```json
// Authenticate
{
  "type": "authenticate",
  "user_id": "bot-{connection_id}",
  "token": "{user_token}"
}

// Start session
{
  "type": "start_session",
  "session_id": "{connection_id}",
  "config": {
    "model": "large-v3",
    "language": "en",
    "enable_vad": true
  }
}

// Audio chunk
{
  "type": "audio_chunk",
  "audio": "<base64>",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

#### Orchestration → Bot

```json
// Segment (deduplicated, speaker-grouped)
{
  "type": "segment",
  "text": "Hello everyone",
  "speaker": "SPEAKER_00",
  "absolute_start_time": "2025-01-15T10:30:00Z",
  "absolute_end_time": "2025-01-15T10:30:03Z",
  "is_final": false,
  "confidence": 0.95
}
```

### HTTP (Bot → Manager)

#### Status Callbacks

```bash
# Started
POST /bots/internal/callback/started
{
  "connection_id": "bot-123",
  "container_id": "abc123..."
}

# Joining
POST /bots/internal/callback/joining
{
  "connection_id": "bot-123",
  "container_id": "abc123..."
}

# Active
POST /bots/internal/callback/active
{
  "connection_id": "bot-123",
  "container_id": "abc123..."
}

# Completed
POST /bots/internal/callback/completed
{
  "connection_id": "bot-123",
  "container_id": "abc123..."
}

# Failed
POST /bots/internal/callback/failed
{
  "connection_id": "bot-123",
  "container_id": "abc123...",
  "error": "Error message",
  "exit_code": 1
}
```

### Redis (Manager → Bot)

#### Commands

```json
// Leave meeting
{
  "action": "leave"
}

// Reconfigure
{
  "action": "reconfigure",
  "language": "es",
  "task": "translate"
}
```

## Testing

### Unit Tests

```bash
pytest tests/ -v -m unit
```

### Integration Tests (requires orchestration running)

```bash
pytest tests/ -v -m integration
```

### All Tests

```bash
pytest tests/ -v
```

## Development

### Setup

```bash
cd modules/bot-container
pip install -r requirements.txt
```

### Run Locally

```bash
# Set environment variables
export MEETING_URL="https://meet.google.com/test"
export CONNECTION_ID="test-bot-123"
export USER_TOKEN="test-token"
export ORCHESTRATION_WS_URL="ws://localhost:3000/ws"

# Run bot
python src/bot_main.py
```

## Phase 3.3 Progress

### Phase 3.3a: Bot Container Creation ✅ Complete

- [x] Create directory structure
- [x] Implement `orchestration_client.py` (WebSocket client)
- [x] Implement `bot_main.py` (entry point)
- [x] Write TDD tests
- [x] Create Dockerfile
- [x] Create requirements.txt
- [x] Tests passing: 4/4 ✅

### Phase 3.3b: Simplify Bot Manager (Next)

- [ ] Replace process management with Docker client
- [ ] Implement callback endpoints in manager
- [ ] Implement Redis pub/sub for commands
- [ ] Remove/merge old bot files
- [ ] Target: Reduce from 8,701 lines to ~3,480 lines

### Phase 3.3c: Integration (After 3.3b)

- [ ] Implement browser automation
- [ ] Implement audio capture
- [ ] End-to-end testing
- [ ] Performance validation

## Related Files

### Orchestration Service (Already Exists!)

- `modules/orchestration-service/src/websocket_frontend_handler.py`
- `modules/orchestration-service/src/streaming_coordinator.py`
- `modules/orchestration-service/src/websocket_whisper_client.py`

### Bot Manager (To Be Simplified)

- `modules/orchestration-service/src/bot/bot_manager.py`

## License

Same as parent project - check root LICENSE file.
