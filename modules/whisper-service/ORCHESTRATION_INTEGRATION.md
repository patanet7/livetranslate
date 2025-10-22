# Whisper Service ↔ Orchestration Service Integration

## Architecture Alignment: Per-Session Buffers + Unified Managers

This document explains how the whisper service's new **per-session buffer architecture** integrates with the orchestration service's **unified manager architecture**.

---

## 🏗️ Architecture Overview

### Orchestration Service (8 Unified Managers)
Following the plan from `modules/orchestration-service/modules/plan.md`:

1. **UnifiedConfigurationManager** - Configuration coordination
2. **UnifiedBotSessionRepository** - Database operations (sessions, audio, transcripts, translations)
3. **UnifiedBotManager** - Bot lifecycle management
4. **WebSocketManager** - WebSocket connection management
5. **HealthMonitor** - System health monitoring
6. **AudioConfigurationManager** - Audio-specific configuration
7. **ConfigurationSyncManager** - Service synchronization
8. **GoogleMeetBotManager** - Google Meet integration

### Whisper Service (Per-Session Architecture)
Following SimulStreaming reference implementation:

- **Per-Session Audio Buffers**: `session_audio_buffers[session_id] = List[torch.Tensor]`
- **Thread-Safe Access**: `session_buffers_lock` for concurrent sessions
- **Session Isolation**: Each session has independent buffer (no cross-contamination)
- **Automatic Cleanup**: Buffers cleared on session end

---

## 🔗 Integration Points

### 1. Session ID Flow

```
Frontend → Orchestration WebSocket → Whisper WebSocket
            ↓                          ↓
    connection_id + session_id    session_id = buffer key
            ↓                          ↓
    WebSocketManager              session_audio_buffers[session_id]
            ↓                          ↓
    UnifiedBotSessionRepository   Per-session transcription
```

**Key Mappings:**
- **Orchestration**: `connection_id` (WebSocket connection) → `session_id` (logical session)
- **Whisper**: `session_id` → isolated audio buffer

**Important**: Multiple WebSocket connections can share the same `session_id` in orchestration, but whisper service uses `session_id` as the buffer key.

### 2. WebSocket Message Flow

```javascript
// Frontend → Orchestration
{
  type: "start_session",
  session_id: "session-xyz",  // Logical session identifier
  config: { model: "large-v3-turbo", language: "en" }
}

// Orchestration → Whisper
{
  event: "join_session",
  session_id: "session-xyz",  // Same session ID propagated
  config: { model: "large-v3-turbo", language: "en" }
}

// Frontend → Orchestration
{
  type: "audio_chunk",
  audio: "base64_audio_data..."
}

// Orchestration → Whisper
{
  event: "transcribe_stream",
  session_id: "session-xyz",  // Same session ID
  audio_data: "base64_audio_data...",
  timestamp: "2025-10-21T12:34:56Z"
}
```

### 3. Database Integration

**UnifiedBotSessionRepository** tracks sessions in PostgreSQL:

```python
# Orchestration creates session in database
await repository.session_create(
    session_id="session-xyz",
    meeting_url="https://meet.google.com/abc-def-ghi",
    bot_config={...},
    metadata={...}
)

# Whisper creates isolated buffer for same session_id
session_audio_buffers["session-xyz"] = []
```

**Data Flow:**
1. **Session Start**: Orchestration creates DB record + Whisper creates buffer
2. **Audio Processing**: Whisper transcribes from buffer → Orchestration stores results in DB
3. **Session End**: Orchestration updates DB status → Whisper cleans up buffer

---

## 🎯 Session Management Coordination

### Orchestration WebSocketManager

```python
@dataclass
class ConnectionInfo:
    connection_id: str          # e.g., "conn-12345"
    session_id: Optional[str]   # e.g., "session-xyz"
    websocket: WebSocket
    # ... other fields

@dataclass
class SessionInfo:
    session_id: str                    # e.g., "session-xyz"
    connection_ids: Set[str]           # Multiple connections per session
    # ... other fields
```

**Key Features:**
- **1:N Mapping**: One session can have multiple connections
- **Session Tracking**: WebSocketManager maintains active sessions
- **Timeout Management**: Sessions auto-expire after 30 minutes

### Whisper Service Session Buffers

```python
class WhisperService:
    def __init__(self):
        # Per-session audio buffers (SimulStreaming pattern)
        self.session_audio_buffers = {}  # session_id → List[torch.Tensor]
        self.session_buffers_lock = threading.Lock()

    def add_audio_chunk(self, audio_chunk: np.ndarray, session_id: str = "default"):
        with self.session_buffers_lock:
            if session_id not in self.session_audio_buffers:
                self.session_audio_buffers[session_id] = []
            self.session_audio_buffers[session_id].append(audio_tensor)
```

**Key Features:**
- **1:1 Mapping**: One buffer per session_id
- **Thread-Safe**: Lock-protected buffer access
- **Automatic Cleanup**: Buffer removed on stream end

---

## 🔄 Lifecycle Coordination

### Session Lifecycle

```
┌─────────────────┐          ┌──────────────────┐
│  Orchestration  │          │  Whisper Service │
└─────────────────┘          └──────────────────┘
        │                            │
        │ 1. WebSocket connect       │
        ├─────────────────────────→  │
        │                            │
        │ 2. start_session           │
        │   (session_id="xyz")       │
        ├─────────────────────────→  │
        │                            │
        │                            │ 3. Create buffer
        │                            │    session_audio_buffers["xyz"] = []
        │                            │
        │ 4. audio_chunk             │
        │   (session_id="xyz")       │
        ├─────────────────────────→  │
        │                            │ 5. add_audio_chunk("xyz")
        │                            │    buffer["xyz"].append(audio)
        │                            │
        │                            │ 6. transcribe_stream()
        │                            │    uses buffer["xyz"]
        │                            │
        │ ←─────────────────────────┤ 7. transcription_result
        │   (session_id="xyz")       │
        │                            │
        │ 8. end_session             │
        │   (session_id="xyz")       │
        ├─────────────────────────→  │
        │                            │ 9. Cleanup buffer
        │                            │    del session_audio_buffers["xyz"]
        │                            │
```

### Database Persistence Flow

```python
# 1. Orchestration: Start session
session = await repository.session_create(
    session_id="session-xyz",
    meeting_url="https://meet.google.com/abc-def-ghi",
    bot_config={"model": "large-v3-turbo"},
    metadata={"user_id": "user-123"}
)
# DB: BotSession record created with status="initializing"

# 2. Whisper: Create buffer (in-memory only)
session_audio_buffers["session-xyz"] = []

# 3. Whisper: Process audio → return transcription
result = await whisper_service.transcribe_stream(
    TranscriptionRequest(session_id="session-xyz", ...)
)

# 4. Orchestration: Store transcription in DB
await repository.transcript_create(
    session_id="session-xyz",
    transcript_text=result.text,
    confidence_score=result.confidence_score,
    speaker_id=None,
    start_time=0.0,
    end_time=duration
)
# DB: Transcript record created

# 5. Orchestration: End session
await repository.session_update_status(
    session_id="session-xyz",
    status="completed"
)
# DB: BotSession status updated

# 6. Whisper: Cleanup buffer
with session_buffers_lock:
    del session_audio_buffers["session-xyz"]
```

---

## 📊 Data Structures Comparison

### Orchestration (PostgreSQL + In-Memory)

**Database Models** (SQLAlchemy):
```python
class BotSession(Base):
    session_id: str         # Primary key
    meeting_url: str
    status: str            # initializing, active, completed, failed
    bot_config: JSON
    created_at: DateTime
    ended_at: DateTime

class AudioFile(Base):
    audio_file_id: str
    session_id: str        # Foreign key to BotSession
    file_path: str
    duration: float

class Transcript(Base):
    transcript_id: str
    session_id: str        # Foreign key to BotSession
    transcript_text: str
    confidence_score: float
    speaker_id: str
```

**In-Memory Tracking** (WebSocketManager):
```python
connections: Dict[str, ConnectionInfo]  # connection_id → info
sessions: Dict[str, SessionInfo]        # session_id → info
```

### Whisper (In-Memory Only)

**Audio Buffers**:
```python
session_audio_buffers: Dict[str, List[torch.Tensor]]  # session_id → audio chunks
```

**Session State**:
```python
streaming_active: bool              # Global streaming flag
session_manager.sessions: Dict      # Session metadata (optional)
```

---

## 🛡️ Thread Safety & Concurrency

### Orchestration Concurrency

```python
# WebSocketManager: Asyncio-based, inherently thread-safe for concurrent connections
async def send_message(self, connection_id: str, message: Dict):
    connection = self.connections.get(connection_id)
    if connection:
        await connection.websocket.send_json(message)

# UnifiedBotSessionRepository: Database transactions handle concurrency
async def session_create(self, session_id: str, ...):
    async with self.db.session() as session:
        # SQLAlchemy handles transaction isolation
        bot_session = BotSession(session_id=session_id, ...)
        session.add(bot_session)
        await session.commit()
```

### Whisper Concurrency

```python
# WhisperService: Thread locks for buffer access
def add_audio_chunk(self, audio_chunk: np.ndarray, session_id: str):
    with self.session_buffers_lock:  # Thread-safe access
        if session_id not in self.session_audio_buffers:
            self.session_audio_buffers[session_id] = []
        self.session_audio_buffers[session_id].append(audio_tensor)

# ModelManager: Inference lock for model access
with self.inference_lock:
    result = model.transcribe(audio_data, ...)
```

**Result**: Both services handle concurrent sessions safely through appropriate locking mechanisms.

---

## 🔍 Session ID Consistency

### Generating Session IDs

**Option 1: Frontend-Generated** (Recommended)
```javascript
// Frontend creates UUID
const sessionId = `session-${Date.now()}-${Math.random().toString(36)}`;

// Pass to orchestration
websocket.send({
  type: "start_session",
  session_id: sessionId,
  config: {...}
});
```

**Option 2: Orchestration-Generated**
```python
# Orchestration creates UUID
session_id = f"session-{uuid.uuid4()}"

# Returns to frontend
await websocket.send_json({
  type: "session_started",
  session_id: session_id
})
```

**Option 3: Fallback to Connection ID**
```python
# Whisper service fallback (if no session_id provided)
session_id = request.session_id or client_id
```

### Session ID Format

**Recommended Format**: `session-{timestamp}-{random}`
- Example: `session-1729512345-a7b3c9`
- Unique across all services
- Sortable by creation time
- URL-safe (no special characters)

---

## ✅ Integration Verification Checklist

### Orchestration Side
- [ ] WebSocketManager properly tracks `session_id` per connection
- [ ] UnifiedBotSessionRepository creates database records with `session_id`
- [ ] SocketIOWhisperClient forwards `session_id` in all messages
- [ ] Session cleanup removes connections AND notifies whisper service

### Whisper Side
- [ ] `api_server.py` receives `session_id` from WebSocket messages
- [ ] `add_audio_chunk()` called with correct `session_id` parameter
- [ ] `transcribe_stream()` uses per-session buffers
- [ ] Buffer cleanup on disconnect/end_session

### End-to-End
- [ ] Same `session_id` flows through: Frontend → Orchestration → Whisper
- [ ] Multiple concurrent sessions don't interfere (no cross-contamination)
- [ ] Database records match whisper processing (same `session_id`)
- [ ] Cleanup works: Orchestration + Whisper both clean up on session end

---

## 🐛 Debugging Session Issues

### Check Session ID Propagation

```bash
# Orchestration logs
grep "session_id" orchestration.log | grep "session-xyz"

# Whisper logs
grep "session-xyz" whisper.log

# Should show same session_id in both services
```

### Check Buffer Isolation

```python
# Add to whisper_service.py for debugging
logger.info(f"[BUFFER] Active sessions: {list(self.session_audio_buffers.keys())}")
logger.info(f"[BUFFER] Session {session_id} buffer size: {len(self.session_audio_buffers[session_id])}")
```

### Check Database-Buffer Sync

```python
# Verify session exists in both places
db_session = await repository.session_get(session_id="session-xyz")
whisper_has_buffer = session_id in whisper_service.session_audio_buffers

if db_session and not whisper_has_buffer:
    logger.warning(f"Session {session_id} in DB but no whisper buffer!")
```

---

## 🎯 Best Practices

1. **Always Pass session_id**: Never rely on defaults or fallbacks in production
2. **Consistent Naming**: Use same `session_id` variable name across all services
3. **Explicit Cleanup**: Always clean up buffers on session end (don't rely on garbage collection)
4. **Database First**: Create DB session before starting whisper processing
5. **Log Session IDs**: Include `session_id` in all log messages for traceability
6. **Monitor Buffer Growth**: Track `session_audio_buffers` size to detect memory leaks
7. **Timeout Alignment**: Match session timeouts between orchestration (30min) and whisper

---

## 📈 Performance Considerations

### Memory Usage

```python
# Orchestration: Minimal per-session overhead
# - ConnectionInfo: ~1KB
# - SessionInfo: ~500 bytes
# - Database connection pool: shared

# Whisper: Significant per-session overhead
# - Audio buffer: ~30MB max (30 seconds @ 16kHz float32)
# - Model memory: shared across sessions
# - Total: ~30MB per active session
```

**Recommendation**: Limit concurrent whisper sessions to ~100 (3GB buffer memory)

### Database Load

```python
# Per session:
# - 1 INSERT (BotSession creation)
# - ~10-100 INSERTs (Transcripts, chunked)
# - 1-10 INSERTs (AudioFiles)
# - 1 UPDATE (BotSession completion)

# Total: ~12-112 queries per session
# At 100 concurrent sessions: ~1200 queries/session-lifetime
```

**Recommendation**: Connection pooling (already implemented in UnifiedBotSessionRepository)

---

## 🔮 Future Enhancements

### Phase 4: Advanced Session Management

1. **Session Migration**: Transfer active session between whisper instances
2. **Session Resume**: Resume interrupted sessions with buffer recovery
3. **Multi-Service Sessions**: Coordinate session across whisper + translation + diarization
4. **Session Analytics**: Track session metrics across orchestration + whisper

### Phase 5: Distributed Sessions

1. **Redis Session Store**: Share session state across multiple orchestration instances
2. **Whisper Load Balancing**: Route sessions to different whisper instances
3. **Session Affinity**: Maintain session-to-instance mapping for buffer locality

---

## 📝 Summary

The orchestration service's **unified manager architecture** and whisper service's **per-session buffer architecture** work together seamlessly:

- **Session ID** is the common identifier flowing through all services
- **WebSocketManager** (orchestration) tracks connections → sessions
- **UnifiedBotSessionRepository** (orchestration) persists session data
- **session_audio_buffers** (whisper) provides isolated audio processing
- **Thread safety** maintained through locks and async patterns
- **Cleanup coordination** ensures no resource leaks

This architecture provides:
✅ **Session isolation**: No cross-contamination between sessions
✅ **Scalability**: Support for 100+ concurrent sessions
✅ **Maintainability**: Clean separation of concerns
✅ **Observability**: Session tracking across all services
✅ **Reliability**: Proper cleanup and error handling
