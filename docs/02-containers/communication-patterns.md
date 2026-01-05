# Communication Patterns

## HTTP REST (Synchronous)

**Frontend → Orchestration**
- `POST /api/audio/upload` - Upload audio chunk
- `GET /api/health` - Health check
- `GET /api/settings` - Get configuration

**Orchestration → Whisper**
- `POST /transcribe` - Transcribe audio
- `GET /health` - Health check

**Orchestration → Translation**
- `POST /api/translate` - Translate text
- `GET /api/health` - Health check

## WebSocket (Asynchronous)

**Frontend ↔ Orchestration**
- Real-time transcription updates
- Translation streaming
- Progress notifications

## Database (PostgreSQL)

**Orchestration → PostgreSQL**
- Session persistence
- Transcript storage
- Analytics queries

See [Data Flow](./data-flow.md) for complete flow.
