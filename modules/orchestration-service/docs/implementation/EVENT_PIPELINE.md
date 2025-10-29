# Event & Queue Architecture Proposal

## Goals
- Decouple request/response handling in the FastAPI orchestration service from long-running audio/Bot workloads.
- Provide a consistent envelope for cross-service messages (API ↔ workers ↔ downstream services).
- Leverage Redis (already deployed) for initial implementation with a migration path to Kafka/NATS if throughput requirements grow.

## Transport Recommendation
- **Phase 1 (local/dev)**: Redis Streams (`XADD`/`XREADGROUP`) – minimal latency, already in docker-compose.  
- **Phase 2 (prod-ready option)**: Wrap queue logic behind an adapter so the same interface can target Kafka/NATS later.
- Use separate consumer groups per worker type (`audio-pipeline`, `bot-orchestrator`, `config-sync`, `monitoring`).

## Envelope Schema (JSON)
```json
{
  "schema_version": "v1",
  "event_id": "uuid4",
  "event_type": "AudioChunkReceived",
  "source": "orchestration-api",
  "timestamp": "2024-10-19T12:34:56.789Z",
  "trace_id": "request-trace-id",
  "payload": { "... domain-specific fields ..." }
}
```

- All payloads use snake_case keys to align with Python models.
- Binary blobs (audio chunks) should be stored in object storage or Redis binary keys with the queue message carrying a reference (`storage_uri`).

## Core Streams & Events

### 1. `stream:audio-ingest`
- **Producer**: API (`routers/audio.py`, WebSocket handlers).  
- **Consumer**: Audio Pipeline Worker.
- **Key payload fields**:
  - `session_id`, `chunk_id`, `user_id`
  - `source_type` (`websocket`, `upload`, `bot`)
  - `storage_uri` or inline base64 (dev only)
  - `audio_config` snapshot (sample rate, overlap)
  - `processing_flags` (transcribe, diarize, translate)

### 2. `stream:audio-results`
- **Producer**: Audio Pipeline Worker.  
- **Consumers**: Translation Worker (optional), API (for WebSocket fan-out), Monitoring.
- **Events**:
  - `TranscriptionCompleted`
  - `TranslationCompleted`
  - `ProcessingFailed`
- **Payload**: transcripts, timing data, speaker map, latency metrics.

### 3. `stream:config-sync`
- **Producer**: Config Sync Worker.  
- **Consumers**: API, whisper/translation services (through existing REST clients) or additional workers.
- **Events**:
  - `ConfigSnapshot`
  - `ConfigDriftDetected`
  - `ConfigRollbackRequested`
- Include checksum/version info for drift detection.

### 4. `stream:bot-control`
- **Producers**: API (start/stop requests), Monitoring (recovery).  
- **Consumer**: Bot Orchestrator Worker.
- **Events**:
  - `BotRequested`
  - `BotStopRequested`
  - `BotStatusUpdate` (worker → API, for dashboards)
- Payload includes meeting metadata, auth tokens (reference), desired capabilities.

### 5. `stream:monitoring`
- **Producer**: Monitoring Worker.  
- **Consumers**: API (for `/api/system/health`), alerting service.
- **Events**:
  - `ServiceHealthReport`
  - `AnomalyDetected`
  - `UsageMetrics`

## Interaction Flow (Example: Audio Chunk)
1. Client uploads/streams audio → API validates and publishes `AudioChunkReceived` to `stream:audio-ingest`.
2. Audio worker claims message (consumer group), pulls chunk from storage, runs pipeline.
3. Worker emits `TranscriptionCompleted` + optional `TranslationRequested`.
4. Translation worker processes text → emits `TranslationCompleted`.
5. API subscribes to `stream:audio-results` (separate consumer group) and delivers updates to WebSocket clients.

## Implementation Notes
- Add a thin queue abstraction (e.g., `infrastructure/queue.py`) with methods `publish(event)` and `subscribe(stream, handler)`.
- Ensure idempotency: include `chunk_id`/`bot_id` in message to deduplicate on retries.
- Add tracing headers (`trace_id`, `span_id`) to correlate logs once OpenTelemetry is introduced.
- Use Redis key TTLs for temporary binary blobs to avoid leaking storage.
