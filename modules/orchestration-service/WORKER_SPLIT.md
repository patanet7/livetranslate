# Orchestration Service – Worker Extraction Notes

## Current Responsibilities (FastAPI Process)
- **HTTP & WebSocket surface**  
  - Entry points in `routers/` (`audio`, `translation`, `settings`, `system`, `websocket`, `analytics`).  
  - Request validation models in `models/` and dependency wiring via `dependencies.py`.  
  - Session management utilities in `gateway/` and `managers/websocket_manager.py`.
- **Synchronous service calls**  
  - Whisper & translation clients located in `clients/`.  
  - Aggregated orchestration logic in `audio/audio_coordinator.py` for on-demand chunk processing.
- **Configuration access**  
  - `managers/config_manager.py` + `config/` for current settings.  
  - Shared state cached inside the API process.

These pieces remain inside the API pod/container after refactor; they should become thin request brokers that enqueue work and stream results back to clients.

## Long-Running / Stateful Modules (Candidates for Background Workers)

| Domain | Modules / Files | Notes |
| --- | --- | --- |
| **Configuration Sync** | `audio/config_sync.py`, `audio/config.py`, `dependencies.get_config_sync_manager` | Manages drift detection, hot reloads, preset reconciliation. Maintains internal polling threads; ideal for a dedicated worker watching config sources and publishing updates. |
| **Audio Pipeline & Chunk Processing** | `audio/audio_coordinator.py`, `audio/chunk_manager.py`, `audio/audio_processor.py`, `audio/speaker_correlator.py`, `audio/timing_coordinator.py`, `audio/database_adapter.py` | Handles chunk queues, numpy transforms, external service calls, DB writes. Should run as async workers consuming events from the API to avoid blocking request threads. |
| **Bot Lifecycle / Google Meet Automation** | `bot/` package (`bot_manager.py`, `bot_lifecycle_manager.py`, `google_meet_*`, `virtual_webcam.py`, `audio_capture.py`, `time_correlation.py`), `managers/bot_manager.py` | Spawns browser automation, manages long-running sessions, persisting stats. Needs its own worker (or pool) with resource-limited concurrency. |
| **Monitoring & Health Polling** | `managers/health_monitor.py`, `monitoring/` | Periodic polling of system metrics and downstream services should move to a worker or scheduled job to avoid tying up API event loop. |
| **Configuration Synchronization Events** | `audio/config_sync.py`, `routers/settings.py` | Currently mixes HTTP handlers with background loops. Separate worker can subscribe to config change topics and publish updates back to API/clients. |

## Suggested Worker Roles
1. **Config Sync Worker**  
   - Watches shared config sources (DB, files, remote services).  
   - Publishes `ConfigUpdated` events and handles drift remediation.
2. **Audio Pipeline Worker**  
   - Processes `AudioChunkReceived` events, executes stage pipeline, persists results, and emits `TranscriptionCompleted` / `TranslationCompleted`.  
   - Responsible for speaker correlation & timing alignment.
3. **Bot Orchestrator Worker**  
   - Manages bot lifecycle queue (`BotRequested`, `BotStatusChanged`).  
   - Triggers Google Meet automation, audio capture, and cleanup.
4. **Monitoring Worker**  
   - Periodically polls downstream services, updates metrics store, and raises alerts.

Workers can be implemented as separate processes (Celery, Dramatiq, FastAPI background service) sharing a common message bus with the API tier.

## Interfaces to Preserve
- **API → Worker**: enqueue jobs when REST/WebSocket endpoints are hit (e.g., `/api/whisper/stream`, `/api/bot/start`).  
- **Worker → API/Clients**: push updates via Redis pub/sub, WebSocket notifications, or database channels consumed by the API.
- **Shared DB/Cache**: Reuse existing `database/` repositories for persistence; ensure models are serializable for queue payloads.

## Outstanding Investigations
- Determine ownership of `monitoring/` dashboards and whether they can consume worker-generated metrics.  
- Audit `dependencies.py` singletons—transition to lazy proxies that talk to workers instead of holding state in-process.  
- Evaluate scope of legacy `gateway/` code when bot orchestration moves out-of-process.
