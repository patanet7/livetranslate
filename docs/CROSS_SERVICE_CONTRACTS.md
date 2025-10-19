# Cross-Service Contract Audit

This document captures the current state of request/response contracts between the orchestration service and dependent modules. It highlights where implementations align and where gaps or mismatches exist so we can prioritise contract tests and follow-up work.

## Legend
- ✅ Implemented in both client and service
- ⚠️ Partial – endpoint exists but payload/behaviour needs confirmation
- ❌ Missing – client references endpoint not present in service (or vice versa)

## 1. Orchestration ↔ Whisper (Audio Service)

| Client Call (modules/orchestration-service/src/clients/audio_service_client.py) | Service Endpoint (modules/whisper-service/src/api_server.py) | Status | Notes |
| --- | --- | --- | --- |
| `GET /health` | `/health` | ✅ | Returns JSON `{status, ...}` used by health monitor. |
| `GET /api/models` | `/api/models` | ✅ | Provides `available_models`. |
| `GET /api/device-info` | `/api/device-info` | ✅ | Device metadata (cpu/gpu). |
| `POST /transcribe` (multipart) | `/transcribe` | ✅ | Accepts audio file. Need to confirm expected form field names (`audio`, `language`, etc.). |
| `POST /transcribe/<model>` | `/transcribe/<model_name>` | ⚠️ | Endpoint exists; verify required payload fields. |
| `POST /api/process-chunk` | ❌ | No matching endpoint. |
| `POST /api/realtime/start` | ✅ | Wrapper around streaming session configuration/start. |
| `POST /api/realtime/audio` | ✅ | Proxies to `/stream/audio`. |
| `POST /api/realtime/stop` | ✅ | Proxies to `/stream/stop`. |
| `GET /api/realtime/status/{session_id}` | ✅ | Returns session config and state. |
| `POST /api/analyze` | ✅ | Provides basic audio metrics (duration/RMS/peak). |
| `GET /performance` | `/performance` | ✅ | Returns metrics JSON. |
| `POST /api/process-pipeline` | ✅ | Executes transcription pipeline and returns metadata. |
| `POST /api/start-streaming` | ✅ | Alias for realtime start. |
| `GET /api/stream-results/{session_id}` | ✅ | Returns recent session transcriptions. |
| `GET /api/processing-stats` | ✅ | Returns service status/metrics. |
| `GET /api/download/{request_id}` | ⚠️ | Returns transcript if stored; otherwise 404. |

**Takeaway:** Core health/model/device endpoints align, but most advanced pipeline/realtime calls in the client are unsupported. We should either implement them in Whisper or remove/gate the client paths before writing contract tests.

## 2. Orchestration ↔ Translation Service

| Client Call (modules/orchestration-service/src/clients/translation_service_client.py) | Service Endpoint (modules/translation-service/src/api_server.py) | Status | Notes |
| --- | --- | --- | --- |
| `GET /api/health` | `/api/health` | ✅ | Returns status JSON. |
| `GET /api/device-info` | `/api/device-info` | ✅ | Provides backend/device data. |
| `GET /api/languages` | ✅ | Wrapper for supported languages list. |
| `POST /api/detect` | ✅ | Wrapper for `/detect_language`. |
| `POST /api/translate` | `/api/translate` | ✅ | JSON contract matches client usage. |
| `GET /api/status` | `/api/status` | ✅ | Already exposed. |
| `GET /api/performance` | ✅ | Returns basic counters (active sessions, cached prompts). |

**Takeaway:** Translation flows (health/device/translate) work, but language/detect/status contracts need adjustment. Consider adding endpoints or updating the client to align.

## 3. Config Sync & Worker Mode

| Producer | Event Type | Consumer | Notes |
| --- | --- | --- | --- |
| Settings API (API mode) | `*Updated` | Monitoring/clients | Emitted after immediate updates. |
| Settings API (worker mode) | `*UpdateRequested` | `worker.config_sync_worker` | Worker currently handles system/service updates; user settings path logs TODO. |

## 4. Bot Lifecycle Events

| API Endpoint | Queue Event | Notes |
| --- | --- | --- |
| `POST /bots/spawn` | `BotRequested` | Worker not yet consuming; event ready for future bot orchestrator. |
| `POST /bots/{id}/terminate` | `BotStopRequested` | Same as above. |

## 5. Monitoring/Health

- Health monitor polls `/health`, `/api/system/health`, `/performance` synchronously. Monitoring worker not yet implemented; metrics cache still local to API.

## Next Steps
1. Decide whether to implement missing Whisper/Translation endpoints or trim the clients (prevents 404s when contract tests land).
2. Add schema assertions for supported endpoints (pytest contracts, or JSONSchema files).
3. Extend config worker to handle user settings events and confirm translation/whisper updates propagate via queue.
4. Once interfaces stabilise, formalise versioned API docs for each service.

### Proposed Contract Test Matrix
- **Whisper health/model/device**: pytest integration hitting `/health`, `/api/models`, `/api/device-info`; assert JSON keys used by orchestration (`status`, `available_models`, etc.).
- **Whisper transcription**: fixture uploading sample WAV to `/transcribe`; confirm response contains `text`, `processing_time`, etc. (requires sample audio + service running or mocked response).
- **Translation translate/device**: call `/api/translate` with known text (stub backend for deterministic result) and `/api/device-info` to validate structure.
- **Config sync events**: emit `SystemSettingsUpdateRequested` into Redis stream via test harness, run worker in async test, assert `ConfigurationSyncManager.update_configuration` called/mutates state.
- **API fallback behaviour**: when `CONFIG_SYNC_MODE=api`, direct updates should succeed even if Redis unavailable (mock `EventPublisher.publish`).
- **Queue publishing**: instrument `EventPublisher` in tests to ensure audio/bot/settings endpoints emit expected event payloads without raising.

These tests can live under `modules/orchestration-service/tests/contracts/` with docker-compose profiles to spin up dependent services or mocked responders.
