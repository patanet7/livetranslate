## FunASR Streaming Demo (ZH → EN) – Working Implementation Plan

This is a living document to guide the implementation of a minimal, discrete FunASR-based streaming demo that integrates with the existing orchestration and translation services. It focuses on realtime microphone capture, chunked streaming ASR, and live translation to English.

### Goals
- Build a standalone `funasr` module with a simple FastAPI WebSocket endpoint for streaming Chinese speech to text.
- Integrate with `@orchestration-service` via a new WebSocket router that relays audio and fans out translations to English.
- Add a minimal FE page to stream mic audio, view partial/final transcript and the English translation in near-realtime.

### Non-Goals (for this demo)
- No diarization or multi-speaker correlation.
- No file uploads; mic-only streaming.
- No long-term persistence; in-memory session state.

---

## 1) Architecture Overview

- Frontend (`modules/frontend-service`)
  - New page: `FunASRDemo` for mic capture and UI
  - WebSocket client to Orchestration: `/api/funasr/realtime/{sessionId}`
  - Sends PCM16 16k mono chunks (100–300 ms) as base64 via WS
  - Displays `asr_partial`/`asr_final` and English `translation_*`

- Orchestration (`modules/orchestration-service`)
  - New router: `routers/funasr.py`
  - WebSocket endpoint: `/api/funasr/realtime/{session_id}`
  - Relays FE `audio_chunk` to FunASR service WS
  - Emits `asr_partial`/`asr_final` upstream to FE
  - Calls translation API for English; emits `translation_partial`/`translation_final`

- FunASR Service (`/funasr` – standalone)
  - WebSocket endpoint: `/ws/stream?session_id=...`
  - Model: Paraformer streaming (Chinese) or SenseVoice with VAD disabled for simplicity
  - Maintains `cache` across chunks; emits partial and final text

Ports & env (defaults):
- FunASR WS: `ws://localhost:5006/ws/stream`
- Orchestration HTTP/WS: `http://localhost:3000`, WS path proxied under `/api/funasr/*`
- Env (orchestration): `FUNASR_WS_URL=ws://localhost:5006/ws/stream`

---

## 2) Data & Stream Contracts

### 2.1 Frontend → Orchestration (WS `/api/funasr/realtime/{session_id}`)
- Client messages
  - `config`
    - Example:
      ```json
      {"type":"config","languages":["en"],"emitPartials":true}
      ```
  - `audio_chunk`
    - Base64 PCM16LE mono 16kHz; chunk 100–300 ms
    - Example:
      ```json
      {"type":"audio_chunk","data":"<base64_pcm16>","sampleRate":16000,"channels":1,"chunkMs":200}
      ```
  - `end`
    - Flush and finalize current session chunking
    - Example:
      ```json
      {"type":"end"}
      ```

- Server messages
  - ASR:
    - `{"type":"asr_partial","text":"..."}`
    - `{"type":"asr_final","text":"..."}`
  - Translation (ZH → EN):
    - `{"type":"translation_partial","lang":"en","text":"..."}`
    - `{"type":"translation_final","lang":"en","text":"..."}`
  - Errors:
    - `{"type":"error","message":"..."}`

### 2.2 Orchestration ↔ FunASR (WS `FUNASR_WS_URL`)
- Orchestration forwards the same `audio_chunk` payloads to FunASR.
- FunASR responds with `asr_partial` / `asr_final` messages; Orchestration relays them to FE and triggers translation.

### 2.3 Orchestration → Translation API
- Preferred: Streaming-like behavior using existing endpoints. For a minimal version, call the standard translate endpoint on each `asr_final` and optionally on `asr_partial` with throttling.
- Endpoints available in `modules/orchestration-service/src/routers/translation.py`:
  - `POST /api/translation/translate` (root `/api/translation/` also supported)
  - Optional streaming route: `POST /api/translation/stream` (if desired for partials)
- Request example:
  ```json
  {"text":"你好世界","source_language":"zh","target_language":"en","model":"default","quality":"balanced"}
  ```
- Response example (normalized by orchestration to FE events):
  ```json
  {"translated_text":"Hello world","source_language":"zh","target_language":"en","confidence":0.95}
  ```

---

## 3) Audio & Model Settings

### 3.1 Audio capture (Frontend)
- Constraints: 16,000 Hz, mono, disable echo/noise/AGC if pipeline handles it
- Chunk size: 200 ms (acceptable 100–300 ms)
- Format: PCM16 little-endian; send as base64

### 3.2 FunASR model
- Option A: `paraformer-zh-streaming`
  - Good accuracy; streaming-friendly; expects 16k mono
- Option B (later): SenseVoiceSmall (more features, keep VAD off for simplicity)

### 3.3 FunASR inference strategy
- Maintain `cache` dict per session across chunks
- Emit `asr_partial` often; emit `asr_final` on `end` or clear boundary
- Normalize inputs: convert PCM16 to float32 [-1, 1]

---

## 4) Step-by-Step Implementation

### Step 0 – Repository scaffolding (funasr module)
- Files to add under `/funasr`:
  - `src/server.py`: FastAPI app with WS `/ws/stream`
  - `src/streaming_asr.py`: FunASR wrapper (load model, handle cache, generate)
  - `requirements.txt` or `pyproject.toml`: `funasr`, `fastapi`, `uvicorn`, `numpy`
  - `scripts/start-funasr.ps1` and `scripts/start-funasr.sh`
  - `README.md` (quickstart)

Verification:
- Run server: `uvicorn src.server:app --host 0.0.0.0 --port 5006`
- Connect via WS client; send dummy PCM16; receive `asr_partial` or errors

### Step 1 – FunASR WS API
- Implement `/ws/stream?session_id=...`:
  - On connect: create session state with model `cache`
  - On `audio_chunk`: decode base64 → Int16 → float32; call `model.generate(..., cache)`
    - If text available: `send({"type":"asr_partial","text":...})`
  - On `end`: call `model.generate(..., is_final=True)` to flush; send `asr_final`
  - On error: send `{"type":"error","message":...}`

Verification:
- Unit-test chunk decode + generate pipeline with 1–3 chunks and a short wav sample
- Confirm partials appear; final sent after `end`

### Step 2 – Orchestration router `funasr.py`
- Add `@router.websocket("/realtime/{session_id}")` with:
  - Accept FE WS; parse `config` (languages default `["en"]`)
  - Open backend WS to `FUNASR_WS_URL` (pass `session_id`)
  - Relay `audio_chunk` and `end`
  - On `asr_partial`/`asr_final` from FunASR:
    - Forward to FE
    - For ZH→EN: call `POST /api/translation/translate` with `{ source_language: "zh", target_language: "en" }`
      - For partials: throttle (e.g., every 800 ms) to reduce load
      - Emit `translation_partial` and `translation_final` (normalized)
- Register the router in `main_fastapi.py` with prefix `/api/funasr`

Verification:
- Connect FE to `/api/funasr/realtime/{session}`; stream test PCM16; observe ASR events relayed
- Stub translation temporarily to validate orchestration flow; then enable real calls

### Step 3 – Frontend `FunASRDemo` page
- UI (simple): device selector, Start/Stop, language select (preselect EN), transcript area, translation area
- Audio capture:
  - Use `AudioWorklet` or `ScriptProcessor` to get PCM16 frames at 16k mono
  - 200 ms chunks; base64 encode; send `audio_chunk` messages
- WS client:
  - Connect to `/api/funasr/realtime/{sessionId}`; send `config` with `{ languages: ["en"] }`
  - Handle `asr_partial`/`asr_final` → display
  - Handle `translation_partial`/`translation_final` → display
  - Handle reconnect and clean stop (`end`)

Verification:
- Visual sanity: typing-like partials; final lines on stop
- Latency target: partials within ~300–600 ms after speech

### Step 4 – Polish & Hardening
- Backpressure: drop or coalesce chunks if FE outpaces network
- Rate-limit translation partials (e.g., debounce 800–1200 ms)
- Health check endpoint in FunASR for readiness/liveness
- Graceful shutdown of model on SIGTERM/SIGINT

---

## 5) API In/Out Verification Matrix

| Stage | Input | Output | Validation |
| --- | --- | --- | --- |
| FE capture | Float32 PCM from MediaStream | Int16 PCM16 base64 | Waveform level is non-zero; 16k mono; chunk size ~200 ms |
| FE→Orch WS | `audio_chunk` JSON | 101 Switching to WS | Orchestration receives messages; logs chunk counts |
| Orch→FunASR WS | `audio_chunk` JSON | `asr_partial`/`asr_final` | FunASR returns partials in <600 ms; final on `end` |
| Orch→Translation | JSON POST translate | translated text JSON | Returns English string; 200–700 ms typical |
| Orch→FE WS | normalized JSON events | UI updates | UI shows partials and finals without flicker |

Edge cases to test:
- Empty audio (silence): no spurious text; final empty or omitted
- Rapid start/stop: no crashes; session cleaned up
- Large spikes in volume: no clipping/NaNs

---

## 6) Configuration & Defaults

- Orchestration env:
  - `FUNASR_WS_URL=ws://localhost:5006/ws/stream`
  - `TRANSLATION_SERVICE_URL` already configured (falls back to default)
- FunASR service env:
  - `MODEL_NAME=paraformer-zh-streaming`
  - `DEVICE=cpu` (or `cuda:0` if available)
  - Optional: `VAD_ENABLED=false` (keep off for this demo)

---

## 7) Operational Runbook (Local Dev)

Order of start (recommend):
1. FunASR service: `uvicorn src.server:app --host 0.0.0.0 --port 5006`
2. Orchestration service (existing dev script)
3. Frontend service (existing dev script)

Quick sanity test:
- Hit `ws://localhost:5006/ws/stream` with a WS client; send 1–2 chunks; observe partial text
- From FE demo page, speak Chinese phrases; observe partials and English translation

---

## 8) Acceptance Criteria

- Mic Start/Stop works reliably; no unhandled errors in console/logs
- Partial ASR text appears within ~300–600 ms after speech begins
- Final ASR and English translation appear on stop or flush
- ZH→EN translation correctness is reasonable for short phrases
- System handles silence and rapid toggles gracefully

---

## 9) Milestones & Tasks (Checklist)

- [ ] FunASR module scaffolding (server, wrapper, deps, scripts)
- [ ] FunASR WebSocket streaming functional with partial/final
- [ ] Orchestration `funasr.py` router WS relay + translation fan-out
- [ ] Router registered at `/api/funasr` and env wired (`FUNASR_WS_URL`)
- [ ] Frontend `FunASRDemo` page with PCM16 capture + WS client
- [ ] Basic UI for partial/final transcripts and English translations
- [ ] Debounce translation partials; final on ASR final
- [ ] Manual end-to-end runbook verified
- [ ] Edge cases and error handling validated

---

## 10) Future Enhancements (Post-demo)

- Add VAD for better finalization boundaries
- Add language detection fallback (SenseVoice)
- Support multiple target languages beyond English
- Add adaptive chunk sizing (network- or latency-aware)
- Optional Opus ingestion path (server-side decode) for bandwidth

---

## 11) Reference Notes (FunASR)

- Paraformer streaming typical settings:
  - `chunk_size = [0, 10, 5]` (≈600 ms windows, 300 ms lookahead)
  - Maintain `cache` across calls; set `is_final=True` to flush
- Input expected at 16kHz mono; convert Int16 PCM to float32 [-1, 1]

---

### Working Notes
- Keep this document updated during implementation with any deviations and their rationale.
- Log any API contract adjustments here and in code-level docstrings.
