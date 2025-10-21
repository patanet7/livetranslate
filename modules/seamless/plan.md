## Seamless Streaming Demo (ZH → EN) – Working Implementation Plan

This document defines a minimal, discrete demo using Meta’s Seamless (SeamlessStreaming preferred) to perform speech-to-text translation from Chinese (cmn/zh) to English (eng) in near-realtime. We reuse the same demo principles as the FunASR plan: browser mic capture → chunked WS stream → backend model → live UI updates.

### Goals
- Standalone `seamless` module exposing a simple FastAPI WebSocket for streaming speech translation (S2TT) ZH → EN.
- Orchestration WebSocket router that relays audio chunks to the Seamless service and emits partial/final translated text to the frontend.
- Minimal FE page to stream mic audio and display partial/final translations.

### Non-Goals (for this demo)
- No diarization or voice cloning.
- No separate translation service calls (Seamless provides translation).
- No persistence; in-memory sessions only.

---

## 1) Architecture Overview

- Frontend (`modules/frontend-service`)
  - New page: `SeamlessDemo`
  - WebSocket to Orchestration: `/api/seamless/realtime/{sessionId}`
  - Sends PCM16 16k mono chunks (100–300 ms) as base64
  - Renders `translation_partial` / `translation_final` for English

- Orchestration (`modules/orchestration-service`)
  - New router: `routers/seamless.py`
  - WS endpoint: `/api/seamless/realtime/{session_id}`
  - Relays FE `audio_chunk` and `end` to Seamless service
  - Forwards Seamless partial/final translations to FE (no extra translation step)

- Seamless Service (`modules/seamless` – standalone)
  - WS endpoint: `/ws/stream?session_id=...` on port 5007
  - Loads SeamlessStreaming (preferred) or a minimal S2TT pipeline that can operate on short windows
  - Produces partial and final English text

Ports & env (defaults):
- Seamless WS: `ws://localhost:5007/ws/stream`
- Orchestration base: `http://localhost:3000`
- Env (orchestration): `SEAMLESS_WS_URL=ws://localhost:5007/ws/stream`

---

## 2) Data & Stream Contracts

### 2.1 Frontend → Orchestration (WS `/api/seamless/realtime/{session_id}`)
- Client messages
  - `config` (optional; Seamless handles translation)
    - Example:
      ```json
      {"type":"config","source":"cmn","target":"eng","emitPartials":true}
      ```
  - `audio_chunk`
    - Base64 PCM16LE mono 16kHz; chunk 100–300 ms
    - Example:
      ```json
      {"type":"audio_chunk","data":"<base64_pcm16>","sampleRate":16000,"channels":1,"chunkMs":200}
      ```
  - `end`
    - Flush and finalize
    - Example:
      ```json
      {"type":"end"}
      ```

- Server messages (to FE)
  - `{"type":"translation_partial","lang":"en","text":"..."}`
  - `{"type":"translation_final","lang":"en","text":"..."}`
  - `{"type":"error","message":"..."}`

### 2.2 Orchestration ↔ Seamless (WS `SEAMLESS_WS_URL`)
- Orchestration forwards `config` (src=`cmn`, tgt=`eng`), `audio_chunk`, and `end`.
- Seamless responds with translation partials/finals; orchestration forwards to FE.

---

## 3) Audio & Model Settings

### 3.1 Audio capture (Frontend)
- 16,000 Hz mono; chunk size 200 ms (100–300 ms acceptable)
- PCM16 little-endian; base64 transport over WS

### 3.2 Seamless model
- Preferred: SeamlessStreaming S2TT (speech-to-text translation) with `src_lang=cmn`, `tgt_lang=eng`.
- Fallback (if streaming API is not accessible): micro-batch windowing (e.g., 1–2 s) to produce pseudo-partials, understanding increased latency.

---

## 4) Step-by-Step Implementation

### Step 0 – Module scaffolding (`modules/seamless`)
- Files:
  - `src/server.py`: FastAPI app with WS `/ws/stream`
  - `src/streaming_st.py`: Seamless wrapper; initialize model; handle chunked inference
  - `requirements.txt` or `pyproject.toml`: deps (fastapi, uvicorn, numpy, torch, fairseq2, seamless_communication as applicable)
  - `scripts/start-seamless.ps1` / `scripts/start-seamless.sh`
  - `README.md`: quickstart and model notes

Verification:
- Start server locally on 5007; connect with WS client; send dummy PCM16; ensure partial/final messages arrive or clear error surfaced.

### Step 1 – Seamless WS API
- `/ws/stream?session_id=...`
  - On connect: establish session state with model, `src=cmm/zh(cmn)`, `tgt=eng`
  - On `config`: allow overrides, default to cmn→eng
  - On `audio_chunk`: base64→Int16→float32; pass to streaming inference; send `translation_partial` when available
  - On `end`: finalize and send `translation_final`
  - On errors: send `{"type":"error","message":...}` and optionally close

Verification:
- Local test with short Mandarin audio verifies reasonable English output.

### Step 2 – Orchestration router `seamless.py`
- `@router.websocket("/realtime/{session_id}")`
  - Accept FE WS; read optional `config`
  - Open backend WS to `SEAMLESS_WS_URL` (propagate `config`)
  - Relay `audio_chunk` and `end`
  - Forward `translation_partial` and `translation_final` to FE (no extra translation step)
- Register in `main_fastapi.py` with prefix `/api/seamless`

Verification:
- WS echo test; then end-to-end with Seamless service running.

### Step 3 – Frontend `SeamlessDemo` page
- UI: device selector, Start/Stop, source/target prefilled (cmn→eng), live translation panel
- Mic capture:
  - Use `AudioWorklet` or `ScriptProcessor` to get PCM16 16k mono frames
  - Chunk to ~200 ms; base64; send as `audio_chunk`
- WS client:
  - Connect `/api/seamless/realtime/{sessionId}`; send `config` on open
  - Render `translation_partial` in-progress; commit on `translation_final`

Verification:
- Latency: partials within ~300–800 ms (hardware dependent)
- Stop/Start stability; no console errors

### Step 4 – Polish
- Debounce partial updates (e.g., 500–800 ms) to reduce flicker
- Health endpoints, graceful shutdown, backpressure handling

---

## 5) API In/Out Verification Matrix

| Stage | Input | Output | Validation |
| --- | --- | --- | --- |
| FE capture | Float32 PCM → Int16 PCM16 | Base64 chunks (200 ms) | Levels non-zero; 16k mono; timing stable |
| FE→Orch WS | `audio_chunk` | 101 WS OK | Orchestration counts chunks, no drops |
| Orch↔Seamless | `audio_chunk` / `end` | `translation_*` | Partials < 800 ms, final on end |
| Orch→FE WS | normalized events | UI updates | Minimal flicker; correct finalization |

Edge cases:
- Silence: no spurious output; final empty or omitted
- Rapid stop/start: session cleanup; no stale messages
- Long speech: periodic partials; memory bounded

---

## 6) Configuration & Defaults

- Orchestration:
  - `SEAMLESS_WS_URL=ws://localhost:5007/ws/stream`
- Seamless service:
  - `SRC_LANG=cmn`
  - `TGT_LANG=eng`
  - `DEVICE=cpu` (or `cuda:0`)

---

## 7) Operational Runbook (Local)

1) Start Seamless service:
   - `uvicorn src.server:app --host 0.0.0.0 --port 5007`
2) Start Orchestration (existing dev script)
3) Start Frontend (existing dev script) → open `SeamlessDemo`

Quick sanity:
- Speak Mandarin; observe streaming English partials and final text.

---

## 8) Acceptance Criteria

- Mic start/stop works; no unhandled errors.
- Partial English translations appear during speech with acceptable latency.
- Final translations appear on end; UI commits text without duplication.
- Robust to silence and rapid toggles.

---

## 9) Tasks (Checklist)

- [ ] Module scaffolding (`modules/seamless`)
- [ ] Seamless WS server (streaming or windowed fallback)
- [ ] Orchestration `seamless.py` router and registration
- [ ] Frontend `SeamlessDemo` page and route
- [ ] End-to-end local verification
- [ ] Debounce/flicker polish

---

## 10) Start Here – Immediate Steps & Commands

1. Create Seamless WS service (minimal skeleton)
   - Files: `modules/seamless/src/server.py`, `modules/seamless/src/streaming_st.py`
   - Deps: `modules/seamless/requirements.txt` (fastapi, uvicorn, numpy; add Seamless deps when ready)
   - Command:
     ```bash
     cd modules/seamless
     uvicorn src.server:app --host 0.0.0.0 --port 5007
     ```

2. Add orchestration router proxy
   - File: `modules/orchestration-service/src/routers/seamless.py`
   - Register in `modules/orchestration-service/src/main_fastapi.py` with prefix `/api/seamless`
   - Env: `SEAMLESS_WS_URL=ws://localhost:5007/ws/stream`

3. Frontend demo page
   - File: `modules/frontend-service/src/pages/SeamlessDemo/index.tsx`
   - Route: add `/seamless-demo` in `src/App.tsx`
   - Behavior: mic capture → PCM16 base64 chunks → WS → render `translation_*`
   - Add Sessions page: `modules/frontend-service/src/pages/SeamlessSessions/index.tsx` and route `/seamless-sessions` (list sessions, download transcripts)

4. Run end-to-end
   - Start Seamless (step 1), Orchestration (dev script), Frontend (dev script)
   - Open `/seamless-demo` and test speech → translation

5. Optional: enable SQLite persistence (next iteration)
   - Add SQLite repo/models and retrieval endpoints under `/api/seamless/*` (DONE)
   - Implement FE Sessions page (DONE)

6. README & start scripts
   - `modules/seamless/README.md` with quick start, API, env
   - `modules/seamless/scripts/start-seamless.(sh|ps1)` to bootstrap service

---

### Working Notes
- If full SeamlessStreaming is heavy to embed, start with a windowed S2TT fallback and swap to true streaming once dependencies are ready.
- Keep contract parity with the FunASR demo for easier UI reuse.

---

## 10) Local Persistence (SQLite)

Purpose: keep a lightweight local record of sessions, streaming events, and final translations for demo auditing and quick analysis.

Storage choice: SQLite (single-file DB), minimal setup, cross-platform, perfect for single-system use.

Defaults
- DB path (orchestration): `SEAMLESS_DB_PATH` env, default `./data/seamless_demo.db`
- Auto-create schema on startup if not present

Schema (proposed)
- `seamless_sessions`
  - `id` TEXT PRIMARY KEY (session_id)
  - `created_at` DATETIME
  - `ended_at` DATETIME NULL
  - `source_lang` TEXT DEFAULT 'cmn'
  - `target_lang` TEXT DEFAULT 'eng'
  - `client_ip` TEXT NULL
  - `user_agent` TEXT NULL

- `seamless_events`
  - `id` INTEGER PRIMARY KEY AUTOINCREMENT
  - `session_id` TEXT REFERENCES seamless_sessions(id) ON DELETE CASCADE
  - `event_type` TEXT CHECK(event_type IN ('audio_chunk','asr_partial','asr_final','translation_partial','translation_final','error'))
  - `payload` TEXT NOT NULL (JSON string; store minimal body)
  - `timestamp_ms` INTEGER NOT NULL
  - INDEX on (session_id, timestamp_ms)

- `seamless_transcripts`
  - `id` INTEGER PRIMARY KEY AUTOINCREMENT
  - `session_id` TEXT REFERENCES seamless_sessions(id) ON DELETE CASCADE
  - `lang` TEXT NOT NULL DEFAULT 'en'
  - `text` TEXT NOT NULL
  - `is_final` INTEGER NOT NULL DEFAULT 1
  - `created_at` DATETIME NOT NULL
  - INDEX on (session_id, lang, is_final)

Write strategy
- On FE WS connect (orchestration): insert `seamless_sessions`
- On incoming events from Seamless or FE:
  - Append to `seamless_events` (throttle audio_chunk persistence if too chatty; e.g., record every Nth chunk or aggregate per second)
- On `translation_final`: insert into `seamless_transcripts` and mark session `ended_at` on `end`

Retrieval API (orchestration)
- `GET /api/seamless/sessions?limit=50&offset=0` → list recent sessions
- `GET /api/seamless/sessions/{session_id}` → summary (langs, created/ended, counts)
- `GET /api/seamless/sessions/{session_id}/events?from_ms=&to_ms=&types=` → paged stream of events
- `GET /api/seamless/sessions/{session_id}/transcripts` → final translations (and optionally partials)
- Optional admin:
  - `DELETE /api/seamless/sessions/{session_id}`

Implementation steps
1) Orchestration dependencies
   - Add `SQLAlchemy` (async) or `aiosqlite`. If reusing existing DB utilities, extend them with a file-backed SQLite config.
2) Create models & repo
   - `modules/orchestration-service/src/database/seamless_models.py`
   - `modules/orchestration-service/src/database/seamless_repository.py`
3) Initialize DB
   - On startup, ensure DB path exists; run `create_all()` for SQLite
4) Wire router
   - In `routers/seamless.py`, on connect/create session; on events, persist per strategy; on end, finalize
5) Add retrieval endpoints under `/api/seamless/*` as listed

Verification
- Start a session, speak a few phrases, stop
- Verify: one row in `seamless_sessions`, multiple `seamless_events`, at least one `translation_final` row in `seamless_transcripts`
- Call retrieval endpoints and confirm shapes


