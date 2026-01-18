Seamless Streaming Demo (ZH → EN)

This demo service performs streaming speech-to-text translation from Mandarin Chinese (cmn) to English (eng) using Meta's Seamless M4T via HuggingFace Transformers. Intended to run locally and integrate with Orchestration and Frontend.

Features
- WebSocket API `/ws/stream` for incremental audio ingestion and streaming translations
- Accepts 16 kHz mono PCM16 chunks (base64) from the browser
- Emits `translation_partial` and `translation_final` messages
- CPU or GPU (`DEVICE=cuda:0`)

Quick Start
1) Install deps:
   - `cd modules/seamless && pip install -r requirements.txt`
2) Start service:
   - Bash: `./scripts/start-seamless.sh`
   - PowerShell: `./scripts/start-seamless.ps1`
3) Orchestration env (optional):
   - `SEAMLESS_WS_URL=ws://localhost:5007/ws/stream`
   - `SEAMLESS_DB_PATH=./data/seamless_demo.db`
4) Frontend: open `/seamless-demo`

API
- WS `ws://localhost:5007/ws/stream?session_id=<id>`
  - Client: `config`, `audio_chunk`, `end`
  - Server: `translation_partial`, `translation_final`, `error`

Notes
- Uses a rolling window (~3s) to produce partials.
- FE demo handles 16k mono PCM16 chunking.
