# Service Overview

Quick reference for active core services and external inference backends.

## Dashboard Service (:5173 dev, :3000 prod)
- **Tech**: SvelteKit + Svelte 5
- **Purpose**: Real-time UI (loopback page, settings, meeting controls)
- **Hardware**: Browser
- **Module**: `modules/dashboard-service/`

## Orchestration Service (:3000)
- **Tech**: Python + FastAPI
- **Purpose**: Central coordinator, WebSocket hub, translation orchestration
- **Hardware**: CPU-optimized
- **Module**: `modules/orchestration-service/`

## Transcription Service (:5001)
- **Tech**: Python (pluggable backends: vLLM-MLX, faster-whisper)
- **Purpose**: Speech-to-text with VAD
- **Hardware**: GPU/Apple Silicon
- **Module**: `modules/transcription-service/`

## External Inference Services

### vLLM-MLX (Apple Silicon)
- **:8005** — Whisper transcription inference
- **:8006** — Qwen3-4B-4bit translation inference (LLM API)
- **Model**: `mlx-community/Qwen3-4B-4bit`

### Ollama (Alternative LLM)
- **:11434** — OpenAI-compatible LLM API
- **Model**: `qwen3.5:7b` (or other compatible models)

---

## Archived Services

| Service | Notes |
|---------|-------|
| Whisper Service (was :5001) | Replaced by Transcription Service with pluggable backends |
| Translation Service (was :5003) | Translation integrated into Orchestration Service via LLM API (vLLM-MLX :8006 or Ollama :11434) |
| Frontend Service | Replaced by Dashboard Service |

See [Container Architecture](./README.md) for detailed breakdown.
