# Transcription Service

Real-time speech-to-text with pluggable backends, Silero VAD, and language detection.

## Architecture

```
src/
├── api.py                  # FastAPI WebSocket /api/stream — main streaming endpoint
├── language_detection.py   # WhisperLanguageDetector (SustainedLanguageDetector adapter)
├── vac_online_processor.py # Async queue-based VAC: prebuffer → stride → overlap retention
├── backends/
│   ├── manager.py          # BackendManager — VRAM budgeting, LRU eviction, circuit breaker
│   ├── vllm_whisper.py     # vLLM-MLX backend (Apple Silicon, port 8005)
│   ├── mlx_whisper.py      # MLX-Whisper backend
│   └── whisper.py          # faster-whisper backend (GPU/CPU)
├── transcription/
│   ├── hallucination_filter.py  # Repetition + degeneration suppression
│   ├── domain_prompt_helper.py  # Language-specific initial prompts
│   └── text_analysis.py         # CJK detection, text normalization
├── silero_vad_iterator.py  # FixedVADIterator for voice activity detection
├── sentence_segmenter.py   # Sentence boundary detection
├── registry.py             # ModelRegistry — hot-reloadable backend configs
└── main.py                 # Entry point
```

## Commands

```bash
uv run pytest modules/transcription-service/tests/ -v                    # All tests
uv run pytest modules/transcription-service/tests/ -v -m "not slow"      # Skip slow tests
uv run pytest modules/transcription-service/tests/test_session_restart.py -v  # Session restart
uv run pytest modules/transcription-service/tests/test_language_detection.py -v  # Language detection
```

## Key Components

### `SessionState` (api.py)
Dataclass holding per-connection state: `source_language`, `lock_language`, `lang_detector`, `vac_processor`. The `source_language` field maps to `SessionConfig.source_language` in orchestration.

### `WhisperLanguageDetector` (language_detection.py)
Adapter wrapping `SustainedLanguageDetector` with hysteresis-based switching. Parameters: `confidence_margin=0.2`, `min_dwell_frames=4`, `min_dwell_ms=10000`. Only runs when `lock_language=False` (interpreter mode).

### `VACOnlineProcessor` (vac_online_processor.py)
Async queue-based chunking processor. `reset()` clears buffer/counters for session restart on language switch. Preserves config (stride, overlap, prebuffer).

### Session Restart
On sustained language switch (`WhisperLanguageDetector.update()` returns non-None): flush `_prev_segment_text`, reset `HallucinationFilter`, reset `VACOnlineProcessor`. Only fires when `lock_language=False`.

## Gotchas

- **`is_final` flag** means "ends with punctuation", NOT "final transcription" — collect ALL segments with text
- **Test fixtures loading ML models MUST use `yield` + teardown with `gc.collect()`** — bare `return` causes OOM on Apple Silicon (unified memory). Past incident: 250GB crash
- **`source_language`** (not `language`) is the field name in `SessionState` — matches orchestration's `SessionConfig`
- **RMS gating removed** from `ready_for_inference()` — Whisper's `no_speech_prob` (checked post-inference) is more reliable than pre-inference energy gating on downsampled browser audio
