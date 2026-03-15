# Plan 6: Streaming Pipeline Tuning & Benchmark Framework

## Context

Plan 5 proved the full e2e pipeline works (Chinese meeting → 13 segments → 13 translations).
Current CER: 22% on Chinese. Target: <10%.

## Headroom Analysis

Inference via vllm-mlx: **1.2s per 5s chunk** (4x faster than real-time).
At stride_s=4.0: **2.8s of free compute per cycle**.

This headroom can be used for:
- Wider overlap (re-process more context per chunk)
- Two-pass refinement (fast draft → accurate final)
- Beam search on second pass
- Concurrent translation (already using this)

## Priority Fixes (CER 22% → 10-12%)

### P0: Immediate Fixes (30 min each)

1. **`temperature=(0.0,)` only** — remove fallback cascade that causes 6x latency on bad chunks
2. **Fix TranslationConfig default** — change `thomas-pc:11434` to `localhost`
3. **`no_speech_threshold=0.6`** — pass to `model.transcribe()` directly

### P1: High Impact (2-3 hours each)

4. **Expose `no_speech_prob` from Whisper `info`** — add to TranscriptionResult, gate on >0.6
5. **Reduce initial_prompt to ≤80 chars** — trim to last complete sentence, not arbitrary boundary
6. **Chinese number normalization** — `cn2an` post-processing (5-8 CER points)
7. **Language consistency filter** — discard segments where detected lang ≠ session lang AND confidence < 0.7
8. **VAD gating in VACOnlineProcessor** — RMS energy check before inference (threshold 0.005)

### P2: Structural Improvements (4-6 hours each)

9. **Enable word_timestamps=True** — time-based dedup replaces brittle word matching
10. **SimpleStabilityTracker** — 30-line time-anchored stable/unstable tracking
11. **Translation triggers on stable text** — not `is_final` (punctuation-based)
12. **Overlap increase to 1.0-1.5s** — with dedup max_overlap bumped to 12 words

### P3: Two-Pass Architecture (new feature)

13. **Fast first pass**: stride=3.0s, overlap=0.5s, beam=1 → real-time captions
14. **Refinement pass**: every 15s, re-process the last 15s as one chunk with beam=3 → corrected transcript
15. **Diff the two passes**: update displayed captions with corrections from refinement

### P4: Benchmark Framework

16. **`tools/transcription_benchmark/`** module with:
    - Per-config WER/CER calculation using `jiwer`
    - Chinese number normalization before CER comparison
    - Config matrix: stride × overlap × prebuffer × beam × language
    - Results appended to `results/transcription_index.jsonl`
    - `just benchmark` recipe
17. **Ground truth alignment** — don't compare per-turn, compare full transcript with edit distance
18. **Translation quality** — BLEU/COMET scores alongside CER
19. **Latency percentiles** — p50, p95, p99 for time-to-first-caption and per-segment gap

## Config Changes

### model_registry.local.yaml (Apple Silicon / vllm-mlx)

```yaml
language_routing:
  en:
    backend: vllm
    model: large-v3-turbo
    chunk_duration_s: 5.0
    stride_s: 4.0          # was 4.5
    overlap_s: 1.0          # was 0.5
    prebuffer_s: 3.0
    beam_size: 1
    vad_threshold: 0.5
  zh:
    backend: vllm
    model: large-v3-turbo
    chunk_duration_s: 5.0
    stride_s: 4.0
    overlap_s: 1.0
    prebuffer_s: 3.0
    beam_size: 1
    vad_threshold: 0.45     # more sensitive for tonal
  "*":
    backend: vllm
    model: large-v3-turbo
    chunk_duration_s: 5.0
    stride_s: 4.0
    overlap_s: 1.0
    prebuffer_s: 3.0
    beam_size: 1
    vad_threshold: 0.5
```

### model_registry.yaml (thomas-pc CUDA)

```yaml
language_routing:
  en:
    backend: whisper
    model: large-v3-turbo
    compute_type: float16
    chunk_duration_s: 5.0
    stride_s: 4.5
    overlap_s: 0.5
    prebuffer_s: 0.3        # GPU inference is fast
    beam_size: 1
    vad_threshold: 0.5
```

## MLOps Changes

### Config Unification
- `.env` (committed): localhost defaults
- `.env.local` (gitignored): MacBook overrides
- `.env.production` (on thomas-pc): production overrides
- TranslationConfig migrated to pydantic-settings (reads .env)

### Observability
- Prometheus metrics: transcription_duration_seconds, audio_queue_depth, backend_circuit_state
- Structured log enrichment: model_id + registry_version on every TranscriptionResult
- Benchmark JSONL: results/transcription_index.jsonl

### Circuit Breaker
- `fallback_chain: [mlx, whisper]` in BackendConfig
- 3 consecutive failures → circuit opens → failover
- Recovery probe every 120s

## Agent Sources

Compiled from 4 specialist agent reviews:
- **media-streaming**: overlap tuning, production system comparison, VAD gating
- **ml-engineer**: Whisper accuracy tuning, hallucination prevention, StabilityTracker
- **mlops-engineer**: config unification, experiment tracking, circuit breaker
- **qa-expert**: benchmark framework design (pending)
