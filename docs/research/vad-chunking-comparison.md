# VAD / Chunking Strategy Comparison: WhisperX, faster-whisper, and FasterWhisperX

**Date:** 2026-03-14
**Purpose:** Inform pluggable transcription backend design for `modules/transcription-service/`
**Author:** Research Analyst (automated synthesis)
**Status:** Final — ready for engineering review

---

## Executive Summary

Three approaches to VAD-driven audio segmentation exist in the open-source Whisper ecosystem. Only two are actively maintained production libraries: **WhisperX** and **faster-whisper**. The third name, "FasterWhisperX," is not a distinct maintained project — it is a colloquial label for WhisperX's v3+ architecture, which adopted faster-whisper as its internal ASR engine. The real design decision for LiveTranslate is: which VAD/chunking patterns from these two libraries to adopt for the `BackendConfig`-driven pluggable backend.

Key finding: **faster-whisper's `BatchedInferencePipeline` with Silero VAD represents the best-fit pattern** for LiveTranslate's real-time streaming use case. WhisperX's forced alignment stage (wav2vec2) adds high-quality word timestamps but introduces ~2–4 GB of additional VRAM overhead and sequential alignment that conflicts with real-time latency requirements. WhisperX's pyannote VAD pipeline requires GPU and is incompatible with streaming modes. For LiveTranslate's batch re-transcription profile, WhisperX patterns become more relevant.

---

## 1. WhisperX — VAD, Segmentation, and Batch Inference

### 1.1 VAD Engine: pyannote-audio

WhisperX uses [pyannote-audio](https://github.com/pyannote/pyannote-audio) as its primary VAD model, specifically a bundled pretrained segmentation model stored at `whisperx/assets/pytorch_model.bin` (the `pyannote/segmentation` family). This is a neural segmentation model — not a lightweight threshold filter — trained for speaker diarization and VAD jointly.

**Pipeline steps:**

1. Audio is passed to `VoiceActivitySegmentation`, a pyannote pipeline wrapping the segmentation model.
2. The model outputs raw speech probabilities at fine temporal resolution.
3. A **Binarize** post-processor applies hysteresis thresholding:
   - Speech starts when probability >= `vad_onset` (default: **0.500**)
   - Speech ends when probability < `vad_offset` (default: **0.363**)
   - The asymmetric thresholds prevent rapid flickering at speech boundaries.
4. Minimum duration filters remove spurious detections:
   - `min_duration_on`: 0.1 s (minimum speech segment)
   - `min_duration_off`: 0.1 s (minimum silence)
5. `pad_onset` and `pad_offset` extend segment boundaries for acoustic context.
6. Adjacent segments with gaps <= 0.5 s are merged.
7. Segments exceeding `max_duration` are split at the lowest-probability internal timestamp.

**Output:** A list of `(start, end)` speech segments, passed to the chunk merger.

### 1.2 Chunk Merging

After VAD produces speech segments, `merge_chunks()` consolidates them into inference-ready windows:

- **`chunk_size`** (default: **30 s**): Maximum duration of a merged chunk fed to Whisper. Adjacent VAD segments are greedily merged until the next segment would push the window beyond `chunk_size`, at which point a new chunk begins.
- The result is a list of chunks, each containing one or more VAD segments, bounded at 30 s to match Whisper's native context window.
- There is **no explicit stride or overlap** between chunks in WhisperX — chunk boundaries fall at VAD-detected silence, not at fixed intervals.

### 1.3 Forced Alignment: wav2vec2

After ASR transcription produces utterance-level text, WhisperX runs a second model pass for word-level timestamps:

1. `load_align_model(language_code)` loads a language-specific wav2vec2 model (from torchaudio or Hugging Face). Supported natively: English, French, German, Spanish, Italian (via torchaudio bundles). Extended to 37 languages via HuggingFace CTC models.
2. The alignment model extracts CTC emission probabilities from the audio.
3. A trellis search (dynamic programming) aligns the transcript phonemes to the emission sequence.
4. Back-tracing produces character-level and word-level timestamps.

**Memory overhead:** The large wav2vec2 bundle (`WAV2VEC2_ASR_LARGE_LV60K_960H`) uses approximately **1.2 GB VRAM**; the base bundle uses ~360 MB. Models are loaded on-demand per language and explicitly unloaded after use. Alignment runs sequentially per segment — it is not batched.

**Quality:** Word-level timestamps are accurate to ~50–100 ms. This is substantially better than Whisper's native token-level timestamps (which can drift 0.5–2 s on long segments).

### 1.4 Batch Inference Strategy

WhisperX uses **faster-whisper as its ASR backend** (since v3.0). The batch inference layer:

- Uses `WhisperModel.generate_segment_batched()` internally.
- Processes all VAD-derived chunks through the encoder simultaneously via a DataLoader.
- Features stacked using a custom collate function, all encoded in a single forward pass.
- Decoding runs with `without_timestamps=True` (required for batched mode — Whisper cannot emit per-token timestamps in batch mode).
- **`batch_size`** parameter (recommended: 4–16 depending on VRAM; API default: 16).
- Achieves **60–70x real-time** speed on large-v2 with < 8 GB VRAM.

### 1.5 Streaming Support

WhisperX has **no streaming mode**. It processes complete audio files. The pyannote VAD model requires GPU and operates on the full signal. The forced alignment stage also requires the complete transcript before running. Attempting to use WhisperX for live streaming requires an external chunking layer (e.g., whisper-streaming wrapping it) and forfeits word-timestamp accuracy in the interim results.

### 1.6 Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_size` | 30 s | Maximum merged chunk fed to ASR |
| `vad_onset` | 0.500 | pyannote speech start threshold |
| `vad_offset` | 0.363 | pyannote speech end threshold |
| `batch_size` | 16 | Chunks processed per forward pass |
| `min_duration_on` | 0.1 s | Minimum speech segment to keep |
| `min_duration_off` | 0.1 s | Minimum silence between segments |

---

## 2. faster-whisper — VAD, Chunking, and BatchedInferencePipeline

### 2.1 VAD Engine: Silero VAD

faster-whisper integrates [Silero VAD v6](https://github.com/snakers4/silero-vad), a lightweight ONNX model for voice activity detection. Unlike pyannote, Silero VAD:

- Runs on **CPU only** via ONNX Runtime (single-threaded, deterministic)
- Processes audio in **512-sample windows** (32 ms at 16 kHz)
- Maintains a **1×1×128 float32 LSTM hidden state** for temporal context across windows
- Requires no authentication tokens, no GPU, and loads in milliseconds
- Model file size: ~2 MB (vs. pyannote segmentation model: ~70–100 MB)

**VAD parameters (`VadOptions` dataclass):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Speech probability threshold; above = speech start |
| `neg_threshold` | 0.35 (threshold - 0.15) | Silence threshold; below = speech end |
| `min_speech_duration_ms` | 0 | Minimum speech segment duration |
| `max_speech_duration_s` | inf | Maximum segment duration before forced split |
| `min_silence_duration_ms` | 2000 | Required silence to finalize a speech segment |
| `speech_pad_ms` | 400 | Padding added before/after each segment |
| `min_silence_at_max_speech` | 98 ms | Minimum silence required when splitting long segments |
| `use_max_poss_sil_at_max_speech` | True | Split at longest silence (True) vs. last silence (False) |

The dual-threshold hysteresis matches pyannote's onset/offset design but is computed over 32 ms frames rather than the pyannote model's finer resolution neural outputs.

### 2.2 `vad_filter` vs. `BatchedInferencePipeline`

faster-whisper exposes VAD through two interfaces:

**`WhisperModel.transcribe()` with `vad_filter=True`:**
- VAD is **optional** (default: off)
- When enabled, removes silent regions before passing audio to Whisper
- Supports `condition_on_previous_text=True` for context continuity across segments
- Processes segments sequentially — one segment at a time through the decoder
- Supports temperature fallback and quality thresholds (no-speech probability, compression ratio)
- Returns fine-grained segments (~1–2 s each, bounded by detected speech)

**`BatchedInferencePipeline.transcribe()`:**
- VAD is **always enabled** (cannot be disabled)
- Segments speech regions and merges them into batches up to `chunk_length` seconds
- Processes multiple 30-second segments simultaneously (default `batch_size=8`)
- Does **not** support `condition_on_previous_text` — each chunk is independent
- Does **not** apply temperature fallback or quality thresholds
- Returns larger segments (~30 s each in typical use)
- Achieves **3–5x throughput** improvement over sequential processing
- Word timestamps are supported (`word_timestamps=True` parameter)

### 2.3 Chunking and Collection

The `collect_chunks()` function consolidates VAD speech segments into inference windows:

- Speech segments shorter than a threshold are merged with adjacent segments.
- The merger respects `max_duration=chunk_length`, splitting at VAD-detected silence boundaries when the accumulated duration approaches the limit.
- A **100 ms silence safety margin** is maintained on either side of chunks to prevent windowing artifacts at audio boundaries (the "two VAD" overlap approach from MobiusML's batched whisper implementation).
- Merging follows WhisperX's min-cut strategy: keeps chunks as close to 30 s as possible while respecting natural voicing boundaries.

**`chunk_length` parameter:** Overrides the default 30 s window for the FeatureExtractor. Reducing it (e.g., to 15 s) lowers latency but reduces context for Whisper's decoder, increasing WER on long sentences. Increasing beyond 30 s is not supported (Whisper's positional encodings cover exactly 30 s).

### 2.4 Condition on Previous Text

`condition_on_previous_text` is available only in `WhisperModel.transcribe()`. It carries the previous segment's transcript as a decoder prompt, improving coherence across segment boundaries — particularly useful for:

- Technical terminology and proper nouns (model "learns" them from context)
- Languages with complex grammatical dependencies
- Audio with frequent background noise causing segmentation artifacts

This feature is **incompatible** with `BatchedInferencePipeline` because batched processing treats each chunk as independent (parallel encoding requires uniform prompt structure).

### 2.5 Streaming Mode

faster-whisper has **no native streaming mode**. Both interfaces return Python generators of `Segment` objects, which enables streaming-style consumption but does not reduce time-to-first-token — the audio must be VAD-processed before any segment is emitted.

For true real-time streaming, [whisper-streaming](https://github.com/ufal/whisper_streaming) wraps faster-whisper with a `LocalAgreement-2` policy:

- Processes audio in successive chunks (configurable `--min-chunk-size`, default behavior)
- Emits transcript text only when 2 consecutive inference passes agree on a common prefix
- Achieves **3.3 s average latency** on unsegmented long-form speech (NVIDIA A40)
- Buffer trimming at sentence boundaries to prevent unbounded memory growth
- The `FasterWhisperASR` wrapper class makes the integration clean and pluggable

### 2.6 Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `vad_filter` | False (WhisperModel) | Enable/disable VAD filtering |
| `vad_parameters` | VadOptions() | Full VAD configuration |
| `chunk_length` | 30 s | Maximum chunk size for batch collection |
| `batch_size` | 8 | Parallel segments in BatchedInferencePipeline |
| `condition_on_previous_text` | True (WhisperModel) | Context continuity across segments |
| `word_timestamps` | False | Enable word-level timestamps |
| `beam_size` | 5 | Beam search width (WhisperModel only) |
| `temperature` | (0.0, 0.2, 0.4, ...) | Temperature fallback schedule (WhisperModel only) |

---

## 3. "FasterWhisperX" — Status Assessment

**"FasterWhisperX" is not a distinct maintained project.**

Research across GitHub, PyPI, and academic sources confirms:

1. **WhisperX v3+ already uses faster-whisper internally.** The batch inference pipeline in WhisperX calls `generate_segment_batched()` from faster-whisper's `WhisperModel`. WhisperX is effectively "WhisperX + faster-whisper" already.

2. **Several community forks carry "FasterWhisperX" in their names** (e.g., `whisperX-silero`, `whisperX-FastAPI`) but these are thin wrappers or deployment scaffolds, not distinct algorithmic approaches.

3. **The `whisperX-silero` forks** (by `cnbeining`, `lukaszliniewicz`, `3manifold`) represent the closest thing to a distinct approach: they replace WhisperX's pyannote VAD with Silero VAD to eliminate the GPU requirement and Hugging Face token authentication. These are unmaintained community patches as of early 2026, not a published package.

4. **The MobiusML "batched faster-whisper" blog** describes the algorithmic basis for `BatchedInferencePipeline` — this is now merged into the official faster-whisper repository (not a separate project).

**Conclusion for LiveTranslate:** Do not treat "FasterWhisperX" as a library dependency. The design decision is which patterns from WhisperX and faster-whisper to adopt in the `BackendConfig`-driven backend.

---

## 4. Comparison Matrix

| Dimension | WhisperX | faster-whisper (sequential) | faster-whisper (BatchedInferencePipeline) | whisper-streaming (on faster-whisper) |
|-----------|----------|-----------------------------|-------------------------------------------|---------------------------------------|
| **VAD engine** | pyannote/segmentation (neural, GPU) | Silero VAD v6 (ONNX, CPU) | Silero VAD v6 (ONNX, CPU, always on) | Silero VAD v6 (optional, --vad flag) |
| **VAD model size** | ~70–100 MB | ~2 MB | ~2 MB | ~2 MB |
| **VAD compute** | GPU required | CPU only | CPU only | CPU only |
| **VAD onset/offset** | 0.500 / 0.363 (configurable) | 0.5 / 0.35 (configurable) | 0.5 / 0.35 (configurable) | 0.5 / 0.35 (configurable) |
| **Chunking strategy** | VAD merge → max 30 s (chunk_size) | VAD segments, sequential | VAD merge → max 30 s (chunk_length) | Sliding window, min-chunk-size param |
| **Stride / overlap** | None (VAD boundaries only) | None (VAD boundaries only) | None (VAD boundaries only) | Fixed stride via min-chunk-size |
| **Context continuity** | No (batch mode) | Yes (condition_on_previous_text) | No | Yes (LocalAgreement-2 policy) |
| **Batch inference** | Yes (60–70x RT, batch_size=16) | No (sequential) | Yes (3–5x speedup, batch_size=8) | No (sequential per chunk) |
| **Word timestamps** | Yes (wav2vec2 forced alignment) | Yes (native Whisper token alignment) | Yes (as of mid-2024 fix) | No (segment-level only) |
| **Word timestamp quality** | High (~50–100 ms accuracy) | Medium (~200–500 ms accuracy) | Medium (~200–500 ms accuracy) | N/A |
| **Streaming / real-time** | No | No (generator only) | No (generator only) | Yes (3.3 s avg latency) |
| **VRAM overhead (beyond ASR)** | +1.2 GB (wav2vec2-large) or +360 MB (base) + pyannote | 0 | 0 | 0 |
| **Auth token required** | Yes (HuggingFace, pyannote) | No | No | No |
| **WER (YouTube public)** | ~12–14% (large-v2) | ~13–15% | ~13.1% | ~13–15% |
| **Throughput (large-v2, RTX)** | ~60–70x RT | ~20x RT | ~64–104x RT | ~17–20x RT |
| **Temperature fallback** | No (batch mode) | Yes | No | No |
| **Speaker diarization** | Optional (pyannote) | No | No | No |
| **Production maturity** | High (active, v3+) | High (active) | High (active, 2024+) | Medium (research project) |

---

## 5. Pattern Recommendations for `BackendConfig`

### 5.1 VAD Strategy: Use Silero VAD (faster-whisper model)

**Adopt:** faster-whisper's Silero VAD integration, not pyannote.

Rationale:
- pyannote requires GPU, HuggingFace authentication, and ~70–100 MB model weight download with token-gated access — all create operational friction for a headless production service.
- Silero VAD loads in < 50 ms, runs on CPU, needs no authentication, and its 2 MB ONNX model is trivially bundleable.
- For VAD accuracy: pyannote outperforms on precision (fewer false positives); Silero outperforms on recall (fewer missed speech regions). For transcription use cases, recall is more critical — missing speech is worse than processing a brief silence.
- Silero's CPU execution is complementary to GPU-heavy transcription: VAD runs in parallel on CPU while the GPU is busy with ASR inference, adding zero wall-clock latency.

**Mapping to `BackendConfig.vad_threshold`:** This field maps directly to Silero's `threshold` parameter. The `neg_threshold` is derived automatically as `vad_threshold - 0.15` (matching faster-whisper's default relationship).

### 5.2 Chunking Strategy: VAD-Bounded with `chunk_duration_s` Ceiling

**Adopt:** WhisperX's merge-then-cap strategy, not fixed-interval chunking.

The optimal chunk collection algorithm:
1. Run Silero VAD to identify speech segments.
2. Greedily merge adjacent speech segments into a growing window.
3. Emit the window when: (a) adding the next segment would exceed `chunk_duration_s`, or (b) a silence gap exceeds `min_silence_duration_ms`.
4. Prepend `prebuffer_s` audio before each chunk for acoustic context (the "pre-roll").
5. Add 100 ms padding on each side of the merged chunk to avoid windowing artifacts.

**`chunk_duration_s`** acts as the ceiling on the Whisper context window. The recommended range:
- Real-time mode (`batch_profile: realtime`): 5–10 s — shorter windows reduce latency but increase WER on long sentences.
- Batch mode (`batch_profile: batch`): 25–30 s — maximizes Whisper's context, matching WhisperX behavior.

**`overlap_s` and `stride_s`:** In VAD-bounded chunking, there is no sliding-window overlap in the traditional sense — the VAD boundaries are natural break points. The `overlap_s` / `stride_s` relationship in `BackendConfig` is most relevant when the backend operates in **fixed-stride** mode (e.g., whisper-streaming-style processing where audio arrives continuously and a sliding window advances regardless of VAD). For the `BatchedInferencePipeline` path, set `overlap_s=0` and `stride_s=chunk_duration_s` to reflect the VAD-bounded behavior. For whisper-streaming-style real-time mode, `overlap_s` represents the amount of audio re-fed as context into the next inference window.

### 5.3 Batch Profile: Two Distinct Operating Modes

**`batch_profile: "realtime"`** maps to:
- `WhisperModel.transcribe()` with `vad_filter=True`
- Sequential segments, small chunks (5–10 s), `condition_on_previous_text=True`
- Temperature fallback enabled (quality guard)
- Produces fine-grained segments (1–3 s each)
- Lower throughput, lower latency, higher coherence across segment boundaries
- Target use: live loopback transcription, meeting bot captions

**`batch_profile: "batch"`** maps to:
- `BatchedInferencePipeline.transcribe()`
- Large chunks (25–30 s), parallel batch processing, `batch_size` driven by VRAM
- No context conditioning between chunks, no temperature fallback
- Produces coarse segments (~30 s each)
- Higher throughput (3–5x over sequential), slightly higher WER on noisy audio
- Target use: post-meeting re-transcription of full recordings

### 5.4 Word Timestamps

**Do not adopt WhisperX's forced alignment** for real-time mode. Reasons:
- wav2vec2 alignment runs sequentially per segment, adding 200–800 ms per chunk (language-dependent)
- Requires loading an additional 360 MB–1.2 GB model per language
- Incompatible with streaming — requires complete transcript text before aligning

**Adopt for batch mode optionally:** For post-meeting transcript refinement, word-level timestamps from forced alignment are valuable. This should be a separate post-processing step triggered after batch transcription completes, not part of the real-time `TranscriptionBackend.transcribe()` path.

**For real-time:** faster-whisper's native `word_timestamps=True` provides adequate segment-level precision (~200–500 ms). This is sufficient for speaker attribution and subtitle display.

### 5.5 `beam_size` Guidance

- Real-time mode: `beam_size=1` (greedy, ~30% faster per segment) or `beam_size=3` (minimal quality gain over greedy for clean audio)
- Batch mode: `beam_size=5` (WhisperX default, maximizes accuracy)

`beam_size` has no effect in `BatchedInferencePipeline` — it uses only the first temperature value (greedy-equivalent).

### 5.6 `prebuffer_s` Guidance

`prebuffer_s` defines the minimum audio accumulated before triggering the first inference call on a new session. Recommended values:
- Real-time mode: 0.3–0.5 s (balances first-word latency against false start risk)
- Batch mode: 0.0 s (full audio already available)

This maps directly to the pre-roll buffer in the VAD chunk collector — audio is held in the buffer until `prebuffer_s` of speech has accumulated.

---

## 6. Impact on `BackendConfig` — Field Analysis

The current `BackendConfig` definition in `modules/shared/src/livetranslate_common/models/registry.py`:

```python
class BackendConfig(BaseModel):
    backend: str
    model: str
    compute_type: str
    chunk_duration_s: float = Field(gt=0)
    stride_s: float = Field(gt=0)
    overlap_s: float = Field(ge=0)
    vad_threshold: float = Field(ge=0.0, le=1.0)
    beam_size: int = Field(ge=1)
    prebuffer_s: float = Field(ge=0)
    batch_profile: Literal["realtime", "batch"] = "realtime"
```

### 6.1 Existing Fields — Verdict

| Field | Status | Notes |
|-------|--------|-------|
| `backend` | Sufficient | "faster-whisper", "sensevoice", "funasr" |
| `model` | Sufficient | "large-v3-turbo", "SenseVoiceSmall", etc. |
| `compute_type` | Sufficient | "float16", "int8", "int8_float16" |
| `chunk_duration_s` | Sufficient | Maps to chunk_length / chunk_size ceiling |
| `stride_s` | Sufficient for sliding-window mode | In VAD-bounded mode, set = chunk_duration_s |
| `overlap_s` | Sufficient | In VAD-bounded mode, set = 0 |
| `vad_threshold` | Sufficient | Maps to Silero `threshold`; neg_threshold derived |
| `beam_size` | Sufficient | Ignored in batch profile (greedy) |
| `prebuffer_s` | Sufficient | Pre-roll before first inference trigger |
| `batch_profile` | Sufficient | Controls sequential vs. BatchedInferencePipeline |

### 6.2 Candidate New Fields

The following fields are not strictly required for the initial implementation but should be considered for a v2 `BackendConfig`:

| Field | Type | Default | Rationale |
|-------|------|---------|-----------|
| `vad_min_silence_ms` | int | 2000 | Silero `min_silence_duration_ms` — controls how long a silence must be before closing a speech segment. Critical for real-time mode (2000 ms is too long; 500–800 ms is more appropriate for live captions). |
| `vad_speech_pad_ms` | int | 400 | Silero `speech_pad_ms` — padding added around each speech segment. Reducing to 100–200 ms lowers latency in real-time mode. |
| `batch_size` | int | 8 | `BatchedInferencePipeline` batch size. Currently hardcoded in the backend adapter; exposing it enables VRAM-budget-aware tuning. |
| `condition_on_previous_text` | bool | True | Only meaningful in `batch_profile: realtime` / sequential mode. Improves coherence for technical content. |
| `word_timestamps` | bool | False | Enable native Whisper word timestamps (faster-whisper). For display/export pipelines that need per-word timing. |

### 6.3 Recommended Additions for v1.1

Prioritized by impact:

**High priority — add to `BackendConfig`:**

```python
vad_min_silence_ms: int = Field(default=2000, ge=100, le=10000)
```

The 2000 ms default is too conservative for live captions. In real-time mode, a speaker pausing for 2 seconds before a new speech segment is emitted creates perceived lag. A value of 500–800 ms is appropriate for meeting transcription. This cannot be derived from existing fields and has direct, measurable impact on end-user latency.

**Medium priority — add to `BackendConfig`:**

```python
batch_size: int = Field(default=8, ge=1, le=64)
```

Currently the batch size is implicit. Exposing it allows the `BackendManager` to tune batch size per backend based on available VRAM budget (larger VRAM → larger batch → higher throughput).

**Lower priority — defer to post-processing config:**

`word_timestamps` and `condition_on_previous_text` can be expressed as backend adapter kwargs rather than `BackendConfig` fields, since they do not affect chunking/VAD behavior. Move them to a separate `InferenceOptions` dataclass if needed.

---

## 7. Recommended Configuration Presets

These presets translate directly into `BackendConfig` instances for the `ModelRegistry`:

### Preset: Real-Time English (Low Latency)

```yaml
backend: "faster-whisper"
model: "large-v3-turbo"
compute_type: "float16"
chunk_duration_s: 8.0
stride_s: 8.0        # = chunk_duration_s (VAD-bounded, no fixed stride)
overlap_s: 0.0
vad_threshold: 0.5
beam_size: 1          # greedy for speed
prebuffer_s: 0.4
batch_profile: "realtime"
# proposed additions:
vad_min_silence_ms: 600
batch_size: 1         # sequential mode, condition_on_previous_text active
```

### Preset: Real-Time Multilingual

```yaml
backend: "faster-whisper"
model: "large-v3-turbo"
compute_type: "float16"
chunk_duration_s: 10.0
stride_s: 10.0
overlap_s: 0.0
vad_threshold: 0.45   # slightly more sensitive for non-English phonemes
beam_size: 3
prebuffer_s: 0.5
batch_profile: "realtime"
vad_min_silence_ms: 800
batch_size: 1
```

### Preset: Post-Meeting Batch Re-Transcription

```yaml
backend: "faster-whisper"
model: "large-v3"
compute_type: "float16"
chunk_duration_s: 28.0
stride_s: 28.0
overlap_s: 0.0
vad_threshold: 0.5
beam_size: 5
prebuffer_s: 0.0
batch_profile: "batch"
vad_min_silence_ms: 2000
batch_size: 16
```

---

## 8. Summary of Adopted Patterns

| Decision | Adopted From | Rationale |
|----------|-------------|-----------|
| Silero VAD (not pyannote) | faster-whisper | CPU-only, no auth, streaming-compatible, low overhead |
| VAD-bounded chunk merging with ceiling | WhisperX chunk_size pattern | Natural boundaries reduce hallucination vs. fixed stride |
| `chunk_duration_s` as context ceiling | Both (chunk_size / chunk_length) | Maps directly to Whisper's 30 s positional encoding limit |
| Two batch profiles (realtime / batch) | faster-whisper dual interface | WhisperModel for streaming; BatchedInferencePipeline for throughput |
| No forced alignment in real-time path | WhisperX lesson (anti-pattern for RT) | wav2vec2 alignment latency is incompatible with live captions |
| `prebuffer_s` pre-roll | whisper-streaming pattern | Accumulate minimum speech before triggering first inference |
| `vad_min_silence_ms` as new field | Silero VadOptions | 2000 ms default too conservative for live meeting transcription |
| `condition_on_previous_text` in realtime | faster-whisper WhisperModel | Maintains coherence for technical/domain vocabulary |

---

## Sources

- [WhisperX GitHub (m-bain/whisperX)](https://github.com/m-bain/whisperX)
- [WhisperX VAD — DeepWiki](https://deepwiki.com/m-bain/whisperX/4.1-voice-activity-detection)
- [WhisperX ASR — DeepWiki](https://deepwiki.com/m-bain/whisperX/3.2-automatic-speech-recognition)
- [WhisperX Forced Alignment — DeepWiki](https://deepwiki.com/m-bain/whisperX/3.3-forced-alignment-system)
- [faster-whisper GitHub (SYSTRAN/faster-whisper)](https://github.com/SYSTRAN/faster-whisper)
- [faster-whisper VAD — DeepWiki](https://deepwiki.com/SYSTRAN/faster-whisper/5.2-voice-activity-detection)
- [faster-whisper — BatchedInferencePipeline VAD PR #936](https://github.com/SYSTRAN/faster-whisper/pull/936)
- [faster-whisper — Word Timestamps Fix PR #921](https://github.com/SYSTRAN/faster-whisper/pull/921)
- [MobiusML: Speeding up Whisper (ASR) — Batched Faster-Whisper Blog](https://mobiusml.github.io/batched_whisper_blog/)
- [whisper-streaming GitHub (ufal/whisper_streaming)](https://github.com/ufal/whisper_streaming)
- [Turning Whisper into a Real-Time Transcription System (arXiv:2307.14743)](https://arxiv.org/html/2307.14743)
- [Modal: Choosing between Whisper variants](https://modal.com/blog/choosing-whisper-variants)
- [Silero VAD — snakers4/silero-vad Discussions](https://github.com/snakers4/silero-vad/discussions/152)
- [WhisperX-Silero community forks](https://github.com/cnbeining/whisperX-silero)
