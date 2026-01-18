# Whisper Service Codebase Analysis Report

**Date**: 2025-10-29  
**Codebase Location**: `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/`  
**Total Python Files**: 74  
**Analysis Focus**: Current implementation vs. FEEDBACK.md code-switching requirements

---

## Executive Summary

The whisper service codebase has **FUNDAMENTAL ARCHITECTURAL INCOMPATIBILITY** with true code-switching (intra-sentence language mixing) according to the FEEDBACK.md requirements. The current implementation:

1. **Attempts code-switching via SimulStreaming framework** - which was never designed for this
2. **Made catastrophic changes** that broke the reference architecture
3. **Has implemented passive LID tracking only** - no actual parallel decoding
4. **Does NOT have the required architecture** for multi-language code-switching

**Current Status**: Code-switching is BROKEN (0-20% accuracy as documented in CODE_SWITCHING_ARCHITECTURE_ANALYSIS.md)

---

## 1. Current Streaming Implementation

### 1.1 SimulStreaming Base Architecture

**Location**: `/src/simul_whisper/simul_whisper.py` (799 lines)

**Current Implementation**:
- Single unified encoder + single decoder (no parallel decoders)
- VAD-first processing with fixed 1.2-second chunks
- KV cache accumulation across entire session
- One tokenizer per session (language pinned at start)
- AlignAtt (Alignment-guided Attention) for frame-level streaming policy

**Critical Issues**:

1. **Line 482-485**: Dynamic language detection on EVERY chunk
   ```python
   should_detect_language = (
       self.cfg.language == "auto" and
       (self.detected_language is None or getattr(self, 'enable_code_switching', False))
   )
   ```
   - This violates reference pattern (detect ONCE)
   - Causes cache dimension mismatches
   - Breaks decoder's language conditioning

2. **Line 251-271**: `update_language_tokens()` method (code-switching attempt)
   ```python
   def update_language_tokens(self, language):
       self.create_tokenizer(language)  # New tokenizer
       self.initial_tokens = torch.tensor(...)  # Reset SOT
       self.tokens = [self.initial_tokens]  # ← RESET token sequence!
       self._clean_cache()  # ← CLEAR KV cache
   ```
   - Resets token sequence (loses all previous context)
   - Clears KV cache (loses language-conditioned patterns)
   - Violates FEEDBACK.md "Never clear KV mid-utterance" principle

3. **Line 467-474**: Newest-segment-only language detection
   ```python
   if getattr(self, 'enable_code_switching', False) and len(self.segments) > 0:
       newest_segment = self.segments[-1]
       mel_newest = log_mel_spectrogram(...)
       encoder_feature_for_lang_detect = self.model.encoder(mel_newest)
   ```
   - Attempts to mitigate majority-language bias
   - Still incompatible with SOT-based language conditioning
   - Creates new encoder feature separate from main flow

### 1.2 VAC (Voice Activity Controller) Online Processor

**Location**: `/src/vac_online_processor.py` (350+ lines)

**Current Implementation**:
- Wraps SimulStreaming with Silero VAD
- Detects speech in small 0.04s VAD chunks
- Buffers audio until 1.2s chunk ready
- Processes only when buffer full OR speech ends

**Critical Issues**:

1. **Line 350-372**: Processing order INVERTED from reference
   ```python
   def process_iter(self):
       # Check buffer size FIRST (should be second!)
       if self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
           return self._process_online_chunk()
       
       # Check VAD silence SECOND (should be first!)
       elif self.is_currently_final:
           return self._finish()
   ```
   - Reference: VAD silence check FIRST (prevents mid-utterance cuts)
   - Current: Buffer size check FIRST (cuts utterances mid-word for code-switching detection)
   - **Result**: Phonetic unit breakage, word duplication

2. **Line 138-139**: Audio chunk accumulation pattern differs from reference
   ```python
   self.audio_chunks = []  # List of chunks
   self.current_online_chunk_buffer_size = 0
   ```
   - Attempts to match SimulStreaming's pattern
   - But processing loop logic is inverted

### 1.3 Language Identification Components

**Passive Tracking Only** - Not coupled to decoder:

#### `SlidingLIDDetector` (/src/sliding_lid_detector.py)
- Tracks language detections in sliding 0.9s window
- Returns majority language in window
- **Purpose**: UI/formatting only, NOT decoder control
- **Limitation**: Passive (no effect on transcription)

#### `TextLanguageDetector` (/src/text_language_detector.py)
- Character-script analysis (CJK, Latin, Cyrillic, Arabic, etc.)
- Detect language from OUTPUT text
- **Purpose**: Post-processing language tagging
- **Limitation**: Cannot affect real-time decoding (text isn't available yet)

### 1.4 Architecture Gaps vs. FEEDBACK.md

| Requirement | Status | Gap |
|-------------|--------|-----|
| **Shared encoder** | ✅ Exists | Single encoder (correct) |
| **Multiple decoders (N ≥ 2)** | ❌ MISSING | Only one decoder per session |
| **Per-decoder KV caches** | ❌ MISSING | Single shared KV cache |
| **Cross-attention masking** | ❌ MISSING | No per-language attention masks |
| **LID-gated logit fusion** | ❌ MISSING | No logit fusion mechanism |
| **Per-language logit masks** | ❌ MISSING | No token-level language gating |
| **Hysteresis/dwell logic** | ❌ MISSING | No language switch stability checks |
| **AlignAtt frame thresholding** | ✅ Partial | Only for latency, not language switching |

---

## 2. Key Files and Their Current Implementation

### 2.1 Core Streaming Files

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `simul_whisper.py` | 799 | Main streaming engine | ⚠️ Broken code-switching |
| `vac_online_processor.py` | 350+ | VAD chunking wrapper | ⚠️ Inverted logic |
| `beam_decoder.py` | ~200 | Beam search decoding | ✅ Implemented |
| `alignatt_decoder.py` | 415 | Frame-level streaming policy | ✅ Implemented (latency-focused) |

### 2.2 Model and Configuration Files

| File | Purpose | Status |
|------|---------|--------|
| `simul_whisper/config.py` | Configuration dataclass | ✅ Complete |
| `simul_whisper/whisper/model.py` | Whisper model wrapper | ✅ Complete |
| `simul_whisper/whisper/tokenizer.py` | Token encoding/decoding | ✅ Complete |
| `models/pytorch_manager.py` | PyTorch model loading | ✅ Implemented |
| `models/openvino_manager.py` | OpenVINO NPU support | ✅ Implemented |

### 2.3 Language Detection Files

| File | Purpose | Decoder Impact |
|------|---------|-----------------|
| `sliding_lid_detector.py` | Sliding window language tracking | ❌ None (passive only) |
| `text_language_detector.py` | Text-based language detection | ❌ None (post-processing) |
| `simul_whisper/whisper/decoding.py` | Whisper's built-in `detect_language()` | ⚠️ Called once per session only |

### 2.4 Audio Processing Files

| File | Purpose | Status |
|------|---------|--------|
| `audio/audio_utils.py` | Audio format conversions | ✅ Complete |
| `audio/vad_processor.py` | Silero VAD wrapper | ✅ Complete |
| `silero_vad_iterator.py` | Fixed-size VAD iteration | ✅ Complete |
| `token_deduplicator.py` | Repetition detection | ✅ Implemented (Phase 5) |
| `utf8_boundary_fixer.py` | Multi-byte character cleanup | ✅ Implemented (Phase 5) |

### 2.5 Support Infrastructure

| File | Purpose | Status |
|------|---------|--------|
| `api_server.py` | Flask/WebSocket server | ✅ Production-ready |
| `websocket_stream_server.py` | WebSocket streaming | ✅ Production-ready |
| `connection_manager.py` | Connection pooling (1000) | ✅ Enterprise-grade |
| `session_manager.py` | Session lifecycle | ✅ Complete |
| `transcript_manager.py` | Transcript buffering | ✅ Complete |
| `speaker_diarization.py` | Speaker attribution | ✅ Implemented |
| `domain_prompt_manager.py` | Domain-specific prompting | ✅ Implemented |

---

## 3. Encoder-Decoder Architecture Analysis

### 3.1 Current Single-Decoder Design

```
┌─────────────────────────────────┐
│      Whisper Encoder            │
│  (Mel-spectrogram → features)   │
│  Input: 30 seconds @ 16kHz      │
│  Output: 1500 frames @ 50fps    │
└──────────────┬──────────────────┘
               │ encoder_features
               │ (shape: [1, 1500, 1280])
               ▼
┌─────────────────────────────────┐
│   Single Decoder Instance       │
│ (features + tokens → logits)    │
│  KV Cache: Single (shared)      │
│  Tokenizer: One (pinned)        │
│  SOT Tokens: Fixed (at start)   │
└──────────────┬──────────────────┘
               │ logits
               │ (shape: [1, seq_len, vocab_size])
               ▼
┌─────────────────────────────────┐
│   Token Generation              │
│ (greedy or beam search)         │
└─────────────────────────────────┘
```

**Limitations**:
- Single language path only
- All tokens condition on single SOT
- Cache mixes language patterns
- No per-language gating

### 3.2 What Code-Switching Would Require

```
┌─────────────────────────────────┐
│      Whisper Encoder (SHARED)   │
│  (Mel-spectrogram → features)   │
│  Input: 30 seconds @ 16kHz      │
│  Output: 1500 frames @ 50fps    │
└──────────────┬──────────────────┘
               │ encoder_features
               │ (shape: [1, 1500, 1280])
               ├─────────────────────────────────────┐
               │                                     │
    ┌──────────▼────────────┐        ┌──────────────▼─────────┐
    │ Decoder EN            │        │ Decoder ZH             │
    │ SOT: <|en|>           │        │ SOT: <|zh|>            │
    │ KV Cache: EN-only     │        │ KV Cache: ZH-only      │
    │ Tokenizer: EN         │        │ Tokenizer: ZH          │
    │ Attn Mask: LID-gated  │        │ Attn Mask: LID-gated   │
    └──────────┬────────────┘        └──────────┬─────────────┘
               │ logits_en           │ logits_zh
               │                     │
               └────────────┬────────┘
                           │
        ┌──────────────────▼─────────────────────┐
        │ LID-Weighted Logit Fusion              │
        │ S_l(t) = log p_dec(t) + λ·log p_LID    │
        │ Pick: l* = argmax_l S_l(t)             │
        │ Stability: Hysteresis + dwell time     │
        └──────────────────┬─────────────────────┘
                           │ fused_token
                           ▼
        ┌──────────────────────────────────────────┐
        │ Commit & Output with Language Tag        │
        │ (Only when stable, AlignAtt approves)    │
        └──────────────────────────────────────────┘
```

**Required Components** (ALL MISSING):
1. Two separate decoder instances (one per language)
2. Per-decoder KV cache management
3. Per-decoder tokenizers
4. Cross-attention masking with LID prior
5. Logit-space fusion function
6. Hysteresis and dwell-time rules
7. Token commitment logic

---

## 4. KV Cache Management Analysis

### 4.1 Current Single-Cache Design

**Location**: `/src/simul_whisper/simul_whisper.py:88-102`

```python
self.kv_cache = {}

def kv_hook(module: torch.nn.Linear, _, net_output: torch.Tensor):
    if module.cache_id not in self.kv_cache or net_output.shape[1] > self.max_text_len:
        self.kv_cache[module.cache_id] = net_output  # First token
    else:
        x = self.kv_cache[module.cache_id]
        self.kv_cache[module.cache_id] = torch.cat([x, net_output], dim=1).detach()
    return self.kv_cache[module.cache_id]
```

**Structure**:
- Single `kv_cache` dict per session
- Keys: module.cache_id (one per attention head)
- Values: Accumulated KV tensors [batch, seq_len, dim]
- Growth: Concatenates on each new token

**Problem for Code-Switching**:
- Cache is **language-conditioned** (trained patterns for specific language)
- Clearing mid-utterance = **losing all context**
- Keeping mid-switch = **language mismatch** (English patterns + Chinese SOT)
- No way to "merge" or "translate" between language caches

### 4.2 What Code-Switching Would Need

```python
# Per-decoder KV cache (NOT IMPLEMENTED)
self.kv_cache_en = {}  # English KV only
self.kv_cache_zh = {}  # Chinese KV only

# Per-decoder context switching (NOT IMPLEMENTED)
def switch_to_decoder(language):
    if language == 'en':
        self.active_kv_cache = self.kv_cache_en
        self.active_decoder = self.decoder_en
    elif language == 'zh':
        self.active_kv_cache = self.kv_cache_zh
        self.active_decoder = self.decoder_zh

# Cross-language context preservation (NOT IMPLEMENTED)
# No way to preserve English tokens + context when switching to Chinese
# Solution: Don't clear, but apply attention masks instead
```

---

## 5. Token Management and SOT Analysis

### 5.1 Current Token Sequence Management

**Location**: `/src/simul_whisper/simul_whisper.py:238-250`

```python
def init_tokens(self):
    self.initial_tokens = torch.tensor(
        self.tokenizer.sot_sequence_including_notimestamps,
        dtype=torch.long,
        device=self.model.device).unsqueeze(0)
    self.initial_token_length = self.initial_tokens.shape[1]
    self.tokens = [self.initial_tokens]  # ← List of token tensors
```

**Normal Flow** (single language):
```
Chunk 1: self.tokens = [initial_tokens, chunk1_tokens]
Chunk 2: self.tokens = [initial_tokens, chunk1_tokens, chunk2_tokens]
Chunk 3: self.tokens = [initial_tokens, chunk1_tokens, chunk2_tokens, chunk3_tokens]
```

**Code-Switching Attempt** (Line 251-271):
```
Chunk 1 (EN): self.tokens = [initial_tokens_EN, chunk1_tokens_EN]
Chunk 2 (EN): self.tokens = [initial_tokens_EN, chunk1_tokens_EN, chunk2_tokens_EN]
Chunk 3 (ZH): update_language_tokens() → self.tokens = [initial_tokens_ZH]  ← RESET!
              ❌ Previous tokens DISCARDED
              ❌ Decoder forgets what it said
```

### 5.2 SOT Token Problem

**English SOT**:
```
[50258, 50259, 50359, 50363]
  ↓      ↓      ↓      ↓
  <|startoftranscript|>  (50258)
  <|en|>                 (50259)  ← LANGUAGE TOKEN
  <|transcribe|>         (50359)
  <|notimestamps|>       (50363)
```

**Chinese SOT**:
```
[50258, 50260, 50359, 50363]
  ↓      ↓      ↓      ↓
  <|startoftranscript|>  (50258)
  <|zh|>                 (50260)  ← DIFFERENT LANGUAGE TOKEN
  <|transcribe|>         (50359)
  <|notimestamps|>       (50363)
```

**Training Assumption Violation**:
- Whisper trained with SOT ONCE at sequence start
- Mid-stream SOT changes violate this assumption
- Decoder doesn't know how to handle "language token changed mid-sequence"

---

## 6. Test Infrastructure

### 6.1 Code-Switching Tests (All Failing)

| File | Purpose | Current Status |
|------|---------|-----------------|
| `tests/stress/test_extended_code_switching.py` | 30+ second mixed audio | ❌ 0.04% accuracy |
| `tests/integration/test_streaming_code_switching.py` | Real-time switching | ❌ 0% accuracy |
| `tests/integration/test_orchestration_code_switching.py` | Bot integration | ❌ BROKEN |
| `tests/stress/test_sustained_detection.py` | Language stability | ⚠️ Partial |

### 6.2 Baseline Tests (Passing)

| Test Type | Location | Status |
|-----------|----------|--------|
| Single-language transcription | `tests/smoke/` | ✅ 75-90% accuracy |
| VAD detection | `tests/unit/test_vad_enhancement.py` | ✅ Working |
| Beam search | `tests/integration/test_beam_search_integration.py` | ✅ Working |
| Speaker diarization | `tests/integration/` | ✅ Implemented |
| Domain prompts | `tests/integration/test_domain_prompts.py` | ✅ Working |
| AlignAtt streaming | `tests/integration/test_alignatt_integration.py` | ✅ Working (latency) |

---

## 7. API Endpoints and Server Integration

### 7.1 Main API Server

**Location**: `/src/api_server.py` (~1000+ lines)

**Endpoints**:
- `POST /api/transcribe` - Batch transcription
- `POST /api/transcribe-streaming` - Streaming transcription
- `WS /ws/transcribe` - WebSocket streaming
- `GET /api/models` - List available models
- `POST /api/configure` - Session configuration
- `GET /api/status` - Server health

**Code-Switching Support**:
- ❌ No configuration option for code-switching
- ❌ No endpoint to switch languages mid-stream
- ✅ Session support (could add logic per-session)

### 7.2 WebSocket Infrastructure

**Location**: `/src/websocket_stream_server.py`

**Features** (All Production-Ready):
- Connection pooling (1000 capacity)
- 20+ error categories
- Heartbeat monitoring
- 30-minute session timeout
- Message buffering (zero-message-loss design)
- Pub-sub routing

**Code-Switching Support**:
- ❌ No per-message language tagging
- ❌ No decoder switching signals
- ✅ Infrastructure to support it (would need API changes)

---

## 8. Hardware Acceleration Status

### 8.1 Implemented

| Hardware | Framework | Status | Location |
|----------|-----------|--------|----------|
| NVIDIA GPU | PyTorch/CUDA | ✅ Primary | `models/pytorch_manager.py` |
| Intel NPU | OpenVINO | ✅ Fallback | `models/openvino_manager.py` |
| Apple MPS | PyTorch | ✅ Fallback | `simul_whisper.py:47-55` |
| CPU | PyTorch | ✅ Fallback | Automatic |

### 8.2 Device Selection Priority

```python
# From simul_whisper.py:47-55
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
else:
    device = "cpu"
```

**Code-Switching Impact**:
- Dual decoders = 2x memory per device
- GPU memory: Need to profile (likely 20-30GB for 2 decoders)
- CPU fallback: Doable but slower
- NPU: May not fit 2 decoders

---

## 9. Critical Gaps Summary

### 9.1 Architecture Gaps

| Component | Required | Current | Gap |
|-----------|----------|---------|-----|
| Encoder | Shared (1) | ✅ 1 | ✅ Complete |
| Decoders | Multiple (2+) | ❌ 1 | ❌ MISSING |
| KV Caches | Per-decoder | ❌ Shared | ❌ MISSING |
| Tokenizers | Per-decoder | ❌ Single | ❌ MISSING |
| Attention Masks | Per-language | ❌ None | ❌ MISSING |
| LID Stream | Parallel | ❌ Passive tracking | ⚠️ Partial |
| Logit Fusion | Weighted | ❌ None | ❌ MISSING |
| Hysteresis Logic | Dwell-time | ❌ None | ❌ MISSING |

### 9.2 Implementation Gaps

| Task | Lines of Code | Effort | Estimated Days |
|------|---------------|--------|-----------------|
| Parallel decoder setup | 200-300 | Medium | 2-3 |
| Per-decoder KV cache | 150-200 | Medium | 2-3 |
| Cross-attention masking | 100-150 | Medium | 2-3 |
| Logit-space fusion | 100-150 | Low-Medium | 1-2 |
| Hysteresis + dwell logic | 100-150 | Low | 1-2 |
| LID-based gating | 150-200 | Medium | 2-3 |
| Testing framework | 300-500 | High | 3-5 |
| **TOTAL** | **1100-1650** | **High** | **13-21 days** |

### 9.3 Broken Components

1. **`update_language_tokens()`** - Resets context (line 251-271)
2. **Processing order** - VAD check second (line 350-372)
3. **Newest-segment detection** - Causes encoder mismatch (line 467-474)
4. **Dynamic language detection** - Every-chunk detection (line 482-485)

---

## 10. Test Results Summary

### 10.1 Documented Failures (from CODE_SWITCHING_ARCHITECTURE_ANALYSIS.md)

| Test | Approach | CER (Chinese) | WER (English) | Overall | Verdict |
|------|----------|---------------|---------------|---------|---------|
| SOT Fix | Update SOT + clear cache | 215% | 77% | 0.00% | ❌ FAILED |
| Context Preserved | SOT without clearing | 60% | N/A | 19.81% | ❌ FAILED |
| Multilingual Tokenizer | No SOT updates | N/A | 68% | 15.91% | ❌ FAILED |
| SimulStream Cache | Full cache clearing | 122% | 77% | 0.04% | ❌ FAILED |

**All 4 attempts** resulted in **96-100% accuracy loss** compared to Phase 1 baseline (75-90%).

### 10.2 What's Working

- Single-language transcription: 75-90% accuracy ✅
- VAD detection: Reliable speech/silence discrimination ✅
- Beam search: +20-30% quality over greedy ✅
- Speaker diarization: Working with timestamps ✅
- Domain prompts: Effective for domain-specific bias ✅
- AlignAtt streaming: -30-50% latency improvement ✅

---

## 11. Comparison: Current vs. FEEDBACK.md Requirements

### 11.1 Parallel Decoder Architecture

**FEEDBACK.md Requirement**:
```python
enc_out = encoder(audio_chunk, cache=enc_cache)  # Single shared encoder
lid_probs = lid_model(audio_chunk)  # Frame-level LID

# Two decoders with INDEPENDENT KV caches
tokens_en, kvc_en = dec_en.step(enc_out, kv=kvc_en, attn_mask=masks['en'])
tokens_zh, kvc_zh = dec_zh.step(enc_out, kv=kvc_zh, attn_mask=masks['zh'])

# Fuse
y = fuse(tokens_en, tokens_zh, lid_probs, lambda_=0.5)
```

**Current Implementation**:
```python
enc_out = encoder(audio_chunk)  # Single encoder ✅
# NO parallel decoding ❌
# Single decoder only
tokens, kvc = dec.step(enc_out, kv=kvc, attn_mask=None)  # No masking ❌
# No fusion ❌
```

### 11.2 LID Stream Requirements

**FEEDBACK.md**:
- 80-120ms hop (frame-level)
- MMS-LID or XLSR-based
- Smooth with Viterbi or hysteresis
- Map to encoder frames

**Current**:
- `SlidingLIDDetector`: Passive window tracking (no effect on decoder) ❌
- `TextLanguageDetector`: Post-processing only ❌
- No active LID model ❌
- No frame-level smoothing ❌

### 11.3 Commit Policy Requirements

**FEEDBACK.md**:
```
Hold tokens in buffer until:
a) LID stays stable for ≥ 200-300 ms
b) AlignAtt shows look-ahead margin < threshold
c) Token entropy < τ
```

**Current**:
- No token buffer for code-switching ❌
- AlignAtt only for latency, not language stability ❌
- No entropy checks ❌
- No dwell-time logic ❌

---

## 12. Recommendations

### 12.1 Immediate Action (Required)

**REVERT code-switching changes** to restore Phase 1 baseline (75-90% accuracy):

1. **Revert `vac_online_processor.py:350-372`** - Check VAD FIRST
2. **Revert `simul_whisper.py:482-485`** - Detect language ONCE
3. **Delete `update_language_tokens()`** - Line 251-271
4. **Revert newest-segment detection** - Line 467-474
5. **Remove `enable_code_switching` flag** - From all files

**Expected Result**: Restore 75-90% accuracy baseline

### 12.2 Short-Term (1-2 Weeks)

**Option A: Session Restart on Language Switch** (Recommended for production)

- Create `MultiLanguageSessionManager` wrapper
- Detect language change → finish session → start new session
- Works for inter-sentence switching
- Maintains high accuracy (70-85%)
- Simple, stable, low-risk

### 12.3 Long-Term (1-2 Months)

**Option B: Standard Whisper + Sliding Window** (Best for true code-switching)

- Replace SimulStreaming with standard Whisper
- Implement sliding window (10s window, 5s stride)
- No language forcing → native code-switching support
- Accuracy: 60-80% (Whisper's native CS capability)
- Latency: 5-10s (higher than SimulStreaming)

### 12.4 Not Recommended

**Option C: Implement Parallel Decoders** (13-21 days, high complexity)

- Would require rewriting major components
- Needs careful KV cache management
- Attention masking complexity
- Logit fusion algorithm
- Extensive testing framework
- **Risk**: Complex, error-prone, still may not achieve >80% accuracy

**Why not?**:
- Whisper wasn't trained for mid-stream SOT changes
- No training data for intra-sentence code-switching
- Decoder expects single-language context
- Architectural constraints fundamental to transformer design

---

## 13. File Reference Guide

### 13.1 Streaming Core

| Path | Lines | Purpose |
|------|-------|---------|
| `/src/simul_whisper/simul_whisper.py` | 799 | Main streaming engine |
| `/src/vac_online_processor.py` | 350+ | VAD chunking wrapper |
| `/src/simul_whisper/config.py` | ~200 | Configuration |
| `/src/simul_whisper/whisper/model.py` | ~300 | Model wrapper |

### 13.2 Language Detection

| Path | Lines | Purpose |
|------|-------|---------|
| `/src/sliding_lid_detector.py` | 210 | Sliding window LID |
| `/src/text_language_detector.py` | 214 | Text-based LID |
| `/src/simul_whisper/whisper/decoding.py` | ~500 | Whisper decoding + lang detection |

### 13.3 Decoders

| Path | Lines | Purpose |
|------|-------|---------|
| `/src/beam_decoder.py` | ~200 | Beam search |
| `/src/alignatt_decoder.py` | 415 | AlignAtt streaming policy |

### 13.4 Audio Processing

| Path | Lines | Purpose |
|------|-------|---------|
| `/src/vac_online_processor.py` | 350+ | VAD-aware chunking |
| `/src/audio/vad_processor.py` | ~200 | Silero VAD wrapper |
| `/src/silero_vad_iterator.py` | ~300 | Fixed-size VAD iteration |
| `/src/token_deduplicator.py` | ~200 | Phase 5 repetition detection |
| `/src/utf8_boundary_fixer.py` | ~150 | Phase 5 multi-byte cleanup |

### 13.5 API & Server

| Path | Lines | Purpose |
|------|-------|---------|
| `/src/api_server.py` | ~1000+ | Flask/WebSocket server |
| `/src/websocket_stream_server.py` | ~500 | WebSocket infrastructure |
| `/src/connection_manager.py` | ~300 | Connection pooling |
| `/src/session_manager.py` | ~300 | Session lifecycle |

### 13.6 Models

| Path | Lines | Purpose |
|------|-------|---------|
| `/src/models/pytorch_manager.py` | ~300 | PyTorch model loading |
| `/src/models/openvino_manager.py` | ~250 | OpenVINO NPU loading |
| `/src/models/model_factory.py` | ~200 | Model factory pattern |

---

## 14. Conclusion

### 14.1 Current Status

The whisper service has **production-ready single-language streaming** (75-90% accuracy) but **fundamentally broken code-switching support** (0-20% accuracy).

The codebase attempted to implement code-switching in SimulStreaming, which violates the core assumptions of the architecture. The FEEDBACK.md requirements for true code-switching are technically sound but would require:

1. **Complete architecture rewrite** OR
2. **Switch to different ASR model** (not SimulStreaming-based)

### 14.2 What's Working

- Single-language transcription ✅
- Streaming infrastructure ✅
- Hardware acceleration ✅
- Speaker diarization ✅
- Domain prompting ✅
- AlignAtt latency optimization ✅
- Beam search quality ✅

### 14.3 What's Broken

- Code-switching ❌ (0% accuracy)
- Dynamic language detection ❌
- Multi-decoder support ❌ (not implemented)
- LID-gated logit fusion ❌ (not implemented)
- Parallel KV caches ❌ (not implemented)

### 14.4 Path Forward

1. **Immediate** (today): Revert code-switching changes → restore 75-90% baseline
2. **Short-term** (1-2 weeks): Implement session-restart approach → 70-85% on inter-sentence switching
3. **Long-term** (1-2 months): Consider standard Whisper + sliding window → true code-switching (60-80%)

This approach:
- Preserves existing high-quality infrastructure
- Avoids complex architectural changes
- Offers clear migration path
- Maintains production stability

---

**Analysis Completed**: 2025-10-29  
**Total Files Analyzed**: 74 Python files  
**Key Findings**: 12 critical gaps, 4 broken components, 1 viable short-term solution
