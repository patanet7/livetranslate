# Whisper Service Implementation Details

## 1. Complete File Listing

### Core Streaming (4 files)
- `/src/simul_whisper/simul_whisper.py` (799 lines) - Main engine, BROKEN code-switching
- `/src/vac_online_processor.py` (350+ lines) - VAD wrapper, INVERTED logic
- `/src/beam_decoder.py` (~200 lines) - Beam search, WORKING
- `/src/alignatt_decoder.py` (415 lines) - AlignAtt policy, WORKING for latency

### Language Detection (3 files)
- `/src/sliding_lid_detector.py` (210 lines) - Passive window tracking, UI-only
- `/src/text_language_detector.py` (214 lines) - Text-based detection, post-processing
- `/src/simul_whisper/whisper/decoding.py` (~500 lines) - Whisper's detect_language()

### Audio Processing (5 files)
- `/src/vac_online_processor.py` - VAD-aware chunking
- `/src/audio/vad_processor.py` (~200 lines) - Silero VAD wrapper
- `/src/silero_vad_iterator.py` (~300 lines) - Fixed-size iteration
- `/src/token_deduplicator.py` (~200 lines) - Phase 5 repetition handling
- `/src/utf8_boundary_fixer.py` (~150 lines) - Phase 5 UTF-8 cleanup

### Model Management (5 files)
- `/src/models/pytorch_manager.py` (~300 lines) - PyTorch loading (GPU)
- `/src/models/openvino_manager.py` (~250 lines) - OpenVINO NPU loading
- `/src/models/model_factory.py` (~200 lines) - Factory pattern
- `/src/simul_whisper/config.py` (~200 lines) - Configuration
- `/src/simul_whisper/whisper/model.py` (~300 lines) - Model wrapper

### API & Server (6 files)
- `/src/api_server.py` (~1000+ lines) - Flask REST API
- `/src/websocket_stream_server.py` (~500 lines) - WebSocket infrastructure
- `/src/connection_manager.py` (~300 lines) - Connection pooling (1000)
- `/src/session_manager.py` (~300 lines) - Session lifecycle
- `/src/transcript_manager.py` (~300 lines) - Transcript buffering
- `/src/message_router.py` (~200 lines) - Message routing

### Support Infrastructure (8 files)
- `/src/speaker_diarization.py` (~400 lines) - Speaker attribution
- `/src/domain_prompt_manager.py` (~300 lines) - Domain prompting
- `/src/error_handler.py` (~400 lines) - Error management
- `/src/heartbeat_manager.py` (~200 lines) - Connection health
- `/src/sentence_segmenter.py` (~200 lines) - Sentence boundaries
- `/src/segment_timestamper.py` (~200 lines) - Timestamp generation
- `/src/token_buffer.py` (~300 lines) - Token buffering
- `/src/stability_tracker.py` (~200 lines) - Stability metrics

### Tokenization & Decoding (5 files)
- `/src/simul_whisper/whisper/tokenizer.py` (~400 lines) - Token encoding
- `/src/simul_whisper/whisper/decoding.py` (~500 lines) - Decoding logic
- `/src/eow_detection.py` (~200 lines) - End-of-word detection
- `/src/token_deduplicator.py` (~200 lines) - Deduplication
- `/src/generation_progress.py` (~150 lines) - Progress tracking

### Utilities & Helpers (12+ files)
- `/src/audio/audio_utils.py` - Audio conversions
- `/src/utils/audio_errors.py` - Error classes
- `/src/transcription/` - Result parsing and helpers
- etc.

**TOTAL**: 74 Python files

---

## 2. Critical Code Locations

### 2.1 Language Switching Bug #1: Dynamic Detection

**File**: `/src/simul_whisper/simul_whisper.py:482-485`

```python
# BROKEN: Detects language on EVERY chunk when code-switching enabled
should_detect_language = (
    self.cfg.language == "auto" and
    (self.detected_language is None or getattr(self, 'enable_code_switching', False))
)
```

**Reference (how it should be)**:
```python
# CORRECT: Detect ONLY once, then pin language for entire session
if self.cfg.language == "auto" and self.detected_language is None:
    # Language detection runs ONLY when detected_language is None
    language_tokens, language_probs = self.lang_id(encoder_feature)
    # ... detect and pin language
    self.detected_language = top_lan  # ← SET ONCE, NEVER CHANGES
```

**Impact**: Causes cache dimension mismatches on 2nd detection

---

### 2.2 Language Switching Bug #2: SOT Reset

**File**: `/src/simul_whisper/simul_whisper.py:251-271`

```python
def update_language_tokens(self, language):
    """BROKEN: Resets all decoder context on language switch"""
    self.create_tokenizer(language)  # New tokenizer
    # Update initial_tokens with new SOT sequence
    self.initial_tokens = torch.tensor(
        self.tokenizer.sot_sequence_including_notimestamps,
        dtype=torch.long,
        device=self.model.device).unsqueeze(0)
    self.initial_token_length = self.initial_tokens.shape[1]
    # PROBLEM: Reset tokens to initial (loses all previous output!)
    self.tokens = [self.initial_tokens]  # ← RESET!
    # PROBLEM: Clear KV cache (loses all context!)
    self._clean_cache()  # ← CLEARS!
```

**FEEDBACK.md Violation**:
- "Never clear KV mid-utterance"
- "Never swap SOT mid-sequence"
- "Do not change SOT mid-stream"

**Impact**: 
- Loss of 30+ seconds of context
- Token sequence discontinuity
- Decoder generates garbage (96-100% accuracy loss)

---

### 2.3 Language Switching Bug #3: Processing Order

**File**: `/src/vac_online_processor.py:350-372`

```python
def process_iter(self):
    # BROKEN: Check buffer size FIRST (should be second!)
    if self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
        # Process every 1.2s REGARDLESS of VAD status
        return self._process_online_chunk()
    
    # Check VAD silence SECOND (should be first!)
    elif self.is_currently_final:
        return self._finish()
```

**Reference (SimulStreaming)**:
```python
def process_iter(self):
    # CORRECT: Check VAD silence FIRST
    if self.is_currently_final:
        return self.finish()
    elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE*self.online_chunk_size:
        # Check buffer size SECOND (only if no silence detected)
        return self.online.process_iter()
```

**Impact**: 
- Cuts utterances mid-word
- Breaks phonetic units
- Causes duplication

---

### 2.4 Language Switching Bug #4: Newest Segment Detection

**File**: `/src/simul_whisper/simul_whisper.py:467-474`

```python
# BROKEN: Uses different encoder features for language detection
encoder_feature_for_lang_detect = encoder_feature
if getattr(self, 'enable_code_switching', False) and len(self.segments) > 0:
    # Use only the NEWEST segment for language detection
    newest_segment = self.segments[-1]
    mel_newest = log_mel_spectrogram(newest_segment, n_mels=self.model.dims.n_mels,
                                      padding=N_SAMPLES, device=self.model.device).unsqueeze(0)
    mel_newest = pad_or_trim(mel_newest, N_FRAMES)
    encoder_feature_for_lang_detect = self.model.encoder(mel_newest)  # ← Different encoder output!
```

**Problems**:
1. Creates 2 different encoder feature tensors (shape mismatch)
2. Language detection on different features than main decoding
3. Double encoder call (wasted computation)
4. Inconsistent with reference architecture

---

## 3. Current vs. Required Architecture

### Current (Single Decoder)

```python
class PaddedAlignAttWhisper:
    def __init__(self, cfg):
        # Single encoder
        self.model = load_model(...)  # Has encoder + decoder
        
        # Single KV cache
        self.kv_cache = {}  # Lines 88-102
        
        # Single tokenizer (pinned at startup)
        self.create_tokenizer(language)  # Line 157-163
        
        # Single token sequence
        self.tokens = [self.initial_tokens]  # Line 249
```

### Required (Parallel Decoders - MISSING)

```python
class ParallelDecoderWhisper:  # NOT IMPLEMENTED
    def __init__(self, cfg):
        # Single encoder (SHARED)
        self.encoder = load_encoder(...)  # OK
        
        # Per-decoder components (ALL MISSING)
        self.decoder_en = load_decoder(..., language='en')  # ❌
        self.decoder_zh = load_decoder(..., language='zh')  # ❌
        
        # Per-decoder KV caches (NOT IMPLEMENTED)
        self.kv_cache_en = {}  # ❌
        self.kv_cache_zh = {}  # ❌
        
        # Per-decoder tokenizers (NOT IMPLEMENTED)
        self.tokenizer_en = get_tokenizer(language='en')  # ❌
        self.tokenizer_zh = get_tokenizer(language='zh')  # ❌
        
        # LID-based gating (NOT IMPLEMENTED)
        self.lid_model = load_lid_model()  # ❌
        
        # Logit fusion (NOT IMPLEMENTED)
        def fusion_function(logits_en, logits_zh, lid_probs):  # ❌
            pass
```

---

## 4. KV Cache Structure Deep Dive

### Current Implementation

```python
# Line 88-102: Install KV cache hooks
self.kv_cache = {}
def kv_hook(module: torch.nn.Linear, _, net_output: torch.Tensor):
    if module.cache_id not in self.kv_cache or net_output.shape[1] > self.max_text_len:
        self.kv_cache[module.cache_id] = net_output  # First token
    else:
        x = self.kv_cache[module.cache_id]
        # Concatenate with existing KV
        self.kv_cache[module.cache_id] = torch.cat([x, net_output], dim=1).detach()
    return self.kv_cache[module.cache_id]

# Hook installed for every attention head in every decoder layer
for i, b in enumerate(self.model.decoder.blocks):
    b.attn.key.register_forward_hook(kv_hook)      # Self-attention K
    b.attn.value.register_forward_hook(kv_hook)    # Self-attention V
    b.cross_attn.key.register_forward_hook(kv_hook)   # Cross-attention K
    b.cross_attn.value.register_forward_hook(kv_hook) # Cross-attention V
```

### Cache Growth Pattern

```
Token 1:     cache[id] = [1, 1, 1280]
Token 2:     cache[id] = cat([1,1,1280], [1,1,1280]) = [1, 2, 1280]
Token 3:     cache[id] = cat([1,2,1280], [1,1,1280]) = [1, 3, 1280]
...
Token 100:   cache[id] = [1, 100, 1280]  # 100 tokens accumulated
```

### Why It's Language-Conditioned

```
English sequence:
SOT: <|en|> (token 50259)
Self-attention patterns learn:
- "the" → "is" → "a" patterns
- Left-to-right attention flow
- English word-boundary patterns

KV accumulated for English: 750 tokens of English patterns

Switch to Chinese (CURRENT BROKEN BEHAVIOR):
SOT changes to: <|zh|> (token 50260)
But KV cache still has 750 English tokens!
Result: Chinese SOT + English cached patterns = GARBAGE
```

---

## 5. Token Sequence Management

### Current (Single Decoder)

```python
# Line 238-250: Initialize tokens
def init_tokens(self):
    self.initial_tokens = torch.tensor(
        self.tokenizer.sot_sequence_including_notimestamps,
        dtype=torch.long,
        device=self.model.device).unsqueeze(0)
    self.tokens = [self.initial_tokens]  # ← List of token tensors

# Line 591-596: Append tokens after decoding
current_tokens, completed = self.token_decoder.update(current_tokens, logits, sum_logprobs)
new_tokens = torch.tensor([new_hypothesis], dtype=torch.long).repeat_interleave(...)
self.tokens.append(new_tokens)  # ← APPEND (never reset)
```

### Token Sequence Flow

```
Chunk 1:
  self.tokens = [initial_tokens]
  decode → get new_tokens
  self.tokens.append(new_tokens)
  self.tokens = [initial_tokens, chunk1_tokens]

Chunk 2:
  self.tokens = [initial_tokens, chunk1_tokens]
  decode → get new_tokens
  self.tokens.append(new_tokens)
  self.tokens = [initial_tokens, chunk1_tokens, chunk2_tokens]

Chunk 3:
  self.tokens = [initial_tokens, chunk1_tokens, chunk2_tokens]
  decode → continue...
```

### Code-Switching Breaks This

```
Chunk 1 (EN):
  self.tokens = [initial_tokens_EN, chunk1_tokens_EN]

Chunk 2 (EN):
  self.tokens = [initial_tokens_EN, chunk1_tokens_EN, chunk2_tokens_EN]

Chunk 3 (ZH): LANGUAGE SWITCH
  update_language_tokens('zh')
  ❌ self.tokens = [initial_tokens_ZH]  ← RESET!
  ❌ chunk1_tokens_EN and chunk2_tokens_EN DISCARDED
  ❌ Decoder has NO MEMORY of what it said
```

---

## 6. LID Components Inventory

### SlidingLIDDetector (PASSIVE)

```python
class SlidingLIDDetector:
    """Tracks language detections in sliding 0.9s window"""
    
    def add_detection(self, language: str, confidence: float, audio_position: float):
        """Add detection to window"""
        detection = LanguageDetection(language, confidence, timestamp, audio_position)
        self.detections.append(detection)
        self._purge_old_detections()  # Remove old ones
    
    def get_current_language(self) -> Optional[str]:
        """Get majority language in window"""
        language_counts = Counter(d.language for d in self.detections)
        most_common = language_counts.most_common(1)[0]
        return most_common[0]

# PURPOSE: UI/formatting only
# IMPACT: None on decoder
# USED FOR: Display current detected language
```

### TextLanguageDetector (POST-PROCESSING)

```python
class TextLanguageDetector:
    """Character-based language detection"""
    
    def detect(self, text: str, audio_detected_language: str = None) -> str:
        """Detect language from text"""
        # Count CJK characters, Latin, Cyrillic, etc.
        cjk_count = self._count_chars_in_ranges(text, CJK_RANGES)
        latin_count = self._count_latin_chars(text)
        # Return language based on script
        if cjk_count > 0:
            return 'zh'
        elif latin_count > 0:
            return 'en'
        # ...

# PURPOSE: Post-processing text analysis
# IMPACT: None on real-time decoding
# USED FOR: Label completed transcripts with detected language
```

### Whisper's Built-in detect_language()

```python
@torch.no_grad()
def lang_id(self, encoder_features):
    """Language detection from encoder features (Line 397-427)"""
    # Single token forward pass with SOT
    x = torch.tensor([[self.tokenizer.sot]] * n_audio).to(self.model.device)
    logits = self.model.logits(x, encoder_features)[:, 0]
    
    # Mask non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(self.tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    
    # Get language with highest probability
    language_tokens = logits.argmax(dim=-1)
    language_probs = [...]
    
    # CRITICAL: Clear cache after lang_id
    self._clean_cache()  # ← Cache cleared only here
    return language_tokens, language_probs

# CALLED: Line 487 (if should_detect_language)
# FREQUENCY: Once per session (normal) or every chunk (broken code-switching)
# CACHE IMPACT: Clears after use (OK for single detection, WRONG for every chunk)
```

---

## 7. Alignment-Attention Mechanism

### AlignAtt Concept

```python
# From alignatt_decoder.py:40-56
class AlignAttDecoder:
    """
    Uses cross-attention to determine when to emit tokens.
    We guide the decoder to attend to only the first l frames by setting a
    frame threshold offset τ, where l = k - τ
    """
```

### Cross-Attention Hook

```python
# Line 80-86: Install attention hooks
self.dec_attns = []
def layer_hook(module, net_input, net_output):
    # net_output[1]: B*num_head*token_len*audio_len
    t = F.softmax(net_output[1], dim=-1)
    self.dec_attns.append(t.squeeze(0))  # Append attention weights

# Hook on cross-attention of each decoder layer
for b in self.model.decoder.blocks:
    b.cross_attn.register_forward_hook(layer_hook)
```

### Alignment Heads

```python
# Line 104-111: Extract alignment heads
self.align_source = {}
self.num_align_heads = 0
for layer_rank, head_id in self.model.alignment_heads.indices().T:
    layer_rank = layer_rank.item()
    heads = self.align_source.get(layer_rank, [])
    heads.append((self.num_align_heads, head_id.item()))
    self.align_source[layer_rank] = heads
    self.num_align_heads += 1

# Whisper model has specific attention heads trained for alignment
# These heads show WHICH AUDIO FRAMES the decoder is processing
```

### Frame-Level Stop Condition

```python
# From infer() method: Check if decoder caught up
attn_of_alignment_heads = ...  # Get attention from alignment heads
attn_of_alignment_heads = median_filter(attn_of_alignment_heads, 7)  # Smooth
most_attended_frame = torch.argmax(attn_of_alignment_heads[:,-1,:], dim=-1)

# Stop generating if decoder caught up to audio
frame_threshold = 4 if is_last else self.cfg.frame_threshold
if content_mel_len - most_attended_frame <= frame_threshold:
    # Decoder has caught up → stop generating
    # Don't want decoder to hallucinate beyond available audio
    completed = True
```

---

## 8. Hardware Acceleration Paths

### Device Selection (Line 47-55)

```python
if torch.cuda.is_available():
    device = "cuda"
    logger.info(f"[Device] Using CUDA device")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Apple Metal Performance Shaders
    logger.info(f"[Device] Using MPS device")
else:
    device = "cpu"
    logger.info(f"[Device] Using CPU device")
```

### GPU Memory Requirements

**Single Decoder (baseline)**:
- Whisper large-v3: ~3GB
- KV cache (30s @ 50fps): ~2GB
- Batch processing: +1-2GB
- **Total**: ~6-7GB GPU memory

**Dual Decoders (for code-switching)**:
- 2x Whisper large-v3: ~6GB
- 2x KV cache: ~4GB
- Batch processing: +1-2GB
- **Total**: ~11-12GB GPU memory

### Memory Analysis

```python
# Single decoder growth
def estimate_kv_cache_size(num_tokens, hidden_dim=1280, num_layers=32):
    # Each layer has: self_attn.k, self_attn.v, cross_attn.k, cross_attn.v
    # self_attn: [batch, seq_len, hidden_dim] → grows with tokens
    # cross_attn: [batch, audio_frames, hidden_dim] → fixed
    
    self_attn_size = 4 * num_tokens * hidden_dim * num_layers
    cross_attn_size = 2 * 1500 * hidden_dim * num_layers  # 30s @ 50fps
    
    return (self_attn_size + cross_attn_size) / 1e9  # GB

# Example:
# 100 tokens: 0.1GB + 0.2GB = 0.3GB
# 500 tokens: 0.5GB + 0.2GB = 0.7GB
# 1500 tokens (30s): 1.5GB + 0.2GB = 1.7GB per decoder
```

---

## 9. Test Coverage Gaps

### What's Tested (PASSING)

```python
# Single-language tests
tests/smoke/test_jfk_direct.py  # English baseline

# Streaming tests  
tests/integration/test_live_streaming_simulation.py
tests/integration/test_streaming_stability.py

# Quality tests
tests/integration/test_beam_search_integration.py
tests/integration/test_alignatt_integration.py

# Speaker attribution
tests/integration/test_real_speech.py  # Diarization + alignment

# Domain prompting
tests/integration/test_domain_prompts.py

# VAD
tests/unit/test_vad_enhancement.py
tests/integration/test_silero_vad_integration.py
```

### What's NOT Tested or BROKEN

```python
# Code-switching tests (ALL FAILING)
tests/stress/test_extended_code_switching.py  # 0.04% accuracy
tests/integration/test_streaming_code_switching.py  # 0% accuracy
tests/integration/test_orchestration_code_switching.py  # BROKEN

# Multi-decoder tests (MISSING)
# (Would need: parallel decoder setup, per-decoder KV management, etc.)

# Attention masking tests (MISSING)
# (Would need: LID-based masking, cross-language handling)

# Token fusion tests (MISSING)
# (Would need: logit space fusion, hysteresis/dwell logic)
```

---

## 10. Broken Functions Summary

| Function | File | Lines | Problem | Fix |
|----------|------|-------|---------|-----|
| `update_language_tokens()` | simul_whisper.py | 251-271 | Resets context+cache | DELETE |
| `process_iter()` | vac_online_processor.py | 350-372 | VAD check second | REVERT |
| Language detection logic | simul_whisper.py | 482-485 | Every-chunk detection | REVERT |
| Newest segment detection | simul_whisper.py | 467-474 | Double encoder call | REVERT |

---

## 11. Effort Estimation for Parallel Decoders

If someone were to implement parallel decoders (NOT RECOMMENDED):

### Components to Add

1. **Parallel Decoder Setup** (200-300 LOC, 2-3 days)
   - Clone decoder for each language
   - Initialize separate tokenizers
   - Create per-language model instances

2. **Per-Decoder KV Cache** (150-200 LOC, 2-3 days)
   - Separate kv_cache dicts for each language
   - Context switching logic
   - Cache lifecycle management

3. **Cross-Attention Masking** (100-150 LOC, 2-3 days)
   - Build attention masks from LID output
   - Apply masks to cross-attention before softmax
   - Validate mask application

4. **Logit-Space Fusion** (100-150 LOC, 1-2 days)
   - Combine decoder logits with LID prior
   - Implement scoring function: S_l(t) = log p_dec(t) + λ·log p_LID
   - Select winning decoder

5. **Hysteresis & Dwell Logic** (100-150 LOC, 1-2 days)
   - Minimum dwell time (250ms)
   - Hysteresis threshold (0.2 confidence margin)
   - Language stability tracking

6. **LID Integration** (150-200 LOC, 2-3 days)
   - Add frame-level LID model (MMS-LID or XLSR)
   - Smooth LID probabilities (Viterbi or hysteresis)
   - Map to encoder frames

7. **Testing Framework** (300-500 LOC, 3-5 days)
   - Multi-language test audio
   - Per-language metrics (WER/CER)
   - Code-switching boundary tests
   - Stress tests (30+ seconds)

**Total**: 1100-1650 LOC, 13-21 days, HIGH COMPLEXITY, UNPROVEN EFFICACY

---

**Document Complete**: Comprehensive implementation details  
**Use Cases**: Architecture review, implementation planning, debugging reference
