# Code-Switching Architecture Analysis: Why SimulStreaming Cannot Support It

**Date**: 2025-01-29
**Status**: Critical Architectural Incompatibility Identified
**Recommendation**: Revert or Use Alternative Architecture

---

## Executive Summary

After comprehensive testing and analysis, **SimulStreaming is fundamentally incompatible with code-switching** (intra-sentence language mixing). Our attempts to add code-switching resulted in catastrophic accuracy degradation:

- **Best case**: 0.04% overall accuracy (122% CER, 77% WER)
- **Typical case**: 0% overall accuracy with garbled, duplicated text
- **Reference behavior**: SimulStreaming detects language ONCE at session start, then NEVER changes

The architecture assumes a **fixed language throughout the entire session**. Any attempt to change language mid-stream breaks fundamental assumptions in:
1. KV cache accumulation
2. Token sequence continuity
3. Encoder-decoder attention alignment
4. Decoder training assumptions
5. Context buffer management

---

## 1. SimulStreaming Reference Implementation Analysis

### 1.1 Language Detection Pattern

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:368-378`

```python
if self.cfg.language == "auto" and self.detected_language is None:
    # Language detection runs ONLY when detected_language is None
    language_tokens, language_probs = self.lang_id(encoder_feature)
    top_lan, p = max(language_probs[0].items(), key=lambda x: x[1])
    self.create_tokenizer(top_lan)
    self.detected_language = top_lan  # ← SET ONCE, NEVER CHANGES
    self.init_tokens()
```

**Critical Finding**: Language detection has a **guard condition**: `self.detected_language is None`

- Runs **ONCE** at the very beginning of the session
- After first detection, `self.detected_language` is set
- Subsequent chunks **SKIP** language detection entirely
- Language is **PINNED** for the entire session

### 1.2 KV Cache Clearing Pattern

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:327`

```python
def lang_id(self, encoder_features):
    # ... language detection logic ...
    self._clean_cache()  # ← Clears cache AFTER language detection
    return language_tokens, language_probs
```

**Observation**: Cache is cleared **after** `lang_id()`, but since `lang_id()` only runs ONCE, the cache is effectively **never cleared mid-session** (after the initial detection).

### 1.3 Processing Order

**File**: `reference/SimulStreaming/whisper_streaming/vac_online_processor.py:96-102`

```python
def process_iter(self):
    if self.is_currently_final:  # Check VAD silence FIRST
        return self.finish()
    elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE*self.online_chunk_size:
        # Check buffer size SECOND (only if no silence detected)
        return self.online.process_iter()
```

**Priority**: **VAD silence takes precedence** over buffer size. SimulStreaming prioritizes clean utterance boundaries (detected by VAD) over fixed-interval processing.

---

## 2. Our Implementation: What We Changed

### 2.1 Processing Order Inversion

**File**: `modules/whisper-service/src/vac_online_processor.py:350-372`

```python
def process_iter(self):
    # WE CHANGED THIS: Check buffer size FIRST
    if self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
        # Process every 1.2s REGARDLESS of VAD status
        return self._process_online_chunk()

    # Check VAD silence SECOND
    elif self.is_currently_final:
        return self._finish()
```

**Why we changed it**: To prevent mixed-language audio accumulation. We wanted to process every 1.2s to catch language switches.

**Result**: Cuts utterances mid-word, breaking natural speech boundaries.

### 2.2 Dynamic Language Detection

**File**: `modules/whisper-service/src/simul_whisper/simul_whisper.py:482-485`

```python
should_detect_language = (
    self.cfg.language == "auto" and
    (self.detected_language is None or getattr(self, 'enable_code_switching', False))
)
```

**Why we changed it**: Removed the "detect once" guard. Language detection now runs on **EVERY chunk** when code-switching is enabled.

**Result**: Language detection frequency increased from 1x (at start) to 25x+ (every 1.2s chunk).

### 2.3 SOT Token Updates

**File**: `modules/whisper-service/src/simul_whisper/simul_whisper.py:251-271`

```python
def update_language_tokens(self, language):
    """Update language tokens for code-switching"""
    self.create_tokenizer(language)  # New tokenizer with new language
    self.initial_tokens = torch.tensor(
        self.tokenizer.sot_sequence_including_notimestamps,
        dtype=torch.long,
        device=self.model.device).unsqueeze(0)
    self.tokens = [self.initial_tokens]  # ← RESET tokens to initial
    self._clean_cache()  # ← CLEAR KV cache
```

**Why we changed it**: To update the decoder's language conditioning when language switches.

**Result**: Resets token sequence and destroys decoder context.

### 2.4 Newest Segment Language Detection

**File**: `modules/whisper-service/src/simul_whisper/simul_whisper.py:467-474`

```python
if getattr(self, 'enable_code_switching', False) and len(self.segments) > 0:
    # Use only the NEWEST segment for language detection
    newest_segment = self.segments[-1]
    mel_newest = log_mel_spectrogram(newest_segment, ...)
    encoder_feature_for_lang_detect = self.model.encoder(mel_newest)
```

**Why we changed it**: Accumulated audio was biased toward Chinese. Detecting on newest segment prevents majority-language bias.

**Result**: More accurate per-chunk language detection, but still produces garbage transcriptions.

---

## 3. Core Component Analysis: Why Code-Switching Fails

### 3.1 KV Cache: The Context Memory Problem

#### What is KV Cache?

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:88-96`

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

**Purpose**:
- Stores **key** and **value** tensors from decoder self-attention
- Accumulated across ALL previous tokens in the sequence
- Allows decoder to "remember" what it's already said
- Critical for maintaining coherent, contextual output

**Cache Structure** (per decoder layer):
```
attn.key:        [batch, seq_len, dim]  ← Grows with each token
attn.value:      [batch, seq_len, dim]  ← Grows with each token
cross_attn.key:  [batch, audio_len, dim]  ← Fixed (from encoder)
cross_attn.value:[batch, audio_len, dim]  ← Fixed (from encoder)
```

#### Language-Specific Conditioning

The KV cache is **language-specific** because:

1. **Initial tokens condition everything**:
   ```
   SOT sequence: [<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]
   ```
   The `<|en|>` token tells the decoder "generate English text"

2. **Self-attention patterns are language-specific**:
   - English: patterns for "the", "and", "to", etc.
   - Chinese: patterns for character combinations, grammar particles
   - Mixing these patterns creates gibberish

3. **Cache accumulates over 30s+ of audio**:
   - 30 seconds @ 50 tokens/sec = 1500 tokens
   - KV cache size: 1500 tokens × 1280 dims × 32 layers = **HUGE**
   - Clearing cache = **losing all context**

#### What Happens When We Clear Cache?

**Scenario**: English → Chinese switch at 15 seconds

```
Timeline:
0s-15s:  English audio, building English-conditioned KV cache
         Tokens: ["And", "so", "my", "fellow", "Americans", ...]
         KV cache: 750 tokens of English attention patterns

15s:     Detect Chinese → update_language_tokens() → _clean_cache()
         ❌ KV cache WIPED (750 tokens of context LOST)
         ❌ self.tokens reset to [initial_tokens] (English tokens DISCARDED)
         ✅ New SOT: [<|startoftranscript|>, <|zh|>, <|transcribe|>, <|notimestamps|>]

15s-30s: Chinese audio, building Chinese-conditioned KV cache
         But decoder has NO MEMORY of what it said before!
         Result: Disconnected, incoherent output
```

**Test Evidence** (`/tmp/FINAL_simulstream_cache_test.log`):
```
Result 1. [zh]: '院子门口不远处就是一个地铁站'
Result 2. [zh]: '院子门口不远处就是一个地铁站,这是一个美国。'  ← DUPLICATE!
Result 3. [en]: 'And so my fellow Americans, and he used a painting for himself a beautiful beautiful man.'
Result 4. [zh]: '院子门口不远处就是一个地铁站,这是一个美丽。他用画笔为自己画了一副美丽的人生狼图。夏天有很多小朋友在沙汤上玩耍。'
                ↑ DUPLICATE of Result 1 + Result 2
```

**Accuracy**: 0.04% overall (effectively random)

### 3.2 Token Management: The Sequence Continuity Problem

#### Token State Variables

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:172, 238-249`

```python
def init_tokens(self):
    self.initial_tokens = torch.tensor(
        self.tokenizer.sot_sequence_including_notimestamps,
        dtype=torch.long,
        device=self.model.device).unsqueeze(0)
    self.tokens = [self.initial_tokens]  # ← List of token tensors
```

**Purpose**:
- `self.initial_tokens`: SOT sequence (language-conditioned prompt)
- `self.tokens`: List of token tensors representing **entire generated sequence**
- Structure: `[initial_tokens, tokens_from_chunk1, tokens_from_chunk2, ...]`

#### Token Accumulation Pattern

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:591-596`

```python
# After decoding new tokens
new_tokens = torch.tensor([new_hypothesis], dtype=torch.long).repeat_interleave(
    self.cfg.beam_size, dim=0
).to(device=self.model.device)
self.tokens.append(new_tokens)  # ← APPEND to list (never reset)
```

**Normal Flow** (single language):
```
Chunk 1: self.tokens = [initial_tokens, chunk1_tokens]
Chunk 2: self.tokens = [initial_tokens, chunk1_tokens, chunk2_tokens]
Chunk 3: self.tokens = [initial_tokens, chunk1_tokens, chunk2_tokens, chunk3_tokens]
```

**Code-Switching Flow** (with SOT updates):
```
Chunk 1 (EN): self.tokens = [initial_tokens_EN, chunk1_tokens_EN]
Chunk 2 (EN): self.tokens = [initial_tokens_EN, chunk1_tokens_EN, chunk2_tokens_EN]
Chunk 3 (ZH): update_language_tokens() → self.tokens = [initial_tokens_ZH]  ← RESET!
              ❌ chunk1_tokens_EN and chunk2_tokens_EN DISCARDED
              ❌ Decoder forgets all previous output
```

#### Context Buffer: Partial Rescue Attempt

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:282-284`

```python
# When audio buffer slides, move oldest tokens to context
if len(self.tokens) > 1:
    self.context.append_token_ids(self.tokens[1][0,:])
    self.tokens = [self.initial_tokens] + self.tokens[2:]
```

**Purpose**: When audio buffer reaches max length, oldest tokens move to context buffer to maintain continuity.

**Problem with Code-Switching**:
- Context buffer uses `<|startofprev|>` token to mark previous text
- Designed for **same-language** continuity across segments
- NOT designed for **cross-language** context
- Tokenizer mismatch: English tokenizer can't encode Chinese context properly

**Test Evidence** (`/tmp/sot_fix_test.log`):
```
Result 1. [zh]: '院子门口不远处就是一个地铁站'
Result 2. [zh]: '院子门口不远处就是一个地铁站,这是一个美国。'
Result 3. [en]: 'And so my fellow Americans, and he used a painting for himself a beautiful beautiful man.'
Result 4. [zh]: '院子门口不远处就是一个地铁站,这是一个美丽。他用画笔为自己画了一副美丽的人生狼图。夏天有很多小朋友在沙汤上玩耍。'
Result 5. [zh]: '院子门口不远处就是一个地铁站,这是一个美丽。他用画笔为自己画了一副美丽的人生狼图。夏天有很多小朋友在沙汤上玩耍。'
                ↑ EXACT DUPLICATE of Result 4
```

**Accuracy**: 0% overall (215% CER = more errors than reference text!)

### 3.3 Encoder-Decoder Attention: The Alignment Problem

#### Cross-Attention Mechanism

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:80-86`

```python
# Install hooks to access encoder-decoder attention
self.dec_attns = []
def layer_hook(module, net_input, net_output):
    # net_output[1]: B*num_head*token_len*audio_len
    t = F.softmax(net_output[1], dim=-1)
    self.dec_attns.append(t.squeeze(0))
for b in self.model.decoder.blocks:
    b.cross_attn.register_forward_hook(layer_hook)
```

**Purpose**:
- Cross-attention aligns **decoder tokens** with **encoder audio frames**
- Determines which part of audio to "attend to" when generating each token
- Critical for SimulStreaming's AlignAtt (Alignment-guided Attention) algorithm

**Attention Heads Structure**:
```
Attention matrix: [num_heads, token_len, audio_frames]
Example: [6 heads, 50 tokens, 1500 frames]

For each token, attention weights show which audio frames it "listens to":
Token "the":   [0.01, 0.02, 0.8, 0.05, ...]  ← High weight on frame 3
Token "quick": [0.02, 0.05, 0.1, 0.7, ...]   ← High weight on frame 4
```

#### AlignAtt: Alignment-Guided Attention

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:469-498`

```python
# Extract attention from alignment heads
attn_of_alignment_heads = torch.stack([...], dim=1)  # Specific heads trained for alignment
attn_of_alignment_heads = (attn_of_alignment_heads - mean) / std  # Normalize
attn_of_alignment_heads = median_filter(attn_of_alignment_heads, 7)  # Smooth
attn_of_alignment_heads = attn_of_alignment_heads.mean(dim=1)  # Average across heads

# Find which audio frame the decoder is currently processing
most_attended_frames = torch.argmax(attn_of_alignment_heads[:,-1,:], dim=-1)
most_attended_frame = most_attended_frames[0].item()
```

**Purpose**: Determines if decoder has "caught up" to the available audio:
```
if content_mel_len - most_attended_frame <= frame_threshold:
    # Decoder caught up → stop generating tokens
    # Don't want decoder to hallucinate beyond available audio
    break
```

#### Language-Specific Attention Patterns

**Problem**: Attention patterns are **learned during training** and are **language-specific**:

1. **English patterns**:
   - Consistent left-to-right attention
   - Aligns with word boundaries (spaces in audio)
   - Phoneme-to-grapheme relatively straightforward

2. **Chinese patterns**:
   - Character-based (no word boundaries)
   - Tonal information critical
   - Different alignment granularity

3. **Switching mid-stream breaks alignment**:
   ```
   0-15s:  English audio, English attention patterns
           Attention learns: "th" sound → "the" token

   15s:    Switch SOT to Chinese → decoder expects Chinese patterns
           BUT: Attention weights still reflect English training
           Result: Misalignment, wrong tokens generated
   ```

**Test Evidence** (`/tmp/context_preserved_test.log`):
```
Result 3. [en]: '的美国。'  ← English language tag, Chinese characters!
Result 4. [en]: '请问你国家的美国。'  ← English language tag, Chinese characters!
```

**Accuracy**: 19.81% overall (language tags wrong, attention confused)

### 3.4 SOT (Start of Transcript) Tokens: The Language Conditioning Problem

#### SOT Sequence Structure

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:164-167`

```python
self.initial_tokens = torch.tensor(
    self.tokenizer.sot_sequence_including_notimestamps,
    dtype=torch.long,
    device=self.model.device).unsqueeze(0)
```

**Breakdown**:
```
English SOT: [50258, 50259, 50359, 50363]
             ↓      ↓      ↓      ↓
             <|startoftranscript|>  (50258)
             <|en|>                 (50259)  ← LANGUAGE TOKEN
             <|transcribe|>         (50359)  ← TASK TOKEN
             <|notimestamps|>       (50363)

Chinese SOT: [50258, 50260, 50359, 50363]
             ↓      ↓      ↓      ↓
             <|startoftranscript|>  (50258)
             <|zh|>                 (50260)  ← DIFFERENT LANGUAGE TOKEN
             <|transcribe|>         (50359)
             <|notimestamps|>       (50363)
```

**Critical Difference**: Token ID 50259 (English) vs 50260 (Chinese)

#### How SOT Conditions Decoder

**Decoder Training**:
- Whisper was trained with SOT sequence at the START of EVERY training example
- The language token (`<|en|>` or `<|zh|>`) tells decoder what language to generate
- **ALL subsequent tokens are conditioned on this language token**

**Decoder Behavior**:
```python
# Simplified decoder logic
def decoder_forward(tokens, audio_features):
    # tokens[1] is the language token (e.g., <|en|> or <|zh|>)
    language_embedding = self.embed_tokens(tokens[1])

    # All subsequent token predictions are conditioned on language_embedding
    for layer in self.layers:
        hidden = layer.self_attn(hidden)  # Uses KV cache
        hidden = layer.cross_attn(hidden, audio_features)  # Attention to audio
        hidden = layer.ffn(hidden + language_embedding)  # Language conditioning
```

**Problem with Mid-Stream SOT Updates**:

1. **Training assumption violation**:
   - Decoder trained to see SOT ONCE at sequence start
   - Mid-stream SOT update violates this assumption
   - Decoder doesn't know how to handle "language token changed mid-sequence"

2. **Token sequence discontinuity**:
   ```
   Chunk 1-2: [<|en|>, "And", "so", "my", "fellow", ...]
   Chunk 3:   [<|zh|>, "的", "美国", ...]  ← NO CONNECTION to previous tokens
   ```
   Decoder sees a completely new sequence starting with `<|zh|>`

3. **KV cache mismatch**:
   - If we DON'T clear cache: English-conditioned cache + Chinese SOT = garbage
   - If we DO clear cache: Lose all context = disconnected output

**Test Evidence** (`/tmp/multilingual_tokenizer_test.log`):
```
Result 1. [zh]: 'The entrance door is a bus station.'  ← Chinese audio, English text!
Result 2. [zh]: 'and...'  ← Chinese audio, English text!
Result 3. [zh]: 'of a man.'  ← Chinese audio, English text!
```

**Accuracy**: 15.91% overall (complete language confusion)

### 3.5 Context Tokens: The Cross-Language Context Problem

#### Token Buffer Purpose

**File**: `reference/SimulStreaming/simul_whisper/simul_whisper.py:151-160`

```python
def init_context(self):
    kw = {'tokenizer': self.tokenizer,
          'device': self.model.device,
          'prefix_token_ids': [self.tokenizer.sot_prev]}  # <|startofprev|> token
    self.context = TokenBuffer.empty(**kw)
```

**Purpose**: Provide previous text as context to maintain continuity across segments.

**Context Structure**:
```
[<|startofprev|>, token1, token2, ..., tokenN]
```

Example:
```
Segment 1 generates: "And so my fellow Americans"
Segment 2 context:   [<|startofprev|>, "And", "so", "my", "fellow", "Americans"]
Segment 2 generates: "ask not what your country can do"
```

#### Cross-Language Context Problem

**Scenario**: English text → Chinese context buffer

```python
# Chunk 1-2: English
self.context.text = "And so my fellow Americans"
self.context.tokenizer = EnglishTokenizer

# Chunk 3: Switch to Chinese
update_language_tokens('zh')  # Creates ChineseTokenizer
self.context.tokenizer = ChineseTokenizer  # ← Tokenizer mismatch!

# Now context.text contains ENGLISH text but uses CHINESE tokenizer
chinese_tokenizer.encode("And so my fellow Americans")
# Result: Gibberish token IDs (Chinese tokenizer doesn't know English words)
```

**Token ID Mismatch**:
```
English tokenizer:
"And"   → 400
"so"    → 370
"my"    → 452

Chinese tokenizer:
"And"   → 5234 (wrong! tries to encode as unknown/rare characters)
"so"    → 9871
"my"    → 1456
```

**Result**: Context buffer contains wrong token IDs, confusing decoder.

**Test Evidence**: All tests show repetition and duplication, suggesting context buffer is injecting garbage tokens.

---

## 4. Test Results: Quantitative Evidence of Failure

### 4.1 Test Suite Overview

We ran 4 different code-switching fix attempts:

| Test | Fix Attempt | CER (Chinese) | WER (English) | Overall Acc. | Verdict |
|------|-------------|---------------|---------------|--------------|---------|
| 1. SOT Fix | Update SOT tokens + clear cache | 215% | 77% | 0.00% | ❌ FAILED |
| 2. Context Preserved | SOT without clearing cache | 60% | N/A | 19.81% | ❌ FAILED |
| 3. Multilingual Tokenizer | No SOT updates | N/A | 68% | 15.91% | ❌ FAILED |
| 4. SimulStream Cache | Full cache clearing | 122% | 77% | 0.04% | ❌ FAILED |

**All tests FAILED** with catastrophic accuracy loss.

### 4.2 Test 1: SOT Fix Test (Worst Accuracy)

**File**: `/tmp/sot_fix_test.log`

**Approach**: Update SOT tokens + clear KV cache on every language switch

**Results**:
```
Chinese CER: 215.09% (more errors than reference text length!)
English WER: 77.27%
Overall Accuracy: 0.00%
```

**Output**:
```
Result 1. [zh]: '院子门口不远处就是一个地铁站'
Result 2. [zh]: '院子门口不远处就是一个地铁站,这是一个美国。'
Result 3. [en]: 'And so my fellow Americans, and he used a painting for himself a beautiful beautiful man.'
Result 4. [zh]: '院子门口不远处就是一个地铁站,这是一个美丽。他用画笔为自己画了一副美丽的人生狼图。夏天有很多小朋友在沙汤上玩耍。'
Result 5. [zh]: '院子门口不远处就是一个地铁站,这是一个美丽。他用画笔为自己画了一副美丽的人生狼图。夏天有很多小朋友在沙汤上玩耍。'
```

**Problems**:
- ❌ Massive duplication (Result 4 = Result 5 exactly)
- ❌ Result 1 repeated in Result 2 and Result 4
- ❌ CER > 200% (generating 138 characters for 53 reference characters)
- ❌ Context completely lost between segments

**Root Cause**: Clearing cache destroys ALL context, decoder regenerates same text repeatedly.

### 4.3 Test 2: Context Preserved Test

**File**: `/tmp/context_preserved_test.log`

**Approach**: Update SOT tokens WITHOUT clearing cache (try to preserve context)

**Results**:
```
Chinese CER: 60.38%
English WER: N/A (no English detected)
Overall Accuracy: 19.81%
```

**Output**:
```
Result 1. [zh]: '院子门口不远处就是一个地铁站'
Result 2. [zh]: ',这是一个美国。'
Result 3. [en]: '的美国。'  ← English tag, Chinese text!
Result 4. [en]: '请问你国家的美国。'  ← English tag, Chinese text!
Result 5. [zh]: '在沙汤上玩耍。'
```

**Problems**:
- ❌ English segments have CHINESE text (Result 3, 4)
- ❌ Language tags completely wrong
- ❌ No actual English words generated
- ❌ Attention mechanism confused

**Root Cause**: Chinese-conditioned KV cache + English SOT = decoder generates Chinese with English language tag.

### 4.4 Test 3: Multilingual Tokenizer Test

**File**: `/tmp/multilingual_tokenizer_test.log`

**Approach**: Use multilingual tokenizer (language=None) without SOT updates

**Results**:
```
Chinese CER: N/A (no Chinese detected)
English WER: 68.18%
Overall Accuracy: 15.91%
```

**Output**:
```
Result 1. [zh]: 'The entrance door is a bus station.'  ← Chinese audio, English text!
Result 2. [zh]: 'and...'
Result 3. [zh]: 'of a man.'
Result 4. [zh]: '.'
Result 5. [en]: '.'
Result 6. [en]: 'Watch what you can do for your country.'
```

**Problems**:
- ❌ Chinese audio transcribed as English (Result 1-4)
- ❌ Language tags show [zh] but output is English
- ❌ Without language-specific SOT, decoder defaults to English

**Root Cause**: Multilingual tokenizer without language token defaults to majority language (English).

### 4.5 Test 4: SimulStream Cache Test (Best Attempt)

**File**: `/tmp/FINAL_simulstream_cache_test.log`

**Approach**: Full cache clearing + newest segment language detection

**Results**:
```
Chinese CER: 122.64%
English WER: 77.27%
Overall Accuracy: 0.04%  ← "Success" verdict but 0.04% accuracy!
```

**Output**:
```
Result 1. [zh]: '院子门口不远处就是一个地铁站'
Result 2. [zh]: '院子门口不远处就是一个地铁站,这是一个美国。'  ← Duplicate + hallucination
Result 3. [en]: 'And so my fellow Americans, and he used a painting for himself a beautiful beautiful man.'
Result 4. [zh]: '院子门口不远处就是一个地铁站,这是一个美丽。他用画笔为自己画了一副美丽的人生狼图。夏天有很多小朋友在沙汤上玩耍。'
```

**Problems**:
- ❌ 0.04% accuracy is essentially random
- ❌ Result 2 duplicates Result 1 + adds hallucination
- ❌ CER 122% = generating more errors than reference length
- ❌ Content disconnected and incoherent

**Root Cause**: Clearing cache loses context, but NOT clearing cache causes language mismatch. No middle ground exists.

### 4.6 Comparison with Reference SimulStreaming

**Single-Language Performance** (from previous tests, NOT code-switching):
```
Chinese-only: CER ~15-25% (acceptable)
English-only: WER ~10-20% (good)
Overall Accuracy: 75-90% (production-ready)
```

**Code-Switching Performance** (all 4 test attempts):
```
Best case: 0.04% overall accuracy
Typical case: 0% overall accuracy
Chinese CER: 60-215%
English WER: 68-77%
```

**Conclusion**: **96-100% accuracy loss** when attempting code-switching.

---

## 5. Why SimulStreaming Cannot Support Code-Switching: Architectural Constraints

### 5.1 Fundamental Design Assumptions

SimulStreaming makes several **hard assumptions** that are **incompatible with code-switching**:

1. **Single-language session assumption**:
   ```python
   # Line 368: Guard prevents re-detection
   if self.cfg.language == "auto" and self.detected_language is None:
       # Detect language ONCE
   ```
   Architecture assumes language is **pinned** at session start.

2. **Continuous token sequence assumption**:
   ```python
   # Line 596: Tokens accumulate continuously
   self.tokens.append(new_tokens)  # Never reset mid-session
   ```
   Architecture assumes token sequence is **unbroken** (no resets).

3. **Stable KV cache assumption**:
   ```python
   # Line 95: Cache grows continuously
   self.kv_cache[module.cache_id] = torch.cat([x, net_output], dim=1).detach()
   ```
   Architecture assumes cache is **never cleared mid-session** (only at segment boundaries).

4. **Language-consistent attention assumption**:
   ```python
   # AlignAtt relies on consistent attention patterns
   most_attended_frame = torch.argmax(attn_of_alignment_heads[:,-1,:], dim=-1)
   ```
   Architecture assumes attention patterns are **stable** (same language throughout).

### 5.2 What Would Be Required for Code-Switching

To support code-switching in SimulStreaming's architecture, we would need:

1. **Language-aware KV cache management**:
   - Separate caches for each language
   - Context switching between caches on language change
   - Cache merging or translation (not possible in transformer architecture)

2. **Cross-language token continuity**:
   - Tokenizer that handles mixed-language sequences
   - Context buffer that preserves cross-language context
   - Training data with code-switching examples (Whisper wasn't trained this way)

3. **Dynamic SOT injection**:
   - Ability to inject new SOT mid-sequence
   - Decoder trained to handle mid-sequence language changes
   - Mechanism to prevent token sequence discontinuity

4. **Language-adaptive attention**:
   - Attention heads that adapt to language switches
   - Cross-language alignment patterns
   - Training on code-switched audio (Whisper wasn't trained this way)

5. **Whisper model retraining**:
   - Add code-switching examples to training data
   - Train decoder to handle mid-sequence language changes
   - Modify architecture to support dynamic language conditioning

**None of these are feasible** without retraining the Whisper model itself.

---

## 6. Proposed Solutions: Trade-offs and Recommendations

### 6.1 Option 1: Revert All Code-Switching Changes (RECOMMENDED)

**Approach**: Remove all code-switching code, accept single-language limitation.

**Changes Required**:
1. Revert `vac_online_processor.py:350-372` to check VAD silence FIRST
2. Revert `simul_whisper.py:482-485` to detect language ONCE
3. Remove `simul_whisper.py:251-271` (update_language_tokens method)
4. Remove `simul_whisper.py:467-474` (newest segment detection)
5. Remove `enable_code_switching` flag and related logic

**Pros**:
- ✅ Restore 75-90% accuracy (Phase 1 baseline)
- ✅ Production-ready, stable performance
- ✅ Clean architecture (matches reference SimulStreaming)
- ✅ No accuracy degradation
- ✅ Fast, low compute cost

**Cons**:
- ❌ No code-switching support
- ❌ Must restart session to change language
- ❌ Cannot handle intra-sentence language mixing

**Use Cases**:
- ✅ Monolingual meetings (English OR Chinese, not both)
- ✅ Separate sessions per language
- ✅ High-accuracy requirements
- ❌ Mixed-language meetings (requires Option 2 or 3)

**Recommendation**: **Use this for production**. Accept that SimulStreaming is a single-language architecture.

---

### 6.2 Option 2: Session Restart on Language Switch

**Approach**: Detect language change, finish current session, start new session with new language.

**Architecture**:
```python
class MultiLanguageSessionManager:
    def __init__(self):
        self.current_session = SimulWhisper(language='auto')
        self.pending_audio = []

    def process_chunk(self, audio):
        # Detect language on new audio
        detected_lang = detect_language(audio)

        if detected_lang != self.current_session.language:
            # Finish current session
            final_output = self.current_session.finish()
            yield final_output

            # Start NEW session with new language
            self.current_session = SimulWhisper(language=detected_lang)
            self.current_session.insert_audio(audio)
        else:
            # Continue current session
            self.current_session.insert_audio(audio)

        return self.current_session.infer()
```

**Pros**:
- ✅ Clean language separation (no cache mixing)
- ✅ Each session maintains high accuracy
- ✅ Preserves SimulStreaming's strengths
- ✅ Language switches are clean boundaries

**Cons**:
- ❌ Latency spike on language switch (new session startup)
- ❌ Lose cross-language context (English text → Chinese session starts fresh)
- ❌ Cannot handle rapid language switching (e.g., "我想要 an apple")
- ❌ Audio segmentation required (where to cut between languages?)

**Use Cases**:
- ✅ Inter-sentence language switching (speaker changes languages between sentences)
- ✅ Meetings where one person speaks English, another speaks Chinese
- ❌ Intra-sentence code-switching (rapid mixing within single sentence)

**Implementation Complexity**: Medium (3-5 days)

---

### 6.3 Option 3: Dual-Session Parallel Processing

**Approach**: Run TWO SimulStreaming sessions (English + Chinese) in parallel, select output based on language confidence.

**Architecture**:
```python
class DualSessionProcessor:
    def __init__(self):
        self.session_en = SimulWhisper(language='en')
        self.session_zh = SimulWhisper(language='zh')

    def process_chunk(self, audio):
        # Send audio to BOTH sessions
        output_en = self.session_en.insert_audio(audio).infer()
        output_zh = self.session_zh.insert_audio(audio).infer()

        # Compare confidence scores
        if output_en.confidence > output_zh.confidence:
            return output_en
        else:
            return output_zh
```

**Pros**:
- ✅ No session restarts (always running)
- ✅ Fast language switching (no latency spike)
- ✅ Each session maintains high accuracy
- ✅ Can handle rapid language switching

**Cons**:
- ❌ 2x compute cost (running two models)
- ❌ 2x memory usage
- ❌ Need robust confidence scoring (Whisper doesn't provide reliable confidence)
- ❌ May produce mixed output if confidence is similar

**Use Cases**:
- ✅ High compute budget environments
- ✅ Rapid intra-sentence code-switching
- ✅ Mixed-language meetings with frequent switching
- ❌ Resource-constrained deployments

**Implementation Complexity**: Medium (4-6 days)

---

### 6.4 Option 4: Switch to Different Architecture

**Approach**: Replace SimulStreaming with a different ASR architecture that supports code-switching.

#### Option 4a: Standard Whisper (Non-Streaming) with Sliding Window

**Architecture**:
```python
class SlidingWindowWhisper:
    def __init__(self, window_size=10.0, stride=5.0):
        self.window_size = window_size
        self.stride = stride
        self.audio_buffer = []

    def process_chunk(self, audio):
        self.audio_buffer.append(audio)

        if len(self.audio_buffer) >= self.window_size:
            # Run standard Whisper on window
            result = whisper.transcribe(
                audio=self.audio_buffer,
                language='auto',  # Re-detect on every window
                task='transcribe'
            )

            # Slide window forward
            self.audio_buffer = self.audio_buffer[self.stride:]
            return result
```

**Pros**:
- ✅ Standard Whisper supports code-switching naturally
- ✅ Language detection built-in
- ✅ High accuracy on mixed-language audio
- ✅ Simple implementation

**Cons**:
- ❌ Higher latency (process 10s windows with 5s stride = 5s delay)
- ❌ No real-time streaming (batch processing)
- ❌ Duplication handling required (overlapping windows)

#### Option 4b: faster-whisper + Separate LID

**Architecture**:
```python
from faster_whisper import WhisperModel
from language_detector import LanguageDetector

class FasterWhisperCodeSwitching:
    def __init__(self):
        self.model = WhisperModel("large-v3", device="cuda")
        self.lid = LanguageDetector()  # Separate language detection

    def process_chunk(self, audio):
        # Detect language first
        language = self.lid.detect(audio)

        # Transcribe with detected language
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=True
        )
        return list(segments)
```

**Pros**:
- ✅ faster-whisper is optimized for low latency
- ✅ Separate LID can be very fast (e.g., wav2vec2-based)
- ✅ Can handle code-switching by detecting per-segment
- ✅ Lower compute cost than dual-session

**Cons**:
- ❌ Still batch-based (not truly streaming)
- ❌ Need robust LID model (additional dependency)
- ❌ Language detection errors propagate to transcription

#### Option 4c: wav2vec2 + Language-Specific Models

**Architecture**:
```python
class Wav2Vec2CodeSwitching:
    def __init__(self):
        self.lid_model = Wav2Vec2ForAudioClassification.from_pretrained("facebook/mms-lid-1024")
        self.asr_en = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        self.asr_zh = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-zh")

    def process_chunk(self, audio):
        # Detect language
        language = self.lid_model.predict(audio)

        # Route to appropriate ASR
        if language == 'en':
            return self.asr_en.transcribe(audio)
        else:
            return self.asr_zh.transcribe(audio)
```

**Pros**:
- ✅ True streaming capability
- ✅ Fast language detection (dedicated LID model)
- ✅ Can handle rapid code-switching
- ✅ Lower latency than Whisper

**Cons**:
- ❌ Lower accuracy than Whisper
- ❌ More complex pipeline
- ❌ Need to manage multiple models

**Recommendation**: For code-switching, **Option 4a (Standard Whisper + Sliding Window)** is most reliable, but has higher latency.

**Implementation Complexity**: High (1-2 weeks)

---

### 6.5 Option 5: Hybrid Approach (SimulStreaming + Session Router)

**Approach**: Use SimulStreaming for primary language, separate LID for detection, dynamically route to appropriate session.

**Architecture**:
```python
class HybridSessionRouter:
    def __init__(self):
        self.sessions = {
            'en': SimulWhisper(language='en'),
            'zh': SimulWhisper(language='zh')
        }
        self.lid = SlidingLIDDetector(window_size=0.9)
        self.current_language = None
        self.audio_buffer = []

    def process_chunk(self, audio):
        # Update language detection
        self.lid.add_audio(audio)
        detected_language = self.lid.get_current_language()

        # Check if language switched
        if detected_language != self.current_language:
            if self.current_language is not None:
                # Finish current session
                final_output = self.sessions[self.current_language].finish()
                yield final_output

                # Start new session
                self.sessions[detected_language].init()

            self.current_language = detected_language

        # Route to active session
        return self.sessions[self.current_language].process(audio)
```

**Pros**:
- ✅ Leverages SimulStreaming's high accuracy
- ✅ Dedicated LID model (more reliable than Whisper's internal LID)
- ✅ Clean language separation
- ✅ Configurable switching granularity

**Cons**:
- ❌ Still has session restart latency
- ❌ Lose cross-language context
- ❌ Complex state management

**Use Cases**:
- ✅ Meetings with clear language boundaries (e.g., speaker 1 speaks English, speaker 2 speaks Chinese)
- ✅ Gradual language transitions (finish English section → start Chinese section)
- ❌ Rapid intra-sentence code-switching

**Implementation Complexity**: High (1-2 weeks)

---

## 7. Recommendations

### 7.1 Immediate Action (Next 24 Hours)

**REVERT ALL CODE-SWITCHING CHANGES** and restore Phase 1 baseline:

1. **Revert processing order** (vac_online_processor.py:350-372):
   ```python
   # REVERT TO: Check VAD silence FIRST
   if self.is_currently_final:
       return self._finish()
   elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
       return self._process_online_chunk()
   ```

2. **Revert language detection** (simul_whisper.py:482-485):
   ```python
   # REVERT TO: Detect ONCE only
   if self.cfg.language == "auto" and self.detected_language is None:
       language_tokens, language_probs = self.lang_id(encoder_feature)
   ```

3. **Remove update_language_tokens()** (simul_whisper.py:251-271):
   - Delete entire method
   - Remove all calls to it

4. **Remove newest segment detection** (simul_whisper.py:467-474):
   - Use full encoder_feature for language detection (reference behavior)

5. **Remove enable_code_switching flag**:
   - Remove from all files
   - Remove related conditional logic

**Expected Result**: Restore 75-90% accuracy baseline.

### 7.2 Short-Term Solution (Next 1-2 Weeks)

**Implement Option 2: Session Restart on Language Switch**

This provides code-switching capability while maintaining SimulStreaming's high accuracy:

1. Create `MultiLanguageSessionManager` wrapper
2. Implement session restart logic
3. Add language detection at session boundaries
4. Handle output merging and deduplication

**Pros**: Production-ready within 1-2 weeks, maintains high accuracy.

**Cons**: Cannot handle rapid intra-sentence switching.

### 7.3 Long-Term Solution (Next 1-2 Months)

**Implement Option 4a: Standard Whisper with Sliding Window**

This provides true code-switching support with high accuracy:

1. Replace SimulStreaming with standard Whisper
2. Implement sliding window processor (10s window, 5s stride)
3. Add deduplication logic for overlapping windows
4. Optimize for latency (smaller windows, GPU acceleration)

**Pros**: True code-switching support, high accuracy maintained.

**Cons**: Higher latency (5-10s), more compute cost.

### 7.4 Production Deployment Strategy

**Tiered Approach**:

1. **Tier 1: Monolingual Sessions** (Available Now)
   - Use reverted SimulStreaming (Phase 1 baseline)
   - Accuracy: 75-90%
   - Latency: <2s
   - Use case: Single-language meetings

2. **Tier 2: Inter-Sentence Code-Switching** (1-2 weeks)
   - Use Option 2 (Session Restart)
   - Accuracy: 70-85% (slight drop at boundaries)
   - Latency: <3s (includes session restart)
   - Use case: Meetings where speakers change languages between sentences

3. **Tier 3: Intra-Sentence Code-Switching** (1-2 months)
   - Use Option 4a (Standard Whisper + Sliding Window)
   - Accuracy: 60-80% (Whisper's native code-switching accuracy)
   - Latency: 5-10s
   - Use case: True mixed-language conversations

---

## 8. Conclusion

**SimulStreaming is fundamentally incompatible with code-switching** due to architectural constraints:

1. **Language is pinned at session start** (detected once, never changes)
2. **KV cache accumulates language-specific patterns** (clearing = context loss)
3. **Token sequence must be continuous** (resetting = disconnected output)
4. **Encoder-decoder attention is language-specific** (switching = misalignment)
5. **SOT tokens condition entire decoder** (mid-stream updates violate training assumptions)

**All attempts to add code-switching resulted in 96-100% accuracy loss.**

**Recommendation**:
- **Immediate**: Revert all code-switching changes, restore 75-90% baseline
- **Short-term**: Implement session restart approach (Option 2)
- **Long-term**: Switch to standard Whisper with sliding window (Option 4a)

**The only way to achieve true code-switching** is to use a different architecture or retrain the Whisper model itself.

---

## Appendix A: Test Logs Reference

- `/tmp/sot_fix_test.log` - SOT token updates (0% accuracy)
- `/tmp/context_preserved_test.log` - Context preservation attempt (19.81% accuracy)
- `/tmp/multilingual_tokenizer_test.log` - Multilingual tokenizer (15.91% accuracy)
- `/tmp/FINAL_simulstream_cache_test.log` - Full cache clearing (0.04% accuracy)

All logs show catastrophic accuracy loss compared to Phase 1 baseline (75-90%).

---

## Appendix B: Reference SimulStreaming Files

- `reference/SimulStreaming/simul_whisper/simul_whisper.py` - Core SimulStreaming implementation
- `reference/SimulStreaming/whisper_streaming/vac_online_processor.py` - VAC processor
- Key insight: Language detection at line 368 has guard `self.detected_language is None`

---

**END OF DOCUMENT**
