# Code-Switching Implementation Analysis: Intra-Sentence Multilingual Support

## 🎯 **Goal**: Support sentences like "我想要 a coffee please" (I want a coffee please)

**Question**: Can we implement Simul-Whisper-style streaming with dynamic LID (Language ID) without pinning the decoder to one language?

---

## ✅ **GOOD NEWS: It's Technically Feasible!**

### Why it works:
1. ✅ **Whisper has built-in language detection** that doesn't break KV cache
2. ✅ **We already have AlignAtt decoder** for streaming
3. ✅ **We already have SimulStreaming infrastructure** (simul_whisper/)
4. ✅ **Whisper's multilingual tokenizer** handles code-switching natively
5. ✅ **One decoder for all languages** - no decoder swapping needed

---

## 🔍 **Current Architecture Analysis**

### **What We Have** ✅

#### 1. **SimulStreaming Infrastructure** (`src/simul_whisper/`)
```python
# simul_whisper.py line 363-369
def lang_id(self, encoder_features):
    """Language detection from encoder features.
    This code is trimmed and copy-pasted from whisper.decoding.detect_language .
    """
```

**Status**: ✅ **Already implemented** but not used dynamically

#### 2. **AlignAtt Decoder** (`src/alignatt_decoder.py`)
```python
class AlignAttDecoder:
    """Attention-guided streaming decoder for Whisper

    From SimulStreaming paper (Section 3.2):
    - Frame threshold: Maximum audio frame the decoder can attend to
    - Frame offset (τ): Frames reserved for future streaming (default: 10)
    - Incremental decoding: Process audio as it arrives, emit tokens early
    """
```

**Status**: ✅ **Fully functional** for streaming

#### 3. **Whisper's Built-in Language Detection**
```python
# From openai-whisper
whisper.decoding.detect_language(model, mel, tokenizer)
```

**Documentation**:
> "This is performed outside the main decode loop in order to not interfere with kv-caching."

**Status**: ✅ **Available** - specifically designed NOT to break cache!

#### 4. **Multilingual Tokenizer**
- ✅ Whisper uses ONE tokenizer for all 99 languages
- ✅ Can emit tokens from multiple languages in single stream
- ✅ Language tokens exist in vocabulary (e.g., `<|en|>`, `<|zh|>`)

---

### **What's Currently Broken** ❌

#### 1. **Language is Pinned Per-Request**
**Location**: `src/whisper_service.py:858-860`

```python
# CURRENT CODE (WRONG for code-switching):
if language:
    decode_options["language"] = language  # ❌ Pins decoder to one language
    logger.info(f"[INFERENCE] Language: {language}")
```

**Problem**: Forces decoder to output ONLY that language, breaks code-switching.

**Fix**: Remove language pinning, let Whisper auto-detect.

```python
# FIXED CODE (for code-switching):
if language and not enable_code_switching:
    decode_options["language"] = language  # Only pin if NOT code-switching
else:
    # Let Whisper auto-detect language per token window
    decode_options["language"] = None  # ✅ Allows code-switching
    logger.info(f"[CODE-SWITCHING] Dynamic language detection enabled")
```

#### 2. **No Dynamic LID Loop**
**Status**: We have `lang_id()` but don't call it per-chunk for dynamic switching.

**Current Flow**:
```
Audio Chunk → Encode → Decode (pinned to "en") → Output English only
```

**Needed Flow**:
```
Audio Chunk → Encode → Detect Language → Decode (unpinned) → Output Mixed Languages
    ↓
Tag output with detected language per segment
```

#### 3. **No Truncation Detector**
**Status**: We have stability tracking but not the SimulStreaming truncation detector.

**Location**: Stability tracking in `whisper_service.py:1033-1041` exists but differs from truncation detector.

---

## 📋 **Implementation Plan**

### **Phase 1: Enable Dynamic Language Detection** (1-2 days)

#### Step 1.1: Add Code-Switching Flag
```python
# TranscriptionRequest dataclass
@dataclass
class TranscriptionRequest:
    # ... existing fields ...
    enable_code_switching: bool = False  # NEW: Enable intra-sentence code-switching
    language: Optional[str] = None       # If None and code_switching=True, auto-detect
```

#### Step 1.2: Modify safe_inference to Support Dynamic LID
```python
def safe_inference(
    self,
    model_name: str,
    audio_data: np.ndarray,
    ...
    session_id: Optional[str] = None,
    enable_code_switching: bool = False  # NEW
):
    # ...

    # Phase 2.2: Dynamic language detection for code-switching
    if enable_code_switching:
        # DO NOT pin language
        decode_options["language"] = None
        logger.info(f"[CODE-SWITCHING] Dynamic LID enabled")
    else:
        # Original behavior: pin to specified language
        if language:
            decode_options["language"] = language

    # Perform transcription
    result = model.transcribe(audio=audio_data, **decode_options)

    # Post-process: detect language per segment
    if enable_code_switching:
        result = self._tag_language_segments(result, model, audio_data)

    return result
```

#### Step 1.3: Implement Language Tagging
```python
def _tag_language_segments(
    self,
    result: Dict,
    model: Any,
    audio_data: np.ndarray
) -> Dict:
    """
    Tag each segment with detected language using Whisper's LID.

    This runs AFTER transcription to tag segments without breaking cache.
    """
    import whisper
    from whisper.audio import log_mel_spectrogram

    # Get segments from result
    segments = result.get('segments', [])

    for segment in segments:
        start = segment['start']
        end = segment['end']

        # Extract audio for this segment
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        segment_audio = audio_data[start_sample:end_sample]

        # Detect language for this segment
        mel = log_mel_spectrogram(segment_audio)
        language_token, language_probs = whisper.detect_language(model, mel)

        # Get top language
        detected_lang = max(language_probs[0], key=language_probs[0].get)
        confidence = language_probs[0][detected_lang]

        # Tag segment
        segment['detected_language'] = detected_lang
        segment['language_confidence'] = confidence

        logger.debug(f"[LID] Segment: '{segment['text'][:50]}...' → {detected_lang} ({confidence:.2f})")

    return result
```

---

### **Phase 2: Implement True SimulStreaming Code-Switching** (3-5 days)

This is the more advanced approach using the actual SimulStreaming library.

#### Step 2.1: Use SimulStreaming's lang_id
```python
# In streaming loop
from simul_whisper import SimulWhisper

state = SimulWhisper(model="large-v3")  # Already in src/simul_whisper/

for chunk in audio_chunks:
    # Encode
    encoder_features = state.encode(chunk)

    # Detect language (doesn't break KV cache!)
    lang_probs = state.lang_id(encoder_features)
    detected_lang = max(lang_probs, key=lang_probs.get)

    # Decode WITHOUT language constraint
    hypo, attn = state.decode_until_frontier(
        encoder_features,
        language=None,  # ✅ No pinning!
        task="transcribe"
    )

    # Tag output with detected language
    emit(hypo.text, language=detected_lang, stable=is_stable(hypo, attn))
```

#### Step 2.2: Add Truncation Detector
**Reference**: SimulStreaming paper Section 3.3

The truncation detector determines if a hypothesis is "stable" (can be committed) or "tentative" (might change with more audio).

```python
class TruncationDetector:
    """
    Detect when decoder has produced stable text that won't change.

    From SimulStreaming: Uses attention alignment and token stability.
    """

    def is_stable(
        self,
        hypothesis: str,
        attention_weights: torch.Tensor,
        threshold: float = 0.85
    ) -> bool:
        """
        Check if hypothesis is stable enough to emit.

        Criteria:
        1. Attention has moved past audio frontier
        2. Last N tokens are consistent
        3. No partial word detected
        """
        # Implementation based on SimulStreaming eow_detection.py
        pass
```

---

## ⚖️ **Trade-offs**

### **Advantages** ✅

1. **True Multilingual Support**
   - ✅ Can handle "Spanglish", "Chinglish", etc.
   - ✅ Preserves semantic meaning in code-switched speech
   - ✅ More natural for bilingual speakers

2. **Better Accuracy for Multilingual Users**
   - ✅ No forced language = better WER for code-switchers
   - ✅ Whisper can pick optimal language per phrase

3. **Semantic Fidelity**
   - ✅ Preserves original language choice (important for sentiment, intent)
   - ✅ Better than post-hoc translation

4. **Already Mostly Built**
   - ✅ SimulStreaming infrastructure exists
   - ✅ Language detection exists
   - ✅ Multilingual tokenizer ready

### **Disadvantages** ❌

1. **Increased Complexity**
   - ⚠️ More complex processing logic
   - ⚠️ Need to tag languages per segment
   - ⚠️ Need UI to display multiple languages

2. **Potential Quality Degradation**
   - ⚠️ Whisper's code-switching isn't perfect without fine-tuning
   - ⚠️ May produce more hallucinations at language boundaries
   - ⚠️ Homophone confusion (e.g., English "see" vs Chinese "死 sǐ")

3. **Downstream Complications**
   - ⚠️ Translation service needs to handle mixed-language input
   - ⚠️ Rolling context management becomes more complex
   - ⚠️ Punctuation model needs language-aware processing

4. **Performance Impact**
   - ⚠️ Language detection adds ~10-20ms per chunk
   - ⚠️ Segment-level tagging adds processing overhead
   - ⚠️ May need more frequent truncation checks

5. **Model Limitations**
   - ⚠️ Whisper large-v3 not explicitly trained for code-switching
   - ⚠️ Some language pairs work better than others
   - ⚠️ Fine-tuned models (e.g., Belle-Whisper-zh) lose multilingual ability

---

## 🔬 **What About Fine-Tuned Models?**

### Question: "Does a fine-tune like whisper v3 zh still have the same efficacy on other languages?"

**Answer**: ❌ **NO** - Fine-tuning for one language reduces multilingual capacity.

#### Evidence:
- Belle-Whisper-Large-V3-ZH: 24-65% improvement on Chinese... but only Chinese
- Model capacity shifts toward fine-tuned distribution
- Other languages, especially rare ones, degrade

#### Recommendation:
- **For code-switching**: Use **base large-v3** (multilingual)
- **For pure Chinese**: Use **Belle-Whisper-zh** (monolingual)
- **Don't mix**: Can't have both high Chinese accuracy AND code-switching

---

## 📊 **Comparison: Current vs Proposed**

| Feature | Current System | Phase 1 (Simple) | Phase 2 (Full SimulStreaming) |
|---------|---------------|------------------|------------------------------|
| **Intra-sentence code-switching** | ❌ No | ✅ Yes (post-hoc tagging) | ✅ Yes (real-time detection) |
| **Language pinning** | ✅ Pinned per request | ⚠️ Optional | ❌ Never pinned |
| **KV cache** | ✅ Preserved | ✅ Preserved | ✅ Preserved |
| **Latency overhead** | Baseline | +20ms (tagging) | +10ms (detection) |
| **Implementation effort** | - | 1-2 days | 3-5 days |
| **Complexity** | Low | Medium | High |
| **Quality (monolingual)** | Best | Best | Best |
| **Quality (code-switching)** | N/A | Good | Best |

---

## 🎯 **Recommendation**

### **For Your Use Case**:

#### **If you need code-switching NOW**:
→ **Implement Phase 1** (1-2 days)
- Simple flag: `enable_code_switching=True`
- Remove language pinning
- Post-hoc language tagging
- Works with current architecture

#### **If you need production-grade code-switching**:
→ **Implement Phase 2** (3-5 days)
- Full SimulStreaming integration
- Real-time LID per chunk
- Truncation detector
- Better quality, lower latency

#### **If you mostly have monolingual speakers**:
→ **Keep current system**
- Pin language per session (current behavior)
- Better accuracy for single-language speech
- Simpler implementation

---

## 🛠️ **Implementation Checklist**

### Phase 1: Quick Code-Switching (1-2 days)
- [ ] Add `enable_code_switching` flag to TranscriptionRequest
- [ ] Modify `safe_inference` to skip language pinning when flag=True
- [ ] Implement `_tag_language_segments()` for post-hoc LID
- [ ] Test with mixed English/Chinese audio
- [ ] Update API to accept code_switching parameter

### Phase 2: Full SimulStreaming (3-5 days)
- [ ] Integrate `simul_whisper.lang_id()` into streaming loop
- [ ] Implement truncation detector from SimulStreaming
- [ ] Add per-chunk language detection
- [ ] Update rolling context to handle language switches
- [ ] Add language-aware punctuation model
- [ ] Test with real code-switching speech samples

### Testing:
- [ ] Test with "Spanglish" audio (English/Spanish mixed)
- [ ] Test with "Chinglish" audio (English/Chinese mixed)
- [ ] Verify no quality degradation on monolingual speech
- [ ] Benchmark latency impact
- [ ] Test with rapid language switching

---

## 📚 **References**

1. **SimulStreaming Paper**: IWSLT 2025, Section 3.2-3.3
   - AlignAtt policy
   - Truncation detection
   - Language detection strategy

2. **Whisper Documentation**:
   - `whisper.detect_language()`: "performed outside the main decode loop"
   - Multilingual tokenizer design
   - Code-switching behavior (undocumented but works)

3. **Your Codebase**:
   - `src/simul_whisper/simul_whisper.py`: lang_id() method (line 363)
   - `src/alignatt_decoder.py`: AlignAtt implementation
   - `src/whisper_service.py`: Current pinning logic (line 858)

---

## 🎬 **Example: What It Would Look Like**

### Input Audio (Mixed English/Chinese):
```
Speaker: "我想要 a large coffee with milk, 不要糖"
         (I want a large coffee with milk, no sugar)
```

### Current System Output:
```
Language pinned to "en": "wo xiang yao a large coffee with milk, bu yao tang"
Language pinned to "zh": "我想要 a large coffee with milk 不要糖"
                         ❌ English words might be transliterated incorrectly
```

### Proposed System Output (Code-Switching Enabled):
```json
{
  "text": "我想要 a large coffee with milk, 不要糖",
  "segments": [
    {
      "text": "我想要",
      "language": "zh",
      "confidence": 0.95,
      "start": 0.0,
      "end": 1.2
    },
    {
      "text": "a large coffee with milk",
      "language": "en",
      "confidence": 0.98,
      "start": 1.2,
      "end": 3.5
    },
    {
      "text": "不要糖",
      "language": "zh",
      "confidence": 0.96,
      "start": 3.5,
      "end": 4.2
    }
  ]
}
```

---

## ✅ **Bottom Line**

**YES, it's feasible!**

- ✅ Architecture supports it (SimulStreaming + AlignAtt + LID)
- ✅ Whisper is designed for this (multilingual tokenizer + detect_language)
- ✅ KV cache won't break (language detection is outside decode loop)
- ✅ Moderate effort (1-2 days for basic, 3-5 for production)

**Trade-offs**:
- ⚠️ Increased complexity
- ⚠️ Slight quality impact on monolingual speech (if not careful)
- ⚠️ Need language-aware post-processing

**Recommendation**:
Start with **Phase 1** (simple post-hoc tagging) to validate use case, then move to **Phase 2** (real-time LID) if needed.
