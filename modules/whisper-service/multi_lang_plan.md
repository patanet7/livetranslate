# Multi-Language Code-Switching Implementation Plan

**Status**: In Progress
**Started**: 2025-01-23
**Target Architecture**: Best-in-class streaming code-switching with sustained detection

---

## Executive Summary

Implementing production-grade code-switching for LiveTranslate with:
- ✅ Sliding LID (0.8-1.0s window) for tagging only
- ✅ Sustained language detection (2.5-3.0s + VAD pause) before SOT re-init
- ✅ Soft bias injection (prepend lang token without decoder reset)
- ✅ Token de-duplication at chunk boundaries
- ✅ N-best rescoring for low confidence
- ✅ Configurable VAD thresholds (min speech 120ms, min silence 220-300ms)

**System Capability Assessment**: ✅ **SUPPORTED** - Strong foundation, requires moderate enhancements

---

## Current System Strengths

✅ SimulStreaming with AlignAtt + truncation detection
✅ Stateful decoder with KV cache preservation
✅ Beam search with n-best tracking
✅ Per-session VAC processors with isolated state
✅ Language detection (`lang_id()`) available

---

## Implementation Phases

### **Phase 1: Configuration Infrastructure** ✅ COMPLETE
*Enable all parameters to flow through the stack*

**Status**: All config parameters flow correctly through WebSocket → VAC → Stateful Model

**Files Modified:**
- ✅ `src/simul_whisper/config.py` - Added AlignAttConfig parameters
- ✅ `src/whisper_service.py` - Added TranscriptionRequest fields
- ✅ `src/api_server.py` - Extract config from WebSocket messages (lines 2160-2192)
- ✅ `tests/test_config_flow.py` - Integrated test suite (4/4 tests passing)
- ☐ `modules/orchestration-service/src/socketio_whisper_client.py` - Add fields (later)
- ☐ `modules/orchestration-service/src/clients/audio_service_client.py` - Add fields (later)

**New Config Parameters:**
```python
sliding_lid_window: float = 0.9  # seconds
sustained_lang_duration: float = 3.0  # seconds
sustained_lang_min_silence: float = 0.25  # seconds
soft_bias_enabled: bool = False
token_dedup_enabled: bool = True
confidence_threshold: float = 0.6
vad_min_speech_ms: int = 120
vad_min_silence_ms: int = 250
```

**Testing Strategy:**
```python
# test_config_flow.py - INTEGRATED TEST
async def test_config_propagation():
    """Test config flows from WebSocket → VAC → Stateful model"""
    # Send WebSocket message with all params
    # Verify extraction in TranscriptionRequest
    # Verify VAC processor receives params
    # Verify stateful model receives params
    # Verify defaults applied when omitted
```

**Success Criteria:**
- ✅ All config parameters flow from WebSocket → VAC → stateful model
- ✅ Defaults match specification
- ✅ Per-session overrides work
- ✅ Config validation prevents invalid values (graceful handling)
- ✅ Force task='transcribe' when code-switching enabled

---

### **Phase 2: VAD Enhancement** ✅ COMPLETE
*Make VAD configurable per requirements*

**Status**: Per-session VAD with full parameter control verified working

**Files Modified:**
1. ✅ `src/silero_vad_iterator.py`:
   - Added `min_speech_duration_ms` parameter (default 120ms)
   - Made `min_silence_duration_ms` configurable (default 250ms)
   - Track speech start time and filter short bursts

2. ✅ `src/vac_online_processor.py`:
   - Accept VAD config parameters (vad_min_speech_ms, vad_min_silence_ms)
   - Pass to FixedVADIterator initialization

3. ✅ `src/api_server.py`:
   - Extract VAD params from TranscriptionRequest (lines 2210-2226)
   - Create per-session FixedVADIterator with custom config (lines 2318-2337)

4. ✅ `tests/test_vad_enhancement.py` - Integrated test suite (3/3 tests passing)

**Testing Strategy:**
```python
# test_vad_thresholds.py - INTEGRATED TEST
async def test_vad_configurable_thresholds():
    """Test VAD respects custom thresholds"""
    # Create session with vad_min_speech_ms=120, vad_min_silence_ms=250
    # Send audio with 100ms speech burst → should be filtered
    # Send audio with 150ms speech → should be detected
    # Send audio with 200ms pause → should NOT end speech
    # Send audio with 300ms pause → should end speech
    # Verify timestamps match expectations
```

**Success Criteria:**
- ✅ VAD respects min speech 120ms (configurable)
- ✅ VAD respects min silence 250ms (configurable)
- ✅ Per-session thresholds work (verified in logs)
- ✅ No regressions in existing VAD behavior
- ✅ Per-session FixedVADIterator instances (not shared)

---

### **Phase 3: Sliding LID Window** ✅ COMPLETE
*Track language detections in rolling window for UI/formatting*

**Status**: Fully working with REAL audio - English and Chinese detection verified!

**New Files Created:**
1. ✅ `src/sliding_lid_detector.py` (190 lines):
   - SlidingLIDDetector class with configurable window (default 0.9s)
   - `add_detection()` - tracks language with timestamp and audio position
   - `get_current_language()` - returns majority language in window
   - `get_sustained_language(min_duration)` - checks if sustained
   - Automatic purging of old detections
   - PASSIVE tracking only - does NOT affect decoder

2. ✅ `tests/test_sliding_lid.py` (341 lines) - Integrated tests
3. ✅ `tests/test_detected_language_real_audio.py` (242 lines) - Real audio validation

**Files Modified:**
- ✅ `src/vac_online_processor.py`:
  - Added `sliding_lid_window` parameter to constructor
  - Initialize `SlidingLIDDetector(window_size)` in __init__
  - Track detected_language after each `infer()` (lines 375-397)
  - Return `detected_language` in result dict

- ✅ `src/simul_whisper/simul_whisper.py`:
  - **CRITICAL FIX**: `set_task()` now updates `self.cfg.language` (lines 190-194)
  - Enables language detection condition: `self.cfg.language == "auto"`

- ✅ `src/api_server.py`:
  - Pass `sliding_lid_window` to VACOnlineASRProcessor (line 2215)
  - Include `detected_language` in WebSocket transcription_data (line 2484)

**Testing Strategy:**
```python
# test_sliding_lid.py - INTEGRATED TEST
async def test_sliding_lid_window():
    """Test sliding window tracks language correctly"""
    # Chinese audio (2s) → English audio (1s) → Chinese audio (2s)
    # Window size: 0.9s
    # Verify detections tracked in window
    # Verify old detections purged after 0.9s
    # Verify get_current_language() returns majority
    # Verify get_sustained_language(2.5) works correctly

    # Test edge cases:
    # - Rapid language switches (< 0.9s)
    # - Single language sustained (> 3s)
    # - Empty window behavior
```

**Success Criteria:**
- ✅ Language tracked in 0.9s sliding window (configurable)
- ✅ `detected_language` field appears in WebSocket responses
- ✅ Window correctly purges old detections
- ✅ Verified with REAL audio: English (JFK) detected as 'en', Chinese detected as 'zh'
- ✅ Per-session sliding window isolation
- ✅ No impact on transcription accuracy

**Known Streaming Artifacts (To be fixed in Phase 5):**
- ⚠️ Chunk boundary artifacts: Chinese text shows `�` characters at chunk edges
  - Example: "院子门口不远**�**" → "**�**就是一个地铁站..."
  - Root cause: Incomplete tokens at streaming chunk boundaries
  - **Will be verified fixed in Phase 5** using `test_detected_language_real_audio.py`
- ⚠️ Sentence truncation: JFK speech cuts off mid-sentence
  - Expected: "...ask what you can do for your **country**"
  - Actual: "...ask what you can do for your" (missing "country")
  - Root cause: Same chunk boundary issue
  - **Phase 5 token deduplication will resolve this**

---

### **Phase 4: Sustained Language Detection** ☐ PENDING
*Only reset decoder on sustained change + VAD pause*

**Files Modified:**
`src/vac_online_processor.py`:
- Add sustained detection logic
- Track silence duration
- Implement `_should_reset_sot()`
- Add cooldown mechanism (max 1 reset per 5s)
- Reset SOT in `_finish()` only when conditions met

**Testing Strategy:**
```python
# test_sustained_detection.py - INTEGRATED TEST
async def test_sustained_language_detection():
    """Test SOT only resets on sustained change"""
    # Scenario 1: Transient switch (should NOT reset)
    # Chinese (3s) → English (0.5s) → Chinese (3s)
    # Verify NO SOT reset (English too short)

    # Scenario 2: Sustained switch without silence (should NOT reset)
    # Chinese (3s) → English (3.5s no pause)
    # Verify NO SOT reset (no VAD silence)

    # Scenario 3: Sustained switch with silence (should reset)
    # Chinese (3s) → silence (0.3s) → English (3s)
    # Verify SOT reset ONCE at silence

    # Scenario 4: Cooldown test
    # Multiple sustained changes within 5s
    # Verify max 1 reset per 5s

    # Verify KV cache not corrupted after reset
    # Verify transcription quality maintained
```

**Success Criteria:**
- [ ] SOT only resets on sustained language (2.5-3.0s + 250ms silence)
- [ ] Transient language switches ignored
- [ ] Max reset frequency: once per 5 seconds
- [ ] KV cache integrity maintained
- [ ] Transcription quality no degradation

---

### **Phase 5: Token De-duplication** ☐ PENDING
*Prevent repeated tokens at chunk boundaries*

**New File:**
`src/token_deduplicator.py` (~80 lines):
```python
class TokenDeduplicator:
    """
    Tracks recent tokens to prevent duplication at chunk boundaries.
    """
    def deduplicate(self, new_tokens: List[int]) -> List[int]
    def reset(self)  # On segment boundaries
```

**Files Modified:**
- `src/vac_online_processor.py`:
  - Add `self.token_deduplicator = TokenDeduplicator()`
  - Call `deduplicate()` after `infer()`
  - Reset on segment boundaries

**Testing Strategy:**
```python
# test_token_dedup.py - INTEGRATED TEST
async def test_token_deduplication():
    """Test tokens deduplicated at chunk boundaries"""
    # Scenario 1: Exact token overlap
    # Chunk 1 ends: [... "hello", "world"]
    # Chunk 2 starts: ["world", "again", ...]
    # Verify "world" deduplicated

    # Scenario 2: Multi-token overlap
    # Chunk 1 ends: [... "hello", "world", "this"]
    # Chunk 2 starts: ["world", "this", "is", ...]
    # Verify 2 tokens deduplicated

    # Scenario 3: No overlap
    # Verify no false positives

    # Scenario 4: Word-level vs token-level
    # Test with multi-token words

    # Performance: Verify < 2ms overhead
```

**Success Criteria:**
- [ ] No repeated phrases at chunk boundaries
- [ ] Handles word-level and token-level overlaps
- [ ] No false positives (removing valid tokens)
- [ ] Latency overhead < 2ms
- [ ] Works with beam search

**Verification Test (Must Re-run):**
- [ ] **Re-run `test_detected_language_real_audio.py`** to verify streaming artifacts fixed:
  - Chinese text should have NO `�` characters at chunk boundaries
  - JFK speech should include complete sentence: "...ask what you can do for your **country**"
  - Baseline (Phase 3): Chinese had `�`, JFK missing "country"
  - After Phase 5: Both should be clean and complete

---

### **Phase 6: N-best Rescoring** ☐ PENDING
*Use beam search alternatives when confidence low*

**New File:**
`src/nbest_rescorer.py` (~70 lines):
```python
class NbestRescorer:
    """
    Rescores beam search hypotheses when primary has low confidence.
    """
    def rescore_if_needed(
        self,
        primary_text,
        primary_confidence,
        beam_hypotheses
    ) -> Tuple[str, float, bool]
```

**Files Modified:**
- `src/simul_whisper/simul_whisper.py`:
  - Add `get_beam_hypotheses()` method
  - Expose `finished_sequences` from BeamSearchDecoder

- `src/vac_online_processor.py`:
  - Add `self.nbest_rescorer = NbestRescorer(threshold)`
  - Call `rescore_if_needed()` after `infer()`

**Testing Strategy:**
```python
# test_nbest_rescoring.py - INTEGRATED TEST
async def test_nbest_rescoring():
    """Test low confidence triggers rescoring"""
    # Scenario 1: High confidence (should NOT rescore)
    # Primary: "hello world" (conf: 0.85)
    # Verify no rescoring triggered

    # Scenario 2: Low confidence with better alternative
    # Primary: "there" (conf: 0.45)
    # Alternative: "their" (conf: 0.78)
    # Verify alternative selected

    # Scenario 3: Low confidence without better alternative
    # Verify primary kept (no thrashing)

    # Scenario 4: Homophone detection
    # "see" vs "sea" - verify correct choice

    # Performance: Verify < 50ms when triggered
```

**Success Criteria:**
- [ ] Low confidence segments (<0.6) trigger rescoring
- [ ] Beam alternatives accessible
- [ ] Fallback graceful if no alternatives
- [ ] No thrashing on ambiguous audio
- [ ] Latency < 50ms when rescoring

---

## Integration Testing

### **End-to-End Code-Switching Test**
```python
# test_e2e_code_switching.py - COMPREHENSIVE INTEGRATED TEST
async def test_end_to_end_code_switching():
    """
    Full pipeline test with realistic mixed-language audio.

    Audio: Chinese (3s) → English (2s) → Chinese (3s) → English (5s)
    Silence: 0.3s gaps between language changes

    Expected behavior:
    1. Phase 3: Sliding LID tracks language throughout
    2. Phase 2: VAD respects 120ms speech, 250ms silence
    3. Phase 4: SOT resets at:
       - 3s mark (Chinese → English, sustained + silence)
       - 8s mark (English → Chinese, sustained + silence)
       - NO reset at 5s mark (English segment too short)
    4. Phase 5: Tokens deduplicated at chunk boundaries
    5. Phase 6: Low confidence segments rescored
    6. Phase 1: All config parameters respected

    Verify:
    - Transcription accuracy > 95%
    - Language tags 100% correct
    - No repeated phrases
    - Latency < 150ms per chunk
    - Memory stable (no leaks)
    """
```

### **Stress Testing**
```python
# test_stress_code_switching.py
async def test_rapid_language_switching():
    """Stress test with rapid language changes"""
    # 10 minutes of audio
    # Language switches every 2-5 seconds
    # Verify no degradation over time
    # Verify memory stable
    # Verify no SOT thrashing

async def test_single_language_no_regression():
    """Verify monolingual performance unchanged"""
    # Pure English audio (10 minutes)
    # Verify WER unchanged from baseline
    # Verify latency unchanged
    # Verify no spurious language detections
```

---

## Test Files

**New Test Files to Create:**
1. `tests/test_config_flow.py` - Phase 1 config propagation
2. `tests/test_vad_thresholds.py` - Phase 2 VAD configuration
3. `tests/test_sliding_lid.py` - Phase 3 sliding window
4. `tests/test_sustained_detection.py` - Phase 4 sustained language
5. `tests/test_token_dedup.py` - Phase 5 deduplication
6. `tests/test_nbest_rescoring.py` - Phase 6 rescoring
7. `tests/test_e2e_code_switching.py` - Full integration
8. `tests/test_stress_code_switching.py` - Stress & regression

**Test Data Required:**
- `tests/audio/chinese_3s.wav` - Pure Chinese 3 seconds
- `tests/audio/english_3s.wav` - Pure English 3 seconds
- `tests/audio/mixed_realistic.wav` - Chinese+English mixed
- `tests/audio/rapid_switching.wav` - Rapid language changes
- Existing: `tests/audio/OSR_cn_000_0072_8k.wav`, `tests/audio/jfk.wav`

---

## Implementation Status

### Phase 1: Configuration Infrastructure ⏳
- [x] AlignAttConfig parameters added
- [x] TranscriptionRequest fields added
- [ ] WebSocket handler extraction
- [ ] Orchestration service integration
- [ ] Test: test_config_flow.py
- [ ] **COMMIT CHECKPOINT**: Config infrastructure complete

### Phase 2: VAD Enhancement ☐
- [ ] VADIterator min_speech_duration
- [ ] VAC processor VAD config
- [ ] API server VAD extraction
- [ ] Test: test_vad_thresholds.py
- [ ] **COMMIT CHECKPOINT**: VAD configurable

### Phase 3: Sliding LID Window ☐
- [ ] SlidingLIDDetector class
- [ ] VAC integration
- [ ] WebSocket response field
- [ ] Test: test_sliding_lid.py
- [ ] **COMMIT CHECKPOINT**: Sliding LID working

### Phase 4: Sustained Detection ☐
- [ ] Sustained change detection logic
- [ ] SOT reset conditions
- [ ] Cooldown mechanism
- [ ] Test: test_sustained_detection.py
- [ ] **COMMIT CHECKPOINT**: Smart SOT reset

### Phase 5: Token Deduplication ☐
- [ ] TokenDeduplicator class
- [ ] VAC integration
- [ ] Chunk boundary handling
- [ ] Test: test_token_dedup.py
- [ ] **COMMIT CHECKPOINT**: Dedup working

### Phase 6: N-best Rescoring ☐
- [ ] NbestRescorer class
- [ ] Beam hypotheses accessor
- [ ] VAC integration
- [ ] Test: test_nbest_rescoring.py
- [ ] **COMMIT CHECKPOINT**: Rescoring working

### Final Integration ☐
- [ ] Test: test_e2e_code_switching.py
- [ ] Test: test_stress_code_switching.py
- [ ] Performance benchmarks
- [ ] Documentation update
- [ ] **COMMIT CHECKPOINT**: Code-switching production-ready

---

## Success Metrics

**Accuracy:**
- Code-switching WER < 5% degradation vs monolingual
- Language tag accuracy > 95%
- No spurious language switches on transients

**Latency:**
- Sliding LID: < 5ms overhead
- Token dedup: < 2ms overhead
- N-best rescore: < 50ms when triggered
- Overall latency < 150ms (same as current)

**Stability:**
- No KV cache corruption
- No memory leaks in sliding window
- SOT reset frequency < 1 per 5 seconds
- No thrashing on ambiguous audio

**Compatibility:**
- Monolingual performance unchanged
- Existing API contracts maintained
- Backward compatible config defaults

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| SOT reset breaks KV cache | HIGH | Only reset on sustained changes (2.5-3s + silence) |
| Sliding LID adds latency | MEDIUM | Lightweight circular buffer, O(1) operations |
| Token dedup false positives | MEDIUM | Conservative overlap detection, extensive testing |
| Beam rescoring adds latency | LOW | Only triggered on low confidence (<10% of segments) |
| Config complexity | LOW | Good defaults, gradual rollout with feature flags |

---

## Rollout Plan

**Week 1:** Phase 1-2 (Config + VAD)
- Low risk, foundational
- Deploy to staging
- Feature flag: `enable_code_switching=false` (default)

**Week 2:** Phase 3 (Sliding LID)
- Test with real mixed audio
- Deploy to staging
- Feature flag: `enable_code_switching=true` (opt-in)

**Week 3:** Phase 4-5 (Sustained detection + Dedup)
- HIGH RISK - extensive testing
- Feature flag: `enable_sustained_detection=false` (default)
- Gradual rollout

**Week 4:** Phase 6 + Integration (Rescoring + E2E)
- Final integration tests
- Performance benchmarks
- Production deployment
- Feature flag: `enable_nbest_rescoring=false` (default)

---

## File Summary

**New Files (7):**
- `src/sliding_lid_detector.py` (~100 lines)
- `src/token_deduplicator.py` (~80 lines)
- `src/nbest_rescorer.py` (~70 lines)
- `tests/test_config_flow.py` (~150 lines)
- `tests/test_sliding_lid.py` (~200 lines)
- `tests/test_sustained_detection.py` (~250 lines)
- `tests/test_e2e_code_switching.py` (~300 lines)

**Modified Files (8):**
- `src/simul_whisper/config.py` (+10 lines)
- `src/simul_whisper/simul_whisper.py` (+30 lines)
- `src/whisper_service.py` (+15 lines)
- `src/vac_online_processor.py` (+120 lines)
- `src/silero_vad_iterator.py` (+30 lines)
- `src/api_server.py` (+20 lines)
- `modules/orchestration-service/src/socketio_whisper_client.py` (+10 lines)
- `modules/orchestration-service/src/clients/audio_service_client.py` (+8 lines)

**Total:** ~1,363 lines of code

---

## Next Actions

1. ✅ Create this plan document
2. ☐ Git commit checkpoint: "CHECKPOINT: Before code-switching implementation"
3. ☐ Start Phase 1: Complete WebSocket config extraction
4. ☐ Write test_config_flow.py (TDD)
5. ☐ Run test, implement until passing
6. ☐ Commit Phase 1 checkpoint

---

## Notes

- All tests must be INTEGRATED (end-to-end through real services)
- No mocks for audio or transcription - use real Whisper
- Test with REAL audio files (Chinese, English, mixed)
- Each phase has clear success criteria
- Commit checkpoint after each phase passes tests
- Feature flags for gradual rollout
- Think wholistically about end product

**Key Principle**: "Test the behavior users will see, not the implementation"
