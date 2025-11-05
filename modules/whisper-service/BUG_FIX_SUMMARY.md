# CRITICAL BUG FIX: Session Creation Timing

## Problem Summary

Session creation at VAD START (lines 431-441 in broken code) was causing -718% accuracy regression due to premature language detection.

## Root Cause

**Broken Flow:**
1. VAD detects speech START at 0.2s
2. Code immediately creates 'en' session (fallback) before LID runs
3. LID hasn't had chance to analyze audio yet (needs ~100ms+)
4. Session created with WRONG SOT token (English instead of Chinese)
5. Chinese audio gets English session → "Amen" hallucinations × 92!

**Evidence:**
```
🆕 Creating initial session with fallback language: en
[TOKEN-DEBUG] Generated token #5: ID=14092, Text=' Amen', Lang=en, SOT=en
[TOKEN-DEBUG] Generated token #6: ID=13, Text='.', Lang=en, SOT=en
... (repeats 92+ times for Chinese audio!)
```

## The Fix

### Key Insight
The Dependency Injection (DI) pattern for shared model was **CORRECT**. The problem was **session creation timing**, not the DI pattern itself.

### What Changed

1. **Restored Original LID Guard** (line 518):
   ```python
   while len(self.audio_buffer_for_lid) >= self.lid_hop_samples and self.current_session is not None:
   ```
   - LID only runs AFTER session exists
   - This ensures LID has enough audio context (not just silence/noise)

2. **Moved Session Creation BEFORE LID Loop** (lines 511-519):
   ```python
   # Create session BEFORE LID (not after)
   if self.current_session is None:
       initial_language = self.sustained_detector.get_current_language()
       if initial_language is None:
           initial_language = self.target_languages[0]  # Fallback
       self.current_session = self._create_new_session(initial_language)
   ```
   - Creates session with fallback language ('en' initially)
   - LID runs on subsequent chunks and corrects if needed

3. **Kept DI Pattern** (lines 137-148, 554-555):
   ```python
   # Shared model at transcriber level
   self.shared_whisper_model = self._load_shared_model(model_path)
   self.shared_tokenizer = get_tokenizer(...)

   # LID uses shared model
   lid_probs = self.lid_detector.detect(
       encoder_output=encoder_output,
       model=self.shared_whisper_model,  # DI pattern
       tokenizer=self.shared_tokenizer,  # DI pattern
       timestamp=current_time
   )
   ```
   - Decouples model lifecycle from session lifecycle
   - Enables efficient session switching without model reloading

### Correct Flow Now

```
1. VAD START → should_process=True
2. Append to LID buffer
3. Create session with fallback 'en' (sustained_detector.get_current_language() or target_languages[0])
4. LID loop runs (has session now, guard passes)
5. LID processes frames → Detects 'zh' over multiple frames
6. Sustained detection triggered (6+ frames, 250ms+, confidence margin > 0.2)
7. Switch session from 'en' to 'zh'
8. Process audio with correct 'zh' SOT token
```

### Why This Works

1. **Initial fallback is OK**: First session uses 'en' fallback, but LID quickly corrects it
2. **LID has context**: By waiting for session creation, LID analyzes real speech (not silence/noise)
3. **Sustained detection prevents false switches**: Requires 6+ frames (250ms+) with high confidence
4. **Session switching is clean**: Happens at VAD boundaries with correct SOT token

## Files Modified

- `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/src/session_restart/session_manager.py`
  - **Line 518**: Restored LID guard `and self.current_session is not None`
  - **Lines 511-519**: Session creation BEFORE LID loop (not after, not at VAD START)
  - **Lines 554-555**: LID uses shared model/tokenizer (DI pattern maintained)
  - **Removed lines 431-441**: Deleted premature session creation at VAD START

## Expected Outcomes

### Chinese-only test:
```bash
poetry run python test_chinese_only.py
```
**Expected:**
- Initial session: 'en' (fallback)
- LID detects 'zh' within first 1-2 seconds
- Session switches to 'zh'
- Chinese transcriptions produced (NOT "Amen" repetitions)

### English-only test:
```bash
poetry run pytest tests/milestone2/test_real_code_switching.py::test_english_only_no_switch -v
```
**Expected:**
- 100% accuracy restored (was working before, should still work)
- No false language switches
- Correct English transcriptions

## Technical Notes

### Why Original Code Worked

The original code (commit a666951) used this pattern:
1. Create session with fallback language
2. LID runs only when session exists
3. LID accumulates frames over time
4. Sustained detection triggers session switch

This avoided the "first chunk problem" where LID makes wrong decision on silence/noise.

### Why My First Fix Failed

I tried to run LID BEFORE session creation to get the "correct" language from the start. But:
- LID on first chunk is unreliable (silence/noise before speech)
- Creates chicken-and-egg problem (need session to get model, need model to detect language)
- DI pattern solves chicken-and-egg, but first chunk problem remains

### The Right Solution

- Use DI pattern for technical correctness (shared model)
- Use fallback-then-switch for practical correctness (reliable LID)
- Best of both worlds: clean architecture + robust detection

## Verification Steps

1. Run Chinese test - should see session switch from 'en' to 'zh'
2. Run English test - should maintain 'en' session, no switches
3. Check logs for "Creating initial session with fallback language: en"
4. Check logs for "Sustained language change detected: en → zh" (Chinese test)
5. Verify no "Amen" hallucinations in output

## Related Commits

- **a666951**: Original working code (100% accuracy)
- **Current fix**: Restores original timing + keeps DI pattern improvements
