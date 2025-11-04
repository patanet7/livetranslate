# Code-Switching Accuracy Validation Report
**Date**: 2025-11-03
**System**: Session-Restart Code-Switching (Milestone 2)
**Target Accuracy**: 70-85% (FEEDBACK.md line 184)

---

## Executive Summary

**Status**: ❌ **CRITICAL BUG IDENTIFIED - NOT PRODUCTION READY**

The code-switching system has a **critical logic error** preventing language detection from running. While the English-only baseline achieves perfect 100% accuracy, code-switching scenarios fail catastrophically with -322% accuracy due to this bug.

### Quick Metrics

| Test Case | Expected | Actual | Status |
|-----------|----------|--------|--------|
| **English-Only (JFK)** | ≥75% | **100.0%** | ✅ PASS |
| **Mixed EN→ZH** | 70-85% | **-322.7%** | ❌ FAIL |
| **Separate Files** | 70-85% | **N/A** | ❌ FAIL |
| **Language Switching** | Detect EN↔ZH | **0 switches** | ❌ FAIL |

---

## Test Environment

```
Working Directory: /Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service
Python: 3.12.4
Platform: macOS-15.6.1-arm64-arm-64bit (Apple Silicon)
Device: MPS (Metal Performance Shaders)
Model: whisper-large-v3-turbo
Sample Rate: 16000 Hz
```

**Test Files**:
- `tests/fixtures/audio/jfk.wav` - JFK speech (11s, English)
- `tests/fixtures/audio/OSR_cn_000_0072_8k.wav` - Chinese speech (20s)
- `tests/fixtures/audio/test_clean_mixed_en_zh.wav` - Mixed EN/ZH (67s)

---

## Test 1: English-Only (Baseline Quality)

**Purpose**: Validate that monolingual English transcription works correctly with no false language switches.

### Results

```
Test Duration: 26.45s
Transcription Sessions: 4
Language Switches: 0 (correct - monolingual)
English WER (normalized): 0.0%
English Accuracy: 100.0% ✅
```

### Ground Truth vs. Transcription

**Expected**:
```
And so my fellow Americans ask not what your country can do for you ask what you can do for your country
```

**Actual** (normalized):
```
and so my fellow americans ask not what your country can do for you ask what you can do for your country
```

### Analysis

✅ **PERFECT MATCH** - Zero word errors after normalization
✅ **No false language switches** - Stayed in English throughout
✅ **VAD-first processing working** - Clean segment boundaries
✅ **Session restart working** - Multiple sessions handled correctly

**Verdict**: English baseline is **production-ready** with excellent accuracy.

---

## Test 2: Mixed Language Code-Switching

**Purpose**: Test real code-switching with mixed English/Chinese audio file.

### Results

```
Test Duration: 137.31s (2:17)
Audio Duration: 66.87s
Transcription Sessions: 18
Language Switches Detected: 0 ❌
Expected Language Switches: Multiple (EN→ZH transitions)
```

### Accuracy Metrics

| Language | Expected Segments | Detected Segments | Accuracy | Status |
|----------|------------------|-------------------|----------|--------|
| **English** | 1 (11s) | 18 (all audio) | **-322.7%** | ❌ |
| **Chinese** | 3 (56s) | 0 | **N/A** | ❌ |
| **Overall** | 70-85% | **-161.4%** | ❌ |

### What Happened

The system transcribed the **entire 67-second audio file as English**, including Chinese segments:

**English Ground Truth** (0-11s):
```
And so my fellow Americans ask not what your country can do for you ask what you can do for your country
```

**Actual Transcription** (included Chinese as English gibberish):
```
And so, my fellow Americans, Ask not! What your country can do for you, ask what you can do for your country.
The entrance door is a bus station. ❌
This is a beautiful and amazing景象 ❌
The tree has grown up with a small and sweet桃. ❌
海豚和鲸鱼的表演是很好看的节目 ❌
The car was in the car. ❌
[... continues with hallucinations ...]
```

**Error Analysis**:
- 93 insertions (hallucinated English words for Chinese audio)
- 0 substitutions
- 0 deletions
- Chinese segments completely lost (transcribed as nonsense English)

**Verdict**: Code-switching is **completely non-functional** due to critical bug.

---

## Test 3: Separate Language Files

**Purpose**: Test explicit EN→ZH transition with separate audio files.

### Results

```
Test Duration: 69.57s (1:09)
Audio Duration: 31.97s (EN: 11s, silence: 1s, ZH: 20s)
Language Switches Expected: 1 (EN→ZH at ~12s)
Language Switches Detected: 1 (MANUAL switch logged)
Automatic Detection: ❌ None
```

### What Happened

Even with **explicit separation** (English file → 1s silence → Chinese file), the system:
- Transcribed English correctly
- Transcribed Chinese as English gibberish
- Did NOT automatically detect the language switch
- Required manual intervention (test code manually switched at 12s)

**Verdict**: Automatic language detection **not working**.

---

## Root Cause Analysis

### The Critical Bug

**Location**: `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/src/session_restart/session_manager.py:456`

**Bug**: Session creation happens AFTER LID check, causing LID to never run.

```python
# Line 450: Add audio to LID buffer
self.audio_buffer_for_lid.append(audio_chunk)

# Line 456: Run LID ONLY if current_session exists
while len(self.audio_buffer_for_lid) >= self.lid_hop_samples and self.current_session is not None:
    # LID code here...
    lid_probs = self.lid_detector.detect(...)  # ❌ NEVER RUNS!

# Line 538-543: Create session if None (AFTER LID check!)
if self.current_session is None:
    initial_language = self.sustained_detector.get_current_language()  # Returns None!
    if initial_language is None:
        initial_language = self.target_languages[0]  # Defaults to 'en'
    self.current_session = self._create_new_session(initial_language)  # Always 'en'
```

### Execution Flow (Broken)

1. **VAD END event** → Current session saved and set to `None`
2. **Next chunk arrives** → Audio buffered for LID (line 450)
3. **LID check** → Fails because `current_session is None` (line 456) ❌
4. **Session creation** → Uses `sustained_detector.get_current_language()` which returns `None`
5. **Default language** → Defaults to `'en'` (line 542)
6. **Result** → All audio transcribed as English, even Chinese segments

### Why It Wasn't Caught Earlier

1. **Unit tests** focused on LID components in isolation (all passing)
2. **Integration tests** didn't verify language field in output
3. **Test logging** showed `[None]` for language but tests still passed
4. **English-only** worked perfectly (default 'en' is correct)

### Evidence from Logs

**Expected**:
```
INFO:language_id.sustained_detector:✅ Initial language set: en (p=0.850)
INFO:session_restart.session_manager:🔍 Sustained language change detected: en → zh
```

**Actual**:
```
INFO:session_restart.session_manager:🆕 Creating new session for language: en
INFO:session_restart.session_manager:🆕 Creating new session for language: en
INFO:session_restart.session_manager:🆕 Creating new session for language: en
... (all sessions in 'en', zero LID logs)
```

**No LID logs = LID never ran!**

---

## Performance Analysis

### What Worked Well

✅ **VAD Processing**: Clean speech/silence detection
✅ **Session Management**: Multiple sessions handled correctly
✅ **Encoder Caching**: Infrastructure in place (unused due to bug)
✅ **Metrics Tracking**: Performance monitoring working
✅ **English Baseline**: 100% accuracy on monolingual audio

### What Didn't Work

❌ **Language Detection**: Never runs (critical bug)
❌ **Code-Switching**: 0 automatic switches detected
❌ **Chinese Transcription**: Completely lost (transcribed as English)
❌ **Accuracy**: -322% (massive hallucinations from wrong language)

### Latency (From Working English-Only Test)

```
Average per-chunk latency: ~2-3s (model loading + inference)
Total processing time: 26.45s for 11s audio = 2.4x real-time
```

**Note**: Real-time performance requires optimization, but accuracy must be fixed first.

---

## The Fix

### Required Change

**Move session creation BEFORE LID check**:

```python
# BEFORE (BROKEN):
# 1. Add to LID buffer
self.audio_buffer_for_lid.append(audio_chunk)
# 2. Run LID (fails if no session)
while len(self.audio_buffer_for_lid) >= self.lid_hop_samples and self.current_session is not None:
    # LID never runs!
# 3. Create session (after LID)
if self.current_session is None:
    self.current_session = self._create_new_session(initial_language)

# AFTER (FIXED):
# 1. Add to LID buffer
self.audio_buffer_for_lid.append(audio_chunk)
# 2. Create session FIRST (before LID)
if self.current_session is None:
    # Start with default language, LID will switch if needed
    self.current_session = self._create_new_session(self.target_languages[0])
# 3. Run LID (now session exists!)
while len(self.audio_buffer_for_lid) >= self.lid_hop_samples:
    # LID runs and can detect language changes!
    lid_probs = self.lid_detector.detect(...)
    switch_event = self.sustained_detector.update(lid_probs, ...)
    if switch_event:
        self._switch_session(switch_event.to_language)
```

### Testing After Fix

1. Re-run `test_mixed_language_transcription` → Should detect EN→ZH switches
2. Verify Chinese CER < 30% (target: ≥70% accuracy)
3. Verify overall accuracy 70-85%
4. Confirm no false positives on English-only test

---

## Recommendations

### Immediate Actions (Critical)

1. ✅ **Fix session/LID ordering** (session_manager.py:456)
2. ✅ **Re-run integration tests** with fixed code
3. ✅ **Add explicit LID logging check** to test assertions
4. ✅ **Verify Chinese transcription quality** after fix

### Testing Improvements

1. **Add language field validation** to all integration tests
2. **Check for LID log messages** in test assertions
3. **Test with real code-switching audio** (already done, good!)
4. **Add performance regression tests** (latency, memory)

### Code Quality

1. **Add defensive assertions** (e.g., `assert current_session is not None before LID`)
2. **Improve logging** (warn if LID skipped unexpectedly)
3. **Add unit test** specifically for session creation timing
4. **Document** the session/LID interaction in comments

---

## Conclusion

### Current Status

**NOT PRODUCTION READY** - Critical bug prevents language detection from running.

### Impact Assessment

| Component | Status | Impact |
|-----------|--------|--------|
| English Transcription | ✅ Working | 100% accuracy |
| VAD Processing | ✅ Working | Clean boundaries |
| Session Management | ✅ Working | Multiple sessions OK |
| **Language Detection** | ❌ **BROKEN** | **Zero switches detected** |
| **Code-Switching** | ❌ **BROKEN** | **-322% accuracy** |
| Performance Optimization | ⚠️ Untested | Infrastructure ready |

### Path to Production

1. **Fix critical bug** (1-2 hours)
2. **Re-run all tests** (30 minutes)
3. **Verify accuracy ≥70%** (Chinese segments)
4. **Performance tuning** (if needed)
5. **Final validation** with diverse audio samples

### Estimated Timeline

- **Bug fix + testing**: 2-3 hours
- **Performance optimization**: 4-6 hours (if needed)
- **Full validation**: 1-2 hours

**Total**: 1 day to production-ready code-switching.

---

## Detailed Test Logs

### Test Execution Commands

```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service

# Test 1: English-only
poetry run pytest tests/milestone2/test_real_code_switching.py::test_english_only_no_switch -v -s

# Test 2: Mixed language
poetry run pytest tests/milestone2/test_real_code_switching.py::test_mixed_language_transcription -v -s

# Test 3: Separate files
poetry run pytest tests/milestone2/test_real_code_switching.py::test_separate_language_files -v -s
```

### Key Log Evidence

**No LID initialization logs**:
```bash
$ grep "Initial language set" /tmp/test_output.log
# (empty - LID never initialized)
```

**No language switch logs**:
```bash
$ grep "Sustained language change detected" /tmp/test_output.log
# (empty - no switches detected)
```

**All sessions in English**:
```bash
$ grep "Creating new session for language:" /tmp/test_output.log
INFO:session_restart.session_manager:🆕 Creating new session for language: en
INFO:session_restart.session_manager:🆕 Creating new session for language: en
INFO:session_restart.session_manager:🆕 Creating new session for language: en
... (all 'en', zero 'zh')
```

---

## Appendix: Full Test Results

### Test 1: English-Only (PASSED)

```
Test Duration: 26.45s
Sessions: 4
Switches: 0 (correct)
WER (normalized): 0.0%
Accuracy: 100.0%
Status: ✅ PASS
```

### Test 2: Mixed Language (FAILED)

```
Test Duration: 137.31s
Audio: 66.87s (EN: 11s, ZH: 56s)
Sessions: 18 (all 'en')
Switches: 0 (expected: multiple)
English WER: 422.7% (hallucinations)
Chinese CER: N/A (not detected)
Overall Accuracy: -161.4%
Status: ❌ FAIL
```

### Test 3: Separate Files (FAILED)

```
Test Duration: 69.57s
Audio: 31.97s (EN: 11s, silence: 1s, ZH: 20s)
Sessions: 8 (all 'en')
Switches: 0 automatic (1 manual)
Status: ❌ FAIL
```

---

**Report Generated**: 2025-11-03
**Engineer**: ML Engineer (Claude Code)
**Next Steps**: Fix session/LID ordering bug and re-validate
