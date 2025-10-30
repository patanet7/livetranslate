# Phase 2 Regression Analysis - Code-Switching Broken

**Date**: 2025-10-28
**Status**: 🔴 CRITICAL - Code-switching completely broken
**Root Cause**: 3 major regressions introduced during Phase 2 Day 7-14 refactoring

---

## Executive Summary

Multi-agent analysis comparing **WORKING legacy** (commit 85d2641) vs **BROKEN current** implementation has identified **3 critical regressions** that broke code-switching:

1. **VAC Online Processor**: Frame-based VAD handling breaks audio continuity
2. **SimulWhisper**: Sustained language detection locks decoder to single language
3. **API Server**: Config priority inversion ignores orchestration settings

All three issues must be fixed to restore working code-switching.

---

## Critical Finding #1: VAC Processor Frame-Based VAD Breaks Audio Continuity

### Location
`src/vac_online_processor.py` lines 153, 274-374

### What Changed

**LEGACY (WORKING)** - Simple status-based approach:
```python
# Lines 273-310 - Simple flag updates, continuous audio flow
if vad_result is not None:
    if 'start' in vad_result:
        self.status = 'voice'  # Just update flag
    elif 'end' in vad_result:
        self.status = 'nonvoice'  # Just update flag
else:
    # Send audio continuously during speech
    if self.status == 'voice':
        if len(self.audio_buffer) > 0:
            self._send_audio_to_online_processor(self.audio_buffer)
```

**CURRENT (BROKEN)** - Complex frame-based slicing:
```python
# Lines 274-374 - Added buffer_offset tracking (line 153)
if vad_result is not None:
    frame = list(vad_result.values())[0] - self.buffer_offset

    if 'start' in vad_result:
        send_audio = self.audio_buffer[frame:]  # ❌ Slice at frame boundary
        self._send_audio_to_online_processor(send_audio)
        self.buffer_offset += len(self.audio_buffer)  # ❌ Track offsets
        self.audio_buffer = torch.tensor([])  # ❌ Clear immediately

    elif 'end' in vad_result:
        send_audio = self.audio_buffer[:frame]  # ❌ Slice at frame boundary
        # ... complex buffer management ...
```

### Why This Breaks Code-Switching

1. **Audio fragmentation**: `audio_buffer[frame:]` slicing can split audio **at language boundaries**
2. **Context loss**: Immediate buffer clearing after VAD events loses cross-language context
3. **Frame position errors**: `buffer_offset` calculations misalign audio segments during rapid language switches
4. **No continuity**: VAD-based slicing doesn't preserve natural speech flow for mixed-language utterances

### The Fix

**Revert to legacy approach**:
- Remove `self.buffer_offset = 0` (line 153)
- Replace lines 274-374 with legacy lines 273-310
- Use simple status flags instead of frame-based slicing
- Send audio continuously during `status == 'voice'`, not on VAD events

---

## Critical Finding #2: Sustained Language Detection Breaks Decoder Continuity

### Location
`src/simul_whisper/simul_whisper.py` lines 135, 459-460, 483-540

### What Changed

**LEGACY (WORKING)** - Immediate tracking, decoder stays rolling:
```python
# Line 475-478 - Simple update, NO state reset
elif getattr(self, 'enable_code_switching', False):
    # Update language but DON'T reset decoder
    self.detected_language = top_lan
    logger.info(f"[CODE-SWITCHING] Language tracked as {top_lan} but decoder unchanged (no cache flush)")
```

**CURRENT (BROKEN)** - Sustained detection with full decoder reset:
```python
# Line 135 - Added sustained detection tracking
self.language_history = []  # ❌ NEW: Track language history

# Lines 459-460 - Added pre-detection cache clearing
if getattr(self, 'enable_code_switching', False) and self.detected_language is not None:
    self._clean_cache()  # ❌ Clear cache before every detection

# Lines 483-540 - Sustained detection logic
elif getattr(self, 'enable_code_switching', False):
    current_time = time.time()
    self.language_history.append((current_time, top_lan, p))

    # Keep 2.5 second sliding window
    cutoff_time = current_time - 2.5
    self.language_history = [(t, l, c) for (t, l, c) in self.language_history if t > cutoff_time]

    # Wait for 3+ consecutive chunks (≈3.6 seconds)
    if len(self.language_history) >= 3:
        recent_langs = [l for (t, l, c) in self.language_history[-3:]]

        if len(set(recent_langs)) == 1:  # All same language
            sustained_lang = recent_langs[0]

            if sustained_lang != self.detected_language:
                # ❌ FULL DECODER RESET
                self._clean_cache()           # Clear KV cache
                self.dec_attns = []           # Clear attention
                self.segments = []            # Clear audio buffer
                self.create_tokenizer(sustained_lang)
                self.detected_language = sustained_lang
                self.init_tokens()            # Reset SOT tokens
                self.init_context()           # Reset context
                self.language_history = []
```

### Why This Breaks Code-Switching

1. **3.6 second delay**: Requires 3 consecutive chunks before recognizing language switch
2. **Decoder locked**: During the delay, `self.detected_language` stays locked to old language
3. **Token tagging wrong**: All tokens tagged with old language even when speaking new language
4. **Aggressive state clearing**: When switch finally happens, `self.segments = []` destroys audio context
5. **Designed for Chinglish filtering**: Meant to prevent single-word false positives, but breaks true code-switching

### Log Evidence

From user's logs showing the problem:
```
INFO:simul_whisper.simul_whisper:[TOKEN-DEBUG] Generated token #26: ID=5289, Text='And', Lang=zh, SOT=zh
INFO:simul_whisper.simul_whisper:[TOKEN-DEBUG] Generated token #27: ID=370, Text=' so', Lang=zh, SOT=zh
INFO:simul_whisper.simul_whisper:[TOKEN-DEBUG] Generated token #29: ID=7177, Text=' fellow', Lang=zh, SOT=zh
INFO:simul_whisper.simul_whisper:[TOKEN-DEBUG] Generated token #30: ID=6280, Text=' Americans', Lang=zh, SOT=zh
```

**English words ALL marked as `Lang=zh, SOT=zh`** because sustained detection keeps old language!

### The Fix

**Revert to legacy approach**:
- Remove `self.language_history = []` initialization (line 135)
- Keep cache pre-cleaning (lines 459-460) - this prevents KV dimension mismatches
- Replace lines 483-540 with legacy simple tracking (4 lines)
- No decoder resets, let Whisper's multilingual tokenizer handle mixing naturally

---

## Critical Finding #3: API Server Config Priority Inversion

### Location
`src/api_server.py` lines 2014-2027 (join_session), 2166-2168 (transcribe_stream)

### What Changed

**LEGACY (WORKING)** - Always use fresh config from orchestration:
```python
# Line 2150 in handle_transcribe_stream
config = data.get('config', {})  # ✅ Always fresh from orchestration
```

**CURRENT (BROKEN)** - Prioritize stale session storage:
```python
# Lines 2166-2168 in handle_transcribe_stream
session_id = data.get('session_id')
config = streaming_sessions.get(session_id, {}) if session_id else data.get('config', {})
# ❌ Uses STALE config from join_session, ignores fresh data from orchestration
```

### Why This Breaks Code-Switching

**The Bug Flow**:

1. **Join Session** (orchestration doesn't send config here):
   ```python
   streaming_sessions[session_id] = {
       "enable_code_switching": config.get('enable_code_switching', False),  # ❌ Defaults to False
       ...
   }
   ```

2. **First Audio Chunk** (orchestration DOES send config here):
   ```python
   # Orchestration sends: data = {"session_id": "...", "config": {"enable_code_switching": true}}
   # CURRENT: config = streaming_sessions[session_id]  # ❌ Gets False from storage
   # LEGACY:  config = data.get('config', {})          # ✅ Gets True from orchestration
   ```

3. **VAC Processor Creation**:
   ```python
   transcription_request = TranscriptionRequest(
       enable_code_switching=config.get('enable_code_switching', False)  # ❌ Gets False!
   )
   ```

4. **Result**: `stateful_whisper.set_task(enable_code_switching=False)` - Code-switching disabled!

### The Fix

**Reverse priority** - Fresh data should override stale storage:
```python
# Change line 2166-2168 from:
config = streaming_sessions.get(session_id, {}) if session_id else data.get('config', {})

# To:
config = data.get('config', streaming_sessions.get(session_id, {})) if session_id else data.get('config', {})
```

This prioritizes fresh orchestration config while falling back to stored session config.

---

## Test Evidence

### Test Bug (Already Fixed)
The test was sending config in `transcribe_stream` instead of `join_session`:
```python
# OLD (WRONG):
sio.emit('join_session', {'session_id': session_id})  # ❌ No config
sio.emit('transcribe_stream', {
    'enable_code_switching': True  # ❌ Too late, session already created
})

# FIXED:
sio.emit('join_session', {
    'session_id': session_id,
    'config': {'enable_code_switching': True}  # ✅ Config during join
})
```

### But Config Still Not Working!
Even with fixed test, logs show:
```
INFO:simul_whisper.simul_whisper:Detected language: zh with p=0.9874
```
**English audio detected as Chinese!** This confirms sustained detection is locking language.

---

## Additional Issue: VAC Processor Caching

### Problem
VAC processors are cached per session_id:
```python
if session_id not in vac_processors:
    # Create NEW processor with current config
else:
    # Reuse OLD processor with OLD config  # ❌ Bug!
```

**Impact**: Running the test multiple times with same session pattern reuses OLD processor with code-switching disabled.

**Workaround**: Restart Whisper service between tests to clear cache.

**Proper Fix**: Session cleanup should remove VAC processor when session ends.

---

## Files Requiring Changes

### 1. `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/src/vac_online_processor.py`
- **Line 153**: Remove `self.buffer_offset = 0`
- **Lines 274-374**: Replace with legacy lines 273-310 (simple status-based VAD)

### 2. `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/src/simul_whisper/simul_whisper.py`
- **Line 135**: Remove `self.language_history = []`
- **Lines 459-460**: KEEP pre-detection cache clearing (prevents KV dimension mismatch)
- **Lines 483-540**: Replace with legacy lines 475-478 (simple immediate tracking)

### 3. `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/src/api_server.py`
- **Lines 2166-2168**: Reverse config priority to favor fresh data over stale storage

### 4. `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/tests/integration/test_streaming_code_switching.py`
- **Already Fixed**: Now sends config in `join_session` (lines 117-129)
- **Improvement Needed**: Use random session IDs to avoid cache reuse

---

## Verification Steps

After applying fixes:

1. **Restart Whisper service** to clear VAC processor cache
2. **Run test**: `python tests/integration/test_streaming_code_switching.py`
3. **Check logs for**:
   - `[VAC] Code-switching enabled → Using task='transcribe'` (should appear)
   - `[CODE-SWITCHING] Language tracked as en but decoder unchanged` (should appear)
   - `[TOKEN-DEBUG] ... Lang=en, SOT=zh` (should see BOTH languages)
4. **Verify transcription**:
   - Chinese portions transcribed in Chinese
   - English portions transcribed in English
   - No "Lang=zh" for English words
   - No 3.6 second delay in language detection

---

## Root Cause Analysis

### Why Phase 2 Broke This

**Intended Goal**: Improve audio processing and fix 15.75s accumulation bug

**What Actually Happened**:
1. **Frame-based VAD**: Added to fix audio accumulation, but broke audio continuity for language mixing
2. **Sustained detection**: Added to filter Chinglish single-word switches, but prevented legitimate code-switching
3. **Config storage**: Added to avoid re-sending config, but created priority inversion bug

**The Problem**: Over-engineering. Simple working code was replaced with complex logic that solved different problems but broke the original functionality.

### Quote from User
> "ULTRATHINK I think when you extracted from monolithic files you just made a huge mess... look through whisper/src and tell me what the hell is going on"

**Analysis**: The Phase 2 extraction created **70+ files** from a **2392-line monolithic file**, but in doing so:
- Created duplicate implementations (buffer_manager, eow_detection)
- Scattered related logic across multiple files
- Introduced subtle bugs through abstraction layers
- Made the system harder to understand and debug

---

## Lessons Learned

### What Worked (Legacy)
- Monolithic but clear organization
- Direct configuration passing
- Simple status-based logic
- Easy to trace execution flow
- Code-switching worked perfectly

### What Broke (Phase 2)
- Premature file splitting without proper testing
- Complex abstractions that obscured data flow
- "Clever" solutions to problems that didn't exist
- Lost sight of core functionality during refactoring

### Moving Forward
1. **Test before refactoring**: Run full test suite before/after extractions
2. **Preserve behavior**: Extract without changing logic
3. **One change at a time**: Don't fix bugs during refactoring
4. **Verify regressions**: Compare line-by-line with working implementation

---

## Appendix: Legacy Files Location

All working implementations saved to:
```
/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/legacy/
├── README.md                          # This analysis
├── api_server_WORKING.py              # 3642 lines, commit 85d2641
├── whisper_service_WORKING.py         # 2392 lines, commit 85d2641
├── vac_online_processor_WORKING.py    # 896 lines, commit 85d2641
├── simul_whisper_WORKING.py          # 33K, commit 85d2641
├── api_server_before_phase2.py        # 3114 lines, commit 20a4c1c
└── whisper_service_before_phase2.py   # 1250 lines, commit 20a4c1c
```

Use these for reference when fixing regressions.
