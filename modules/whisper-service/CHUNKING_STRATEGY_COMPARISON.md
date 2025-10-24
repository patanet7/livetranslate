# Chunking Strategy Comparison: Our Implementation vs SimulStreaming vs vexa

## TL;DR - The Critical Difference

| System | Client Sends | Server Processes | Key Strategy |
|--------|-------------|------------------|--------------|
| **Our Implementation** | 1.0s chunks | **WAITS FOR VAD SILENCE** → Can accumulate 9s+ | ❌ **PROBLEM**: Mixed languages in one chunk |
| **SimulStreaming** | 0.04s chunks | **Every 1.2s REGARDLESS of VAD** | ✅ Fixed interval processing |
| **vexa/WhisperLive** | Small chunks | **Every 0.4s minimum** | ✅ Frequent processing |

---

## 1. Our Implementation (PROBLEMATIC)

### Client → Server
```python
# Test sends: 1.0 second chunks
chunk_size = 16000 samples = 1.0 second
send_interval = 200ms between sends
```

### Server Processing Strategy
```python
# vac_online_processor.py
online_chunk_size = 1.2s  # Target size

def process_iter(self):
    if self.is_currently_final:
        # ❌ WAITS for VAD silence before processing!
        return self._finish()

    elif len(self.audio_buffer) >= self.SAMPLING_RATE * self.online_chunk_size:
        return self._process_online_chunk()
    else:
        return {}  # Keep buffering
```

### What Actually Happens (Code-Switching Test):
```
Timeline:
t=0s:   Client sends chunk 1 (1s EN audio)
t=0.2s: Client sends chunk 2 (1s EN audio)
t=0.4s: Client sends chunk 3 (1s ZH audio)  ← Language switch!
t=0.6s: Client sends chunk 4 (1s EN audio)  ← Switch again!
t=0.8s: Client sends chunk 5 (1s ZH audio)
... continues ...
t=2.0s: VAD processes first 2s → Result #1: "And so, my fellow Americans..." (en) ✅

t=2.2s: Client sends more chunks...
t=11.0s: All 11 chunks sent
t=11.0s: VAD detects silence → Processes 9s accumulated audio!
         ❌ Audio contains: 1s ZH + 1s EN + 3s ZH + 4s EN MIXED
         ❌ Whisper detects 'zh' from audio features
         ❌ But transcribes English text!
         → Result #2: "ask not what your country..." (labeled 'zh' but text is 'en')
```

**THE PROBLEM**: We wait for VAD silence, so we buffer **9 seconds of mixed-language audio** and process it all at once!

---

## 2. SimulStreaming (CORRECT)

### Client → Server
```python
# Sends TINY chunks frequently
chunk_size = 0.04 seconds (640 samples @ 16kHz)
# Very frequent sending
```

### Server Processing Strategy
```python
# whisper_streaming/vac_online_processor.py (lines 96-103)

def process_iter(self):
    if self.is_currently_final:
        return self.finish()  # VAD silence

    # ✅ KEY: Process every 1.2s EVEN IF SPEECH CONTINUING!
    elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE*self.online_chunk_size:
        self.current_online_chunk_buffer_size = 0
        return self.online.process_iter()  # ← Regular processing!
    else:
        return None  # Keep buffering
```

###  SimulStreaming 30s Buffer Confusion - EXPLAINED

**You asked**: "Does SimulStreaming build a 30s buffer and run the whole thing?"

**Answer**: NO! The 30s is the **MAXIMUM** buffer size (`audio_max_len`), but it processes **every 1.2s**!

```python
# simulstreaming_whisper.py line 22-23
audio_max_len = 30.0s  # MAX buffer (prevents OOM)
segment_length = 1.2s  # Actual processing interval

# How it works:
- Client sends tiny 0.04s chunks continuously
- Server accumulates in rolling buffer (max 30s)
- Server processes every 1.2s regardless of VAD
- After processing, OLD audio is removed from buffer
- Buffer slides forward like a rolling window
```

**Segmentation Strategy**:
```
Buffer: [----------------- 30s MAX -----------------]
Process:     ↓1.2s      ↓1.2s      ↓1.2s     ↓1.2s
          Result#1   Result#2   Result#3   Result#4

Each result uses last 1.2-2s of audio (overlapping for context)
Old audio removed after processing
```

### What Happens with Mixed Languages:
```
Timeline:
t=0-1.2s:   Accumulate EN audio → Process → Result: "And so" (en) ✅
t=1.2-2.4s: Accumulate ZH audio → Process → Result: "院子门口" (zh) ✅
t=2.4-3.6s: Accumulate EN audio → Process → Result: "my fellow" (en) ✅
t=3.6-4.8s: Accumulate ZH audio → Process → Result: "不远就是" (zh) ✅

✅ Each 1.2s chunk is mostly ONE language
✅ Whisper's language detection matches actual text
```

---

## 3. vexa/WhisperLive (MOST AGGRESSIVE)

### Client → Server
```python
# Sends frequent small chunks
# Exact size varies, but very frequent
```

### Server Processing Strategy
```python
# whisper_live/server.py line 2143-2145

input_bytes, duration = self.get_audio_chunk_for_processing()
if duration < 0.4:
    continue  # Wait for more

# ✅ Process every 0.4s minimum!
self.transcribe_audio(input_sample)
```

### Key Features:
- **Minimum chunk**: 0.4 seconds
- **Frequent processing**: Even faster than SimulStreaming
- **Timestamp-based**: Uses absolute ISO timestamps for deduplication
- **No VAD waiting**: Processes based on time, not silence

### What Happens with Mixed Languages:
```
Timeline:
t=0-0.4s:   EN audio → Process → Result #1
t=0.4-0.8s: EN audio → Process → Result #2
t=0.8-1.2s: ZH audio → Process → Result #3  ← Clean switch
t=1.2-1.6s: EN audio → Process → Result #4  ← Clean switch
t=1.6-2.0s: ZH audio → Process → Result #5  ← Clean switch

✅ Each 0.4s chunk is VERY LIKELY one language
✅ Near-perfect language labels
```

---

## Comparison Table

| Aspect | Our Implementation | SimulStreaming | vexa/WhisperLive |
|--------|-------------------|----------------|------------------|
| **Client chunk size** | 1.0s | 0.04s | Variable (small) |
| **Processing trigger** | VAD silence | Every 1.2s | Every 0.4s |
| **Waits for VAD?** | ✅ YES | ❌ NO | ❌ NO |
| **Can mix languages?** | ❌ YES (9s+) | ✅ Minimal (1.2s max) | ✅ Minimal (0.4s max) |
| **Code-switching ready?** | ❌ NO | ✅ YES | ✅ YES |
| **Latency** | High (waits for silence) | Medium (1.2s) | Low (0.4s) |
| **Compute efficiency** | High (fewer calls) | Medium | Lower (more calls) |

---

## Why Our Approach Fails for Code-Switching

### Example: JFK + Chinese Mixed Stream

**Client sends** (our test):
```
Chunk 1:  1s EN audio
Chunk 2:  1s EN audio
Chunk 3:  1s ZH audio  ← Language switch!
Chunk 4:  1s EN audio  ← Switch back!
Chunk 5:  1s ZH audio  ← Switch again!
Chunk 6:  1s ZH audio
Chunk 7:  1s ZH audio
Chunk 8:  1s EN audio
Chunk 9:  1s EN audio
Chunk 10: 1s EN audio
Chunk 11: 1s EN audio
```

**Our server processes**:
```
First 2s: ✅ Process → "And so, my fellow Americans" (en) CORRECT

VAD continues during chunks 3-11 (no silence detected)
Buffer accumulates: 1 ZH + 1 EN + 3 ZH + 4 EN = 9s MIXED

When silence finally detected at end:
❌ Process 9s mixed audio
❌ Whisper language detection: 'zh' (from audio features)
❌ Transcription: "ask not what your country..." (English text!)
❌ MISMATCH: Language='zh', Text='en'
```

**SimulStreaming would process**:
```
0-1.2s:   "And so, my" (en) ✅
1.2-2.4s: "fellow Americans 院" (mixed but short) ⚠️
2.4-3.6s: "子门口不" (zh) ✅
3.6-4.8s: "远 ask not" (mixed but short) ⚠️
4.8-6.0s: "what 就是" (mixed but short) ⚠️
6.0-7.2s: "一个地铁站" (zh) ✅
7.2-8.4s: "your country" (en) ✅
8.4-9.6s: "can do for" (en) ✅

Much better! Shorter chunks = less mixing
```

**vexa would process**:
```
0-0.4s:   "And" (en) ✅
0.4-0.8s: "so, my" (en) ✅
0.8-1.2s: "fellow" (en) ✅
1.2-1.6s: "Americans" (en) ✅
1.6-2.0s: "院子" (zh) ✅  ← Clean switch!
2.0-2.4s: "门口" (zh) ✅
2.4-2.8s: "不远" (zh) ✅
2.8-3.2s: "ask" (en) ✅  ← Clean switch!
3.2-3.6s: "not what" (en) ✅

Nearly perfect! Very short chunks = almost no mixing
```

---

## The Solution

### Option A: Match SimulStreaming (Recommended)

**Change our `process_iter()`**:
```python
def process_iter(self) -> Dict[str, Any]:
    # Process when buffer full, REGARDLESS of VAD status
    if len(self.audio_buffer) >= self.SAMPLING_RATE * self.online_chunk_size:
        self._send_audio_to_online_processor(self.audio_buffer)
        self.audio_buffer = torch.tensor([], dtype=torch.float32)
        return self._process_online_chunk()

    # ALSO process on VAD silence (for clean boundaries)
    elif self.is_currently_final:
        if len(self.audio_buffer) > 0:
            self._send_audio_to_online_processor(self.audio_buffer)
            self.audio_buffer = torch.tensor([], dtype=torch.float32)
        return self._finish()

    else:
        return {}  # Keep buffering
```

**Effect**: Process every 1.2s even if speech continuing → Max 1.2s of mixed audio

### Option B: Reduce Chunk Size

```python
online_chunk_size = 0.5s  # Instead of 1.2s
```

**Effect**: Process every 0.5s → Even less mixing

### Option C: Both (Best for Code-Switching)

```python
online_chunk_size = 0.6s  # Smaller chunks
# + Process on timer, not just VAD
```

**Effect**: Max 0.6s of mixed audio → Clean language switching

---

## Client Chunk Size - Should We Change It?

**Current**: We send 1.0s chunks from client

**SimulStreaming**: Sends 0.04s chunks

**vexa**: Sends variable small chunks

### Should we reduce client chunk size?

**Arguments FOR smaller chunks**:
- Lower latency (server can process sooner)
- Better network efficiency (WebSocket streaming)
- Matches reference implementations

**Arguments AGAINST**:
- More network overhead
- More frequent SocketIO emissions
- Current 1s chunks work fine IF server processes frequently

### Recommendation:

**Keep 1s client chunks** BUT **fix server processing**:
- Server should process every 1.2s (or less)
- Don't wait for VAD silence
- This way we get:
  - ✅ Reasonable network usage (1s chunks)
  - ✅ Frequent processing (1.2s intervals)
  - ✅ Clean language switching (max 1.2s mixing)

If we want even lower latency:
- Reduce to 0.5s client chunks
- Process every 0.6s on server
- Near-vexa performance with reasonable overhead

---

## Conclusion

**Root Cause**: We wait for VAD silence before processing, which causes massive (9s+) mixed-language chunks.

**Fix**: Process on **fixed time intervals** (every 1.2s or less) like SimulStreaming and vexa, regardless of VAD status.

**Result**:
- ✅ Clean language switching
- ✅ Correct language labels
- ✅ Production-ready code-switching

**Next Step**: Implement Option C (0.6s chunks + timer-based processing)
