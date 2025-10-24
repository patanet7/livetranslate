# Chunk Tracking Architecture - Streaming Audio Processing

## Problem Statement

When streaming audio chunks to the Whisper service, results are generated asynchronously and may arrive after the client has disconnected. This leads to:

1. **Missing Results**: Late-processed chunks generate results after client timeout
2. **No Correlation**: Client cannot determine which chunks have been fully processed
3. **Arbitrary Wait Times**: Client uses fixed timeouts (45s) instead of chunk-based completion tracking

### Example Failure Scenario

```
Client Timeline:
├─ t=0s:    Send chunks 0-19 (5 seconds total)
├─ t=5s:    All chunks sent
├─ t=20s:   Receive Result 1 (chunks 0-5)
├─ t=35s:   Receive Result 2 (chunks 6-12)
├─ t=45s:   DISCONNECT (timeout)
└─ t=50s:   Result 3 generated ❌ (chunks 13-19) - CLIENT ALREADY GONE!

Server Timeline:
├─ t=0-5s:  Accumulate chunks 0-19 in buffer
├─ t=5-20s: Process chunks 0-5 → Emit Result 1
├─ t=20-35s: Process chunks 6-12 → Emit Result 2
└─ t=35-50s: Process chunks 13-19 → Emit Result 3 ❌ (no one listening!)
```

## Root Cause Analysis

### 1. **Asynchronous Processing Gap**
- Client sends chunks faster than server can process (200ms intervals vs 15-20s processing)
- Server accumulates chunks in buffer before processing
- Processing latency is unpredictable (depends on model, audio content, hardware)

### 2. **No Chunk-to-Result Mapping**
Current SocketIO API:
```python
# Client sends:
sio.emit('transcribe_stream', {
    'session_id': 'test-123',
    'audio_data': chunk_b64,
    # ❌ NO: chunk_index, timestamp, or correlation ID
})

# Server emits:
sio.emit('transcription_result', {
    'text': 'transcription...',
    'is_final': True,
    # ❌ NO: chunk_range, processed_through_index, or completion status
})
```

### 3. **Misunderstood Semantics**
- `is_final=True` means "**complete sentence**", NOT "**all chunks processed**"
- Server continues processing remaining chunks after emitting `is_final=True`
- Client incorrectly assumes `is_final=True` means "session complete"

## Solution: Chunk Tracking with Metadata

### Phase 1: Add Chunk Metadata (Client → Server)

**Enhanced Request Format:**
```python
sio.emit('transcribe_stream', {
    'session_id': 'test-123',
    'audio_data': chunk_b64,

    # 🆕 Chunk Tracking Metadata
    'chunk_index': 5,              # Sequential chunk number (0-based)
    'chunk_timestamp': 1.0,        # Audio timestamp in seconds
    'chunk_duration': 1.0,         # Duration of this chunk (seconds)
    'total_duration_sent': 6.0,    # Total audio sent so far (seconds)
    'is_last_chunk': False,        # True if this is the final chunk
})
```

**Benefits:**
- Server knows which chunk is which
- Server knows total audio duration sent
- Server knows when all chunks have been received

### Phase 2: Add Processing Status (Server → Client)

**Enhanced Response Format:**
```python
sio.emit('transcription_result', {
    'text': 'transcription...',
    'is_final': True,              # Existing: "complete sentence"

    # 🆕 Processing Status Metadata
    'processed_through_index': 12,     # Processed up to and including chunk 12
    'processed_through_timestamp': 13.0, # Processed up to 13.0 seconds
    'chunks_in_buffer': 7,             # Chunks received but not yet processed
    'is_session_complete': False,      # True when ALL chunks processed
})
```

**Benefits:**
- Client knows exactly which chunks have been processed
- Client can wait until `processed_through_index == last_sent_index`
- Client can display progress: "Processed 13/20 chunks"

### Phase 3: End-of-Stream Signal

**Client Sends:**
```python
sio.emit('end_stream', {
    'session_id': 'test-123',
    'total_chunks_sent': 20,
    'total_duration_sent': 20.0,
})
```

**Server Responds:**
```python
# After processing ALL chunks:
sio.emit('stream_complete', {
    'session_id': 'test-123',
    'total_results_emitted': 3,
    'total_chunks_processed': 20,
    'total_duration_processed': 20.0,
})
```

**Benefits:**
- Client explicitly signals "no more chunks coming"
- Server knows when to finalize processing
- Clear completion signal (not ambiguous like `is_final`)

## Comparison with SimulStreaming and vexa Architectures

### SimulStreaming's Approach (Frame-based Tracking)

SimulStreaming (from `src/simul_whisper/simul_whisper.py`) tracks:

1. **Frame-based Attention**:
   ```python
   # Line 601-605: Track which audio frames decoder is attending to
   most_attended_frames = torch.argmax(attn_of_alignment_heads[:,-1,:], dim=-1)
   most_attended_frame = most_attended_frames[0].item()

   # Line 631-635: Stop when attention reaches end of audio
   if content_mel_len - most_attended_frame <= frame_threshold:
       # Decoder has processed all available audio
       break
   ```

2. **Audio Segment Metadata**:
   ```python
   # Lines 334-336: Track total audio duration
   def segments_len(self):
       segments_len = sum(s.shape[0] for s in self.segments) / 16000
       return segments_len

   # Lines 346-362: Track removed audio and adjust attention position
   removed_len = self.segments[0].shape[0] / 16000
   self.last_attend_frame -= int(TOKENS_PER_SECOND * removed_len)
   ```

3. **Generation Metadata**:
   ```python
   # Lines 502-518: Track decoding progress
   generation = {
       "token_len_before_decoding": token_len_before_decoding,
       "frames_len": content_mel_len,
       "frames_threshold": frame_threshold,
       "progress": generation_progress,  # Per-step metadata
   }
   ```

**Key Insight**: SimulStreaming knows EXACTLY which audio frames correspond to which tokens via attention tracking.

### vexa/WhisperLive's Approach (Absolute Time-based Tracking)

vexa (from `reference/vexa/services/WhisperLive/`) uses **absolute timestamps** instead of chunk indices:

1. **Absolute Time Tracking**:
   ```python
   # reference/vexa/services/WhisperLive/whisper_live/server.py:1639
   self.timestamp_offset = 0.0  # Tracks absolute time position in audio stream

   # Lines 1780-1784: Check timestamp offset
   if self.timestamp_offset < self.frames_offset:
       self.timestamp_offset = self.frames_offset

   # Lines 1793-1800: Update timestamp based on buffer
   if self.frames_np[...].shape[0] > self.clip_if_no_segment_s * self.RATE:
       self.timestamp_offset = self.frames_offset + duration - self.clip_retain_s
   ```

2. **Segment Deduplication via Absolute Timestamps**:
   ```python
   # reference/vexa/testing/ws_realtime_transcription.py:95-99
   # Bootstrap from REST API - seed map by absolute_start_time
   for segment in segments:
       abs_start = segment.get('absolute_start_time')
       if abs_start and segment.get('text', '').strip():
           self.transcript_by_abs_start[abs_start] = segment

   # Lines 116-119: Deduplicate by updated_at timestamp
   existing = self.transcript_by_abs_start.get(abs_start)
   if existing and existing.get('updated_at') and segment.get('updated_at'):
       if segment['updated_at'] < existing['updated_at']:
           continue  # Keep existing (newer)
   ```

3. **Segment Format with Absolute Times**:
   ```python
   # Each segment has:
   {
       'text': 'transcription text',
       'absolute_start_time': '2025-01-23T10:15:30.123Z',  # ISO 8601
       'absolute_end_time': '2025-01-23T10:15:32.456Z',
       'updated_at': '2025-01-23T10:15:35.789Z',
       'speaker': 'SPEAKER_00',
       'start': 0.123,  # Relative to segment
       'end': 2.333,    # Relative to segment
   }
   ```

4. **Redis Stream Publishing with Timestamps**:
   ```python
   # reference/vexa/services/WhisperLive/whisper_live/server.py:280-285
   now = datetime.datetime.utcnow()
   timestamp_iso = now.isoformat() + "Z"
   redis_message_payload["server_received_timestamp_iso"] = timestamp_iso

   result = self.redis_client.xadd(
       self.stream_key,
       redis_message_payload
   )
   ```

**Key Insights from vexa**:
- ✅ **Time-based, not index-based**: Uses absolute ISO 8601 timestamps
- ✅ **Deduplication by timestamp**: `absolute_start_time` as unique key
- ✅ **Updated tracking**: Keeps newer version based on `updated_at`
- ✅ **Server timestamp tracking**: `timestamp_offset` tracks stream position
- ✅ **No chunk indices needed**: Natural deduplication via absolute time

### Our Current Approach (No Tracking)

Current implementation:

**Issues:**
- No chunk_id, timestamp, or correlation tracking
- No way to know which results correspond to which chunks
- Client uses arbitrary timeouts (45s) instead of completion tracking

**Needs:**
- Add timestamp metadata to chunks (vexa-style)
- Track processed audio duration (SimulStreaming-style)
- Emit completion signals when all audio processed

## Implementation Roadmap

### Immediate (Test Fix)

**File**: `tests/test_detected_language_real_audio.py`

```python
# Track sent chunks
chunks_sent = []
for i in range(total_chunks):
    chunk_metadata = {
        'chunk_index': i,
        'chunk_timestamp': i * 1.0,  # Each chunk is 1 second
        'chunk_duration': len(chunk) / 16000,
        'total_duration_sent': (i + 1) * 1.0,
        'is_last_chunk': (i == total_chunks - 1),
    }
    chunks_sent.append(chunk_metadata)

    request_data = {
        "session_id": session_id,
        "audio_data": chunk_b64,
        **chunk_metadata,  # 🆕 Include metadata
    }
    sio.emit('transcribe_stream', request_data)

# Wait for all chunks to be processed (not just arbitrary timeout)
max_wait = 60
processed_through = -1

for i in range(max_wait):
    time.sleep(1)
    # Check if server reported processing all chunks
    if processed_through >= len(chunks_sent) - 1:
        print(f"✅ All {len(chunks_sent)} chunks processed!")
        break
    if i % 5 == 0:
        print(f"Waiting... processed {processed_through + 1}/{len(chunks_sent)} chunks")
```

### Short-term (Whisper Service SocketIO API)

**File**: `src/api_server.py` (SocketIO handlers)

**Add to `handle_transcribe_stream()`**:
```python
@sio.on('transcribe_stream')
async def handle_transcribe_stream(sid, data):
    # Extract chunk metadata
    chunk_index = data.get('chunk_index')
    chunk_timestamp = data.get('chunk_timestamp')
    is_last_chunk = data.get('is_last_chunk', False)

    # Store in session metadata
    session_data['chunks_received'].append({
        'index': chunk_index,
        'timestamp': chunk_timestamp,
        'received_at': time.time(),
    })

    # Track if this is the last chunk
    if is_last_chunk:
        session_data['all_chunks_received'] = True
        session_data['total_chunks'] = chunk_index + 1
```

**Add to transcription result emission**:
```python
result = {
    'text': text,
    'is_final': is_final,

    # 🆕 Processing status
    'processed_through_index': session_data['last_processed_chunk'],
    'processed_through_timestamp': session_data['last_processed_timestamp'],
    'chunks_in_buffer': len(session_data['pending_chunks']),
    'is_session_complete': (
        session_data.get('all_chunks_received', False) and
        len(session_data['pending_chunks']) == 0
    ),
}

await sio.emit('transcription_result', result, room=sid)

# If session complete, send final signal
if result['is_session_complete']:
    await sio.emit('stream_complete', {
        'session_id': session_id,
        'total_results': session_data['results_count'],
        'total_chunks_processed': session_data['total_chunks'],
    }, room=sid)
```

### Medium-term (Orchestration Integration)

**File**: `modules/orchestration-service/src/clients/audio_service_client.py`

**Enhance `send_realtime_audio()`**:
```python
async def send_realtime_audio(
    self,
    session_id: str,
    audio_chunk: bytes,
    chunk_index: int = None,  # 🆕
    chunk_timestamp: float = None,  # 🆕
    is_last_chunk: bool = False,  # 🆕
) -> Optional[Dict[str, Any]]:
    data = aiohttp.FormData()
    data.add_field("session_id", session_id)
    data.add_field("audio_chunk", audio_chunk, ...)

    # 🆕 Add chunk metadata
    if chunk_index is not None:
        data.add_field("chunk_index", str(chunk_index))
    if chunk_timestamp is not None:
        data.add_field("chunk_timestamp", str(chunk_timestamp))
    data.add_field("is_last_chunk", str(is_last_chunk).lower())

    # ... rest of implementation
```

## Migration Path

### 1. **Backward Compatibility**
- Make chunk metadata **optional** in initial release
- Server falls back to current behavior if metadata absent
- Clients can migrate incrementally

### 2. **Gradual Rollout**
1. **Phase 1**: Add metadata fields (optional, informational only)
2. **Phase 2**: Server uses metadata for progress tracking
3. **Phase 3**: Client uses progress tracking to optimize waits
4. **Phase 4**: Deprecate old API without metadata

### 3. **Testing Strategy**
```python
# Test with metadata
test_streaming_with_chunk_tracking()  # 🆕

# Test without metadata (backward compat)
test_streaming_legacy_api()  # Existing behavior

# Test mixed scenarios
test_streaming_partial_metadata()  # Some chunks have metadata
```

## Expected Benefits

### 1. **Reliability**
- ✅ No more missed results (client waits for all chunks)
- ✅ Deterministic completion (not arbitrary timeouts)
- ✅ Clear error recovery (know which chunks failed)

### 2. **Performance**
- ✅ Faster completion (stop waiting immediately when done)
- ✅ Better resource usage (no 45s wait for 20s audio)
- ✅ Progress feedback ("Processed 15/20 chunks")

### 3. **Debugging**
- ✅ Trace chunk → result mapping
- ✅ Identify slow chunks (processing time per chunk)
- ✅ Detect bottlenecks (chunk accumulation vs processing)

## References

- **SimulStreaming Architecture**: `src/simul_whisper/simul_whisper.py`
  - Frame-based attention tracking (lines 601-635)
  - Audio segment duration tracking (lines 334-362)
  - Generation metadata (lines 502-518)

- **Orchestration Audio Client**: `modules/orchestration-service/src/clients/audio_service_client.py`
  - Current realtime API (lines 616-695)
  - Needs chunk tracking enhancement

- **Whisper Service SocketIO API**: `src/api_server.py`
  - Current `transcribe_stream` handler
  - Needs chunk metadata extraction and progress tracking

## Next Steps

1. ✅ Document architecture findings (this file)
2. ⏳ Update test to track chunks and wait intelligently
3. ⏳ Enhance Whisper service SocketIO API with chunk tracking
4. ⏳ Update orchestration service to send chunk metadata
5. ⏳ Add progress tracking to frontend (optional)

---

**Author**: Architecture investigation based on SimulStreaming reference implementation
**Date**: 2025-01-23
**Status**: Proposed (Phase 1 implementation pending)
