# Hybrid Tracking System - Combining SimulStreaming Attention + vexa Timestamps

## Executive Summary

**Goal**: Combine the internal precision of SimulStreaming's attention tracking with the external correlation of vexa's timestamps to create a robust, observable streaming transcription system.

**Key Insight**: Keep BOTH tracking mechanisms:
- **SimulStreaming (Internal)**: Frame-based attention tracking for processing decisions
- **vexa (External)**: Absolute timestamp tracking for client correlation and completion

## Critical Semantic Clarification

### ⚠️ `is_final` Does NOT Mean "Session Done"!

```python
# WRONG INTERPRETATION ❌
if result['is_final'] == True:
    disconnect()  # Too early! More chunks still processing!

# CORRECT INTERPRETATION ✅
if result['is_final'] == True:
    # This is a complete sentence/thought
    # Server MAY continue processing remaining audio
    # Client should wait for is_session_complete=True
```

### Three States to Track

1. **`is_final`** (Sentence Boundary)
   - Meaning: "This sentence/thought is complete"
   - Source: SimulStreaming's end-of-word detection
   - Use: Formatting, translation triggers, UI display

2. **`is_session_complete`** (Stream Exhausted)
   - Meaning: "ALL audio chunks have been processed"
   - Source: Hybrid tracking (attention + timestamps)
   - Use: Client disconnection, final cleanup

3. **`is_draft`** (Unstable Text)
   - Meaning: "This text may still change"
   - Source: Stability tracking
   - Use: UI rendering (gray out unstable text)

## Hybrid Tracking Architecture

### Layer 1: SimulStreaming Attention Tracking (Internal)

**Purpose**: Track which audio frames the decoder has processed

**Implementation** (in `vac_online_processor.py`):

```python
class VACOnlineProcessor:
    def __init__(self):
        # Existing SimulStreaming tracking
        self.last_attend_frame = -self.cfg.rewind_threshold
        self.segments = []  # Audio buffers
        self.timestamp_offset = 0.0  # vexa-style position tracking

        # 🆕 Hybrid tracking additions
        self.audio_start_time = 0.0  # Session start (seconds)
        self.audio_end_time = 0.0    # Latest chunk end (seconds)
        self.frames_to_time_offset = 0.0  # Mapping between frames and absolute time

    def _process_online_chunk(self, audio_chunk, chunk_metadata):
        """Process chunk with hybrid tracking"""

        # Extract vexa-style metadata from chunk
        chunk_start_time = chunk_metadata.get('audio_start_time', self.audio_end_time)
        chunk_end_time = chunk_metadata.get('audio_end_time', chunk_start_time + len(audio_chunk)/16000)

        # Update absolute time tracking
        self.audio_end_time = max(self.audio_end_time, chunk_end_time)

        # Existing SimulStreaming processing
        new_hypothesis, generation = self.model.infer(is_last=False)

        # 🆕 Extract attention tracking from generation metadata
        most_attended_frame = generation.get('most_attended_frame', 0)
        content_mel_len = generation.get('frames_len', 0)

        # 🆕 Convert frames to absolute time
        # TOKENS_PER_SECOND = 50 (from whisper)
        processed_through_time = self.frames_to_time_offset + (most_attended_frame / 50.0)

        # 🆕 Check if we've processed all received audio
        is_caught_up = (content_mel_len - most_attended_frame) <= self.cfg.frame_threshold

        return {
            'text': text,
            'is_final': is_final,  # SimulStreaming: complete sentence

            # 🆕 Hybrid tracking metadata
            'attention_tracking': {
                'most_attended_frame': most_attended_frame,
                'content_mel_len': content_mel_len,
                'is_caught_up': is_caught_up,
            },
            'timestamp_tracking': {
                'processed_through_time': processed_through_time,
                'audio_received_through': self.audio_end_time,
                'is_session_complete': is_caught_up and chunk_metadata.get('is_last_chunk', False),
            },

            # vexa-style segment metadata
            'absolute_start_time': chunk_start_time,
            'absolute_end_time': chunk_end_time,
        }
```

### Layer 2: vexa Timestamp Correlation (External)

**Purpose**: Map chunks to results, detect completion

**Client-Side Tracking** (in `test_detected_language_real_audio.py`):

```python
class ChunkTracker:
    """Track sent chunks and received results with hybrid metadata"""

    def __init__(self):
        self.chunks_sent = []  # List of {index, start_time, end_time, sent_at}
        self.results_by_abs_time = {}  # vexa-style deduplication
        self.latest_processed_time = 0.0
        self.total_audio_sent = 0.0

    def track_sent_chunk(self, chunk_index, audio_data, sent_at):
        """Track a sent chunk with timestamp metadata"""
        chunk_duration = len(audio_data) / 16000  # Assuming 16kHz
        chunk_start = self.total_audio_sent
        chunk_end = chunk_start + chunk_duration

        chunk_metadata = {
            'chunk_index': chunk_index,
            'audio_start_time': chunk_start,
            'audio_end_time': chunk_end,
            'chunk_duration': chunk_duration,
            'sent_at': sent_at,
        }

        self.chunks_sent.append(chunk_metadata)
        self.total_audio_sent = chunk_end

        return chunk_metadata

    def track_received_result(self, result):
        """Track received result with hybrid metadata"""
        abs_start = result.get('absolute_start_time')

        # vexa-style deduplication by absolute time
        if abs_start:
            existing = self.results_by_abs_time.get(abs_start)
            updated_at_new = result.get('updated_at', 0)
            updated_at_old = existing.get('updated_at', 0) if existing else 0

            # Keep newer version (vexa pattern)
            if updated_at_new >= updated_at_old:
                self.results_by_abs_time[abs_start] = result

        # Update latest processed time from hybrid tracking
        timestamp_tracking = result.get('timestamp_tracking', {})
        processed_through = timestamp_tracking.get('processed_through_time', 0)
        self.latest_processed_time = max(self.latest_processed_time, processed_through)

    def is_complete(self):
        """Check if all sent audio has been processed"""
        # Allow small tolerance (0.1s) for floating point comparison
        tolerance = 0.1
        return (self.total_audio_sent - self.latest_processed_time) < tolerance

    def get_processing_progress(self):
        """Get processing progress percentage"""
        if self.total_audio_sent == 0:
            return 0.0
        return (self.latest_processed_time / self.total_audio_sent) * 100.0

    def get_unprocessed_chunks(self):
        """Get list of chunks not yet processed (for debugging)"""
        unprocessed = []
        for chunk in self.chunks_sent:
            if chunk['audio_end_time'] > self.latest_processed_time:
                unprocessed.append(chunk)
        return unprocessed
```

### Layer 3: Enhanced Result Format (Combined Metadata)

**Server Response** (from `api_server.py`):

```python
result = {
    # Original fields
    'text': text,
    'detected_language': detected_language,
    'is_final': is_final,  # ⚠️ SENTENCE complete, NOT session complete!

    # 🆕 SimulStreaming attention tracking (internal precision)
    'attention_tracking': {
        'most_attended_frame': most_attended_frame,
        'content_mel_len': content_mel_len,
        'frame_threshold': frame_threshold,
        'is_caught_up': is_caught_up,  # Decoder caught up to available audio
    },

    # 🆕 vexa timestamp tracking (external correlation)
    'timestamp_tracking': {
        'processed_through_time': processed_through_time,  # Seconds processed
        'audio_received_through': audio_received_through,  # Seconds received
        'is_session_complete': is_session_complete,        # ALL audio processed
        'lag_seconds': audio_received_through - processed_through_time,
    },

    # 🆕 vexa-style segment metadata
    'absolute_start_time': absolute_start_time,  # ISO 8601 or seconds
    'absolute_end_time': absolute_end_time,
    'updated_at': time.time(),  # For deduplication

    # 🆕 Debugging/observability
    'metadata': {
        'chunk_index_range': [first_chunk_idx, last_chunk_idx],
        'generation_stats': {
            'tokens_generated': len(tokens),
            'beam_size': beam_size,
            'decode_time_ms': decode_time_ms,
        },
    },
}
```

## Enhanced Client Wait Logic

**Test Implementation** (replaces arbitrary 45s timeout):

```python
def test_english_audio():
    print("\n" + "="*80)
    print("TEST: English Audio (JFK) - Hybrid Tracking")
    print("="*80)

    # Load audio
    audio_path = os.path.join(os.path.dirname(__file__), 'audio', 'jfk.wav')
    audio_int16, sample_rate = load_wav_file(audio_path)

    sio = socketio.Client()
    tracker = ChunkTracker()  # 🆕 Hybrid tracker
    results_received = []

    @sio.on('transcription_result')
    def on_result(data):
        results_received.append(data)
        tracker.track_received_result(data)

        # Extract both tracking systems
        is_final = data.get('is_final', False)
        timestamp_tracking = data.get('timestamp_tracking', {})
        attention_tracking = data.get('attention_tracking', {})

        is_session_complete = timestamp_tracking.get('is_session_complete', False)
        processed_through = timestamp_tracking.get('processed_through_time', 0)
        most_attended_frame = attention_tracking.get('most_attended_frame', 0)

        print(f"  Result: text='{data.get('text', '')[:50]}...'")
        print(f"    is_final={is_final} (sentence complete)")
        print(f"    is_session_complete={is_session_complete} (all audio processed)")
        print(f"    processed_through_time={processed_through:.2f}s")
        print(f"    most_attended_frame={most_attended_frame}")
        print(f"    progress={tracker.get_processing_progress():.1f}%")

    # Connect and stream chunks
    sio.connect(SERVICE_URL)
    session_id = f"test-english-{int(time.time())}"
    sio.emit('join_session', {'session_id': session_id})

    # Resample to 16kHz if needed (omitted for brevity)

    # Stream audio in chunks with timestamp metadata
    chunk_size = 16000 * 1  # 1 second chunks
    total_chunks = len(audio_int16) // chunk_size + (1 if len(audio_int16) % chunk_size else 0)

    print(f"\n  Streaming {total_chunks} chunks (1s each)...")

    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(audio_int16))
        chunk = audio_int16[start_idx:end_idx]

        # 🆕 Track sent chunk and get metadata
        chunk_metadata = tracker.track_sent_chunk(i, chunk, time.time())

        chunk_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')

        request_data = {
            "session_id": session_id,
            "audio_data": chunk_b64,
            "model_name": "base",
            "sample_rate": 16000,
            "enable_code_switching": True,

            # 🆕 Hybrid tracking metadata
            "chunk_index": chunk_metadata['chunk_index'],
            "audio_start_time": chunk_metadata['audio_start_time'],
            "audio_end_time": chunk_metadata['audio_end_time'],
            "chunk_duration": chunk_metadata['chunk_duration'],
            "is_last_chunk": (i == total_chunks - 1),

            "config": {
                "sliding_lid_window": 0.9,
            }
        }

        print(f"  Chunk {i+1}/{total_chunks}: {chunk_metadata['audio_start_time']:.2f}s - {chunk_metadata['audio_end_time']:.2f}s")
        sio.emit('transcribe_stream', request_data)
        time.sleep(0.2)

    print(f"\n  All {total_chunks} chunks sent (total: {tracker.total_audio_sent:.2f}s)")
    print("  ⚠️  IMPORTANT: is_final=True means 'sentence complete', NOT 'session done'!")
    print("  Waiting for is_session_complete=True OR tracker.is_complete()...")

    # 🆕 Intelligent wait using hybrid tracking
    max_wait_seconds = 60
    check_interval = 1.0

    for i in range(int(max_wait_seconds / check_interval)):
        time.sleep(check_interval)

        # Check hybrid completion criteria
        if tracker.is_complete():
            print(f"\n  ✅ Tracker reports complete: {tracker.latest_processed_time:.2f}s / {tracker.total_audio_sent:.2f}s")
            break

        # Check server-reported completion
        if results_received:
            last_result = results_received[-1]
            timestamp_tracking = last_result.get('timestamp_tracking', {})
            if timestamp_tracking.get('is_session_complete', False):
                print(f"\n  ✅ Server reports is_session_complete=True")
                break

        # Progress update every 5 seconds
        if i % 5 == 0:
            progress = tracker.get_processing_progress()
            unprocessed = len(tracker.get_unprocessed_chunks())
            print(f"    Waiting... {progress:.1f}% processed, {unprocessed} chunks pending")
    else:
        print(f"\n  ⚠️  Timeout after {max_wait_seconds}s")
        print(f"    Final progress: {tracker.get_processing_progress():.1f}%")
        print(f"    Unprocessed chunks: {len(tracker.get_unprocessed_chunks())}")

    # Summary
    print(f"\n  Received {len(results_received)} total results")
    for i, result in enumerate(results_received):
        is_final = result.get('is_final', False)
        timestamp_tracking = result.get('timestamp_tracking', {})
        processed = timestamp_tracking.get('processed_through_time', 0)
        print(f"    Result {i+1}: is_final={is_final}, processed_through={processed:.2f}s, text='{result.get('text', '')[:60]}...'")

    # Cleanup
    sio.emit('leave_session', {'session_id': session_id})
    time.sleep(0.5)
    sio.disconnect()

    return tracker.is_complete()
```

## Benefits of Hybrid Tracking

### 1. **Internal Precision** (SimulStreaming)
✅ Know EXACTLY which frames decoder is processing
✅ Detect when decoder is caught up to available audio
✅ Optimize buffering and segment boundaries
✅ Debug attention patterns and model behavior

### 2. **External Correlation** (vexa)
✅ Map results back to sent chunks
✅ Natural deduplication via timestamps
✅ Handle out-of-order results gracefully
✅ Client knows completion without guessing

### 3. **Combined Power**
✅ **Precise AND observable**: Internal decoder state + external correlation
✅ **Debugging**: See both "what frame" and "what timestamp"
✅ **Robust**: Multiple completion signals (attention + timestamps)
✅ **Production-ready**: Best of both proven systems

## Observability Dashboard (Future Enhancement)

With hybrid tracking, we can build rich observability:

```
┌─────────────────────────────────────────────────────────────┐
│ Session: test-english-1234567890                            │
├─────────────────────────────────────────────────────────────┤
│ Audio Sent:      20.0s ████████████████████████████ 100%    │
│ Audio Processed: 13.5s ████████████████░░░░░░░░░░░░  67%    │
│ Lag:              6.5s                                       │
├─────────────────────────────────────────────────────────────┤
│ Attention Tracking:                                          │
│   Most Attended Frame: 675 / 1000 frames                    │
│   Is Caught Up: False                                        │
│   Frame Threshold: 4                                         │
├─────────────────────────────────────────────────────────────┤
│ Results Received: 2                                          │
│   Result 1: is_final=False, 0.0s-5.2s, "My fellow Americans"│
│   Result 2: is_final=True,  5.2s-13.5s, "ask not what..."   │
├─────────────────────────────────────────────────────────────┤
│ Status: 🟡 Processing (3 chunks pending)                     │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Server-Side Enhancement (Immediate)
1. ✅ Add attention tracking extraction from `generation` metadata
2. ✅ Add timestamp offset tracking (vexa-style)
3. ✅ Emit hybrid result format with both tracking systems
4. ✅ Add `is_session_complete` distinct from `is_final`

### Phase 2: Client-Side Enhancement (Test Update)
1. ✅ Create `ChunkTracker` class with hybrid tracking
2. ✅ Send chunk metadata with timestamps
3. ✅ Intelligent wait based on `is_complete()` instead of timeout
4. ✅ Progress reporting during wait

### Phase 3: Orchestration Integration
1. Update `audio_service_client.py` to send timestamps
2. Propagate hybrid metadata through orchestration
3. Frontend progress indicators

### Phase 4: Observability
1. Logging/metrics for attention patterns
2. Dashboard for real-time tracking
3. Alerting for stuck processing

## Migration & Backward Compatibility

**Principle**: Make all new fields optional

```python
# Server checks if client sent metadata
chunk_metadata = data.get('chunk_metadata')
if chunk_metadata:
    # Use hybrid tracking
    process_with_timestamps(chunk_metadata)
else:
    # Fall back to legacy behavior
    process_without_metadata()

# Client checks if server sent metadata
timestamp_tracking = result.get('timestamp_tracking')
if timestamp_tracking:
    # Use intelligent wait
    wait_for_completion(timestamp_tracking)
else:
    # Fall back to timeout
    time.sleep(45)
```

## References

- **SimulStreaming**: `src/simul_whisper/simul_whisper.py`
  - Lines 601-635: Attention tracking
  - Lines 502-518: Generation metadata

- **vexa**: `reference/vexa/services/WhisperLive/`
  - `whisper_live/server.py:1639-1800`: Timestamp offset tracking
  - `testing/ws_realtime_transcription.py:95-119`: Deduplication

- **CHUNK_TRACKING_ARCHITECTURE.md**: Detailed comparison of both systems

---

**Status**: Design Complete - Ready for Implementation
**Next Step**: Implement Phase 1 (Server-Side Enhancement) + Phase 2 (Test Update)
