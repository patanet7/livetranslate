# 🎯 SimulStreaming Stability Tracking + AlignAtt Integration Plan

## Executive Summary

This plan implements a complete SimulStreaming system that combines:
- **READ Policy (AlignAtt)**: When to read more audio - uses attention patterns
- **WRITE Policy (Stability)**: When to emit tokens - prevents unstable output

Together these provide low-latency, high-quality simultaneous transcription and translation.

---

## Phase 1: Fix Current Streaming Foundation (IMMEDIATE)
**Goal**: Get basic streaming working with AlignAtt

### 1.1 Simplify Buffer Management
```python
# whisper_service.py - replace complex buffer_manager
class WhisperService:
    def __init__(self):
        # Simple SimulStreaming-style buffer
        self.audio_segments = []  # List[torch.Tensor]
        self.audio_max_len = 30.0  # seconds
        self.audio_min_len = 1.0   # seconds
```

**Key Changes**:
- Remove `buffer_manager` attribute
- Remove imports of `BufferConfig`, `RollingBufferManager`
- Replace with simple list `self.audio_segments`
- Match SimulStreaming reference implementation

### 1.2 Update transcribe_stream() to use AlignAtt properly
```python
async def transcribe_stream(self, request):
    # Add audio chunk to buffer
    self.audio_segments.append(audio_tensor)

    # Maintain rolling window
    while segments_len > self.audio_max_len:
        self.audio_segments.pop(0)

    # Process ENTIRE buffer (AlignAtt tracks progress internally)
    if segments_len >= self.audio_min_len:
        result = await self.process_with_alignatt(
            audio=torch.cat(self.audio_segments),
            is_last=False
        )
        yield result
```

**Critical Understanding**:
- Feed ENTIRE buffer to model each time
- AlignAtt decoder tracks what's already been decoded internally
- Don't manually track "new vs old" audio - that's AlignAtt's job
- `frame_threshold` parameter controls how far to decode

### 1.3 Remove VAD from transcribe()
```python
# Remove this block:
if request.enable_vad and self.buffer_manager:
    speech_start, speech_end = self.buffer_manager.find_speech_boundaries(audio_data)
```

**Reason**: VAD is handled internally by AlignAtt/BeamSearch decoders

---

## Phase 2: Implement Stability Tracking System
**Goal**: Add WRITE policy for stable token emission

### 2.1 Create `stability_tracker.py`

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import math

@dataclass
class StabilityConfig:
    """Configuration for stability detection"""
    stability_threshold: float = 0.85      # Min confidence for stable
    min_stable_words: int = 2              # Min words before emitting
    min_hold_time: float = 0.3             # Min time to observe token
    max_latency: float = 2.0               # Max delay before forcing emit
    word_boundary_bonus: float = 0.1       # Boost at word boundaries

@dataclass
class TokenState:
    """State of a single token"""
    token_id: int
    text: str
    logprob: float
    first_seen: float      # When first generated
    last_updated: float    # Last time it appeared
    update_count: int      # How many times it's been consistent
    confidence: float      # Current confidence score
    is_word_boundary: bool # True if ends a word

class StabilityTracker:
    """
    Tracks token stability for incremental MT updates.

    Key Concept:
    - stable_prefix: Tokens we're confident won't change → send to MT
    - unstable_tail: Tokens that might still change → hold back

    Display Visualization:
    - Black text = stable_prefix (confirmed, sent to MT)
    - Grey text = unstable_tail (might change, held back)
    """

    def __init__(self, config: StabilityConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Token tracking
        self.stable_prefix: List[TokenState] = []
        self.unstable_tail: List[TokenState] = []

        # Stability detection
        self.last_emit_time = time.time()

    def update(self,
               new_tokens: List[int],
               logprobs: List[float],
               timestamp: float) -> Tuple[List[TokenState], bool]:
        """
        Update with new tokens from decoder.

        Returns:
            (new_stable_tokens, is_forced_emit)
        """
        # Compare with existing tokens to find what changed
        aligned = self._align_tokens(new_tokens, logprobs)

        # Detect stable prefix boundary
        stable_idx = self._find_stable_boundary(aligned, timestamp)

        # Extract new stable tokens
        new_stable = self.unstable_tail[:stable_idx]
        self.stable_prefix.extend(new_stable)
        self.unstable_tail = self.unstable_tail[stable_idx:] + aligned

        # Check if we should force emit due to max latency
        should_force = self._should_force_emit(timestamp)

        return new_stable, should_force

    def _align_tokens(self, new_tokens, logprobs):
        """
        Align new tokens with existing unstable tail.

        Strategy:
        - Find longest common prefix with existing unstable tokens
        - Increment update_count for matched tokens
        - Create new TokenState for unmatched tokens
        """
        matched = 0
        for i, (old, new) in enumerate(zip(self.unstable_tail, new_tokens)):
            if old.token_id == new:
                matched = i + 1
                old.update_count += 1
                old.last_updated = time.time()
                old.confidence = self._compute_confidence(old, logprobs[i])
            else:
                break

        # Create new TokenState for unmatched tokens
        new_states = [
            TokenState(
                token_id=tok,
                text=self.tokenizer.decode([tok]),
                logprob=lp,
                first_seen=time.time(),
                last_updated=time.time(),
                update_count=1,
                confidence=math.exp(lp),
                is_word_boundary=self._is_word_boundary(tok)
            )
            for tok, lp in zip(new_tokens[matched:], logprobs[matched:])
        ]

        return self.unstable_tail[:matched] + new_states

    def _compute_confidence(self, token: TokenState, new_logprob: float) -> float:
        """
        Compute confidence based on:
        - Current logprob
        - Update count (consistency)
        - Time observed
        """
        base_conf = math.exp(new_logprob)
        consistency_bonus = min(token.update_count * 0.05, 0.2)
        time_bonus = min((time.time() - token.first_seen) * 0.1, 0.1)

        return min(base_conf + consistency_bonus + time_bonus, 1.0)

    def _find_stable_boundary(self, tokens: List[TokenState], timestamp: float) -> int:
        """
        Find index where tokens transition from stable to unstable.

        Stability Criteria:
        1. Seen at least 3 times (update_count >= 3)
        2. High confidence (>= stability_threshold)
        3. Observed long enough (>= min_hold_time)
        4. Bonus for word boundaries
        """
        for i, token in enumerate(tokens):
            age = timestamp - token.first_seen

            # Check stability criteria
            if (token.update_count < 3 or
                token.confidence < self.config.stability_threshold or
                age < self.config.min_hold_time):
                return i  # Everything before this is stable

            # Bonus: word boundaries are more stable
            if token.is_word_boundary:
                # Check if next few tokens are stable too
                if self._check_word_stability(tokens[i+1:i+4]):
                    return i + 1

        return len(tokens)  # All tokens are stable

    def _check_word_stability(self, tokens: List[TokenState]) -> bool:
        """Check if a word (sequence of tokens) is stable"""
        if not tokens:
            return True
        return all(
            t.update_count >= 2 and
            t.confidence >= self.config.stability_threshold * 0.9
            for t in tokens[:3]
        )

    def _is_word_boundary(self, token_id: int) -> bool:
        """Check if token represents word boundary"""
        text = self.tokenizer.decode([token_id])
        return text.strip() != text or text.endswith(('.', '!', '?', ',', ';'))

    def _should_force_emit(self, timestamp: float) -> bool:
        """Check if we should force emit due to max latency"""
        elapsed = timestamp - self.last_emit_time
        return elapsed >= self.config.max_latency

    def finalize_segment(self) -> List[TokenState]:
        """
        Mark all tokens as stable at segment boundary.
        Called when segment is complete (silence, punctuation, etc.)
        """
        all_stable = self.stable_prefix + self.unstable_tail
        self.stable_prefix = []
        self.unstable_tail = []
        self.last_emit_time = time.time()
        return all_stable
```

### 2.2 Integrate with AlignAtt Decoder

```python
# whisper_service.py
class WhisperService:
    def __init__(self):
        # ... existing init ...
        if not self.orchestration_mode:
            self.stability_tracker = StabilityTracker(
                config=StabilityConfig(),
                tokenizer=self.model_manager.tokenizer
            )

    async def process_with_alignatt(self, audio, is_last=False):
        """
        Process audio with AlignAtt + Stability tracking.

        Returns:
            TranscriptionResult with stable/unstable separation
        """

        # AlignAtt inference (returns incremental tokens)
        tokens, logprobs, generation_progress = await self.alignatt_decoder.infer(
            audio=audio,
            is_last=is_last
        )

        # Stability detection
        new_stable, is_forced = self.stability_tracker.update(
            new_tokens=tokens,
            logprobs=logprobs,
            timestamp=time.time()
        )

        # If segment is final, get all tokens
        if is_last:
            all_tokens = self.stability_tracker.finalize_segment()
            return self._create_final_result(all_tokens, generation_progress)

        # Otherwise return incremental result
        return self._create_draft_result(
            new_stable=new_stable,
            is_forced=is_forced,
            generation_progress=generation_progress
        )
```

---

## Phase 3: Draft vs Final Emission Protocol
**Goal**: Distinguish between incremental drafts and final segments

### 3.1 Enhanced TranscriptionResult

```python
# whisper_service.py
@dataclass
class TranscriptionResult:
    """Enhanced result with stability info"""

    # Text representations
    text: str                          # Full text (stable + unstable)
    stable_text: str                   # Only stable prefix (black in UI)
    unstable_text: str                 # Only unstable tail (grey in UI)

    # Token-level data
    stable_tokens: List[TokenState]    # Confirmed tokens → send to MT
    unstable_tokens: List[TokenState]  # Uncertain tokens → hold back

    # Segment metadata
    is_final: bool                     # True = segment boundary reached
    is_draft: bool                     # True = incremental update
    is_forced: bool                    # True = forced by max_latency

    # Timestamps
    stable_end_time: float             # Time of last stable token
    segment_start_time: float
    segment_end_time: float

    # Confidence metrics
    stability_score: float             # Avg confidence of stable tokens
    segment_confidence: float          # Overall segment confidence

    # For MT integration
    should_translate: bool             # True if enough stable text for MT
    translation_mode: str              # "draft" or "final"

    # AlignAtt metadata
    generation_progress: dict          # Frame-level attention info
    segments: List[dict]               # Segment-level timestamps
```

### 3.2 Emission Logic

```python
# whisper_service.py
async def transcribe_stream(self, request):
    """
    Stream with draft/final emission.

    Emission Strategy:
    1. DRAFT: Emit when stable prefix grows
    2. FINAL: Emit at segment boundaries

    Yields:
        TranscriptionResult - draft or final
    """
    self.last_stable_count = 0

    while self.streaming_active:
        # Get new audio chunk
        audio_chunk = await self.get_next_chunk()
        if audio_chunk is None:
            break

        self.audio_segments.append(audio_chunk)

        # Maintain rolling window
        while self._segments_len() > self.audio_max_len:
            removed = self.audio_segments.pop(0)
            # Update context if needed

        # Skip if too short
        if self._segments_len() < self.audio_min_len:
            continue

        # Process with AlignAtt
        audio_concat = torch.cat(self.audio_segments, dim=0)
        result = await self.process_with_alignatt(
            audio=audio_concat,
            is_last=False
        )

        # DRAFT EMISSION: Stable prefix grew
        if len(result.stable_tokens) > self.last_stable_count:
            result.is_draft = True
            result.is_final = False
            result.should_translate = len(result.stable_tokens) >= 2
            result.translation_mode = "draft"

            yield result
            self.last_stable_count = len(result.stable_tokens)

        # FINAL EMISSION: Detect segment boundary
        if self._detect_segment_boundary(result):
            # Force finalize current segment
            final_result = await self.process_with_alignatt(
                audio=audio_concat,
                is_last=True
            )

            final_result.is_draft = False
            final_result.is_final = True
            final_result.should_translate = True
            final_result.translation_mode = "final"

            yield final_result

            # Reset for next segment
            self.audio_segments = []
            self.last_stable_count = 0
            self.stability_tracker = StabilityTracker(
                config=self.stability_config,
                tokenizer=self.model_manager.tokenizer
            )

def _detect_segment_boundary(self, result) -> bool:
    """
    Detect if we've reached a segment boundary.

    Criteria:
    - Long silence detected
    - Strong punctuation (. ! ?)
    - Max segment length reached
    """
    # Check for silence (from VAD or audio energy)
    if self._has_long_silence():
        return True

    # Check for strong punctuation
    if result.stable_text.rstrip().endswith(('.', '!', '?')):
        return True

    # Check max segment length
    if self._segments_len() >= self.audio_max_len * 0.9:
        return True

    return False
```

---

## Phase 4: Downstream Integration
**Goal**: Pass draft/final to translation service

### 4.1 WebSocket Event Schema

```python
# api_server.py - Socket.IO events

@socketio.on('transcribe_stream')
async def handle_transcribe_stream(data):
    """
    Handle streaming transcription with draft/final emission.

    Events emitted:
    - transcription_draft: Incremental stable prefix updates
    - transcription_final: Complete segment
    """
    try:
        # Create request
        request = TranscriptionRequest(
            audio_data=audio_array,
            model_name=data.get('model', 'large-v3-turbo'),
            language=data.get('language'),
            session_id=data.get('session_id'),
            streaming=True,
            sample_rate=data.get('sample_rate', 16000)
        )

        # Stream results
        async for result in whisper_service.transcribe_stream(request):
            if result.is_draft:
                # DRAFT: Incremental update
                socketio.emit('transcription_draft', {
                    # Display data
                    'stable_text': result.stable_text,
                    'unstable_text': result.unstable_text,
                    'full_text': result.text,

                    # Token data
                    'stable_tokens': [t.text for t in result.stable_tokens],
                    'unstable_tokens': [t.text for t in result.unstable_tokens],

                    # Metadata
                    'confidence': result.stability_score,
                    'timestamp': result.stable_end_time,
                    'session_id': data['session_id'],
                    'is_forced': result.is_forced,

                    # For MT service
                    'translate': {
                        'mode': 'draft',
                        'text': result.stable_text,
                        'tokens': [
                            {
                                'text': t.text,
                                'confidence': t.confidence,
                                'timestamp': t.last_updated
                            }
                            for t in result.stable_tokens
                        ]
                    }
                }, room=client_id)

            elif result.is_final:
                # FINAL: Complete segment
                socketio.emit('transcription_final', {
                    # Display data
                    'text': result.text,
                    'segments': result.segments,

                    # Token data
                    'tokens': [t.text for t in result.stable_tokens],

                    # Metadata
                    'confidence': result.segment_confidence,
                    'start': result.segment_start_time,
                    'end': result.segment_end_time,
                    'session_id': data['session_id'],

                    # For MT service
                    'translate': {
                        'mode': 'final',
                        'text': result.text,
                        'replace_draft': True,  # Tell MT to replace draft
                        'tokens': [
                            {
                                'text': t.text,
                                'confidence': t.confidence,
                                'timestamp': t.last_updated
                            }
                            for t in result.stable_tokens
                        ]
                    }
                }, room=client_id)

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        socketio.emit('transcription_error', {
            'error': str(e),
            'session_id': data.get('session_id')
        }, room=client_id)
```

### 4.2 Translation Service Integration

```python
# translation-service/src/stream_manager.py

class TranslationStreamManager:
    """
    Manages streaming translations with draft/final modes.

    Strategy:
    - DRAFT mode: Fast, lower quality for incremental updates
    - FINAL mode: Slow, higher quality for complete segments
    """

    def __init__(self):
        self.draft_translations = {}  # session_id -> current draft
        self.final_translations = []  # completed segments
        self.translation_context = {}  # session_id -> context

    async def handle_draft(self, data):
        """
        Handle draft transcription → produce draft translation.

        Strategy:
        - Use streaming mode (lower quality, lower latency)
        - Update incrementally as stable prefix grows
        - Don't store in history (will be replaced by final)
        """
        session_id = data['session_id']
        source_text = data['translate']['text']

        # Skip if too short
        if len(source_text.split()) < 2:
            return

        # Get previous context
        context = self.translation_context.get(session_id, '')

        # Translate in streaming mode
        draft = await self.translate(
            text=source_text,
            mode='streaming',  # Fast, may sacrifice quality
            context=context,
            max_length=100  # Limit length for speed
        )

        # Store draft
        self.draft_translations[session_id] = {
            'source': source_text,
            'translation': draft,
            'timestamp': time.time()
        }

        # Emit draft translation
        emit('translation_draft', {
            'source_text': source_text,
            'translation': draft,
            'is_draft': True,
            'confidence': data.get('confidence', 0.0),
            'session_id': session_id
        })

    async def handle_final(self, data):
        """
        Handle final transcription → produce final translation.

        Strategy:
        - Use quality mode (higher quality, higher latency OK)
        - Use context from previous segments
        - Store in history for future context
        """
        session_id = data['session_id']
        source_text = data['translate']['text']

        # Get context from last 3 segments
        recent_context = ' '.join(self.final_translations[-3:])

        # High-quality translation
        final = await self.translate(
            text=source_text,
            mode='quality',  # Slower, higher quality
            context=recent_context,
            beam_size=5  # More beams for better quality
        )

        # Store final translation
        self.final_translations.append(final)
        self.translation_context[session_id] = final

        # Clear draft
        if session_id in self.draft_translations:
            del self.draft_translations[session_id]

        # Emit final translation
        emit('translation_final', {
            'source_text': source_text,
            'translation': final,
            'is_final': True,
            'replace_draft': True,  # Frontend should replace draft
            'confidence': data.get('confidence', 1.0),
            'session_id': session_id,
            'segment_id': len(self.final_translations)
        })
```

---

## Phase 5: Frontend Display
**Goal**: Show stable (black) vs unstable (grey) text

### 5.1 React Component

```typescript
// frontend-service/src/components/LiveTranscription.tsx

interface TokenDisplay {
  text: string;
  confidence: number;
  isStable: boolean;
}

interface TranscriptionSegment {
  stableText: string;      // Black text - confirmed
  unstableText: string;    // Grey text - might change
  isDraft: boolean;
  isFinal: boolean;
  timestamp: number;
}

interface TranslationSegment {
  sourceText: string;
  translation: string;
  isDraft: boolean;
  isFinal: boolean;
}

export function LiveTranscriptionView() {
  const [currentTranscription, setCurrentTranscription] = useState<TranscriptionSegment | null>(null);
  const [currentTranslation, setCurrentTranslation] = useState<TranslationSegment | null>(null);
  const [history, setHistory] = useState<Array<{
    transcription: string;
    translation: string;
    timestamp: number;
  }>>([]);

  useEffect(() => {
    // DRAFT transcription updates
    socket.on('transcription_draft', (data) => {
      setCurrentTranscription({
        stableText: data.stable_text,
        unstableText: data.unstable_text,
        isDraft: true,
        isFinal: false,
        timestamp: data.timestamp
      });
    });

    // FINAL transcription updates
    socket.on('transcription_final', (data) => {
      // Add to history
      const segment = {
        transcription: data.text,
        translation: currentTranslation?.translation || '',
        timestamp: data.end
      };
      setHistory(prev => [...prev, segment]);

      // Clear current
      setCurrentTranscription(null);
    });

    // DRAFT translation updates
    socket.on('translation_draft', (data) => {
      setCurrentTranslation({
        sourceText: data.source_text,
        translation: data.translation,
        isDraft: true,
        isFinal: false
      });
    });

    // FINAL translation updates
    socket.on('translation_final', (data) => {
      // Update most recent history entry
      setHistory(prev => {
        const updated = [...prev];
        if (updated.length > 0) {
          updated[updated.length - 1].translation = data.translation;
        }
        return updated;
      });

      // Clear current
      setCurrentTranslation(null);
    });

    return () => {
      socket.off('transcription_draft');
      socket.off('transcription_final');
      socket.off('translation_draft');
      socket.off('translation_final');
    };
  }, [currentTranslation]);

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* History */}
      <div className="flex flex-col gap-2">
        {history.map((segment, i) => (
          <div key={i} className="border-b pb-2">
            <div className="text-sm text-gray-600">
              {new Date(segment.timestamp * 1000).toLocaleTimeString()}
            </div>
            <div className="font-medium">{segment.transcription}</div>
            <div className="text-blue-600 italic">{segment.translation}</div>
          </div>
        ))}
      </div>

      {/* Current (live) */}
      {currentTranscription && (
        <div className="bg-yellow-50 p-4 rounded border-2 border-yellow-200">
          <div className="text-xs text-gray-500 mb-1">LIVE</div>

          {/* Transcription: Black (stable) + Grey (unstable) */}
          <div className="text-lg mb-2">
            <span className="text-black font-medium">
              {currentTranscription.stableText}
            </span>
            <span className="text-gray-400 font-light">
              {currentTranscription.unstableText}
            </span>
          </div>

          {/* Translation: Draft */}
          {currentTranslation && (
            <div className="text-blue-600 italic">
              <span className="text-xs text-gray-500">DRAFT: </span>
              {currentTranslation.translation}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

### 5.2 Visual Design

```
┌─────────────────────────────────────────────┐
│ HISTORY                                     │
├─────────────────────────────────────────────┤
│ 10:23:15                                    │
│ Hello, how are you today?                   │
│ Bonjour, comment allez-vous aujourd'hui ?   │
├─────────────────────────────────────────────┤
│ 10:23:18                                    │
│ I'm doing great, thanks for asking.         │
│ Je vais très bien, merci de demander.       │
├─────────────────────────────────────────────┤
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ LIVE                                    │ │
│ │                                         │ │
│ │ What about you?                         │ │ ← Black (stable)
│ │ How is your da                          │ │ ← Grey (unstable)
│ │                                         │ │
│ │ DRAFT: Et toi ? Comment est            │ │ ← Translation draft
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

## Implementation Order

### ✅ WEEK 1: Foundation (Phase 1)
**Files to modify**:
- `whisper_service.py`
  - Remove `buffer_manager` attribute
  - Replace with `self.audio_segments = []`
  - Fix `transcribe()` to remove VAD calls
  - Fix `transcribe_stream()` to use simple buffer
  - Remove references to `get_new_audio_only()`, `mark_audio_as_processed()`

**Testing**:
- Basic streaming works
- Audio accumulates in buffer
- AlignAtt decoder receives full buffer
- No crashes from missing methods

### ✅ WEEK 2: Stability Tracking (Phase 2)
**Files to create**:
- `src/stability_tracker.py`
  - Implement `StabilityConfig`
  - Implement `TokenState`
  - Implement `StabilityTracker`

**Files to modify**:
- `whisper_service.py`
  - Add `stability_tracker` initialization
  - Create `process_with_alignatt()` method
  - Integrate stability detection

**Testing**:
- Tokens are correctly aligned
- Stable prefix is identified
- Confidence scoring works
- Word boundaries detected

### ✅ WEEK 3: Draft/Final Protocol (Phase 3)
**Files to modify**:
- `whisper_service.py`
  - Enhance `TranscriptionResult` dataclass
  - Implement draft emission logic
  - Implement final emission logic
  - Add segment boundary detection

**Testing**:
- Draft events emitted when stable prefix grows
- Final events emitted at segment boundaries
- Correct `is_draft` / `is_final` flags
- Translation mode set correctly

### ✅ WEEK 4: Integration (Phase 4 + 5)
**Files to modify**:
- `api_server.py`
  - Add `transcription_draft` event
  - Add `transcription_final` event
  - Update event payloads

**Files to create** (translation service):
- `translation-service/src/stream_manager.py`
  - Implement `TranslationStreamManager`
  - Handle draft/final modes

**Files to create** (frontend):
- `frontend-service/src/components/LiveTranscription.tsx`
  - Display stable (black) + unstable (grey)
  - Handle draft/final updates
  - Maintain history

**Testing**:
- End-to-end flow works
- Draft updates smooth
- Final updates replace drafts
- No missed segments

---

## Tunable Parameters

Users can adjust these for different use cases:

### Low Latency (News, Sports Commentary)
```python
StabilityConfig(
    stability_threshold=0.75,  # Lower = faster emission
    min_hold_time=0.1,         # Emit quickly
    max_latency=1.0,           # Force every 1 second
    min_stable_words=1         # Emit single words
)
```

### High Quality (Lectures, Medical, Legal)
```python
StabilityConfig(
    stability_threshold=0.95,  # Higher = more accurate
    min_hold_time=0.5,         # Wait longer
    max_latency=3.0,           # Can tolerate delay
    min_stable_words=3         # Wait for phrases
)
```

### Balanced (General Conversation)
```python
StabilityConfig(
    stability_threshold=0.85,
    min_hold_time=0.3,
    max_latency=2.0,
    min_stable_words=2
)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Audio Input Stream                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Rolling Audio Buffer                           │
│  [segment1, segment2, segment3, ...]                        │
│  Max: 30s, Min: 1s for processing                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           AlignAtt Decoder (READ Policy)                    │
│  - Tracks attention to frames                               │
│  - Decides which frames to decode                           │
│  - Outputs: tokens + logprobs + attention                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        Stability Tracker (WRITE Policy)                     │
│  Input: new_tokens, logprobs                                │
│  ├─ Token Alignment                                         │
│  ├─ Confidence Scoring                                      │
│  ├─ Stability Detection                                     │
│  └─ Boundary Detection                                      │
│  Output: stable_tokens, unstable_tokens                     │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
        DRAFT  │                      │  FINAL
               ▼                      ▼
┌──────────────────────┐  ┌──────────────────────┐
│  transcription_draft │  │  transcription_final │
│  - stable_text       │  │  - full_text         │
│  - unstable_text     │  │  - segments          │
│  - confidence        │  │  - timestamps        │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐
│  translation_draft   │  │  translation_final   │
│  (fast, streaming)   │  │  (quality, batched)  │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └────────────┬────────────┘
                        ▼
           ┌────────────────────────┐
           │   Frontend Display     │
           │  ┌──────────────────┐  │
           │  │ Black: Stable    │  │
           │  │ Grey:  Unstable  │  │
           │  └──────────────────┘  │
           └────────────────────────┘
```

---

## Success Metrics

### Latency
- **Draft Emission**: < 500ms after tokens become stable
- **Final Emission**: < 1s after segment boundary
- **End-to-End**: < 2s from speech to translation

### Quality
- **Stability Accuracy**: > 95% of stable tokens don't change
- **Segment Accuracy**: > 98% of final segments correct
- **Translation Quality**: BLEU score comparable to offline

### User Experience
- **Smooth Updates**: No jarring rewrites
- **Clear Feedback**: Visual distinction between stable/unstable
- **Responsive**: Feels real-time, not laggy

---

## Future Enhancements

1. **Adaptive Stability**: Adjust thresholds based on confidence trends
2. **Speaker Diarization Integration**: Track stability per speaker
3. **Multi-language Support**: Language-specific stability rules
4. **Context Carryover**: Maintain context across segments
5. **Hallucination Detection**: Detect and suppress hallucinated tokens

---

## References

- SimulStreaming Paper: https://arxiv.org/abs/2406.04541
- AlignAtt Implementation: `reference/SimulStreaming/`
- Whisper Live: `reference/vexa/services/WhisperLive/`
- Token Buffer: `reference/SimulStreaming/token_buffer.py`
