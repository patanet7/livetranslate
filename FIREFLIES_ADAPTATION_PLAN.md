# Fireflies.ai Integration & Real-Time Caption Overlay

> **Status**: ✅ Phase 5 Complete - Core Services Ready
> **Last Updated**: 2026-01-10
> **Target**: Replace internal meeting bot + transcription with Fireflies.ai, add real-time translated captions to OBS/screen overlay
>
> ### Progress Summary
> | Phase | Status | Details |
> |-------|--------|---------|
> | Phase 1: Fireflies Integration | ✅ Complete | Client, router, models all implemented |
> | Phase 2: Sentence Aggregation | ✅ Complete | Hybrid boundary detection with 42 tests |
> | Phase 3: Rolling Window Translation | ✅ Complete | Context windows + glossary injection (34 tests) |
> | Phase 4: Glossary System | ✅ Complete | DB + service + 69 tests |
> | Phase 5: Caption Output | ✅ Complete | CaptionBuffer + speaker colors (46 tests) |
> | Phase 6: Testing | ✅ Strong | 295 tests (all passing, 0 skipped) |

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Fireflies API Integration](#fireflies-api-integration)
4. [Sentence Aggregation System](#sentence-aggregation-system)
5. [Rolling Window Translation](#rolling-window-translation)
6. [Glossary & Context System](#glossary--context-system)
7. [Caption Output Options](#caption-output-options)
8. [Database Schema](#database-schema)
9. [Implementation Checklist](#implementation-checklist)
10. [File Structure](#file-structure)
11. [Configuration](#configuration)
12. [References](#references)

---

## Executive Summary

### What We're Building

Replace the internal meeting bot + Whisper transcription pipeline with **Fireflies.ai** managed service, then:

1. **Receive** real-time transcripts via Fireflies WebSocket API
2. **Aggregate** chunks into complete sentences (hybrid boundary detection)
3. **Translate** with rolling context window + glossary injection
4. **Display** captions via OBS overlay / Electron transparent window

### Why This Approach

| Current System | New System |
|----------------|------------|
| Self-hosted meeting bot | Fireflies managed bot |
| Whisper transcription (NPU) | Fireflies transcription |
| Complex audio pipeline | Simple WebSocket integration |
| Maintenance overhead | API-based, minimal maintenance |

### What We Keep (90% Reusable)

- Translation Service (vLLM/Ollama/Groq)
- WebSocket infrastructure
- Database schema (transcripts, translations, speakers)
- Virtual webcam rendering engine
- Session management
- Configuration sync

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FIREFLIES.AI SERVICE                                │
│              (Bot joins meeting, transcription, speaker ID)                 │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ WebSocket (wss://...)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              FIREFLIES REALTIME CLIENT (NEW)                                │
│  modules/orchestration-service/src/clients/fireflies_client.py              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐          │
│  │ GraphQL Client  │  │ WebSocket Client │  │ Event Dispatcher   │          │
│  │ (active_meetings)│  │ (transcripts)    │  │ (internal routing) │          │
│  └─────────────────┘  └──────────────────┘  └────────────────────┘          │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────────┐  ┌─────────────────────────┐  ┌─────────────────────┐
│ SENTENCE          │  │ ROLLING WINDOW          │  │ DATABASE            │
│ AGGREGATOR (NEW)  │  │ TRANSLATOR (NEW)        │  │ (EXISTS)            │
│                   │  │                         │  │                     │
│ • Pause detection │  │ • Context window (3-5)  │  │ • transcripts       │
│ • Punctuation     │  │ • Glossary injection    │  │ • translations      │
│ • Speaker turns   │  │ • Multi-language        │  │ • speakers          │
│ • NLP (spaCy)     │  │ • Only output current   │  │ • glossaries (NEW)  │
│ • Buffer limits   │  │                         │  │ • sessions          │
└───────────────────┘  └─────────────────────────┘  └─────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CAPTION OUTPUT SYSTEM (NEW)                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ OPTION 1: OBS Integration                                             │  │
│  │   • obs-websocket-js → SetInputSettings (Text Source)                 │  │
│  │   • SendStreamCaption (CEA-608 for Twitch/YouTube)                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ OPTION 2: Electron Overlay                                            │  │
│  │   • Transparent, always-on-top window                                 │  │
│  │   • Click-through capability                                          │  │
│  │   • Works without OBS                                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ OPTION 3: Browser Source                                              │  │
│  │   • React component with transparent background                       │  │
│  │   • Add to OBS as Browser Source                                      │  │
│  │   • WebSocket connection to orchestration                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Fireflies API Integration

### API Endpoints Used

| Endpoint | Purpose | Documentation |
|----------|---------|---------------|
| GraphQL `active_meetings` | Get meeting IDs for connecting | [Docs](https://docs.fireflies.ai/graphql-api/query/active-meetings) |
| WebSocket Realtime | Receive live transcripts | [Docs](https://docs.fireflies.ai/realtime-api/overview) |

### GraphQL Query: Active Meetings

```graphql
query ActiveMeetings($email: String, $states: [MeetingState!]) {
  active_meetings(input: { email: $email, states: $states }) {
    id
    title
    organizer_email
    meeting_link
    start_time
    end_time
    privacy
    state
  }
}
```

### WebSocket Events

| Event | Description |
|-------|-------------|
| `auth.success` | Authentication confirmed |
| `auth.failed` | Auth failed, socket disconnects |
| `connection.established` | Ready to receive transcripts |
| `connection.error` | Connection or authorization error |
| `transcription.broadcast` | Live transcript segment |

### Transcript Event Schema

```json
{
  "transcript_id": "abc123",
  "chunk_id": "chunk_001",
  "text": "Hello world",
  "speaker_name": "Alice",
  "start_time": 0.0,
  "end_time": 1.25
}
```

**Key Fields:**
- `chunk_id`: Unique per segment, same ID = update to previous chunk
- `speaker_name`: Speaker attribution from Fireflies
- `start_time` / `end_time`: Timing in seconds (for pause detection)

---

## Sentence Aggregation System

### The Problem

Fireflies sends real-time chunks that may not align with sentence boundaries:

```
chunk_001: "So I think we should"          ← incomplete
chunk_002: "probably move forward with"    ← incomplete
chunk_003: "the implementation. Next we"   ← boundary mid-chunk!
chunk_004: "need to discuss the budget."   ← complete
```

Translating incomplete chunks produces poor quality translations.

### Solution: Hybrid Boundary Detection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SENTENCE AGGREGATOR PIPELINE                        │
│                                                                         │
│   Incoming Chunk                                                        │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ 1. SPEAKER CHANGE CHECK                                         │   │
│   │    If speaker_name != previous → FLUSH buffer, start new        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ 2. PAUSE DETECTION                                              │   │
│   │    If gap between chunks > 800ms → FLUSH (natural pause)        │   │
│   │    Uses: new_chunk.start_time - last_chunk.end_time             │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ 3. ADD TO BUFFER                                                │   │
│   │    Accumulate text, track timing, update word count             │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ 4. PUNCTUATION CHECK                                            │   │
│   │    If buffer ends with .?! → Extract sentence, TRANSLATE        │   │
│   │    Handle abbreviations: Dr. Mr. Mrs. etc. (don't split)        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ 5. NLP BOUNDARY DETECTION (if buffer > 10 words)                │   │
│   │    Use spaCy to find sentence boundaries mid-buffer             │   │
│   │    Extract complete sentences, keep remainder buffered          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ 6. BUFFER LIMITS (safety net)                                   │   │
│   │    If buffer > 30 words OR > 5 seconds → FORCE FLUSH            │   │
│   │    Find best break point, carry context forward                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        ▼                                                                │
│   Complete Sentence(s) → TRANSLATION                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Boundary Detection Accuracy

| Approach | Latency | Accuracy | Use Case |
|----------|---------|----------|----------|
| Punctuation only | 0ms | 60-70% | Only if ASR punctuates reliably |
| Pause detection | 0ms | 75-85% | Good baseline |
| Speaker turn | 0ms | N/A | Always use as boundary |
| NLP (spaCy) | 5-15ms | 90-95% | Best balance |
| **Hybrid (all above)** | 5-15ms | 92-95% | **Recommended** |

### Edge Cases Handled

| Edge Case | Solution |
|-----------|----------|
| Abbreviations (Dr. Mr. Inc.) | Regex negative lookbehind, abbreviation list |
| Decimals (3.50 dollars) | Pattern: `\d+\.\d+` excluded from split |
| Ellipsis (...) | Treat as soft boundary, wait for more |
| Mid-chunk boundary | NLP finds boundary, splits correctly |
| Run-on speech | Force flush at 30 words, find best break |
| Non-English punctuation | Unicode-aware: 。？！¿¡ |

### Implementation

```python
class SentenceAggregator:
    """
    Aggregates streaming transcript chunks into complete sentences.
    Uses multiple signals: punctuation, pauses, speaker turns, NLP.
    """

    def __init__(
        self,
        pause_threshold_ms: float = 800,
        max_buffer_words: int = 30,
        max_buffer_seconds: float = 5.0,
        min_words_for_translation: int = 3,
        nlp_threshold_words: int = 10,
    ):
        self.buffers: dict[str, SpeakerBuffer] = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.abbreviations = {'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'inc.', 'ltd.', 'etc.', 'e.g.', 'i.e.'}

    def process_chunk(self, chunk: FirefliesChunk) -> list[TranslationUnit]:
        """
        Process incoming chunk, return complete sentences ready for translation.
        """
        speaker = chunk.speaker_name
        results = []

        if speaker not in self.buffers:
            self.buffers[speaker] = SpeakerBuffer(speaker)

        buffer = self.buffers[speaker]

        # 1. Speaker change → flush previous speaker's buffer
        # 2. Pause detection (timing gap)
        if buffer.chunks and self._is_pause(buffer.last_chunk, chunk):
            results.extend(self._flush_buffer(buffer))

        # 3. Add to buffer
        buffer.add(chunk)

        # 4. Check for sentence-ending punctuation
        results.extend(self._extract_punctuated_sentences(buffer))

        # 5. NLP boundary detection
        if buffer.word_count >= self.nlp_threshold_words:
            results.extend(self._extract_nlp_sentences(buffer))

        # 6. Force flush if limits exceeded
        if self._exceeds_limits(buffer):
            results.extend(self._flush_buffer(buffer, force=True))

        return results
```

---

## Rolling Window Translation

### The Pattern

Context for understanding, but **only translate the current sentence**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ROLLING WINDOW TRANSLATION                           │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Previous Context (NOT translated, just for understanding):    │   │
│   │                                                                 │   │
│   │  [n-3] "We discussed the API changes yesterday."                │   │
│   │  [n-2] "John mentioned the authentication flow needs work."     │   │
│   │  [n-1] "He suggested using OAuth instead."                      │   │
│   │         ▲                                                       │   │
│   │         │ LLM reads this to understand "He" = John              │   │
│   │         │ and "it" = authentication flow                        │   │
│   └─────────┼───────────────────────────────────────────────────────┘   │
│             │                                                           │
│   ┌─────────▼───────────────────────────────────────────────────────┐   │
│   │  Current Sentence (THIS gets translated):                       │   │
│   │                                                                 │   │
│   │  [n] "I agree, it would simplify the whole process."            │   │
│   │       ▲                                                         │   │
│   │       └── "it" correctly understood as OAuth/auth flow         │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   Output: ONLY the translation of sentence [n]                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Translation Prompt Template

```python
TRANSLATION_PROMPT = """You are a professional real-time translator.

Target Language: {target_language}

Glossary (use these exact translations for these terms):
{glossary}

Previous context (DO NOT translate, only use for understanding references):
{context_window}

---

Translate ONLY the following sentence:
{current_sentence}

Translation:"""
```

### Example

**Glossary:**
```
API → API
OAuth → OAuth
sprint → sprint
deployment → déploiement
```

**Rolling Window:**
```
[n-2] "The sprint planning went well."
[n-1] "We decided to prioritize the OAuth integration."
```

**Current Sentence:**
```
[n] "It should be ready for deployment by Friday."
```

**LLM Output:**
```
Ça devrait être prêt pour le déploiement d'ici vendredi.
```

✅ "It" correctly understood as OAuth integration (from context)
✅ "deployment" uses glossary term "déploiement"
✅ Only current sentence translated

### Implementation

```python
class RollingWindowTranslator:
    def __init__(
        self,
        window_size: int = 3,
        translation_service: TranslationServiceClient,
        glossary_service: GlossaryService,
    ):
        self.window_size = window_size
        self.speaker_windows: dict[str, deque[str]] = {}
        self.global_window: deque[str] = deque(maxlen=window_size)

    async def translate(
        self,
        sentence: str,
        speaker: str,
        session_id: str,
        target_language: str,
    ) -> TranslationResult:
        # Get speaker's context window
        if speaker not in self.speaker_windows:
            self.speaker_windows[speaker] = deque(maxlen=self.window_size)
        speaker_window = self.speaker_windows[speaker]

        # Build context
        context_sentences = list(speaker_window)
        context_window = "\n".join(context_sentences) if context_sentences else "(No previous context)"

        # Load glossary
        glossary = await self.glossary_service.get_glossary(session_id, target_language)

        # Build prompt
        prompt = TRANSLATION_PROMPT.format(
            target_language=target_language,
            glossary=self._format_glossary(glossary),
            context_window=context_window,
            current_sentence=sentence
        )

        # Translate
        result = await self.translation_service.translate_with_prompt(prompt)

        # Update windows AFTER translation
        speaker_window.append(sentence)
        self.global_window.append(sentence)

        return TranslationResult(
            original=sentence,
            translated=result.text.strip(),
            speaker=speaker,
            target_language=target_language,
            confidence=result.confidence,
        )
```

### Why This Approach

| Approach | Problem |
|----------|---------|
| Translate each chunk independently | Loses context, pronouns wrong |
| Re-translate entire window each time | Slow, expensive, inconsistent |
| Translate window, extract last sentence | Hard to extract cleanly |
| **Context + translate only current** | ✅ Fast, consistent, context-aware |

---

## Glossary & Context System

### Database Table

```sql
CREATE TABLE bot_sessions.glossaries (
    glossary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES bot_sessions.sessions(session_id),  -- NULL for global
    source_term TEXT NOT NULL,
    target_term TEXT NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    domain VARCHAR(50),  -- 'medical', 'legal', 'tech', etc.
    priority INT DEFAULT 0,  -- Higher = preferred when conflicts
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(session_id, source_term, target_language)
);

CREATE INDEX idx_glossaries_session ON bot_sessions.glossaries(session_id);
CREATE INDEX idx_glossaries_language ON bot_sessions.glossaries(target_language);
CREATE INDEX idx_glossaries_domain ON bot_sessions.glossaries(domain);
```

### Glossary Service

```python
class GlossaryService:
    async def get_glossary(
        self,
        session_id: str,
        target_language: str,
        domain: str = None
    ) -> dict[str, str]:
        """
        Get glossary terms for translation.
        Merges global + session-specific, session takes priority.
        """
        # Global glossary (session_id IS NULL)
        global_terms = await self._get_terms(None, target_language, domain)

        # Session-specific glossary
        session_terms = await self._get_terms(session_id, target_language, domain)

        # Merge (session overrides global)
        return {**global_terms, **session_terms}

    async def add_term(
        self,
        source_term: str,
        target_term: str,
        target_language: str,
        session_id: str = None,
        domain: str = None
    ):
        """Add a glossary term."""
        ...
```

---

## Caption Output Options

### Option 1: OBS WebSocket Integration (Recommended for Streaming)

```python
import obsws_python as obs

class OBSCaptionOutput:
    def __init__(self, host: str = "localhost", port: int = 4455, password: str = ""):
        self.client = obs.ReqClient(host=host, port=port, password=password)
        self.text_source_name = "CaptionText"

    async def update_caption(self, text: str, speaker: str = None):
        """Update OBS text source with caption."""
        display_text = f"{speaker}: {text}" if speaker else text

        self.client.set_input_settings(
            name=self.text_source_name,
            settings={"text": display_text},
            overlay=True
        )

    async def send_stream_caption(self, text: str):
        """Send CEA-608 caption for Twitch/YouTube."""
        self.client.send_stream_caption(caption_text=text)
```

### Option 2: Electron Transparent Overlay

```javascript
// Main process
const { BrowserWindow } = require('electron');

const overlay = new BrowserWindow({
  width: 1200,
  height: 200,
  transparent: true,
  frame: false,
  alwaysOnTop: true,
  skipTaskbar: true,
  focusable: false,
  hasShadow: false,
  webPreferences: {
    nodeIntegration: true
  }
});

// Click-through
overlay.setIgnoreMouseEvents(true, { forward: true });
overlay.setAlwaysOnTop(true, 'screen-saver');
```

### Option 3: Browser Source (Reuses Frontend)

React component with transparent background, add to OBS as Browser Source.

```tsx
// CaptionOverlay.tsx
export const CaptionOverlay: React.FC = () => {
  const [captions, setCaptions] = useState<Caption[]>([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:3000/api/captions/stream');
    ws.onmessage = (event) => {
      const caption = JSON.parse(event.data);
      setCaptions(prev => [...prev.slice(-4), caption]);
    };
    return () => ws.close();
  }, []);

  return (
    <div className="caption-overlay" style={{ background: 'transparent' }}>
      {captions.map(caption => (
        <CaptionLine key={caption.id} caption={caption} />
      ))}
    </div>
  );
};
```

---

## Database Schema

### New Tables Required

```sql
-- Fireflies sessions
CREATE TABLE bot_sessions.fireflies_sessions (
    fireflies_session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES bot_sessions.sessions(session_id),
    fireflies_transcript_id VARCHAR(100) NOT NULL,
    fireflies_meeting_id VARCHAR(100),
    connection_status VARCHAR(20) DEFAULT 'disconnected',
    last_chunk_id VARCHAR(100),
    last_chunk_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Glossaries (as defined above)
CREATE TABLE bot_sessions.glossaries (
    glossary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES bot_sessions.sessions(session_id),
    source_term TEXT NOT NULL,
    target_term TEXT NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    domain VARCHAR(50),
    priority INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, source_term, target_language)
);
```

### Existing Tables Used

- `bot_sessions.sessions` - Session management
- `bot_sessions.transcripts` - Store incoming transcripts (source_type = 'fireflies')
- `bot_sessions.translations` - Store translations
- `bot_sessions.speaker_identities` - Speaker mapping

---

## Implementation Checklist

### Phase 1: Fireflies Integration (Core) ✅ COMPLETE

- [x] **1.1** Create Fireflies client module
  - [x] GraphQL client for `active_meetings` query
  - [x] WebSocket client for realtime transcripts
  - [x] Event parsing (auth, connection, transcription)
  - [x] Automatic reconnection with exponential backoff
  - [x] Chunk deduplication using `chunk_id`
  - **File**: `src/clients/fireflies_client.py`

- [x] **1.2** Create Fireflies router
  - [x] `POST /api/fireflies/connect` - Start session with transcript ID
  - [x] `POST /api/fireflies/disconnect` - End session
  - [x] `GET /api/fireflies/meetings` - List active meetings
  - [x] `GET /api/fireflies/status` - Connection status
  - **File**: `src/routers/fireflies.py`

- [x] **1.3** Create Fireflies models
  - [x] `FirefliesChunk` - Incoming transcript chunk
  - [x] `FirefliesSession` - Session state
  - [x] `FirefliesSessionConfig` - API configuration
  - [x] `SpeakerBuffer` - Per-speaker chunk accumulator
  - [x] `TranslationUnit` - Complete sentence for translation
  - [x] `TranslationContext` - Context window + glossary
  - [x] `TranslationResult` - Translation output
  - [x] `CaptionEntry` / `CaptionBroadcast` - Caption display models
  - **File**: `src/models/fireflies.py`

### Phase 2: Sentence Aggregation ✅ COMPLETE

- [x] **2.1** Create sentence aggregator service
  - [x] Per-speaker buffer management
  - [x] Pause detection (timing gaps > 800ms threshold)
  - [x] Punctuation boundary detection
  - [x] Abbreviation handling (Dr., Mr., Inc., etc.)
  - [x] Decimal number handling (3.50 not split)
  - [x] Ellipsis handling (... not treated as boundary)
  - [x] Buffer limits (30 words / 5 seconds)
  - **File**: `src/services/sentence_aggregator.py`

- [x] **2.2** Add NLP sentence boundary detection
  - [x] spaCy integration (lazy loading)
  - [x] Mid-buffer sentence extraction
  - [x] Remainder carryover

- [x] **2.3** Create translation unit producer
  - [x] Package complete sentences for translation
  - [x] Include speaker, timing, metadata
  - [x] Boundary type tracking (punctuation, pause, speaker_change, nlp, forced)

- [x] **2.4** Unit tests
  - [x] 42 comprehensive tests passing
  - **File**: `tests/fireflies/unit/test_sentence_aggregator.py`

### Phase 3: Rolling Window Translation ✅ COMPLETE

- [x] **3.1** Create rolling window translator
  - [x] Per-speaker context windows (deque-based)
  - [x] Global cross-speaker context window
  - [x] Context window size configuration (default: 3)
  - **File**: `src/services/rolling_window_translator.py`

- [x] **3.2** Implement translation prompt
  - [x] Template with context + glossary + sentence
  - [x] LLM output parsing with fallback
  - [x] Confidence extraction from response
  - [x] Glossary term injection

- [x] **3.3** Integrate with existing translation service
  - [x] Use existing `TranslationServiceClient`
  - [x] Prompt-based translation method
  - [x] Multi-language parallel translation support

- [x] **3.4** Unit tests
  - [x] 34 comprehensive tests passing
  - **File**: `tests/fireflies/unit/test_rolling_window_translator.py`

### Phase 4: Glossary System ✅ COMPLETE

- [x] **4.1** Create database tables
  - [x] `glossaries` table with domain, language settings
  - [x] `glossary_entries` table with translations JSON
  - [x] Indexes for performance
  - [x] Alembic migration script
  - **File**: `src/database/models.py` (Glossary, GlossaryEntry)
  - **Migration**: `alembic/versions/001_initial_schema_and_glossary.py`
  - **Status**: Migration applied to PostgreSQL ✅

- [x] **4.2** Create glossary service ✅
  - [x] CRUD operations for glossaries and entries
  - [x] Global vs session-specific merge (session overrides default)
  - [x] Domain filtering
  - [x] Priority-based term selection
  - [x] Term matching with whole-word and case-sensitivity
  - [x] Bulk import support
  - [x] Integration with TranslationContext model
  - **File**: `src/services/glossary_service.py`
  - **Unit Tests**: `tests/fireflies/unit/test_glossary_service.py` (56 tests)
  - **Integration Tests**: `tests/fireflies/integration/test_glossary_integration.py` (13 tests)
  - **Status**: Fully implemented ✅

- [ ] **4.3** Create glossary API endpoints (Optional - for admin UI)
  - [ ] `GET /api/glossary/{session_id}` - Get terms
  - [ ] `POST /api/glossary` - Add term
  - [ ] `DELETE /api/glossary/{term_id}` - Remove term
  - [ ] `POST /api/glossary/import` - Bulk import

### Phase 5: Caption Output ✅ COMPLETE (Core)

- [x] **5.1** Create caption buffer service
  - [x] Display queue management with time-based expiration
  - [x] Timing control (configurable duration, min display time)
  - [x] Max captions limit with priority-based overflow
  - [x] Speaker color assignment (material design palette)
  - [x] Thread-safe operations (RLock)
  - [x] Callbacks for caption added/expired events
  - [x] SessionCaptionManager for multi-session support
  - **File**: `src/services/caption_buffer.py`
  - **Tests**: 46 tests passing
  - **File**: `tests/fireflies/unit/test_caption_buffer.py`

- [ ] **5.2** Create OBS WebSocket output (Optional)
  - [ ] Connection to OBS
  - [ ] Text source updates
  - [ ] Stream caption (CEA-608)

- [ ] **5.3** Create WebSocket broadcast endpoint (Optional)
  - [ ] `/api/captions/stream` WebSocket
  - [ ] Caption event format
  - [ ] Client connection management

- [ ] **5.4** (Optional) Create Electron overlay app
  - [ ] Transparent window
  - [ ] WebSocket client
  - [ ] Caption display component

- [ ] **5.5** Create browser source component (Optional)
  - [ ] React caption overlay
  - [ ] Transparent background styling
  - [ ] OBS Browser Source ready

### Phase 6: Testing & Integration 🚧 PARTIAL

- [x] **6.1** Unit tests
  - [x] Fireflies models tests (20 tests)
  - [x] Fireflies client tests (22 tests)
  - [x] Fireflies router tests (fixed - import chain issues resolved)
  - [x] Sentence aggregator tests (43 tests - NLP mock working)
  - [x] Rolling window translator tests (34 tests)
  - [x] Glossary service tests (56 tests)
  - [x] Caption buffer tests (46 tests)
  - **Files**: `tests/fireflies/unit/test_*.py`
  - **Status**: 240+ unit tests passing ✅

- [x] **6.2** Integration tests
  - [x] Glossary + TranslationContext integration (13 tests)
  - [x] Multi-language glossary handling
  - [x] Translation pipeline simulation
  - [x] Full translation pipeline integration (14 tests)
  - [x] Translation contract validation (14 tests)
  - [x] Fireflies integration tests (22 tests - fixed import issues)
  - [ ] Fireflies mock server (pending)
  - [ ] End-to-end flow test (pending)
  - [ ] OBS integration test (pending)
  - **Files**: `tests/fireflies/integration/test_*.py`
  - **Status**: 50+ integration tests passing ✅

- [ ] **6.3** Performance testing
  - [ ] Latency measurement (target: <500ms e2e)
  - [ ] Concurrent speaker handling
  - [ ] Long session memory profiling

**Total Tests: 281 (all passing)**

---

## File Structure

```
modules/orchestration-service/src/
├── clients/
│   ├── fireflies_client.py          # NEW: Fireflies API client
│   └── translation_service_client.py # EXISTS: Translation client
├── models/
│   └── fireflies.py                  # NEW: Fireflies data models
├── routers/
│   ├── fireflies.py                  # NEW: Fireflies API endpoints
│   ├── glossary.py                   # NEW: Glossary management API
│   └── captions.py                   # NEW: Caption streaming API
├── services/
│   ├── __init__.py                   # ✅ Services module init
│   ├── sentence_aggregator.py        # ✅ Chunk → sentence (42 tests)
│   ├── rolling_window_translator.py  # ✅ Context-aware translation (34 tests)
│   ├── glossary_service.py           # ✅ Glossary management (69 tests)
│   └── caption_buffer.py             # ✅ Caption display queue (46 tests)
├── outputs/
│   ├── obs_output.py                 # NEW: OBS WebSocket integration
│   └── websocket_output.py           # NEW: WebSocket broadcast
└── database/
    └── migrations/
        └── 003_fireflies_glossary.sql # NEW: Schema additions

modules/frontend-service/src/
├── components/
│   └── CaptionOverlay/               # NEW: Browser source component
│       ├── CaptionOverlay.tsx
│       ├── CaptionLine.tsx
│       └── CaptionOverlay.css
└── pages/
    └── Captions/                     # NEW: Caption settings page
        └── CaptionSettings.tsx

caption-overlay/                      # NEW: Optional Electron app
├── main.js
├── preload.js
├── renderer/
│   ├── index.html
│   └── overlay.js
└── package.json

scripts/
└── migrations/
    └── 003_fireflies_glossary.sql    # NEW: Database migration
```

---

## Configuration

```python
@dataclass
class FirefliesConfig:
    # Fireflies API
    api_key: str
    graphql_endpoint: str = "https://api.fireflies.ai/graphql"
    websocket_endpoint: str = "wss://api.fireflies.ai/realtime"

    # Sentence aggregation
    pause_threshold_ms: float = 800
    max_buffer_words: int = 30
    max_buffer_seconds: float = 5.0
    min_words_for_translation: int = 3
    use_nlp_boundary_detection: bool = True

    # Rolling window translation
    context_window_size: int = 3
    include_cross_speaker_context: bool = True

    # Translation
    target_languages: list[str] = field(default_factory=lambda: ["es", "fr", "de"])
    translation_backend: str = "vllm"  # vllm, ollama, groq

    # Caption display
    max_captions_displayed: int = 5
    caption_duration_seconds: float = 8.0
    show_speaker_names: bool = True
    show_original_text: bool = False

    # OBS integration
    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = ""
    obs_text_source: str = "CaptionText"
```

---

## References

### Fireflies Documentation
- [Realtime API Overview](https://docs.fireflies.ai/realtime-api/overview)
- [Event Schema](https://docs.fireflies.ai/realtime-api/event-schema)
- [Active Meetings Query](https://docs.fireflies.ai/graphql-api/query/active-meetings)

### OBS Integration
- [OBS WebSocket Protocol](https://github.com/obsproject/obs-websocket)
- [obs-websocket-js](https://github.com/obsproject/obs-websocket-js)
- [LocalVocal Plugin](https://obsproject.com/forum/resources/localvocal-local-live-captions-translation-on-the-go.1769/)

### Translation & NLP
- [DeepL Glossary API](https://developers.deepl.com/api-reference/glossaries)
- [spaCy Sentence Segmentation](https://spacy.io/usage/linguistic-features#sbd)
- [LiveCaptions-Translator](https://github.com/SakiRinn/LiveCaptions-Translator)

### Caption Display
- [Electron Transparent Windows](https://www.electronjs.org/docs/latest/tutorial/window-customization)
- [OBS Browser Source](https://obsproject.com/wiki/Sources-Guide#browser-source)

---

## Notes

- **Fireflies API is in beta** - Features may change
- **spaCy model**: Use `en_core_web_sm` for speed, `en_core_web_lg` for accuracy
- **Translation latency target**: <500ms end-to-end
- **Multi-language**: Can translate to multiple languages simultaneously using existing translation service

---

*Document generated: 2025-01-08*
*Ready for implementation*
