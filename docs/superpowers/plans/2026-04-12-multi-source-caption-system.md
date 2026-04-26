# Multi-Source Caption System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the caption display system to support multiple audio/transcription sources (Local Mic, Fireflies, ScreenCaptureKit) through the loopback page with all display modes.

**Architecture:** Frontend uses source adapters to abstract WebSocket connections. Backend handles ScreenCaptureKit via subprocess. Unified caption store normalizes both `SegmentMessage` and `CaptionEvent` formats.

**Tech Stack:** SvelteKit 5 (runes), TypeScript, Python/FastAPI, Swift (ScreenCaptureKit), Socket.IO (Fireflies)

**Spec:** `docs/superpowers/specs/2026-04-12-multi-source-caption-system-design.md`

---

## File Structure

### Phase 1: Unified Store + Fireflies

| File | Action | Responsibility |
|------|--------|----------------|
| `modules/dashboard-service/src/lib/stores/caption.svelte.ts` | Create | Unified caption store accepting both formats |
| `modules/dashboard-service/src/lib/audio/source-adapter.ts` | Create | Source adapter abstraction + implementations |
| `modules/dashboard-service/src/lib/types/ws-messages.ts` | Modify | Add `source` field to `StartSessionMessage` |
| `modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte` | Modify | Source selector + meeting picker UI |
| `modules/dashboard-service/src/routes/(app)/loopback/+page.svelte` | Modify | Wire to new store + adapters |
| `modules/dashboard-service/src/lib/components/loopback/SplitView.svelte` | Modify | Import from new store |
| `modules/dashboard-service/src/lib/components/loopback/SubtitleView.svelte` | Modify | Import from new store |
| `modules/dashboard-service/src/lib/components/loopback/InterpreterView.svelte` | Modify | Import from new store |
| `modules/dashboard-service/src/lib/components/loopback/TranscriptView.svelte` | Modify | Import from new store |
| `modules/orchestration-service/src/routers/fireflies.py` | Modify | Enhance active-meetings endpoint |
| `modules/orchestration-service/src/meeting/pipeline.py` | Modify | Add auto_record parameter |
| `modules/orchestration-service/src/routers/captions.py` | Modify | Add protocol_version to connected event |

### Phase 2: ScreenCaptureKit

| File | Action | Responsibility |
|------|--------|----------------|
| `tools/screencapture/Package.swift` | Create | Swift package manifest |
| `tools/screencapture/Sources/main.swift` | Create | ScreenCaptureKit CLI entry point |
| `tools/screencapture/Sources/AudioCapture.swift` | Create | Core audio capture logic |
| `modules/orchestration-service/src/audio/screencapture_source.py` | Create | Python subprocess wrapper |
| `modules/orchestration-service/src/audio/__init__.py` | Create | Package init |
| `modules/orchestration-service/src/routers/system.py` | Create | System capabilities endpoint |
| `modules/orchestration-service/src/routers/audio/websocket_audio.py` | Modify | Handle screencapture source |

---

## Phase 1: Unified Store + Fireflies Integration

### Task 1: Create Unified Caption Store

**Files:**
- Create: `modules/dashboard-service/src/lib/stores/caption.svelte.ts`
- Test: `modules/dashboard-service/tests/unit/stores/caption.test.ts`

- [ ] **Step 1: Create test file with first test**

```typescript
// modules/dashboard-service/tests/unit/stores/caption.test.ts
import { describe, it, expect, beforeEach } from 'vitest';

// We'll import the store after creating it
describe('captionStore', () => {
  describe('ingestSegment', () => {
    it('should add a new caption from SegmentMessage', () => {
      // Test will be filled in after store exists
      expect(true).toBe(true);
    });
  });
});
```

- [ ] **Step 2: Run test to verify setup works**

Run: `cd modules/dashboard-service && npm test -- --run tests/unit/stores/caption.test.ts`
Expected: PASS (placeholder test)

- [ ] **Step 3: Create the unified caption store skeleton**

```typescript
// modules/dashboard-service/src/lib/stores/caption.svelte.ts
import { SPEAKER_COLORS } from '$lib/theme';
import type { SegmentMessage, TranslationMessage, TranslationChunkMessage } from '$lib/types/ws-messages';
import type { CaptionEvent, Caption } from '$lib/types/caption';

export type CaptionSource = 'local' | 'screencapture' | 'fireflies';
export type DisplayMode = 'split' | 'subtitle' | 'interpreter' | 'transcript';
export type TranslationState = 'pending' | 'draft' | 'streaming' | 'complete';
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface UnifiedCaption {
  id: string;
  text: string;
  stableText: string;
  unstableText: string;
  translation: string | null;
  translationState: TranslationState;
  speaker: string | null;
  speakerColor: string;
  language: string;
  confidence: number;
  timestamp: number;
  isFinal: boolean;
  isDraft: boolean;
}

const MAX_CAPTIONS = 5000;
const STORAGE_KEY = 'livetranslate:caption-config';

interface PersistedConfig {
  sourceLanguage: string | null;
  targetLanguage: string;
  displayMode: DisplayMode;
  captionSource: CaptionSource;
  interpreterLangA: string;
  interpreterLangB: string;
}

function createCaptionStore() {
  // State
  let captions = $state<UnifiedCaption[]>([]);
  let interimText = $state('');
  let interimConfidence = $state(0);
  let captionSource = $state<CaptionSource>('local');
  let connectionState = $state<ConnectionState>('disconnected');
  let firefliesSessionId = $state<string | null>(null);
  let displayMode = $state<DisplayMode>('split');
  let sourceLanguage = $state<string | null>(null);
  let targetLanguage = $state('zh');
  let detectedLanguage = $state<string | null>(null);
  let interpreterLangA = $state('zh');
  let interpreterLangB = $state('en');
  let transcriptionStatus = $state<'up' | 'down'>('down');
  let translationStatus = $state<'up' | 'down'>('down');
  let isCapturing = $state(false);
  let isRecording = $state(false);
  let recordingChunks = $state(0);
  let chunksSent = $state(0);
  let segmentsReceived = $state(0);
  let translationsReceived = $state(0);
  let lastError = $state<string | null>(null);
  let isMeetingActive = $state(false);
  let meetingSessionId = $state<string | null>(null);
  let meetingStartedAt = $state<string | null>(null);

  const speakerColorMap = new Map<string, string>();
  const seenCaptionIds = new Set<string>();

  function getSpeakerColor(speaker: string | null): string {
    if (!speaker) return SPEAKER_COLORS[0];
    if (!speakerColorMap.has(speaker)) {
      speakerColorMap.set(speaker, SPEAKER_COLORS[speakerColorMap.size % SPEAKER_COLORS.length]);
    }
    return speakerColorMap.get(speaker)!;
  }

  function persistConfig(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        sourceLanguage,
        targetLanguage,
        displayMode,
        captionSource,
        interpreterLangA,
        interpreterLangB,
      }));
    } catch { /* ignore */ }
  }

  function restoreConfig(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as Partial<PersistedConfig>;
      if (saved.sourceLanguage !== undefined) sourceLanguage = saved.sourceLanguage;
      if (saved.targetLanguage !== undefined) targetLanguage = saved.targetLanguage;
      if (saved.displayMode !== undefined) displayMode = saved.displayMode;
      if (saved.captionSource !== undefined) captionSource = saved.captionSource;
      if (saved.interpreterLangA !== undefined) interpreterLangA = saved.interpreterLangA;
      if (saved.interpreterLangB !== undefined) interpreterLangB = saved.interpreterLangB;
    } catch { /* ignore */ }
  }

  // Restore on creation
  if (typeof localStorage !== 'undefined') {
    restoreConfig();
  }

  function ingestSegment(msg: SegmentMessage): void {
    if (typeof msg.text !== 'string') return;
    segmentsReceived++;

    // Prefix ID for source isolation
    const id = `lb_${msg.segment_id}`;
    const existingIdx = captions.findIndex(c => c.id === id);

    const caption: UnifiedCaption = {
      id,
      text: [msg.stable_text, msg.unstable_text].filter(Boolean).join(' ') || msg.text,
      stableText: msg.stable_text ?? msg.text,
      unstableText: msg.unstable_text ?? '',
      translation: existingIdx >= 0 ? captions[existingIdx].translation : null,
      translationState: existingIdx >= 0 ? captions[existingIdx].translationState : 'pending',
      speaker: msg.speaker_id,
      speakerColor: getSpeakerColor(msg.speaker_id),
      language: msg.language,
      confidence: msg.confidence,
      timestamp: existingIdx >= 0 ? captions[existingIdx].timestamp : Date.now(),
      isFinal: msg.is_final,
      isDraft: msg.is_draft ?? false,
    };

    if (existingIdx >= 0) {
      const existing = captions[existingIdx];
      if (!existing.isDraft && caption.isDraft) return;
      captions = captions.map((c, i) => i === existingIdx ? caption : c);
    } else {
      captions = [...captions.slice(-(MAX_CAPTIONS - 1)), caption];
    }

    if (msg.is_final) {
      interimText = '';
      interimConfidence = 0;
    }
    if (msg.language) {
      detectedLanguage = msg.language;
    }
  }

  function ingestTranslation(msg: TranslationMessage): void {
    if (typeof msg.text !== 'string') return;
    translationsReceived++;

    const id = `lb_${msg.transcript_id}`;
    const caption = captions.find(c => c.id === id);
    if (!caption) return;

    const isDraft = msg.is_draft ?? false;
    if (caption.translationState === 'complete') return;
    if (isDraft && caption.translationState !== 'pending' && caption.translationState !== 'draft') return;

    caption.translation = msg.text;
    caption.translationState = isDraft ? 'draft' : 'complete';
  }

  function ingestTranslationChunk(msg: TranslationChunkMessage): void {
    const id = `lb_${msg.transcript_id}`;
    const caption = captions.find(c => c.id === id);
    if (!caption) return;
    if (caption.translationState === 'complete') return;

    if (caption.translationState === 'draft') {
      caption.translation = '';
    }
    caption.translationState = 'streaming';
    caption.translation = (caption.translation ?? '') + msg.delta;
  }

  function ingestCaptionEvent(event: CaptionEvent): void {
    if (event.event === 'caption_added' || event.event === 'caption_updated') {
      const cap = event.caption;
      const id = `ff_${cap.id}`;

      // Skip duplicates on reconnect
      if (event.event === 'caption_added' && seenCaptionIds.has(id)) return;
      seenCaptionIds.add(id);

      const existingIdx = captions.findIndex(c => c.id === id);

      const caption: UnifiedCaption = {
        id,
        text: cap.original_text || cap.text,
        stableText: cap.original_text || cap.text,
        unstableText: '',
        translation: cap.translated_text !== cap.original_text ? cap.translated_text : null,
        translationState: cap.translated_text ? 'complete' : 'pending',
        speaker: cap.speaker_name,
        speakerColor: cap.speaker_color || getSpeakerColor(cap.speaker_name),
        language: cap.target_language || 'auto',
        confidence: cap.confidence,
        timestamp: cap.receivedAt || Date.now(),
        isFinal: true,
        isDraft: false,
      };

      if (existingIdx >= 0) {
        captions = captions.map((c, i) => i === existingIdx ? caption : c);
      } else {
        captions = [...captions.slice(-(MAX_CAPTIONS - 1)), caption];
        segmentsReceived++;
        if (caption.translation) translationsReceived++;
      }
    } else if (event.event === 'session_cleared') {
      captions = [];
      seenCaptionIds.clear();
    }
    // caption_expired: keep for history
  }

  function ingestInterim(text: string, confidence: number): void {
    interimText = text;
    interimConfidence = confidence;
  }

  function startMeeting(sessionId: string, startedAt: string): void {
    isMeetingActive = true;
    meetingSessionId = sessionId;
    meetingStartedAt = startedAt;
  }

  function endMeeting(): void {
    isMeetingActive = false;
    meetingSessionId = null;
    meetingStartedAt = null;
    isRecording = false;
    recordingChunks = 0;
  }

  function clear(): void {
    captions = [];
    interimText = '';
    interimConfidence = 0;
    chunksSent = 0;
    segmentsReceived = 0;
    translationsReceived = 0;
    lastError = null;
    speakerColorMap.clear();
    seenCaptionIds.clear();
  }

  return {
    get captions() { return captions; },
    get interimText() { return interimText; },
    get interimConfidence() { return interimConfidence; },
    get captionSource() { return captionSource; },
    set captionSource(v: CaptionSource) { captionSource = v; persistConfig(); },
    get connectionState() { return connectionState; },
    set connectionState(v: ConnectionState) { connectionState = v; },
    get firefliesSessionId() { return firefliesSessionId; },
    set firefliesSessionId(v: string | null) { firefliesSessionId = v; },
    get displayMode() { return displayMode; },
    set displayMode(v: DisplayMode) { displayMode = v; persistConfig(); },
    get sourceLanguage() { return sourceLanguage; },
    set sourceLanguage(v: string | null) { sourceLanguage = v; persistConfig(); },
    get targetLanguage() { return targetLanguage; },
    set targetLanguage(v: string) { targetLanguage = v; persistConfig(); },
    get detectedLanguage() { return detectedLanguage; },
    set detectedLanguage(v: string | null) { detectedLanguage = v; },
    get interpreterLangA() { return interpreterLangA; },
    set interpreterLangA(v: string) { interpreterLangA = v; persistConfig(); },
    get interpreterLangB() { return interpreterLangB; },
    set interpreterLangB(v: string) { interpreterLangB = v; persistConfig(); },
    get transcriptionStatus() { return transcriptionStatus; },
    set transcriptionStatus(v: 'up' | 'down') { transcriptionStatus = v; },
    get translationStatus() { return translationStatus; },
    set translationStatus(v: 'up' | 'down') { translationStatus = v; },
    get isCapturing() { return isCapturing; },
    set isCapturing(v: boolean) { isCapturing = v; },
    get isRecording() { return isRecording; },
    set isRecording(v: boolean) { isRecording = v; },
    get recordingChunks() { return recordingChunks; },
    set recordingChunks(v: number) { recordingChunks = v; },
    get chunksSent() { return chunksSent; },
    set chunksSent(v: number) { chunksSent = v; },
    get segmentsReceived() { return segmentsReceived; },
    get translationsReceived() { return translationsReceived; },
    get lastError() { return lastError; },
    set lastError(v: string | null) { lastError = v; },
    get isMeetingActive() { return isMeetingActive; },
    get meetingSessionId() { return meetingSessionId; },
    get meetingStartedAt() { return meetingStartedAt; },
    getSpeakerColor,
    ingestSegment,
    ingestTranslation,
    ingestTranslationChunk,
    ingestCaptionEvent,
    ingestInterim,
    startMeeting,
    endMeeting,
    clear,
    restoreConfig,
  };
}

export const captionStore = createCaptionStore();
```

- [ ] **Step 4: Write real tests for ingestSegment**

```typescript
// modules/dashboard-service/tests/unit/stores/caption.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { captionStore } from '$lib/stores/caption.svelte';
import type { SegmentMessage } from '$lib/types/ws-messages';

describe('captionStore', () => {
  beforeEach(() => {
    captionStore.clear();
  });

  describe('ingestSegment', () => {
    it('should add a new caption from SegmentMessage', () => {
      const msg: SegmentMessage = {
        type: 'segment',
        segment_id: 1,
        text: 'Hello world',
        stable_text: 'Hello',
        unstable_text: 'world',
        language: 'en',
        confidence: 0.95,
        is_final: false,
        is_draft: false,
        speaker_id: 'speaker1',
        start_ms: 0,
        end_ms: 1000,
      };

      captionStore.ingestSegment(msg);

      expect(captionStore.captions.length).toBe(1);
      expect(captionStore.captions[0].id).toBe('lb_1');
      expect(captionStore.captions[0].text).toBe('Hello world');
      expect(captionStore.captions[0].speaker).toBe('speaker1');
    });

    it('should update existing caption with same segment_id', () => {
      const msg1: SegmentMessage = {
        type: 'segment',
        segment_id: 1,
        text: 'Hello',
        stable_text: 'Hello',
        unstable_text: '',
        language: 'en',
        confidence: 0.9,
        is_final: false,
        is_draft: true,
        speaker_id: null,
        start_ms: null,
        end_ms: null,
      };

      const msg2: SegmentMessage = {
        ...msg1,
        text: 'Hello world',
        stable_text: 'Hello world',
        is_draft: false,
        confidence: 0.95,
      };

      captionStore.ingestSegment(msg1);
      captionStore.ingestSegment(msg2);

      expect(captionStore.captions.length).toBe(1);
      expect(captionStore.captions[0].text).toBe('Hello world');
      expect(captionStore.captions[0].isDraft).toBe(false);
    });

    it('should not overwrite final with draft', () => {
      const final: SegmentMessage = {
        type: 'segment',
        segment_id: 1,
        text: 'Final text',
        stable_text: 'Final text',
        unstable_text: '',
        language: 'en',
        confidence: 0.95,
        is_final: true,
        is_draft: false,
        speaker_id: null,
        start_ms: null,
        end_ms: null,
      };

      const draft: SegmentMessage = {
        ...final,
        text: 'Draft text',
        is_draft: true,
      };

      captionStore.ingestSegment(final);
      captionStore.ingestSegment(draft);

      expect(captionStore.captions[0].text).toBe('Final text');
    });
  });

  describe('ingestCaptionEvent', () => {
    it('should add caption from Fireflies event', () => {
      captionStore.ingestCaptionEvent({
        event: 'caption_added',
        caption: {
          id: 'ff123',
          text: 'Translated text',
          original_text: 'Original text',
          translated_text: 'Translated text',
          speaker_name: 'Alice',
          speaker_color: '#ff0000',
          target_language: 'zh',
          confidence: 0.9,
          duration_seconds: 4,
          created_at: new Date().toISOString(),
          expires_at: new Date().toISOString(),
        },
      });

      expect(captionStore.captions.length).toBe(1);
      expect(captionStore.captions[0].id).toBe('ff_ff123');
      expect(captionStore.captions[0].text).toBe('Original text');
      expect(captionStore.captions[0].translation).toBe('Translated text');
    });

    it('should skip duplicate captions on reconnect', () => {
      const event = {
        event: 'caption_added' as const,
        caption: {
          id: 'ff123',
          text: 'Text',
          original_text: 'Text',
          translated_text: 'Text',
          speaker_name: 'Alice',
          speaker_color: '#ff0000',
          target_language: 'zh',
          confidence: 0.9,
          duration_seconds: 4,
          created_at: new Date().toISOString(),
          expires_at: new Date().toISOString(),
        },
      };

      captionStore.ingestCaptionEvent(event);
      captionStore.ingestCaptionEvent(event);

      expect(captionStore.captions.length).toBe(1);
    });
  });
});
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd modules/dashboard-service && npm test -- --run tests/unit/stores/caption.test.ts`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add modules/dashboard-service/src/lib/stores/caption.svelte.ts modules/dashboard-service/tests/unit/stores/caption.test.ts
git commit -m "feat(dashboard): add unified caption store

Supports both SegmentMessage (loopback) and CaptionEvent (Fireflies) formats.
Prefixes IDs by source (lb_, ff_) to prevent collisions.
Tracks seen caption IDs to prevent duplicates on reconnect."
```

---

### Task 2: Create Source Adapter Abstraction

**Files:**
- Create: `modules/dashboard-service/src/lib/audio/source-adapter.ts`
- Test: `modules/dashboard-service/tests/unit/audio/source-adapter.test.ts`

- [ ] **Step 1: Create source adapter types and interfaces**

```typescript
// modules/dashboard-service/src/lib/audio/source-adapter.ts
import { captionStore, type ConnectionState } from '$lib/stores/caption.svelte';
import { WS_BASE } from '$lib/config';
import type { ServerMessage } from '$lib/types/ws-messages';
import type { CaptionEvent } from '$lib/types/caption';

export interface SourceConfig {
  sourceLanguage: string | null;
  targetLanguage: string;
  interpreterLanguages?: [string, string];
  deviceId?: string;
  sampleRate?: number;
  channels?: number;
  firefliesSessionId?: string;
}

export interface SourceAdapter {
  connect(config: SourceConfig): Promise<void>;
  disconnect(): void;
  sendConfig(config: Record<string, unknown>): void;
  sendAudio?(data: Float32Array): void;
}

/**
 * Loopback adapter - connects to /ws/loopback for browser mic or screencapture.
 */
export class LoopbackAdapter implements SourceAdapter {
  private ws: WebSocket | null = null;
  private source: 'mic' | 'screencapture';
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(source: 'mic' | 'screencapture' = 'mic') {
    this.source = source;
  }

  async connect(config: SourceConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `${WS_BASE}/ws/loopback`;
      this.ws = new WebSocket(url);
      let settled = false;

      this.ws.onopen = () => {
        captionStore.connectionState = 'connecting';
        this.reconnectAttempts = 0;
        this.ws?.send(JSON.stringify({
          type: 'start_session',
          sample_rate: config.sampleRate ?? 48000,
          channels: config.channels ?? 1,
          device_id: config.deviceId,
          source: this.source,
        }));
      };

      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as ServerMessage;
        this.handleMessage(msg, () => {
          if (!settled) {
            settled = true;
            resolve();
          }
        }, (err) => {
          if (!settled) {
            settled = true;
            reject(err);
          }
        });
      };

      this.ws.onerror = () => {
        captionStore.connectionState = 'error';
        if (!settled) {
          settled = true;
          reject(new Error('WebSocket connection failed'));
        }
      };

      this.ws.onclose = () => {
        captionStore.connectionState = 'disconnected';
      };
    });
  }

  private handleMessage(
    msg: ServerMessage,
    resolve?: () => void,
    reject?: (err: Error) => void
  ): void {
    switch (msg.type) {
      case 'connected':
        captionStore.connectionState = 'connected';
        resolve?.();
        break;
      case 'segment':
        captionStore.ingestSegment(msg);
        break;
      case 'interim':
        captionStore.ingestInterim(msg.text, msg.confidence);
        break;
      case 'translation':
        captionStore.ingestTranslation(msg);
        break;
      case 'translation_chunk':
        captionStore.ingestTranslationChunk(msg);
        break;
      case 'meeting_started':
        if (typeof msg.session_id === 'string' && typeof msg.started_at === 'string') {
          captionStore.startMeeting(msg.session_id, msg.started_at);
        }
        break;
      case 'recording_status':
        captionStore.isRecording = msg.recording;
        captionStore.recordingChunks = msg.chunks_written;
        break;
      case 'service_status':
        captionStore.transcriptionStatus = msg.transcription as 'up' | 'down';
        captionStore.translationStatus = msg.translation as 'up' | 'down';
        break;
      case 'language_detected':
        captionStore.detectedLanguage = msg.language;
        break;
      case 'error':
        captionStore.lastError = msg.message;
        if (!msg.recoverable) {
          reject?.(new Error(msg.message));
        }
        break;
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.send(JSON.stringify({ type: 'end_session' }));
      this.ws.close();
      this.ws = null;
    }
  }

  sendConfig(config: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'config', ...config }));
    }
  }

  sendAudio(data: Float32Array): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data.buffer);
    }
  }
}

/**
 * Fireflies adapter - connects to /api/captions/stream/{sessionId}.
 */
export class FirefliesAdapter implements SourceAdapter {
  private ws: WebSocket | null = null;

  async connect(config: SourceConfig): Promise<void> {
    const sessionId = config.firefliesSessionId;
    if (!sessionId) {
      throw new Error('Fireflies session ID required');
    }

    return new Promise((resolve, reject) => {
      const langParam = config.targetLanguage ? `?target_language=${config.targetLanguage}` : '';
      const url = `${WS_BASE}/api/captions/stream/${sessionId}${langParam}`;
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        captionStore.connectionState = 'connecting';
      };

      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as CaptionEvent;
        this.handleMessage(msg, resolve);
      };

      this.ws.onerror = () => {
        captionStore.connectionState = 'error';
        reject(new Error('Fireflies WebSocket connection failed'));
      };

      this.ws.onclose = () => {
        captionStore.connectionState = 'disconnected';
      };
    });
  }

  private handleMessage(msg: CaptionEvent, resolve?: () => void): void {
    if (msg.event === 'connected') {
      captionStore.connectionState = 'connected';
      // Ingest initial captions
      for (const cap of msg.current_captions) {
        captionStore.ingestCaptionEvent({ event: 'caption_added', caption: cap });
      }
      resolve?.();
    } else {
      captionStore.ingestCaptionEvent(msg);
    }
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }

  sendConfig(config: Record<string, unknown>): void {
    if (config.target_language && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        event: 'set_language',
        language: config.target_language,
      }));
    }
  }
}

/**
 * Factory function to create the appropriate adapter.
 */
export function createSourceAdapter(source: 'local' | 'screencapture' | 'fireflies'): SourceAdapter {
  switch (source) {
    case 'local':
      return new LoopbackAdapter('mic');
    case 'screencapture':
      return new LoopbackAdapter('screencapture');
    case 'fireflies':
      return new FirefliesAdapter();
  }
}
```

- [ ] **Step 2: Create basic test for factory function**

```typescript
// modules/dashboard-service/tests/unit/audio/source-adapter.test.ts
import { describe, it, expect } from 'vitest';
import { createSourceAdapter, LoopbackAdapter, FirefliesAdapter } from '$lib/audio/source-adapter';

describe('createSourceAdapter', () => {
  it('should create LoopbackAdapter for local source', () => {
    const adapter = createSourceAdapter('local');
    expect(adapter).toBeInstanceOf(LoopbackAdapter);
  });

  it('should create LoopbackAdapter for screencapture source', () => {
    const adapter = createSourceAdapter('screencapture');
    expect(adapter).toBeInstanceOf(LoopbackAdapter);
  });

  it('should create FirefliesAdapter for fireflies source', () => {
    const adapter = createSourceAdapter('fireflies');
    expect(adapter).toBeInstanceOf(FirefliesAdapter);
  });
});
```

- [ ] **Step 3: Run tests**

Run: `cd modules/dashboard-service && npm test -- --run tests/unit/audio/source-adapter.test.ts`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add modules/dashboard-service/src/lib/audio/source-adapter.ts modules/dashboard-service/tests/unit/audio/source-adapter.test.ts
git commit -m "feat(dashboard): add source adapter abstraction

LoopbackAdapter for mic/screencapture via /ws/loopback
FirefliesAdapter for Fireflies via /api/captions/stream
Factory function createSourceAdapter() for source selection"
```

---

### Task 3: Extend StartSessionMessage Protocol

**Files:**
- Modify: `modules/dashboard-service/src/lib/types/ws-messages.ts`

- [ ] **Step 1: Add source field to StartSessionMessage**

```typescript
// In modules/dashboard-service/src/lib/types/ws-messages.ts
// Find the StartSessionMessage interface and update it:

export interface StartSessionMessage {
  type: 'start_session';
  sample_rate: number;
  channels: number;
  encoding?: string;
  device_id?: string;
  source?: 'mic' | 'screencapture';  // NEW: audio source type
}
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/types/ws-messages.ts
git commit -m "feat(protocol): add source field to StartSessionMessage

Allows frontend to specify mic or screencapture audio source"
```

---

### Task 4: Enhance Fireflies Active Meetings Endpoint

**Files:**
- Modify: `modules/orchestration-service/src/routers/fireflies.py`
- Test: `modules/orchestration-service/tests/fireflies/unit/test_fireflies_router.py`

- [ ] **Step 1: Write test for auto-select logic**

```python
# Add to modules/orchestration-service/tests/fireflies/unit/test_fireflies_router.py

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_active_meetings_auto_select_single():
    """When exactly one meeting is active, auto_select should be True."""
    from models.fireflies import FirefliesMeeting, MeetingState
    from datetime import datetime, UTC
    
    mock_meeting = FirefliesMeeting(
        id="meeting123",
        title="Daily Standup",
        organizer_email="test@example.com",
        start_time=datetime.now(UTC),
        state=MeetingState.IN_PROGRESS,
    )
    
    with patch("routers.fireflies.FirefliesGraphQLClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.get_active_meetings = AsyncMock(return_value=[mock_meeting])
        
        # Import after patching
        from routers.fireflies import get_active_meetings
        from config import FirefliesSettings
        
        settings = FirefliesSettings(fireflies_api_key="test-key")
        result = await get_active_meetings(settings=settings)
        
        assert result["auto_select"] is True
        assert result["auto_select_id"] == "meeting123"
        assert result["count"] == 1


@pytest.mark.asyncio
async def test_active_meetings_no_auto_select_multiple():
    """When multiple meetings are active, auto_select should be False."""
    from models.fireflies import FirefliesMeeting, MeetingState
    from datetime import datetime, UTC
    
    meetings = [
        FirefliesMeeting(
            id="meeting1",
            title="Meeting 1",
            organizer_email="test@example.com",
            start_time=datetime.now(UTC),
            state=MeetingState.IN_PROGRESS,
        ),
        FirefliesMeeting(
            id="meeting2",
            title="Meeting 2",
            organizer_email="test@example.com",
            start_time=datetime.now(UTC),
            state=MeetingState.IN_PROGRESS,
        ),
    ]
    
    with patch("routers.fireflies.FirefliesGraphQLClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.get_active_meetings = AsyncMock(return_value=meetings)
        
        from routers.fireflies import get_active_meetings
        from config import FirefliesSettings
        
        settings = FirefliesSettings(fireflies_api_key="test-key")
        result = await get_active_meetings(settings=settings)
        
        assert result["auto_select"] is False
        assert result["auto_select_id"] is None
        assert result["count"] == 2
```

- [ ] **Step 2: Run test to verify it fails (endpoint doesn't return auto_select yet)**

Run: `cd modules/orchestration-service && uv run pytest tests/fireflies/unit/test_fireflies_router.py -v -k "auto_select"`
Expected: FAIL (KeyError: 'auto_select')

- [ ] **Step 3: Update the active-meetings endpoint**

Find the `get_active_meetings` endpoint in `modules/orchestration-service/src/routers/fireflies.py` and update the return statement to include auto_select logic:

```python
# In the get_active_meetings function, update the return statement:

    meeting_list = [
        {
            "id": m.id,
            "title": m.title or "Untitled Meeting",
            "started_at": m.start_time.isoformat() if m.start_time else None,
            "organizer": m.organizer_email,
            "state": m.state.value if m.state else "unknown",
        }
        for m in meetings
    ]

    return {
        "meetings": meeting_list,
        "count": len(meeting_list),
        "auto_select": len(meeting_list) == 1,
        "auto_select_id": meeting_list[0]["id"] if len(meeting_list) == 1 else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd modules/orchestration-service && uv run pytest tests/fireflies/unit/test_fireflies_router.py -v -k "auto_select"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/routers/fireflies.py modules/orchestration-service/tests/fireflies/unit/test_fireflies_router.py
git commit -m "feat(fireflies): add auto-select to active meetings endpoint

Returns auto_select=True when exactly one meeting is active.
Returns auto_select_id for automatic connection."
```

---

### Task 5: Add Protocol Version to Captions WebSocket

**Files:**
- Modify: `modules/orchestration-service/src/routers/captions.py`

- [ ] **Step 1: Update the connected event to include protocol_version**

In `modules/orchestration-service/src/routers/captions.py`, find the `caption_stream` WebSocket endpoint and update the initial message:

```python
# In the caption_stream function, update the send_json call:

        await websocket.send_json(
            {
                "event": "connected",
                "protocol_version": 1,  # ADD THIS LINE
                "session_id": session_id,
                "current_captions": [c.to_dict() for c in current_captions],
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
```

- [ ] **Step 2: Commit**

```bash
git add modules/orchestration-service/src/routers/captions.py
git commit -m "feat(captions): add protocol_version to connected event

Enables frontend version negotiation for future protocol changes"
```

---

### Task 6: Add Auto-Record to MeetingPipeline

**Files:**
- Modify: `modules/orchestration-service/src/meeting/pipeline.py`
- Test: `modules/orchestration-service/tests/meeting/test_pipeline.py`

- [ ] **Step 1: Write test for auto_record behavior**

```python
# Add to modules/orchestration-service/tests/meeting/test_pipeline.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile


@pytest.mark.asyncio
async def test_pipeline_auto_record_starts_recorder():
    """When auto_record=True, recorder should start on pipeline.start()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("meeting.pipeline.MeetingSessionManager") as MockSessionMgr:
            mock_session_mgr = MockSessionMgr.return_value
            mock_session_mgr.create_session = AsyncMock(return_value=123)
            
            from meeting.pipeline import MeetingPipeline
            
            pipeline = MeetingPipeline(
                session_manager=mock_session_mgr,
                recording_base_path=Path(tmpdir),
                source_type="loopback",
                sample_rate=48000,
                channels=2,
                auto_record=True,
            )
            
            await pipeline.start()
            
            assert pipeline.recorder is not None
            assert pipeline.recorder._running is True
            
            # Cleanup
            pipeline.recorder.stop()


@pytest.mark.asyncio
async def test_pipeline_no_auto_record():
    """When auto_record=False, recorder should not start."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("meeting.pipeline.MeetingSessionManager") as MockSessionMgr:
            mock_session_mgr = MockSessionMgr.return_value
            mock_session_mgr.create_session = AsyncMock(return_value=123)
            
            from meeting.pipeline import MeetingPipeline
            
            pipeline = MeetingPipeline(
                session_manager=mock_session_mgr,
                recording_base_path=Path(tmpdir),
                source_type="loopback",
                sample_rate=48000,
                channels=2,
                auto_record=False,
            )
            
            await pipeline.start()
            
            assert pipeline.recorder is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd modules/orchestration-service && uv run pytest tests/meeting/test_pipeline.py -v -k "auto_record"`
Expected: FAIL (no auto_record parameter)

- [ ] **Step 3: Add auto_record parameter to MeetingPipeline**

In `modules/orchestration-service/src/meeting/pipeline.py`, update the `__init__` and `start` methods:

```python
# In __init__, add auto_record parameter:
def __init__(
    self,
    session_manager: MeetingSessionManager,
    recording_base_path: Path,
    source_type: str = "loopback",
    sample_rate: int = 48000,
    channels: int = 2,
    auto_record: bool = True,  # NEW
):
    self.session_manager = session_manager
    self.recording_base_path = recording_base_path
    self.source_type = source_type
    self.sample_rate = sample_rate
    self.channels = channels
    self.auto_record = auto_record
    self.recorder: FlacChunkRecorder | None = None
    # ... rest of init

# In start(), add auto-record logic:
async def start(self) -> None:
    self.session_id = await self.session_manager.create_session(
        source_type=self.source_type,
        sample_rate=self.sample_rate,
        channels=self.channels,
    )
    
    if self.auto_record:
        from meeting.recorder import FlacChunkRecorder
        self.recorder = FlacChunkRecorder(
            session_id=str(self.session_id),
            base_path=self.recording_base_path,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        self.recorder.start()
        logger.info("auto_record_started", session_id=self.session_id)
    
    # ... rest of start
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd modules/orchestration-service && uv run pytest tests/meeting/test_pipeline.py -v -k "auto_record"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/meeting/pipeline.py modules/orchestration-service/tests/meeting/test_pipeline.py
git commit -m "feat(pipeline): add auto_record parameter

Records audio immediately on session start by default.
Quick captures no longer require promote_to_meeting."
```

---

### Task 7: Add Source Selector to Toolbar

**Files:**
- Modify: `modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte`

- [ ] **Step 1: Add source selector UI to Toolbar**

In `modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte`, add the source selector:

```svelte
<script lang="ts">
  // Add import at top
  import { captionStore } from '$lib/stores/caption.svelte';
  
  // Add state for screencapture availability
  let screenCaptureAvailable = $state(false);
  
  // Add to onMount
  onMount(async () => {
    // ... existing code ...
    
    // Check screencapture availability
    try {
      const res = await fetch('/api/system/screencapture-available');
      if (res.ok) {
        screenCaptureAvailable = (await res.json()).available;
      }
    } catch {
      screenCaptureAvailable = false;
    }
  });
</script>

<!-- Add source selector before device selector -->
<div class="source-selector">
  <span class="label">Source:</span>
  <div class="source-options">
    <label class="source-option">
      <input
        type="radio"
        name="source"
        value="local"
        checked={captionStore.captionSource === 'local'}
        onchange={() => captionStore.captionSource = 'local'}
        disabled={loopbackStore.isCapturing}
      />
      <span>Mic</span>
    </label>
    
    <label class="source-option" class:disabled={!screenCaptureAvailable}>
      <input
        type="radio"
        name="source"
        value="screencapture"
        checked={captionStore.captionSource === 'screencapture'}
        onchange={() => captionStore.captionSource = 'screencapture'}
        disabled={loopbackStore.isCapturing || !screenCaptureAvailable}
      />
      <span>System Audio</span>
      {#if !screenCaptureAvailable}
        <span class="badge">Install</span>
      {/if}
    </label>
    
    <label class="source-option">
      <input
        type="radio"
        name="source"
        value="fireflies"
        checked={captionStore.captionSource === 'fireflies'}
        onchange={() => captionStore.captionSource = 'fireflies'}
        disabled={loopbackStore.isCapturing}
      />
      <span>Fireflies</span>
    </label>
  </div>
</div>

<style>
  .source-selector {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .source-options {
    display: flex;
    gap: 12px;
  }
  
  .source-option {
    display: flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
    font-size: 13px;
  }
  
  .source-option.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .badge {
    font-size: 10px;
    padding: 1px 4px;
    background: #ef4444;
    color: white;
    border-radius: 3px;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte
git commit -m "feat(toolbar): add source selector UI

Allows switching between Mic, System Audio, and Fireflies sources.
System Audio shows Install badge when screencapture binary not available."
```

---

### Task 8: Add Fireflies Meeting Picker

**Files:**
- Modify: `modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte`

- [ ] **Step 1: Add meeting picker state and fetch logic**

```svelte
<script lang="ts">
  // Add meeting state
  interface FirefliesMeeting {
    id: string;
    title: string;
    started_at: string | null;
    organizer: string;
    state: string;
  }
  
  let meetings = $state<FirefliesMeeting[]>([]);
  let loadingMeetings = $state(false);
  let selectedMeetingId = $state('');
  
  // Fetch meetings when Fireflies source is selected
  $effect(() => {
    if (captionStore.captionSource === 'fireflies' && !loopbackStore.isCapturing) {
      fetchActiveMeetings();
    }
  });
  
  async function fetchActiveMeetings() {
    loadingMeetings = true;
    try {
      const res = await fetch('/api/fireflies/active-meetings');
      if (res.ok) {
        const data = await res.json();
        meetings = data.meetings;
        if (data.auto_select && data.auto_select_id) {
          selectedMeetingId = data.auto_select_id;
          captionStore.firefliesSessionId = data.auto_select_id;
        }
      }
    } catch (err) {
      console.error('Failed to fetch Fireflies meetings:', err);
      meetings = [];
    } finally {
      loadingMeetings = false;
    }
  }
  
  function formatTimeAgo(isoDate: string | null): string {
    if (!isoDate) return '';
    const diff = Date.now() - new Date(isoDate).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    return `${Math.floor(mins / 60)}h ago`;
  }
</script>

<!-- Add meeting picker after source selector when Fireflies is selected -->
{#if captionStore.captionSource === 'fireflies'}
  <div class="meeting-picker">
    {#if loadingMeetings}
      <span class="loading">Loading meetings...</span>
    {:else if meetings.length === 0}
      <span class="no-meetings">No active Fireflies meetings</span>
    {:else if meetings.length === 1}
      <span class="auto-selected">
        {meetings[0].title}
      </span>
    {:else}
      <select
        bind:value={selectedMeetingId}
        onchange={() => captionStore.firefliesSessionId = selectedMeetingId}
        disabled={loopbackStore.isCapturing}
      >
        <option value="">Select meeting...</option>
        {#each meetings as meeting}
          <option value={meeting.id}>
            {meeting.title} ({formatTimeAgo(meeting.started_at)})
          </option>
        {/each}
      </select>
    {/if}
  </div>
{/if}

<style>
  .meeting-picker {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .meeting-picker select {
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid #374151;
    background: #1f2937;
    color: #e5e7eb;
    font-size: 13px;
  }
  
  .auto-selected {
    font-size: 13px;
    color: #10b981;
  }
  
  .no-meetings {
    font-size: 13px;
    color: #ef4444;
  }
  
  .loading {
    font-size: 13px;
    color: #9ca3af;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte
git commit -m "feat(toolbar): add Fireflies meeting picker

Auto-selects when exactly one meeting active.
Shows picker dropdown for multiple meetings.
Fetches meetings when Fireflies source selected."
```

---

### Task 9: Wire Loopback Page to New Store and Adapters

**Files:**
- Modify: `modules/dashboard-service/src/routes/(app)/loopback/+page.svelte`

- [ ] **Step 1: Update imports and adapter usage**

This is a larger refactor. Update the loopback page to use the new captionStore and source adapters:

```svelte
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { beforeNavigate } from '$app/navigation';
  import { captionStore } from '$lib/stores/caption.svelte';
  import { createSourceAdapter, type SourceAdapter } from '$lib/audio/source-adapter';
  import { AudioCapture } from '$lib/audio/capture';
  import Toolbar from '$lib/components/loopback/Toolbar.svelte';
  import SplitView from '$lib/components/loopback/SplitView.svelte';
  import SubtitleView from '$lib/components/loopback/SubtitleView.svelte';
  import TranscriptView from '$lib/components/loopback/TranscriptView.svelte';
  import InterpreterView from '$lib/components/loopback/InterpreterView.svelte';

  let devices = $state<MediaDeviceInfo[]>([]);
  let selectedDeviceId = $state('');
  let elapsedTime = $state('00:00:00');
  let elapsedTimerInterval: ReturnType<typeof setInterval> | null = null;
  let captureError = $state<string | null>(null);
  let audioLevel = $state(0);

  let capture: AudioCapture | null = null;
  let adapter: SourceAdapter | null = null;
  let stopping = false;

  async function startCapture() {
    if (captionStore.isCapturing) return;
    captureError = null;

    // Create appropriate adapter based on source
    adapter = createSourceAdapter(captionStore.captionSource);

    try {
      // For local/screencapture, also start AudioCapture
      if (captionStore.captionSource === 'local') {
        capture = new AudioCapture();
        await capture.start({
          deviceId: selectedDeviceId || undefined,
          sourceType: 'mic',
          onChunk: (data) => {
            adapter?.sendAudio?.(data);
            captionStore.chunksSent++;
          },
          onError: (error) => {
            console.error('Audio capture error:', error);
            stopCapture();
          },
          onLevel: (rms) => {
            audioLevel = rms;
          },
        });
      }

      // Connect the adapter
      await adapter.connect({
        sourceLanguage: captionStore.sourceLanguage,
        targetLanguage: captionStore.targetLanguage,
        deviceId: selectedDeviceId,
        sampleRate: capture?.sampleRate ?? 48000,
        channels: 1,
        firefliesSessionId: captionStore.firefliesSessionId ?? undefined,
        interpreterLanguages: captionStore.displayMode === 'interpreter'
          ? [captionStore.interpreterLangA, captionStore.interpreterLangB]
          : undefined,
      });

      captionStore.isCapturing = true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      captureError = msg;
      captionStore.connectionState = 'error';
      capture?.stop();
      capture = null;
      adapter = null;
    }
  }

  async function stopCapture() {
    if (stopping) return;
    stopping = true;
    try {
      capture?.stop();
      capture = null;
      adapter?.disconnect();
      adapter = null;
      captionStore.isCapturing = false;
      captionStore.connectionState = 'disconnected';
      audioLevel = 0;
      stopElapsedTimer();
    } finally {
      stopping = false;
    }
  }

  function handleConfigChange(config: Record<string, unknown>) {
    adapter?.sendConfig(config);
  }

  function startElapsedTimer(startedAt: string) {
    stopElapsedTimer();
    const startTime = new Date(startedAt).getTime();
    elapsedTimerInterval = setInterval(() => {
      const diff = Date.now() - startTime;
      const hours = Math.floor(diff / 3600000).toString().padStart(2, '0');
      const minutes = Math.floor((diff % 3600000) / 60000).toString().padStart(2, '0');
      const seconds = Math.floor((diff % 60000) / 1000).toString().padStart(2, '0');
      elapsedTime = `${hours}:${minutes}:${seconds}`;
    }, 1000);
  }

  function stopElapsedTimer() {
    if (elapsedTimerInterval) {
      clearInterval(elapsedTimerInterval);
      elapsedTimerInterval = null;
    }
    elapsedTime = '00:00:00';
  }

  onMount(async () => {
    captionStore.isCapturing = false;
    captionStore.connectionState = 'disconnected';

    try {
      devices = await AudioCapture.getDevices();
      if (devices.length > 0 && !selectedDeviceId) {
        selectedDeviceId = devices[0].deviceId;
      }
    } catch (err) {
      console.error('Failed to enumerate audio devices:', err);
    }
  });

  beforeNavigate(() => {
    stopCapture();
  });

  onDestroy(() => {
    stopCapture();
  });
</script>

<!-- Rest of template stays mostly the same, but use captionStore instead of loopbackStore -->
<div class="loopback-page">
  <Toolbar
    {devices}
    bind:selectedDeviceId
    {audioLevel}
    onStartCapture={startCapture}
    onStopCapture={stopCapture}
    onConfigChange={handleConfigChange}
  />

  {#if captureError}
    <div class="capture-error" role="alert">
      <span>{captureError}</span>
      <button class="dismiss-btn" onclick={() => captureError = null}>&times;</button>
    </div>
  {/if}

  {#if captionStore.isMeetingActive}
    <div class="meeting-bar">
      <!-- ... meeting bar content using captionStore ... -->
    </div>
  {/if}

  <div class="display-area">
    {#if captionStore.displayMode === 'split'}
      <SplitView />
    {:else if captionStore.displayMode === 'subtitle'}
      <SubtitleView />
    {:else if captionStore.displayMode === 'interpreter'}
      <InterpreterView />
    {:else}
      <TranscriptView />
    {/if}
  </div>
</div>

<!-- ... styles ... -->
```

- [ ] **Step 2: Update display components to use captionStore**

Update each display component (`SplitView.svelte`, `SubtitleView.svelte`, `InterpreterView.svelte`, `TranscriptView.svelte`) to import from the new store:

```svelte
<!-- In each component, change: -->
<script lang="ts">
  // OLD:
  import { loopbackStore } from '$lib/stores/loopback.svelte';
  
  // NEW:
  import { captionStore } from '$lib/stores/caption.svelte';
</script>

<!-- And update all references from loopbackStore to captionStore -->
```

- [ ] **Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/loopback/+page.svelte
git add modules/dashboard-service/src/lib/components/loopback/SplitView.svelte
git add modules/dashboard-service/src/lib/components/loopback/SubtitleView.svelte
git add modules/dashboard-service/src/lib/components/loopback/InterpreterView.svelte
git add modules/dashboard-service/src/lib/components/loopback/TranscriptView.svelte
git commit -m "refactor(loopback): wire page to unified store and adapters

- Use captionStore instead of loopbackStore
- Use source adapters for connection management
- Support Fireflies source alongside local audio"
```

---

### Task 10: Integration Test - Fireflies Flow

**Files:**
- Create: `modules/dashboard-service/tests/integration/fireflies-flow.test.ts`

- [ ] **Step 1: Write integration test**

```typescript
// modules/dashboard-service/tests/integration/fireflies-flow.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { captionStore } from '$lib/stores/caption.svelte';

describe('Fireflies Integration Flow', () => {
  beforeEach(() => {
    captionStore.clear();
    captionStore.captionSource = 'fireflies';
  });

  it('should process Fireflies caption events into unified store', () => {
    // Simulate connected event with initial captions
    captionStore.ingestCaptionEvent({
      event: 'connected',
      session_id: 'test-session',
      current_captions: [
        {
          id: 'cap1',
          text: 'Hello',
          original_text: 'Hello',
          translated_text: 'Hola',
          speaker_name: 'Alice',
          speaker_color: '#ff0000',
          target_language: 'es',
          confidence: 0.95,
          duration_seconds: 4,
          created_at: new Date().toISOString(),
          expires_at: new Date().toISOString(),
        },
      ],
      timestamp: new Date().toISOString(),
    });

    expect(captionStore.captions.length).toBe(1);
    expect(captionStore.captions[0].id).toBe('ff_cap1');
    expect(captionStore.captions[0].text).toBe('Hello');
    expect(captionStore.captions[0].translation).toBe('Hola');

    // Simulate caption_added
    captionStore.ingestCaptionEvent({
      event: 'caption_added',
      caption: {
        id: 'cap2',
        text: 'World',
        original_text: 'World',
        translated_text: 'Mundo',
        speaker_name: 'Bob',
        speaker_color: '#00ff00',
        target_language: 'es',
        confidence: 0.9,
        duration_seconds: 4,
        created_at: new Date().toISOString(),
        expires_at: new Date().toISOString(),
      },
    });

    expect(captionStore.captions.length).toBe(2);
    expect(captionStore.captions[1].id).toBe('ff_cap2');
  });

  it('should handle source switching', () => {
    // Add Fireflies caption
    captionStore.ingestCaptionEvent({
      event: 'caption_added',
      caption: {
        id: 'ff1',
        text: 'Fireflies text',
        original_text: 'Fireflies text',
        translated_text: 'Texto de Fireflies',
        speaker_name: 'Speaker',
        speaker_color: '#ff0000',
        target_language: 'es',
        confidence: 0.95,
        duration_seconds: 4,
        created_at: new Date().toISOString(),
        expires_at: new Date().toISOString(),
      },
    });

    // Switch to local source
    captionStore.captionSource = 'local';

    // Add loopback segment
    captionStore.ingestSegment({
      type: 'segment',
      segment_id: 1,
      text: 'Local text',
      stable_text: 'Local text',
      unstable_text: '',
      language: 'en',
      confidence: 0.95,
      is_final: true,
      is_draft: false,
      speaker_id: null,
      start_ms: null,
      end_ms: null,
    });

    // Both captions should exist with different ID prefixes
    expect(captionStore.captions.length).toBe(2);
    expect(captionStore.captions[0].id).toBe('ff_ff1');
    expect(captionStore.captions[1].id).toBe('lb_1');
  });
});
```

- [ ] **Step 2: Run test**

Run: `cd modules/dashboard-service && npm test -- --run tests/integration/fireflies-flow.test.ts`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add modules/dashboard-service/tests/integration/fireflies-flow.test.ts
git commit -m "test(dashboard): add Fireflies integration flow tests

Verifies caption ingestion and source switching behavior"
```

---

## Phase 2: ScreenCaptureKit Integration

### Task 11: Create Swift CLI Project Structure

**Files:**
- Create: `tools/screencapture/Package.swift`
- Create: `tools/screencapture/Sources/main.swift`

- [ ] **Step 1: Create Swift package manifest**

```swift
// tools/screencapture/Package.swift
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "livetranslate-capture",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "livetranslate-capture",
            path: "Sources"
        )
    ]
)
```

- [ ] **Step 2: Create main.swift entry point**

```swift
// tools/screencapture/Sources/main.swift
import Foundation
import ScreenCaptureKit
import AVFoundation

// Command line arguments
struct Config {
    var sampleRate: Int = 48000
    var channels: Int = 1
    var format: String = "f32le"
    var listSources: Bool = false
    var device: String? = nil
}

func parseArgs() -> Config {
    var config = Config()
    var args = CommandLine.arguments.dropFirst()
    
    while let arg = args.popFirst() {
        switch arg {
        case "--sample-rate":
            if let value = args.popFirst(), let rate = Int(value) {
                config.sampleRate = rate
            }
        case "--channels":
            if let value = args.popFirst(), let ch = Int(value) {
                config.channels = ch
            }
        case "--format":
            if let value = args.popFirst() {
                config.format = value
            }
        case "--device":
            config.device = args.popFirst()
        case "--list-sources":
            config.listSources = true
        case "--help":
            printUsage()
            exit(0)
        default:
            break
        }
    }
    return config
}

func printUsage() {
    fputs("""
    Usage: livetranslate-capture [OPTIONS]
    
    Options:
      --sample-rate <HZ>   Sample rate (default: 48000)
      --channels <N>       Number of channels (default: 1)
      --format <FMT>       Output format: f32le (default)
      --device <NAME>      Specific audio device
      --list-sources       List available audio sources
      --help               Show this help
    
    Outputs raw PCM audio to stdout.
    """, stderr)
}

@main
struct LiveTranslateCapture {
    static func main() async {
        let config = parseArgs()
        
        if config.listSources {
            await listAudioSources()
            return
        }
        
        do {
            try await startCapture(config: config)
        } catch {
            fputs("Error: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
    
    static func listAudioSources() async {
        do {
            let content = try await SCShareableContent.current
            print("Available audio sources:")
            for app in content.applications {
                print("  - \(app.applicationName)")
            }
        } catch {
            fputs("Error listing sources: \(error.localizedDescription)\n", stderr)
            exit(2)
        }
    }
    
    static func startCapture(config: Config) async throws {
        let capture = try await AudioCapture(config: config)
        
        // Handle SIGTERM/SIGINT for graceful shutdown
        signal(SIGTERM) { _ in
            Task { await AudioCapture.shared?.stop() }
        }
        signal(SIGINT) { _ in
            Task { await AudioCapture.shared?.stop() }
        }
        
        try await capture.start()
        
        // Keep running until stopped
        await withCheckedContinuation { continuation in
            capture.onStop = { continuation.resume() }
        }
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd tools/screencapture && swift build`
Expected: Build succeeds (may have warnings about unimplemented AudioCapture)

- [ ] **Step 4: Commit**

```bash
git add tools/screencapture/Package.swift tools/screencapture/Sources/main.swift
git commit -m "feat(screencapture): add Swift CLI project structure

Command-line interface for ScreenCaptureKit audio capture.
Outputs raw f32le PCM to stdout."
```

---

### Task 12: Implement ScreenCaptureKit Audio Capture

**Files:**
- Create: `tools/screencapture/Sources/AudioCapture.swift`

- [ ] **Step 1: Implement AudioCapture class**

```swift
// tools/screencapture/Sources/AudioCapture.swift
import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia

actor AudioCapture: NSObject, SCStreamDelegate, SCStreamOutput {
    static var shared: AudioCapture?
    
    private let config: Config
    private var stream: SCStream?
    private var isRunning = false
    var onStop: (() -> Void)?
    
    init(config: Config) async throws {
        self.config = config
        super.init()
        AudioCapture.shared = self
    }
    
    func start() async throws {
        // Check permission
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(
                false,
                onScreenWindowsOnly: false
            )
        } catch {
            throw CaptureError.permissionDenied
        }
        
        // Find the display to capture audio from
        guard let display = content.displays.first else {
            throw CaptureError.noDisplays
        }
        
        // Configure stream for audio only
        let streamConfig = SCStreamConfiguration()
        streamConfig.capturesAudio = true
        streamConfig.excludesCurrentProcessAudio = false
        streamConfig.sampleRate = config.sampleRate
        streamConfig.channelCount = config.channels
        
        // Minimal video config (required but we ignore it)
        streamConfig.width = 2
        streamConfig.height = 2
        streamConfig.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        
        // Create content filter
        let filter = SCContentFilter(display: display, excludingWindows: [])
        
        // Create and start stream
        stream = SCStream(filter: filter, configuration: streamConfig, delegate: self)
        
        try stream?.addStreamOutput(self, type: .audio, sampleHandlerQueue: .global(qos: .userInteractive))
        
        try await stream?.startCapture()
        isRunning = true
        
        fputs("Capturing system audio at \(config.sampleRate)Hz, \(config.channels) channel(s)\n", stderr)
    }
    
    func stop() async {
        guard isRunning else { return }
        isRunning = false
        
        try? await stream?.stopCapture()
        stream = nil
        
        fputs("Capture stopped\n", stderr)
        onStop?()
    }
    
    // MARK: - SCStreamOutput
    
    nonisolated func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        
        // Extract audio data
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        
        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        
        let status = CMBlockBufferGetDataPointer(
            blockBuffer,
            atOffset: 0,
            lengthAtOffsetOut: nil,
            totalLengthOut: &length,
            dataPointerOut: &dataPointer
        )
        
        guard status == kCMBlockBufferNoErr, let data = dataPointer else { return }
        
        // Write raw PCM to stdout
        let bytesWritten = fwrite(data, 1, length, stdout)
        fflush(stdout)
        
        if bytesWritten != length {
            fputs("Warning: Only wrote \(bytesWritten) of \(length) bytes\n", stderr)
        }
    }
    
    // MARK: - SCStreamDelegate
    
    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("Stream stopped with error: \(error.localizedDescription)\n", stderr)
        Task { await self.stop() }
    }
}

enum CaptureError: Error, LocalizedError {
    case permissionDenied
    case noDisplays
    
    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Screen Recording permission denied. Grant access in System Settings > Privacy & Security > Screen Recording."
        case .noDisplays:
            return "No displays available for capture."
        }
    }
}
```

- [ ] **Step 2: Build and test locally**

Run: `cd tools/screencapture && swift build -c release`
Expected: Build succeeds

Test: `./build/release/livetranslate-capture --list-sources`
Expected: Lists available audio sources (or permission prompt)

- [ ] **Step 3: Commit**

```bash
git add tools/screencapture/Sources/AudioCapture.swift
git commit -m "feat(screencapture): implement ScreenCaptureKit audio capture

Captures system audio via ScreenCaptureKit and outputs raw f32le PCM.
Requires macOS 13+ and Screen Recording permission."
```

---

### Task 13: Create Backend ScreenCaptureAudioSource

**Files:**
- Create: `modules/orchestration-service/src/audio/__init__.py`
- Create: `modules/orchestration-service/src/audio/screencapture_source.py`
- Test: `modules/orchestration-service/tests/audio/test_screencapture_source.py`

- [ ] **Step 1: Create package init**

```python
# modules/orchestration-service/src/audio/__init__.py
"""Audio capture sources."""

from .screencapture_source import ScreenCaptureAudioSource

__all__ = ["ScreenCaptureAudioSource"]
```

- [ ] **Step 2: Create ScreenCaptureAudioSource**

```python
# modules/orchestration-service/src/audio/screencapture_source.py
"""ScreenCaptureKit audio source - spawns native capture and pipes to pipeline."""

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
from livetranslate_common.logging import get_logger

logger = get_logger()

CAPTURE_BINARY = "livetranslate-capture"
# 4096 samples * 4 bytes = 16384 bytes (matches browser worklet granularity)
CHUNK_SIZE = 16384
WATCHDOG_TIMEOUT = 5.0  # seconds


class ScreenCaptureAudioSource:
    """Captures system audio via ScreenCaptureKit CLI, injects into audio pipeline."""

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        on_audio: Callable[[np.ndarray], Awaitable[Any]] | None = None,
        on_error: Callable[[str], Awaitable[Any]] | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.on_audio = on_audio
        self.on_error = on_error
        self._process: asyncio.subprocess.Process | None = None
        self._running = False
        self._read_task: asyncio.Task[None] | None = None
        self._last_data_time: float = 0

    @staticmethod
    def is_available() -> bool:
        """Check if the capture binary is installed and accessible."""
        return shutil.which(CAPTURE_BINARY) is not None

    async def start(self) -> bool:
        """Start the capture subprocess and begin reading audio."""
        if self._running:
            return True

        if not self.is_available():
            if self.on_error:
                await self.on_error(
                    f"ScreenCapture binary not found. Install {CAPTURE_BINARY}."
                )
            return False

        try:
            self._process = await asyncio.create_subprocess_exec(
                CAPTURE_BINARY,
                "--sample-rate", str(self.sample_rate),
                "--channels", str(self.channels),
                "--format", "f32le",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._running = True
            self._last_data_time = asyncio.get_event_loop().time()
            self._read_task = asyncio.create_task(self._read_loop())
            logger.info("screencapture_started", pid=self._process.pid)
            return True

        except Exception as e:
            logger.error("screencapture_start_failed", error=str(e))
            if self.on_error:
                await self.on_error(f"Failed to start screen capture: {e}")
            return False

    async def _read_loop(self) -> None:
        """Continuously read audio from subprocess stdout with watchdog."""
        try:
            while self._running and self._process and self._process.stdout:
                try:
                    chunk = await asyncio.wait_for(
                        self._process.stdout.read(CHUNK_SIZE),
                        timeout=WATCHDOG_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    # Watchdog triggered - check if process is alive
                    if self._process.returncode is not None:
                        logger.error("screencapture_process_died", returncode=self._process.returncode)
                        if self.on_error:
                            await self.on_error("Screen capture process died unexpectedly")
                        break
                    logger.warning("screencapture_watchdog_timeout")
                    continue

                if not chunk:
                    # EOF - process closed stdout
                    break

                self._last_data_time = asyncio.get_event_loop().time()

                # Convert bytes to numpy float32 array
                audio = np.frombuffer(chunk, dtype=np.float32)

                if self.on_audio:
                    await self.on_audio(audio)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("screencapture_read_error", error=str(e))
            if self.on_error:
                await self.on_error(f"Screen capture read error: {e}")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the capture subprocess with graceful shutdown."""
        if not self._running:
            return

        self._running = False

        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._process:
            # Drain any remaining data before terminating
            if self._process.stdout:
                try:
                    remaining = await asyncio.wait_for(
                        self._process.stdout.read(),
                        timeout=0.5,
                    )
                    if remaining and self.on_audio:
                        audio = np.frombuffer(remaining, dtype=np.float32)
                        await self.on_audio(audio)
                except (asyncio.TimeoutError, Exception):
                    pass

            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            self._process = None

        logger.info("screencapture_stopped")
```

- [ ] **Step 3: Create test**

```python
# modules/orchestration-service/tests/audio/test_screencapture_source.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio


@pytest.mark.asyncio
async def test_is_available_when_binary_exists():
    """is_available returns True when binary is in PATH."""
    with patch("shutil.which", return_value="/usr/local/bin/livetranslate-capture"):
        from audio.screencapture_source import ScreenCaptureAudioSource
        assert ScreenCaptureAudioSource.is_available() is True


@pytest.mark.asyncio
async def test_is_available_when_binary_missing():
    """is_available returns False when binary not in PATH."""
    with patch("shutil.which", return_value=None):
        from audio.screencapture_source import ScreenCaptureAudioSource
        assert ScreenCaptureAudioSource.is_available() is False


@pytest.mark.asyncio
async def test_start_fails_when_binary_missing():
    """start() returns False and calls on_error when binary missing."""
    with patch("shutil.which", return_value=None):
        from audio.screencapture_source import ScreenCaptureAudioSource
        
        on_error = AsyncMock()
        source = ScreenCaptureAudioSource(on_error=on_error)
        
        result = await source.start()
        
        assert result is False
        on_error.assert_called_once()
        assert "not found" in on_error.call_args[0][0]


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    """stop() can be called multiple times safely."""
    with patch("shutil.which", return_value=None):
        from audio.screencapture_source import ScreenCaptureAudioSource
        
        source = ScreenCaptureAudioSource()
        
        # Should not raise
        await source.stop()
        await source.stop()
```

- [ ] **Step 4: Run tests**

Run: `cd modules/orchestration-service && uv run pytest tests/audio/test_screencapture_source.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/audio/__init__.py
git add modules/orchestration-service/src/audio/screencapture_source.py
git add modules/orchestration-service/tests/audio/test_screencapture_source.py
git commit -m "feat(orchestration): add ScreenCaptureAudioSource

Python wrapper for livetranslate-capture subprocess.
Includes watchdog timer and graceful shutdown with buffer drain."
```

---

### Task 14: Add ScreenCapture Availability Endpoint

**Files:**
- Create: `modules/orchestration-service/src/routers/system.py`
- Modify: `modules/orchestration-service/src/main_fastapi.py`

- [ ] **Step 1: Create system router**

```python
# modules/orchestration-service/src/routers/system.py
"""System capabilities router."""

from fastapi import APIRouter
from audio.screencapture_source import ScreenCaptureAudioSource

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/screencapture-available")
async def get_screencapture_available() -> dict:
    """Check if ScreenCaptureKit capture is available."""
    return {
        "available": ScreenCaptureAudioSource.is_available(),
    }
```

- [ ] **Step 2: Register router in main_fastapi.py**

```python
# In modules/orchestration-service/src/main_fastapi.py
# Add import:
from routers.system import router as system_router

# Add to router registration section:
app.include_router(system_router, prefix="/api")
```

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/routers/system.py
git add modules/orchestration-service/src/main_fastapi.py
git commit -m "feat(orchestration): add system capabilities endpoint

GET /api/system/screencapture-available for frontend feature detection"
```

---

### Task 15: Integrate ScreenCapture into WebSocket Handler

**Files:**
- Modify: `modules/orchestration-service/src/routers/audio/websocket_audio.py`

- [ ] **Step 1: Add screencapture source handling**

In `websocket_audio.py`, update the `start_session` handling to support screencapture source:

```python
# Add import at top
from audio.screencapture_source import ScreenCaptureAudioSource

# In the start_session handling section, after checking for source type:

            # Handle screencapture source
            source_type = getattr(msg, 'source', 'mic')
            screencapture: ScreenCaptureAudioSource | None = None
            
            if source_type == 'screencapture':
                async def handle_screencapture_audio(audio: np.ndarray) -> None:
                    """Inject screencapture audio into pipeline."""
                    nonlocal _stable_text_buffer
                    
                    if fixture_recorder:
                        fixture_recorder.write_audio(audio)
                    
                    if pipeline:
                        downsampled = await pipeline.process_audio(audio)
                        if transcription_ws:
                            await transcription_ws.send_bytes(downsampled.tobytes())
                
                async def handle_screencapture_error(error: str) -> None:
                    """Handle screencapture errors."""
                    await safe_send(ErrorMessage(message=error, recoverable=False))
                
                screencapture = ScreenCaptureAudioSource(
                    sample_rate=sample_rate,
                    channels=channels,
                    on_audio=handle_screencapture_audio,
                    on_error=handle_screencapture_error,
                )
                
                if not await screencapture.start():
                    continue  # Error already sent to client
                
                _active_sessions[session_id]["screencapture"] = screencapture
                logger.info("screencapture_source_started", session_id=session_id)

# In cleanup section, add:
        if "screencapture" in _active_sessions.get(session_id, {}):
            await _active_sessions[session_id]["screencapture"].stop()
```

- [ ] **Step 2: Commit**

```bash
git add modules/orchestration-service/src/routers/audio/websocket_audio.py
git commit -m "feat(websocket): integrate ScreenCaptureAudioSource

Handles source='screencapture' in start_session.
Audio flows through same pipeline as browser audio."
```

---

## Summary

This plan implements the multi-source caption system in 15 tasks across 2 phases:

**Phase 1 (Tasks 1-10):** Unified store, source adapters, Fireflies integration, toolbar UI
**Phase 2 (Tasks 11-15):** Swift CLI, backend subprocess wrapper, WebSocket integration

Each task follows TDD with explicit test→implement→verify→commit steps. The review findings (backpressure, watchdog, chunk sizing, ID prefixing) are incorporated into the implementations.

**Estimated time:** 8 days total (3 days Phase 1, 5 days Phase 2)
