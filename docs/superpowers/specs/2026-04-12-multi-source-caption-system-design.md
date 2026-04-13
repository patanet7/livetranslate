# Multi-Source Caption System Design

**Date:** 2026-04-12
**Status:** Draft
**Author:** Claude + Thomas

## Overview

Unify the caption display system to support multiple audio/transcription sources through a single multi-view interface. Currently the loopback page only supports browser audio capture, while Fireflies sessions are displayed in a separate fixed-layout page. This design adds:

1. **Fireflies integration** — Connect to active Fireflies sessions for transcription
2. **ScreenCaptureKit capture** — Native macOS system audio capture without virtual devices
3. **Unified store** — Single caption store that handles all source formats
4. **Auto-recording** — Record audio immediately on session start, not just promoted meetings

## Goals

- Single page (`/loopback`) supports all input sources with all display modes
- Audio from ScreenCaptureKit flows through existing pipeline (recorded + transcribed)
- Fireflies meeting auto-discovery with single-meeting auto-connect
- Quick capture recording without requiring "promote to meeting"
- Clean source adapter abstraction for future sources

## Non-Goals

- Windows/Linux ScreenCaptureKit equivalent (macOS only for now)
- Fireflies audio recording (Fireflies handles their own recording)
- Real-time diarization for ScreenCaptureKit (future enhancement)

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AUDIO SOURCES                                   │
├───────────────────┬─────────────────────┬───────────────────────────────────┤
│   Browser Mic     │   ScreenCaptureKit  │      Fireflies Desktop            │
│   (existing)      │   (new - Phase 2)   │      (external app)               │
│   getUserMedia()  │   Swift CLI → PCM   │      captures system audio        │
└─────────┬─────────┴──────────┬──────────┴─────────────────┬─────────────────┘
          │                    │                            │
          │     Audio Path     │                            │ Transcript Path
          ▼                    ▼                            │ (no audio)
┌─────────────────────────────────────────────┐             │
│         ORCHESTRATION SERVICE               │             │
│  /ws/loopback                               │             │
│    │                                        │             │
│    ▼                                        │             │
│  MeetingPipeline                            │             │
│    ├── FlacChunkRecorder (48kHz .flac)     │             │
│    └── Downsampler (→16kHz)                │             │
│           │                                 │             │
│           ▼                                 │             │
│    WebSocketTranscriptionClient             │             │
│           │                                 │             │
│           ▼                                 │             │
│    TRANSCRIPTION SERVICE                    │             │
│    (vLLM-MLX Whisper)                       │             │
│           │                                 │             │
│           ▼                                 │             │
│    SegmentStore ◄───────────────────────────┼─────────────┘
│           │                                 │   Fireflies sends
│           ▼                                 │   transcripts directly
│    TranslationService (LLM)                 │
│           │                                 │
│           ▼                                 │
│    CaptionBuffer → WebSocket broadcast      │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DASHBOARD                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Source: [Local Mic ▼] | [System Audio ▼] | [Fireflies ▼]                   │
│                                                                              │
│  UnifiedCaptionStore ◄── LoopbackAdapter | FirefliesAdapter                 │
│                                                                              │
│  Display: SplitView | SubtitleView | InterpreterView | TranscriptView       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **ScreenCaptureKit audio goes through orchestration** — Same path as browser audio for recording and pipeline consistency
2. **Fireflies skips audio path** — Fireflies already transcribed; we only translate
3. **Unified frontend store** — Accepts both `SegmentMessage` and `CaptionEvent` formats
4. **Auto-record by default** — All sessions record immediately; "promote to meeting" adds metadata/persistence

---

## Component Specifications

### 1. ScreenCaptureKit Swift CLI

**Binary:** `livetranslate-capture`

**Purpose:** Capture macOS system audio using ScreenCaptureKit (macOS 13+), output raw PCM to stdout.

**Interface:**
```bash
# Basic usage - outputs f32le PCM to stdout
livetranslate-capture --sample-rate 48000 --channels 1 --format f32le

# With explicit audio device (optional)
livetranslate-capture --device "BlackHole 2ch" --sample-rate 48000

# List available audio sources
livetranslate-capture --list-sources
```

**Output Format:**
- Raw PCM, little-endian float32 (`f32le`)
- 48000 Hz sample rate (matches browser audio)
- Mono (1 channel) or Stereo (2 channels)
- Continuous stream to stdout until SIGTERM/SIGINT

**Implementation Notes:**
```swift
// Pseudocode - actual implementation in tools/screencapture/
import ScreenCaptureKit
import AVFoundation

class SystemAudioCapture {
    let stream: SCStream
    let audioSettings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: 48000,
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
    ]
    
    func start() async throws {
        let content = try await SCShareableContent.current
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = 48000
        config.channelCount = 1
        
        stream = SCStream(filter: SCContentFilter(...), configuration: config, delegate: self)
        try await stream.startCapture()
    }
    
    func stream(_ stream: SCStream, didOutputSampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        // Extract audio buffer, write to stdout as f32le
        let audioBuffer = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(...)
        FileHandle.standardOutput.write(audioBuffer.data)
    }
}
```

**Requirements:**
- macOS 13.0+ (Ventura)
- Screen Recording permission (system prompt on first run)
- Code signed for distribution
- Notarized for Gatekeeper

**Error Handling:**
- Exit code 1 + stderr message if permission denied
- Exit code 2 if no audio sources available
- Graceful shutdown on SIGTERM (flush buffers)

---

### 2. ScreenCaptureAudioSource (Backend)

**File:** `modules/orchestration-service/src/audio/screencapture_source.py`

**Purpose:** Spawn the Swift CLI as a subprocess, pipe audio into the existing WebSocket audio pipeline.

```python
"""ScreenCaptureKit audio source - spawns native capture and pipes to pipeline."""

import asyncio
import numpy as np
from pathlib import Path
from livetranslate_common.logging import get_logger

logger = get_logger()

CAPTURE_BINARY = "livetranslate-capture"
CHUNK_SIZE = 4096  # bytes per read (1024 float32 samples)


class ScreenCaptureAudioSource:
    """Captures system audio via ScreenCaptureKit CLI, injects into audio pipeline."""

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        on_audio: callable = None,  # async callback(np.ndarray)
        on_error: callable = None,  # async callback(str)
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.on_audio = on_audio
        self.on_error = on_error
        self._process: asyncio.subprocess.Process | None = None
        self._running = False
        self._read_task: asyncio.Task | None = None

    @staticmethod
    def is_available() -> bool:
        """Check if the capture binary is installed and accessible."""
        import shutil
        return shutil.which(CAPTURE_BINARY) is not None

    async def start(self) -> bool:
        """Start the capture subprocess and begin reading audio."""
        if self._running:
            return True

        if not self.is_available():
            if self.on_error:
                await self.on_error(
                    f"ScreenCapture binary not found. Install livetranslate-capture."
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
            self._read_task = asyncio.create_task(self._read_loop())
            logger.info("screencapture_started", pid=self._process.pid)
            return True

        except Exception as e:
            logger.error("screencapture_start_failed", error=str(e))
            if self.on_error:
                await self.on_error(f"Failed to start screen capture: {e}")
            return False

    async def _read_loop(self):
        """Continuously read audio from subprocess stdout."""
        try:
            while self._running and self._process:
                chunk = await self._process.stdout.read(CHUNK_SIZE)
                if not chunk:
                    break
                
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

    async def stop(self):
        """Stop the capture subprocess."""
        if not self._running:
            return

        self._running = False

        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None

        logger.info("screencapture_stopped")
```

**Integration with WebSocket Handler:**

```python
# In websocket_audio.py - add source parameter handling

async def handle_start_session(msg: StartSessionMessage, ...):
    source = getattr(msg, 'source', 'mic')  # 'mic' | 'screencapture'
    
    if source == 'screencapture':
        # Create ScreenCaptureAudioSource instead of waiting for browser audio
        screencapture = ScreenCaptureAudioSource(
            sample_rate=msg.sample_rate,
            channels=msg.channels,
            on_audio=lambda audio: pipeline.process_audio(audio),
            on_error=lambda err: safe_send(ErrorMessage(message=err, recoverable=False)),
        )
        if not await screencapture.start():
            return  # Error already sent
        
        # Store for cleanup
        _active_sessions[session_id]["screencapture"] = screencapture
```

---

### 3. Auto-Recording on Session Start

**File:** `modules/orchestration-service/src/meeting/pipeline.py`

**Change:** Add `auto_record` parameter to start recording immediately.

```python
class MeetingPipeline:
    def __init__(
        self,
        session_manager: MeetingSessionManager,
        recording_base_path: Path,
        source_type: str = "loopback",
        sample_rate: int = 48000,
        channels: int = 2,
        auto_record: bool = True,  # NEW: record immediately
    ):
        self.auto_record = auto_record
        self.recorder: FlacChunkRecorder | None = None
        # ... existing init

    async def start(self) -> None:
        """Start the pipeline. If auto_record=True, begin recording immediately."""
        self.session_id = await self.session_manager.create_session(
            source_type=self.source_type,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        
        if self.auto_record:
            self.recorder = FlacChunkRecorder(
                session_id=str(self.session_id),
                base_path=self.recording_base_path,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )
            self.recorder.start()
            logger.info("auto_record_started", session_id=self.session_id)

    async def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process audio: record at native rate, return downsampled for transcription."""
        # Record at native sample rate (48kHz)
        if self.recorder:
            self.recorder.write(audio)
        
        # Downsample to 16kHz for transcription
        return self.downsampler.process(audio)
```

**Recording Location:**
- Quick captures: `/tmp/livetranslate/recordings/{session_id}/`
- Promoted meetings: `{RECORDING_BASE_PATH}/{session_id}/` (persistent)

---

### 4. Unified Caption Store (Frontend)

**File:** `modules/dashboard-service/src/lib/stores/caption.svelte.ts`

**Purpose:** Single store that handles both loopback (`SegmentMessage`) and Fireflies (`CaptionEvent`) formats.

```typescript
/**
 * Unified caption store supporting multiple sources.
 * 
 * Handles both event formats:
 * - Loopback: SegmentMessage + TranslationMessage (separate events)
 * - Fireflies: CaptionEvent (combined text + translation)
 */

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
  // --- State ---
  let captions = $state<UnifiedCaption[]>([]);
  let interimText = $state('');
  let interimConfidence = $state(0);
  
  // Source & connection
  let captionSource = $state<CaptionSource>('local');
  let connectionState = $state<ConnectionState>('disconnected');
  let firefliesSessionId = $state<string | null>(null);
  
  // Display
  let displayMode = $state<DisplayMode>('split');
  
  // Language
  let sourceLanguage = $state<string | null>(null);
  let targetLanguage = $state('zh');
  let detectedLanguage = $state<string | null>(null);
  let interpreterLangA = $state('zh');
  let interpreterLangB = $state('en');
  
  // Status
  let transcriptionStatus = $state<'up' | 'down'>('down');
  let translationStatus = $state<'up' | 'down'>('down');
  let isCapturing = $state(false);
  let isRecording = $state(false);
  let recordingChunks = $state(0);
  
  // Stats
  let chunksSent = $state(0);
  let segmentsReceived = $state(0);
  let translationsReceived = $state(0);
  let lastError = $state<string | null>(null);
  
  // Internal
  let nextId = 0;
  const speakerColorMap = new Map<string, string>();

  // --- Helpers ---
  
  function getSpeakerColor(speaker: string | null): string {
    if (!speaker) return SPEAKER_COLORS[0];
    if (!speakerColorMap.has(speaker)) {
      speakerColorMap.set(speaker, SPEAKER_COLORS[speakerColorMap.size % SPEAKER_COLORS.length]);
    }
    return speakerColorMap.get(speaker)!;
  }

  // --- Ingest Methods (handle both formats) ---

  /**
   * Ingest a loopback SegmentMessage.
   */
  function ingestSegment(msg: SegmentMessage): void {
    if (typeof msg.text !== 'string') return;
    segmentsReceived++;

    const existingIdx = captions.findIndex(c => c.id === String(msg.segment_id));
    
    const caption: UnifiedCaption = {
      id: String(msg.segment_id),
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
      // Don't overwrite final with draft
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

  /**
   * Ingest a loopback TranslationMessage.
   */
  function ingestTranslation(msg: TranslationMessage): void {
    if (typeof msg.text !== 'string') return;
    translationsReceived++;

    const caption = captions.find(c => c.id === String(msg.transcript_id));
    if (!caption) return;

    const isDraft = msg.is_draft ?? false;
    
    // State guards
    if (caption.translationState === 'complete') return;
    if (isDraft && caption.translationState !== 'pending' && caption.translationState !== 'draft') return;

    caption.translation = msg.text;
    caption.translationState = isDraft ? 'draft' : 'complete';
  }

  /**
   * Ingest a loopback TranslationChunkMessage (streaming).
   */
  function ingestTranslationChunk(msg: TranslationChunkMessage): void {
    const caption = captions.find(c => c.id === String(msg.transcript_id));
    if (!caption) return;
    if (caption.translationState === 'complete') return;

    if (caption.translationState === 'draft') {
      caption.translation = '';
    }
    caption.translationState = 'streaming';
    caption.translation = (caption.translation ?? '') + msg.delta;
  }

  /**
   * Ingest a Fireflies CaptionEvent.
   */
  function ingestCaptionEvent(event: CaptionEvent): void {
    if (event.event === 'caption_added' || event.event === 'caption_updated') {
      const cap = event.caption;
      const existingIdx = captions.findIndex(c => c.id === cap.id);
      
      const caption: UnifiedCaption = {
        id: cap.id,
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
    } else if (event.event === 'caption_expired') {
      // Keep for history (don't remove)
    } else if (event.event === 'session_cleared') {
      captions = [];
    }
  }

  /**
   * Ingest interim text update.
   */
  function ingestInterim(text: string, confidence: number): void {
    interimText = text;
    interimConfidence = confidence;
  }

  // --- Lifecycle ---

  function clear(): void {
    captions = [];
    interimText = '';
    interimConfidence = 0;
    chunksSent = 0;
    segmentsReceived = 0;
    translationsReceived = 0;
    lastError = null;
    nextId = 0;
    speakerColorMap.clear();
  }

  // --- Persistence ---
  
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

  // --- Public API ---
  
  return {
    // State (reactive getters)
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
    
    // Methods
    getSpeakerColor,
    ingestSegment,
    ingestTranslation,
    ingestTranslationChunk,
    ingestCaptionEvent,
    ingestInterim,
    clear,
    restoreConfig,
  };
}

export const captionStore = createCaptionStore();
```

---

### 5. Source Adapters (Frontend)

**File:** `modules/dashboard-service/src/lib/audio/source-adapter.ts`

**Purpose:** Abstract WebSocket connection for different sources.

```typescript
/**
 * Source adapters abstract the WebSocket connection for different caption sources.
 * Each adapter connects to its respective backend endpoint and normalizes events
 * to the unified caption store format.
 */

import { captionStore, type ConnectionState } from '$lib/stores/caption.svelte';
import { WS_BASE } from '$lib/config';
import type { ServerMessage } from '$lib/types/ws-messages';
import type { CaptionEvent } from '$lib/types/caption';

export interface SourceAdapter {
  connect(config: SourceConfig): Promise<void>;
  disconnect(): void;
  sendConfig(config: Record<string, unknown>): void;
  sendAudio?(data: Float32Array): void;  // Only for local/screencapture
}

export interface SourceConfig {
  sourceLanguage: string | null;
  targetLanguage: string;
  interpreterLanguages?: [string, string];
  deviceId?: string;
  sampleRate?: number;
  channels?: number;
  firefliesSessionId?: string;  // For Fireflies source
}

/**
 * Loopback adapter - connects to /ws/loopback for browser mic or screencapture.
 */
export class LoopbackAdapter implements SourceAdapter {
  private ws: WebSocket | null = null;
  private source: 'mic' | 'screencapture';

  constructor(source: 'mic' | 'screencapture' = 'mic') {
    this.source = source;
  }

  async connect(config: SourceConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `${WS_BASE}/ws/loopback`;
      this.ws = new WebSocket(url);
      
      this.ws.onopen = () => {
        captionStore.connectionState = 'connecting';
        // Send start_session with source type
        this.ws?.send(JSON.stringify({
          type: 'start_session',
          sample_rate: config.sampleRate ?? 48000,
          channels: config.channels ?? 1,
          device_id: config.deviceId,
          source: this.source,  // 'mic' or 'screencapture'
        }));
      };

      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as ServerMessage;
        this.handleMessage(msg, resolve, reject);
      };

      this.ws.onerror = () => {
        captionStore.connectionState = 'error';
        reject(new Error('WebSocket connection failed'));
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
    this.ws?.send(JSON.stringify({ type: 'config', ...config }));
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
    // Fireflies doesn't support runtime config changes
    // Could implement language filter update via set_language event
    if (config.target_language) {
      this.ws?.send(JSON.stringify({
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

---

### 6. Fireflies Meeting Discovery

**Backend Endpoint:** `GET /api/fireflies/active-meetings`

Already exists in `routers/fireflies.py`. Add auto-select logic:

```python
# routers/fireflies.py - enhance active meetings endpoint

@router.get("/active-meetings")
async def get_active_meetings(
    settings: FirefliesSettings = Depends(get_settings),
) -> dict:
    """
    Get active Fireflies meetings for the configured account.
    
    Returns auto_select=True and auto_select_id if exactly one meeting is active.
    """
    if not settings.fireflies_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fireflies API key not configured",
        )

    client = FirefliesGraphQLClient(api_key=settings.fireflies_api_key)
    
    try:
        meetings = await client.get_active_meetings()
    except FirefliesRateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Fireflies rate limit exceeded",
        )
    except FirefliesAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fireflies API error: {e}",
        )

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

**Frontend Meeting Picker:**

```svelte
<!-- In Toolbar.svelte - add meeting selector when Fireflies is selected -->

{#if captionStore.captionSource === 'fireflies'}
  <div class="meeting-selector">
    {#if loadingMeetings}
      <span class="loading">Loading meetings...</span>
    {:else if meetings.length === 0}
      <span class="no-meetings">No active Fireflies meetings</span>
    {:else if meetings.length === 1}
      <span class="auto-selected">
        Auto-connected: {meetings[0].title}
      </span>
    {:else}
      <select bind:value={selectedMeetingId}>
        <option value="">Select a meeting...</option>
        {#each meetings as meeting}
          <option value={meeting.id}>
            {meeting.title} ({formatTimeAgo(meeting.started_at)})
          </option>
        {/each}
      </select>
    {/if}
  </div>
{/if}
```

---

### 7. Toolbar Source Selector

**File:** `modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte`

Add source selection UI:

```svelte
<script lang="ts">
  import { captionStore } from '$lib/stores/caption.svelte';
  
  // Source options with availability check
  let screenCaptureAvailable = $state(false);
  
  onMount(async () => {
    // Check if screencapture is available
    try {
      const res = await fetch('/api/system/screencapture-available');
      screenCaptureAvailable = (await res.json()).available;
    } catch {
      screenCaptureAvailable = false;
    }
  });
</script>

<div class="source-selector">
  <label>
    <input
      type="radio"
      name="source"
      value="local"
      bind:group={captionStore.captionSource}
      disabled={isCapturing}
    />
    Local Mic
  </label>
  
  <label class:disabled={!screenCaptureAvailable}>
    <input
      type="radio"
      name="source"
      value="screencapture"
      bind:group={captionStore.captionSource}
      disabled={isCapturing || !screenCaptureAvailable}
    />
    System Audio
    {#if !screenCaptureAvailable}
      <span class="badge">Install required</span>
    {/if}
  </label>
  
  <label>
    <input
      type="radio"
      name="source"
      value="fireflies"
      bind:group={captionStore.captionSource}
      disabled={isCapturing}
    />
    Fireflies
  </label>
</div>
```

---

## Protocol Changes

### StartSessionMessage Extension

```typescript
// ws-messages.ts - extend start_session

export interface StartSessionMessage {
  type: 'start_session';
  sample_rate: number;
  channels: number;
  encoding?: string;
  device_id?: string;
  source?: 'mic' | 'screencapture';  // NEW: audio source type
}
```

### Backend Handling

```python
# websocket_audio.py - handle source parameter

@dataclass
class StartSessionPayload:
    sample_rate: int
    channels: int
    encoding: str = "float32"
    device_id: str | None = None
    source: Literal["mic", "screencapture"] = "mic"  # NEW
```

---

## Implementation Phases

### Phase 1: Unified Store + Fireflies Integration (3 days)

| Task | Files | Effort |
|------|-------|--------|
| 1a. Create unified caption store | `stores/caption.svelte.ts` | 4h |
| 1b. Create source adapter abstraction | `audio/source-adapter.ts` | 3h |
| 1c. Implement Fireflies adapter | `audio/source-adapter.ts` | 2h |
| 1d. Add meeting discovery endpoint enhancement | `routers/fireflies.py` | 1h |
| 1e. Add meeting picker to Toolbar | `Toolbar.svelte` | 2h |
| 1f. Wire display components to new store | `SplitView.svelte`, etc. | 3h |
| 1g. Add source selector UI | `Toolbar.svelte` | 2h |
| 1h. Add auto-record on session start | `meeting/pipeline.py` | 2h |
| 1i. Tests | `*.test.ts`, `test_*.py` | 4h |

### Phase 2: ScreenCaptureKit Integration (5 days)

| Task | Files | Effort |
|------|-------|--------|
| 2a. Swift CLI project setup | `tools/screencapture/` | 2h |
| 2b. ScreenCaptureKit audio capture | `Sources/main.swift` | 6h |
| 2c. Stdout PCM output | `Sources/main.swift` | 2h |
| 2d. Error handling + permissions | `Sources/main.swift` | 2h |
| 2e. Backend ScreenCaptureAudioSource | `audio/screencapture_source.py` | 4h |
| 2f. WebSocket handler integration | `websocket_audio.py` | 3h |
| 2g. Availability check endpoint | `routers/system.py` | 1h |
| 2h. Frontend screencapture adapter | Uses LoopbackAdapter | 1h |
| 2i. macOS signing + notarization | Build scripts | 4h |
| 2j. Installer/DMG packaging | Build scripts | 4h |
| 2k. Tests | Integration tests | 4h |

---

## Testing Strategy

### Unit Tests

- Caption store ingestion (both formats)
- Source adapter connection/disconnection
- ScreenCaptureAudioSource subprocess management
- Auto-record pipeline behavior

### Integration Tests

- Loopback → store → display flow
- Fireflies → store → display flow
- ScreenCapture → pipeline → transcription → store flow
- Meeting discovery + auto-select

### E2E Tests

- Source switching during idle
- Full capture → transcription → translation → display flow
- Fireflies session connection with real API (optional, gated)

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| ScreenCapture binary not found | Show "Install required" badge, disable option |
| ScreenCapture permission denied | Show error toast with "Open System Preferences" link |
| Fireflies API key missing | Show "Configure Fireflies" link in source selector |
| Fireflies no active meetings | Show "No active meetings. Start the Fireflies desktop app." |
| WebSocket disconnect | Auto-reconnect with exponential backoff (existing behavior) |
| Transcription service down | Show status indicator, buffer audio if recording |

---

## Security Considerations

1. **ScreenCaptureKit permissions** — Requires Screen Recording permission; user must grant explicitly
2. **Fireflies API key** — Stored in server-side settings, not exposed to frontend
3. **Audio recording** — Saved to local filesystem only; user controls when to promote/persist
4. **Code signing** — Swift binary must be signed and notarized for distribution

---

## Future Enhancements

1. **Windows/Linux system audio capture** — PulseAudio/WASAPI equivalents
2. **Real-time diarization for ScreenCapture** — Speaker identification from system audio
3. **Multiple simultaneous sources** — Capture both mic and system audio
4. **Cloud recording upload** — Optional upload of recordings to cloud storage

---

## Open Questions

1. Should ScreenCaptureKit capture include all system audio or allow app-specific filtering?
2. Should we bundle the Swift CLI in an Electron/Tauri wrapper for easier installation?
3. Do we need a "test audio" feature before starting capture?
