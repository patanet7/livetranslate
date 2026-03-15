# Plan 2: SvelteKit Loopback Page

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the loopback page in the SvelteKit dashboard — mic + system audio capture → binary WebSocket to orchestration → live captions in split/subtitle/transcript modes → meeting promotion UI.

**Architecture:** The loopback page captures audio via `getUserMedia()` (mic) and optionally a virtual loopback device (system audio). An AudioWorklet processes audio and sends `Float32Array` binary frames over WebSocket to orchestration. Text frames carry control messages (start_session, promote_to_meeting, etc.) and incoming transcription/translation results. The page renders three switchable display modes. All types come from Plan 0's shared contracts.

**Tech Stack:** SvelteKit (Svelte 5 runes), TypeScript, Web Audio API (AudioWorklet), WebSocket (binary frames)

**Spec:** `docs/superpowers/specs/2026-03-14-loopback-transcription-translation-design.md` — Plan 2 section

**Depends on:** Plan 0 (TypeScript type definitions in `dashboard-service/src/lib/types/`)

> **BLOCKING DEPENDENCY:** Plan 0 Chunk 3, Task 7 (TypeScript types for WebSocket messages, `$lib/types/ws-messages`) **must be completed before any work in this plan begins.** All imports from `$lib/types/ws-messages` (used in Task 2, Task 3, and Task 8) depend on those type definitions existing. Do not start Plan 2 until Plan 0 Task 7 is merged.

---

## Chunk 1: Audio Capture & WebSocket

### Task 1: AudioWorklet processor for mic capture

**Files:**
- Create: `modules/dashboard-service/static/audio-worklet-processor.js`
- Create: `modules/dashboard-service/src/lib/audio/capture.ts`

- [ ] **Step 1: Write the AudioWorklet processor**

This runs in the audio rendering thread. It collects Float32Array chunks and posts them to the main thread.

```javascript
// modules/dashboard-service/static/audio-worklet-processor.js

/**
 * AudioWorklet processor that forwards raw Float32Array chunks to the main thread.
 * Runs at native sample rate (typically 48kHz). Downsampling happens server-side.
 */
class AudioChunkProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._bufferSize = 4096; // ~85ms at 48kHz — good balance of latency vs overhead
    // Pre-allocated ring buffer to avoid GC pressure on the audio rendering thread.
    // A growable JS array would create garbage on every flush; a fixed Float32Array
    // ring buffer keeps allocations off the hot path entirely.
    this._ringBuffer = new Float32Array(this._bufferSize);
    this._writeIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Take first channel (mono or left channel of stereo)
    const channelData = input[0];
    if (!channelData || channelData.length === 0) return true;

    // Accumulate samples into the pre-allocated ring buffer
    for (let i = 0; i < channelData.length; i++) {
      this._ringBuffer[this._writeIndex++] = channelData[i];

      // When buffer is full, send to main thread and reset write index
      if (this._writeIndex >= this._bufferSize) {
        // Copy into a new buffer for transfer (original stays allocated)
        const chunk = new Float32Array(this._ringBuffer);
        this.port.postMessage({ type: 'audio_chunk', data: chunk.buffer }, [chunk.buffer]);
        this._writeIndex = 0;
      }
    }

    return true; // Keep processor alive
  }
}

registerProcessor('audio-chunk-processor', AudioChunkProcessor);
```

- [ ] **Step 2: Write the audio capture module**

```typescript
// modules/dashboard-service/src/lib/audio/capture.ts

/**
 * Audio capture manager — handles mic and system audio via getUserMedia.
 *
 * Key design decisions:
 * - AudioWorklet (not ScriptProcessorNode) for glitch-free capture
 * - Native sample rate capture — downsampling is server-side
 * - No echoCancellation/noiseSuppression — these attenuate loopback audio
 */

export type AudioSourceType = 'mic' | 'system' | 'both';

export interface CaptureOptions {
  deviceId?: string;
  systemDeviceId?: string;  // Required when sourceType is 'both' — the loopback device ID
  sourceType: AudioSourceType;
  onChunk: (data: Float32Array) => void;
  onError: (error: Error) => void;
}

export class AudioCapture {
  private context: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private stream: MediaStream | null = null;
  private _systemStream: MediaStream | null = null;  // For 'both' mode — system audio stream
  private _isCapturing = false;

  get isCapturing(): boolean {
    return this._isCapturing;
  }

  get sampleRate(): number {
    return this.context?.sampleRate ?? 48000;
  }

  async start(options: CaptureOptions): Promise<void> {
    if (this._isCapturing) return;

    try {
      // Get audio stream(s) based on source type.
      // 'mic': getUserMedia for microphone input
      // 'system': getUserMedia targeting virtual loopback device (BlackHole/PulseAudio monitor)
      // 'both': capture mic and system audio as separate streams, merge into one source
      const audioConstraints: MediaTrackConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      };

      if (options.sourceType === 'system') {
        // System audio requires a virtual loopback device — deviceId must be provided
        if (!options.deviceId) {
          throw new Error('System audio capture requires a loopback device ID (e.g., BlackHole)');
        }
        audioConstraints.deviceId = { exact: options.deviceId };
      } else if (options.sourceType === 'mic') {
        if (options.deviceId) {
          audioConstraints.deviceId = { exact: options.deviceId };
        }
      }

      if (options.sourceType === 'both') {
        // Capture both mic and system audio as separate streams, merge via ChannelMerger
        const micConstraints: MediaStreamConstraints = {
          audio: {
            ...audioConstraints,
            ...(options.deviceId ? { deviceId: { exact: options.deviceId } } : {}),
          },
        };
        // System audio requires a separate loopback deviceId passed via options.systemDeviceId
        const systemConstraints: MediaStreamConstraints = {
          audio: {
            ...audioConstraints,
            ...(options.systemDeviceId
              ? { deviceId: { exact: options.systemDeviceId } }
              : {}),
          },
        };

        const [micStream, systemStream] = await Promise.all([
          navigator.mediaDevices.getUserMedia(micConstraints),
          navigator.mediaDevices.getUserMedia(systemConstraints),
        ]);

        // Store both streams for cleanup
        this.stream = micStream;
        this._systemStream = systemStream;
        this.context = new AudioContext();

        // Merge both streams into a single mono source
        const micSource = this.context.createMediaStreamSource(micStream);
        const systemSource = this.context.createMediaStreamSource(systemStream);
        const merger = this.context.createChannelMerger(2);
        micSource.connect(merger, 0, 0);
        systemSource.connect(merger, 0, 1);

        await this.context.audioWorklet.addModule('/audio-worklet-processor.js');
        this.workletNode = new AudioWorkletNode(this.context, 'audio-chunk-processor');
        this.workletNode.port.onmessage = (event) => {
          if (event.data.type === 'audio_chunk') {
            options.onChunk(new Float32Array(event.data.data));
          }
        };
        merger.connect(this.workletNode);

        this._isCapturing = true;
        return;
      }

      // Single-source path (mic or system)
      const constraints: MediaStreamConstraints = { audio: audioConstraints };
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.context = new AudioContext();

      // Load AudioWorklet
      await this.context.audioWorklet.addModule('/audio-worklet-processor.js');

      // Create nodes
      const source = this.context.createMediaStreamSource(this.stream);
      this.workletNode = new AudioWorkletNode(this.context, 'audio-chunk-processor');

      // Handle chunks from worklet
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'audio_chunk') {
          const chunk = new Float32Array(event.data.data);
          options.onChunk(chunk);
        }
      };

      // Connect: source → worklet
      source.connect(this.workletNode);
      // Don't connect worklet to destination — we don't want to play back

      this._isCapturing = true;
    } catch (err) {
      options.onError(err instanceof Error ? err : new Error(String(err)));
    }
  }

  async stop(): Promise<void> {
    if (!this._isCapturing) return;

    this.workletNode?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this._systemStream?.getTracks().forEach((t) => t.stop());
    await this.context?.close();

    this.workletNode = null;
    this.stream = null;
    this._systemStream = null;
    this.context = null;
    this._isCapturing = false;
  }

  static async getDevices(): Promise<MediaDeviceInfo[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'audioinput');
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add modules/dashboard-service/static/audio-worklet-processor.js modules/dashboard-service/src/lib/audio/capture.ts
git commit -m "feat(dashboard): add AudioWorklet capture with native sample rate"
```

---

### Task 2: WebSocket connection manager for loopback

**Files:**
- Create: `modules/dashboard-service/src/lib/audio/websocket.ts`

- [ ] **Step 1: Write the WebSocket manager**

```typescript
// modules/dashboard-service/src/lib/audio/websocket.ts

/**
 * WebSocket connection manager for the loopback audio pipeline.
 *
 * Binary frames: raw Float32Array audio → orchestration
 * Text frames: JSON control messages (start_session, segment, translation, etc.)
 *
 * Reconnects automatically with exponential backoff.
 */

import type {
  ClientMessage,
  ServerMessage,
  StartSessionMessage,
} from '$lib/types/ws-messages';
import { parseServerMessage, PROTOCOL_VERSION } from '$lib/types/ws-messages';

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface LoopbackWSOptions {
  url: string;
  onMessage: (msg: ServerMessage) => void;
  onStateChange: (state: ConnectionState) => void;
  onError?: (error: Event) => void;
}

export class LoopbackWebSocket {
  private ws: WebSocket | null = null;
  private options: LoopbackWSOptions;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _state: ConnectionState = 'disconnected';
  private _sessionId: string | null = null;

  get state(): ConnectionState {
    return this._state;
  }

  get sessionId(): string | null {
    return this._sessionId;
  }

  constructor(options: LoopbackWSOptions) {
    this.options = options;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.setState('connecting');

    this.ws = new WebSocket(this.options.url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      // Don't set 'connected' yet — wait for ConnectedMessage with protocol_version
    };

    this.ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        const msg = parseServerMessage(event.data);
        if (!msg) return;

        if (msg.type === 'connected') {
          if (msg.protocol_version !== PROTOCOL_VERSION) {
            console.warn(
              `Protocol version mismatch: expected ${PROTOCOL_VERSION}, got ${msg.protocol_version}`
            );
          }
          this._sessionId = msg.session_id;
          this.setState('connected');
        }

        this.options.onMessage(msg);
      }
    };

    this.ws.onerror = (event) => {
      this.options.onError?.(event);
    };

    this.ws.onclose = () => {
      this._sessionId = null;
      this.setState('disconnected');
      this.scheduleReconnect();
    };
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectAttempts = this.maxReconnectAttempts; // prevent auto-reconnect
    this.ws?.close();
    this.ws = null;
    this._sessionId = null;
    this.setState('disconnected');
  }

  /** Send a binary audio chunk (Float32Array) */
  sendAudio(data: Float32Array): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(data.buffer);
  }

  /** Send a JSON control message */
  sendMessage(msg: ClientMessage): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify(msg));
  }

  /** Send start_session with audio parameters */
  startSession(sampleRate: number, channels: number, deviceId?: string): void {
    const msg: StartSessionMessage = {
      type: 'start_session',
      sample_rate: sampleRate,
      channels,
      device_id: deviceId,
    };
    this.sendMessage(msg);
  }

  private setState(state: ConnectionState): void {
    this._state = state;
    this.options.onStateChange(state);
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;

    const delay = Math.min(1000 * 2 ** this.reconnectAttempts, 30000);
    this.reconnectAttempts++;

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/audio/websocket.ts
git commit -m "feat(dashboard): add LoopbackWebSocket with binary frames and auto-reconnect"
```

---

### Task 3: Loopback Svelte store (reactive state)

**Files:**
- Create: `modules/dashboard-service/src/lib/stores/loopback.svelte.ts`

- [ ] **Step 1: Write the loopback store**

This is the central state for the loopback page — manages captions, translations, connection state, meeting state.

```typescript
// modules/dashboard-service/src/lib/stores/loopback.svelte.ts

/**
 * Reactive state for the loopback page using Svelte 5 runes.
 *
 * Manages: captions, translations, connection state, meeting state,
 * display mode, and audio source configuration.
 */

import type { SegmentMessage, InterimMessage, TranslationMessage } from '$lib/types/ws-messages';

export type DisplayMode = 'split' | 'subtitle' | 'transcript';

export interface CaptionEntry {
  id: number;
  text: string;
  language: string;
  confidence: number;
  speakerId: string | null;
  isFinal: boolean;
  translation: string | null;
  timestamp: number;
}

// Speaker color palette
const SPEAKER_COLORS = [
  '#3b82f6', '#a855f7', '#22c55e', '#f97316', '#ec4899',
  '#06b6d4', '#eab308', '#ef4444', '#8b5cf6', '#14b8a6',
];

function createLoopbackStore() {
  let captions = $state<CaptionEntry[]>([]);
  let interimText = $state('');
  let interimConfidence = $state(0);
  let displayMode = $state<DisplayMode>('split');
  let connectionState = $state<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  let isCapturing = $state(false);
  let isMeetingActive = $state(false);
  let meetingSessionId = $state<string | null>(null);
  let meetingStartedAt = $state<string | null>(null);
  let transcriptionStatus = $state<'up' | 'down'>('down');
  let translationStatus = $state<'up' | 'down'>('down');
  let isRecording = $state(false);
  let recordingChunks = $state(0);
  let sourceLanguage = $state<string | null>(null);
  let targetLanguage = $state('en');
  let nextId = 0;
  const speakerColorMap = new Map<string, string>();

  function getSpeakerColor(speakerId: string | null): string {
    if (!speakerId) return SPEAKER_COLORS[0];
    if (!speakerColorMap.has(speakerId)) {
      speakerColorMap.set(speakerId, SPEAKER_COLORS[speakerColorMap.size % SPEAKER_COLORS.length]);
    }
    return speakerColorMap.get(speakerId)!;
  }

  function addSegment(msg: SegmentMessage) {
    const entry: CaptionEntry = {
      id: nextId++,
      text: msg.stable_text || msg.text,
      language: msg.language,
      confidence: msg.confidence,
      speakerId: msg.speaker_id,
      isFinal: msg.is_final,
      translation: null,
      timestamp: Date.now(),
    };
    captions = [...captions, entry];

    // Auto-detect source language on first segment
    if (!sourceLanguage) {
      sourceLanguage = msg.language;
    }
  }

  function updateInterim(msg: InterimMessage) {
    interimText = msg.text;
    interimConfidence = msg.confidence;
  }

  function addTranslation(msg: TranslationMessage) {
    // Match translation to its source caption by transcript_id
    captions = captions.map((c) =>
      c.id === msg.transcript_id ? { ...c, translation: msg.text } : c
    );
  }

  function startMeeting(sessionId: string, startedAt: string) {
    isMeetingActive = true;
    meetingSessionId = sessionId;
    meetingStartedAt = startedAt;
  }

  function endMeeting() {
    isMeetingActive = false;
    meetingSessionId = null;
    meetingStartedAt = null;
    isRecording = false;
    recordingChunks = 0;
  }

  function clear() {
    captions = [];
    interimText = '';
    interimConfidence = 0;
    nextId = 0;
  }

  return {
    get captions() { return captions; },
    get interimText() { return interimText; },
    get interimConfidence() { return interimConfidence; },
    get displayMode() { return displayMode; },
    set displayMode(v: DisplayMode) { displayMode = v; },
    get connectionState() { return connectionState; },
    set connectionState(v: typeof connectionState) { connectionState = v; },
    get isCapturing() { return isCapturing; },
    set isCapturing(v: boolean) { isCapturing = v; },
    get isMeetingActive() { return isMeetingActive; },
    get meetingSessionId() { return meetingSessionId; },
    get meetingStartedAt() { return meetingStartedAt; },
    get transcriptionStatus() { return transcriptionStatus; },
    set transcriptionStatus(v: 'up' | 'down') { transcriptionStatus = v; },
    get translationStatus() { return translationStatus; },
    set translationStatus(v: 'up' | 'down') { translationStatus = v; },
    get isRecording() { return isRecording; },
    set isRecording(v: boolean) { isRecording = v; },
    get recordingChunks() { return recordingChunks; },
    set recordingChunks(v: number) { recordingChunks = v; },
    get sourceLanguage() { return sourceLanguage; },
    set sourceLanguage(v: string | null) { sourceLanguage = v; },
    get targetLanguage() { return targetLanguage; },
    set targetLanguage(v: string) { targetLanguage = v; },
    getSpeakerColor,
    addSegment,
    updateInterim,
    addTranslation,
    startMeeting,
    endMeeting,
    clear,
  };
}

export const loopbackStore = createLoopbackStore();
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/stores/loopback.svelte.ts
git commit -m "feat(dashboard): add loopback reactive store with Svelte 5 runes"
```

---

## Chunk 2: Display Modes

### Task 4: Split view display mode

**Files:**
- Create: `modules/dashboard-service/src/lib/components/loopback/SplitView.svelte`

- [ ] **Step 1: Write SplitView component**

```svelte
<!-- modules/dashboard-service/src/lib/components/loopback/SplitView.svelte -->
<script lang="ts">
  import { loopbackStore } from '$lib/stores/loopback.svelte';

  let captionsEndOriginal: HTMLElement;
  let captionsEndTranslation: HTMLElement;

  $effect(() => {
    // Auto-scroll when new captions arrive
    if (loopbackStore.captions.length > 0) {
      captionsEndOriginal?.scrollIntoView({ behavior: 'smooth' });
      captionsEndTranslation?.scrollIntoView({ behavior: 'smooth' });
    }
  });
</script>

<div class="split-view">
  <!-- Original language panel -->
  <div class="panel panel-original">
    <div class="panel-header">
      Original ({loopbackStore.sourceLanguage ?? 'detecting...'})
    </div>
    <div class="panel-content">
      {#each loopbackStore.captions as caption (caption.id)}
        <div
          class="caption-entry"
          class:interim={!caption.isFinal}
          style="border-left-color: {loopbackStore.getSpeakerColor(caption.speakerId)}"
        >
          {#if caption.speakerId}
            <span class="speaker" style="color: {loopbackStore.getSpeakerColor(caption.speakerId)}">
              {caption.speakerId}:
            </span>
          {/if}
          <span class="text">{caption.text}</span>
        </div>
      {/each}
      {#if loopbackStore.interimText}
        <div class="caption-entry interim">
          <span class="text">{loopbackStore.interimText}</span>
        </div>
      {/if}
      <div bind:this={captionsEndOriginal}></div>
    </div>
  </div>

  <!-- Translation panel -->
  <div class="panel panel-translation">
    <div class="panel-header">
      Translation ({loopbackStore.targetLanguage})
    </div>
    <div class="panel-content">
      {#each loopbackStore.captions.filter(c => c.isFinal) as caption (caption.id)}
        <div
          class="caption-entry"
          style="border-left-color: {loopbackStore.getSpeakerColor(caption.speakerId)}"
        >
          {#if caption.speakerId}
            <span class="speaker" style="color: {loopbackStore.getSpeakerColor(caption.speakerId)}">
              {caption.speakerId}:
            </span>
          {/if}
          <span class="text">
            {caption.translation ?? '...'}
          </span>
        </div>
      {/each}
      <div bind:this={captionsEndTranslation}></div>
    </div>
  </div>
</div>

<style>
  .split-view {
    display: flex;
    gap: 2px;
    height: 100%;
    min-height: 400px;
  }
  .panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-header {
    padding: 8px 16px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid var(--border, #333);
  }
  .panel-original .panel-header { color: #ffd700; }
  .panel-translation .panel-header { color: #90ee90; }
  .panel-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }
  .caption-entry {
    padding: 8px;
    margin-bottom: 8px;
    border-left: 3px solid;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.03);
  }
  .caption-entry.interim {
    opacity: 0.5;
    font-style: italic;
  }
  .speaker {
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/components/loopback/SplitView.svelte
git commit -m "feat(dashboard): add SplitView display mode component"
```

---

### Task 5: Subtitle overlay display mode

**Files:**
- Create: `modules/dashboard-service/src/lib/components/loopback/SubtitleView.svelte`

- [ ] **Step 1: Write SubtitleView component**

```svelte
<!-- modules/dashboard-service/src/lib/components/loopback/SubtitleView.svelte -->
<script lang="ts">
  import { loopbackStore } from '$lib/stores/loopback.svelte';

  interface Props {
    fontSize?: number;   // Base font size in px (default: 16). Translation text is fontSize + 2.
    bgOpacity?: number;  // Background opacity 0-1 (default: 0.75)
  }

  let { fontSize = 16, bgOpacity = 0.75 }: Props = $props();

  // Show last 2 final captions as subtitles
  const recentCaptions = $derived(
    loopbackStore.captions
      .filter(c => c.isFinal)
      .slice(-2)
  );

  /** Open subtitle view in a separate browser window for screen-sharing. */
  function popOut() {
    const popupWidth = 800;
    const popupHeight = 300;
    const left = (screen.width - popupWidth) / 2;
    const top = screen.height - popupHeight - 100;

    // Open a new window pointing to a dedicated subtitle-only route
    window.open(
      '/loopback/subtitle-popout',
      'subtitle-popout',
      `width=${popupWidth},height=${popupHeight},left=${left},top=${top},toolbar=no,menubar=no,location=no`
    );
  }
</script>

<div class="subtitle-view">
  <button class="popout-btn" onclick={popOut} title="Pop out subtitles for screen-sharing">
    Pop Out
  </button>
  <div class="subtitle-area">
    {#each recentCaptions as caption (caption.id)}
      <div class="subtitle-line" style="background: rgba(0, 0, 0, {bgOpacity})">
        <div class="original" style="font-size: {fontSize}px">{caption.text}</div>
        {#if caption.translation}
          <div class="translation" style="font-size: {fontSize + 2}px">{caption.translation}</div>
        {/if}
      </div>
    {/each}
    {#if loopbackStore.interimText}
      <div class="subtitle-line interim" style="background: rgba(0, 0, 0, {bgOpacity})">
        <div class="original" style="font-size: {fontSize}px">{loopbackStore.interimText}</div>
      </div>
    {/if}
  </div>
</div>

<style>
  .subtitle-view {
    height: 100%;
    min-height: 400px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 20px;
    position: relative;
  }
  .popout-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    padding: 4px 10px;
    font-size: 11px;
    background: rgba(255, 255, 255, 0.1);
    color: #ccc;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    cursor: pointer;
  }
  .popout-btn:hover {
    background: rgba(255, 255, 255, 0.2);
  }
  .subtitle-area {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .subtitle-line {
    padding: 12px 24px;
    border-radius: 8px;
    text-align: center;
  }
  .subtitle-line.interim {
    opacity: 0.6;
    font-style: italic;
  }
  .original {
    color: #ffd700;
    margin-bottom: 4px;
  }
  .translation {
    color: #90ee90;
  }
</style>
```

> **Note:** The pop-out window (`/loopback/subtitle-popout`) should be a minimal route that imports `SubtitleView` and renders it full-viewport with no toolbar/nav. It reads from the same `loopbackStore` singleton. If cross-window reactivity is needed (the pop-out is a separate browsing context), use a `BroadcastChannel` to relay caption updates from the main page to the pop-out window.

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/components/loopback/SubtitleView.svelte
git commit -m "feat(dashboard): add SubtitleView display mode"
```

---

### Task 6: Transcript display mode

**Files:**
- Create: `modules/dashboard-service/src/lib/components/loopback/TranscriptView.svelte`

- [ ] **Step 1: Write TranscriptView component**

```svelte
<!-- modules/dashboard-service/src/lib/components/loopback/TranscriptView.svelte -->
<script lang="ts">
  import { loopbackStore } from '$lib/stores/loopback.svelte';

  let endRef: HTMLElement;

  $effect(() => {
    if (loopbackStore.captions.length > 0) {
      endRef?.scrollIntoView({ behavior: 'smooth' });
    }
  });

  function formatTime(ts: number): string {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour12: false });
  }
</script>

<div class="transcript-view">
  {#each loopbackStore.captions.filter(c => c.isFinal) as caption (caption.id)}
    <div
      class="transcript-entry"
      style="border-left-color: {loopbackStore.getSpeakerColor(caption.speakerId)}"
    >
      <div class="entry-header">
        {#if caption.speakerId}
          <span class="speaker" style="color: {loopbackStore.getSpeakerColor(caption.speakerId)}">
            {caption.speakerId}
          </span>
        {/if}
        <span class="timestamp">{formatTime(caption.timestamp)}</span>
      </div>
      <div class="original">{caption.text}</div>
      {#if caption.translation}
        <div class="translation">{caption.translation}</div>
      {/if}
    </div>
  {/each}
  {#if loopbackStore.interimText}
    <div class="transcript-entry interim">
      <div class="original">{loopbackStore.interimText}</div>
    </div>
  {/if}
  <div bind:this={endRef}></div>
</div>

<style>
  .transcript-view {
    padding: 16px;
    overflow-y: auto;
    height: 100%;
    min-height: 400px;
  }
  .transcript-entry {
    padding: 12px;
    margin-bottom: 12px;
    border-left: 3px solid;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.03);
  }
  .transcript-entry.interim {
    opacity: 0.5;
    font-style: italic;
  }
  .entry-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
  }
  .speaker {
    font-weight: 600;
    font-size: 13px;
  }
  .timestamp {
    color: #666;
    font-size: 11px;
  }
  .original {
    color: #ffd700;
    margin-bottom: 4px;
  }
  .translation {
    color: #90ee90;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/components/loopback/TranscriptView.svelte
git commit -m "feat(dashboard): add TranscriptView display mode"
```

---

## Chunk 3: Loopback Page & Toolbar

### Task 7: Loopback toolbar component

**Files:**
- Create: `modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte`

- [ ] **Step 1: Write toolbar with all controls**

```svelte
<!-- modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte -->
<script lang="ts">
  import { loopbackStore, type DisplayMode } from '$lib/stores/loopback.svelte';
  import { Button } from '$lib/components/ui/button';

  interface Props {
    devices: MediaDeviceInfo[];
    selectedDeviceId: string;
    onDeviceChange: (deviceId: string) => void;
    onStartCapture: () => void;
    onStopCapture: () => void;
    onStartMeeting: () => void;
    onEndMeeting: () => void;
  }

  let {
    devices,
    selectedDeviceId,
    onDeviceChange,
    onStartCapture,
    onStopCapture,
    onStartMeeting,
    onEndMeeting,
  }: Props = $props();

  /** Model override — empty string means "auto" (registry decides based on detected language). */
  let modelOverride = $state('');

  /** Whether the End Meeting confirmation dialog is open. */
  let showEndMeetingConfirm = $state(false);

  const modes: { value: DisplayMode; label: string }[] = [
    { value: 'split', label: 'Split' },
    { value: 'subtitle', label: 'Subtitle' },
    { value: 'transcript', label: 'Transcript' },
  ];

  /** Available source languages for manual override. Empty = auto-detect. */
  const sourceLanguages = [
    { value: '', label: 'Auto' },
    { value: 'en', label: 'English' },
    { value: 'zh', label: 'Chinese' },
    { value: 'ja', label: 'Japanese' },
    { value: 'es', label: 'Spanish' },
    { value: 'fr', label: 'French' },
  ];

  /** Available model overrides. Empty = registry default for detected language. */
  const modelOptions = [
    { value: '', label: 'Auto (best for language)' },
    { value: 'large-v3-turbo', label: 'Whisper large-v3-turbo' },
    { value: 'SenseVoiceSmall', label: 'SenseVoice Small' },
  ];

  function handleEndMeetingClick() {
    showEndMeetingConfirm = true;
  }

  function confirmEndMeeting() {
    showEndMeetingConfirm = false;
    onEndMeeting();
  }

  function cancelEndMeeting() {
    showEndMeetingConfirm = false;
  }
</script>

<div class="toolbar">
  <!-- Audio source -->
  <div class="toolbar-group">
    <label class="toolbar-label">Source</label>
    <select
      value={selectedDeviceId}
      onchange={(e) => onDeviceChange(e.currentTarget.value)}
      class="toolbar-select"
    >
      {#each devices as device}
        <option value={device.deviceId}>
          {device.label || `Mic ${device.deviceId.slice(0, 8)}`}
        </option>
      {/each}
    </select>
  </div>

  <!-- Language selectors -->
  <div class="toolbar-group">
    <label class="toolbar-label">Source Lang</label>
    <select
      value={loopbackStore.sourceLanguage ?? ''}
      onchange={(e) => {
        const val = e.currentTarget.value;
        loopbackStore.sourceLanguage = val || null;
      }}
      class="toolbar-select"
    >
      {#each sourceLanguages as lang}
        <option value={lang.value}>{lang.label}</option>
      {/each}
    </select>
    {#if loopbackStore.sourceLanguage}
      <span class="auto-badge">{loopbackStore.sourceLanguage}</span>
    {:else}
      <span class="auto-badge">detecting...</span>
    {/if}
  </div>

  <div class="toolbar-group">
    <label class="toolbar-label">Target</label>
    <select
      value={loopbackStore.targetLanguage}
      onchange={(e) => loopbackStore.targetLanguage = e.currentTarget.value}
      class="toolbar-select"
    >
      <option value="en">English</option>
      <option value="zh">Chinese</option>
      <option value="ja">Japanese</option>
      <option value="es">Spanish</option>
      <option value="fr">French</option>
    </select>
  </div>

  <!-- Model override — defaults to registry's best for detected language -->
  <div class="toolbar-group">
    <label class="toolbar-label">Model</label>
    <select
      value={modelOverride}
      onchange={(e) => modelOverride = e.currentTarget.value}
      class="toolbar-select"
    >
      {#each modelOptions as opt}
        <option value={opt.value}>{opt.label}</option>
      {/each}
    </select>
  </div>

  <!-- Display mode switcher -->
  <div class="toolbar-group">
    <label class="toolbar-label">Mode</label>
    <div class="mode-switcher">
      {#each modes as mode}
        <button
          class="mode-btn"
          class:active={loopbackStore.displayMode === mode.value}
          onclick={() => loopbackStore.displayMode = mode.value}
        >
          {mode.label}
        </button>
      {/each}
    </div>
  </div>

  <!-- Capture controls -->
  <div class="toolbar-group">
    {#if loopbackStore.isCapturing}
      <Button variant="destructive" size="sm" onclick={onStopCapture}>
        Stop
      </Button>
    {:else}
      <Button size="sm" onclick={onStartCapture}>
        Start Capture
      </Button>
    {/if}
  </div>

  <!-- Meeting controls -->
  <div class="toolbar-group">
    {#if loopbackStore.isMeetingActive}
      <button class="meeting-btn active" onclick={handleEndMeetingClick}>
        <span class="recording-dot"></span>
        End Meeting
      </button>
    {:else}
      <button class="meeting-btn" onclick={onStartMeeting} disabled={!loopbackStore.isCapturing}>
        Start Meeting
      </button>
    {/if}
  </div>

  <!-- End Meeting confirmation dialog -->
  {#if showEndMeetingConfirm}
    <div class="confirm-overlay" onclick={cancelEndMeeting}>
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <div class="confirm-dialog" onclick|stopPropagation>
        <p>End this meeting? Recording and transcription will stop.</p>
        <div class="confirm-actions">
          <button class="confirm-btn cancel" onclick={cancelEndMeeting}>Cancel</button>
          <button class="confirm-btn confirm" onclick={confirmEndMeeting}>End Meeting</button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Connection status -->
  <div class="toolbar-group status-group">
    <div class="status-dot" class:up={loopbackStore.transcriptionStatus === 'up'}></div>
    <span class="status-label">STT</span>
    <div class="status-dot" class:up={loopbackStore.translationStatus === 'up'}></div>
    <span class="status-label">MT</span>
  </div>
</div>

<style>
  .toolbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border, #333);
    flex-wrap: wrap;
  }
  .toolbar-group {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .toolbar-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #888;
  }
  .toolbar-select {
    background: var(--bg-secondary, #1e293b);
    color: inherit;
    border: 1px solid var(--border, #333);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 13px;
  }
  .auto-badge {
    background: var(--bg-secondary, #1e293b);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    color: #60a5fa;
  }
  .mode-switcher {
    display: flex;
    border: 1px solid var(--border, #333);
    border-radius: 4px;
    overflow: hidden;
  }
  .mode-btn {
    padding: 4px 10px;
    font-size: 12px;
    background: transparent;
    color: inherit;
    border: none;
    cursor: pointer;
  }
  .mode-btn.active {
    background: var(--primary, #3b82f6);
    color: white;
  }
  .meeting-btn {
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 13px;
    cursor: pointer;
    background: #166534;
    color: #4ade80;
    border: 1px solid #22c55e;
  }
  .meeting-btn.active {
    background: #7f1d1d;
    color: #fca5a5;
    border-color: #ef4444;
  }
  .meeting-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .recording-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    margin-right: 4px;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  .status-group {
    margin-left: auto;
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
  }
  .status-dot.up {
    background: #22c55e;
  }
  .status-label {
    font-size: 10px;
    color: #888;
    margin-right: 8px;
  }
  .confirm-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  .confirm-dialog {
    background: var(--bg-secondary, #1e293b);
    border: 1px solid var(--border, #333);
    border-radius: 8px;
    padding: 20px 24px;
    max-width: 360px;
  }
  .confirm-dialog p {
    margin: 0 0 16px;
    font-size: 14px;
  }
  .confirm-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }
  .confirm-btn {
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 13px;
    cursor: pointer;
    border: 1px solid var(--border, #333);
  }
  .confirm-btn.cancel {
    background: transparent;
    color: inherit;
  }
  .confirm-btn.confirm {
    background: #7f1d1d;
    color: #fca5a5;
    border-color: #ef4444;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/components/loopback/Toolbar.svelte
git commit -m "feat(dashboard): add loopback toolbar with mode switcher, model override, and meeting controls"
```

---

### Task 8: Main loopback page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/loopback/+page.svelte`

- [ ] **Step 1: Write the loopback page**

This is the main page that wires everything together: audio capture → WebSocket → store → display components.

```svelte
<!-- modules/dashboard-service/src/routes/(app)/loopback/+page.svelte -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { loopbackStore } from '$lib/stores/loopback.svelte';
  import { AudioCapture } from '$lib/audio/capture';
  import { LoopbackWebSocket } from '$lib/audio/websocket';
  import type { ServerMessage } from '$lib/types/ws-messages';
  import Toolbar from '$lib/components/loopback/Toolbar.svelte';
  import SplitView from '$lib/components/loopback/SplitView.svelte';
  import SubtitleView from '$lib/components/loopback/SubtitleView.svelte';
  import TranscriptView from '$lib/components/loopback/TranscriptView.svelte';

  let devices = $state<MediaDeviceInfo[]>([]);
  let selectedDeviceId = $state('');

  /** Live elapsed timer string computed from meetingStartedAt. */
  let elapsedTime = $state('00:00:00');
  let elapsedTimerInterval: ReturnType<typeof setInterval> | null = null;

  const capture = new AudioCapture();
  let ws: LoopbackWebSocket | null = null;

  function updateElapsedTime() {
    if (!loopbackStore.meetingStartedAt) {
      elapsedTime = '00:00:00';
      return;
    }
    const start = new Date(loopbackStore.meetingStartedAt).getTime();
    const diff = Math.max(0, Date.now() - start);
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    elapsedTime = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }

  function handleMessage(msg: ServerMessage) {
    switch (msg.type) {
      case 'segment':
        loopbackStore.addSegment(msg);
        break;
      case 'interim':
        loopbackStore.updateInterim(msg);
        break;
      case 'translation':
        loopbackStore.addTranslation(msg);
        break;
      case 'meeting_started':
        loopbackStore.startMeeting(msg.session_id, msg.started_at);
        // Start elapsed timer
        elapsedTimerInterval = setInterval(updateElapsedTime, 1000);
        updateElapsedTime();
        break;
      case 'service_status':
        loopbackStore.transcriptionStatus = msg.transcription;
        loopbackStore.translationStatus = msg.translation;
        break;
      case 'recording_status':
        loopbackStore.isRecording = msg.recording;
        loopbackStore.recordingChunks = msg.chunks_written;
        break;
    }
  }

  async function startCapture() {
    // Connect WebSocket first, then wait for the ConnectedMessage before
    // sending start_session. This avoids the race condition where
    // startSession fires before the WebSocket is fully open and the
    // server has acknowledged with a session_id.
    const wsUrl = `ws://${window.location.hostname}:3000/api/audio/stream`;

    const connectedPromise = new Promise<void>((resolve, reject) => {
      ws = new LoopbackWebSocket({
        url: wsUrl,
        onMessage: (msg) => {
          // Resolve the promise when we get the 'connected' ack
          if (msg.type === 'connected') {
            resolve();
          }
          handleMessage(msg);
        },
        onStateChange: (state) => {
          loopbackStore.connectionState = state;
          if (state === 'error') {
            reject(new Error('WebSocket connection failed'));
          }
        },
        onError: (event) => {
          reject(new Error('WebSocket error'));
        },
      });
    });

    ws!.connect();

    // Wait for the server to send the ConnectedMessage with session_id
    await connectedPromise;

    // Start audio capture
    await capture.start({
      sourceType: 'mic',
      deviceId: selectedDeviceId || undefined,
      onChunk: (data) => {
        ws?.sendAudio(data);
      },
      onError: (err) => {
        console.error('Audio capture error:', err);
      },
    });

    loopbackStore.isCapturing = true;

    // Now safe to send start_session — WebSocket is connected and session_id is set
    ws!.startSession(capture.sampleRate, 1, selectedDeviceId || undefined);
  }

  async function stopCapture() {
    ws?.sendMessage({ type: 'end_session' });
    await capture.stop();
    ws?.disconnect();
    ws = null;
    loopbackStore.isCapturing = false;
    loopbackStore.connectionState = 'disconnected';
    if (elapsedTimerInterval) {
      clearInterval(elapsedTimerInterval);
      elapsedTimerInterval = null;
    }
  }

  function startMeeting() {
    ws?.sendMessage({ type: 'promote_to_meeting' });
  }

  function endMeeting() {
    ws?.sendMessage({ type: 'end_meeting' });
    loopbackStore.endMeeting();
    if (elapsedTimerInterval) {
      clearInterval(elapsedTimerInterval);
      elapsedTimerInterval = null;
    }
  }

  onMount(async () => {
    devices = await AudioCapture.getDevices();
    if (devices.length > 0) {
      selectedDeviceId = devices[0].deviceId;
    }
  });

  // Svelte does NOT await async onDestroy callbacks. Use a synchronous
  // teardown that fires stopCapture() without awaiting it.
  onDestroy(() => {
    stopCapture();
  });
</script>

<div class="loopback-page">
  <Toolbar
    {devices}
    {selectedDeviceId}
    onDeviceChange={(id) => selectedDeviceId = id}
    onStartCapture={startCapture}
    onStopCapture={stopCapture}
    onStartMeeting={startMeeting}
    onEndMeeting={endMeeting}
  />

  <!-- Meeting info bar -->
  {#if loopbackStore.isMeetingActive}
    <div class="meeting-bar">
      <span class="recording-indicator"></span>
      Meeting: {loopbackStore.meetingSessionId?.slice(0, 8)}
      | Elapsed: {elapsedTime}
      {#if loopbackStore.sourceLanguage}
        | Source: {loopbackStore.sourceLanguage}
      {/if}
      {#if loopbackStore.isRecording}
        | Chunks: {loopbackStore.recordingChunks}
      {/if}
    </div>
  {/if}

  <!-- Display area -->
  <div class="display-area">
    {#if loopbackStore.displayMode === 'split'}
      <SplitView />
    {:else if loopbackStore.displayMode === 'subtitle'}
      <SubtitleView />
    {:else}
      <TranscriptView />
    {/if}
  </div>
</div>

<style>
  .loopback-page {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  .meeting-bar {
    padding: 6px 16px;
    background: #7f1d1d;
    color: #fca5a5;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .recording-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    animation: pulse 1.5s ease-in-out infinite;
  }
  .display-area {
    flex: 1;
    overflow: hidden;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/loopback/+page.svelte
git commit -m "feat(dashboard): add loopback page wiring audio capture → WebSocket → display"
```

---

### Task 9: Add loopback to sidebar navigation

**Files:**
- Modify: `modules/dashboard-service/src/routes/(app)/+layout.svelte` (add nav link)

- [ ] **Step 1: Read the current layout**

Read the `+layout.svelte` file to find where navigation links are defined.

- [ ] **Step 2: Add "Loopback" link to the navigation**

Add a nav entry for `/loopback` alongside the existing routes (Config, Sessions, etc.).

- [ ] **Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/+layout.svelte
git commit -m "feat(dashboard): add loopback page to sidebar navigation"
```

---

## Summary

**Total tasks:** 9 tasks, ~20 steps
**Branch:** `plan-2/loopback-page`

After completing Plan 2:
- AudioWorklet captures mic/system/both audio at native quality (48kHz) with GC-friendly ring buffer
- Binary WebSocket sends Float32Array frames to orchestration (race-condition-free connect flow)
- Three display modes: Split (default), Subtitle overlay (configurable font/opacity, pop-out window), Transcript
- Toolbar with device selector, source language override, target language, model override, mode switcher
- Meeting promotion UI (Start/End Meeting with confirmation dialog, elapsed timer, detected language)
- Connection status dots for transcription and translation services
- Reactive Svelte 5 store manages all loopback state
- Loopback page accessible from dashboard sidebar navigation
