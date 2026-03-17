<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { beforeNavigate } from '$app/navigation';
  import { loopbackStore } from '$lib/stores/loopback.svelte';
  import { AudioCapture } from '$lib/audio/capture';
  import { LoopbackWebSocket } from '$lib/audio/websocket';
  import type { ServerMessage } from '$lib/types/ws-messages';
  import { WS_BASE } from '$lib/config';
  import { runDemo, type DemoHandle } from '$lib/loopback/demo-script';
  import { startHealthPoller, type HealthPollerHandle } from '$lib/loopback/health-poller';
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
  let demoHandle = $state<DemoHandle | null>(null);
  let isDemoRunning = $derived(demoHandle !== null);
  let isDraining = $state(false);

  let capture: AudioCapture | null = null;
  let ws: LoopbackWebSocket | null = null;
  let healthPoller: HealthPollerHandle | null = null;
  let stopping = false;  // C3: Re-entrancy guard for stopCapture

  function handleMessage(msg: ServerMessage) {
    switch (msg.type) {
      case 'segment':
        loopbackStore.addSegment(msg);
        break;
      case 'interim':
        loopbackStore.updateInterim(msg);
        break;
      case 'translation_chunk':
        loopbackStore.appendTranslationChunk(msg);
        break;
      case 'translation':
        loopbackStore.addTranslation(msg);
        break;
      case 'meeting_started':
        // C3: Guard against malformed message
        if (typeof msg.session_id === 'string' && typeof msg.started_at === 'string') {
          loopbackStore.startMeeting(msg.session_id, msg.started_at);
          startElapsedTimer(msg.started_at);
        }
        break;
      case 'service_status':
        // C3: Guard field types
        if (typeof msg.transcription === 'string' && typeof msg.translation === 'string') {
          loopbackStore.transcriptionStatus = msg.transcription as 'up' | 'down';
          loopbackStore.translationStatus = msg.translation as 'up' | 'down';
        }
        break;
      case 'recording_status':
        if (typeof msg.recording === 'boolean' && typeof msg.chunks_written === 'number') {
          loopbackStore.isRecording = msg.recording;
          loopbackStore.recordingChunks = msg.chunks_written;
        }
        break;
      // M6: Handle language_detected and backend_switched messages
      // Note: language_detected is informational — it must NOT overwrite
      // sourceLanguage (the user's dropdown selection). Otherwise auto-detect
      // gets locked to the first detected language on the next session start.
      case 'language_detected':
        if (typeof msg.language === 'string') {
          loopbackStore.detectedLanguage = msg.language;
        }
        break;
      case 'backend_switched':
        break;
      case 'error':
        loopbackStore.lastError = msg.message;
        captureError = msg.message;
        break;
    }
  }

  async function startCapture() {
    if (loopbackStore.isCapturing) return;
    captureError = null;

    // S2: Use WS_BASE from config instead of hardcoded port
    const wsUrl = `${WS_BASE}/ws/loopback`;

    // C2 fix: Use a settled flag to prevent calling resolve/reject multiple times.
    // Both onError and onStateChange('error') can fire, and reject after resolve is benign
    // but masks the actual failure sequence.
    let settled = false;
    let resolveConnected: () => void;
    let rejectConnected: (reason: Error) => void;
    const connectedPromise = new Promise<void>((resolve, reject) => {
      resolveConnected = resolve;
      rejectConnected = reject;
    });

    ws = new LoopbackWebSocket({
      url: wsUrl,
      onMessage: (msg) => {
        if (msg.type === 'connected' && !settled) {
          settled = true;
          resolveConnected();
        }
        handleMessage(msg);
      },
      onStateChange: (state) => {
        loopbackStore.connectionState = state;
        if (state === 'error' && !settled) {
          settled = true;
          rejectConnected(new Error('WebSocket connection failed'));
        }
      },
      onError: () => {
        if (!settled) {
          settled = true;
          rejectConnected(new Error('WebSocket error'));
        }
      },
      // C1 fix: Re-send start_session after auto-reconnect so the server
      // has session context for incoming audio frames.
      onReconnect: () => {
        if (capture) {
          ws?.startSession(capture.sampleRate ?? 48000, 1, selectedDeviceId || undefined);
        }
      },
    });

    loopbackStore.connectionState = 'connecting';
    ws.connect();

    try {
      await connectedPromise;
    } catch {
      captureError = 'Could not connect to orchestration service. Is it running?';
      loopbackStore.connectionState = 'error';
      ws.disconnect();
      ws = null;
      return;
    }

    // Start audio capture with selected source type
    capture = new AudioCapture();
    try {
      await capture.start({
        deviceId: selectedDeviceId || undefined,
        sourceType: 'mic',
        onChunk: (data) => {
          ws?.sendAudio(data);
          loopbackStore.chunksSent++;
        },
        onError: (error) => {
          console.error('Audio capture error:', error);
          stopCapture();
        },
        onLevel: (rms) => {
          audioLevel = rms;
        },
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // Categorize error: permission vs device config vs generic
      if (msg.includes('Permission') || msg.includes('NotAllowed')) {
        captureError = `Microphone access denied. Please allow microphone access in your browser.`;
      } else if (msg.includes('loopback') || msg.includes('BlackHole') || msg.includes('device')) {
        captureError = msg;
      } else {
        captureError = `Audio capture failed: ${msg}`;
      }
      loopbackStore.connectionState = 'error';
      ws?.disconnect();
      ws = null;
      capture = null;
      return;
    }

    // Send start_session with sample rate (null-safe fallback to 48000)
    ws.startSession(capture.sampleRate ?? 48000, 1, selectedDeviceId);

    // Send initial config so server knows target language, source language,
    // and model from the toolbar state at capture start (not just on change).
    // Always send `language` (even null for auto-detect) so the server has
    // the complete state after a page refresh.
    if (loopbackStore.displayMode === 'interpreter') {
      ws.sendMessage({
        type: 'config',
        interpreter_languages: [loopbackStore.interpreterLangA, loopbackStore.interpreterLangB],
      });
    } else {
      ws.sendMessage({
        type: 'config',
        target_language: loopbackStore.targetLanguage,
        language: loopbackStore.sourceLanguage,
      });
    }

    loopbackStore.isCapturing = true;
  }

  async function stopCapture() {
    // C3: Re-entrancy guard — stopCapture can be called from onError callback
    // during the stop() sequence itself, causing interleaved cleanup.
    if (stopping) return;
    stopping = true;
    try {
      if (capture) {
        await capture.stop();
        capture = null;
      }

      if (ws && loopbackStore.isCapturing) {
        // Drain: send end_session and wait for server to finish sending
        // pending translations before closing the WebSocket.
        isDraining = true;
        await ws.drainAndDisconnect(5000);
        isDraining = false;
        ws = null;
      } else if (ws) {
        ws.disconnect();
        ws = null;
      }

      loopbackStore.isCapturing = false;
      loopbackStore.connectionState = 'disconnected';
      audioLevel = 0;
      stopElapsedTimer();
    } finally {
      isDraining = false;
      stopping = false;
    }
  }

  function startMeeting() {
    ws?.sendMessage({ type: 'promote_to_meeting' });
  }

  function endMeeting() {
    ws?.sendMessage({ type: 'end_meeting' });
    loopbackStore.endMeeting();
    stopElapsedTimer();
  }

  function startDemo() {
    if (loopbackStore.isCapturing || isDemoRunning) return;
    loopbackStore.clear();
    captureError = null;
    demoHandle = runDemo(loopbackStore, {
      onComplete: () => { demoHandle = null; },
      onMessage: handleMessage,
    });
  }

  function stopDemo() {
    demoHandle?.stop();
    demoHandle = null;
  }

  /** I1/I2: Send config changes (model, language, target_language, interpreter_languages) to server via WebSocket */
  function handleConfigChange(config: { model?: string; language?: string | null; target_language?: string; interpreter_languages?: [string, string] | null }) {
    ws?.sendMessage({
      type: 'config',
      ...(config.language !== undefined ? { language: config.language } : {}),
      ...(config.model !== undefined ? { model: config.model } : {}),
      ...(config.target_language !== undefined ? { target_language: config.target_language } : {}),
      ...(config.interpreter_languages !== undefined ? { interpreter_languages: config.interpreter_languages } : {}),
    });
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
    // I4: Reset transient state on mount — the singleton store persists
    // across SvelteKit navigations but AudioCapture/WS are recreated.
    loopbackStore.isCapturing = false;
    loopbackStore.connectionState = 'disconnected';

    try {
      devices = await AudioCapture.getDevices();
      if (devices.length > 0 && !selectedDeviceId) {
        selectedDeviceId = devices[0].deviceId;
      }
    } catch (err) {
      console.error('Failed to enumerate audio devices:', err);
    }

    // Poll actual service health from /api/health every 5s
    healthPoller = startHealthPoller((health) => {
      // Don't overwrite demo-driven status
      if (!isDemoRunning) {
        loopbackStore.transcriptionStatus = health.transcription;
        loopbackStore.translationStatus = health.translation;
      }
    });
  });

  // I6: Clean up audio capture before SvelteKit navigates away.
  // beforeNavigate callbacks must be synchronous — async callbacks return a
  // Promise that SvelteKit ignores, so the navigation proceeds immediately.
  // We fire-and-forget stopCapture(); the re-entrancy guard inside it
  // prevents double-cleanup if onDestroy also fires.
  beforeNavigate(() => {
    healthPoller?.stop();
    stopDemo();
    stopCapture();
  });

  onDestroy(() => {
    // Fallback for non-navigation unmounts (e.g., HMR)
    healthPoller?.stop();
    stopDemo();
    stopCapture();
  });
</script>

<div class="loopback-page">
  <Toolbar
    {devices}
    bind:selectedDeviceId
    {audioLevel}
    onStartCapture={startCapture}
    onStopCapture={stopCapture}
    onStartMeeting={startMeeting}
    onEndMeeting={endMeeting}
    onConfigChange={handleConfigChange}
    onStartDemo={startDemo}
    onStopDemo={stopDemo}
    {isDemoRunning}
    {isDraining}
  />

  {#if captureError}
    <div class="capture-error" role="alert">
      <span>{captureError}</span>
      <button class="dismiss-btn" onclick={() => captureError = null} aria-label="Dismiss error">&times;</button>
    </div>
  {/if}

  {#if loopbackStore.isMeetingActive}
    <div class="meeting-bar">
      <div class="meeting-bar-left">
        <span class="recording-dot"></span>
        <span class="meeting-label">Meeting</span>
        {#if loopbackStore.meetingSessionId}
          <span class="session-id">{loopbackStore.meetingSessionId}</span>
        {/if}
      </div>
      <div class="meeting-bar-center">
        <span class="elapsed">{elapsedTime}</span>
      </div>
      <div class="meeting-bar-right">
        {#if loopbackStore.detectedLanguage}
          <span class="lang-badge">{loopbackStore.detectedLanguage}</span>
        {/if}
        {#if loopbackStore.isRecording}
          <span class="chunks-badge">{loopbackStore.recordingChunks} chunks</span>
        {/if}
      </div>
    </div>
  {/if}

  <div class="display-area">
    {#if loopbackStore.displayMode === 'split'}
      <SplitView />
    {:else if loopbackStore.displayMode === 'subtitle'}
      <SubtitleView />
    {:else if loopbackStore.displayMode === 'interpreter'}
      <InterpreterView />
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
    min-height: 0;
  }

  .meeting-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 16px;
    background: #991b1b;
    color: #fecaca;
    font-size: 13px;
    gap: 12px;
  }

  .meeting-bar-left,
  .meeting-bar-center,
  .meeting-bar-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .meeting-bar-left {
    flex: 1;
  }

  .meeting-bar-right {
    flex: 1;
    justify-content: flex-end;
  }

  .recording-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .meeting-label {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .session-id {
    font-family: monospace;
    font-size: 11px;
    opacity: 0.7;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .elapsed {
    font-family: monospace;
    font-weight: 600;
    font-size: 14px;
  }

  .lang-badge,
  .chunks-badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    background: rgba(255, 255, 255, 0.15);
  }

  .capture-error {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: #7f1d1d;
    color: #fecaca;
    font-size: 13px;
    gap: 12px;
  }

  .dismiss-btn {
    background: none;
    border: none;
    color: #fecaca;
    font-size: 18px;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
  }

  .dismiss-btn:hover {
    color: white;
  }

  .display-area {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }
</style>
