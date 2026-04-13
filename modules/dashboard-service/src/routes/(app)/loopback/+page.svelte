<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { beforeNavigate } from '$app/navigation';
  import { captionStore } from '$lib/stores/caption.svelte';
  import { AudioCapture } from '$lib/audio/capture';
  import { createSourceAdapter, type SourceAdapter } from '$lib/audio/source-adapter';
  import type { ServerMessage } from '$lib/types/ws-messages';
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
  let adapter: SourceAdapter | null = null;
  let healthPoller: HealthPollerHandle | null = null;
  let stopping = false;  // C3: Re-entrancy guard for stopCapture

  // handleMessage is used only by the demo script, which routes through onMessage
  function handleMessage(msg: ServerMessage) {
    switch (msg.type) {
      case 'segment':
        captionStore.ingestSegment(msg);
        break;
      case 'interim':
        captionStore.ingestInterim(msg.text, msg.confidence);
        break;
      case 'translation_chunk':
        captionStore.ingestTranslationChunk(msg);
        break;
      case 'translation':
        captionStore.ingestTranslation(msg);
        break;
      case 'meeting_started':
        if (typeof msg.session_id === 'string' && typeof msg.started_at === 'string') {
          captionStore.startMeeting(msg.session_id, msg.started_at);
          startElapsedTimer(msg.started_at);
        }
        break;
      case 'service_status':
        if (typeof msg.transcription === 'string' && typeof msg.translation === 'string') {
          captionStore.transcriptionStatus = msg.transcription as 'up' | 'down';
          captionStore.translationStatus = msg.translation as 'up' | 'down';
        }
        break;
      case 'recording_status':
        if (typeof msg.recording === 'boolean' && typeof msg.chunks_written === 'number') {
          captionStore.isRecording = msg.recording;
          captionStore.recordingChunks = msg.chunks_written;
        }
        break;
      // Note: language_detected is informational — it must NOT overwrite
      // sourceLanguage (the user's dropdown selection). Otherwise auto-detect
      // gets locked to the first detected language on the next session start.
      case 'language_detected':
        if (typeof msg.language === 'string') {
          captionStore.detectedLanguage = msg.language;
        }
        break;
      case 'backend_switched':
        break;
      case 'error':
        captionStore.lastError = msg.message;
        captureError = msg.message;
        break;
    }
  }

  async function startCapture() {
    if (captionStore.isCapturing) return;
    captureError = null;

    const source = captionStore.captionSource;
    adapter = createSourceAdapter(source);

    if (source === 'local') {
      // Local mic: start AudioCapture, then connect adapter
      capture = new AudioCapture();
      try {
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
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes('Permission') || msg.includes('NotAllowed')) {
          captureError = `Microphone access denied. Please allow microphone access in your browser.`;
        } else if (msg.includes('loopback') || msg.includes('BlackHole') || msg.includes('device')) {
          captureError = msg;
        } else {
          captureError = `Audio capture failed: ${msg}`;
        }
        captionStore.connectionState = 'error';
        adapter = null;
        capture = null;
        return;
      }

      try {
        await adapter.connect({
          sourceLanguage: captionStore.sourceLanguage,
          targetLanguage: captionStore.targetLanguage,
          interpreterLanguages: captionStore.displayMode === 'interpreter'
            ? [captionStore.interpreterLangA, captionStore.interpreterLangB]
            : undefined,
          deviceId: selectedDeviceId || undefined,
          sampleRate: capture.sampleRate ?? 48000,
          channels: 1,
        });
      } catch {
        captureError = 'Could not connect to orchestration service. Is it running?';
        captionStore.connectionState = 'error';
        await capture.stop();
        capture = null;
        adapter = null;
        return;
      }
    } else if (source === 'screencapture') {
      // ScreenCaptureKit: server captures audio via Swift binary, no browser AudioCapture needed
      try {
        await adapter.connect({
          sourceLanguage: captionStore.sourceLanguage,
          targetLanguage: captionStore.targetLanguage,
          interpreterLanguages: captionStore.displayMode === 'interpreter'
            ? [captionStore.interpreterLangA, captionStore.interpreterLangB]
            : undefined,
          onLevel: (rms) => {
            audioLevel = rms;
          },
        });
      } catch {
        captureError = 'Could not start system audio capture. Is the orchestration service running?';
        captionStore.connectionState = 'error';
        adapter = null;
        return;
      }
    } else {
      // Fireflies: just connect adapter, no AudioCapture needed
      try {
        await adapter.connect({
          sourceLanguage: captionStore.sourceLanguage,
          targetLanguage: captionStore.targetLanguage,
          firefliesSessionId: captionStore.firefliesSessionId ?? undefined,
        });
      } catch {
        captureError = 'Could not connect to Fireflies stream. Check the session ID.';
        captionStore.connectionState = 'error';
        adapter = null;
        return;
      }
    }

    captionStore.isCapturing = true;
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

      if (adapter) {
        isDraining = true;
        adapter.disconnect();
        isDraining = false;
        adapter = null;
      }

      captionStore.isCapturing = false;
      captionStore.connectionState = 'disconnected';
      audioLevel = 0;
      stopElapsedTimer();
    } finally {
      isDraining = false;
      stopping = false;
    }
  }

  function startMeeting() {
    adapter?.sendConfig({ type: 'promote_to_meeting' });
  }

  function endMeeting() {
    adapter?.sendConfig({ type: 'end_meeting' });
    captionStore.endMeeting();
    stopElapsedTimer();
  }

  function startDemo() {
    if (captionStore.isCapturing || isDemoRunning) return;
    captionStore.clear();
    captureError = null;
    // Shim: demo script uses addSegment/addTranslation but we always pass onMessage,
    // so those fallback paths are never hit. Status and connection fields are wired
    // directly to captionStore.
    const demoStoreShim = {
      addSegment: () => {},
      addTranslation: () => {},
      clear: () => captionStore.clear(),
      get transcriptionStatus() { return captionStore.transcriptionStatus; },
      set transcriptionStatus(v: 'up' | 'down') { captionStore.transcriptionStatus = v; },
      get translationStatus() { return captionStore.translationStatus; },
      set translationStatus(v: 'up' | 'down') { captionStore.translationStatus = v; },
      get connectionState() { return captionStore.connectionState; },
      set connectionState(v: 'disconnected' | 'connecting' | 'connected' | 'error') { captionStore.connectionState = v; },
      get isCapturing() { return captionStore.isCapturing; },
      set isCapturing(v: boolean) { captionStore.isCapturing = v; },
    };
    demoHandle = runDemo(demoStoreShim, {
      onComplete: () => { demoHandle = null; },
      onMessage: handleMessage,
    });
  }

  function stopDemo() {
    demoHandle?.stop();
    demoHandle = null;
  }

  /** Send config changes (model, language, target_language, interpreter_languages) to server via adapter */
  function handleConfigChange(config: { model?: string; language?: string | null; target_language?: string; interpreter_languages?: [string, string] | null }) {
    adapter?.sendConfig({
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
    // Reset transient state on mount — the singleton store persists
    // across SvelteKit navigations but AudioCapture/adapter are recreated.
    captionStore.isCapturing = false;
    captionStore.connectionState = 'disconnected';

    try {
      // Request permission first to get device labels (browsers hide labels without permission)
      devices = await AudioCapture.requestPermissionAndGetDevices();
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
        captionStore.transcriptionStatus = health.transcription;
        captionStore.translationStatus = health.translation;
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

  {#if captionStore.isMeetingActive}
    <div class="meeting-bar">
      <div class="meeting-bar-left">
        <span class="recording-dot"></span>
        <span class="meeting-label">Meeting</span>
        {#if captionStore.meetingSessionId}
          <span class="session-id">{captionStore.meetingSessionId}</span>
        {/if}
      </div>
      <div class="meeting-bar-center">
        <span class="elapsed">{elapsedTime}</span>
      </div>
      <div class="meeting-bar-right">
        {#if captionStore.detectedLanguage}
          <span class="lang-badge">{captionStore.detectedLanguage}</span>
        {/if}
        {#if captionStore.isRecording}
          <span class="chunks-badge">{captionStore.recordingChunks} chunks</span>
        {/if}
      </div>
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
