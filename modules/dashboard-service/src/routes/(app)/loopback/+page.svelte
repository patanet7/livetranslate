<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { beforeNavigate } from '$app/navigation';
  import { captionStore } from '$lib/stores/caption.svelte';
  import { audioStore } from '$lib/stores/audio.svelte';
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
  import EarwyrmMini from '$lib/components/brand/EarwyrmMini.svelte';

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
            audioStore.push(rms);
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
          onMeetingStarted: (sessionId, startedAt) => {
            startElapsedTimer(startedAt);
          },
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
            audioStore.push(rms);
          },
          onMeetingStarted: (sessionId, startedAt) => {
            startElapsedTimer(startedAt);
          },
          onChunkCount: (count) => {
            captionStore.chunksSent = count;
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

    // Auto-start meeting when capture starts (simplifies UX — no separate button needed)
    startMeeting();
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
      audioStore.reset();
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

<svelte:head>
  <title>Loopback — LiveTranslate</title>
</svelte:head>

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
      <span class="alert-mark" aria-hidden="true"></span>
      <span class="alert-body">{captureError}</span>
      <button class="dismiss-btn" onclick={() => captureError = null} aria-label="Dismiss error">×</button>
    </div>
  {/if}

  {#if captionStore.isMeetingActive}
    <!-- Editorial meeting strip — replaces the red bar with a magazine running-head -->
    <div class="meeting-strip" role="status">
      <div class="ms-section ms-section--left">
        <span class="record-pip" aria-hidden="true"></span>
        <span class="byline">recording</span>
        {#if captionStore.meetingSessionId}
          <span class="session-id font-mono" title={captionStore.meetingSessionId}>
            {captionStore.meetingSessionId.slice(0, 8)}
          </span>
        {/if}
      </div>
      <div class="ms-section ms-section--center">
        <span class="elapsed font-mono tabular-nums">{elapsedTime}</span>
      </div>
      <div class="ms-section ms-section--right">
        {#if captionStore.detectedLanguage}
          <span class="meta-tag">
            <span class="eyebrow">detected</span>
            <span class="font-mono">{captionStore.detectedLanguage}</span>
          </span>
        {/if}
        {#if captionStore.isRecording}
          <span class="meta-tag">
            <span class="eyebrow">chunks</span>
            <span class="font-mono tabular-nums">{captionStore.recordingChunks}</span>
          </span>
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

  <!--
    Loopback Earwyrm — bottom-right live indicator (D4.6).
    Ear cups pulse with audioStore.rms; ring color reflects capture state.
    Out of the way until you're looking for it; nice when you are.
  -->
  <div class="loopback-mascot" aria-hidden={!captionStore.isCapturing}>
    <EarwyrmMini
      size={42}
      audioRms={audioStore.rms}
      state={captionStore.isCapturing ? 'live' : (captionStore.connectionState === 'error' ? 'offline' : 'idle')}
      title={captionStore.isCapturing ? 'Earwyrm — listening' : 'Earwyrm — idle'}
    />
  </div>
</div>

<style>
  .loopback-page {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    background: var(--paper);
    position: relative;
  }

  /* ── Loopback Earwyrm corner mascot — D4.6 ────────────────────
     Sits in the bottom-right of the loopback page above the page
     content (z-index above scroll, below modals). The mini variant
     handles audio-RMS reactivity through the audioStore. */
  .loopback-mascot {
    position: absolute;
    right: 1.25rem;
    bottom: 1.25rem;
    z-index: 5;
    pointer-events: none;
    /* Background ring softens contrast on dense transcripts */
    background: var(--paper);
    padding: 0.4rem;
    border-radius: 9999px;
    border: 1px solid var(--rule);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.05);
  }

  /* ── Meeting running-strip ─────────────────────────────────────
     Replaces the harsh red meeting-bar with a magazine running-head
     printed on a faint peach-tinted plate. Three columns: recording
     state, elapsed timer, metadata tags. */
  .meeting-strip {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    padding: 0.5rem 1.25rem;
    gap: 1rem;
    background: color-mix(in srgb, var(--peach) 16%, var(--paper));
    border-bottom: 1px solid var(--rule);
  }
  .ms-section {
    display: flex;
    align-items: center;
    gap: 0.625rem;
  }
  .ms-section--right { justify-content: flex-end; }
  .ms-section--center { justify-content: center; }

  .record-pip {
    width: 0.625rem;
    height: 0.625rem;
    border-radius: 9999px;
    background: var(--peach-deep);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--peach-deep) 18%, transparent);
    animation: record-breathe 2.4s ease-in-out infinite;
  }

  @keyframes record-breathe {
    0%, 100% { box-shadow: 0 0 0 3px color-mix(in srgb, var(--peach-deep) 18%, transparent); }
    50%      { box-shadow: 0 0 0 6px color-mix(in srgb, var(--peach-deep) 8%, transparent); }
  }

  .session-id {
    font-size: 0.6875rem;
    color: var(--ink-soft);
    max-width: 12rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .elapsed {
    font-size: 0.9375rem;
    color: var(--ink);
    letter-spacing: 0.04em;
  }

  .meta-tag {
    display: inline-flex;
    align-items: baseline;
    gap: 0.375rem;
    padding: 0.125rem 0.5rem;
    border: 1px solid var(--rule);
    border-radius: 9999px;
    background: var(--paper);
    font-size: 0.75rem;
    color: var(--ink-soft);
  }
  .meta-tag .eyebrow { color: var(--ink-faint); }

  /* ── Capture-error banner — editorial alert ─────────────────── */
  .capture-error {
    display: flex;
    align-items: center;
    gap: 0.875rem;
    padding: 0.75rem 1.25rem;
    background: color-mix(in srgb, var(--oxblood) 12%, var(--paper));
    border-bottom: 1px solid var(--oxblood);
    color: var(--ink);
    font-family: var(--font-body);
    font-size: 0.9375rem;
  }
  .alert-mark {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 9999px;
    background: var(--oxblood);
    flex-shrink: 0;
  }
  .alert-body {
    flex: 1;
  }
  .dismiss-btn {
    background: none;
    border: none;
    color: var(--ink-soft);
    font-size: 1.25rem;
    line-height: 1;
    cursor: pointer;
    padding: 0 0.25rem;
    transition: color 160ms ease;
  }
  .dismiss-btn:hover {
    color: var(--ink);
  }

  .display-area {
    flex: 1;
    min-height: 0;
    overflow: hidden;
    /* Page-color paper underneath the active view */
    background: var(--paper);
  }

  @media (prefers-reduced-motion: reduce) {
    .record-pip { animation: none; }
  }
</style>
