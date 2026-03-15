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
  let selectedDeviceId = $state<string | undefined>(undefined);
  let elapsedTime = $state('00:00:00');
  let elapsedTimerInterval: ReturnType<typeof setInterval> | null = null;

  let capture: AudioCapture | null = null;
  let ws: LoopbackWebSocket | null = null;

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
        startElapsedTimer(msg.started_at);
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
    if (loopbackStore.isCapturing) return;

    // Build WebSocket URL from current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:3000/ws/loopback`;

    // Create a promise that resolves when WS connects, so we can await it
    // before starting audio capture (avoids sending audio before session starts).
    let resolveConnected: () => void;
    let rejectConnected: (reason: Error) => void;
    const connectedPromise = new Promise<void>((resolve, reject) => {
      resolveConnected = resolve;
      rejectConnected = reject;
    });

    ws = new LoopbackWebSocket({
      url: wsUrl,
      onMessage: (msg) => {
        if (msg.type === 'connected') {
          resolveConnected();
        }
        handleMessage(msg);
      },
      onStateChange: (state) => {
        loopbackStore.connectionState = state;
        if (state === 'error') {
          rejectConnected(new Error('WebSocket connection failed'));
        }
      },
      onError: () => {
        rejectConnected(new Error('WebSocket error'));
      },
    });

    loopbackStore.connectionState = 'connecting';
    ws.connect();

    try {
      await connectedPromise;
    } catch {
      loopbackStore.connectionState = 'error';
      ws.disconnect();
      ws = null;
      return;
    }

    // Start audio capture
    capture = new AudioCapture();
    try {
      await capture.start({
        deviceId: selectedDeviceId,
        sourceType: 'system',
        onChunk: (data) => {
          ws?.sendAudio(data);
        },
        onError: (error) => {
          console.error('Audio capture error:', error);
          stopCapture();
        },
      });
    } catch (err) {
      console.error('Failed to start audio capture:', err);
      loopbackStore.connectionState = 'error';
      ws?.disconnect();
      ws = null;
      capture = null;
      return;
    }

    // Send start_session with sample rate (null-safe fallback to 48000)
    ws.startSession(capture.sampleRate ?? 48000, 1, selectedDeviceId);
    loopbackStore.isCapturing = true;
  }

  async function stopCapture() {
    if (ws && loopbackStore.isCapturing) {
      ws.sendMessage({ type: 'end_session' });
    }

    if (capture) {
      await capture.stop();
      capture = null;
    }

    if (ws) {
      ws.disconnect();
      ws = null;
    }

    loopbackStore.isCapturing = false;
    loopbackStore.connectionState = 'disconnected';
    stopElapsedTimer();
  }

  function startMeeting() {
    ws?.sendMessage({ type: 'promote_to_meeting' });
  }

  function endMeeting() {
    ws?.sendMessage({ type: 'end_meeting' });
    loopbackStore.endMeeting();
    stopElapsedTimer();
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
    try {
      devices = await AudioCapture.getDevices();
      if (devices.length > 0 && !selectedDeviceId) {
        selectedDeviceId = devices[0].deviceId;
      }
    } catch (err) {
      console.error('Failed to enumerate audio devices:', err);
    }
  });

  onDestroy(() => {
    // Synchronous teardown: fire-and-forget the async stop
    stopCapture();
  });
</script>

<div class="loopback-page">
  <Toolbar
    {devices}
    bind:selectedDeviceId
    onStart={startCapture}
    onStop={stopCapture}
    onStartMeeting={startMeeting}
    onEndMeeting={endMeeting}
  />

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
        {#if loopbackStore.sourceLanguage}
          <span class="lang-badge">{loopbackStore.sourceLanguage}</span>
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

  .display-area {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }
</style>
