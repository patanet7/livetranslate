import { captionStore } from '$lib/stores/caption.svelte';
import { WS_BASE } from '$lib/config';
import type { ServerMessage } from '$lib/types/ws-messages';
import type { CaptionEvent } from '$lib/stores/caption.svelte';
import { LoopbackWebSocket, type ConnectionState as WSState } from './websocket';

export interface SourceConfig {
  sourceLanguage: string | null;
  targetLanguage: string;
  interpreterLanguages?: [string, string];
  deviceId?: string;
  sampleRate?: number;
  channels?: number;
  firefliesSessionId?: string;
  /** Callback for audio level updates (0-1 range) from server-side capture */
  onLevel?: (rms: number) => void;
  /** Callback when meeting starts (for elapsed timer) */
  onMeetingStarted?: (sessionId: string, startedAt: string) => void;
  /** Callback for chunk count updates (server-side capture) */
  onChunkCount?: (count: number) => void;
}

export interface SourceAdapter {
  connect(config: SourceConfig): Promise<void>;
  disconnect(): void;
  sendConfig(config: Record<string, unknown>): void;
  sendAudio?(data: Float32Array): void;
}

/**
 * Loopback adapter — connects to /ws/loopback for browser mic or screencapture.
 *
 * Delegates to LoopbackWebSocket for transport, gaining auto-reconnect with
 * exponential backoff, 1 MB send-buffer backpressure, and defensive buffer
 * slicing on sendAudio. Previously this class hand-rolled a raw WebSocket
 * and lost all of those properties on every regression of this code path.
 */
export class LoopbackAdapter implements SourceAdapter {
  private ws: LoopbackWebSocket | null = null;
  private source: 'mic' | 'screencapture';
  private startSessionPayload: { sample_rate: number; channels: number; device_id?: string } | null = null;
  private onLevel?: (rms: number) => void;
  private onMeetingStarted?: (sessionId: string, startedAt: string) => void;
  private onChunkCount?: (count: number) => void;

  constructor(source: 'mic' | 'screencapture' = 'mic') {
    this.source = source;
  }

  async connect(config: SourceConfig): Promise<void> {
    this.onLevel = config.onLevel;
    this.onMeetingStarted = config.onMeetingStarted;
    this.onChunkCount = config.onChunkCount;
    this.startSessionPayload = {
      sample_rate: config.sampleRate ?? 48000,
      channels: config.channels ?? 1,
      device_id: config.deviceId,
    };

    return new Promise((resolve, reject) => {
      let settled = false;

      const ws = new LoopbackWebSocket({
        url: `${WS_BASE}/ws/loopback`,
        onMessage: (msg) => {
          this.handleMessage(
            msg,
            () => { if (!settled) { settled = true; resolve(); } },
            (err) => { if (!settled) { settled = true; reject(err); } },
          );
        },
        onStateChange: (state: WSState) => {
          // Mirror the transport state onto the caption store. The "connected"
          // transition is also fired here, but we drive connect()'s resolve
          // off the actual ConnectedMessage in handleMessage so callers know
          // the server-side session is ready, not just the socket.
          captionStore.connectionState = state;
        },
        onError: () => {
          captionStore.connectionState = 'error';
          if (!settled) { settled = true; reject(new Error('WebSocket connection failed')); }
        },
        onReconnect: () => {
          // After auto-reconnect, the server has a fresh socket with no session
          // context. Re-send start_session so transcription resumes.
          if (this.startSessionPayload) {
            ws.startSession(
              this.startSessionPayload.sample_rate,
              this.startSessionPayload.channels,
              this.startSessionPayload.device_id,
            );
          }
        },
      });
      this.ws = ws;
      ws.connect();

      // Send start_session as soon as the socket reaches 'connecting' isn't
      // sufficient — we have to wait for the server to send ConnectedMessage
      // first (which sets state to 'connected'). LoopbackWebSocket already
      // routes that as an onMessage with type='connected'; handleMessage will
      // then call ws.startSession.
    });
  }

  private handleMessage(
    msg: ServerMessage,
    resolve?: () => void,
    reject?: (err: Error) => void
  ): void {
    switch (msg.type) {
      case 'connected':
        // Server is ready for this session — send start_session now, then
        // resolve the connect() promise.
        if (this.ws && this.startSessionPayload) {
          this.ws.startSession(
            this.startSessionPayload.sample_rate,
            this.startSessionPayload.channels,
            this.startSessionPayload.device_id,
          );
        }
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
          this.onMeetingStarted?.(msg.session_id, msg.started_at);
        }
        break;
      case 'recording_status':
        captionStore.isRecording = msg.recording;
        captionStore.recordingChunks = msg.chunks_written;
        break;
      case 'service_status':
        captionStore.transcriptionStatus = msg.transcription === 'up' ? 'up' : 'down';
        captionStore.translationStatus = msg.translation === 'up' ? 'up' : 'down';
        break;
      case 'language_detected':
        captionStore.detectedLanguage = msg.language;
        break;
      case 'audio_level':
        // Server-side audio level (screencapture mode)
        if (typeof msg.rms === 'number') {
          this.onLevel?.(msg.rms);
        }
        if (typeof msg.chunks === 'number') {
          this.onChunkCount?.(msg.chunks);
        }
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
    // drainAndDisconnect is fire-and-forget here: SourceAdapter.disconnect()
    // is sync in the interface, and stopCapture() in the page already handles
    // the elapsed-timer wind-down. Letting the server drain in-flight final
    // translations is a nice-to-have, not a correctness requirement.
    this.ws?.drainAndDisconnect().catch(() => { /* ignore */ });
    this.ws = null;
    this.startSessionPayload = null;
  }

  sendConfig(config: Record<string, unknown>): void {
    this.ws?.sendMessage({ type: 'config', ...config } as never);
  }

  sendAudio(data: Float32Array): void {
    // LoopbackWebSocket does the defensive buffer slice + backpressure check.
    this.ws?.sendAudio(data);
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
      const langParam = config.targetLanguage
        ? `?target_language=${config.targetLanguage}`
        : '';
      const url = `${WS_BASE}/api/captions/stream/${sessionId}${langParam}`;
      this.ws = new WebSocket(url);
      let settled = false;

      this.ws.onopen = () => {
        captionStore.connectionState = 'connecting';
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data as string) as Record<string, unknown>;
          this.handleMessage(
            msg,
            () => { if (!settled) { settled = true; resolve(); } },
            (err) => { if (!settled) { settled = true; reject(err); } }
          );
        } catch (e) {
          captionStore.lastError = `Invalid JSON from server: ${e}`;
        }
      };

      this.ws.onerror = () => {
        captionStore.connectionState = 'error';
        if (!settled) {
          settled = true;
          reject(new Error('Fireflies WebSocket connection failed'));
        }
      };

      this.ws.onclose = () => {
        captionStore.connectionState = 'disconnected';
      };
    });
  }

  private handleMessage(
    msg: Record<string, unknown>,
    resolve?: () => void,
    _reject?: (err: Error) => void
  ): void {
    if (msg.event === 'connected') {
      captionStore.connectionState = 'connected';
      const currentCaptions = msg.current_captions as Array<Record<string, unknown>> | undefined;
      if (currentCaptions) {
        for (const cap of currentCaptions) {
          // Server sends raw caption objects; wrap as CaptionEvent for store ingestion
          captionStore.ingestCaptionEvent({
            event: 'caption_added',
            caption: cap,
          } as unknown as CaptionEvent);
        }
      }
      resolve?.();
    } else {
      captionStore.ingestCaptionEvent(msg as CaptionEvent);
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
export function createSourceAdapter(
  source: 'local' | 'screencapture' | 'fireflies'
): SourceAdapter {
  switch (source) {
    case 'local':
      return new LoopbackAdapter('mic');
    case 'screencapture':
      return new LoopbackAdapter('screencapture');
    case 'fireflies':
      return new FirefliesAdapter();
  }
}
