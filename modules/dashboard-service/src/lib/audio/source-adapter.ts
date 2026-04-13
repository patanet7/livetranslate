import { captionStore } from '$lib/stores/caption.svelte';
import { WS_BASE } from '$lib/config';
import type { ServerMessage } from '$lib/types/ws-messages';
import type { CaptionEvent } from '$lib/stores/caption.svelte';

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
  private readonly maxReconnectAttempts = 5;
  private readonly reconnectDelay = 1000;

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
        const msg = JSON.parse(event.data as string) as ServerMessage;
        this.handleMessage(
          msg,
          () => {
            if (!settled) {
              settled = true;
              resolve();
            }
          },
          (err) => {
            if (!settled) {
              settled = true;
              reject(err);
            }
          }
        );
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
      const langParam = config.targetLanguage
        ? `?target_language=${config.targetLanguage}`
        : '';
      const url = `${WS_BASE}/api/captions/stream/${sessionId}${langParam}`;
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        captionStore.connectionState = 'connecting';
      };

      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data as string) as Record<string, unknown>;
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

  private handleMessage(
    msg: Record<string, unknown>,
    resolve?: () => void
  ): void {
    if (msg.event === 'connected') {
      captionStore.connectionState = 'connected';
      const currentCaptions = msg.current_captions as Array<Record<string, unknown>> | undefined;
      if (currentCaptions) {
        for (const cap of currentCaptions) {
          captionStore.ingestCaptionEvent({
            event: 'caption_added',
            caption: cap,
          } as CaptionEvent);
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
