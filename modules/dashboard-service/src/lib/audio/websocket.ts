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

    // I4 fix: Reset reconnect counter so auto-reconnect works after
    // a manual disconnect() → connect() cycle.
    this.reconnectAttempts = 0;
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
          // S5: Warn-and-continue on version mismatch is intentional.
          // The protocol is designed for forward-compatibility — older clients
          // can connect to newer servers. A hard disconnect here would break
          // rolling deployments where server upgrades before client cache expires.
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
    // C2 fix: Defensive slice — data.buffer could be a view on a larger
    // ArrayBuffer or reference a detached buffer if the typed array was
    // created from a transferred buffer. Slicing ensures we send exactly
    // the bytes that correspond to the Float32Array.
    this.ws.send(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
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
