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
  /** Called after auto-reconnect succeeds (server sends ConnectedMessage).
   * Use this to re-send start_session so the server has session context. */
  onReconnect?: () => void;
}

export class LoopbackWebSocket {
  private ws: WebSocket | null = null;
  private options: LoopbackWSOptions;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _state: ConnectionState = 'disconnected';
  private _sessionId: string | null = null;
  private _hasConnectedOnce = false;  // C1: Track if this is a reconnect

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

          // C1 fix: If this is a reconnect (not the initial connect),
          // fire onReconnect so the page can re-send start_session.
          // Without this, the server receives raw audio on a fresh
          // connection with no session context.
          if (this._hasConnectedOnce) {
            this.options.onReconnect?.();
          }
          this._hasConnectedOnce = true;
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
    this._hasConnectedOnce = false;
    this.ws?.close();
    this.ws = null;
    this._sessionId = null;
    this.setState('disconnected');
  }

  /**
   * Send end_session, then wait for the server to finish draining
   * in-flight translations before disconnecting. Messages received
   * during the drain period are still dispatched to onMessage.
   *
   * @param timeoutMs Max time to wait for server close (default 5s)
   */
  async drainAndDisconnect(timeoutMs = 5000): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.disconnect();
      return;
    }

    // Prevent auto-reconnect during drain
    this.reconnectAttempts = this.maxReconnectAttempts;

    this.sendMessage({ type: 'end_session' });

    // Wait for the server to close the connection (after draining translations)
    // or for our timeout to expire.
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        resolve();
      }, timeoutMs);

      const wsRef = this.ws;
      if (wsRef) {
        const origOnClose = wsRef.onclose;
        wsRef.onclose = () => {
          clearTimeout(timer);
          if (origOnClose) origOnClose.call(wsRef, new CloseEvent('close'));
          resolve();
        };
      }
    });

    // Force-close if server didn't close within timeout
    this._hasConnectedOnce = false;
    this.ws?.close();
    this.ws = null;
    this._sessionId = null;
    this.setState('disconnected');
  }

  /** Send a binary audio chunk (Float32Array) */
  sendAudio(data: Float32Array): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    // I5: Backpressure — drop frames if send buffer exceeds 1MB.
    // Audio is real-time and lossy by nature; dropping under congestion
    // is better than unbounded memory growth in long sessions.
    if (this.ws.bufferedAmount > 1_048_576) return;
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
