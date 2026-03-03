import { browser } from '$app/environment';

export type WsStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export class WebSocketStore {
	url = $state('');
	status = $state<WsStatus>('disconnected');
	reconnectAttempt = $state(0);
	lastCaptionId = $state<string | null>(null);

	#socket: WebSocket | null = null;
	#reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	#heartbeatTimer: ReturnType<typeof setInterval> | null = null;
	#pongTimer: ReturnType<typeof setTimeout> | null = null;
	#outboundBuffer: unknown[] = [];

	#maxReconnectDelay = 30_000;
	#maxReconnectAttempts = 10;
	#heartbeatIntervalMs = 15_000;
	#pongTimeoutMs = 5_000;

	onMessage: ((event: MessageEvent) => void) | null = null;
	onStatusChange: ((status: WsStatus) => void) | null = null;

	connect(url: string) {
		if (!browser) return;
		this.disconnect();
		this.url = url;
		this.#setStatus('connecting');
		this.#openSocket(url);
	}

	send(data: unknown) {
		if (this.#socket?.readyState === WebSocket.OPEN) {
			this.#socket.send(JSON.stringify(data));
		} else if (this.status === 'reconnecting') {
			this.#outboundBuffer.push(data);
		}
	}

	disconnect() {
		this.#clearTimers();
		if (this.#socket) {
			this.#socket.onclose = null;
			this.#socket.close(1000, 'Client disconnect');
			this.#socket = null;
		}
		this.#outboundBuffer = [];
		this.#setStatus('disconnected');
		this.reconnectAttempt = 0;
	}

	retry() {
		if (this.status === 'error' && this.url) {
			this.reconnectAttempt = 0;
			this.#setStatus('connecting');
			this.#openSocket(this.url);
		}
	}

	#openSocket(url: string) {
		try {
			this.#socket = new WebSocket(url);
		} catch {
			this.#setStatus('error');
			return;
		}

		this.#socket.onopen = () => {
			this.#setStatus('connected');

			// Send resume if reconnecting with a known last caption
			if (this.reconnectAttempt > 0 && this.lastCaptionId) {
				this.#socket?.send(
					JSON.stringify({ event: 'resume', last_caption_id: this.lastCaptionId })
				);
			}

			this.reconnectAttempt = 0;
			this.#flushBuffer();
			this.#startHeartbeat();
		};

		this.#socket.onmessage = (event) => {
			// Handle pong
			try {
				const data = JSON.parse(event.data);
				if (data.event === 'pong') {
					this.#clearPongTimer();
					return;
				}
				// Track last caption ID for resume
				if (data.caption?.id) {
					this.lastCaptionId = data.caption.id;
				} else if (data.caption_id) {
					this.lastCaptionId = data.caption_id;
				}
			} catch {
				// Non-JSON message, pass through
			}
			this.onMessage?.(event);
		};

		this.#socket.onclose = (event) => {
			this.#stopHeartbeat();
			if (!event.wasClean) {
				this.#scheduleReconnect();
			} else {
				this.#setStatus('disconnected');
			}
		};

		this.#socket.onerror = () => {
			// onerror is always followed by onclose, so we let onclose handle reconnection
		};
	}

	#startHeartbeat() {
		this.#stopHeartbeat();
		this.#heartbeatTimer = setInterval(() => {
			if (this.#socket?.readyState === WebSocket.OPEN) {
				this.#socket.send(JSON.stringify({ event: 'ping' }));
				this.#clearPongTimer();
				this.#pongTimer = setTimeout(() => {
					// No pong received — force reconnect
					this.#socket?.close(4000, 'Heartbeat timeout');
				}, this.#pongTimeoutMs);
			}
		}, this.#heartbeatIntervalMs);
	}

	#stopHeartbeat() {
		if (this.#heartbeatTimer) {
			clearInterval(this.#heartbeatTimer);
			this.#heartbeatTimer = null;
		}
		this.#clearPongTimer();
	}

	#clearPongTimer() {
		if (this.#pongTimer) {
			clearTimeout(this.#pongTimer);
			this.#pongTimer = null;
		}
	}

	#flushBuffer() {
		const messages = this.#outboundBuffer;
		this.#outboundBuffer = [];
		for (const msg of messages) {
			if (this.#socket?.readyState === WebSocket.OPEN) {
				this.#socket.send(JSON.stringify(msg));
			}
		}
	}

	#scheduleReconnect() {
		if (this.reconnectAttempt >= this.#maxReconnectAttempts || !this.url) {
			this.#setStatus('error');
			return;
		}

		const delay = Math.min(1000 * 2 ** this.reconnectAttempt, this.#maxReconnectDelay);
		this.reconnectAttempt++;
		this.#setStatus('reconnecting');
		this.#reconnectTimer = setTimeout(() => this.#openSocket(this.url), delay);
	}

	#clearTimers() {
		if (this.#reconnectTimer) {
			clearTimeout(this.#reconnectTimer);
			this.#reconnectTimer = null;
		}
		this.#stopHeartbeat();
	}

	#setStatus(status: WsStatus) {
		this.status = status;
		this.onStatusChange?.(status);
	}
}

export const wsStore = new WebSocketStore();
