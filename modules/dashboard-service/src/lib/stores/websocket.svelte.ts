import { browser } from '$app/environment';

export class WebSocketStore {
	url = $state('');
	status = $state<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
	#socket: WebSocket | null = null;
	#reconnectAttempt = 0;
	#reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	#maxReconnectDelay = 30_000;
	onMessage: ((event: MessageEvent) => void) | null = null;

	connect(url: string) {
		if (!browser) return;
		this.disconnect();
		this.url = url;
		this.status = 'connecting';

		this.#socket = new WebSocket(url);
		this.#socket.onopen = () => {
			this.status = 'connected';
			this.#reconnectAttempt = 0;
		};
		this.#socket.onmessage = (event) => this.onMessage?.(event);
		this.#socket.onclose = (event) => {
			this.status = 'disconnected';
			if (!event.wasClean) this.#scheduleReconnect();
		};
		this.#socket.onerror = () => {
			this.status = 'error';
		};
	}

	send(data: unknown) {
		if (this.#socket?.readyState === WebSocket.OPEN) {
			this.#socket.send(JSON.stringify(data));
		}
	}

	disconnect() {
		if (this.#reconnectTimer) {
			clearTimeout(this.#reconnectTimer);
			this.#reconnectTimer = null;
		}
		if (this.#socket) {
			this.#socket.onclose = null;
			this.#socket.close(1000, 'Client disconnect');
			this.#socket = null;
		}
		this.status = 'disconnected';
		this.#reconnectAttempt = 0;
	}

	#scheduleReconnect() {
		const delay = Math.min(1000 * 2 ** this.#reconnectAttempt, this.#maxReconnectDelay);
		this.#reconnectAttempt++;
		this.#reconnectTimer = setTimeout(() => this.connect(this.url), delay);
	}
}

export const wsStore = new WebSocketStore();
