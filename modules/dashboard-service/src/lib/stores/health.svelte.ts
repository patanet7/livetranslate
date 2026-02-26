import { browser } from '$app/environment';

class HealthStore {
	status = $state<'healthy' | 'degraded' | 'down' | 'unknown'>('unknown');
	services = $state<Record<string, boolean>>({});
	#interval: ReturnType<typeof setInterval> | null = null;

	startPolling(intervalMs = 30_000) {
		if (!browser) return;
		this.#poll();
		this.#interval = setInterval(() => this.#poll(), intervalMs);
	}

	stopPolling() {
		if (this.#interval) {
			clearInterval(this.#interval);
			this.#interval = null;
		}
	}

	async #poll() {
		try {
			const res = await fetch('/api/health');
			const data = await res.json();
			this.status = data.status ?? 'unknown';
			this.services = data.services ?? {};
		} catch {
			this.status = 'down';
		}
	}
}

export const healthStore = new HealthStore();
