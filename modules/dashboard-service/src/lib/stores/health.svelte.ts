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

			const rawStatus = data.status ?? 'unknown';
			this.status = rawStatus === 'down' ? 'down' : rawStatus as typeof this.status;

			// The backend returns services as Record<string, { status: string, ... }>.
			// Normalise to a boolean map so consumers can do simple truthiness checks.
			const raw: Record<string, unknown> = data.services ?? {};
			const boolMap: Record<string, boolean> = {};
			for (const [name, value] of Object.entries(raw)) {
				if (typeof value === 'boolean') {
					boolMap[name] = value;
				} else if (value !== null && typeof value === 'object') {
					boolMap[name] = (value as { status?: string }).status === 'healthy';
				} else {
					boolMap[name] = false;
				}
			}
			this.services = boolMap;
		} catch {
			this.status = 'down';
		}
	}
}

export const healthStore = new HealthStore();
