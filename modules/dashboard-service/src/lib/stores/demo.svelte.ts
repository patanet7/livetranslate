import { browser } from '$app/environment';

class DemoStore {
	active = $state(false);
	sessionId = $state('');
	mode = $state<'passthrough' | 'pretranslated'>('passthrough');
	speakers = $state<string[]>([]);
	loading = $state(false);

	async checkStatus() {
		if (!browser) return;
		try {
			const res = await fetch('/api/fireflies/demo/status');
			if (res.ok) {
				const data = await res.json();
				if (data.active) {
					this.active = true;
					this.sessionId = data.session_id ?? '';
					this.mode = data.mode ?? 'passthrough';
					this.speakers = data.speakers ?? [];
				}
			}
		} catch {
			/* ignore */
		}
	}

	async start(mode: 'passthrough' | 'pretranslated' = 'passthrough') {
		this.loading = true;
		try {
			const res = await fetch(`/api/fireflies/demo/start?mode=${mode}`, { method: 'POST' });
			if (res.ok) {
				const data = await res.json();
				this.active = true;
				this.sessionId = data.session_id ?? '';
				this.mode = mode;
				this.speakers = data.speakers ?? [];
			}
		} finally {
			this.loading = false;
		}
	}

	async stop() {
		this.loading = true;
		try {
			await fetch('/api/fireflies/demo/stop', { method: 'POST' });
			this.active = false;
			this.sessionId = '';
			this.speakers = [];
		} finally {
			this.loading = false;
		}
	}
}

export const demoStore = new DemoStore();
