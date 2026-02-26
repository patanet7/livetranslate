import { describe, it, expect, vi } from 'vitest';

vi.mock('$app/environment', () => ({ browser: true }));

describe('HealthStore', () => {
	it('starts with unknown status', async () => {
		const { healthStore } = await import('$lib/stores/health.svelte');
		expect(healthStore.status).toBe('unknown');
	});
});

describe('CaptionStore', () => {
	it('adds and retrieves captions', async () => {
		const { CaptionStore } = await import('$lib/stores/captions.svelte');
		const store = new CaptionStore();
		store.addCaption({
			id: 'cap1',
			text: 'Hola mundo',
			original_text: 'Hello world',
			speaker_name: 'Alice',
			speaker_color: '#4CAF50',
			target_language: 'es',
			confidence: 0.95,
			duration_seconds: 4,
			created_at: new Date().toISOString(),
			expires_at: new Date(Date.now() + 10000).toISOString()
		});
		expect(store.captions.length).toBe(1);
		expect(store.captions[0].text).toBe('Hola mundo');
	});

	it('updates interim text', async () => {
		const { CaptionStore } = await import('$lib/stores/captions.svelte');
		const store = new CaptionStore();
		store.updateInterim('Hello wor');
		expect(store.interim).toBe('Hello wor');
		store.updateInterim('Hello world');
		expect(store.interim).toBe('Hello world');
	});
});
