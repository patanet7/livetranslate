import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('$app/environment', () => ({ browser: true }));

const makeCaption = (overrides = {}) => ({
	id: `cap-${Math.random().toString(36).slice(2, 8)}`,
	text: 'Hola mundo',
	original_text: 'Hello world',
	speaker_name: 'Alice',
	speaker_color: '#4CAF50',
	target_language: 'es',
	confidence: 0.95,
	duration_seconds: 4,
	created_at: new Date().toISOString(),
	expires_at: new Date(Date.now() + 10000).toISOString(),
	...overrides
});

describe('HealthStore', () => {
	it('starts with unknown status', async () => {
		const { healthStore } = await import('$lib/stores/health.svelte');
		expect(healthStore.status).toBe('unknown');
	});

	it('starts with empty services', async () => {
		const { healthStore } = await import('$lib/stores/health.svelte');
		expect(Object.keys(healthStore.services)).toHaveLength(0);
	});
});

describe('CaptionStore', () => {
	let CaptionStore: typeof import('$lib/stores/captions.svelte').CaptionStore;

	beforeEach(async () => {
		const mod = await import('$lib/stores/captions.svelte');
		CaptionStore = mod.CaptionStore;
	});

	it('adds and retrieves captions', () => {
		const store = new CaptionStore();
		store.addCaption(makeCaption({ id: 'cap1', text: 'Hola mundo' }));
		expect(store.captions.length).toBe(1);
		expect(store.captions[0].text).toBe('Hola mundo');
	});

	it('updates interim text', () => {
		const store = new CaptionStore();
		store.updateInterim('Hello wor');
		expect(store.interim).toBe('Hello wor');
		store.updateInterim('Hello world');
		expect(store.interim).toBe('Hello world');
	});

	it('updates an existing caption by id', () => {
		const store = new CaptionStore();
		store.addCaption(makeCaption({ id: 'cap1', text: 'Hola' }));
		store.updateCaption(makeCaption({ id: 'cap1', text: 'Hola mundo' }));
		expect(store.captions[0].text).toBe('Hola mundo');
	});

	it('removes a caption by id', () => {
		const store = new CaptionStore();
		store.addCaption(makeCaption({ id: 'cap1' }));
		store.addCaption(makeCaption({ id: 'cap2' }));
		expect(store.captions.length).toBe(2);
		store.removeCaption('cap1');
		expect(store.captions.length).toBe(1);
		expect(store.captions[0].id).toBe('cap2');
	});

	it('clears all captions and interim', () => {
		const store = new CaptionStore();
		store.addCaption(makeCaption({ id: 'cap1' }));
		store.addCaption(makeCaption({ id: 'cap2' }));
		store.updateInterim('typing...');
		store.clear();
		expect(store.captions.length).toBe(0);
		expect(store.interim).toBe('');
	});

	it('respects maxCaptions limit', () => {
		const store = new CaptionStore();
		store.maxCaptions = 3;
		for (let i = 0; i < 5; i++) {
			store.addCaption(makeCaption({ id: `cap${i}` }));
		}
		expect(store.captions.length).toBe(3);
		expect(store.captions[0].id).toBe('cap2');
		expect(store.captions[2].id).toBe('cap4');
	});

	it('preserves receivedAt on update', () => {
		const store = new CaptionStore();
		store.addCaption(makeCaption({ id: 'cap1' }));
		const originalReceivedAt = store.captions[0].receivedAt;
		store.updateCaption(makeCaption({ id: 'cap1', text: 'Updated' }));
		expect(store.captions[0].receivedAt).toBe(originalReceivedAt);
	});
});

describe('ToastStore', () => {
	let toastStore: typeof import('$lib/stores/toast.svelte').toastStore;

	beforeEach(async () => {
		vi.useFakeTimers();
		const mod = await import('$lib/stores/toast.svelte');
		toastStore = mod.toastStore;
		// Clear any leftover toasts
		toastStore.toasts = [];
	});

	it('adds a toast with default info type', () => {
		toastStore.add('Hello');
		expect(toastStore.toasts.length).toBe(1);
		expect(toastStore.toasts[0].type).toBe('info');
		expect(toastStore.toasts[0].message).toBe('Hello');
	});

	it('convenience methods set correct types', () => {
		toastStore.success('OK');
		toastStore.error('Fail');
		toastStore.warning('Careful');
		toastStore.info('FYI');
		expect(toastStore.toasts.map((t) => t.type)).toEqual([
			'success',
			'error',
			'warning',
			'info'
		]);
	});

	it('dismisses a toast by id', () => {
		const id = toastStore.add('Temp');
		expect(toastStore.toasts.length).toBe(1);
		toastStore.dismiss(id);
		expect(toastStore.toasts.length).toBe(0);
	});

	it('auto-dismisses after duration', () => {
		toastStore.add('Auto', 'info', 3000);
		expect(toastStore.toasts.length).toBe(1);
		vi.advanceTimersByTime(3000);
		expect(toastStore.toasts.length).toBe(0);
	});

	it('error toasts have 8s duration', () => {
		toastStore.error('Oops');
		expect(toastStore.toasts.length).toBe(1);
		vi.advanceTimersByTime(5000);
		expect(toastStore.toasts.length).toBe(1); // still visible at 5s
		vi.advanceTimersByTime(3000);
		expect(toastStore.toasts.length).toBe(0); // gone at 8s
	});
});
