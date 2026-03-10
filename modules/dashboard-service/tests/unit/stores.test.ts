import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('$app/environment', () => ({ browser: true }));

const makeCaption = (overrides = {}) => ({
	id: `cap-${Math.random().toString(36).slice(2, 8)}`,
	text: 'Hola mundo',
	original_text: 'Hello world',
	translated_text: 'Hola mundo',
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

vi.mock('svelte-sonner', () => ({
	toast: {
		success: vi.fn(),
		error: vi.fn(),
		warning: vi.fn(),
		info: vi.fn()
	}
}));

describe('ToastStore', () => {
	let toastStore: typeof import('$lib/stores/toast.svelte').toastStore;
	let toast: typeof import('svelte-sonner').toast;

	beforeEach(async () => {
		const mod = await import('$lib/stores/toast.svelte');
		toastStore = mod.toastStore;
		const sonner = await import('svelte-sonner');
		toast = sonner.toast;
		vi.clearAllMocks();
	});

	it('add() defaults to info type and delegates to svelte-sonner', () => {
		toastStore.add('Hello');
		expect(toast.info).toHaveBeenCalledWith('Hello', { duration: 5000 });
	});

	it('success() delegates to toast.success', () => {
		toastStore.success('OK');
		expect(toast.success).toHaveBeenCalledWith('OK', { duration: 5000 });
	});

	it('error() delegates to toast.error with 8s duration', () => {
		toastStore.error('Fail');
		expect(toast.error).toHaveBeenCalledWith('Fail', { duration: 8000 });
	});

	it('warning() delegates to toast.warning', () => {
		toastStore.warning('Careful');
		expect(toast.warning).toHaveBeenCalledWith('Careful', { duration: 5000 });
	});

	it('info() delegates to toast.info', () => {
		toastStore.info('FYI');
		expect(toast.info).toHaveBeenCalledWith('FYI', { duration: 5000 });
	});

	it('add() with custom duration passes it through', () => {
		toastStore.add('Custom', 'success', 3000);
		expect(toast.success).toHaveBeenCalledWith('Custom', { duration: 3000 });
	});
});
