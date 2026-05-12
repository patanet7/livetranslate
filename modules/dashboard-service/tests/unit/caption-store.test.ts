/**
 * Tests for caption store translation lifecycle.
 *
 * Covers: translationState field, last-writer-wins guards,
 * chunk-on-draft clear, segment replacement preserves translation,
 * localStorage persistence.
 *
 * Migrated from loopback-store.test.ts when loopback.svelte.ts was
 * superseded by caption.svelte.ts (single store for both loopback and
 * Fireflies sources). API names changed: addSegment → ingestSegment,
 * addTranslation → ingestTranslation, appendTranslationChunk → ingestTranslationChunk.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { SegmentMessage, TranslationMessage, TranslationChunkMessage } from '$lib/types/ws-messages';

const STORAGE_KEY = 'livetranslate:caption-config';

function makeSegment(overrides: Partial<SegmentMessage> = {}): SegmentMessage {
	return {
		type: 'segment',
		segment_id: 1,
		text: '你好世界',
		language: 'zh',
		confidence: 0.95,
		stable_text: '你好世界',
		unstable_text: '',
		is_final: false,
		is_draft: true,
		speaker_id: null,
		start_ms: null,
		end_ms: null,
		...overrides,
	};
}

function makeTranslation(overrides: Partial<TranslationMessage> = {}): TranslationMessage {
	return {
		type: 'translation',
		text: 'Hello world',
		source_lang: 'zh',
		target_lang: 'en',
		transcript_id: 1,
		context_used: 0,
		is_draft: false,
		...overrides,
	};
}

function makeChunk(overrides: Partial<TranslationChunkMessage> = {}): TranslationChunkMessage {
	return {
		type: 'translation_chunk',
		transcript_id: 1,
		delta: 'Hello',
		source_lang: 'zh',
		target_lang: 'en',
		...overrides,
	};
}

describe('Caption Store — Translation Lifecycle', () => {
	let store: typeof import('$lib/stores/caption.svelte').captionStore;

	beforeEach(async () => {
		const mod = await import('$lib/stores/caption.svelte');
		store = mod.captionStore;
		store.clear();
	});

	it('ingestTranslation with is_draft=true sets translationState to draft', () => {
		store.ingestSegment(makeSegment({ segment_id: 1, is_draft: true }));
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough draft' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('rough draft');
		expect(caption.translationState).toBe('draft');
	});

	it('ingestTranslation with is_draft=false sets translationState to complete', () => {
		store.ingestSegment(makeSegment({ segment_id: 1, is_draft: false }));
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final translation' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('final translation');
		expect(caption.translationState).toBe('complete');
	});

	it('last-writer-wins: final blocks subsequent draft', () => {
		store.ingestSegment(makeSegment({ segment_id: 1 }));
		// Final arrives first
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final' }));
		// Late draft arrives — should be ignored
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'late draft' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('final');
		expect(caption.translationState).toBe('complete');
	});

	it('last-writer-wins: complete blocks all updates', () => {
		store.ingestSegment(makeSegment({ segment_id: 1 }));
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'done' }));

		// Another final — should be ignored
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'duplicate' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('done');
	});

	it('draft can be overwritten by final', () => {
		store.ingestSegment(makeSegment({ segment_id: 1 }));
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough' }));
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'polished' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('polished');
		expect(caption.translationState).toBe('complete');
	});

	it('ingestTranslationChunk clears draft text on first streaming chunk', () => {
		store.ingestSegment(makeSegment({ segment_id: 1 }));
		// Draft translation arrives
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough draft' }));
		expect(store.captions[0].translationState).toBe('draft');

		// First streaming chunk — should CLEAR draft text, then append
		store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hello' }));

		const caption = store.captions[0];
		// Should be 'Hello', NOT 'rough draftHello'
		expect(caption.translation).toBe('Hello');
		expect(caption.translationState).toBe('streaming');
	});

	it('ingestTranslationChunk ignores after complete', () => {
		store.ingestSegment(makeSegment({ segment_id: 1 }));
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final' }));

		// Late chunk — should be ignored
		store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: ' extra' }));

		expect(store.captions[0].translation).toBe('final');
	});

	it('ingestSegment draft→final preserves existing draft translation', () => {
		// Draft segment arrives with draft translation
		store.ingestSegment(makeSegment({ segment_id: 1, is_draft: true }));
		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough' }));

		// Final segment replaces draft — should preserve translation
		store.ingestSegment(makeSegment({ segment_id: 1, is_draft: false, is_final: false }));

		const caption = store.captions[0];
		expect(caption.isDraft).toBe(false);
		expect(caption.translation).toBe('rough');  // preserved!
	});

	it('new segment starts with translationState pending', () => {
		store.ingestSegment(makeSegment({ segment_id: 1 }));
		expect(store.captions[0].translationState).toBe('pending');
	});

	it('streaming transitions: pending → streaming → complete', () => {
		store.ingestSegment(makeSegment({ segment_id: 1 }));
		expect(store.captions[0].translationState).toBe('pending');

		store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hel' }));
		expect(store.captions[0].translationState).toBe('streaming');

		store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'Hello' }));
		expect(store.captions[0].translationState).toBe('complete');
	});

	// Regression coverage for the in-place mutation rewrite (#10):
	// streaming chunks must mutate the same caption object, not allocate new ones.
	it('ingestTranslationChunk mutates the same caption object (in-place)', () => {
		store.ingestSegment(makeSegment({ segment_id: 1 }));
		const before = store.captions[0];
		store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hel' }));
		const after = store.captions[0];
		// Object identity must be preserved — derivations that hold a
		// reference to the old object should still see the mutation.
		expect(after).toBe(before);
		expect(after.translation).toBe('Hel');
	});
});

describe('Caption Store — localStorage Persistence', () => {
	let store: typeof import('$lib/stores/caption.svelte').captionStore;
	const storage = new Map<string, string>();

	beforeEach(async () => {
		storage.clear();
		const mockStorage = {
			getItem: (key: string) => storage.get(key) ?? null,
			setItem: (key: string, value: string) => { storage.set(key, value); },
			removeItem: (key: string) => { storage.delete(key); },
			clear: () => { storage.clear(); },
			get length() { return storage.size; },
			key: (i: number) => [...storage.keys()][i] ?? null,
		};
		Object.defineProperty(globalThis, 'localStorage', { value: mockStorage, writable: true, configurable: true });

		const mod = await import('$lib/stores/caption.svelte');
		store = mod.captionStore;
		store.clear();
	});

	it('persists targetLanguage to localStorage on change', () => {
		store.targetLanguage = 'ja';

		const saved = JSON.parse(storage.get(STORAGE_KEY) || '{}');
		expect(saved.targetLanguage).toBe('ja');
	});

	it('persists sourceLanguage to localStorage on change', () => {
		store.sourceLanguage = 'en';

		const saved = JSON.parse(storage.get(STORAGE_KEY) || '{}');
		expect(saved.sourceLanguage).toBe('en');
	});

	it('persists displayMode to localStorage on change', () => {
		store.displayMode = 'transcript';

		const saved = JSON.parse(storage.get(STORAGE_KEY) || '{}');
		expect(saved.displayMode).toBe('transcript');
	});

	it('persists interpreterLangA and interpreterLangB', () => {
		store.interpreterLangA = 'ja';
		store.interpreterLangB = 'en';

		const saved = JSON.parse(storage.get(STORAGE_KEY) || '{}');
		expect(saved.interpreterLangA).toBe('ja');
		expect(saved.interpreterLangB).toBe('en');
	});

	it('restores settings from localStorage on load', () => {
		storage.set(STORAGE_KEY, JSON.stringify({
			sourceLanguage: 'en',
			targetLanguage: 'ja',
			displayMode: 'transcript',
			interpreterLangA: 'ja',
			interpreterLangB: 'en',
		}));

		store.restoreConfig();

		expect(store.sourceLanguage).toBe('en');
		expect(store.targetLanguage).toBe('ja');
		expect(store.displayMode).toBe('transcript');
		expect(store.interpreterLangA).toBe('ja');
		expect(store.interpreterLangB).toBe('en');
	});

	it('gracefully handles missing localStorage', () => {
		store.restoreConfig();
		expect(store.targetLanguage).toBeDefined();
	});

	it('gracefully handles corrupted localStorage', () => {
		storage.set(STORAGE_KEY, 'not-json{{{');
		store.restoreConfig();
		expect(store.targetLanguage).toBeDefined();
	});

	it('does NOT persist transient state (captions, connection)', () => {
		store.targetLanguage = 'ja';
		store.connectionState = 'connected';

		const saved = JSON.parse(storage.get(STORAGE_KEY) || '{}');
		expect(saved.connectionState).toBeUndefined();
		expect(saved.captions).toBeUndefined();
		expect(saved.isCapturing).toBeUndefined();
	});
});
