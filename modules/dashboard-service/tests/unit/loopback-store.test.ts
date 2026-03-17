/**
 * Tests for loopback store translation lifecycle.
 *
 * Covers: translationState field, last-writer-wins guards,
 * chunk-on-draft clear, segment replacement preserves translation.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { SegmentMessage, TranslationMessage, TranslationChunkMessage } from '$lib/types/ws-messages';

// Helper: minimal SegmentMessage
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

describe('Loopback Store — Translation Lifecycle', () => {
	let store: typeof import('$lib/stores/loopback.svelte').loopbackStore;

	beforeEach(async () => {
		// Dynamic import to get fresh store instance
		const mod = await import('$lib/stores/loopback.svelte');
		// createLoopbackStore is not exported — use loopbackStore singleton
		// and clear it between tests
		store = mod.loopbackStore;
		store.clear();
	});

	it('addTranslation with is_draft=true sets translationState to draft', () => {
		store.addSegment(makeSegment({ segment_id: 1, is_draft: true }));
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough draft' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('rough draft');
		expect(caption.translationState).toBe('draft');
	});

	it('addTranslation with is_draft=false sets translationState to complete', () => {
		store.addSegment(makeSegment({ segment_id: 1, is_draft: false }));
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final translation' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('final translation');
		expect(caption.translationState).toBe('complete');
	});

	it('last-writer-wins: final blocks subsequent draft', () => {
		store.addSegment(makeSegment({ segment_id: 1 }));
		// Final arrives first
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final' }));
		// Late draft arrives — should be ignored
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'late draft' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('final');
		expect(caption.translationState).toBe('complete');
	});

	it('last-writer-wins: complete blocks all updates', () => {
		store.addSegment(makeSegment({ segment_id: 1 }));
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'done' }));

		// Another final — should be ignored
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'duplicate' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('done');
	});

	it('draft can be overwritten by final', () => {
		store.addSegment(makeSegment({ segment_id: 1 }));
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough' }));
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'polished' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('polished');
		expect(caption.translationState).toBe('complete');
	});

	it('appendTranslationChunk clears draft text on first streaming chunk', () => {
		store.addSegment(makeSegment({ segment_id: 1 }));
		// Draft translation arrives
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough draft' }));
		expect(store.captions[0].translationState).toBe('draft');

		// First streaming chunk — should CLEAR draft text, then append
		store.appendTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hello' }));

		const caption = store.captions[0];
		// Should be 'Hello', NOT 'rough draftHello'
		expect(caption.translation).toBe('Hello');
		expect(caption.translationState).toBe('streaming');
	});

	it('appendTranslationChunk ignores after complete', () => {
		store.addSegment(makeSegment({ segment_id: 1 }));
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final' }));

		// Late chunk — should be ignored
		store.appendTranslationChunk(makeChunk({ transcript_id: 1, delta: ' extra' }));

		expect(store.captions[0].translation).toBe('final');
	});

	it('addSegment draft→final preserves existing draft translation', () => {
		// Draft segment arrives with draft translation
		store.addSegment(makeSegment({ segment_id: 1, is_draft: true }));
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough' }));

		// Final segment replaces draft — should preserve translation
		store.addSegment(makeSegment({ segment_id: 1, is_draft: false, is_final: false }));

		const caption = store.captions[0];
		expect(caption.isDraft).toBe(false);
		expect(caption.translation).toBe('rough');  // preserved!
	});

	it('new segment starts with translationState pending', () => {
		store.addSegment(makeSegment({ segment_id: 1 }));
		expect(store.captions[0].translationState).toBe('pending');
	});

	it('streaming transitions: pending → streaming → complete', () => {
		store.addSegment(makeSegment({ segment_id: 1 }));
		expect(store.captions[0].translationState).toBe('pending');

		store.appendTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hel' }));
		expect(store.captions[0].translationState).toBe('streaming');

		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'Hello' }));
		expect(store.captions[0].translationState).toBe('complete');
	});
});

describe('Loopback Store — localStorage Persistence', () => {
	let store: typeof import('$lib/stores/loopback.svelte').loopbackStore;
	const storage = new Map<string, string>();

	// Mock localStorage for Node environment
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

		const mod = await import('$lib/stores/loopback.svelte');
		store = mod.loopbackStore;
		store.clear();
	});

	it('persists targetLanguage to localStorage on change', () => {
		store.targetLanguage = 'ja';

		const saved = JSON.parse(storage.get('livetranslate:loopback-config') || '{}');
		expect(saved.targetLanguage).toBe('ja');
	});

	it('persists sourceLanguage to localStorage on change', () => {
		store.sourceLanguage = 'en';

		const saved = JSON.parse(storage.get('livetranslate:loopback-config') || '{}');
		expect(saved.sourceLanguage).toBe('en');
	});

	it('persists displayMode to localStorage on change', () => {
		store.displayMode = 'transcript';

		const saved = JSON.parse(storage.get('livetranslate:loopback-config') || '{}');
		expect(saved.displayMode).toBe('transcript');
	});

	it('persists interpreterLangA and interpreterLangB', () => {
		store.interpreterLangA = 'ja';
		store.interpreterLangB = 'en';

		const saved = JSON.parse(storage.get('livetranslate:loopback-config') || '{}');
		expect(saved.interpreterLangA).toBe('ja');
		expect(saved.interpreterLangB).toBe('en');
	});

	it('restores settings from localStorage on load', () => {
		// Pre-populate localStorage
		storage.set('livetranslate:loopback-config', JSON.stringify({
			sourceLanguage: 'en',
			targetLanguage: 'ja',
			displayMode: 'transcript',
			interpreterLangA: 'ja',
			interpreterLangB: 'en',
		}));

		store.restoreFromLocalStorage();

		expect(store.sourceLanguage).toBe('en');
		expect(store.targetLanguage).toBe('ja');
		expect(store.displayMode).toBe('transcript');
		expect(store.interpreterLangA).toBe('ja');
		expect(store.interpreterLangB).toBe('en');
	});

	it('gracefully handles missing localStorage', () => {
		// No localStorage entry — restoreFromLocalStorage should not throw
		store.restoreFromLocalStorage();
		expect(store.targetLanguage).toBeDefined();
	});

	it('gracefully handles corrupted localStorage', () => {
		storage.set('livetranslate:loopback-config', 'not-json{{{');
		// Should not throw
		store.restoreFromLocalStorage();
		expect(store.targetLanguage).toBeDefined();
	});

	it('does NOT persist transient state (captions, connection)', () => {
		store.targetLanguage = 'ja';
		store.connectionState = 'connected';

		const saved = JSON.parse(storage.get('livetranslate:loopback-config') || '{}');
		expect(saved.connectionState).toBeUndefined();
		expect(saved.captions).toBeUndefined();
		expect(saved.isCapturing).toBeUndefined();
	});
});
