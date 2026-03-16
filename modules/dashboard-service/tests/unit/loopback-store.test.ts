/**
 * Tests for loopback store draft→final translation lifecycle.
 *
 * Covers: translationIsDraft field, last-writer-wins guards,
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

describe('Loopback Store — Draft Translation Lifecycle', () => {
	let store: typeof import('$lib/stores/loopback.svelte').loopbackStore;

	beforeEach(async () => {
		// Dynamic import to get fresh store instance
		const mod = await import('$lib/stores/loopback.svelte');
		// createLoopbackStore is not exported — use loopbackStore singleton
		// and clear it between tests
		store = mod.loopbackStore;
		store.clear();
	});

	it('addTranslation with is_draft=true sets translationIsDraft', () => {
		store.addSegment(makeSegment({ segment_id: 1, is_draft: true }));
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough draft' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('rough draft');
		expect(caption.translationIsDraft).toBe(true);
		expect(caption.translationComplete).toBe(false);
	});

	it('addTranslation with is_draft=false sets translationComplete', () => {
		store.addSegment(makeSegment({ segment_id: 1, is_draft: false }));
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final translation' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('final translation');
		expect(caption.translationIsDraft).toBe(false);
		expect(caption.translationComplete).toBe(true);
	});

	it('last-writer-wins: final blocks subsequent draft', () => {
		store.addSegment(makeSegment({ segment_id: 1 }));
		// Final arrives first
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final' }));
		// Late draft arrives — should be ignored
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'late draft' }));

		const caption = store.captions[0];
		expect(caption.translation).toBe('final');
		expect(caption.translationComplete).toBe(true);
	});

	it('last-writer-wins: translationComplete blocks all updates', () => {
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
		expect(caption.translationComplete).toBe(true);
		expect(caption.translationIsDraft).toBe(false);
	});

	it('appendTranslationChunk clears draft text on first final chunk', () => {
		store.addSegment(makeSegment({ segment_id: 1 }));
		// Draft translation arrives
		store.addTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough draft' }));
		expect(store.captions[0].translationIsDraft).toBe(true);

		// First streaming final chunk — should CLEAR draft text, then append
		store.appendTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hello' }));

		const caption = store.captions[0];
		// Should be 'Hello', NOT 'rough draftHello'
		expect(caption.translation).toBe('Hello');
		expect(caption.translationIsDraft).toBe(false);
	});

	it('appendTranslationChunk ignores after translationComplete', () => {
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
});
