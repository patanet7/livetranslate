/**
 * Tests for unified caption store.
 *
 * Covers: ingestSegment (lb_ prefix, upsert, draft guard),
 * ingestCaptionEvent (ff_ prefix, dedup on reconnect),
 * ingestTranslation / ingestTranslationChunk lifecycle.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { SegmentMessage, TranslationMessage, TranslationChunkMessage } from '$lib/types/ws-messages';
import type { CaptionEvent, CaptionEventCaption } from '$lib/stores/caption.svelte';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

let _captionIdCounter = 0;
function makeFirefliesCaption(overrides: Partial<CaptionEventCaption> = {}): CaptionEventCaption {
  const id = `ff-cap-${++_captionIdCounter}`;
  return {
    id,
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
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('captionStore — ingestSegment', () => {
  let store: typeof import('$lib/stores/caption.svelte').captionStore;

  beforeEach(async () => {
    const mod = await import('$lib/stores/caption.svelte');
    store = mod.captionStore;
    store.clear();
  });

  it('adds a new caption with lb_ prefix', () => {
    store.ingestSegment(makeSegment({ segment_id: 42 }));
    expect(store.captions).toHaveLength(1);
    expect(store.captions[0].id).toBe('lb_42');
  });

  it('caption starts with translationState pending', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    expect(store.captions[0].translationState).toBe('pending');
  });

  it('updates existing caption in-place', () => {
    store.ingestSegment(makeSegment({ segment_id: 1, stable_text: 'draft', text: 'draft', is_draft: true }));
    store.ingestSegment(makeSegment({ segment_id: 1, stable_text: 'final', text: 'final', is_draft: false }));
    expect(store.captions).toHaveLength(1);
    expect(store.captions[0].stableText).toBe('final');
  });

  it('preserves existing translation when segment is updated', () => {
    store.ingestSegment(makeSegment({ segment_id: 1, is_draft: true }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough' }));
    store.ingestSegment(makeSegment({ segment_id: 1, is_draft: false }));
    expect(store.captions[0].translation).toBe('rough');
  });

  it('does NOT overwrite a final segment with a later draft', () => {
    store.ingestSegment(makeSegment({ segment_id: 1, stable_text: 'final text', text: 'final text', is_draft: false }));
    store.ingestSegment(makeSegment({ segment_id: 1, stable_text: 'late draft', text: 'late draft', is_draft: true }));
    expect(store.captions[0].stableText).toBe('final text');
    expect(store.captions[0].isDraft).toBe(false);
  });

  it('sets detectedLanguage on each ingest', () => {
    store.ingestSegment(makeSegment({ language: 'ja' }));
    expect(store.detectedLanguage).toBe('ja');
  });

  it('clears interim on is_final=true segment', () => {
    store.ingestInterim('Hello wor', 0.8);
    expect(store.interimText).toBe('Hello wor');
    store.ingestSegment(makeSegment({ is_final: true }));
    expect(store.interimText).toBe('');
  });

  it('increments segmentsReceived counter', () => {
    const before = store.segmentsReceived;
    store.ingestSegment(makeSegment({ segment_id: 99 }));
    expect(store.segmentsReceived).toBe(before + 1);
  });
});

describe('captionStore — ingestTranslation', () => {
  let store: typeof import('$lib/stores/caption.svelte').captionStore;

  beforeEach(async () => {
    const mod = await import('$lib/stores/caption.svelte');
    store = mod.captionStore;
    store.clear();
  });

  it('sets translation and state to draft on is_draft=true', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough' }));
    expect(store.captions[0].translation).toBe('rough');
    expect(store.captions[0].translationState).toBe('draft');
  });

  it('sets translation and state to complete on is_draft=false', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final' }));
    expect(store.captions[0].translation).toBe('final');
    expect(store.captions[0].translationState).toBe('complete');
  });

  it('final blocks subsequent draft (last-writer-wins)', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final' }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'late draft' }));
    expect(store.captions[0].translation).toBe('final');
    expect(store.captions[0].translationState).toBe('complete');
  });

  it('complete blocks all subsequent updates', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'done' }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'duplicate' }));
    expect(store.captions[0].translation).toBe('done');
  });

  it('draft can be overwritten by final', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough' }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'polished' }));
    expect(store.captions[0].translation).toBe('polished');
    expect(store.captions[0].translationState).toBe('complete');
  });

  it('ignores translation for unknown segment id', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslation(makeTranslation({ transcript_id: 999, text: 'orphan' }));
    expect(store.captions[0].translation).toBeNull();
  });
});

describe('captionStore — ingestTranslationChunk', () => {
  let store: typeof import('$lib/stores/caption.svelte').captionStore;

  beforeEach(async () => {
    const mod = await import('$lib/stores/caption.svelte');
    store = mod.captionStore;
    store.clear();
  });

  it('transitions state to streaming and appends delta', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hel' }));
    expect(store.captions[0].translationState).toBe('streaming');
    expect(store.captions[0].translation).toBe('Hel');
  });

  it('accumulates multiple deltas', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hel' }));
    store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: 'lo' }));
    expect(store.captions[0].translation).toBe('Hello');
  });

  it('clears draft text on first streaming chunk', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: true, text: 'rough draft' }));
    store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hello' }));
    expect(store.captions[0].translation).toBe('Hello');
  });

  it('ignores chunk after complete', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'final' }));
    store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: ' extra' }));
    expect(store.captions[0].translation).toBe('final');
  });

  it('streaming → complete transition', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestTranslationChunk(makeChunk({ transcript_id: 1, delta: 'Hel' }));
    expect(store.captions[0].translationState).toBe('streaming');
    store.ingestTranslation(makeTranslation({ transcript_id: 1, is_draft: false, text: 'Hello world' }));
    expect(store.captions[0].translationState).toBe('complete');
    expect(store.captions[0].translation).toBe('Hello world');
  });
});

describe('captionStore — ingestCaptionEvent', () => {
  let store: typeof import('$lib/stores/caption.svelte').captionStore;

  beforeEach(async () => {
    const mod = await import('$lib/stores/caption.svelte');
    store = mod.captionStore;
    store.clear();
    _captionIdCounter = 0;
  });

  it('adds caption with ff_ prefix on caption_added', () => {
    const cap = makeFirefliesCaption({ id: 'abc123' });
    const event: CaptionEvent = { event: 'caption_added', caption: cap };
    store.ingestCaptionEvent(event);
    expect(store.captions).toHaveLength(1);
    expect(store.captions[0].id).toBe('ff_abc123');
  });

  it('sets translation from translated_text when different from original_text', () => {
    const cap = makeFirefliesCaption({ original_text: 'Hello', translated_text: 'Hola' });
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    expect(store.captions[0].translation).toBe('Hola');
    expect(store.captions[0].translationState).toBe('complete');
  });

  it('sets translation to null when translated_text equals original_text', () => {
    const cap = makeFirefliesCaption({ original_text: 'Hello', translated_text: 'Hello' });
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    expect(store.captions[0].translation).toBeNull();
  });

  it('skips duplicate caption_added on reconnect', () => {
    const cap = makeFirefliesCaption({ id: 'dup1' });
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    expect(store.captions).toHaveLength(1);
  });

  it('allows caption_updated even if already seen', () => {
    const cap = makeFirefliesCaption({ id: 'upd1', original_text: 'v1', translated_text: 'v1t' });
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    const updated = { ...cap, original_text: 'v2', translated_text: 'v2t' };
    store.ingestCaptionEvent({ event: 'caption_updated', caption: updated });
    expect(store.captions).toHaveLength(1);
    expect(store.captions[0].text).toBe('v2');
    expect(store.captions[0].translation).toBe('v2t');
  });

  it('clears captions on session_cleared', () => {
    store.ingestCaptionEvent({ event: 'caption_added', caption: makeFirefliesCaption() });
    store.ingestCaptionEvent({ event: 'caption_added', caption: makeFirefliesCaption() });
    expect(store.captions).toHaveLength(2);
    store.ingestCaptionEvent({ event: 'session_cleared' });
    expect(store.captions).toHaveLength(0);
  });

  it('allows new caption_added after session_cleared resets dedup', () => {
    const cap = makeFirefliesCaption({ id: 'reuse1' });
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    store.ingestCaptionEvent({ event: 'session_cleared' });
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    expect(store.captions).toHaveLength(1);
  });

  it('caption_expired does not remove caption (kept for history)', () => {
    const cap = makeFirefliesCaption({ id: 'exp1' });
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    store.ingestCaptionEvent({ event: 'caption_expired', caption_id: 'ff_exp1' });
    expect(store.captions).toHaveLength(1);
  });

  it('uses speaker_color from event if provided', () => {
    const cap = makeFirefliesCaption({ speaker_color: '#FF0000' });
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    expect(store.captions[0].speakerColor).toBe('#FF0000');
  });

  it('marks captions as isFinal and not isDraft', () => {
    const cap = makeFirefliesCaption();
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    expect(store.captions[0].isFinal).toBe(true);
    expect(store.captions[0].isDraft).toBe(false);
  });
});

describe('captionStore — clear and counters', () => {
  let store: typeof import('$lib/stores/caption.svelte').captionStore;

  beforeEach(async () => {
    const mod = await import('$lib/stores/caption.svelte');
    store = mod.captionStore;
    store.clear();
    _captionIdCounter = 0;
  });

  it('clear() resets captions, counters, and interim', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    store.ingestInterim('typing...', 0.5);
    store.clear();
    expect(store.captions).toHaveLength(0);
    expect(store.interimText).toBe('');
    expect(store.segmentsReceived).toBe(0);
  });

  it('ingestInterim updates interimText and interimConfidence', () => {
    store.ingestInterim('Hello wor', 0.75);
    expect(store.interimText).toBe('Hello wor');
    expect(store.interimConfidence).toBe(0.75);
  });

  it('lb_ and ff_ captions coexist without id collision', () => {
    store.ingestSegment(makeSegment({ segment_id: 1 }));
    const cap = makeFirefliesCaption({ id: '1' }); // ff_1 vs lb_1
    store.ingestCaptionEvent({ event: 'caption_added', caption: cap });
    expect(store.captions).toHaveLength(2);
    expect(store.captions.map(c => c.id).sort()).toEqual(['ff_1', 'lb_1']);
  });
});
