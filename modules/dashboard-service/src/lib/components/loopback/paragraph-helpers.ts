/**
 * Shared helpers for paragraph-level translation state.
 * Used by SplitView, TranscriptView, and SubtitleView.
 */
import type { TranslationState } from '$lib/stores/loopback.svelte';

interface CaptionLike {
  translation: string | null;
  isDraft: boolean;
  translationState: TranslationState;
}

/** Combine translations from all captions in a paragraph. */
export function paragraphTranslation(captions: CaptionLike[]): string {
	return captions
		.map((c) => c.translation)
		.filter(Boolean)
		.join(' ');
}

/** Does this paragraph have any captions still waiting for translation? */
export function hasPendingTranslation(captions: CaptionLike[]): boolean {
	return captions.some((c) => !c.isDraft && c.translationState !== 'complete');
}

/**
 * Derive the visual phase of a paragraph's translation.
 * Used by TranslationText to determine opacity and indicator state.
 *
 * The final translation path accumulates stable_text across segments and
 * only fires a TranslationMessage for the segment that triggers the flush.
 * Earlier segments in the buffer keep their draft translations. So a
 * paragraph is "complete" once ANY caption has a final translation — the
 * other captions have draft text that's visually sufficient.
 */
export function translationPhase(captions: CaptionLike[]): TranslationState {
	const anyComplete = captions.some((c) => c.translationState === 'complete');
	const anyStreaming = captions.some((c) => c.translationState === 'streaming');
	const anyDraft = captions.some((c) => c.translationState === 'draft');

	if (anyComplete) return 'complete';
	if (anyStreaming) return 'streaming';
	if (anyDraft) return 'draft';
	return 'pending';
}
