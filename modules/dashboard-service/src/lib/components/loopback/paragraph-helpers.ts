/**
 * Shared helpers for paragraph-level translation state.
 * Used by SplitView, TranscriptView, and SubtitleView.
 */
import type { CaptionEntry } from '$lib/stores/loopback.svelte';

/** Combine translations from all captions in a paragraph. */
export function paragraphTranslation(captions: CaptionEntry[]): string {
	return captions
		.map((c) => c.translation)
		.filter(Boolean)
		.join(' ');
}

/** Does this paragraph have any captions still waiting for translation? */
export function hasPendingTranslation(captions: CaptionEntry[]): boolean {
	return captions.some((c) => !c.isDraft && !c.translationComplete);
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
export function translationPhase(
	captions: CaptionEntry[]
): 'waiting' | 'draft' | 'streaming' | 'complete' {
	const hasTranslation = captions.some((c) => c.translation !== null);
	const anyFinalComplete = captions.some((c) => c.translationComplete);
	const hasDraft = captions.some((c) => c.translationIsDraft && !c.translationComplete);

	if (anyFinalComplete) return 'complete';
	if (hasDraft) return 'draft';
	if (hasTranslation) return 'streaming';
	return 'waiting';
}
