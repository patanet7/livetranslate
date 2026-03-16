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
 */
export function translationPhase(
	captions: CaptionEntry[]
): 'waiting' | 'draft' | 'streaming' | 'complete' {
	const hasTranslation = captions.some((c) => c.translation !== null);
	const allComplete = captions.every((c) => c.translationComplete || c.isDraft);
	const hasDraft = captions.some((c) => c.translationIsDraft);
	const anyFinalComplete = captions.some((c) => c.translationComplete);

	if (anyFinalComplete && !hasPendingTranslation(captions)) return 'complete';
	if (hasDraft && hasTranslation) return 'draft';
	if (hasTranslation && !allComplete) return 'streaming';
	if (!hasTranslation) return 'waiting';
	return 'complete';
}
