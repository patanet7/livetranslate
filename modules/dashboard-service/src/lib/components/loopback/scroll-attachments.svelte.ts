/**
 * Shared `{@attach}` helpers for the loopback view components.
 *
 * Replaces the bind:this + $effect-watch pattern that the Svelte autofixer
 * flagged across SplitView / WireView / TranscriptView / InterpreterView.
 *
 * Two birds with one stone:
 *  - eliminates `bind:this` (autofixer style suggestion)
 *  - eliminates manual `prevCount` state + outer $effect (cleaner reactivity)
 *
 * Pattern source: svelte.dev `{@attach}` docs, "Controlling when attachments
 * re-run" — the getter form prevents the outer attachment from re-creating
 * on every count change.
 */
import type { Attachment } from 'svelte/attachments';


/**
 * Scrolls the attached element into view (smooth) whenever ``getCount`` returns
 * a higher value than the previous read. Per-element ``prev`` counter is
 * scoped to the attachment, so two elements bound to the same getter each
 * decide independently when to scroll (correct for facing-pages spreads
 * where both panels should scroll on every new caption).
 *
 * Example:
 *   <div {@attach scrollIntoViewOnGrow(() => captionStore.captions.length)}></div>
 */
export function scrollIntoViewOnGrow(getCount: () => number): Attachment {
  return (node) => {
    let prev = 0;
    $effect(() => {
      const c = getCount();
      if (c > prev) {
        prev = c;
        // Use Element.scrollIntoView when available — falls through to noop
        // for non-Element targets (Attachment generic defaults to Element).
        (node as Element).scrollIntoView?.({ behavior: 'smooth' });
      }
    });
  };
}
