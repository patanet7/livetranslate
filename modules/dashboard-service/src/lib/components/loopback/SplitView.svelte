<script lang="ts">
  import { loopbackStore, type CaptionEntry } from '$lib/stores/loopback.svelte';
  import { paragraphTranslation, translationPhase } from './paragraph-helpers';
  import TranslationText from './TranslationText.svelte';

  // Allow undefined before mount
  let captionsEndOriginal: HTMLElement | undefined;
  let captionsEndTranslation: HTMLElement | undefined;

  /** Gap threshold to start a new paragraph. Must exceed stride time (~4.5-6s)
   *  plus inference latency (~1-2s). 10s means: silence > 10s = new paragraph. */
  const PARAGRAPH_GAP_MS = 10000;

  interface Paragraph {
    id: number;             // First caption's id (stable key)
    captions: CaptionEntry[];
    speakerId: string | null;
  }

  /**
   * Group captions into paragraphs. A new paragraph starts when:
   *  - The time gap between consecutive segments exceeds PARAGRAPH_GAP_MS
   *  - The speaker changes
   */
  const paragraphs = $derived.by(() => {
    const result: Paragraph[] = [];
    let current: Paragraph | null = null;

    for (const cap of loopbackStore.captions) {
      if (current === null) {
        current = { id: cap.id, captions: [cap], speakerId: cap.speakerId };
        result.push(current);
        continue;
      }
      const lastTs = current.captions[current.captions.length - 1].timestamp;
      const shouldBreak = cap.speakerId !== current.speakerId
        || (cap.timestamp - lastTs > PARAGRAPH_GAP_MS);

      if (shouldBreak) {
        current = { id: cap.id, captions: [cap], speakerId: cap.speakerId };
        result.push(current);
      } else {
        current.captions.push(cap);
      }
    }
    return result;
  });

  // Scroll on new paragraphs or when the last paragraph grows
  let prevCaptionCount = 0;
  $effect(() => {
    const count = loopbackStore.captions.length;
    if (count > prevCaptionCount) {
      prevCaptionCount = count;
      captionsEndOriginal?.scrollIntoView({ behavior: 'smooth' });
      captionsEndTranslation?.scrollIntoView({ behavior: 'smooth' });
    }
  });

  /** CJK languages don't use spaces between words. */
  function isCjk(lang: string): boolean {
    return ['zh', 'ja', 'ko'].includes(lang);
  }
</script>

<div class="split-view">
  <!-- Original language panel -->
  <div class="panel panel-original">
    <div class="panel-header">
      Original ({loopbackStore.detectedLanguage ?? loopbackStore.sourceLanguage ?? 'detecting...'})
    </div>
    <div class="panel-content" role="log" aria-live="polite" aria-label="Original captions">
      {#each paragraphs as para (para.id)}
        <div
          class="paragraph"
          style="border-left-color: {loopbackStore.getSpeakerColor(para.speakerId)}"
        >
          {#if para.speakerId}
            <span class="speaker" style="color: {loopbackStore.getSpeakerColor(para.speakerId)}">
              {para.speakerId}:
            </span>
          {/if}
          <span class="para-text">
            {#each para.captions as cap, i}
              {#if i > 0 && cap.stableText && !isCjk(cap.language)}{' '}{/if}
              <span class:is-draft={cap.isDraft}>{cap.stableText}</span>
              {#if cap.unstableText}<span class="unstable">{#if !isCjk(cap.language)}{' '}{/if}{cap.unstableText}</span>{/if}
            {/each}
          </span>
        </div>
      {/each}
      {#if loopbackStore.interimText}
        <div class="paragraph interim">
          <span class="para-text">{loopbackStore.interimText}</span>
        </div>
      {/if}
      <div bind:this={captionsEndOriginal}></div>
    </div>
  </div>

  <!-- Translation panel -->
  <div class="panel panel-translation">
    <div class="panel-header">
      Translation ({loopbackStore.targetLanguage})
    </div>
    <div class="panel-content" role="log" aria-live="polite" aria-label="Translations">
      {#each paragraphs as para (para.id)}
        {@const trans = paragraphTranslation(para.captions)}
        {@const phase = translationPhase(para.captions)}
        <div
          class="paragraph"
          style="border-left-color: {loopbackStore.getSpeakerColor(para.speakerId)}"
        >
          {#if para.speakerId}
            <span class="speaker" style="color: {loopbackStore.getSpeakerColor(para.speakerId)}">
              {para.speakerId}:
            </span>
          {/if}
          <span class="para-text">
            <TranslationText text={trans} {phase} />
          </span>
        </div>
      {/each}
      <div bind:this={captionsEndTranslation}></div>
    </div>
  </div>
</div>

<style>
  .split-view {
    display: flex;
    gap: 2px;
    height: 100%;
    min-height: 400px;
  }
  .panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-header {
    padding: 8px 16px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid var(--border, #333);
  }
  .panel-original .panel-header { color: var(--color-original, #ffd700); }
  .panel-translation .panel-header { color: var(--color-translation, #90ee90); }
  .panel-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }
  .paragraph {
    padding: 10px 12px;
    margin-bottom: 10px;
    border-left: 3px solid;
    border-radius: 4px;
    background: var(--bg-entry, rgba(255, 255, 255, 0.03));
    line-height: 1.6;
  }
  .paragraph.interim {
    opacity: 0.5;
    font-style: italic;
  }
  .speaker {
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
  }
  .para-text {
    transition: opacity 0.3s ease;
  }
  .is-draft {
    opacity: 0.85;
  }
  .unstable {
    opacity: 0.45;
    font-style: italic;
    transition: opacity 0.3s ease;
  }
</style>
