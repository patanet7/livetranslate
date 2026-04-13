<script lang="ts">
  import { captionStore, type UnifiedCaption as CaptionEntry } from '$lib/stores/caption.svelte';
  import { paragraphTranslation, translationPhase } from './paragraph-helpers';
  import TranslationText from './TranslationText.svelte';

  let captionsEndA: HTMLElement | undefined;
  let captionsEndB: HTMLElement | undefined;

  /** Gap threshold to start a new paragraph (matches SplitView). */
  const PARAGRAPH_GAP_MS = 10000;

  /** Language display names for panel headers. */
  const LANG_NAMES: Record<string, string> = {
    zh: 'Chinese', en: 'English', ja: 'Japanese',
    es: 'Spanish', fr: 'French', ko: 'Korean',
  };

  function langLabel(code: string): string {
    return LANG_NAMES[code] ?? code;
  }

  interface Paragraph {
    id: string;
    captions: CaptionEntry[];
    speaker: string | null;
    language: string;
  }

  /**
   * Group captions into paragraphs. A new paragraph starts when:
   *  - The time gap between consecutive segments exceeds PARAGRAPH_GAP_MS
   *  - The speaker changes
   *  - The detected language changes (interpreter mode: language switch = new block)
   */
  const paragraphs = $derived.by(() => {
    const result: Paragraph[] = [];
    let current: Paragraph | null = null;

    for (const cap of captionStore.captions) {
      if (current === null) {
        current = { id: cap.id, captions: [cap], speaker: cap.speaker, language: cap.language };
        result.push(current);
        continue;
      }
      const lastTs = current.captions[current.captions.length - 1].timestamp;
      const shouldBreak = cap.speaker !== current.speaker
        || (cap.timestamp - lastTs > PARAGRAPH_GAP_MS)
        || cap.language !== current.language;

      if (shouldBreak) {
        current = { id: cap.id, captions: [cap], speaker: cap.speaker, language: cap.language };
        result.push(current);
      } else {
        current.captions.push(cap);
      }
    }
    return result;
  });

  // Scroll on new captions
  let prevCaptionCount = 0;
  $effect(() => {
    const count = captionStore.captions.length;
    if (count > prevCaptionCount) {
      prevCaptionCount = count;
      captionsEndA?.scrollIntoView({ behavior: 'smooth' });
      captionsEndB?.scrollIntoView({ behavior: 'smooth' });
    }
  });

  /** CJK languages don't use spaces between words. */
  function isCjk(lang: string): boolean {
    return ['zh', 'ja', 'ko'].includes(lang);
  }

  /**
   * Determine what a panel shows for a given paragraph.
   * - If paragraph's language matches this panel's language → show transcription (native)
   * - If paragraph's language is the OTHER language → show translation (which is IN this panel's language)
   */
  function isNative(para: Paragraph, panelLang: string): boolean {
    return para.language === panelLang;
  }
</script>

<div class="interpreter-view" data-testid="interpreter-view">
  <!-- Language A panel: shows text that IS in Language A -->
  <div class="panel panel-a" data-testid="panel-lang-a">
    <div class="panel-header panel-header-a">
      {langLabel(captionStore.interpreterLangA)} ({captionStore.interpreterLangA})
    </div>
    <div class="panel-content" role="log" aria-live="polite" aria-label="{langLabel(captionStore.interpreterLangA)} content">
      {#each paragraphs as para (para.id)}
        {@const native = isNative(para, captionStore.interpreterLangA)}
        {@const trans = paragraphTranslation(para.captions)}
        {@const phase = translationPhase(para.captions)}
        <div
          class="paragraph"
          data-testid="paragraph"
          data-para-id={para.id}
          data-translation-state={native ? 'native' : phase}
          class:is-translation={!native}
          style="border-left-color: {captionStore.getSpeakerColor(para.speaker)}"
        >
          {#if para.speaker}
            <span class="speaker" style="color: {captionStore.getSpeakerColor(para.speaker)}">
              {para.speaker}:
            </span>
          {/if}
          {#if native}
            <!-- This panel's language was spoken — show transcription -->
            <span class="para-text">
              {#each para.captions as cap, i}
                {#if i > 0 && cap.stableText && !isCjk(cap.language)}{' '}{/if}
                <span class:is-draft={cap.isDraft}>{cap.stableText}</span>
                {#if cap.unstableText}<span class="unstable">{#if !isCjk(cap.language)}{' '}{/if}{cap.unstableText}</span>{/if}
              {/each}
            </span>
          {:else}
            <!-- Other language was spoken — show translation (which is in THIS panel's language) -->
            <span class="para-text">
              <TranslationText text={trans} {phase} />
            </span>
          {/if}
        </div>
      {/each}
      <div bind:this={captionsEndA}></div>
    </div>
  </div>

  <!-- Language B panel: shows text that IS in Language B -->
  <div class="panel panel-b" data-testid="panel-lang-b">
    <div class="panel-header panel-header-b">
      {langLabel(captionStore.interpreterLangB)} ({captionStore.interpreterLangB})
    </div>
    <div class="panel-content" role="log" aria-live="polite" aria-label="{langLabel(captionStore.interpreterLangB)} content">
      {#each paragraphs as para (para.id)}
        {@const native = isNative(para, captionStore.interpreterLangB)}
        {@const trans = paragraphTranslation(para.captions)}
        {@const phase = translationPhase(para.captions)}
        <div
          class="paragraph"
          data-testid="paragraph"
          data-para-id={para.id}
          data-translation-state={native ? 'native' : phase}
          class:is-translation={!native}
          style="border-left-color: {captionStore.getSpeakerColor(para.speaker)}"
        >
          {#if para.speaker}
            <span class="speaker" style="color: {captionStore.getSpeakerColor(para.speaker)}">
              {para.speaker}:
            </span>
          {/if}
          {#if native}
            <!-- This panel's language was spoken — show transcription -->
            <span class="para-text">
              {#each para.captions as cap, i}
                {#if i > 0 && cap.stableText && !isCjk(cap.language)}{' '}{/if}
                <span class:is-draft={cap.isDraft}>{cap.stableText}</span>
                {#if cap.unstableText}<span class="unstable">{#if !isCjk(cap.language)}{' '}{/if}{cap.unstableText}</span>{/if}
              {/each}
            </span>
          {:else}
            <!-- Other language was spoken — show translation (which is in THIS panel's language) -->
            <span class="para-text">
              <TranslationText text={trans} {phase} />
            </span>
          {/if}
        </div>
      {/each}
      <div bind:this={captionsEndB}></div>
    </div>
  </div>
</div>

<style>
  .interpreter-view {
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
  .panel-header-a { color: var(--color-original, #ffd700); }
  .panel-header-b { color: var(--color-translation, #90ee90); }
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
  .paragraph.is-translation {
    background: var(--bg-entry-translation, rgba(255, 255, 255, 0.015));
    opacity: 0.9;
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
