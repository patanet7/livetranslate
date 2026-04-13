<script lang="ts">
  import { captionStore, type UnifiedCaption as CaptionEntry } from '$lib/stores/caption.svelte';
  import { paragraphTranslation, translationPhase } from './paragraph-helpers';
  import TranslationText from './TranslationText.svelte';

  let endRef: HTMLElement | undefined;

  const PARAGRAPH_GAP_MS = 10000;

  interface Paragraph {
    id: string;
    captions: CaptionEntry[];
    speaker: string | null;
    timestamp: number;  // first caption's timestamp
  }

  const paragraphs = $derived.by(() => {
    const result: Paragraph[] = [];
    let current: Paragraph | null = null;

    for (const cap of captionStore.captions) {
      if (current === null) {
        current = { id: cap.id, captions: [cap], speaker: cap.speaker, timestamp: cap.timestamp };
        result.push(current);
        continue;
      }
      const lastTs = current.captions[current.captions.length - 1].timestamp;
      const shouldBreak = cap.speaker !== current.speaker
        || (cap.timestamp - lastTs > PARAGRAPH_GAP_MS);

      if (shouldBreak) {
        current = { id: cap.id, captions: [cap], speaker: cap.speaker, timestamp: cap.timestamp };
        result.push(current);
      } else {
        current.captions.push(cap);
      }
    }
    return result;
  });

  let prevLength = 0;
  $effect(() => {
    const len = captionStore.captions.length;
    if (len > prevLength) {
      prevLength = len;
      endRef?.scrollIntoView({ behavior: 'smooth' });
    }
  });

  function formatTime(ts: number): string {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, { hour12: false });
  }

  /** CJK languages don't use spaces between words. */
  function isCjk(lang: string): boolean {
    return ['zh', 'ja', 'ko'].includes(lang);
  }
</script>

<div class="transcript-view" data-testid="transcript-view" role="log" aria-live="polite" aria-label="Transcript">
  {#each paragraphs as para (para.id)}
    {@const trans = paragraphTranslation(para.captions)}
    {@const phase = translationPhase(para.captions)}
    <div
      class="transcript-entry"
      data-testid="transcript-entry"
      data-entry-id={para.id}
      data-translation-state={phase}
      style="border-left-color: {captionStore.getSpeakerColor(para.speaker)}"
    >
      <div class="entry-header">
        {#if para.speaker}
          <span class="speaker" data-testid="speaker" style="color: {captionStore.getSpeakerColor(para.speaker)}">
            {para.speaker}
          </span>
        {/if}
        <span class="timestamp" data-testid="entry-timestamp">{formatTime(para.timestamp)}</span>
      </div>
      <div class="original" data-testid="original-text">
        {#each para.captions as cap, i}
          {#if i > 0 && cap.stableText && !isCjk(cap.language)}{' '}{/if}
          <span class:is-draft={cap.isDraft} data-testid="caption-text" data-segment-id={cap.id}>{cap.stableText}</span>
          {#if cap.unstableText}<span class="unstable">{#if !isCjk(cap.language)}{' '}{/if}{cap.unstableText}</span>{/if}
        {/each}
      </div>
      <div class="translation" data-testid="translation-text">
        <TranslationText text={trans} {phase} />
      </div>
    </div>
  {/each}
  {#if captionStore.interimText}
    <div class="transcript-entry interim">
      <div class="original">{captionStore.interimText}</div>
    </div>
  {/if}
  <div bind:this={endRef}></div>
</div>

<style>
  .transcript-view {
    padding: 16px;
    overflow-y: auto;
    height: 100%;
    min-height: 400px;
  }
  .transcript-entry {
    padding: 12px;
    margin-bottom: 12px;
    border-left: 3px solid;
    border-radius: 4px;
    background: var(--bg-entry, rgba(255, 255, 255, 0.03));
    line-height: 1.6;
  }
  .transcript-entry.interim {
    opacity: 0.5;
    font-style: italic;
  }
  .entry-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
  }
  .speaker {
    font-weight: 600;
    font-size: 13px;
  }
  .timestamp {
    color: var(--color-timestamp, #666);
    font-size: 11px;
  }
  .original {
    color: var(--color-original, #ffd700);
    margin-bottom: 4px;
  }
  .translation {
    color: var(--color-translation, #90ee90);
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
