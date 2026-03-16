<script lang="ts">
  import { loopbackStore, type CaptionEntry } from '$lib/stores/loopback.svelte';

  let endRef: HTMLElement | undefined;

  const PARAGRAPH_GAP_MS = 10000;

  interface Paragraph {
    id: number;
    captions: CaptionEntry[];
    speakerId: string | null;
    timestamp: number;  // first caption's timestamp
  }

  const paragraphs = $derived.by(() => {
    const result: Paragraph[] = [];
    let current: Paragraph | null = null;

    for (const cap of loopbackStore.captions) {
      if (current === null) {
        current = { id: cap.id, captions: [cap], speakerId: cap.speakerId, timestamp: cap.timestamp };
        result.push(current);
        continue;
      }
      const lastTs = current.captions[current.captions.length - 1].timestamp;
      const shouldBreak = cap.speakerId !== current.speakerId
        || (cap.timestamp - lastTs > PARAGRAPH_GAP_MS);

      if (shouldBreak) {
        current = { id: cap.id, captions: [cap], speakerId: cap.speakerId, timestamp: cap.timestamp };
        result.push(current);
      } else {
        current.captions.push(cap);
      }
    }
    return result;
  });

  let prevLength = 0;
  $effect(() => {
    const len = loopbackStore.captions.length;
    if (len > prevLength) {
      prevLength = len;
      endRef?.scrollIntoView({ behavior: 'smooth' });
    }
  });

  function formatTime(ts: number): string {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, { hour12: false });
  }

  function paragraphTranslation(captions: CaptionEntry[]): string {
    return captions.map(c => c.translation).filter(Boolean).join(' ');
  }

  /** CJK languages don't use spaces between words. */
  function isCjk(lang: string): boolean {
    return ['zh', 'ja', 'ko'].includes(lang);
  }
</script>

<div class="transcript-view" role="log" aria-live="polite" aria-label="Transcript">
  {#each paragraphs as para (para.id)}
    <div
      class="transcript-entry"
      style="border-left-color: {loopbackStore.getSpeakerColor(para.speakerId)}"
    >
      <div class="entry-header">
        {#if para.speakerId}
          <span class="speaker" style="color: {loopbackStore.getSpeakerColor(para.speakerId)}">
            {para.speakerId}
          </span>
        {/if}
        <span class="timestamp">{formatTime(para.timestamp)}</span>
      </div>
      <div class="original">
        {#each para.captions as cap, i}
          {#if i > 0 && cap.stableText && !isCjk(cap.language)}{' '}{/if}
          <span class:is-draft={cap.isDraft}>{cap.stableText}</span>
          {#if cap.unstableText}<span class="unstable">{#if !isCjk(cap.language)}{' '}{/if}{cap.unstableText}</span>{/if}
        {/each}
      </div>
      {#if paragraphTranslation(para.captions)}
        <div class="translation">{paragraphTranslation(para.captions)}</div>
      {/if}
    </div>
  {/each}
  {#if loopbackStore.interimText}
    <div class="transcript-entry interim">
      <div class="original">{loopbackStore.interimText}</div>
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
