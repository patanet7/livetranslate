<script lang="ts">
  import { loopbackStore } from '$lib/stores/loopback.svelte';

  // 2d: Allow undefined before mount
  let endRef: HTMLElement | undefined;

  // 1b: Extract filtered captions to $derived
  const finalCaptions = $derived(loopbackStore.captions.filter(c => c.isFinal));

  // 1a: Only scroll on new captions, not translation patches
  let prevLength = 0;
  $effect(() => {
    const len = loopbackStore.captions.length;
    if (len > prevLength) {
      prevLength = len;
      endRef?.scrollIntoView({ behavior: 'smooth' });
    }
  });

  // 3b: Use browser locale instead of hardcoded 'en-US'
  function formatTime(ts: number): string {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, { hour12: false });
  }
</script>

<!-- 2a: ARIA live region for streaming transcript -->
<div class="transcript-view" role="log" aria-live="polite" aria-label="Transcript">
  {#each finalCaptions as caption (caption.id)}
    <div
      class="transcript-entry"
      style="border-left-color: {loopbackStore.getSpeakerColor(caption.speakerId)}"
    >
      <div class="entry-header">
        {#if caption.speakerId}
          <span class="speaker" style="color: {loopbackStore.getSpeakerColor(caption.speakerId)}">
            {caption.speakerId}
          </span>
        {/if}
        <span class="timestamp">{formatTime(caption.timestamp)}</span>
      </div>
      <div class="original">{caption.text}</div>
      {#if caption.translation}
        <div class="translation">{caption.translation}</div>
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
  /* 2b: Use CSS custom properties for theme-able colors */
  .original {
    color: var(--color-original, #ffd700);
    margin-bottom: 4px;
  }
  .translation {
    color: var(--color-translation, #90ee90);
  }
</style>
