<script lang="ts">
  import { loopbackStore } from '$lib/stores/loopback.svelte';

  let endRef: HTMLElement;

  $effect(() => {
    if (loopbackStore.captions.length > 0) {
      endRef?.scrollIntoView({ behavior: 'smooth' });
    }
  });

  function formatTime(ts: number): string {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour12: false });
  }
</script>

<div class="transcript-view">
  {#each loopbackStore.captions.filter(c => c.isFinal) as caption (caption.id)}
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
    background: rgba(255, 255, 255, 0.03);
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
    color: #666;
    font-size: 11px;
  }
  .original {
    color: #ffd700;
    margin-bottom: 4px;
  }
  .translation {
    color: #90ee90;
  }
</style>
