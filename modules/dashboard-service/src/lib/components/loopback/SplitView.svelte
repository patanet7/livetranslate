<script lang="ts">
  import { loopbackStore } from '$lib/stores/loopback.svelte';

  let captionsEndOriginal: HTMLElement;
  let captionsEndTranslation: HTMLElement;

  $effect(() => {
    // Auto-scroll when new captions arrive
    if (loopbackStore.captions.length > 0) {
      captionsEndOriginal?.scrollIntoView({ behavior: 'smooth' });
      captionsEndTranslation?.scrollIntoView({ behavior: 'smooth' });
    }
  });
</script>

<div class="split-view">
  <!-- Original language panel -->
  <div class="panel panel-original">
    <div class="panel-header">
      Original ({loopbackStore.sourceLanguage ?? 'detecting...'})
    </div>
    <div class="panel-content">
      {#each loopbackStore.captions as caption (caption.id)}
        <div
          class="caption-entry"
          class:interim={!caption.isFinal}
          style="border-left-color: {loopbackStore.getSpeakerColor(caption.speakerId)}"
        >
          {#if caption.speakerId}
            <span class="speaker" style="color: {loopbackStore.getSpeakerColor(caption.speakerId)}">
              {caption.speakerId}:
            </span>
          {/if}
          <span class="text">{caption.text}</span>
        </div>
      {/each}
      {#if loopbackStore.interimText}
        <div class="caption-entry interim">
          <span class="text">{loopbackStore.interimText}</span>
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
    <div class="panel-content">
      {#each loopbackStore.captions.filter(c => c.isFinal) as caption (caption.id)}
        <div
          class="caption-entry"
          style="border-left-color: {loopbackStore.getSpeakerColor(caption.speakerId)}"
        >
          {#if caption.speakerId}
            <span class="speaker" style="color: {loopbackStore.getSpeakerColor(caption.speakerId)}">
              {caption.speakerId}:
            </span>
          {/if}
          <span class="text">
            {caption.translation ?? '...'}
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
  .panel-original .panel-header { color: #ffd700; }
  .panel-translation .panel-header { color: #90ee90; }
  .panel-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }
  .caption-entry {
    padding: 8px;
    margin-bottom: 8px;
    border-left: 3px solid;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.03);
  }
  .caption-entry.interim {
    opacity: 0.5;
    font-style: italic;
  }
  .speaker {
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
  }
</style>
