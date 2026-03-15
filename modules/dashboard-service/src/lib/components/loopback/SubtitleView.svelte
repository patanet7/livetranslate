<script lang="ts">
  import { loopbackStore } from '$lib/stores/loopback.svelte';

  interface Props {
    fontSize?: number;   // Base font size in px (default: 16). Translation text is fontSize + 2.
    bgOpacity?: number;  // Background opacity 0-1 (default: 0.75)
  }

  let { fontSize = 16, bgOpacity = 0.75 }: Props = $props();

  // 3e: Clamp opacity to valid range
  const clampedOpacity = $derived(Math.max(0, Math.min(1, bgOpacity)));

  // Show last 2 final captions as subtitles
  const recentCaptions = $derived(
    loopbackStore.captions
      .filter(c => c.isFinal)
      .slice(-2)
  );

  /** Open subtitle view in a separate browser window for screen-sharing. */
  function popOut() {
    const popupWidth = 800;
    const popupHeight = 300;
    // 3c: Use availWidth/availHeight to account for taskbars
    const left = (screen.availWidth - popupWidth) / 2;
    const top = screen.availHeight - popupHeight - 100;

    // 3c: Handle popup blocker
    const popup = window.open(
      '/loopback/subtitle-popout',
      'subtitle-popout',
      `width=${popupWidth},height=${popupHeight},left=${left},top=${top},toolbar=no,menubar=no,location=no`
    );
    if (!popup) {
      console.warn('Popup blocked — allow popups for this site to use subtitle pop-out');
    }
  }
</script>

<div class="subtitle-view">
  <!-- 2c: aria-label for screen readers -->
  <button
    class="popout-btn"
    onclick={popOut}
    aria-label="Pop out subtitles into a separate window for screen-sharing"
  >
    Pop Out
  </button>
  <!-- 2a: ARIA live region for real-time subtitles -->
  <div class="subtitle-area" role="status" aria-live="assertive" aria-label="Subtitles">
    {#each recentCaptions as caption (caption.id)}
      <div class="subtitle-line" style="background: rgba(0, 0, 0, {clampedOpacity})">
        <div class="original" style="font-size: {fontSize}px">{caption.text}</div>
        {#if caption.translation}
          <div class="translation" style="font-size: {fontSize + 2}px">{caption.translation}</div>
        {/if}
      </div>
    {/each}
    {#if loopbackStore.interimText}
      <div class="subtitle-line interim" style="background: rgba(0, 0, 0, {clampedOpacity})">
        <div class="original" style="font-size: {fontSize}px">{loopbackStore.interimText}</div>
      </div>
    {/if}
  </div>
</div>

<style>
  .subtitle-view {
    height: 100%;
    min-height: 400px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 20px;
    position: relative;
  }
  .popout-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    padding: 4px 10px;
    font-size: 11px;
    background: rgba(255, 255, 255, 0.1);
    color: var(--color-muted, #ccc);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    cursor: pointer;
  }
  .popout-btn:hover {
    background: rgba(255, 255, 255, 0.2);
  }
  .subtitle-area {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .subtitle-line {
    padding: 12px 24px;
    border-radius: 8px;
    text-align: center;
  }
  .subtitle-line.interim {
    opacity: 0.6;
    font-style: italic;
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
