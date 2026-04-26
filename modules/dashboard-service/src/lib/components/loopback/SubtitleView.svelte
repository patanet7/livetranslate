<!--
  Editorial subtitle treatment (D4.2).

  Bottom-anchored captions printed on an ink plate. Original line uses
  Newsreader body; translation reads as the kicker (italic, slightly
  larger, peach-soft). Drop-cap when speaker changes — even in a
  two-line subtitle frame it works because we only show the most-recent
  pair anyway.
-->
<script lang="ts">
  import { captionStore } from "$lib/stores/caption.svelte";
  import TranslationText from "./TranslationText.svelte";

  interface Props {
    fontSize?: number;
    bgOpacity?: number;
  }

  let { fontSize = 18, bgOpacity = 0.88 }: Props = $props();

  const clampedOpacity = $derived(Math.max(0, Math.min(1, bgOpacity)));
  const recentCaptions = $derived(captionStore.captions.slice(-2));

  function popOut() {
    const sessionId = captionStore.meetingSessionId;
    if (!sessionId) {
      console.warn("No active meeting session — start a meeting first to use subtitle pop-out");
      return;
    }
    const popupWidth = 800;
    const popupHeight = 300;
    const left = (screen.availWidth - popupWidth) / 2;
    const top = screen.availHeight - popupHeight - 100;
    const popup = window.open(
      `/captions?session=${sessionId}&position=bottom&maxCaptions=2&showStatus=false`,
      "subtitle-popout",
      `width=${popupWidth},height=${popupHeight},left=${left},top=${top},toolbar=no,menubar=no,location=no`,
    );
    if (!popup) {
      console.warn("Popup blocked — allow popups for this site to use subtitle pop-out");
    }
  }
</script>

<div class="subtitle-view" data-testid="subtitle-view">
  <button
    type="button"
    class="popout-btn"
    data-testid="popout-button"
    onclick={popOut}
    aria-label="Pop out subtitles into a separate window for screen-sharing"
  >
    <span class="byline">pop out</span>
  </button>

  <div
    class="subtitle-area"
    data-testid="subtitle-area"
    role="status"
    aria-live="assertive"
    aria-label="Subtitles"
  >
    {#each recentCaptions as caption (caption.id)}
      {@const captionPhase = caption.translationState}
      {@const speakerColor = captionStore.getSpeakerColor(caption.speaker)}
      <article
        class="subtitle-line"
        data-testid="subtitle-line"
        data-caption-id={caption.id}
        data-translation-state={captionPhase}
        style="--plate-opacity: {clampedOpacity}; --speaker-color: {speakerColor};"
      >
        {#if caption.speaker}
          <p class="byline speaker-line" data-testid="speaker-label">
            {caption.speaker}
          </p>
        {/if}
        <p class="original" data-testid="subtitle-original" style="font-size: {fontSize}px">
          <span class="caption-text" class:is-draft={caption.isDraft}>
            {caption.stableText}{#if caption.unstableText}<span class="unstable"> {caption.unstableText}</span>{/if}
          </span>
        </p>
        <p
          class="translation kicker"
          data-testid="subtitle-translation"
          style="font-size: {fontSize + 2}px"
        >
          <TranslationText text={caption.translation ?? ""} phase={captionPhase} />
        </p>
      </article>
    {/each}

    {#if captionStore.interimText}
      <article class="subtitle-line interim" style="--plate-opacity: {clampedOpacity};">
        <p class="original" style="font-size: {fontSize}px">
          {captionStore.interimText}
        </p>
      </article>
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
    padding: 1.5rem 2rem 2rem;
    position: relative;
    background: var(--paper);
  }

  .popout-btn {
    position: absolute;
    top: 0.875rem;
    right: 1rem;
    padding: 0.375rem 0.75rem;
    border: 1px solid var(--rule);
    border-radius: 9999px;
    background: var(--paper);
    color: var(--ink-soft);
    transition: color 160ms ease, border-color 160ms ease;
  }
  .popout-btn:hover {
    color: var(--ink);
    border-color: var(--ink-soft);
  }

  .subtitle-area {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .subtitle-line {
    padding: 1.125rem 1.75rem;
    background: color-mix(in srgb, var(--ink) calc(var(--plate-opacity, 0.88) * 100%), var(--paper));
    color: var(--paper);
    border-radius: 0.25rem;
    /* Speaker color as a left-edge "ply" — barely-there ink stripe */
    box-shadow: inset 4px 0 0 var(--speaker-color, transparent);
    text-align: left;
  }

  .subtitle-line.interim {
    opacity: 0.55;
    font-style: italic;
  }

  .speaker-line {
    margin: 0 0 0.375rem;
    color: var(--speaker-color, var(--peach));
    font-size: 0.6875rem;
    letter-spacing: 0.18em;
    line-height: 1;
  }

  .original {
    margin: 0;
    font-family: var(--font-body);
    font-variation-settings: "opsz" 24;
    line-height: 1.4;
    color: var(--paper);
  }

  .translation {
    margin: 0.375rem 0 0;
    color: color-mix(in srgb, var(--peach) 55%, var(--paper));
    line-height: 1.35;
  }

  .caption-text { transition: opacity 0.3s ease; }
  .caption-text.is-draft { opacity: 0.78; }

  .unstable {
    opacity: 0.45;
    font-style: italic;
    transition: opacity 0.3s ease;
  }
</style>
