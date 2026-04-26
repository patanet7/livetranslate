<!--
  Editorial transcript — ledger / running-script feel (D4.3).

  Two-column row layout per entry:
    [ JetBrains Mono timestamp gutter | speaker-coloured byline + body ]

  Drop cap fires for the first entry of a new speaker. Translation is
  printed beneath the original in a quieter ink — same ink-soft tone as
  the kicker. Hairline rules between entries instead of card backgrounds.
-->
<script lang="ts">
  import { captionStore, type UnifiedCaption as CaptionEntry } from "$lib/stores/caption.svelte";
  import { paragraphTranslation, translationPhase } from "./paragraph-helpers";
  import TranslationText from "./TranslationText.svelte";

  let endRef: HTMLElement | undefined;

  const PARAGRAPH_GAP_MS = 10000;

  interface Paragraph {
    id: string;
    captions: CaptionEntry[];
    speaker: string | null;
    timestamp: number;
  }

  const paragraphs = $derived.by(() => {
    const result: Paragraph[] = [];
    let current: Paragraph | null = null;
    let prevSpeaker: string | null | undefined = undefined;

    for (const cap of captionStore.captions) {
      if (current === null) {
        current = { id: cap.id, captions: [cap], speaker: cap.speaker, timestamp: cap.timestamp };
        result.push(current);
        prevSpeaker = cap.speaker;
        continue;
      }
      const lastTs = current.captions[current.captions.length - 1].timestamp;
      const shouldBreak =
        cap.speaker !== current.speaker || cap.timestamp - lastTs > PARAGRAPH_GAP_MS;

      if (shouldBreak) {
        current = { id: cap.id, captions: [cap], speaker: cap.speaker, timestamp: cap.timestamp };
        result.push(current);
      } else {
        current.captions.push(cap);
      }
    }
    return result;
  });

  /** Whether this paragraph opens a new speaker section (drop-cap trigger). */
  function isSectionOpener(idx: number): boolean {
    if (idx === 0) return true;
    return paragraphs[idx].speaker !== paragraphs[idx - 1].speaker;
  }

  let prevLength = 0;
  $effect(() => {
    const len = captionStore.captions.length;
    if (len > prevLength) {
      prevLength = len;
      endRef?.scrollIntoView({ behavior: "smooth" });
    }
  });

  function formatTime(ts: number): string {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, { hour12: false });
  }

  function isCjk(lang: string): boolean {
    return ["zh", "ja", "ko"].includes(lang);
  }
</script>

<div
  class="transcript-view"
  data-testid="transcript-view"
  role="log"
  aria-live="polite"
  aria-label="Transcript"
>
  {#each paragraphs as para, idx (para.id)}
    {@const trans = paragraphTranslation(para.captions)}
    {@const phase = translationPhase(para.captions)}
    {@const speakerColor = captionStore.getSpeakerColor(para.speaker)}
    {@const opensSection = isSectionOpener(idx)}
    <article
      class="entry"
      class:opens-section={opensSection}
      data-testid="transcript-entry"
      data-entry-id={para.id}
      data-translation-state={phase}
      style="--speaker-color: {speakerColor};"
    >
      <!-- Mono timestamp gutter -->
      <time class="ts font-mono tabular-nums" data-testid="entry-timestamp">
        {formatTime(para.timestamp)}
      </time>

      <div class="body">
        {#if para.speaker && opensSection}
          <p class="byline speaker" data-testid="speaker">
            {para.speaker}
          </p>
        {/if}

        <p class="original" data-testid="original-text" class:drop-cap={opensSection}>
          {#each para.captions as cap, i}
            {#if i > 0 && cap.stableText && !isCjk(cap.language)}{" "}{/if}
            <span
              class:is-draft={cap.isDraft}
              data-testid="caption-text"
              data-segment-id={cap.id}>{cap.stableText}</span
            >
            {#if cap.unstableText}<span class="unstable"
                >{#if !isCjk(cap.language)}{" "}{/if}{cap.unstableText}</span
              >{/if}
          {/each}
        </p>

        {#if trans}
          <p class="translation" data-testid="translation-text">
            <TranslationText text={trans} {phase} />
          </p>
        {/if}
      </div>
    </article>
  {/each}

  {#if captionStore.interimText}
    <article class="entry interim">
      <span class="ts font-mono">…</span>
      <div class="body">
        <p class="original">{captionStore.interimText}</p>
      </div>
    </article>
  {/if}

  <div bind:this={endRef}></div>
</div>

<style>
  .transcript-view {
    padding: 1.75rem 2.5rem;
    overflow-y: auto;
    height: 100%;
    min-height: 400px;
    background: var(--paper);
    /* Reading column max-width — typeset prose breathes better when not stretched */
    max-width: 76rem;
    margin: 0 auto;
  }

  .entry {
    display: grid;
    grid-template-columns: 5.25rem 1fr;
    column-gap: 1.5rem;
    padding: 0.875rem 0;
    border-bottom: 1px solid var(--rule-soft);
  }
  .entry:last-of-type {
    border-bottom: none;
  }

  /* Section openers get a thicker top rule — visual paragraph break */
  .entry.opens-section {
    border-top: 1px solid var(--rule);
    margin-top: 0.5rem;
    padding-top: 1.25rem;
  }
  .entry.opens-section:first-of-type {
    border-top: none;
    margin-top: 0;
  }

  .ts {
    color: var(--ink-faint);
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    line-height: 1.7;
    text-align: right;
    /* The speaker color reads as a single dot before the timestamp */
    position: relative;
    padding-right: 0.5rem;
  }
  .opens-section .ts::after {
    content: "";
    position: absolute;
    right: -0.625rem;
    top: 0.55rem;
    width: 0.4375rem;
    height: 0.4375rem;
    border-radius: 9999px;
    background: var(--speaker-color, var(--ink-faint));
  }

  .body {
    min-width: 0;
  }

  .speaker {
    /* WCAG: speaker byline at 12px needs 4.5:1, which several earth-tone
       speaker hues fail on paper. Preserve identity via the colored pip
       (line 197); render the byline text in --ink for AA. */
    margin: 0 0 0.375rem;
    color: var(--ink);
    font-size: 0.75rem;
    letter-spacing: 0.16em;
    line-height: 1;
  }

  .original {
    margin: 0;
    font-family: var(--font-body);
    font-variation-settings: "opsz" 16;
    line-height: 1.55;
    color: var(--ink);
    font-size: 1rem;
  }

  .translation {
    margin: 0.5rem 0 0;
    font-family: var(--font-body);
    font-style: italic;
    font-variation-settings: "opsz" 18;
    line-height: 1.5;
    color: var(--ink-soft);
  }

  .interim {
    opacity: 0.55;
    font-style: italic;
  }

  .is-draft {
    opacity: 0.78;
  }

  .unstable {
    opacity: 0.45;
    font-style: italic;
    transition: opacity 0.3s ease;
  }

  /* Mobile: collapse the timestamp gutter */
  @media (max-width: 720px) {
    .transcript-view {
      padding: 1.25rem;
    }
    .entry {
      grid-template-columns: 1fr;
      column-gap: 0;
    }
    .ts {
      text-align: left;
      margin-bottom: 0.25rem;
    }
    .opens-section .ts::after {
      right: auto;
      left: 1.25rem;
      top: 0.6rem;
    }
  }
</style>
