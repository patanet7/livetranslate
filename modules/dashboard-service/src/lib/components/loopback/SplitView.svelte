<!--
  Editorial facing-pages spread (D4.4).

  Reads as the open spread of a translation anthology:
    [ source page  ‖  centre gutter  ‖  target page ]

  The centre gutter holds the speaker byline + JBM timestamp that span
  both pages — the same way running heads do across a magazine spread.
  Each paragraph block has a hairline rule above it; section-openers get
  a thicker rule + drop-cap on both sides simultaneously.
-->
<script lang="ts">
  import { captionStore, type UnifiedCaption as CaptionEntry } from "$lib/stores/caption.svelte";
  import { paragraphTranslation, translationPhase } from "./paragraph-helpers";
  import TranslationText from "./TranslationText.svelte";

  let captionsEndOriginal: HTMLElement | undefined;
  let captionsEndTranslation: HTMLElement | undefined;

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

    for (const cap of captionStore.captions) {
      if (current === null) {
        current = { id: cap.id, captions: [cap], speaker: cap.speaker, timestamp: cap.timestamp };
        result.push(current);
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

  function isSectionOpener(idx: number): boolean {
    if (idx === 0) return true;
    return paragraphs[idx].speaker !== paragraphs[idx - 1].speaker;
  }

  let prevCaptionCount = 0;
  $effect(() => {
    const count = captionStore.captions.length;
    if (count > prevCaptionCount) {
      prevCaptionCount = count;
      captionsEndOriginal?.scrollIntoView({ behavior: "smooth" });
      captionsEndTranslation?.scrollIntoView({ behavior: "smooth" });
    }
  });

  function isCjk(lang: string): boolean {
    return ["zh", "ja", "ko"].includes(lang);
  }

  function formatTime(ts: number): string {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, { hour12: false });
  }

  const sourceLabel = $derived(
    captionStore.detectedLanguage ?? captionStore.sourceLanguage ?? "detecting…",
  );
</script>

<div class="spread" data-testid="split-view">
  <!-- Spread header — small-caps department labels for each page -->
  <header class="spread-head">
    <p class="byline page-label original-label">
      original
      <span class="lang font-mono">{sourceLabel}</span>
    </p>
    <p class="byline page-label translation-label">
      translation
      <span class="lang font-mono">{captionStore.targetLanguage}</span>
    </p>
  </header>

  <!-- Source page -->
  <section class="page page-original" data-testid="panel-original" role="log" aria-live="polite" aria-label="Original captions">
    {#each paragraphs as para, idx (para.id)}
      {@const opens = isSectionOpener(idx)}
      {@const speakerColor = captionStore.getSpeakerColor(para.speaker)}
      <article
        class="para"
        class:opens-section={opens}
        data-testid="paragraph"
        data-para-id={para.id}
        style="--speaker-color: {speakerColor};"
      >
        {#if opens}
          <header class="para-head">
            {#if para.speaker}
              <span class="byline speaker">{para.speaker}</span>
            {/if}
            <span class="ts font-mono tabular-nums">{formatTime(para.timestamp)}</span>
          </header>
        {/if}
        <p class="prose original" class:drop-cap={opens && para.speaker !== null}>
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
      </article>
    {/each}
    {#if captionStore.interimText}
      <article class="para interim">
        <p class="prose original">{captionStore.interimText}</p>
      </article>
    {/if}
    <div bind:this={captionsEndOriginal}></div>
  </section>

  <!-- Centre gutter — visual rule between pages -->
  <div class="gutter" aria-hidden="true"></div>

  <!-- Target page -->
  <section class="page page-translation" data-testid="panel-translation" role="log" aria-live="polite" aria-label="Translations">
    {#each paragraphs as para, idx (para.id)}
      {@const trans = paragraphTranslation(para.captions)}
      {@const phase = translationPhase(para.captions)}
      {@const opens = isSectionOpener(idx)}
      {@const speakerColor = captionStore.getSpeakerColor(para.speaker)}
      <article
        class="para"
        class:opens-section={opens}
        data-testid="paragraph"
        data-para-id={para.id}
        data-translation-state={phase}
        style="--speaker-color: {speakerColor};"
      >
        {#if opens}
          <header class="para-head">
            {#if para.speaker}
              <span class="byline speaker">{para.speaker}</span>
            {/if}
            <span class="ts font-mono tabular-nums">{formatTime(para.timestamp)}</span>
          </header>
        {/if}
        <p class="prose translation" class:drop-cap={opens && para.speaker !== null && !!trans}>
          <TranslationText text={trans} {phase} />
        </p>
      </article>
    {/each}
    <div bind:this={captionsEndTranslation}></div>
  </section>
</div>

<style>
  .spread {
    display: grid;
    grid-template-columns: 1fr 1px 1fr;
    grid-template-rows: auto 1fr;
    grid-template-areas:
      "head head head"
      "left gutter right";
    height: 100%;
    min-height: 400px;
    background: var(--paper);
  }

  .spread-head {
    grid-area: head;
    display: grid;
    grid-template-columns: 1fr 1fr;
    column-gap: 1px;
    padding: 1rem 2rem 0.5rem;
    border-bottom: 1px solid var(--rule);
    background: var(--paper);
  }

  .page-label {
    margin: 0;
    color: var(--ink-soft);
    display: inline-flex;
    align-items: baseline;
    gap: 0.5rem;
  }
  .page-label .lang {
    color: var(--ink-faint);
    letter-spacing: 0.04em;
    font-feature-settings: "tnum";
  }
  .original-label {
    /* left page label sits on the left edge */
    justify-self: start;
  }
  .translation-label {
    justify-self: start;
    padding-left: 1rem;
  }

  .page {
    overflow-y: auto;
    padding: 1.5rem 2rem 4rem;
  }

  .page-original {
    grid-area: left;
  }

  .gutter {
    grid-area: gutter;
    background: var(--rule);
  }

  .page-translation {
    grid-area: right;
  }

  /* ── Paragraphs ────────────────────────────────────────────── */
  .para {
    padding: 0.5rem 0 0.875rem;
  }
  .para.opens-section {
    border-top: 1px solid var(--rule);
    margin-top: 1rem;
    padding-top: 1rem;
  }
  .para.opens-section:first-of-type {
    border-top: none;
    margin-top: 0;
    padding-top: 0;
  }

  .para-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.5rem;
  }

  .speaker {
    color: var(--speaker-color, var(--ink));
    font-size: 0.75rem;
    letter-spacing: 0.16em;
    line-height: 1;
  }
  .ts {
    color: var(--ink-faint);
    font-size: 0.6875rem;
    letter-spacing: 0.04em;
  }

  .prose {
    margin: 0;
    font-family: var(--font-body);
    line-height: 1.55;
    color: var(--ink);
  }
  .prose.original {
    font-variation-settings: "opsz" 16;
    font-size: 1rem;
  }
  .prose.translation {
    font-variation-settings: "opsz" 18;
    font-style: italic;
    color: var(--ink-soft);
    font-size: 1.0625rem;
  }

  .is-draft {
    opacity: 0.78;
  }
  .unstable {
    opacity: 0.45;
    font-style: italic;
    transition: opacity 0.3s ease;
  }

  .interim {
    opacity: 0.55;
    font-style: italic;
  }

  /* ── Mobile: stack the spread vertically ──────────────────────── */
  @media (max-width: 820px) {
    .spread {
      grid-template-columns: 1fr;
      grid-template-areas:
        "head"
        "left"
        "right";
    }
    .gutter {
      display: none;
    }
    .page-translation {
      border-top: 1px solid var(--rule);
    }
    .spread-head {
      grid-template-columns: 1fr;
      gap: 0.25rem;
    }
  }
</style>
