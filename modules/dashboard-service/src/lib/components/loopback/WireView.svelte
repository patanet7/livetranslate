<!--
  WireView — teleprinter dispatch display (D8.1).

  Reads the same captionStore as SplitView/TranscriptView, but renders
  each paragraph as an AP-style wire dispatch:

      :: SLUG : LIVETRANS-2604-220537-EN-ES-SPK01
      :: SPKR : ● thomas
      :: STAT : [ FILED ]
      :: TIME : 22:05:37 · 247ms

      EN → we should ship the wire-service prototype before the standup
      ES → deberíamos enviar el prototipo del servicio antes de la reunión

                                ─ 30 ─

  Status reads off the existing translation phase + isDraft / unstableText:
    DRAFT  — any caption still has unstable text or isDraft=true
    STABLE — source done, translation pending or streaming
    FILED  — at least one caption has a complete translation

  No type-on animation: live captions arrive already-formed. The peach
  keystrike caret marks "still arriving" — appears on DRAFT source, on
  STABLE pending-target, and on streaming translations.

  Sister to SplitView (editorial spread). Pick whichever fits — both
  drive off the same store.
-->
<script lang="ts">
  import { captionStore, type UnifiedCaption as CaptionEntry } from "$lib/stores/caption.svelte";
  import { paragraphTranslation } from "./paragraph-helpers";
  import { scrollIntoViewOnGrow } from "./scroll-attachments.svelte";

  const PARAGRAPH_GAP_MS = 10000;
  const MAX_VISIBLE = 12;

  interface Paragraph {
    id: string;
    captions: CaptionEntry[];
    speaker: string | null;
    timestamp: number;
    index: number;
  }

  type Status = "DRAFT" | "STABLE" | "FILED";

  // Mirror SplitView's paragraph grouping so the two views read the same
  // session identically — switching modes mid-session shouldn't shuffle
  // the speaker/paragraph boundaries.
  const paragraphs = $derived.by(() => {
    const result: Paragraph[] = [];
    let current: Paragraph | null = null;
    let idx = 0;

    for (const cap of captionStore.captions) {
      if (current === null) {
        current = {
          id: cap.id,
          captions: [cap],
          speaker: cap.speaker,
          timestamp: cap.timestamp,
          index: idx++,
        };
        result.push(current);
        continue;
      }
      const lastTs = current.captions[current.captions.length - 1].timestamp;
      const shouldBreak =
        cap.speaker !== current.speaker || cap.timestamp - lastTs > PARAGRAPH_GAP_MS;
      if (shouldBreak) {
        current = {
          id: cap.id,
          captions: [cap],
          speaker: cap.speaker,
          timestamp: cap.timestamp,
          index: idx++,
        };
        result.push(current);
      } else {
        current.captions.push(cap);
      }
    }
    return result;
  });

  // Cap visible dispatches; older ones scroll out of the buffer (still in
  // captionStore.captions, just not painted on the wire feed).
  const visibleParagraphs = $derived(paragraphs.slice(-MAX_VISIBLE));

  function paragraphStatus(p: Paragraph): Status {
    const last = p.captions[p.captions.length - 1];
    const hasUnstable = p.captions.some((c) => c.unstableText && c.unstableText.length > 0);
    const allFinal = p.captions.every((c) => !c.isDraft);
    if (!allFinal || hasUnstable) return "DRAFT";
    if (last.translationState === "complete") return "FILED";
    return "STABLE";
  }

  function paragraphLanguage(p: Paragraph): string {
    return p.captions[p.captions.length - 1].language || "auto";
  }

  function paragraphStableSource(p: Paragraph): string {
    return p.captions
      .map((c) => c.stableText)
      .filter(Boolean)
      .join(" ");
  }

  function paragraphUnstableSource(p: Paragraph): string {
    const last = p.captions[p.captions.length - 1];
    return last.unstableText ?? "";
  }

  function paragraphLatency(p: Paragraph): number {
    // Approximate: time from first caption to most recent. Real backend
    // latency isn't tracked per-paragraph — this stands in as a proxy.
    const last = p.captions[p.captions.length - 1];
    if (last.translationState !== "complete") return 0;
    return Math.max(0, last.timestamp - p.timestamp + 220);
  }

  function makeSlug(p: Paragraph): string {
    const d = new Date(p.timestamp);
    const date = `${d.getUTCDate().toString().padStart(2, "0")}${(d.getUTCMonth() + 1).toString().padStart(2, "0")}`;
    const time = `${d.getUTCHours().toString().padStart(2, "0")}${d.getUTCMinutes().toString().padStart(2, "0")}${d.getUTCSeconds().toString().padStart(2, "0")}`;
    const src = paragraphLanguage(p).toUpperCase();
    const tgt = captionStore.targetLanguage.toUpperCase();
    return `LIVETRANS-${date}-${time}-${src}-${tgt}-SPK${(p.index + 1).toString().padStart(2, "0")}`;
  }

  // Auto-scroll on caption arrival — uses the shared {@attach} helper
  // attached to the trailing div at the bottom of the feed.

  function fmtClock(ts: number): string {
    const d = new Date(ts);
    return `${d.getUTCHours().toString().padStart(2, "0")}:${d.getUTCMinutes().toString().padStart(2, "0")}:${d.getUTCSeconds().toString().padStart(2, "0")}`;
  }
</script>

<div class="terminal" data-testid="wire-view">
  <aside class="sprocket" aria-hidden="true">
    {#each Array(40) as _, i (i)}<span class="hole"></span>{/each}
  </aside>

  <header class="masthead">
    <div class="brand">
      <span class="display">Livetrans</span>
      <span class="department">/ wire terminal</span>
    </div>
    <div class="meta">
      <span class="cell">
        <span class="lbl">issue</span>
        <span class="val tabular-nums">{paragraphs.length.toString().padStart(3, "0")}</span>
      </span>
      <span class="cell">
        <span class="lbl">src</span>
        <span class="val font-mono"
          >{captionStore.detectedLanguage ?? captionStore.sourceLanguage ?? "auto"}</span
        >
      </span>
      <span class="cell">
        <span class="lbl arrow-lbl">→</span>
        <span class="val font-mono">{captionStore.targetLanguage}</span>
      </span>
      <span class="cell live" class:hot={captionStore.isCapturing}>
        <span class="dot"></span>
        <span class="lbl">{captionStore.isCapturing ? "audio in" : "standby"}</span>
      </span>
    </div>
  </header>

  <hr class="rule" />

  <main class="feed" role="log" aria-live="polite" aria-label="wire dispatches">
    {#if paragraphs.length === 0}
      <p class="awaiting">
        <span class="prompt">$</span>
        {#if captionStore.isCapturing}
          awaiting first dispatch
        {:else}
          terminal idle — start capture or demo
        {/if}
        <span class="blink">▍</span>
      </p>
    {/if}

    {#each visibleParagraphs as p (p.id)}
      {@const status = paragraphStatus(p)}
      {@const speakerColor = captionStore.getSpeakerColor(p.speaker)}
      {@const stable = paragraphStableSource(p)}
      {@const unstable = paragraphUnstableSource(p)}
      {@const trans = paragraphTranslation(p.captions)}
      {@const latency = paragraphLatency(p)}
      <article class="dispatch" data-status={status} data-testid="dispatch">
        <ol class="slug-block">
          <li><span class="lbl">slug</span><span class="val slug-val">{makeSlug(p)}</span></li>
          <li>
            <span class="lbl">spkr</span>
            <span class="val speaker">
              <span class="pip" style="background: {speakerColor}"></span>
              {p.speaker ?? "—"}
            </span>
          </li>
          <li>
            <span class="lbl">stat</span>
            <span class="val flag flag-{status.toLowerCase()}">[ {status} ]</span>
          </li>
          <li>
            <span class="lbl">time</span>
            <span class="val tabular-nums"
              >{fmtClock(p.timestamp)}{latency > 0 ? ` · ${latency}ms` : ""}</span
            >
          </li>
        </ol>

        <div class="lines">
          <p class="line src">
            <span class="lang">{paragraphLanguage(p).toUpperCase()}</span>
            <span class="arrow">→</span>
            <span class="text">
              {stable}
              {#if unstable}
                {" "}<span class="unstable-tx">{unstable}</span>
              {/if}
              {#if status === "DRAFT"}<span class="caret">▍</span>{/if}
            </span>
          </p>
          {#if trans}
            <p class="line tgt" class:filed={status === "FILED"}>
              <span class="lang">{captionStore.targetLanguage.toUpperCase()}</span>
              <span class="arrow">→</span>
              <span class="text">
                {trans}
                {#if status !== "FILED"}<span class="caret">▍</span>{/if}
              </span>
            </p>
          {:else if status !== "DRAFT"}
            <p class="line tgt awaiting-tgt">
              <span class="lang">{captionStore.targetLanguage.toUpperCase()}</span>
              <span class="arrow">→</span>
              <span class="text">
                <span class="dim">awaiting translation</span>
                <span class="caret">▍</span>
              </span>
            </p>
          {/if}
        </div>

        {#if status === "FILED"}
          <p class="thirty" aria-label="end of dispatch">─ 30 ─</p>
        {/if}
      </article>
    {/each}

    <div {@attach scrollIntoViewOnGrow(() => captionStore.captions.length)}></div>
  </main>
</div>

<style>
  .terminal {
    position: relative;
    display: grid;
    grid-template-rows: auto auto 1fr;
    height: 100%;
    padding: 1.25rem 2.5rem 1rem 4rem;
    background: var(--paper);
    color: var(--ink);
    font-family: var(--font-mono);
    font-size: 13px;
    line-height: 1.55;
    font-feature-settings: "tnum", "ss01", "calt";
    overflow: hidden;
  }

  /* Sprocket gutter — fanfold tractor-feed paper. Decorative only. */
  .sprocket {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 1rem;
    width: 1.4rem;
    border-right: 1px dashed var(--rule);
    display: grid;
    grid-template-rows: repeat(40, 1fr);
    align-items: center;
    justify-items: center;
    pointer-events: none;
  }
  .sprocket .hole {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--paper);
    box-shadow: inset 0 0 0 1px var(--rule);
    opacity: 0.5;
  }

  .masthead {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 2rem;
    flex-wrap: wrap;
  }
  .brand {
    display: inline-flex;
    align-items: baseline;
    gap: 0.6rem;
  }
  .brand .display {
    font-family: var(--font-display);
    font-variation-settings: "opsz" 96, "SOFT" 30;
    font-size: 1.875rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--ink);
    line-height: 1;
  }
  .brand .department {
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    color: var(--ink-faint);
    letter-spacing: 0.12em;
    text-transform: lowercase;
  }
  .meta {
    display: inline-flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.75rem;
    text-transform: uppercase;
    flex-wrap: wrap;
  }
  .meta .cell {
    display: inline-flex;
    gap: 0.4rem;
    align-items: baseline;
  }
  .meta .lbl {
    color: var(--ink-faint);
    letter-spacing: 0.1em;
  }
  .meta .arrow-lbl {
    letter-spacing: 0;
  }
  .meta .val {
    color: var(--ink);
    font-weight: 500;
  }
  .meta .live {
    align-items: center;
    gap: 0.45rem;
    padding: 0.18rem 0.55rem;
    border: 1px solid var(--rule);
    border-radius: 999px;
  }
  .meta .live .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ink-faint);
    transition: background 220ms ease, box-shadow 220ms ease;
  }
  .meta .live.hot .dot {
    background: var(--peach-deep);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--peach-deep) 22%, transparent);
    animation: live-breathe 2.4s ease-in-out infinite;
  }
  .meta .live.hot .lbl {
    color: var(--ink);
  }
  @keyframes live-breathe {
    0%, 100% { box-shadow: 0 0 0 3px color-mix(in srgb, var(--peach-deep) 22%, transparent); }
    50%      { box-shadow: 0 0 0 6px color-mix(in srgb, var(--peach-deep) 8%, transparent); }
  }

  .rule {
    height: 1px;
    border: 0;
    background: var(--rule);
    margin: 0.875rem 0 0.625rem;
  }

  .feed {
    overflow-y: auto;
    padding: 0.5rem 0 1.5rem;
    min-height: 0;
  }
  .awaiting {
    color: var(--ink-faint);
    font-size: 0.875rem;
    padding: 1rem 0;
  }
  .awaiting .prompt {
    color: var(--purple-deep);
    margin-right: 0.5rem;
  }
  .blink {
    animation: blink 0.8s steps(2) infinite;
  }
  @keyframes blink {
    50% { opacity: 0; }
  }

  /* Dispatch */
  .dispatch {
    padding: 1rem 0 1.25rem;
    border-top: 1px solid var(--rule);
    animation: arrive 280ms ease-out;
  }
  .dispatch:first-of-type {
    border-top: none;
    padding-top: 0;
  }
  @keyframes arrive {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .slug-block {
    list-style: none;
    margin: 0 0 0.75rem;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.1rem 1.25rem;
    font-size: 0.75rem;
  }
  .slug-block li {
    display: inline-flex;
    gap: 0.5rem;
    align-items: baseline;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .slug-block .lbl {
    color: var(--ink-faint);
    width: 2.6rem;
    flex-shrink: 0;
  }
  .slug-block .lbl::after {
    content: ":";
  }
  .slug-block .val {
    color: var(--ink);
  }
  .slug-val {
    color: var(--purple-deep);
    font-feature-settings: "tnum", "ss01";
  }
  .speaker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    text-transform: lowercase;
  }
  .speaker .pip {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  /* Status flags — colored wash + ink text. Identity in box, legibility in type. */
  .flag {
    padding: 0.05rem 0.4rem 0.08rem;
    border-radius: 2px;
    font-weight: 600;
    letter-spacing: 0.08em;
    background: color-mix(in srgb, var(--rule) 50%, transparent);
    color: var(--ink);
  }
  .flag-draft {
    background: color-mix(in srgb, var(--oxblood) 18%, transparent);
  }
  .flag-stable {
    background: color-mix(in srgb, var(--ochre) 22%, transparent);
  }
  .flag-filed {
    background: color-mix(in srgb, var(--sage) 26%, transparent);
  }

  /* Source / target lines */
  .lines {
    margin: 0.25rem 0 0.5rem;
  }
  .line {
    margin: 0.18rem 0;
    display: grid;
    grid-template-columns: 2.4rem 1.1rem 1fr;
    gap: 0.3rem;
    align-items: baseline;
    font-size: 0.9375rem;
  }
  .line .lang {
    color: var(--purple-deep);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .line .arrow {
    color: var(--ink-faint);
    font-weight: 300;
  }
  .line .text {
    color: var(--ink);
  }
  .line.tgt .text {
    color: color-mix(in srgb, var(--ink) 70%, var(--paper));
    font-style: italic;
  }
  .line.tgt.filed .text {
    color: var(--ink);
    font-style: normal;
  }
  .line.awaiting-tgt .text .dim {
    color: var(--ink-faint);
    font-style: italic;
  }

  /* Unstable text — wire-aesthetic version of the ink-drying underline.
     Uses peach-tinted ink that lifts on the next stable arrival. */
  .unstable-tx {
    color: var(--peach-ink);
    text-decoration: underline;
    text-decoration-style: dotted;
    text-decoration-thickness: 0.075em;
    text-underline-offset: 0.18em;
    text-decoration-color: color-mix(in srgb, var(--peach-deep) 35%, transparent);
  }

  /* Keystrike caret */
  .caret {
    display: inline-block;
    width: 0.5ch;
    margin-left: 0.05ch;
    color: var(--peach);
    animation: keystrike 0.18s steps(1) infinite;
  }
  @keyframes keystrike {
    50% { opacity: 0.3; }
  }

  /* End-of-dispatch glyph */
  .thirty {
    margin: 0.6rem 0 0;
    color: var(--peach-deep);
    text-align: center;
    font-size: 0.8125rem;
    letter-spacing: 0.4em;
    animation: thirty-fade 360ms ease-out;
  }
  @keyframes thirty-fade {
    from { opacity: 0; letter-spacing: 0.1em; }
    to { opacity: 1; letter-spacing: 0.4em; }
  }

  @media (prefers-reduced-motion: reduce) {
    .blink, .caret { animation: none; }
    .thirty { animation: none; }
    .dispatch { animation: none; }
    .meta .live.hot .dot { animation: none; }
  }
</style>
