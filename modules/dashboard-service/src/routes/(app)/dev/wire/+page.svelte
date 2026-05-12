<!--
  /dev/wire — Wire-Service exploration (sister to /dev/earwyrm).

  An alternative aesthetic direction for the loopback / live-translation
  surface, in the spirit of an AP/Reuters teleprinter terminal rather
  than a magazine spread. Everything you'd see here ships in a real
  newsroom: dateline, slug line, status flag, latency stamp, end-of-
  dispatch glyph.

  Why a sister direction:
    The editorial Riso identity (D0..D7) reads as a magazine *about*
    publishing, not as the *act* of listening-and-translating. Wire
    feeds are explicitly real-time: every dispatch is live, fresh, and
    moments later it's history. The Earwyrm flips from listener to
    operator at the keys.

  Stylistic moves:
    · JetBrains Mono dominates; Fraunces only appears in the masthead.
    · Each dispatch types itself in character-by-character with a peach
      keystrike caret — replaces the "ink-drying" underline.
    · Status flips DRAFT -> STABLE -> FILED with bracketed tags.
    · Dispatch ends with the journalistic '- 30 -' marker.
    · Sprocket-hole gutter down the left margin (fanfold paper).
    · Live RMS is a single column of bars on the right, audio-meter style.

  Mock data only — wired to setInterval, not the caption store. This is
  an aesthetic exploration. If picked, it can be refactored to read
  from $lib/stores/caption.svelte.
-->
<script lang="ts">
  type Status = "DRAFT" | "STABLE" | "FILED";

  interface Dispatch {
    id: string;
    slug: string;
    speaker: string;
    speakerColor: string;
    timestamp: number;
    sourceLang: string;
    targetLang: string;
    sourceFull: string;
    sourceTyped: string;
    targetFull: string;
    targetTyped: string;
    status: Status;
    latencyMs: number;
  }

  // [speaker, color, srcLang, tgtLang, source, target]
  const SAMPLES: Array<[string, string, string, string, string, string]> = [
    [
      "thomas",
      "#C26F49",
      "en",
      "es",
      "we should ship the wire-service prototype before the standup",
      "deberíamos enviar el prototipo del servicio antes de la reunión",
    ],
    [
      "akira",
      "#7A8C5C",
      "ja",
      "en",
      "翻訳のレイテンシは三百ミリ秒以下に抑えたい",
      "we want translation latency held under three hundred milliseconds",
    ],
    [
      "maría",
      "#C8893E",
      "es",
      "en",
      "el sistema detectó cinco hablantes simultáneos",
      "the system detected five simultaneous speakers",
    ],
    [
      "jens",
      "#5F6B85",
      "de",
      "en",
      "die Untertitel werden direkt in den Browser-Stream gerendert",
      "the captions render directly into the browser stream",
    ],
    [
      "li",
      "#6B4A6B",
      "zh",
      "en",
      "我们刚才检测到了说话人切换",
      "we just detected a speaker change",
    ],
  ];

  let dispatches = $state<Dispatch[]>([]);
  let nextSampleIdx = 0;
  let dispatchCounter = $state(0);
  let now = $state(Date.now());
  let audioRms = $state(0);

  const sessionStart = Date.now();

  // Running clock + animated RMS so the meter feels live without a mic.
  $effect(() => {
    const id = setInterval(() => {
      now = Date.now();
      audioRms = Math.max(0, Math.min(1, audioRms + (Math.random() - 0.45) * 0.3));
    }, 80);
    return () => clearInterval(id);
  });

  // Spawn a new dispatch every ~7.5s and type it on.
  $effect(() => {
    const spawn = () => {
      const [speaker, color, src, tgt, sourceFull, targetFull] =
        SAMPLES[nextSampleIdx % SAMPLES.length];
      nextSampleIdx++;
      dispatchCounter++;
      const d: Dispatch = {
        id: `d-${dispatchCounter}`,
        slug: makeSlug(src, tgt, dispatchCounter),
        speaker,
        speakerColor: color,
        timestamp: Date.now(),
        sourceLang: src,
        targetLang: tgt,
        sourceFull,
        sourceTyped: "",
        targetFull,
        targetTyped: "",
        status: "DRAFT",
        latencyMs: 0,
      };
      dispatches = [...dispatches, d].slice(-6);
      typeOn(d);
    };
    const first = setTimeout(spawn, 800);
    const id = setInterval(spawn, 7500);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  });

  function makeSlug(src: string, tgt: string, n: number): string {
    const d = new Date();
    const date = `${d.getUTCDate().toString().padStart(2, "0")}${(d.getUTCMonth() + 1).toString().padStart(2, "0")}`;
    const time = `${d.getUTCHours().toString().padStart(2, "0")}${d.getUTCMinutes().toString().padStart(2, "0")}${d.getUTCSeconds().toString().padStart(2, "0")}`;
    return `LIVETRANS-${date}-${time}-${src.toUpperCase()}-${tgt.toUpperCase()}-SPK${n.toString().padStart(2, "0")}`;
  }

  function typeOn(d: Dispatch) {
    let i = 0;
    const sourceTimer = setInterval(() => {
      i++;
      const updated = { ...d, sourceTyped: d.sourceFull.slice(0, i) };
      patch(updated);
      if (i >= d.sourceFull.length) {
        clearInterval(sourceTimer);
        setTimeout(() => {
          patch({ ...updated, status: "STABLE" });
          let j = 0;
          const targetTimer = setInterval(() => {
            j++;
            const t: Dispatch = {
              ...updated,
              status: "STABLE",
              targetTyped: d.targetFull.slice(0, j),
            };
            patch(t);
            if (j >= d.targetFull.length) {
              clearInterval(targetTimer);
              setTimeout(() => {
                patch({
                  ...t,
                  status: "FILED",
                  latencyMs: 220 + Math.floor(Math.random() * 240),
                });
              }, 380);
            }
          }, 28);
        }, 420);
      }
    }, 32);
  }

  function patch(d: Dispatch) {
    dispatches = dispatches.map((x) => (x.id === d.id ? d : x));
  }

  function fmtClock(ts: number): string {
    const d = new Date(ts);
    return `${d.getUTCHours().toString().padStart(2, "0")}:${d.getUTCMinutes().toString().padStart(2, "0")}:${d.getUTCSeconds().toString().padStart(2, "0")}`;
  }

  function fmtSinceStart(ts: number): string {
    const s = Math.floor((ts - sessionStart) / 1000);
    return `T+${s.toString().padStart(4, "0")}s`;
  }

  const METER_BARS = 12;
  const meterBars = $derived(
    Array.from({ length: METER_BARS }, (_, i) => audioRms * METER_BARS > i),
  );

  const filed = $derived(dispatches.filter((d) => d.status === "FILED").length);
  const inFlight = $derived(dispatches.filter((d) => d.status !== "FILED").length);
</script>

<svelte:head>
  <title>Wire Terminal — LiveTrans</title>
</svelte:head>

<div class="terminal">
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
        <span class="val tabular-nums">{dispatchCounter.toString().padStart(3, "0")}</span>
      </span>
      <span class="cell">
        <span class="lbl">utc</span>
        <span class="val tabular-nums">{fmtClock(now)}</span>
      </span>
      <span class="cell">
        <span class="lbl">since</span>
        <span class="val tabular-nums">{fmtSinceStart(now)}</span>
      </span>
      <span class="cell live" class:hot={audioRms > 0.15}>
        <span class="dot"></span>
        <span class="lbl">audio in</span>
      </span>
    </div>
  </header>

  <hr class="rule" />

  <div class="ticker" aria-label="recent dispatches">
    <div class="ticker-track">
      {#each dispatches.slice().reverse() as d (d.id + "-tk")}
        <span class="ticker-item">
          <span class="ticker-slug">{d.slug.split("-").slice(-3).join("·")}</span>
          <span class="ticker-text">{d.targetTyped || d.sourceTyped || "…"}</span>
          <span class="ticker-sep">::</span>
        </span>
      {/each}
      {#if dispatches.length === 0}
        <span class="ticker-item idle">awaiting first dispatch ::</span>
      {/if}
    </div>
  </div>

  <hr class="rule" />

  <main class="feed" aria-live="polite">
    {#each dispatches as d (d.id)}
      <article class="dispatch" data-status={d.status}>
        <ol class="slug-block">
          <li><span class="lbl">slug</span><span class="val slug-val">{d.slug}</span></li>
          <li>
            <span class="lbl">spkr</span>
            <span class="val speaker">
              <span class="pip" style="background: {d.speakerColor}"></span>
              {d.speaker}
            </span>
          </li>
          <li>
            <span class="lbl">stat</span>
            <span class="val flag flag-{d.status.toLowerCase()}">[ {d.status} ]</span>
          </li>
          <li>
            <span class="lbl">laty</span>
            <span class="val tabular-nums">{d.latencyMs > 0 ? `${d.latencyMs}ms` : "—"}</span>
          </li>
        </ol>

        <div class="lines">
          <p class="line src">
            <span class="lang">{d.sourceLang.toUpperCase()}</span>
            <span class="arrow">→</span>
            <span class="text"
              >{d.sourceTyped}<span
                class="caret"
                class:done={d.sourceTyped === d.sourceFull}>▍</span
              ></span
            >
          </p>
          <p class="line tgt" class:filed={d.status === "FILED"}>
            <span class="lang">{d.targetLang.toUpperCase()}</span>
            <span class="arrow">→</span>
            <span class="text"
              >{d.targetTyped}<span
                class="caret"
                class:done={d.targetTyped === d.targetFull && d.status === "FILED"}>▍</span
              ></span
            >
          </p>
        </div>

        {#if d.status === "FILED"}
          <p class="thirty" aria-label="end of dispatch">─ 30 ─</p>
        {/if}
      </article>
    {/each}

    {#if dispatches.length === 0}
      <p class="awaiting">
        <span class="prompt">$</span> awaiting wire input <span class="blink">▍</span>
      </p>
    {/if}
  </main>

  <footer class="strip">
    <div class="op">
      <span class="op-label">earwyrm·op</span>
      <span class="op-state">{inFlight > 0 ? "filing" : "standby"}</span>
    </div>
    <div class="meter" aria-hidden="true">
      {#each meterBars as lit, i (i)}
        <span class="bar" class:lit style="--i: {i}"></span>
      {/each}
    </div>
    <div class="counters">
      <span><span class="lbl">filed</span> <span class="val tabular-nums">{filed.toString().padStart(2, "0")}</span></span>
      <span><span class="lbl">inflt</span> <span class="val tabular-nums">{inFlight.toString().padStart(2, "0")}</span></span>
      <span><span class="lbl">avg</span> <span class="val tabular-nums">312ms</span></span>
    </div>
  </footer>
</div>

<style>
  .terminal {
    position: relative;
    display: grid;
    grid-template-rows: auto auto auto auto 1fr auto;
    min-height: calc(100vh - 4rem);
    padding: 2rem 2.5rem 1rem 4.25rem;
    background: var(--paper);
    color: var(--ink);
    font-family: var(--font-mono);
    font-size: 13px;
    line-height: 1.55;
    font-feature-settings: "tnum", "ss01", "calt";
    overflow: hidden;
  }

  /* Sprocket gutter — fanfold tractor-feed paper. Decorative. */
  .sprocket {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 1rem;
    width: 1.5rem;
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

  /* Masthead */
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
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--ink);
    line-height: 1;
  }
  .brand .department {
    font-family: var(--font-mono);
    font-size: 0.875rem;
    color: var(--ink-faint);
    letter-spacing: 0.12em;
    text-transform: lowercase;
  }
  .meta {
    display: inline-flex;
    align-items: center;
    gap: 1.1rem;
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
  .meta .val {
    color: var(--ink);
    font-weight: 500;
  }
  .meta .live {
    align-items: center;
    gap: 0.45rem;
    padding: 0.2rem 0.55rem;
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
  }
  .meta .live.hot .lbl {
    color: var(--ink);
  }

  .rule {
    height: 1px;
    border: 0;
    background: var(--rule);
    margin: 1rem 0 0.75rem;
  }

  /* Ticker — perforated baseline (sprocket-hole pattern), continuous slide */
  .ticker {
    overflow: hidden;
    height: 1.75rem;
    border-block: 1px solid var(--rule);
    padding: 0.3rem 0;
    background:
      repeating-linear-gradient(
        90deg,
        transparent 0,
        transparent 22px,
        color-mix(in srgb, var(--rule) 60%, transparent) 22px,
        color-mix(in srgb, var(--rule) 60%, transparent) 23px
      );
    background-size: 100% 4px;
    background-position: 0 100%;
    background-repeat: no-repeat;
  }
  .ticker-track {
    display: inline-flex;
    gap: 1.25rem;
    white-space: nowrap;
    padding-left: 100%;
    animation: ticker 36s linear infinite;
    font-size: 0.8125rem;
    color: var(--ink-soft);
  }
  .ticker-item {
    display: inline-flex;
    gap: 0.6rem;
    align-items: baseline;
  }
  .ticker-item.idle {
    color: var(--ink-faint);
    font-style: italic;
  }
  .ticker-slug {
    color: var(--purple-deep);
  }
  .ticker-text {
    color: var(--ink);
  }
  .ticker-sep {
    color: var(--ink-faint);
  }
  @keyframes ticker {
    from { transform: translateX(0); }
    to { transform: translateX(-100%); }
  }

  /* Feed */
  .feed {
    padding: 1rem 0 1.5rem;
    overflow-y: auto;
  }
  .awaiting {
    color: var(--ink-faint);
    font-size: 0.875rem;
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
  .dispatch:first-child {
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

  /* Caret — telex keystrike. Hidden when the line is fully typed. */
  .caret {
    display: inline-block;
    width: 0.5ch;
    margin-left: 0.05ch;
    color: var(--peach);
    animation: keystrike 0.18s steps(1) infinite;
  }
  .caret.done {
    opacity: 0;
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

  /* Footer status strip — operator + audio meter + counters */
  .strip {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 1.5rem;
    align-items: center;
    padding: 0.6rem 0 0;
    border-top: 1px solid var(--rule);
    font-size: 0.75rem;
  }
  .op {
    display: inline-flex;
    gap: 0.5rem;
    align-items: baseline;
  }
  .op-label {
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  .op-state {
    color: var(--purple-deep);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .meter {
    display: inline-flex;
    gap: 2px;
    height: 1rem;
    align-items: end;
    justify-self: center;
  }
  .meter .bar {
    width: 4px;
    height: calc(40% + (var(--i) * 4%));
    background: var(--rule);
    transition: background 80ms linear, transform 80ms linear;
    transform-origin: bottom;
  }
  .meter .bar.lit {
    background: var(--peach-deep);
    transform: scaleY(1.05);
  }
  .counters {
    display: inline-flex;
    gap: 1.25rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .counters .lbl {
    color: var(--ink-faint);
    margin-right: 0.4rem;
  }
  .counters .val {
    color: var(--ink);
    font-weight: 600;
  }

  @media (prefers-reduced-motion: reduce) {
    .ticker-track { animation: none; transform: translateX(0); }
    .blink, .caret { animation: none; }
    .thirty { animation: none; }
    .dispatch { animation: none; }
    .meter .bar { transition: none; transform: none; }
  }
</style>
