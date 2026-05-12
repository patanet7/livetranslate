<!--
  /dev/stress — offline UI stress harness for the caption store.

  Drives synthetic segments + translation chunks through the real captionStore
  while measuring frame-budget violations. Renders the actual loopback view
  components on the same page so the test exercises real Svelte reactivity,
  not just store-state mutation.

  No backend needed. Open in `npm run dev` and click "Start".

  Use cases:
    - Reproduce the loopback lockup deterministically.
    - A/B current vs proposed caption-store fixes on identical input.
    - Compare display-mode performance side-by-side (Split vs Wire vs Transcript).
-->
<script lang="ts">
  import { captionStore, type DisplayMode } from '$lib/stores/caption.svelte';
  import { runStress, DEFAULT_STRESS_CONFIG, type StressMetrics, type StressHandle } from '$lib/loopback/stress-driver';
  import SplitView from '$lib/components/loopback/SplitView.svelte';
  import TranscriptView from '$lib/components/loopback/TranscriptView.svelte';
  import InterpreterView from '$lib/components/loopback/InterpreterView.svelte';
  import WireView from '$lib/components/loopback/WireView.svelte';

  // Local config (bound to inputs)
  let totalSegments = $state(DEFAULT_STRESS_CONFIG.totalSegments);
  let chunksPerTranslation = $state(DEFAULT_STRESS_CONFIG.chunksPerTranslation);
  let segmentIntervalMs = $state(DEFAULT_STRESS_CONFIG.segmentIntervalMs);
  let chunkIntervalMs = $state(DEFAULT_STRESS_CONFIG.chunkIntervalMs);
  let slowFrameThresholdMs = $state(DEFAULT_STRESS_CONFIG.slowFrameThresholdMs);

  // Live state
  let handle = $state<StressHandle | null>(null);
  let isRunning = $derived(handle !== null);
  let metrics = $state<StressMetrics | null>(null);
  let liveCounts = $state({ segments: 0, chunks: 0, translations: 0 });
  let liveTimer: ReturnType<typeof setInterval> | null = null;

  // Display-mode picker (drives which view component renders below)
  const MODES: DisplayMode[] = ['split', 'wire', 'transcript', 'interpreter'];

  async function start() {
    if (handle) return;
    captionStore.clear?.();
    metrics = null;
    liveCounts = { segments: 0, chunks: 0, translations: 0 };

    handle = runStress({
      totalSegments,
      chunksPerTranslation,
      segmentIntervalMs,
      chunkIntervalMs,
      slowFrameThresholdMs,
    });

    liveTimer = setInterval(() => {
      if (handle) liveCounts = { ...handle.live };
    }, 250);

    metrics = await handle.done;
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    handle = null;
  }

  async function stop() {
    if (!handle) return;
    metrics = await handle.stop();
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    handle = null;
  }

  function setMode(m: DisplayMode) {
    captionStore.displayMode = m;
  }

  function fmt(n: number, digits = 1): string {
    return n.toFixed(digits);
  }

  function verdict(m: StressMetrics): { label: string; color: string } {
    if (m.maxFrameMs > 250) return { label: 'LOCKED', color: 'var(--oxblood, #8a2a2a)' };
    if (m.slowFrames > m.frameCount * 0.1) return { label: 'JANK', color: 'var(--peach-deep, #d4783d)' };
    if (m.maxFrameMs > m.slowFrameThresholdMs) return { label: 'TIGHT', color: 'var(--ochre, #c5a04a)' };
    return { label: 'SMOOTH', color: 'var(--sage, #6a8a5a)' };
  }
</script>

<svelte:head><title>UI Stress — LiveTranslate Dev</title></svelte:head>

<div class="page">
  <header>
    <h1>Caption Store Stress Harness</h1>
    <p class="sub">
      Synthetic segments + streaming chunks → real captionStore → real view components.
      No backend. Measures frame-budget violations to reproduce / verify-fix the loopback lockup.
    </p>
  </header>

  <section class="controls">
    <div class="grid">
      <label>Total segments
        <input type="number" min="10" max="5000" bind:value={totalSegments} disabled={isRunning} />
      </label>
      <label>Chunks / translation
        <input type="number" min="1" max="100" bind:value={chunksPerTranslation} disabled={isRunning} />
      </label>
      <label>Segment interval (ms)
        <input type="number" min="0" max="2000" bind:value={segmentIntervalMs} disabled={isRunning} />
      </label>
      <label>Chunk interval (ms)
        <input type="number" min="0" max="200" bind:value={chunkIntervalMs} disabled={isRunning} />
      </label>
      <label>Slow-frame threshold (ms)
        <input type="number" min="16" max="500" bind:value={slowFrameThresholdMs} disabled={isRunning} />
      </label>
    </div>
    <div class="actions">
      {#if isRunning}
        <button onclick={stop} class="danger">Stop</button>
      {:else}
        <button onclick={start} class="primary">Start</button>
      {/if}
      <span class="mode-row">
        view:
        {#each MODES as m (m)}
          <button class:active={captionStore.displayMode === m} onclick={() => setMode(m)}>{m}</button>
        {/each}
      </span>
    </div>
  </section>

  <section class="metrics">
    {#if isRunning && handle}
      <div class="live">
        <span><b>{liveCounts.segments}</b> segments</span>
        <span><b>{liveCounts.chunks}</b> chunks</span>
        <span><b>{liveCounts.translations}</b> translations</span>
        <span class="running-dot">running…</span>
      </div>
    {:else if metrics}
      {@const v = verdict(metrics)}
      <div class="result" style="--accent: {v.color}">
        <div class="verdict">{v.label}</div>
        <table>
          <tbody>
            <tr><th>max frame</th><td><b>{fmt(metrics.maxFrameMs)} ms</b></td><th>frames observed</th><td>{metrics.frameCount}</td></tr>
            <tr><th>p95 frame</th><td>{fmt(metrics.p95FrameMs)} ms</td><th>slow frames (&gt;{metrics.slowFrameThresholdMs}ms)</th><td>{metrics.slowFrames}</td></tr>
            <tr><th>duration</th><td>{fmt(metrics.durationMs / 1000, 2)} s</td><th>avg fps</th><td>{fmt(metrics.frameCount / (metrics.durationMs / 1000))}</td></tr>
            <tr><th>segments</th><td>{metrics.segmentsInjected}</td><th>chunks / final</th><td>{metrics.chunksInjected} / {metrics.translationsInjected}</td></tr>
          </tbody>
        </table>
      </div>
    {:else}
      <p class="hint">Configure parameters above, then click Start. Watch the view below jank in real time; the metrics summarise at the end.</p>
    {/if}
  </section>

  <section class="view-host">
    {#if captionStore.displayMode === 'split'}
      <SplitView />
    {:else if captionStore.displayMode === 'wire'}
      <WireView />
    {:else if captionStore.displayMode === 'transcript'}
      <TranscriptView />
    {:else if captionStore.displayMode === 'interpreter'}
      <InterpreterView />
    {/if}
  </section>
</div>

<style>
  .page {
    display: grid;
    grid-template-rows: auto auto auto 1fr;
    height: 100vh;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
    background: var(--paper, #f7f4ee);
    color: var(--ink, #1f1b16);
    font-family: var(--font-body, system-ui);
  }
  header h1 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  header .sub {
    margin: 0.25rem 0 0;
    font-size: 0.875rem;
    color: var(--ink-faint, #7a746b);
  }
  .controls {
    border: 1px solid var(--rule, #d6cfc1);
    padding: 0.75rem;
    border-radius: 4px;
    background: color-mix(in srgb, var(--paper) 50%, white);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.5rem 1rem;
  }
  label {
    display: flex;
    flex-direction: column;
    font-size: 0.75rem;
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  input[type="number"] {
    margin-top: 0.25rem;
    padding: 0.35rem 0.5rem;
    border: 1px solid var(--rule);
    border-radius: 3px;
    font-family: var(--font-mono, ui-monospace);
    background: white;
  }
  .actions {
    margin-top: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
  }
  button {
    padding: 0.4rem 0.85rem;
    border: 1px solid var(--rule);
    background: white;
    border-radius: 3px;
    font: inherit;
    cursor: pointer;
  }
  button.primary { background: var(--ink); color: var(--paper); border-color: var(--ink); }
  button.danger  { background: var(--oxblood, #8a2a2a); color: white; border-color: var(--oxblood, #8a2a2a); }
  button.active  { background: var(--ink); color: var(--paper); border-color: var(--ink); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .mode-row {
    margin-left: 1rem;
    display: inline-flex;
    gap: 0.35rem;
    align-items: center;
    color: var(--ink-faint);
    font-size: 0.8125rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .metrics .live {
    display: inline-flex;
    gap: 1.5rem;
    padding: 0.5rem 0.75rem;
    background: color-mix(in srgb, var(--peach-deep, #d4783d) 12%, white);
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 0.8125rem;
  }
  .running-dot::before {
    content: "●";
    color: var(--peach-deep, #d4783d);
    margin-right: 0.4rem;
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse { 50% { opacity: 0.35; } }
  .result {
    border: 1px solid var(--rule);
    border-left: 6px solid var(--accent);
    padding: 0.6rem 0.85rem;
    border-radius: 3px;
    background: white;
  }
  .verdict {
    font-size: 1.125rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8125rem;
    font-family: var(--font-mono);
  }
  table th {
    text-align: left;
    color: var(--ink-faint);
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.7rem;
    padding: 0.15rem 0.6rem 0.15rem 0;
    width: 18%;
  }
  table td {
    padding: 0.15rem 1.25rem 0.15rem 0;
  }
  .hint {
    color: var(--ink-faint);
    font-size: 0.875rem;
    margin: 0;
  }
  .view-host {
    border: 1px solid var(--rule);
    border-radius: 4px;
    overflow: hidden;
    min-height: 0;
    background: var(--paper);
  }
</style>
