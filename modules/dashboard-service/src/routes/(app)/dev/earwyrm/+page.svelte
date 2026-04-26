<!--
  Dev-only preview for the Earwyrm mascot. Not linked from the main nav.
  Used during D2.x to verify the component visually + iterate on the SVG
  + animations. Will be removed (or absorbed into a Storybook-like surface)
  before D7 ships.
-->
<script lang="ts">
  import Earwyrm from "$lib/components/brand/Earwyrm.svelte";
  import EarwyrmMini from "$lib/components/brand/EarwyrmMini.svelte";

  // Fake audio RMS that bounces — lets us see the ear-cup pulse without
  // having to plug a microphone in.
  let rms = $state(0);
  let listening = $state(false);

  $effect(() => {
    const id = setInterval(() => {
      // Lazy random walk between 0..1 with bias toward "talking"
      rms = Math.max(0, Math.min(1, rms + (Math.random() - 0.45) * 0.35));
    }, 150);
    return () => clearInterval(id);
  });

  function pingListening() {
    listening = true;
    // Re-trigger via toggle so the CSS animation can fire again
    setTimeout(() => (listening = false), 1300);
  }
</script>

<div class="px-12 py-16 mx-auto max-w-6xl">
  <header class="mb-16">
    <p class="running-head mb-3">brand · component preview</p>
    <h1 class="font-display text-6xl mb-3">Earwyrm</h1>
    <p class="kicker text-lg">A peach worm with purple headphones — and a real job.</p>
  </header>

  <section class="grid grid-cols-2 gap-12 mb-16">
    <div>
      <p class="eyebrow mb-4">Hero — Earwyrm @ 240px</p>
      <div class="rounded p-8" style="background: var(--paper-cream);">
        <Earwyrm size={240} audioRms={rms} {listening} />
      </div>
      <p class="mt-3 font-mono text-xs" style="color: var(--ink-soft);">
        rms = <span class="tabular-nums">{rms.toFixed(2)}</span>
      </p>
      <button
        type="button"
        onclick={pingListening}
        class="mt-3 px-3 py-1.5 byline text-xs"
        style="background: var(--ink); color: var(--paper);"
      >
        ping listening tilt
      </button>
    </div>

    <div>
      <p class="eyebrow mb-4">Mini — connection-status badges</p>
      <div class="rounded p-8 flex flex-col gap-6" style="background: var(--paper-cream);">
        <div class="flex items-center gap-4">
          <EarwyrmMini size={36} state="live" audioRms={rms} title="live capture" />
          <div>
            <p class="byline text-sm">live</p>
            <p class="text-xs" style="color: var(--ink-soft);">capturing audio · ear cups pulse with input</p>
          </div>
        </div>
        <div class="flex items-center gap-4">
          <EarwyrmMini size={36} state="idle" title="idle" />
          <div>
            <p class="byline text-sm">idle</p>
            <p class="text-xs" style="color: var(--ink-soft);">connected, waiting</p>
          </div>
        </div>
        <div class="flex items-center gap-4">
          <EarwyrmMini size={36} state="offline" title="offline" />
          <div>
            <p class="byline text-sm">offline</p>
            <p class="text-xs" style="color: var(--ink-soft);">disconnected · dashed ring</p>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="mb-16">
    <p class="eyebrow mb-4">Scale ladder</p>
    <div class="rounded p-8 flex items-end gap-8 flex-wrap" style="background: var(--paper-cream);">
      <EarwyrmMini size={20} state="live" audioRms={rms} />
      <EarwyrmMini size={28} state="live" audioRms={rms} />
      <EarwyrmMini size={40} state="live" audioRms={rms} />
      <EarwyrmMini size={56} state="live" audioRms={rms} />
      <EarwyrmMini size={80} state="live" audioRms={rms} />
      <Earwyrm size={120} audioRms={rms} />
    </div>
  </section>

  <footer class="border-t border-rule pt-6">
    <p class="running-head">design / d2 component verification</p>
  </footer>
</div>
