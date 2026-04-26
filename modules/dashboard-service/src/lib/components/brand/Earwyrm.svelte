<!--
  Earwyrm — the LiveTranslate mascot.

  A peach worm wearing purple headphones. Used as a hero illustration
  on the home page and as a wandering live-indicator on the loopback
  page. The mini variant lives in `EarwyrmMini.svelte`.

  Drives off the editorial palette via CSS vars (var(--peach), var(--purple))
  so it adapts automatically to evening edition.

  Props:
    size      — pixel height (default 240). Width derives from viewBox.
    idle      — when true (default), the worm has a slow breathing wobble.
    audioRms  — 0..1 audio amplitude. Drives ear-cup pulse + glow.
                Surface this from the loopback store (D2.3) to make the
                mascot react to live audio.
    listening — small head-tilt to convey "paying attention". Toggled by
                the loopback page when a new caption arrives.
-->
<script lang="ts">
  type Props = {
    size?: number;
    idle?: boolean;
    audioRms?: number;
    listening?: boolean;
    title?: string;
  };

  let {
    size = 240,
    idle = true,
    audioRms = 0,
    listening = false,
    title = "Earwyrm — listening",
  }: Props = $props();

  // Clamp + soften the audioRms so a dead-quiet room doesn't read 0
  // and a loud transient doesn't blow the ear cups out of frame.
  const earScale = $derived(0.92 + Math.min(1, Math.max(0, audioRms)) * 0.18);
  const earGlow = $derived(Math.min(1, audioRms * 1.8));
</script>

<svg
  class="earwyrm"
  class:idle
  class:listening
  width={size}
  height={size}
  viewBox="0 0 280 280"
  xmlns="http://www.w3.org/2000/svg"
  role="img"
  aria-label={title}
  style="--ear-scale: {earScale}; --ear-glow: {earGlow};"
>
  <title>{title}</title>

  <!-- Soft drop shadow under the worm — adds weight to the page -->
  <defs>
    <filter id="earwyrm-shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="2.5" />
      <feOffset dx="0" dy="2" result="offsetBlur" />
      <feComponentTransfer><feFuncA type="linear" slope="0.18" /></feComponentTransfer>
      <feMerge>
        <feMergeNode />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>

    <!-- Subtle radial highlight on the head — gives the peach tonality life -->
    <radialGradient id="head-shine" cx="40%" cy="35%" r="60%">
      <stop offset="0%" stop-color="rgb(255 255 255 / 0.35)" />
      <stop offset="60%" stop-color="rgb(255 255 255 / 0)" />
    </radialGradient>

    <!-- Glow filter for the ear cups when audio is active -->
    <filter id="ear-glow-filter" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>

  <g class="body-group" filter="url(#earwyrm-shadow)">
    <!--
      Worm body — a soft S-curve, drawn as a stroke so the body has
      consistent thickness. The path flows from the back of the head,
      down and around in a relaxed coil that suggests "settled in for
      a long listen" rather than "wormy".
    -->
    <path
      class="body"
      d="
        M 140 110
        C 195 130, 220 175, 195 210
        C 175 240, 130 240, 115 215
        C 100 190, 130 175, 160 195
        C 190 215, 200 245, 175 260
      "
      fill="none"
      stroke="var(--peach, #e8b4a0)"
      stroke-width="36"
      stroke-linecap="round"
      stroke-linejoin="round"
    />

    <!-- Head — slightly oval, sits forward on the body -->
    <ellipse
      class="head"
      cx="140"
      cy="105"
      rx="58"
      ry="50"
      fill="var(--peach, #e8b4a0)"
    />
    <ellipse
      cx="140"
      cy="105"
      rx="58"
      ry="50"
      fill="url(#head-shine)"
      pointer-events="none"
    />

    <!-- Headphone band — a slightly dipped arc, not a perfect curve -->
    <path
      class="band"
      d="M 91 92 C 110 60, 170 60, 189 92"
      fill="none"
      stroke="var(--purple, #7c6cf0)"
      stroke-width="9"
      stroke-linecap="round"
    />

    <!-- Left ear cup — pulses with audioRms via CSS var -->
    <g class="ear ear-left">
      <circle
        cx="91"
        cy="100"
        r="22"
        fill="var(--purple, #7c6cf0)"
      />
      <!-- Glow ring — opacity driven by --ear-glow -->
      <circle
        cx="91"
        cy="100"
        r="22"
        fill="none"
        stroke="var(--purple, #7c6cf0)"
        stroke-width="3"
        class="ear-glow-ring"
        filter="url(#ear-glow-filter)"
      />
    </g>

    <!-- Right ear cup -->
    <g class="ear ear-right">
      <circle
        cx="189"
        cy="100"
        r="22"
        fill="var(--purple, #7c6cf0)"
      />
      <circle
        cx="189"
        cy="100"
        r="22"
        fill="none"
        stroke="var(--purple, #7c6cf0)"
        stroke-width="3"
        class="ear-glow-ring"
        filter="url(#ear-glow-filter)"
      />
    </g>

    <!-- Eyes — tall ovals with tiny catchlights for liveliness -->
    <g class="eye eye-left">
      <ellipse cx="124" cy="108" rx="9" ry="14" fill="var(--ink, #181410)" />
      <ellipse cx="121" cy="103" rx="2.4" ry="3" fill="var(--paper, #FAF6EE)" opacity="0.9" />
    </g>
    <g class="eye eye-right">
      <ellipse cx="156" cy="108" rx="9" ry="14" fill="var(--ink, #181410)" />
      <ellipse cx="153" cy="103" rx="2.4" ry="3" fill="var(--paper, #FAF6EE)" opacity="0.9" />
    </g>

    <!-- Tiny mouth — barely there, just enough to suggest contentment -->
    <path
      class="mouth"
      d="M 134 132 Q 140 135 146 132"
      fill="none"
      stroke="var(--ink, #181410)"
      stroke-width="1.6"
      stroke-linecap="round"
      opacity="0.55"
    />
  </g>
</svg>

<style>
  .earwyrm {
    display: block;
    overflow: visible;
  }

  /* ── Idle: a slow body breathe + occasional eye blink ───────── */
  .earwyrm.idle .body-group {
    transform-origin: 140px 200px;
    animation: earwyrm-breathe 4.8s ease-in-out infinite;
  }

  .earwyrm.idle .eye-left ellipse:first-child,
  .earwyrm.idle .eye-right ellipse:first-child {
    transform-origin: center;
    transform-box: fill-box;
    animation: earwyrm-blink 7.2s ease-in-out infinite;
  }

  @keyframes earwyrm-breathe {
    0%, 100% {
      transform: translateY(0) scale(1);
    }
    50% {
      transform: translateY(-1.5px) scale(1.012);
    }
  }

  @keyframes earwyrm-blink {
    0%, 91%, 96%, 100% { transform: scaleY(1); }
    93% { transform: scaleY(0.08); }
  }

  /* ── Listening tilt: a small head-cock when listening prop is true ── */
  .earwyrm.listening .head,
  .earwyrm.listening .band,
  .earwyrm.listening .ear,
  .earwyrm.listening .eye,
  .earwyrm.listening .mouth {
    transform-origin: 140px 110px;
    transform-box: fill-box;
    animation: earwyrm-tilt 1.2s ease-out;
  }

  @keyframes earwyrm-tilt {
    0% { transform: rotate(0deg); }
    40% { transform: rotate(-4deg); }
    70% { transform: rotate(-2deg); }
    100% { transform: rotate(0deg); }
  }

  /* ── Audio-reactive ear cups ──────────────────────────────────
     The --ear-scale CSS var is set inline (computed from audioRms)
     and drives a subtle scale on the ear cup containers. The glow
     ring's opacity is driven by --ear-glow. */
  .ear {
    transform-origin: center;
    transform-box: fill-box;
    transform: scale(var(--ear-scale, 1));
    transition: transform 80ms ease-out;
  }

  .ear-glow-ring {
    opacity: var(--ear-glow, 0);
    transition: opacity 120ms ease-out;
  }

  /* Reduced-motion: kill the ambient animations entirely; keep the
     audio-reactive transforms but smooth them. */
  @media (prefers-reduced-motion: reduce) {
    .earwyrm.idle .body-group,
    .earwyrm.idle .eye-left ellipse:first-child,
    .earwyrm.idle .eye-right ellipse:first-child,
    .earwyrm.listening .head {
      animation: none;
    }
    .ear {
      transition: transform 240ms ease-out;
    }
  }
</style>
