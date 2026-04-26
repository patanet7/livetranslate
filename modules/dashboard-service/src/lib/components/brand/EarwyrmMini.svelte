<!--
  EarwyrmMini — circular badge variant of the mascot.

  Used for:
  - Connection-status pip in the top status strip (D3.2)
  - Bottom-corner live indicator on the loopback page (D4.6)
  - Sidebar logo slot (D3.1)

  Displays just the head + headphones in a circular crop. Same audio-RMS
  hook as the full mascot so its ear cups pulse with live audio.

  Props:
    size      — pixel size of the circular badge (default 32).
    audioRms  — 0..1, drives ear-cup pulse + glow.
    state     — 'idle' | 'live' | 'offline'. Frames the ring color.
    title     — for screen readers + tooltip.
-->
<script lang="ts">
  type State = "idle" | "live" | "offline";
  type Props = {
    size?: number;
    audioRms?: number;
    state?: State;
    title?: string;
  };

  let {
    size = 32,
    audioRms = 0,
    state = "idle",
    title = "Earwyrm",
  }: Props = $props();

  const earScale = $derived(0.92 + Math.min(1, Math.max(0, audioRms)) * 0.18);
  const earGlow = $derived(Math.min(1, audioRms * 1.8));
</script>

<span
  class="earwyrm-mini"
  data-state={state}
  style="--size: {size}px; --ear-scale: {earScale}; --ear-glow: {earGlow};"
  role="img"
  aria-label={title}
  title={title}
>
  <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <!-- Ring — color reflects state (live=peach, idle=rule, offline=oxblood) -->
    <circle cx="50" cy="50" r="48" class="ring" fill="none" stroke-width="2" />

    <!-- Inner head — peach -->
    <circle cx="50" cy="56" r="34" fill="var(--peach, #e8b4a0)" />

    <!-- Headphone band -->
    <path
      d="M 22 48 C 30 22, 70 22, 78 48"
      fill="none"
      stroke="var(--purple, #7c6cf0)"
      stroke-width="6"
      stroke-linecap="round"
    />

    <!-- Ear cups -->
    <g class="ear ear-left">
      <circle cx="22" cy="54" r="11" fill="var(--purple, #7c6cf0)" />
    </g>
    <g class="ear ear-right">
      <circle cx="78" cy="54" r="11" fill="var(--purple, #7c6cf0)" />
    </g>

    <!-- Eyes — chunky black ovals; tiny highlight -->
    <ellipse cx="40" cy="58" rx="4.5" ry="7" fill="var(--ink, #181410)" />
    <ellipse cx="60" cy="58" rx="4.5" ry="7" fill="var(--ink, #181410)" />
    <circle cx="38.5" cy="55" r="1.4" fill="var(--paper, #FAF6EE)" opacity="0.9" />
    <circle cx="58.5" cy="55" r="1.4" fill="var(--paper, #FAF6EE)" opacity="0.9" />
  </svg>
</span>

<style>
  .earwyrm-mini {
    display: inline-block;
    width: var(--size);
    height: var(--size);
    line-height: 0;
    flex-shrink: 0;
  }

  .earwyrm-mini svg {
    width: 100%;
    height: 100%;
    display: block;
    overflow: visible;
  }

  /* Ring color reflects connection state */
  .ring {
    stroke: var(--rule);
    transition: stroke 200ms ease;
  }
  [data-state="live"] .ring {
    stroke: var(--peach);
    /* Subtle pulse on the ring when live, regardless of audio activity */
    animation: mini-ring-pulse 2.4s ease-in-out infinite;
  }
  [data-state="offline"] .ring {
    stroke: var(--oxblood);
    stroke-dasharray: 4 3;
  }

  @keyframes mini-ring-pulse {
    0%, 100% { stroke-opacity: 0.55; }
    50%      { stroke-opacity: 1; }
  }

  /* Audio-reactive ear cups */
  .ear {
    transform-origin: center;
    transform-box: fill-box;
    transform: scale(var(--ear-scale, 1));
    transition: transform 80ms ease-out;
  }

  @media (prefers-reduced-motion: reduce) {
    [data-state="live"] .ring { animation: none; }
    .ear { transition: transform 240ms ease-out; }
  }
</style>
