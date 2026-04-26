<!--
  Status pip — editorial restyle (D3.2).

  Replaces Material primary colors with the warm earth-tone palette so
  the status indicators sit naturally inside the editorial chrome.
  Healthy = sage (calm green). Degraded = ochre. Down = oxblood.
  Connecting / reconnecting pulses gently. Disconnected reads as faint ink.
-->
<script lang="ts">
  interface Props {
    status:
      | "healthy" | "degraded" | "down" | "unknown"
      | "connected" | "connecting" | "reconnecting"
      | "disconnected" | "error";
    label?: string;
    size?: "sm" | "md";
  }

  let { status, label, size = "sm" }: Props = $props();

  // Map states to editorial-palette CSS vars.
  const COLOR_VAR: Record<string, string> = {
    healthy: "var(--sage)",
    connected: "var(--sage)",
    degraded: "var(--ochre)",
    connecting: "var(--ochre)",
    reconnecting: "var(--ochre)",
    down: "var(--oxblood)",
    error: "var(--oxblood)",
    disconnected: "var(--ink-faint)",
    unknown: "var(--ink-faint)",
  };

  const PULSING = new Set(["connecting", "reconnecting"]);

  const dotSize = $derived(size === "md" ? "0.5rem" : "0.4375rem");
</script>

<span class="status-pip" class:pulsing={PULSING.has(status)}>
  <span
    aria-hidden="true"
    class="dot"
    style="--dot-color: {COLOR_VAR[status] ?? 'var(--ink-faint)'}; --dot-size: {dotSize};"
  ></span>
  {#if label}
    <span class="byline label-text">{label}</span>
  {/if}
</span>

<style>
  .status-pip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
  }

  .dot {
    width: var(--dot-size);
    height: var(--dot-size);
    border-radius: 9999px;
    background: var(--dot-color);
    flex-shrink: 0;
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--dot-color) 18%, transparent);
  }

  .pulsing .dot {
    animation: pip-pulse 1.6s ease-in-out infinite;
  }

  .label-text {
    font-size: 0.6875rem;
    color: var(--ink-soft);
  }

  @keyframes pip-pulse {
    0%, 100% { box-shadow: 0 0 0 2px color-mix(in srgb, var(--dot-color) 18%, transparent); }
    50%      { box-shadow: 0 0 0 5px color-mix(in srgb, var(--dot-color) 8%, transparent); }
  }

  @media (prefers-reduced-motion: reduce) {
    .pulsing .dot { animation: none; }
  }
</style>
