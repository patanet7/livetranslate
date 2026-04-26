<!--
  Editorial top status strip (D3.2).

  Reads as a magazine "running head": current location on the left,
  technical status on the right. Replaces shadcn Badge pills with
  small-caps text + warm earth-tone status pips.
-->
<script lang="ts">
  import { page } from "$app/stores";
  import { demoStore } from "$lib/stores/demo.svelte";
  import StatusIndicator from "./StatusIndicator.svelte";
  import { goto } from "$app/navigation";
  import MenuIcon from "@lucide/svelte/icons/menu";
  import ChevronDownIcon from "@lucide/svelte/icons/chevron-down";

  interface Props {
    health: "healthy" | "degraded" | "down" | "unknown";
    onMenuToggle?: () => void;
  }

  let { health, onMenuToggle }: Props = $props();

  let apiConnected = $derived(health === "healthy" || health === "degraded");
  let translationOnline = $derived(health === "healthy");

  // Derive a department-style label from the URL — used as the running-head text.
  // "/loopback/foo" → "loopback". Root → "overview".
  const departmentLabel = $derived.by(() => {
    const path = $page.url.pathname;
    if (path === "/") return "overview";
    return path.split("/").filter(Boolean)[0] ?? "overview";
  });

  let demoOpen = $state(false);
  let selectedMode = $state<"passthrough" | "pretranslated" | "replay">("replay");

  const demoModes = [
    { value: "replay" as const, label: "Replay Real Meeting" },
    { value: "passthrough" as const, label: "Live Passthrough" },
    { value: "pretranslated" as const, label: "Pre-translated ES" },
  ];

  async function launchDemo() {
    await demoStore.start(selectedMode);
    demoOpen = false;
    goto("/fireflies/live-feed");
  }

  async function stopDemo() {
    await demoStore.stop();
    demoOpen = false;
  }

  function selectMode(mode: typeof selectedMode) {
    selectedMode = mode;
    demoOpen = false;
  }

  function handleClickOutside(event: MouseEvent) {
    const target = event.target as HTMLElement;
    if (!target.closest("[data-demo-menu]")) demoOpen = false;
  }
</script>

<svelte:document onclick={demoOpen ? handleClickOutside : undefined} />

<header class="topbar">
  <!-- ── Left: hamburger + running-head department ─────── -->
  <div class="left">
    <button
      class="md:hidden inline-flex items-center justify-center size-9 text-ink-soft hover:text-ink transition-colors"
      aria-label="Toggle sidebar"
      onclick={onMenuToggle}
    >
      <MenuIcon class="size-5" />
    </button>

    <p class="running-head">
      issue 01 · vol 1
      <span class="sep">·</span>
      <span class="department">{departmentLabel}</span>
    </p>
  </div>

  <!-- ── Right: technical status ──────────────────────── -->
  <div class="right">
    <span class="status-cluster hidden lg:inline-flex">
      <StatusIndicator status={apiConnected ? "connected" : "disconnected"} label="api" />
      <StatusIndicator status={translationOnline ? "connected" : "down"} label="mt" />
    </span>

    <!-- Demo controls — small editorial dropdown -->
    <div class="relative demo-menu" data-demo-menu>
      <button
        type="button"
        class="demo-trigger"
        onclick={() => (demoOpen = !demoOpen)}
        aria-expanded={demoOpen}
      >
        <span class="dot" class:active={demoStore.active}></span>
        <span class="byline">demo</span>
        <ChevronDownIcon class="size-3 chevron {demoOpen ? 'open' : ''}" />
      </button>

      {#if demoOpen}
        <div class="demo-popover">
          <p class="eyebrow px-3 pt-2 pb-1.5">demo mode</p>
          {#each demoModes as mode}
            <button
              type="button"
              class="demo-option"
              class:selected={selectedMode === mode.value}
              onclick={() => selectMode(mode.value)}
            >
              <span class="radio" class:on={selectedMode === mode.value}></span>
              <span>{mode.label}</span>
            </button>
          {/each}
          <hr class="demo-divider" />
          <button
            type="button"
            class="demo-action"
            class:active={demoStore.active}
            disabled={demoStore.loading}
            onclick={demoStore.active ? stopDemo : launchDemo}
          >
            {demoStore.active ? "Stop Demo" : "Launch Demo"}
          </button>
        </div>
      {/if}
    </div>

    <span class="rule-vert"></span>

    <StatusIndicator status={health} label="services" />
  </div>
</header>

<style>
  .topbar {
    height: 2.75rem;
    padding: 0 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--rule);
    background: var(--paper);
  }

  .left, .right {
    display: flex;
    align-items: center;
    gap: 0.875rem;
  }

  /* Running-head text — magazine masthead-style */
  .running-head {
    margin: 0;
    color: var(--ink-faint);
  }
  .sep {
    margin: 0 0.4em;
    color: var(--rule);
  }
  .department {
    color: var(--ink-soft);
  }

  /* Cluster of status pips, separated by hairlines */
  .status-cluster {
    display: inline-flex;
    align-items: center;
    gap: 0.875rem;
    padding-right: 0.875rem;
    border-right: 1px solid var(--rule);
  }

  /* Demo trigger — minimal editorial pill */
  .demo-trigger {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.625rem;
    border: 1px solid var(--rule);
    border-radius: 9999px;
    background: var(--paper);
    color: var(--ink-soft);
    transition: color 160ms ease, border-color 160ms ease;
  }
  .demo-trigger:hover {
    color: var(--ink);
    border-color: var(--ink-soft);
  }
  .demo-trigger .dot {
    width: 0.4375rem;
    height: 0.4375rem;
    border-radius: 9999px;
    background: var(--ink-faint);
  }
  .demo-trigger .dot.active {
    background: var(--peach-deep);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--peach-deep) 22%, transparent);
    animation: dot-pulse 2s ease-in-out infinite;
  }
  /* Popover */
  .demo-popover {
    position: absolute;
    right: 0;
    top: calc(100% + 0.5rem);
    z-index: 50;
    width: 14rem;
    padding: 0.25rem 0;
    background: var(--popover);
    border: 1px solid var(--rule);
    border-radius: 0.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.08), 0 1px 0 var(--rule);
  }

  .demo-option {
    display: flex;
    width: 100%;
    align-items: center;
    gap: 0.625rem;
    padding: 0.5rem 0.75rem;
    font-family: var(--font-body);
    font-size: 0.8125rem;
    text-align: left;
    color: var(--ink-soft);
    transition: background-color 120ms ease, color 120ms ease;
  }
  .demo-option:hover {
    background: var(--paper-cream);
    color: var(--ink);
  }
  .demo-option.selected {
    color: var(--ink);
    background: color-mix(in srgb, var(--peach) 12%, transparent);
  }
  .demo-option .radio {
    width: 0.625rem;
    height: 0.625rem;
    border-radius: 9999px;
    border: 1.5px solid var(--ink-faint);
    flex-shrink: 0;
  }
  .demo-option .radio.on {
    border-color: var(--peach-deep);
    background: var(--peach-deep);
    box-shadow: inset 0 0 0 2px var(--paper);
  }

  .demo-divider {
    border: none;
    border-top: 1px solid var(--rule);
    margin: 0.25rem 0;
  }

  .demo-action {
    display: block;
    width: calc(100% - 0.5rem);
    margin: 0.25rem 0.25rem;
    padding: 0.5rem 0.75rem;
    text-align: center;
    font-family: var(--font-display);
    font-variation-settings: "opsz" 14;
    font-feature-settings: "smcp", "c2sc";
    letter-spacing: 0.08em;
    text-transform: lowercase;
    font-size: 0.75rem;
    background: var(--ink);
    color: var(--paper);
    border-radius: 0.25rem;
    transition: background-color 160ms ease;
  }
  .demo-action:hover { background: color-mix(in srgb, var(--ink) 88%, var(--peach)); }
  .demo-action.active { background: var(--oxblood); color: var(--paper); }
  .demo-action.active:hover { background: color-mix(in srgb, var(--oxblood) 88%, var(--ink)); }
  .demo-action:disabled { opacity: 0.5; cursor: not-allowed; }

  .rule-vert {
    display: inline-block;
    width: 1px;
    height: 1.125rem;
    background: var(--rule);
  }

  @keyframes dot-pulse {
    0%, 100% { box-shadow: 0 0 0 2px color-mix(in srgb, var(--peach-deep) 22%, transparent); }
    50%      { box-shadow: 0 0 0 5px color-mix(in srgb, var(--peach-deep) 10%, transparent); }
  }

  @media (prefers-reduced-motion: reduce) {
    .demo-trigger .dot.active { animation: none; }
  }
</style>
