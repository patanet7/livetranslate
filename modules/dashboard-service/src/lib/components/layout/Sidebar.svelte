<!--
  Editorial sidebar — reads like a magazine table of contents (D3.1).

  - Wordmark in Fraunces display with the Earwyrm mini sitting next to it
  - Sections grouped under small-caps eyebrow labels
  - Active route: peach gutter-mark + serif body, no pill background
  - Sub-items render as indented marginalia rather than nested pills
  - Collapse toggle preserved
-->
<script lang="ts">
  import { page } from "$app/stores";
  import { APP_NAME } from "$lib/config";
  import { audioStore } from "$lib/stores/audio.svelte";
  import EarwyrmMini from "$lib/components/brand/EarwyrmMini.svelte";
  import HomeIcon from "@lucide/svelte/icons/house";
  import CalendarDaysIcon from "@lucide/svelte/icons/calendar-days";
  import MicIcon from "@lucide/svelte/icons/mic";
  import DatabaseIcon from "@lucide/svelte/icons/database";
  import LightbulbIcon from "@lucide/svelte/icons/lightbulb";
  import AudioWaveformIcon from "@lucide/svelte/icons/audio-waveform";
  import HeadphonesIcon from "@lucide/svelte/icons/headphones";
  import GlobeIcon from "@lucide/svelte/icons/globe";
  import SettingsIcon from "@lucide/svelte/icons/settings";
  import MessageSquareIcon from "@lucide/svelte/icons/message-square";
  import TerminalIcon from "@lucide/svelte/icons/terminal";
  import PanelLeftCloseIcon from "@lucide/svelte/icons/panel-left-close";
  import PanelLeftOpenIcon from "@lucide/svelte/icons/panel-left-open";
  import type { Component } from "svelte";

  interface Props {
    open?: boolean;
    collapsed?: boolean;
    onclose?: () => void;
    oncollapse?: (collapsed: boolean) => void;
  }

  let { open = false, collapsed = false, onclose, oncollapse }: Props = $props();

  interface NavChild { label: string; href: string; }
  interface NavItem { label: string; href: string; icon: Component; children?: NavChild[]; }
  interface NavSection { label: string; items: NavItem[]; }

  const navSections: NavSection[] = [
    {
      label: "overview",
      items: [
        { label: "Dashboard", href: "/", icon: HomeIcon },
      ],
    },
    {
      label: "the loop",
      items: [
        { label: "Loopback", href: "/loopback", icon: HeadphonesIcon },
        { label: "Meetings", href: "/meetings", icon: CalendarDaysIcon },
        { label: "Sessions", href: "/sessions", icon: TerminalIcon },
        {
          label: "Fireflies",
          href: "/fireflies",
          icon: MicIcon,
          children: [
            { label: "Connect", href: "/fireflies" },
            { label: "Live Feed", href: "/fireflies/live-feed" },
            { label: "Glossary", href: "/fireflies/glossary" },
          ],
        },
      ],
    },
    {
      label: "the workshop",
      items: [
        {
          label: "Translation",
          href: "/translation",
          icon: GlobeIcon,
          children: [
            { label: "Test Bench", href: "/translation/test" },
            { label: "Config", href: "/config/translation" },
          ],
        },
        { label: "Intelligence", href: "/intelligence", icon: LightbulbIcon },
        { label: "Diarization", href: "/diarization", icon: AudioWaveformIcon },
        { label: "Data & Logs", href: "/data", icon: DatabaseIcon },
        { label: "Chat", href: "/chat", icon: MessageSquareIcon },
      ],
    },
    {
      label: "the desk",
      items: [
        {
          label: "Config",
          href: "/config",
          icon: SettingsIcon,
          children: [
            { label: "Connections", href: "/config/connections" },
            { label: "Audio", href: "/config/audio" },
            { label: "System", href: "/config/system" },
            { label: "Settings", href: "/config/settings" },
          ],
        },
      ],
    },
  ];

  function isActive(href: string): boolean {
    const pathname = $page.url.pathname;
    if (href === "/") return pathname === "/";
    return pathname === href || pathname.startsWith(href + "/");
  }

  function isParentActive(item: NavItem): boolean {
    if (isActive(item.href)) return true;
    if (item.children) return item.children.some((c) => isActive(c.href));
    return false;
  }

  function isChildExact(href: string): boolean {
    return $page.url.pathname === href;
  }

  function handleNavClick() {
    onclose?.();
  }

  // Earwyrm mini state — peach pulsing ring when audio is flowing
  const earwyrmState = $derived<"live" | "idle" | "offline">(
    audioStore.isFlowing ? "live" : "idle",
  );
</script>

<!-- Mobile backdrop -->
{#if open}
  <button
    class="fixed inset-0 z-40 bg-ink/40 md:hidden"
    aria-label="Close sidebar"
    onclick={onclose}
    tabindex="-1"
  ></button>
{/if}

<aside
  class="sidebar bg-sidebar text-sidebar-foreground flex flex-col h-full transition-[width,transform] duration-200 ease-in-out
    fixed inset-y-0 left-0 z-50
    md:relative md:translate-x-0 md:z-auto
    {open ? 'translate-x-0' : '-translate-x-full'}
    {collapsed ? 'w-14' : 'w-60'}"
>
  <!-- ── Wordmark + mascot ────────────────────────────── -->
  <div class="px-5 pt-7 pb-5 flex items-center gap-3 border-b border-rule {collapsed ? 'justify-center px-0' : ''}">
    {#if collapsed}
      <EarwyrmMini size={28} state={earwyrmState} audioRms={audioStore.rms} title={APP_NAME} />
    {:else}
      <EarwyrmMini size={32} state={earwyrmState} audioRms={audioStore.rms} title="{APP_NAME} — Earwyrm" />
      <h1 class="font-display text-2xl leading-none m-0" style="font-variation-settings: 'opsz' 36, 'SOFT' 40, 'WONK' 1; letter-spacing: -0.02em;">
        Live<span class="text-purple">Translate</span>
      </h1>
    {/if}
  </div>

  <!-- ── Sections ─────────────────────────────────────── -->
  <nav class="flex-1 px-3 py-4 overflow-y-auto">
    {#each navSections as section, sIdx (section.label)}
      {#if !collapsed}
        <p class="eyebrow mt-{sIdx === 0 ? 0 : 5} mb-2 px-2">{section.label}</p>
      {:else if sIdx > 0}
        <hr class="my-3 border-t border-rule" />
      {/if}

      <ul class="space-y-px">
        {#each section.items as item (item.href)}
          {@const parentActive = isParentActive(item)}
          {@const Icon = item.icon}
          <li class="relative">
            <!-- Peach gutter-mark on active items -->
            {#if parentActive && !collapsed}
              <span aria-hidden="true" class="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-peach rounded-r-full"></span>
            {/if}
            <a
              href={item.children ? item.children[0].href : item.href}
              aria-current={parentActive ? "page" : undefined}
              title={collapsed ? item.label : undefined}
              class="group flex items-center transition-colors text-sm
                {collapsed
                  ? 'justify-center px-2 py-2'
                  : 'gap-3 pl-4 pr-2 py-2'}
                {parentActive ? 'text-ink' : 'text-ink-soft hover:text-ink'}"
              onclick={handleNavClick}
            >
              <Icon class="size-4 shrink-0 {parentActive ? 'text-purple' : 'opacity-70'}" />
              {#if !collapsed}
                <span class={parentActive ? "font-display tracking-tight" : ""} style={parentActive ? "font-variation-settings: 'opsz' 24, 'SOFT' 30; font-weight: 500;" : ""}>
                  {item.label}
                </span>
              {/if}
            </a>

            {#if !collapsed && item.children && parentActive}
              <ul class="mt-0.5 mb-2 ml-7 pl-3 border-l border-rule space-y-px">
                {#each item.children as child, cIdx (child.href)}
                  <li class="flex items-baseline gap-2">
                    <span class="font-mono text-[10px] tabular-nums text-ink-faint w-4">{String(cIdx + 1).padStart(2, "0")}</span>
                    <a
                      href={child.href}
                      class="block py-1 text-xs transition-colors
                        {isChildExact(child.href)
                          ? 'text-ink font-medium'
                          : 'text-ink-soft hover:text-ink'}"
                      onclick={handleNavClick}
                    >
                      {child.label}
                    </a>
                  </li>
                {/each}
              </ul>
            {/if}
          </li>
        {/each}
      </ul>
    {/each}
  </nav>

  <!-- ── Collapse control ─────────────────────────────── -->
  <div class="border-t border-rule {collapsed ? 'p-2' : 'px-3 py-3'}">
    <button
      type="button"
      class="hidden md:flex items-center justify-center w-full py-1.5 rounded text-ink-faint hover:text-ink-soft hover:bg-paper-cream transition-colors"
      title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      onclick={() => oncollapse?.(!collapsed)}
    >
      {#if collapsed}
        <PanelLeftOpenIcon class="size-4" />
      {:else}
        <PanelLeftCloseIcon class="size-4" />
      {/if}
    </button>
  </div>
</aside>

<style>
  /* The sidebar sits ON paper; introduce a fine right-edge rule to read
     as a column gutter rather than a hard panel split. */
  aside.sidebar {
    border-right: 1px solid var(--rule);
  }
</style>
