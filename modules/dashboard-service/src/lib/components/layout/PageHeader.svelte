<!--
  Editorial page header (D5).

  Title in Fraunces with display optical-size + WONK; description as a
  Newsreader italic kicker. Actions slot sits right-aligned. Below the
  block is a 1px ink rule that doubles as the page's masthead under-line.

  Optional `eyebrow` prop renders a small-caps department label above
  the title — the magazine-style "department / department / page".
-->
<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    title: string;
    eyebrow?: string;
    description?: string;
    actions?: Snippet;
  }

  let { title, eyebrow, description, actions }: Props = $props();
</script>

<header class="page-header">
  <div class="page-header-row">
    <div class="page-header-left">
      {#if eyebrow}
        <p class="eyebrow" data-reveal="0">{eyebrow}</p>
      {/if}
      <h1 class="font-display title" data-reveal="1">{title}</h1>
      {#if description}
        <p class="kicker description" data-reveal="2">{description}</p>
      {/if}
    </div>
    {#if actions}
      <div class="page-header-actions" data-reveal="3">
        {@render actions()}
      </div>
    {/if}
  </div>
  <hr class="masthead-rule" data-rule-draw />
</header>

<style>
  .page-header {
    margin-bottom: 1.75rem;
  }

  .page-header-row {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.5rem;
    padding-bottom: 1.25rem;
  }

  .masthead-rule {
    /* The typesetter's rule beneath the masthead. `data-rule-draw` in
       app.css wipes it in left-to-right after the title appears. */
    border: 0;
    height: 1px;
    background: var(--rule);
    margin: 0;
  }

  .page-header-left { flex: 1; min-width: 0; }

  .title {
    margin: 0;
    font-size: 2.75rem;
    line-height: 1;
    color: var(--ink);
    font-variation-settings: "opsz" 96, "SOFT" 50, "WONK" 1;
    letter-spacing: -0.025em;
  }

  .description {
    margin: 0.5rem 0 0;
    font-size: 0.9375rem;
    max-width: 50rem;
  }

  .page-header-actions {
    display: flex;
    align-items: center;
    gap: 0.625rem;
    flex-shrink: 0;
  }

  @media (max-width: 640px) {
    .page-header-row {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.875rem;
    }
    .title { font-size: 2.25rem; }
  }
</style>
