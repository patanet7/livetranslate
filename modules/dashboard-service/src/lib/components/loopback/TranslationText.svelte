<script lang="ts">
  /**
   * Shared translation text renderer with phase-based visual feedback.
   *
   * Phases:
   *   pending   → pulsing dot, no text
   *   draft     → 65% opacity, solid dot
   *   streaming → 85% opacity, fast pulse + glow
   *   complete  → 100% opacity, no indicator
   */

  interface Props {
    text: string;
    phase: 'pending' | 'draft' | 'streaming' | 'complete';
  }

  let { text, phase }: Props = $props();
</script>

<span class="translation-text phase-{phase}">
  {#if text}{text}{/if}
  {#if phase !== 'complete' && phase !== 'pending'}
    <span class="indicator indicator-{phase}"></span>
  {:else if phase === 'pending'}
    <span class="indicator indicator-pending"></span>
  {/if}
</span>

<style>
  .translation-text {
    transition: opacity 0.3s ease;
  }

  .phase-pending {
    opacity: 1;
  }

  .phase-draft {
    opacity: 0.65;
  }

  .phase-streaming {
    opacity: 0.85;
  }

  .phase-complete {
    opacity: 1;
  }

  .indicator {
    display: inline-block;
    width: 6px;
    height: 6px;
    margin-left: 6px;
    border-radius: 50%;
    vertical-align: middle;
  }

  .indicator-pending {
    background: var(--color-translation, #90ee90);
    opacity: 0.7;
    animation: pulse-dot 1.2s ease-in-out infinite;
  }

  .indicator-draft {
    background: var(--color-translation, #90ee90);
    opacity: 0.5;
  }

  .indicator-streaming {
    background: var(--color-translation, #90ee90);
    opacity: 0.9;
    animation: fast-pulse 0.6s ease-in-out infinite;
    box-shadow: 0 0 6px var(--color-translation, #90ee90);
  }

  @keyframes pulse-dot {
    0%, 100% { opacity: 0.3; transform: scale(0.8); }
    50% { opacity: 1; transform: scale(1.2); }
  }

  @keyframes fast-pulse {
    0%, 100% { opacity: 0.4; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.3); }
  }
</style>
