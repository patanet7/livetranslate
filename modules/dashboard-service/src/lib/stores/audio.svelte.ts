/**
 * Live audio amplitude store (D2.3).
 *
 * Single source of truth for "what is the mic / system audio level right now?"
 * The loopback page pushes RMS values from its `onLevel` callback; visual
 * components (Earwyrm mascot, future VU meters) read from here.
 *
 * Kept separate from caption.svelte.ts on purpose — captions are textual,
 * audio level is a continuously-updating metric. Different lifecycle.
 */

const FLOW_TIMEOUT_MS = 500;

function createAudioStore() {
  let rms = $state(0);
  let peak = $state(0);
  let lastUpdateMs = $state(0);
  // Separate reactive flag for "is audio flowing right now?" — distinct from
  // lastUpdateMs because consumers (Sidebar's Earwyrm) need to re-evaluate
  // *at the timeout deadline* when audio stops, not only on the next push.
  // Without this flag, the derived state was stuck at "live" indefinitely
  // after silence (Date.now() comparison inside a getter is not reactive).
  let flowing = $state(false);
  let flowTimer: ReturnType<typeof setTimeout> | null = null;

  function scheduleFlowExpiry(): void {
    if (flowTimer) clearTimeout(flowTimer);
    flowTimer = setTimeout(() => {
      flowing = false;
      flowTimer = null;
    }, FLOW_TIMEOUT_MS);
  }

  return {
    /** Current RMS (root-mean-square) audio amplitude, 0..1. */
    get rms() { return rms; },
    /** Most recent peak amplitude, 0..1. More visually responsive than RMS. */
    get peak() { return peak; },
    /** ms timestamp of the last update — for diagnostics, not reactivity. */
    get lastUpdateMs() { return lastUpdateMs; },
    /** Audio considered "alive" if a push arrived within FLOW_TIMEOUT_MS.
     *  Reactive: transitions back to false via a setTimeout after the last push. */
    get isFlowing() { return flowing; },

    /** Push a new level reading. Called from AudioCapture's `onLevel` callback. */
    push(value: number, peakValue?: number): void {
      rms = value;
      if (peakValue !== undefined) peak = peakValue;
      lastUpdateMs = Date.now();
      flowing = true;
      scheduleFlowExpiry();
    },

    /** Reset to silent state. Call when capture stops. */
    reset(): void {
      rms = 0;
      peak = 0;
      lastUpdateMs = 0;
      flowing = false;
      if (flowTimer) { clearTimeout(flowTimer); flowTimer = null; }
    },
  };
}

export const audioStore = createAudioStore();
