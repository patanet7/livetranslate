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

function createAudioStore() {
  let rms = $state(0);
  let peak = $state(0);
  let lastUpdateMs = $state(0);

  return {
    /** Current RMS (root-mean-square) audio amplitude, 0..1. */
    get rms() { return rms; },
    /** Most recent peak amplitude, 0..1. More visually responsive than RMS. */
    get peak() { return peak; },
    /** ms timestamp of the last update — used to detect "no audio for N seconds". */
    get lastUpdateMs() { return lastUpdateMs; },
    /** Audio considered "alive" if updated in the last 500ms. */
    get isFlowing() {
      return lastUpdateMs > 0 && Date.now() - lastUpdateMs < 500;
    },

    /** Push a new level reading. Called from AudioCapture's `onLevel` callback. */
    push(value: number, peakValue?: number): void {
      rms = value;
      if (peakValue !== undefined) peak = peakValue;
      lastUpdateMs = Date.now();
    },

    /** Reset to silent state. Call when capture stops. */
    reset(): void {
      rms = 0;
      peak = 0;
      lastUpdateMs = 0;
    },
  };
}

export const audioStore = createAudioStore();
