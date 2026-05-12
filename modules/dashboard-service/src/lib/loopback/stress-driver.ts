/**
 * Offline UI stress driver — reproduces the loopback "lockup" without a backend.
 *
 * Injects synthetic SegmentMessage + TranslationChunkMessage events into the
 * captionStore at controllable rates, while measuring main-thread frame budget
 * violations via requestAnimationFrame.
 *
 * Use cases:
 *  - Reproduce the production lockup deterministically.
 *  - A/B current vs proposed caption-store fixes (Task #10) on identical input.
 *  - Audit Toolbar reactivity overhead — flip displayMode mid-stress, watch
 *    whether the captionSource $effect causes extra work.
 *
 * Pure TypeScript — no Svelte template imports — so it runs identically in
 * Node (vitest + happy-dom) and in the browser dev route.
 */
import { captionStore } from '$lib/stores/caption.svelte';
import type { SegmentMessage, TranslationChunkMessage, TranslationMessage } from '$lib/types/ws-messages';

export interface StressMetrics {
  /** Number of rAF frames observed. */
  frameCount: number;
  /** Longest gap between rAF callbacks (ms). 60fps ideal = 16.67ms. */
  maxFrameMs: number;
  /** 95th-percentile frame gap (ms). */
  p95FrameMs: number;
  /** Number of frames that exceeded `slowFrameThresholdMs`. */
  slowFrames: number;
  /** Threshold used for slowFrames counting. */
  slowFrameThresholdMs: number;
  /** Total segments injected. */
  segmentsInjected: number;
  /** Total translation chunks injected. */
  chunksInjected: number;
  /** Total final translations injected. */
  translationsInjected: number;
  /** Wall-clock duration of the run (ms). */
  durationMs: number;
}

export interface StressConfig {
  /** Total number of segments to inject before stopping. */
  totalSegments: number;
  /** Average chunks per translation (streaming bursts). */
  chunksPerTranslation: number;
  /** Time between segment arrivals (ms). 0 = as-fast-as-possible. */
  segmentIntervalMs: number;
  /** Time between chunks within a single translation burst (ms). */
  chunkIntervalMs: number;
  /** Anything slower than this counts as a "slow frame". */
  slowFrameThresholdMs: number;
  /** Optional: rotate through this many speakers. */
  speakers: number;
}

export const DEFAULT_STRESS_CONFIG: StressConfig = {
  totalSegments: 200,
  chunksPerTranslation: 25,
  segmentIntervalMs: 200,    // 5 segments/sec — realistic VAC stride cadence
  chunkIntervalMs: 30,       // ~33 chunks/sec — realistic LLM streaming token rate
  slowFrameThresholdMs: 50,  // >50ms = visible jank
  speakers: 3,
};

const CN_PHRASES = [
  '你好世界,我喜欢编程。',
  '今天天气很好,我们应该出去走走。',
  '我建议你有空去看一下我们前端的代码。',
  '能在本地跑起来之后可以做一些实验。',
  '前端然后和交互都能做OK那我建议你。',
];

const EN_PHRASES = [
  'We should ship the wire-service prototype before the standup.',
  'I think the streaming pipeline needs more backpressure.',
  'Translation latency under one second is the goal we want.',
  'The interpreter mode should switch cleanly between languages.',
  'Capture loops must be identical regardless of input source.',
];

function pickPhrase(idx: number): { text: string; language: string } {
  const useChinese = idx % 2 === 0;
  const arr = useChinese ? CN_PHRASES : EN_PHRASES;
  return { text: arr[idx % arr.length], language: useChinese ? 'zh' : 'en' };
}

function makeSegment(id: number, speakers: number): SegmentMessage {
  const { text, language } = pickPhrase(id);
  return {
    type: 'segment',
    segment_id: id,
    text,
    language,
    confidence: 0.85 + (id % 10) * 0.01,
    stable_text: text,
    unstable_text: '',
    is_final: true,
    is_draft: false,
    speaker_id: `spk_${id % speakers}`,
    start_ms: id * 1000,
    end_ms: id * 1000 + 800,
  };
}

function makeChunk(transcriptId: number, delta: string): TranslationChunkMessage {
  return {
    type: 'translation_chunk',
    transcript_id: transcriptId,
    delta,
    source_lang: 'zh',
    target_lang: 'en',
  };
}

function makeFinal(transcriptId: number, fullText: string): TranslationMessage {
  return {
    type: 'translation',
    transcript_id: transcriptId,
    text: fullText,
    source_lang: 'zh',
    target_lang: 'en',
  } as TranslationMessage;
}

/**
 * Frame-budget meter. Starts a rAF loop that records the gap between each
 * frame. Stop with `.stop()` to freeze metrics.
 */
class FrameMeter {
  private gaps: number[] = [];
  private last: number = 0;
  private rafId: number = 0;
  private running = false;
  readonly slowFrameThresholdMs: number;
  startedAt = 0;
  stoppedAt = 0;

  constructor(slowFrameThresholdMs: number) {
    this.slowFrameThresholdMs = slowFrameThresholdMs;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.startedAt = performance.now();
    this.last = this.startedAt;
    const tick = (now: number) => {
      if (!this.running) return;
      this.gaps.push(now - this.last);
      this.last = now;
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  stop(): void {
    if (!this.running) return;
    this.running = false;
    this.stoppedAt = performance.now();
    cancelAnimationFrame(this.rafId);
  }

  snapshot(): Omit<StressMetrics, 'segmentsInjected' | 'chunksInjected' | 'translationsInjected'> {
    const sorted = [...this.gaps].sort((a, b) => a - b);
    const p95Idx = Math.floor(sorted.length * 0.95);
    return {
      frameCount: this.gaps.length,
      maxFrameMs: sorted.length ? sorted[sorted.length - 1] : 0,
      p95FrameMs: sorted.length ? sorted[p95Idx] : 0,
      slowFrames: this.gaps.filter(g => g > this.slowFrameThresholdMs).length,
      slowFrameThresholdMs: this.slowFrameThresholdMs,
      durationMs: (this.stoppedAt || performance.now()) - this.startedAt,
    };
  }
}

export interface StressHandle {
  stop(): Promise<StressMetrics>;
  /** Promise that resolves with final metrics when the stress run completes naturally. */
  done: Promise<StressMetrics>;
  /** Live counters (read during a run for progress UI). */
  live: { segments: number; chunks: number; translations: number };
}

/**
 * Run a stress scenario against the live captionStore. Returns a handle whose
 * `done` promise resolves with frame-budget metrics when the run completes.
 */
export function runStress(config: Partial<StressConfig> = {}): StressHandle {
  const cfg: StressConfig = { ...DEFAULT_STRESS_CONFIG, ...config };
  const meter = new FrameMeter(cfg.slowFrameThresholdMs);
  meter.start();

  const live = { segments: 0, chunks: 0, translations: 0 };
  let stopped = false;

  const resolveRef: { fn: ((m: StressMetrics) => void) | null } = { fn: null };
  const done = new Promise<StressMetrics>((resolve) => {
    resolveRef.fn = resolve;
  });

  function finish(): void {
    if (stopped) return;
    stopped = true;
    meter.stop();
    const m: StressMetrics = {
      ...meter.snapshot(),
      segmentsInjected: live.segments,
      chunksInjected: live.chunks,
      translationsInjected: live.translations,
    };
    resolveRef.fn?.(m);
  }

  // Driver loop: inject segments at segmentIntervalMs; for each segment, inject
  // chunksPerTranslation chunks at chunkIntervalMs; emit a final translation
  // when the burst completes.
  let nextSegmentAt = performance.now();
  let segmentId = 1;

  function injectNextSegment(): void {
    if (stopped || segmentId > cfg.totalSegments) {
      finish();
      return;
    }

    const seg = makeSegment(segmentId, cfg.speakers);
    captionStore.ingestSegment(seg);
    live.segments++;

    // Start streaming-chunk burst for this segment.
    const transcriptId = seg.segment_id;
    const fullText = `[en] ${seg.text} (translated)`;
    const chunks = splitIntoChunks(fullText, cfg.chunksPerTranslation);
    let chunkIdx = 0;

    function injectNextChunk(): void {
      if (stopped) {
        finish();
        return;
      }
      if (chunkIdx < chunks.length) {
        captionStore.ingestTranslationChunk(makeChunk(transcriptId, chunks[chunkIdx]));
        live.chunks++;
        chunkIdx++;
        setTimeout(injectNextChunk, cfg.chunkIntervalMs);
        return;
      }
      // Burst complete — emit final translation.
      captionStore.ingestTranslation(makeFinal(transcriptId, fullText));
      live.translations++;
      segmentId++;
      nextSegmentAt += cfg.segmentIntervalMs;
      const wait = Math.max(0, nextSegmentAt - performance.now());
      setTimeout(injectNextSegment, wait);
    }

    setTimeout(injectNextChunk, cfg.chunkIntervalMs);
  }

  injectNextSegment();

  return {
    stop(): Promise<StressMetrics> {
      finish();
      return done;
    },
    done,
    live,
  };
}

/** Split ``text`` into roughly ``n`` chunks, simulating LLM token streaming. */
function splitIntoChunks(text: string, n: number): string[] {
  if (n <= 0) return [text];
  const chunkSize = Math.max(1, Math.ceil(text.length / n));
  const out: string[] = [];
  for (let i = 0; i < text.length; i += chunkSize) {
    out.push(text.slice(i, i + chunkSize));
  }
  return out;
}
