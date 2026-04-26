/**
 * Behavioral test: init_script audio capture pipeline.
 *
 * Validates Phase 9.2.b — the bot can capture audio from a MediaStream
 * (returned by our overridden getUserMedia) and expose 20ms int16 PCM chunks
 * via window.__livedemoAudio.takeChunk().
 *
 * In production the source MediaStream is the meeting audio (Meet's combined
 * `<audio>` output once we tap into it). For this test we feed it via
 * AudioContext.createOscillator → MediaStreamDestination so we exercise the
 * exact same WebAudio path with deterministic input.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium, Browser, Page } from 'playwright';
import { INIT_SCRIPT } from '../src/init_script';

let browser: Browser;
let page: Page;

beforeAll(async () => {
  browser = await chromium.launch({
    headless: true,
    args: ['--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
  });
  const context = await browser.newContext();
  await context.addInitScript(INIT_SCRIPT);
  page = await context.newPage();
  await page.goto('https://example.com');
}, 60_000);

afterAll(async () => {
  await browser?.close();
});

describe('audio capture pipeline', () => {
  it('exposes window.__livedemoAudio API', async () => {
    const has = await page.evaluate(() => {
      return !!(window as any).__livedemoAudio
        && typeof (window as any).__livedemoAudio.startCapture === 'function'
        && typeof (window as any).__livedemoAudio.takeChunk === 'function';
    });
    expect(has).toBe(true);
  });

  it('startCapture + takeChunk yields int16 PCM chunks', async () => {
    const result = await page.evaluate(async () => {
      const ctx = new AudioContext({ sampleRate: 48000 });
      const osc = ctx.createOscillator();
      osc.frequency.value = 440;
      const dest = ctx.createMediaStreamDestination();
      osc.connect(dest);
      osc.start();

      // Feed the synthetic stream into our capture pipeline
      // @ts-ignore
      await (window as any).__livedemoAudio.startCapture(dest.stream);

      // Wait for ~5 chunks to accumulate (at 16kHz, 320 samples = 20ms)
      await new Promise(res => setTimeout(res, 200));

      const chunks: number[] = [];
      for (let i = 0; i < 5; i++) {
        // @ts-ignore
        const c = (window as any).__livedemoAudio.takeChunk();
        if (c) chunks.push(c.byteLength);
      }
      osc.stop();
      return { chunkCount: chunks.length, chunkSizes: chunks };
    });
    expect(result.chunkCount).toBeGreaterThan(0);
    // 320 samples * 2 bytes = 640 bytes per 20ms int16 chunk
    expect(result.chunkSizes.every((s: number) => s === 640)).toBe(true);
  });
});
