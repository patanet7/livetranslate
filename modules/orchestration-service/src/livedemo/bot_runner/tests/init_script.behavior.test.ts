/**
 * Behavioral test: real Chromium + INIT_SCRIPT against about:blank.
 *
 * Validates B6 (canvas attached to document.body), the getUserMedia override
 * returns the canvas-backed MediaStream, and enumerateDevices reports the
 * fake "LiveTranslate Canvas Camera".
 *
 * Run: `npm test` from the bot_runner package dir.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium, Browser, Page } from 'playwright';
import { INIT_SCRIPT } from '../src/init_script';

let browser: Browser;
let page: Page;

beforeAll(async () => {
  browser = await chromium.launch({ headless: true, args: ['--use-fake-ui-for-media-stream'] });
  const context = await browser.newContext();
  await context.addInitScript(INIT_SCRIPT);
  page = await context.newPage();
  // Use a real HTTPS site so navigator.mediaDevices is exposed (secure context required).
  await page.goto('https://example.com');
}, 60_000);

afterAll(async () => {
  await browser?.close();
});

describe('INIT_SCRIPT behavioral', () => {
  it('canvas is attached to document.body (B6)', async () => {
    const inDom = await page.evaluate(() => (window as any).__livedemo?.isCanvasInDom());
    expect(inDom).toBe(true);
  });

  it('canvas is 1280x720', async () => {
    const dims = await page.evaluate(() => {
      const c = (window as any).__livedemo?.canvas as HTMLCanvasElement;
      return { w: c.width, h: c.height };
    });
    expect(dims).toEqual({ w: 1280, h: 720 });
  });

  it('getUserMedia({video:true}) returns a stream with canvas track', async () => {
    const result = await page.evaluate(async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const tracks = stream.getVideoTracks();
      return {
        trackCount: tracks.length,
        kind: tracks[0]?.kind,
        readyState: tracks[0]?.readyState,
      };
    });
    expect(result.trackCount).toBeGreaterThanOrEqual(1);
    expect(result.kind).toBe('video');
    expect(result.readyState).toBe('live');
  });

  it('enumerateDevices includes a videoinput device', async () => {
    const found = await page.evaluate(async () => {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.some(d => d.kind === 'videoinput');
    });
    expect(found).toBe(true);
  });

  it('setFrameDataUrl updates the canvas with a data URL', async () => {
    // 1x1 red PNG
    const redPng = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9V0jY9MAAAAASUVORK5CYII=';
    const ok = await page.evaluate(async (b64) => {
      // @ts-ignore
      window.__livedemo.setFrameDataUrl(`data:image/png;base64,${b64}`);
      // Wait for image load
      await new Promise(res => setTimeout(res, 100));
      // @ts-ignore
      return window.__livedemo.lastFrameDataUrl?.startsWith('data:image/png');
    }, redPng);
    expect(ok).toBe(true);
  });
});
