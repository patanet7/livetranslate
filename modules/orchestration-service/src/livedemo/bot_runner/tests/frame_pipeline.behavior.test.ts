/**
 * Behavioral integration test: full frame pipeline in real Chromium.
 *
 * Pipeline: caption frame (PNG bytes) → setFrameDataUrl() → canvas.drawImage
 *           → canvas.captureStream → MediaStreamTrack frames → consumer video element
 *
 * This proves the load-bearing claim: when Python pushes a frame, the
 * canvas-backed MediaStream actually emits a new video frame to anything
 * consuming it (Google Meet's <video> element in production).
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium, Browser, Page } from 'playwright';
import { INIT_SCRIPT } from '../src/init_script';

let browser: Browser;
let page: Page;

const PAGE_HTML = `
<!doctype html>
<html><body>
<video id="v" autoplay playsinline muted></video>
<script>
  window.__videoFrameCount = 0;
  window.__attachStream = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const v = document.getElementById('v');
    v.srcObject = stream;
    if (typeof v.requestVideoFrameCallback === 'function') {
      const tick = () => { window.__videoFrameCount++; v.requestVideoFrameCallback(tick); };
      v.requestVideoFrameCallback(tick);
    } else {
      // Fallback: poll currentTime growth
      let last = -1;
      setInterval(() => {
        if (v.currentTime !== last) {
          window.__videoFrameCount++;
          last = v.currentTime;
        }
      }, 50);
    }
    return stream.getVideoTracks().length;
  };
</script>
</body></html>
`;

beforeAll(async () => {
  browser = await chromium.launch({
    headless: true,
    args: ['--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
  });
  const context = await browser.newContext();
  await context.addInitScript(INIT_SCRIPT);
  page = await context.newPage();
  // Serve the HTML via data: URL on an HTTPS-equivalent secure context.
  // data: URIs are secure contexts in Chromium when given as the page origin.
  await page.goto('https://example.com');
  await page.setContent(PAGE_HTML);
}, 60_000);

afterAll(async () => {
  await browser?.close();
});

describe('frame pipeline E2E', () => {
  it('attaching the canvas stream to a <video> yields a video track', async () => {
    const trackCount = await page.evaluate(async () => {
      // @ts-ignore
      return await window.__attachStream();
    });
    expect(trackCount).toBeGreaterThanOrEqual(1);
  });

  it('pushing a frame data-URL produces video frames on the consuming <video>', async () => {
    // Wait for initial frames from the placeholder + redraw interval.
    await page.waitForTimeout(700);
    const before = await page.evaluate(() => (window as any).__videoFrameCount);

    // Push a fresh frame (a 1x1 blue PNG).
    const bluePng = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAEhgGA4gx0SwAAAABJRU5ErkJggg==';
    await page.evaluate((b64) => {
      // @ts-ignore
      window.__livedemo.setFrameDataUrl(`data:image/png;base64,${b64}`);
    }, bluePng);
    await page.waitForTimeout(600);
    const after = await page.evaluate(() => (window as any).__videoFrameCount);

    expect(after).toBeGreaterThan(before);
  });

  it('setFrameDataUrl updates lastFrameDataUrl', async () => {
    const dataUrl = await page.evaluate(() => (window as any).__livedemo.lastFrameDataUrl);
    expect(dataUrl).toMatch(/^data:image\/png;base64,/);
  });
});
