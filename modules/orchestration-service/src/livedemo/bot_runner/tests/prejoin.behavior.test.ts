/**
 * Behavioral test: muteMic + ensureCameraOn helpers run against a stub
 * pre-join page that mimics Meet's mic/cam toggles.
 *
 * Validates Phase 9.5 of PLAN_7:
 *  - mic must be muted BEFORE clickJoin (we don't want bot to transmit audio)
 *  - camera must be ON BEFORE clickJoin (canvas stream is the bot's video)
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { chromium, Browser, Page } from 'playwright';
import { muteMic, ensureCameraOn } from '../src/runner';

let browser: Browser;

const PREJOIN_HTML_MIC_ON_CAM_OFF = `
<!doctype html>
<html><body>
<button aria-label="Turn off microphone" id="mic">Turn off microphone</button>
<button aria-label="Turn on camera" id="cam">Turn on camera</button>
<script>
  window.__micClicks = 0;
  window.__camClicks = 0;
  document.getElementById('mic').addEventListener('click', () => { window.__micClicks++; });
  document.getElementById('cam').addEventListener('click', () => { window.__camClicks++; });
</script>
</body></html>
`;

const PREJOIN_HTML_MIC_ALREADY_OFF = `
<!doctype html>
<html><body>
<button aria-label="Turn on microphone" id="mic">Turn on microphone</button>
<button aria-label="Turn on camera" id="cam">Turn on camera</button>
<script>
  window.__micClicks = 0;
  window.__camClicks = 0;
  document.getElementById('mic').addEventListener('click', () => { window.__micClicks++; });
  document.getElementById('cam').addEventListener('click', () => { window.__camClicks++; });
</script>
</body></html>
`;

beforeAll(async () => {
  browser = await chromium.launch({ headless: true });
}, 60_000);

afterAll(async () => {
  await browser?.close();
});

describe('pre-join mic/cam helpers', () => {
  it('muteMic clicks the "Turn off microphone" button when mic is on', async () => {
    const page = await (await browser.newContext()).newPage();
    await page.goto('https://example.com');
    await page.setContent(PREJOIN_HTML_MIC_ON_CAM_OFF);

    await muteMic(page);
    const micClicks = await page.evaluate(() => (window as any).__micClicks);
    expect(micClicks).toBe(1);
  });

  it('muteMic is a no-op when mic is already off (no "Turn off microphone" button visible)', async () => {
    const page = await (await browser.newContext()).newPage();
    await page.goto('https://example.com');
    await page.setContent(PREJOIN_HTML_MIC_ALREADY_OFF);

    await muteMic(page);
    const micClicks = await page.evaluate(() => (window as any).__micClicks);
    expect(micClicks).toBe(0);
  });

  it('ensureCameraOn clicks "Turn on camera" when camera is off', async () => {
    const page = await (await browser.newContext()).newPage();
    await page.goto('https://example.com');
    await page.setContent(PREJOIN_HTML_MIC_ON_CAM_OFF);

    await ensureCameraOn(page);
    const camClicks = await page.evaluate(() => (window as any).__camClicks);
    expect(camClicks).toBe(1);
  });

  it('ensureCameraOn is a no-op when camera is already on (no "Turn on camera" visible)', async () => {
    const cam_on_html = `
<!doctype html><html><body>
<button aria-label="Turn off camera" id="cam">Turn off camera</button>
<script>
  window.__camClicks = 0;
  document.getElementById('cam').addEventListener('click', () => { window.__camClicks++; });
</script></body></html>`;
    const page = await (await browser.newContext()).newPage();
    await page.goto('https://example.com');
    await page.setContent(cam_on_html);

    await ensureCameraOn(page);
    const camClicks = await page.evaluate(() => (window as any).__camClicks);
    expect(camClicks).toBe(0);
  });
});
