import { chromium, Page, BrowserContext } from 'playwright';
import config from '../config';

/**
 * Creates a new browser context and page for meeting bot automation.
 * Uses headless Chrome with specific permissions for screen capture.
 */
export default async function createBrowserContext(
  url: string,
  correlationId: string
): Promise<Page> {
  const browser = await chromium.launch({
    headless: true,
    executablePath: config.chromeExecutablePath,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--disable-gpu',
      '--window-size=1920,1080',
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      '--auto-select-desktop-capture-source=Entire screen',
      '--autoplay-policy=no-user-gesture-required',
      '--enable-features=SharedArrayBuffer',
    ],
  });

  const context: BrowserContext = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    permissions: ['camera', 'microphone'],
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ignoreHTTPSErrors: true,
  });

  const page = await context.newPage();

  // Set extra headers for correlation
  await page.setExtraHTTPHeaders({
    'X-Correlation-ID': correlationId,
  });

  return page;
}
