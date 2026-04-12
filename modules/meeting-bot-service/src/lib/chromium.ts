import { chromium, Page, BrowserContext } from 'playwright';
import fs from 'fs';
import config from '../config';

/**
 * Options for creating a browser context.
 */
interface CreateBrowserOptions {
  usePersistentProfile?: boolean;
  headless?: boolean;
}

/**
 * Browser launch arguments for stealth and media handling.
 */
const BROWSER_ARGS = [
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
];

/**
 * Context options for browser automation.
 */
const CONTEXT_OPTIONS = {
  viewport: { width: 1920, height: 1080 },
  permissions: ['camera', 'microphone'],
  userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  ignoreHTTPSErrors: true,
};

/**
 * Injects stealth scripts to avoid browser automation detection.
 */
async function injectStealthScripts(page: Page): Promise<void> {
  await page.addInitScript(() => {
    // Override webdriver detection
    Object.defineProperty(navigator, 'webdriver', {
      get: () => undefined,
    });

    // Override plugins to appear normal
    Object.defineProperty(navigator, 'plugins', {
      get: () => [
        {
          name: 'Chrome PDF Plugin',
          description: 'Portable Document Format',
          filename: 'internal-pdf-viewer',
          version: '1.0',
        },
      ],
    });

    // Override languages
    Object.defineProperty(navigator, 'languages', {
      get: () => ['en-US', 'en'],
    });

    // Override chrome runtime
    if (typeof window !== 'undefined') {
      (window as any).chrome = {
        runtime: {},
      };
    }
  });
}

/**
 * Creates a new browser context and page for meeting bot automation.
 * Uses headless Chrome with specific permissions for screen capture.
 * Supports persistent profiles for Google authentication.
 */
export default async function createBrowserContext(
  url: string,
  correlationId: string,
  options: CreateBrowserOptions = {}
): Promise<Page> {
  const { usePersistentProfile = true, headless = true } = options;
  const profileDir = config.chromeProfileDir;
  const profileExists = fs.existsSync(profileDir);

  // Use persistent context if profile exists and requested
  if (usePersistentProfile && profileExists) {
    const context = await chromium.launchPersistentContext(profileDir, {
      headless,
      executablePath: config.chromeExecutablePath,
      args: BROWSER_ARGS,
      ...CONTEXT_OPTIONS,
    });

    const page = context.pages()[0] || await context.newPage();
    await injectStealthScripts(page);

    // Set extra headers for correlation
    await page.setExtraHTTPHeaders({
      'X-Correlation-ID': correlationId,
    });

    return page;
  }

  // Fall back to ephemeral context
  const browser = await chromium.launch({
    headless,
    executablePath: config.chromeExecutablePath,
    args: BROWSER_ARGS,
  });

  const context: BrowserContext = await browser.newContext(CONTEXT_OPTIONS);
  const page = await context.newPage();

  await injectStealthScripts(page);

  // Set extra headers for correlation
  await page.setExtraHTTPHeaders({
    'X-Correlation-ID': correlationId,
  });

  return page;
}
