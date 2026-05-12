// Capture before/after screenshots of every dashboard surface.
// Usage (run from modules/dashboard-service/):
//   node scripts/screenshot.mjs before               # dark (default)
//   node scripts/screenshot.mjs after                # dark
//   node scripts/screenshot.mjs after light          # daytime/print mode
//
// Outputs to <repo-root>/design/<phase>[-<theme>]/ at viewport 1440x900.
// The dashboard must be running on http://localhost:5180.
// Pages render their chrome even without backend services up — empty
// data states are an acceptable baseline for the design comparison.

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const phase = process.argv[2] ?? "before";
const theme = process.argv[3] ?? "dark"; // "dark" (default) | "light"
const repoRoot = resolve(import.meta.dirname, "../../..");
const dirSuffix = theme === "dark" ? phase : `${phase}-${theme}`;
const outDir = resolve(repoRoot, `design/${dirSuffix}`);
await mkdir(outDir, { recursive: true });

const ROUTES = [
  ["home", "/"],
  ["loopback", "/loopback"],
  ["sessions", "/sessions"],
  ["meetings", "/meetings"],
  ["translation", "/translation/test"],
  ["intelligence", "/intelligence"],
  ["fireflies", "/fireflies"],
  ["chat", "/chat"],
  ["config", "/config"],
  ["data", "/data"],
  ["overlay-captions", "/captions"],
];

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();

for (const [name, route] of ROUTES) {
  try {
    // Vite HMR keeps a WebSocket open; networkidle never lands. Use
    // domcontentloaded + explicit waits for fonts and the reveal cascade.
    await page.goto(`http://localhost:5180${route}`, {
      waitUntil: "domcontentloaded",
      timeout: 12000,
    });
    // Wait for Google Fonts (Fraunces / Newsreader / JetBrains Mono) so
    // headlines render in the editorial faces, not a fallback.
    await page.evaluate(() => document.fonts.ready);
    // Theme override — app.html hardcodes class="dark" so light mode
    // requires a runtime swap on the <html> element. Done after fonts
    // are ready so the swap doesn't fight FOUC mitigation.
    if (theme === "light") {
      await page.evaluate(() => {
        document.documentElement.classList.remove("dark");
      });
    }
    // Settle the staggered editorial reveal (D6.2): max delay 480ms
    // + 380ms transition + rule-draw 540ms@320ms delay = 860ms ceiling.
    await page.waitForTimeout(1100);
    await page.screenshot({ path: `${outDir}/${name}.png`, fullPage: true });
    console.log(`✓ ${name.padEnd(18)} ${route}  [${theme}]`);
  } catch (err) {
    console.log(`✗ ${name.padEnd(18)} ${route}  (${err.message.split("\n")[0]})`);
  }
}

await browser.close();
console.log(`\nSaved to ${outDir}/`);
