// Capture before/after screenshots of every dashboard surface.
// Usage (run from modules/dashboard-service/):
//   node scripts/screenshot.mjs before
//   node scripts/screenshot.mjs after
//
// Outputs to <repo-root>/design/<phase>/ at viewport 1440x900.
// The dashboard must be running on http://localhost:5180.
// Pages render their chrome even without backend services up — empty
// data states are an acceptable baseline for the design comparison.

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const phase = process.argv[2] ?? "before";
const repoRoot = resolve(import.meta.dirname, "../../..");
const outDir = resolve(repoRoot, `design/${phase}`);
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
    await page.goto(`http://localhost:5180${route}`, {
      waitUntil: "networkidle",
      timeout: 8000,
    });
    await page.waitForTimeout(400); // settle animations
    await page.screenshot({ path: `${outDir}/${name}.png`, fullPage: true });
    console.log(`✓ ${name.padEnd(18)} ${route}`);
  } catch (err) {
    console.log(`✗ ${name.padEnd(18)} ${route}  (${err.message.split("\n")[0]})`);
  }
}

await browser.close();
console.log(`\nSaved to ${outDir}/`);
