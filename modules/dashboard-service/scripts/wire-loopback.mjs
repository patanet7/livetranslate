// Capture WireView running on the production /loopback page,
// driven by the built-in demo runner so it gets real captionStore
// data (not the prototype's mock corpus). Confirms the new component
// integrates with the existing toolbar, display-mode switcher, and
// caption pipeline.
//
// Usage: node scripts/wire-loopback.mjs

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const repoRoot = resolve(import.meta.dirname, "../../..");
const outDir = resolve(repoRoot, "design/wire-production");
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch();

async function run(theme, label, waitMs) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  // Pre-seed displayMode = "wire" in localStorage before mount so the
  // page comes up directly in wire mode. This avoids a click-flow that
  // would race the demo's first dispatches.
  await page.addInitScript(() => {
    localStorage.setItem(
      "livetranslate:caption-config",
      JSON.stringify({
        sourceLanguage: null,
        targetLanguage: "zh",
        displayMode: "wire",
        captionSource: "local",
        interpreterLangA: "zh",
        interpreterLangB: "en",
      }),
    );
  });

  await page.goto("http://localhost:5180/loopback", { waitUntil: "domcontentloaded" });
  await page.evaluate(() => document.fonts.ready);
  if (theme === "light") {
    await page.evaluate(() => document.documentElement.classList.remove("dark"));
  }

  // The TopBar has its own "Demo" dropdown that navigates to /fireflies.
  // The loopback Toolbar has a Demo button that drives runDemo() locally
  // without navigation — that's the one we want. Scope by class.
  await page.locator('.toolbar button:has-text("Demo")').first().click();
  await page.waitForTimeout(waitMs);
  await page.screenshot({ path: `${outDir}/${theme}-${label}.png`, fullPage: true });
  console.log(`✓ ${theme}-${label}  (waited ${waitMs}ms)`);
  await context.close();
}

for (const theme of ["dark", "light"]) {
  await run(theme, "01-early", 3500);
  await run(theme, "02-mid", 12000);
  await run(theme, "03-late", 22000);
}

await browser.close();
console.log(`\nSaved to ${outDir}/`);
