// Capture timed frames of the /dev/wire prototype to evaluate the
// teleprinter aesthetic: empty / mid-typing / fully populated, in
// both daytime and evening-edition modes.
//
// Usage: node scripts/wire-prototype.mjs

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const repoRoot = resolve(import.meta.dirname, "../../..");
const outDir = resolve(repoRoot, "design/wire-prototype");
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch();

async function snap(theme, label, waitMs) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  await page.goto("http://localhost:5180/dev/wire", { waitUntil: "domcontentloaded" });
  await page.evaluate(() => document.fonts.ready);
  if (theme === "light") {
    await page.evaluate(() => document.documentElement.classList.remove("dark"));
  }
  // The page spawns its first mock dispatch ~800ms after mount and
  // each one types over ~2.5s, so the windows of interest are:
  //   t≈ 1500ms — first dispatch mid-source-type
  //   t≈ 3800ms — first dispatch FILED, second arriving
  //   t≈14000ms — feed has ~3 filed + 1 in-flight
  await page.waitForTimeout(waitMs);
  await page.screenshot({ path: `${outDir}/${theme}-${label}.png`, fullPage: true });
  console.log(`✓ ${theme}-${label}  (waited ${waitMs}ms)`);
  await context.close();
}

for (const theme of ["dark", "light"]) {
  await snap(theme, "01-mid-typing", 1500);
  await snap(theme, "02-first-filed", 3800);
  await snap(theme, "03-populated", 14000);
}

await browser.close();
console.log(`\nSaved to ${outDir}/`);
