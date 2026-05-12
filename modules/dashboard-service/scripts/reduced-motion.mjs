// Verify prefers-reduced-motion correctly short-circuits editorial
// animations: the reveal cascade, the rule-draw, the button :active
// transform, and the .unstable ink-drying underline pulse.
//
// Strategy: snap a fast mid-cascade frame (180ms after nav). With
// motion enabled, content is mid-stagger (partial opacity, blur, rule
// not yet drawn). With motion reduced, content should already be at
// final state (opaque, sharp, rule fully drawn). If both look the
// same, the @media (prefers-reduced-motion: reduce) overrides aren't
// firing.
//
// Usage: node scripts/reduced-motion.mjs

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const repoRoot = resolve(import.meta.dirname, "../../..");
const outDir = resolve(repoRoot, "design/reduced-motion");
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch();

async function snapEarly(reducedMotion, label) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    reducedMotion,
  });
  const page = await context.newPage();
  await page.goto("http://localhost:5180/", { waitUntil: "domcontentloaded" });
  await page.evaluate(() => document.fonts.ready);
  // 180ms — past first-paint but before ANY of: the longest reveal
  // delay (480ms), the reveal transition (380ms), or the rule-draw
  // (540ms@320ms). With motion enabled, content here is mid-stagger.
  // With motion reduced, all transitions are no-op so end-state is
  // already painted.
  await page.waitForTimeout(180);
  await page.screenshot({
    path: `${outDir}/${label}.png`,
    clip: { x: 120, y: 0, width: 900, height: 280 },
  });
  console.log(`✓ ${label.padEnd(28)} reducedMotion=${reducedMotion}`);
  await context.close();
}

await snapEarly("no-preference", "motion-on-180ms");
await snapEarly("reduce", "motion-reduced-180ms");

// Also capture the post-cascade settled state for both, as a control.
async function snapSettled(reducedMotion, label) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    reducedMotion,
  });
  const page = await context.newPage();
  await page.goto("http://localhost:5180/", { waitUntil: "domcontentloaded" });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(1100);
  await page.screenshot({
    path: `${outDir}/${label}.png`,
    clip: { x: 120, y: 0, width: 900, height: 280 },
  });
  console.log(`✓ ${label.padEnd(28)} reducedMotion=${reducedMotion}`);
  await context.close();
}
await snapSettled("no-preference", "motion-on-settled");
await snapSettled("reduce", "motion-reduced-settled");

await browser.close();
console.log(`\nSaved to ${outDir}/`);
console.log(
  "\nExpected:" +
    "\n  motion-on-180ms     mid-cascade — content fading in, rule partially drawn" +
    "\n  motion-reduced-180ms already settled — content opaque, rule fully drawn (proves override)" +
    "\n  *-settled           both fully painted (control)",
);
