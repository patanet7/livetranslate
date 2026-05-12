// Capture interactive editorial states the static screenshot can't show.
// Usage: node scripts/interactive-states.mjs [light|dark]
//
// Outputs to <repo-root>/design/interactive[-light]/.
// Requires the dashboard running on http://localhost:5180.
//
// What this captures:
//   reveal-mid      — Home page captured 200ms after navigation, mid-cascade
//   buttons-rest    — Home Refresh button at rest
//   buttons-hover   — Home Refresh button hovered (peach under-shadow)
//   buttons-press   — Home Refresh button mid-click (1px translateY + inset)
//   focus-ring      — Sessions page first input, focus-visible (purple ring)
//   selection       — Kicker line on home, peach selection wash

import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const theme = process.argv[2] ?? "dark";
const repoRoot = resolve(import.meta.dirname, "../../..");
const outDir = resolve(
  repoRoot,
  theme === "dark" ? "design/interactive" : `design/interactive-${theme}`,
);
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();

async function setTheme() {
  if (theme === "light") {
    await page.evaluate(() => document.documentElement.classList.remove("dark"));
  }
}
async function settle(extraMs = 1100) {
  await page.evaluate(() => document.fonts.ready);
  await setTheme();
  await page.waitForTimeout(extraMs);
}

// ── 1. Reveal cascade — capture mid-cascade so the stagger is visible
await page.goto("http://localhost:5180/", { waitUntil: "domcontentloaded" });
await page.evaluate(() => document.fonts.ready);
await setTheme();
await page.waitForTimeout(200);
await page.screenshot({
  path: `${outDir}/reveal-mid.png`,
  clip: { x: 120, y: 0, width: 900, height: 220 },
});
console.log(`✓ reveal-mid          (mid-cascade, 200ms after nav)  [${theme}]`);

// ── 2. Buttons rest / hover / press — Home Refresh button (always enabled)
await page.goto("http://localhost:5180/", { waitUntil: "domcontentloaded" });
await settle();
const refreshBtn = page.locator('button:has-text("Refresh")').first();
await refreshBtn.scrollIntoViewIfNeeded();
const btnBox = await refreshBtn.boundingBox();
if (!btnBox) throw new Error("Refresh button not found on home");
const clip = {
  x: Math.max(0, btnBox.x - 60),
  y: Math.max(0, btnBox.y - 30),
  width: btnBox.width + 120,
  height: btnBox.height + 60,
};

await page.screenshot({ path: `${outDir}/buttons-rest.png`, clip });
console.log(`✓ buttons-rest        Refresh button, no interaction  [${theme}]`);

await refreshBtn.hover();
await page.waitForTimeout(220); // letterpress shadow transition: 180ms
await page.screenshot({ path: `${outDir}/buttons-hover.png`, clip });
console.log(`✓ buttons-hover       Refresh hovered (letterpress shadow)  [${theme}]`);

// Press: dispatch mousedown only, capture before mouseup
await refreshBtn.dispatchEvent("mousedown");
await page.waitForTimeout(140);
await page.screenshot({ path: `${outDir}/buttons-press.png`, clip });
await refreshBtn.dispatchEvent("mouseup");
console.log(`✓ buttons-press       Refresh :active (mid-press)  [${theme}]`);

// Move mouse away to clear hover state for subsequent captures
await page.mouse.move(0, 0);

// ── 3. Focus ring — Sessions page Session ID input
await page.goto("http://localhost:5180/sessions", { waitUntil: "domcontentloaded" });
await settle();
const idInput = page.locator('input[placeholder*="session ID"]').first();
await idInput.focus();
await page.waitForTimeout(120);
const inputBox = await idInput.boundingBox();
if (inputBox) {
  await page.screenshot({
    path: `${outDir}/focus-ring.png`,
    clip: {
      x: Math.max(0, inputBox.x - 20),
      y: Math.max(0, inputBox.y - 20),
      width: inputBox.width + 40,
      height: inputBox.height + 40,
    },
  });
  console.log(`✓ focus-ring          Session ID input, focus-visible (purple ring)  [${theme}]`);
}

// ── 4. Text selection — kicker line on home
await page.goto("http://localhost:5180/", { waitUntil: "domcontentloaded" });
await settle();
const kicker = page.locator("p.kicker, p[class*='kicker']").first();
const kickerBox = await kicker.boundingBox();
if (kickerBox) {
  await kicker.click({ clickCount: 3 });
  await page.waitForTimeout(100);
  await page.screenshot({
    path: `${outDir}/selection.png`,
    clip: {
      x: Math.max(0, kickerBox.x - 20),
      y: Math.max(0, kickerBox.y - 20),
      width: Math.min(900, kickerBox.width + 40),
      height: kickerBox.height + 40,
    },
  });
  console.log(`✓ selection           Kicker line, peach selection wash  [${theme}]`);
}

await browser.close();
console.log(`\nSaved to ${outDir}/`);
