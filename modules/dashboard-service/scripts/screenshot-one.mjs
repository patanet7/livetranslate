// One-off screenshot — usage:
//   node scripts/screenshot-one.mjs <path> <name>
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const route = process.argv[2] ?? "/";
const name = process.argv[3] ?? "shot";
const repoRoot = resolve(import.meta.dirname, "../../..");
const outDir = resolve(repoRoot, "design/scratch");
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
const page = await ctx.newPage();
await page.goto(`http://localhost:5180${route}`, { waitUntil: "networkidle", timeout: 8000 });
await page.waitForTimeout(800);
const out = `${outDir}/${name}.png`;
await page.screenshot({ path: out, fullPage: true });
await browser.close();
console.log(out);
