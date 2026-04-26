/**
 * livedemo bot runner — Playwright subprocess that:
 *   1. Opens Chromium with persistent profile + stealth args
 *   2. Injects init_script (canvas in DOM, getUserMedia override)
 *   3. Joins the meeting (handles "Switch here", modal-dismiss)
 *   4. Connects to Python harness over WS (port from BOT_WS_PORT env)
 *   5. On each `{type:"frame"}` message: paints the data URL onto the canvas
 *   6. Sends `{type:"in_call"}` to harness once past the lobby (B5)
 *
 * Configuration via env (set by Python harness when spawning):
 *   BOT_WS_PORT        — port to connect to Python harness on
 *   BOT_MEETING_URL    — meeting to join
 *   BOT_PROFILE_DIR    — Chrome profile path
 *   BOT_HEADLESS       — "1" or "0" (default 0 — headed required for camera)
 */

import { chromium, BrowserContext, Page } from 'playwright';
import WebSocket from 'ws';
import { INIT_SCRIPT } from './init_script';
import {
  JOIN_BUTTON_SELECTORS,
  LEAVE_BUTTON_SELECTORS,
  PEOPLE_BUTTON_SELECTORS,
  GOT_IT_SELECTORS,
  CLOSE_BUTTON_SELECTORS,
  MIC_OFF_SELECTORS,
  MIC_ON_SELECTORS,
  CAM_OFF_SELECTORS,
  CAM_ON_SELECTORS,
  CHAT_BUTTON_SELECTORS,
  CHAT_INPUT_SELECTORS,
  clickFirst,
  findVisible,
  waitForAny,
} from './lib/selectors';
import { fileURLToPath } from 'url';

const WS_PORT = parseInt(process.env.BOT_WS_PORT || '7081', 10);
const MEETING_URL = process.env.BOT_MEETING_URL || '';
const PROFILE_DIR = process.env.BOT_PROFILE_DIR || '';
const HEADLESS = process.env.BOT_HEADLESS === '1';

interface FrameMsg { type: 'frame'; data: string; ts: number; }
interface HelloMsg { type: 'hello'; version: number; }
type ServerMsg = FrameMsg | HelloMsg;

function logToHarness(ws: WebSocket | null, level: string, msg: string): void {
  console.log(`[bot ${level}] ${msg}`);
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'log', level, msg }));
  }
}

async function dismissModals(page: Page): Promise<void> {
  await clickFirst(page, GOT_IT_SELECTORS, { timeout: 400 });
  await page.waitForTimeout(150);
  await clickFirst(page, CLOSE_BUTTON_SELECTORS, { timeout: 400 });
  await page.waitForTimeout(150);
  // Belt-and-suspenders for any straggler dialog.
  await page.keyboard.press('Escape').catch(() => {});
  await page.waitForTimeout(150);
  await page.keyboard.press('Escape').catch(() => {});
}

/**
 * Mute the bot's mic before join. No-op if mic is already muted.
 *
 * Pre-join, Meet shows "Turn off microphone" when mic is ON. Clicking it
 * toggles to muted. Once muted, the button changes to "Turn on microphone"
 * (matched by MIC_ON_SELECTORS) and we stop.
 */
export async function muteMic(page: Page): Promise<boolean> {
  return clickFirst(page, MIC_OFF_SELECTORS, { timeout: 1000 });
}

/**
 * Ensure the bot's camera is ON so the canvas-backed MediaStream is the bot's
 * video tile. No-op if camera is already on.
 *
 * Pre-join, Meet shows "Turn on camera" when cam is OFF. Clicking it toggles
 * to ON. Once on, button changes to CAM_OFF and we stop.
 */
export async function ensureCameraOn(page: Page): Promise<boolean> {
  return clickFirst(page, CAM_ON_SELECTORS, { timeout: 1000 });
}

/**
 * Send a chat message into Meet's chat panel. Opens the panel if needed.
 *
 * Meet's input is contenteditable — `page.fill` doesn't work, so we set
 * textContent + dispatch InputEvent before clicking the send button.
 *
 * Returns true if the send button was clicked, false if the input never appeared.
 */
export async function sendChatMessage(page: Page, text: string): Promise<boolean> {
  // Try to find the chat input directly (panel may already be open).
  let chatInput = await findVisible(page, CHAT_INPUT_SELECTORS);
  if (!chatInput) {
    // Open the chat panel and retry.
    const opened = await clickFirst(page, CHAT_BUTTON_SELECTORS, { timeout: 1000 });
    if (!opened) return false;
    await page.waitForTimeout(500);
    chatInput = await findVisible(page, CHAT_INPUT_SELECTORS);
    if (!chatInput) return false;
  }

  await chatInput.click();
  await chatInput.evaluate((el: HTMLElement, msg: string) => {
    el.textContent = msg;
    el.dispatchEvent(new InputEvent('input', { inputType: 'insertText', bubbles: true }));
  }, text);

  // Click send. Try aria-label "Send a message" first; fall back to Enter.
  const sendBtn = page.locator('[aria-label="Send a message"]').first();
  if (await sendBtn.isVisible({ timeout: 500 }).catch(() => false)) {
    await sendBtn.click({ timeout: 1500 }).catch(() => {});
    return true;
  }
  await page.keyboard.press('Enter').catch(() => {});
  return true;
}

/**
 * Click the Leave button. No-op if not visible.
 *
 * Triggered by harness sending `{type:"leave_request"}` (e.g. when the
 * orchestration `/stop` command fires).
 */
export async function handleLeaveRequest(page: Page): Promise<void> {
  await clickFirst(page, LEAVE_BUTTON_SELECTORS, { timeout: 1500 });
}

const JOIN_CHAT_MESSAGE =
  'LiveTranslate bot active. Pin my video for subtitles. Type /help for commands.';

async function clickJoin(page: Page): Promise<void> {
  // "Switch here" is a session-handoff variant of join — include it as a fallback.
  const joinSelectors = [
    ...JOIN_BUTTON_SELECTORS,
    'button:has-text("Switch here")',
  ];
  const clicked = await clickFirst(page, joinSelectors, { timeout: 5000 });
  if (!clicked) {
    throw new Error('No join button visible');
  }
}

/**
 * B5 — robust in-call signal. Three independent checks; any one passes.
 *   1. People-panel button is rendered (only exists in-call)
 *   2. Self-video tile is rendered with [data-self-name]
 *   3. "Still trying to get in..." text is absent for >3s
 */
async function waitForInCall(page: Page, timeoutMs = 90_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const peoplePanel = await findVisible(page, PEOPLE_BUTTON_SELECTORS);
    if (peoplePanel) return;
    const selfTile = await page.locator('[data-self-name]').first()
      .isVisible({ timeout: 200 }).catch(() => false);
    if (selfTile) return;
    const stuck = await page.locator('text=/still trying to get in/i').first()
      .isVisible({ timeout: 200 }).catch(() => false);
    if (!stuck) {
      // Not stuck-text either — Leave button as last resort.
      const leave = await findVisible(page, LEAVE_BUTTON_SELECTORS);
      if (leave) return;
    }
    await page.waitForTimeout(500);
  }
  throw new Error('waitForInCall timed out — bot likely stuck in lobby');
}

async function connectHarness(): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://127.0.0.1:${WS_PORT}`);
    const timeout = setTimeout(() => reject(new Error('Harness handshake timeout')), 10_000);
    ws.on('open', () => {
      clearTimeout(timeout);
      // Wait for the hello before resolving — Python sends it immediately.
    });
    ws.once('message', (raw) => {
      const msg = JSON.parse(raw.toString());
      if (msg.type === 'hello' && typeof msg.version === 'number') {
        clearTimeout(timeout);
        resolve(ws);
      } else {
        reject(new Error(`Unexpected first message: ${JSON.stringify(msg)}`));
      }
    });
    ws.on('error', (err) => { clearTimeout(timeout); reject(err); });
  });
}

async function main(): Promise<void> {
  if (!MEETING_URL) throw new Error('BOT_MEETING_URL not set');
  if (!PROFILE_DIR) throw new Error('BOT_PROFILE_DIR not set');

  const ws = await connectHarness();
  logToHarness(ws, 'info', `WS connected on :${WS_PORT}`);

  const context: BrowserContext = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: HEADLESS,
    viewport: { width: 1280, height: 800 },
    permissions: ['camera', 'microphone'],
    args: [
      '--no-sandbox',
      '--disable-blink-features=AutomationControlled',
      '--disable-dev-shm-usage',
      '--use-fake-ui-for-media-stream',
      '--autoplay-policy=no-user-gesture-required',
    ],
  });
  const page = await context.newPage();
  await page.addInitScript(INIT_SCRIPT);

  page.on('console', (m) => logToHarness(ws, 'info', `page: ${m.text()}`));

  logToHarness(ws, 'info', `Navigating to ${MEETING_URL}`);
  await page.goto(MEETING_URL, { waitUntil: 'networkidle', timeout: 60_000 });
  await page.waitForTimeout(2000);

  // Pre-join: mute mic + ensure camera is on so the canvas stream becomes the bot's video.
  await muteMic(page).catch(() => {});
  await ensureCameraOn(page).catch(() => {});
  await page.waitForTimeout(300);

  await clickJoin(page);
  logToHarness(ws, 'info', 'Clicked join — waiting for in-call signal...');
  await waitForInCall(page);
  logToHarness(ws, 'info', 'IN-CALL detected');
  ws.send(JSON.stringify({ type: 'in_call' }));

  await dismissModals(page);

  // Post the canonical join message to chat. Best-effort — failure is logged
  // but doesn't abort the run.
  sendChatMessage(page, JOIN_CHAT_MESSAGE)
    .then((ok) => logToHarness(ws, 'info', `join chat message sent: ${ok}`))
    .catch((err) => logToHarness(ws, 'warn', `join chat send failed: ${err.message}`));

  // Start audio capture: feed the canvas-stream's audio (preserved through
  // getUserMedia override) into the init_script's PCM tap.
  await page.evaluate(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // @ts-ignore - injected by init_script
    await window.__livedemoAudio?.startCapture(stream);
  }).catch((err) => logToHarness(ws, 'warn', `audio capture init failed: ${err.message}`));

  // Audio forwarding loop: poll the page's chunk queue and ship binary frames
  // over the canvas WS. ~20ms intervals match the 20ms PCM chunk size.
  const audioPump = setInterval(async () => {
    if (ws.readyState !== WebSocket.OPEN) return;
    try {
      // Drain up to 5 chunks per tick to keep latency low under bursts.
      for (let i = 0; i < 5; i++) {
        const chunk = await page.evaluate(() => {
          // @ts-ignore
          const buf = window.__livedemoAudio?.takeChunk();
          if (!buf) return null;
          return Array.from(new Uint8Array(buf));
        });
        if (!chunk || chunk.length === 0) break;
        ws.send(Buffer.from(chunk));
      }
    } catch (err) {
      // Page may have navigated mid-poll; tolerate transient errors.
    }
  }, 20);

  // Frame + control message loop — drives canvas paints + handles leave_request.
  ws.on('message', async (raw) => {
    let msg: any;
    try { msg = JSON.parse(raw.toString()); } catch { return; }
    if (msg.type === 'frame' && typeof msg.data === 'string') {
      const dataUrl = `data:image/png;base64,${msg.data}`;
      await page.evaluate((url) => {
        // @ts-ignore - injected by init_script
        window.__livedemo?.setFrameDataUrl(url);
      }, dataUrl).catch((err) => logToHarness(ws, 'warn', `frame paint failed: ${err.message}`));
    } else if (msg.type === 'leave_request') {
      logToHarness(ws, 'info', 'Received leave_request — leaving meeting');
      await handleLeaveRequest(page).catch(() => {});
      try { ws.close(); } catch {}
    } else if (msg.type === 'chat_send' && typeof msg.text === 'string') {
      await sendChatMessage(page, msg.text).catch(() => {});
    }
  });

  ws.on('close', async () => {
    logToHarness(ws, 'info', 'Harness disconnected — leaving');
    clearInterval(audioPump);
    await clickFirst(page, LEAVE_BUTTON_SELECTORS, { timeout: 3000 }).catch(() => {});
    await context.close();
    process.exit(0);
  });
}

// Only run main() when invoked as the entry script (not on import).
const isMainModule = import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
  main().catch((err) => {
    console.error('[bot fatal]', err);
    process.exit(1);
  });
}
