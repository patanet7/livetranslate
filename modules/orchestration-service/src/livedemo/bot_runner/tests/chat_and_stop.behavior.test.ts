/**
 * Behavioral test: sendChatMessage helper + leave_request WS handler.
 *
 * Phase 9.6:
 *  - sendChatMessage opens chat panel, fills the contenteditable input via
 *    InputEvent (page.fill doesn't work on contenteditable), clicks send
 *  - leave_request handler clicks LEAVE_BUTTON_SELECTORS
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium, Browser, Page } from 'playwright';
import { sendChatMessage, handleLeaveRequest } from '../src/runner';

let browser: Browser;

const CHAT_HTML = `
<!doctype html><html><body>
<button aria-label="Chat with everyone" id="open-chat">Chat</button>
<div id="panel" style="display:none">
  <div contenteditable="true" aria-label="Send a message to everyone" id="input"></div>
  <button aria-label="Send a message" id="send">Send</button>
</div>
<script>
  window.__sentMessages = [];
  document.getElementById('open-chat').addEventListener('click', () => {
    document.getElementById('panel').style.display = 'block';
  });
  document.getElementById('input').addEventListener('input', () => {
    window.__inputText = document.getElementById('input').textContent;
  });
  document.getElementById('send').addEventListener('click', () => {
    window.__sentMessages.push(window.__inputText || '');
  });
</script>
</body></html>
`;

const LEAVE_HTML = `
<!doctype html><html><body>
<button aria-label="Leave call" id="leave">Leave call</button>
<script>
  window.__leaveClicks = 0;
  document.getElementById('leave').addEventListener('click', () => { window.__leaveClicks++; });
</script>
</body></html>
`;

beforeAll(async () => {
  browser = await chromium.launch({ headless: true });
}, 60_000);

afterAll(async () => {
  await browser?.close();
});

describe('sendChatMessage', () => {
  it('opens chat panel, fills input, clicks send', async () => {
    const page = await (await browser.newContext()).newPage();
    await page.goto('https://example.com');
    await page.setContent(CHAT_HTML);

    const ok = await sendChatMessage(page, 'Hello from livedemo bot');
    expect(ok).toBe(true);
    const sent = await page.evaluate(() => (window as any).__sentMessages);
    expect(sent).toEqual(['Hello from livedemo bot']);
  });

  it('returns false when chat input never appears', async () => {
    const page = await (await browser.newContext()).newPage();
    await page.goto('https://example.com');
    await page.setContent('<html><body>nothing here</body></html>');

    const ok = await sendChatMessage(page, 'whatever');
    expect(ok).toBe(false);
  });
});

describe('handleLeaveRequest', () => {
  it('clicks the Leave call button', async () => {
    const page = await (await browser.newContext()).newPage();
    await page.goto('https://example.com');
    await page.setContent(LEAVE_HTML);

    await handleLeaveRequest(page);
    const clicks = await page.evaluate(() => (window as any).__leaveClicks);
    expect(clicks).toBe(1);
  });

  it('is a no-op when no Leave button is visible', async () => {
    const page = await (await browser.newContext()).newPage();
    await page.goto('https://example.com');
    await page.setContent('<html><body>no leave button</body></html>');
    // Should not throw
    await handleLeaveRequest(page);
  });
});
