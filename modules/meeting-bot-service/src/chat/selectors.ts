/**
 * Google Meet DOM selectors — single source of truth.
 *
 * Update ONLY this file when Google Meet changes its DOM.
 * Primary strategy: aria-label (survives most restructures).
 * Secondary: button text content (breaks with locale changes).
 *
 * Each selector group has multiple fallbacks tried in priority order.
 * 2026 Google Meet UI updates included.
 */

import { Page, Locator } from 'playwright';

// ---------------------------------------------------------------------------
// PRE-JOIN
// ---------------------------------------------------------------------------

export const MIC_OFF_SELECTORS = [
  'button[aria-label*="Turn off microphone"]',
  'button[aria-label*="Mute microphone"]',
  'button[aria-label="Microphone"]',
  'button:has-text("Turn off microphone")',
];

export const CAM_OFF_SELECTORS = [
  'button[aria-label*="Turn off camera"]',
  'button[aria-label*="Disable camera"]',
  'button[aria-label="Camera"]',
  'button:has-text("Turn off camera")',
];

export const JOIN_BUTTON_SELECTORS = [
  'button[aria-label*="Join now"]',
  'button[aria-label*="Ask to join"]',
  'button[aria-label*="Join anyway"]',
  'button:has-text("Join now")',
  'button:has-text("Ask to join")',
  'button:has-text("Join anyway")',
];

export const NAME_INPUT =
  'input[type="text"][aria-label="Your name"]';

export const CONTINUE_WITHOUT_SELECTORS = [
  'button[aria-label*="Continue without microphone"]',
  'button:has-text("Continue without microphone and camera")',
  'button:has-text("Continue without mic and camera")',
];

// ---------------------------------------------------------------------------
// IN-MEETING
// ---------------------------------------------------------------------------

export const LEAVE_BUTTON_SELECTORS = [
  'button[aria-label="Leave call"]',
  'button[aria-label*="Leave"]',
  'button[aria-label*="End call"]',
  'button:has-text("Leave call")',
];

export const CAPTIONS_ON_SELECTORS = [
  'button[aria-label="Turn on captions"]',
  'button[aria-label*="Turn on captions"]',
  'button[aria-label*="Enable captions"]',
  'button:has-text("Turn on captions")',
];

export const CAPTIONS_OFF_SELECTORS = [
  'button[aria-label="Turn off captions"]',
  'button[aria-label*="Turn off captions"]',
  'button[aria-label*="Disable captions"]',
  'button:has-text("Turn off captions")',
];

export const CAPTIONS_REGION_SELECTORS = [
  '[jscontroller="D1tHje"]',
  '[aria-label="Captions"]',
  '.a4cQT',
  '[data-self-name] .a4cQT',
];

// ---------------------------------------------------------------------------
// CHAT
// ---------------------------------------------------------------------------

export const CHAT_BUTTON_SELECTORS = [
  'button[aria-label="Chat with everyone"]',
  'button[aria-label*="Chat"]',
  '[data-panel-id="chat"] button',
  'button:has-text("Chat")',
];

export const CHAT_INPUT_SELECTORS = [
  '[aria-label="Send a message to everyone"]',
  'textarea[aria-label*="message"]',
  'input[aria-label*="message"]',
  '[contenteditable="true"][aria-label*="message"]',
];

export const CHAT_MESSAGES_SELECTORS = [
  '[aria-label="Chat messages"]',
  '[aria-label="Chat panel"]',
  '[data-panel-id="chat"]',
];

export const CHAT_MESSAGE_ITEM =
  '[data-message-id], .chat-message, [jsname="hc1s4"]';

// ---------------------------------------------------------------------------
// DISMISSALS
// ---------------------------------------------------------------------------

export const GOT_IT_SELECTORS = [
  'button:has-text("Got it")',
  'button[jsname="EszDEe"]',
  '[data-is-touch-wrapper="true"] button',
  'button[aria-label="Got it"]',
];

export const CLOSE_BUTTON_SELECTORS = [
  'button[aria-label="Close"]',
  'button[aria-label="Dismiss"]',
  '[data-dismiss]',
  '.VfPpkd-Bz112c-LgbsSe',
];

// ---------------------------------------------------------------------------
// EXIT / POST-MEETING
// ---------------------------------------------------------------------------

export const LEFT_MEETING_SELECTORS = [
  '[aria-label="You left the meeting"]',
  'h1:has-text("You left the meeting")',
  'h2:has-text("You left the meeting")',
  'div:has-text("You left the meeting")',
];

// ---------------------------------------------------------------------------
// PEOPLE
// ---------------------------------------------------------------------------

export const PEOPLE_BUTTON_SELECTORS = [
  'button[aria-label="People"]',
  'button[aria-label^="People"]',
  'button[aria-label*="participants"]',
  'button:has-text("People")',
];

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/**
 * Try multiple selectors in priority order, return first visible locator.
 */
export async function findVisible(page: Page, selectors: string[]): Promise<Locator | null> {
  for (const sel of selectors) {
    try {
      const loc = page.locator(sel).first();
      if (await loc.isVisible({ timeout: 1000 })) {
        return loc;
      }
    } catch {
      // Try next selector
    }
  }
  return null;
}

/**
 * Click the first visible element matching any of the given selectors.
 * Returns true if a click was performed, false if none matched.
 */
export async function clickFirst(
  page: Page,
  selectors: string[],
  options?: { timeout?: number },
): Promise<boolean> {
  const timeout = options?.timeout ?? 1000;
  for (const sel of selectors) {
    try {
      const loc = page.locator(sel).first();
      if (await loc.isVisible({ timeout })) {
        await loc.click({ timeout });
        return true;
      }
    } catch {
      // Try next selector
    }
  }
  return false;
}

/**
 * Wait until any of the given selectors becomes visible.
 * Returns the first visible locator, or null if timeout expires.
 */
export async function waitForAny(
  page: Page,
  selectors: string[],
  timeout = 10000,
): Promise<Locator | null> {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    for (const sel of selectors) {
      try {
        const loc = page.locator(sel).first();
        if (await loc.isVisible({ timeout: 500 })) {
          return loc;
        }
      } catch {
        // Try next selector
      }
    }
    await page.waitForTimeout(200);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Legacy single-value exports for backward compatibility
// ---------------------------------------------------------------------------

/** @deprecated Use CHAT_BUTTON_SELECTORS */
export const CHAT_BUTTON = CHAT_BUTTON_SELECTORS[0];

/** @deprecated Use CHAT_INPUT_SELECTORS */
export const CHAT_INPUT = CHAT_INPUT_SELECTORS[0];

/** @deprecated Use CHAT_MESSAGES_SELECTORS */
export const CHAT_MESSAGES_CONTAINER = CHAT_MESSAGES_SELECTORS[0];

/** @deprecated Use CAPTIONS_ON_SELECTORS */
export const CC_BUTTON = CAPTIONS_ON_SELECTORS[0];

/** @deprecated Use CAPTIONS_REGION_SELECTORS */
export const CC_CONTAINER = CAPTIONS_REGION_SELECTORS[0];

/** @deprecated Use CAPTIONS_REGION_SELECTORS */
export const CC_TEXT_SPAN = CAPTIONS_REGION_SELECTORS[2];

/** @deprecated Use LEAVE_BUTTON_SELECTORS */
export const LEAVE_CALL_BUTTON = LEAVE_BUTTON_SELECTORS[0];

/** @deprecated Use PEOPLE_BUTTON_SELECTORS */
export const PEOPLE_BUTTON_PREFIX = PEOPLE_BUTTON_SELECTORS[1];

/** @deprecated Use CAM_OFF_SELECTORS */
export const CAMERA_BUTTON = 'button[aria-label="Turn on camera"]';

/** @deprecated Use MIC_OFF_SELECTORS */
export const MIC_BUTTON = 'button[aria-label="Turn on microphone"]';

/** @deprecated Use JOIN_BUTTON_SELECTORS */
export const ASK_TO_JOIN_BUTTON = JOIN_BUTTON_SELECTORS[1];

/** @deprecated Use JOIN_BUTTON_SELECTORS */
export const JOIN_NOW_BUTTON = JOIN_BUTTON_SELECTORS[0];

/** @deprecated Use CONTINUE_WITHOUT_SELECTORS */
export const CONTINUE_WITHOUT_MIC_CAM = CONTINUE_WITHOUT_SELECTORS[1];

/** @deprecated Use GOT_IT_SELECTORS */
export const GOT_IT_BUTTON = GOT_IT_SELECTORS[0];

/** @deprecated Use CHAT_INPUT_SELECTORS */
export const CHAT_SEND_BUTTON = '[aria-label="Send a message"]';

/** @deprecated Use CHAT_MESSAGES_SELECTORS */
export const CHAT_PANEL = CHAT_MESSAGES_SELECTORS[0];
