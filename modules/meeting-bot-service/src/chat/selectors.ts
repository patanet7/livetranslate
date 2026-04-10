/**
 * Google Meet DOM selectors — single source of truth.
 *
 * Update ONLY this file when Google Meet changes its DOM.
 * Primary strategy: aria-label (survives most restructures).
 * Secondary: button text content (breaks with locale changes).
 */

import { Page, Locator } from 'playwright';

// Chat panel
export const CHAT_BUTTON = '[aria-label="Chat with everyone"]';
export const CHAT_PANEL = '[aria-label="Chat panel"]';
export const CHAT_INPUT = '[aria-label="Send a message to everyone"]';
export const CHAT_SEND_BUTTON = '[aria-label="Send a message"]';
export const CHAT_MESSAGES_CONTAINER = '[aria-label="Chat messages"]';

// Meeting controls
export const LEAVE_CALL_BUTTON = '[aria-label="Leave call"]';
export const PEOPLE_BUTTON_PREFIX = 'button[aria-label^="People"]';
export const CAMERA_BUTTON = '[aria-label="Turn on camera"]';
export const MIC_BUTTON = '[aria-label="Turn on microphone"]';

// Pre-join
export const NAME_INPUT = 'input[type="text"][aria-label="Your name"]';
export const ASK_TO_JOIN_BUTTON = 'button:has-text("Ask to join")';
export const JOIN_NOW_BUTTON = 'button:has-text("Join now")';
export const CONTINUE_WITHOUT_MIC_CAM = 'button:has-text("Continue without microphone and camera")';
export const GOT_IT_BUTTON = 'button:has-text("Got it")';

// Built-in closed captions (Meet CC)
export const CC_BUTTON = '[aria-label="Turn on captions"]';
export const CC_CONTAINER = '[jscontroller="D1tHje"]';
export const CC_TEXT_SPAN = '.a4cQT';

/**
 * Try multiple selectors in priority order, return first visible one.
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
