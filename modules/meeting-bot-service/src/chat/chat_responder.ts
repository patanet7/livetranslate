/**
 * Chat Responder — types bot responses into Google Meet chat.
 *
 * Handles the contenteditable div (page.fill doesn't work on these).
 * Uses evaluate() to set text + dispatch InputEvent, then clicks send.
 */

import { Page } from 'playwright';
import { Logger } from 'winston';
import { CHAT_INPUT, CHAT_SEND_BUTTON, CHAT_BUTTON } from './selectors';

export class ChatResponder {
  private page: Page;
  private logger: Logger;

  constructor(page: Page, logger: Logger) {
    this.page = page;
    this.logger = logger;
  }

  async sendMessage(text: string): Promise<boolean> {
    try {
      // Ensure chat panel is open
      const chatInput = this.page.locator(CHAT_INPUT).first();
      if (!await chatInput.isVisible({ timeout: 1000 })) {
        // Try opening chat panel
        const chatButton = this.page.locator(CHAT_BUTTON).first();
        if (await chatButton.isVisible({ timeout: 1000 })) {
          await chatButton.click();
          await this.page.waitForTimeout(500);
        }
      }

      // Click the input to focus
      await chatInput.click();

      // Set text via evaluate (page.fill doesn't work on contenteditable)
      await chatInput.evaluate((el: HTMLElement, msg: string) => {
        el.textContent = msg;
        el.dispatchEvent(new InputEvent('input', {
          inputType: 'insertText',
          bubbles: true,
        }));
      }, text);

      // Click send button (more reliable than pressing Enter)
      const sendButton = this.page.locator(CHAT_SEND_BUTTON).first();
      await sendButton.click({ timeout: 2000 });

      this.logger.info('Chat message sent', { text: text.substring(0, 50) });
      return true;
    } catch (err) {
      this.logger.warn('Failed to send chat message', {
        error: (err as Error).message,
        text: text.substring(0, 50),
      });
      return false;
    }
  }

  async sendJoinMessage(): Promise<void> {
    await this.sendMessage(
      'LiveTranslate bot active. Pin my video for subtitles. Type /help for commands.'
    );
  }
}
