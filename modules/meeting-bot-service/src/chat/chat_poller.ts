/**
 * Chat Poller — polls Google Meet chat DOM every 500ms.
 *
 * Opens the chat panel, reads new messages, detects /commands.
 * More reliable than MutationObserver (survives panel open/close,
 * Google Meet DOM restructures).
 */

import { Page } from 'playwright';
import { Logger } from 'winston';
import { CHAT_BUTTON, CHAT_MESSAGES_CONTAINER } from './selectors';

export interface ChatMessage {
  sender: string;
  text: string;
  timestamp: number;
}

export type OnCommandCallback = (command: string, sender: string) => void;

export class ChatPoller {
  private page: Page;
  private logger: Logger;
  private onCommand: OnCommandCallback;
  private pollInterval: ReturnType<typeof setInterval> | null = null;
  private seenMessages: Set<string> = new Set();
  private isRunning = false;

  constructor(page: Page, logger: Logger, onCommand: OnCommandCallback) {
    this.page = page;
    this.logger = logger;
    this.onCommand = onCommand;
  }

  async start(): Promise<void> {
    if (this.isRunning) return;

    // Open chat panel
    try {
      const chatButton = this.page.locator(CHAT_BUTTON).first();
      if (await chatButton.isVisible({ timeout: 3000 })) {
        await chatButton.click();
        this.logger.info('Chat panel opened');
        await this.page.waitForTimeout(500);
      }
    } catch (err) {
      this.logger.warn('Could not open chat panel', { error: (err as Error).message });
    }

    this.isRunning = true;
    this.pollInterval = setInterval(() => this.poll(), 500);
    this.logger.info('Chat poller started');
  }

  stop(): void {
    this.isRunning = false;
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    this.logger.info('Chat poller stopped');
  }

  private async poll(): Promise<void> {
    if (!this.isRunning) return;

    try {
      const messages = await this.page.evaluate((containerSelector: string) => {
        const container = document.querySelector(containerSelector);
        if (!container) return [];

        const items = container.querySelectorAll('[data-message-text]');
        const results: Array<{ sender: string; text: string; key: string }> = [];

        items.forEach((item) => {
          const text = item.getAttribute('data-message-text') || item.textContent || '';
          // Try to find the sender name from the message structure
          const senderEl = item.closest('[data-sender-name]');
          const sender = senderEl?.getAttribute('data-sender-name') || 'Unknown';
          const key = `${sender}:${text}`;
          results.push({ sender, text, key });
        });

        return results;
      }, CHAT_MESSAGES_CONTAINER);

      for (const msg of messages) {
        if (this.seenMessages.has(msg.key)) continue;
        this.seenMessages.add(msg.key);

        if (msg.text.trim().startsWith('/')) {
          this.logger.info('Chat command detected', { sender: msg.sender, command: msg.text });
          this.onCommand(msg.text.trim(), msg.sender);
        }
      }
    } catch (err) {
      // Chat panel might be closed or DOM changed — log and continue
      this.logger.debug('Chat poll error', { error: (err as Error).message });
    }
  }
}
