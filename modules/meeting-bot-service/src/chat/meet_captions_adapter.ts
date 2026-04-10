/**
 * MeetCaptionsAdapter — scrapes Google Meet's built-in closed captions.
 *
 * Polls the CC overlay DOM, extracts speaker + text, deduplicates,
 * and forwards as caption events. Third caption source alongside
 * bot_audio and fireflies.
 */

import { Page } from 'playwright';
import { Logger } from 'winston';
import { CC_BUTTON, CC_CONTAINER, CC_TEXT_SPAN } from './selectors';

export interface CaptionEntry {
  speaker: string;
  text: string;
}

export function parseCaptionText(raw: string): CaptionEntry | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;

  const colonIdx = trimmed.indexOf(':');
  if (colonIdx > 0 && colonIdx < 30) {
    const speaker = trimmed.substring(0, colonIdx).trim();
    const text = trimmed.substring(colonIdx + 1).trim();
    if (text) {
      return { speaker, text };
    }
  }

  return { speaker: 'Unknown', text: trimmed };
}

export class MeetCaptionsAdapter {
  private page: Page;
  private logger: Logger;
  private onCaption: ((entry: CaptionEntry) => void) | null = null;
  private pollInterval: ReturnType<typeof setInterval> | null = null;
  private seenTexts: Set<string> = new Set();
  private isRunning = false;

  constructor(page: Page, logger: Logger) {
    this.page = page;
    this.logger = logger;
  }

  setOnCaption(callback: (entry: CaptionEntry) => void): void {
    this.onCaption = callback;
  }

  async start(): Promise<void> {
    if (this.isRunning) return;

    try {
      const ccButton = this.page.locator(CC_BUTTON).first();
      if (await ccButton.isVisible({ timeout: 3000 })) {
        await ccButton.click();
        this.logger.info('Enabled Meet closed captions');
        await this.page.waitForTimeout(500);
      }
    } catch (err) {
      this.logger.warn('Could not enable CC', { error: (err as Error).message });
    }

    this.isRunning = true;
    this.pollInterval = setInterval(() => this.poll(), 300);
    this.logger.info('MeetCaptionsAdapter started');
  }

  stop(): void {
    this.isRunning = false;
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    this.logger.info('MeetCaptionsAdapter stopped');
  }

  private async poll(): Promise<void> {
    if (!this.isRunning) return;

    try {
      const texts = await this.page.evaluate((containerSel: string, textSel: string) => {
        const container = document.querySelector(containerSel);
        if (!container) return [];
        const spans = container.querySelectorAll(textSel);
        return Array.from(spans).map(s => s.textContent || '');
      }, CC_CONTAINER, CC_TEXT_SPAN);

      for (const rawText of texts) {
        if (!rawText.trim()) continue;
        const key = rawText.trim();
        if (this.seenTexts.has(key)) continue;
        this.seenTexts.add(key);

        if (this.seenTexts.size > 500) {
          const firstKey = this.seenTexts.values().next().value;
          if (firstKey) this.seenTexts.delete(firstKey);
        }

        const entry = parseCaptionText(rawText);
        if (entry && this.onCaption) {
          this.onCaption(entry);
        }
      }
    } catch (err) {
      this.logger.debug('CC poll error', { error: (err as Error).message });
    }
  }
}
