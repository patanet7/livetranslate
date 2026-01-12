/**
 * SimpleMeetBot - Simplified Google Meet bot that successfully bypasses detection
 *
 * This bot uses the proven approach from our tests that successfully joins meetings:
 * 1. Dismiss Google Sign In popup
 * 2. Dismiss device permission dialog
 * 3. Fill in bot name
 * 4. Click join button
 */

import createBrowserContext from '../lib/chromium';
import { Logger } from 'winston';

export interface SimpleMeetBotConfig {
  meetingUrl: string;
  botName: string;
  correlationId: string;
  logger: Logger;
}

export class SimpleMeetBot {
  private config: SimpleMeetBotConfig;
  private page: any = null;

  constructor(config: SimpleMeetBotConfig) {
    this.config = config;
  }

  async join(): Promise<void> {
    const { meetingUrl, botName, correlationId, logger } = this.config;

    try {
      logger.info('Creating browser context...', { correlationId });
      this.page = await createBrowserContext(meetingUrl, correlationId);

      logger.info('Navigating to meeting...', { meetingUrl, correlationId });
      await this.page.goto(meetingUrl, { waitUntil: 'networkidle', timeout: 60000 });

      logger.info('Waiting for page to load...', { correlationId });
      await this.page.waitForTimeout(5000);

      // CRITICAL: Dismiss Google Sign In popup
      await this.dismissGoogleSignInPopup();

      // CRITICAL: Dismiss device permission dialog
      await this.dismissDevicePermissionDialog();

      // Fill in bot name
      await this.fillBotName(botName);

      // Click join button
      await this.clickJoinButton();

      logger.info('Bot successfully joined meeting!', { correlationId, botName });
    } catch (error: any) {
      this.config.logger.error('Failed to join meeting', {
        error: error.message,
        correlationId
      });
      throw error;
    }
  }

  private async dismissGoogleSignInPopup(): Promise<void> {
    const { logger, correlationId } = this.config;

    logger.info('Checking for Google Sign In popup...', { correlationId });

    try {
      const closeButtons = [
        'button[aria-label="Close"]',
        'button[aria-label="Dismiss"]',
        '[data-dismiss]',
        '.VfPpkd-Bz112c-LgbsSe', // Google Material Design close button
      ];

      for (const selector of closeButtons) {
        try {
          const button = await this.page.locator(selector).first();
          if (await button.isVisible({ timeout: 2000 })) {
            logger.info(`Found close button with selector: ${selector}`, { correlationId });
            await button.click();
            logger.info('Dismissed Google Sign In popup', { correlationId });
            await this.page.waitForTimeout(1000);
            return;
          }
        } catch (e) {
          // Button not found or not visible, continue
        }
      }

      logger.info('No Google Sign In popup found', { correlationId });
    } catch (error: any) {
      logger.warn('Error checking for Google Sign In popup', {
        error: error.message,
        correlationId
      });
    }
  }

  private async dismissDevicePermissionDialog(): Promise<void> {
    const { logger, correlationId } = this.config;

    logger.info('Checking for device permission dialog...', { correlationId });

    try {
      const deviceButton = await this.page.getByRole('button', {
        name: 'Continue without microphone and camera'
      });

      if (await deviceButton.isVisible({ timeout: 5000 })) {
        logger.info('Found device permission dialog', { correlationId });
        await deviceButton.click();
        logger.info('Dismissed device permission dialog', { correlationId });
        await this.page.waitForTimeout(2000);
      } else {
        logger.info('No device permission dialog found', { correlationId });
      }
    } catch (error: any) {
      logger.info('No device permission dialog found', { correlationId });
    }
  }

  private async fillBotName(botName: string): Promise<void> {
    const { logger, correlationId } = this.config;

    logger.info('Looking for name input...', { correlationId });

    // Try multiple selectors for the name input
    const nameSelectors = [
      'input[type="text"][aria-label="Your name"]',
      'input[type="text"][placeholder*="name" i]',
      'input[type="text"]',
    ];

    for (const selector of nameSelectors) {
      try {
        const input = await this.page.locator(selector).first();
        if (await input.isVisible({ timeout: 3000 })) {
          logger.info(`Found name input with selector: ${selector}`, { correlationId });
          logger.info(`Filling in name: ${botName}`, { correlationId });
          await input.fill(botName);
          logger.info('Name filled successfully', { correlationId, botName });
          await this.page.waitForTimeout(2000);
          return;
        }
      } catch (e) {
        // Try next selector
      }
    }

    logger.warn('Could not find name input', { correlationId });
    throw new Error('Could not find name input field');
  }

  private async clickJoinButton(): Promise<void> {
    const { logger, correlationId } = this.config;

    logger.info('Looking for join button...', { correlationId });

    const joinButtonTexts = ['Ask to join', 'Join now', 'Join anyway'];

    for (const text of joinButtonTexts) {
      try {
        const button = await this.page.locator(`button:has-text("${text}")`).first();
        if (await button.isVisible({ timeout: 2000 })) {
          logger.info(`Found join button: "${text}"`, { correlationId });
          await button.click();
          logger.info('Clicked join button successfully', { correlationId });
          await this.page.waitForTimeout(5000); // Wait for join to process
          return;
        }
      } catch (e) {
        // Try next button text
      }
    }

    logger.warn('Could not find join button', { correlationId });
    throw new Error('Could not find join button');
  }

  async leave(): Promise<void> {
    const { logger, correlationId } = this.config;

    try {
      if (this.page) {
        logger.info('Closing browser...', { correlationId });
        await this.page.context().browser()?.close();
        logger.info('Browser closed', { correlationId });
      }
    } catch (error: any) {
      logger.error('Error closing browser', {
        error: error.message,
        correlationId
      });
    }
  }

  getPage(): any {
    return this.page;
  }
}
