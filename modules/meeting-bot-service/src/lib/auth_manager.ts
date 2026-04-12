import { Logger } from 'winston';
import { chromium } from 'playwright';
import * as fs from 'fs';
import * as path from 'path';
import config from '../config';

/**
 * Represents the current authentication status and profile state
 */
export interface AuthStatus {
  authenticated: boolean;
  profileExists: boolean;
  account?: string;
  lastAuthTime?: string;
}

/**
 * Manages persistent Google authentication using browser profiles.
 * Stores browser context in a persistent profile directory for reuse across sessions.
 */
export class AuthManager {
  private _logger: Logger;
  private profileDir: string;

  constructor(logger: Logger) {
    this._logger = logger;
    this.profileDir = config.chromeProfileDir;
  }

  /**
   * Returns the path to the browser profile directory
   */
  getProfileDir(): string {
    return this.profileDir;
  }

  /**
   * Checks if the browser profile exists and contains authentication cookies
   * @returns Promise<AuthStatus> - Current authentication status
   */
  async checkStatus(): Promise<AuthStatus> {
    try {
      const profileDir = this.getProfileDir();
      const defaultDir = path.join(profileDir, 'Default');
      const cookiesFile = path.join(defaultDir, 'Cookies');
      const localStateFile = path.join(profileDir, 'Local State');

      const cookiesExist = fs.existsSync(cookiesFile);
      const localStateExists = fs.existsSync(localStateFile);
      const profileExists = cookiesExist && localStateExists;

      this._logger.debug('Checking auth status', {
        profileDir,
        profileExists,
        cookiesExist,
        localStateExists,
      });

      const status: AuthStatus = {
        authenticated: profileExists,
        profileExists,
      };

      // Try to read last modification time for lastAuthTime
      if (cookiesExist) {
        const stats = fs.statSync(cookiesFile);
        status.lastAuthTime = stats.mtime.toISOString();
      }

      return status;
    } catch (error) {
      this._logger.error('Error checking auth status', {
        error: error instanceof Error ? error.message : String(error),
      });
      return {
        authenticated: false,
        profileExists: false,
      };
    }
  }

  /**
   * Launches a headed browser for manual Google sign-in.
   * Waits for user to complete authentication or close the browser.
   * @returns Promise<AuthStatus> - Updated authentication status after sign-in
   */
  async launchSetupFlow(): Promise<AuthStatus> {
    const profileDir = this.getProfileDir();
    const defaultDir = path.join(profileDir, 'Default');

    // Ensure profile directory exists
    if (!fs.existsSync(profileDir)) {
      fs.mkdirSync(profileDir, { recursive: true });
      this._logger.info('Created profile directory', { profileDir });
    }

    let browser;
    try {
      this._logger.info('Launching setup flow for Google authentication', {
        profileDir,
      });

      // Launch persistent context (headed) for manual authentication
      // Use stealth args to avoid Google's bot detection
      const context = await chromium.launchPersistentContext(profileDir, {
        headless: false,
        channel: 'chrome',
        args: [
          '--disable-blink-features=AutomationControlled',
          '--disable-features=IsolateOrigins,site-per-process',
          '--disable-infobars',
          '--no-sandbox',
          '--window-size=1280,800',
        ],
        ignoreDefaultArgs: ['--enable-automation'],
      });

      browser = context.browser();

      // Create a new page
      const page = await context.newPage();

      // Navigate to Google sign-in
      await page.goto('https://accounts.google.com/signin', {
        waitUntil: 'domcontentloaded',
      });

      this._logger.info('Navigated to Google sign-in page');

      // Wait for authentication to complete by listening for navigation to known Google pages
      // The user will manually complete the sign-in flow
      let authCompleted = false;
      const timeoutMs = 5 * 60 * 1000; // 5 minutes
      const startTime = Date.now();

      try {
        // Wait for redirect to one of these URLs which indicates successful authentication
        await Promise.race([
          page.waitForURL('**/myaccount.google.com/**', { timeout: timeoutMs }),
          page.waitForURL('**/mail.google.com/**', { timeout: timeoutMs }),
        ]);
        authCompleted = true;
        this._logger.info('User successfully authenticated');
      } catch (waitError) {
        // Timeout or other error - check if time exceeded
        const elapsed = Date.now() - startTime;
        if (elapsed >= timeoutMs) {
          this._logger.warn('Authentication flow timeout after 5 minutes');
        } else {
          this._logger.debug('Waiting for authentication ended', {
            reason: waitError instanceof Error ? waitError.message : 'unknown',
          });
        }
      }

      // Wait a moment for the profile to be fully saved
      await new Promise(resolve => setTimeout(resolve, 1000));

      await context.close();

      // Check if authentication was successful
      const status = await this.checkStatus();

      if (authCompleted && status.authenticated) {
        this._logger.info('Authentication setup flow completed successfully');
      } else if (authCompleted) {
        this._logger.warn('User completed sign-in but profile cookies not found');
      }

      return status;
    } catch (error) {
      this._logger.error('Error during authentication setup flow', {
        error: error instanceof Error ? error.message : String(error),
      });

      // Close browser if still open
      if (browser) {
        try {
          await browser.close();
        } catch (closeError) {
          this._logger.error('Error closing browser', {
            error:
              closeError instanceof Error ? closeError.message : String(closeError),
          });
        }
      }

      // Return current status even if flow failed
      return this.checkStatus();
    }
  }
}
