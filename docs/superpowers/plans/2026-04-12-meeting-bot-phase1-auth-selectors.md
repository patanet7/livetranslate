# Meeting Bot Phase 1: Authentication & Selectors

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable the meeting bot to join Google Meet without bot detection by using persistent browser profiles with a signed-in Google account.

**Architecture:** Playwright browser launches with a persistent `userDataDir` instead of ephemeral context. One-time manual sign-in saves cookies/session. Subsequent headless launches reuse the authenticated profile.

**Tech Stack:** Playwright, TypeScript, Express

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/lib/chromium.ts` | Browser context factory — add persistent profile support |
| `src/lib/auth_manager.ts` | **NEW** — Auth session manager (check status, launch setup flow) |
| `src/chat/selectors.ts` | Google Meet DOM selectors — update for 2026 UI |
| `src/api_server.ts` | HTTP endpoints — add `/api/auth/setup`, `/api/auth/status` |
| `src/bots/GoogleMeetBot.ts` | Bot class — wire persistent profile path |
| `src/config.ts` | Config — add `CHROME_PROFILE_DIR` env var |

---

## Task 1: Add CHROME_PROFILE_DIR to Config

**Files:**
- Modify: `modules/meeting-bot-service/src/config.ts`

- [ ] **Step 1: Read config.ts to understand structure**

Run: Read the config file to see existing pattern

- [ ] **Step 2: Add chromeProfileDir field**

```typescript
// Add to the config object
chromeProfileDir: process.env.CHROME_PROFILE_DIR || '/data/chrome-profile',
```

- [ ] **Step 3: Commit**

```bash
git add modules/meeting-bot-service/src/config.ts
git commit -m "$(cat <<'EOF'
feat(meeting-bot): add CHROME_PROFILE_DIR config

Persistent browser profile directory for Google auth.
EOF
)"
```

---

## Task 2: Create AuthManager Module

**Files:**
- Create: `modules/meeting-bot-service/src/lib/auth_manager.ts`
- Test: Manual verification (auth check involves filesystem)

- [ ] **Step 1: Create auth_manager.ts with profile checking**

```typescript
/**
 * AuthManager — handles persistent browser profile authentication.
 * 
 * The profile directory stores cookies, localStorage, and session data
 * from a manual Google sign-in, allowing subsequent headless launches
 * to be already authenticated.
 */

import { chromium, Browser, Page } from 'playwright';
import * as fs from 'fs';
import * as path from 'path';
import config from '../config';
import { Logger } from 'winston';

export interface AuthStatus {
  authenticated: boolean;
  profileExists: boolean;
  account?: string;
  lastAuthTime?: string;
}

export class AuthManager {
  private logger: Logger;
  private profileDir: string;

  constructor(logger: Logger) {
    this.logger = logger;
    this.profileDir = config.chromeProfileDir;
  }

  /**
   * Check if an authenticated profile exists.
   */
  async checkStatus(): Promise<AuthStatus> {
    const profileExists = fs.existsSync(this.profileDir);
    
    if (!profileExists) {
      return { authenticated: false, profileExists: false };
    }

    // Check for key files that indicate a complete profile
    const cookiesPath = path.join(this.profileDir, 'Default', 'Cookies');
    const localStatePath = path.join(this.profileDir, 'Local State');
    
    const hasCookies = fs.existsSync(cookiesPath);
    const hasLocalState = fs.existsSync(localStatePath);

    if (!hasCookies || !hasLocalState) {
      return { authenticated: false, profileExists: true };
    }

    // Profile exists and has auth data — assume authenticated
    // Full verification would require launching browser
    const stats = fs.statSync(cookiesPath);
    
    return {
      authenticated: true,
      profileExists: true,
      lastAuthTime: stats.mtime.toISOString(),
    };
  }

  /**
   * Launch a HEADED browser for manual Google sign-in.
   * Returns when user completes sign-in or closes browser.
   */
  async launchSetupFlow(): Promise<AuthStatus> {
    this.logger.info('Launching headed browser for Google sign-in...', {
      profileDir: this.profileDir,
    });

    // Ensure profile directory exists
    if (!fs.existsSync(this.profileDir)) {
      fs.mkdirSync(this.profileDir, { recursive: true });
    }

    const browser = await chromium.launchPersistentContext(this.profileDir, {
      headless: false, // HEADED for manual sign-in
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--window-size=1280,800',
      ],
      viewport: { width: 1280, height: 800 },
    });

    const page = browser.pages()[0] || await browser.newPage();

    // Navigate to Google sign-in
    await page.goto('https://accounts.google.com/signin');
    this.logger.info('Opened Google sign-in page. Waiting for user to complete sign-in...');

    // Wait for either:
    // 1. User navigates to myaccount.google.com (successful sign-in)
    // 2. Browser closes (user cancelled)
    try {
      await Promise.race([
        page.waitForURL('**/myaccount.google.com/**', { timeout: 300000 }), // 5 min
        page.waitForURL('**/mail.google.com/**', { timeout: 300000 }),
        page.waitForURL('**/google.com/?**', { timeout: 300000 }),
        this.waitForBrowserClose(browser),
      ]);

      // Check if we reached an authenticated state
      const url = page.url();
      const isAuthenticated = 
        url.includes('myaccount.google.com') ||
        url.includes('mail.google.com') ||
        (url.includes('google.com') && !url.includes('accounts.google.com'));

      if (isAuthenticated) {
        this.logger.info('Google sign-in successful!', { url });
        
        // Try to get account email
        let account: string | undefined;
        try {
          // Navigate to account page to get email
          await page.goto('https://myaccount.google.com/email');
          await page.waitForTimeout(2000);
          const emailElement = await page.$('text=@gmail.com');
          if (emailElement) {
            account = await emailElement.textContent() ?? undefined;
          }
        } catch (e) {
          // Not critical if we can't get the email
        }

        await browser.close();
        return {
          authenticated: true,
          profileExists: true,
          account,
          lastAuthTime: new Date().toISOString(),
        };
      }
    } catch (error: any) {
      this.logger.warn('Auth setup flow ended', { error: error.message });
    }

    // Browser closed or timeout
    try {
      await browser.close();
    } catch {
      // Already closed
    }

    return this.checkStatus();
  }

  private waitForBrowserClose(browser: any): Promise<void> {
    return new Promise((resolve) => {
      browser.on('disconnected', () => resolve());
    });
  }

  /**
   * Get the profile directory path for use by chromium.ts
   */
  getProfileDir(): string {
    return this.profileDir;
  }
}
```

- [ ] **Step 2: Export from lib/index.ts**

Add to `modules/meeting-bot-service/src/lib/index.ts`:

```typescript
export { AuthManager, AuthStatus } from './auth_manager';
```

- [ ] **Step 3: Commit**

```bash
git add modules/meeting-bot-service/src/lib/auth_manager.ts modules/meeting-bot-service/src/lib/index.ts
git commit -m "$(cat <<'EOF'
feat(meeting-bot): add AuthManager for persistent Google auth

- checkStatus(): verify profile exists with cookies
- launchSetupFlow(): headed browser for manual sign-in
- Waits for user to complete Google sign-in flow
EOF
)"
```

---

## Task 3: Update chromium.ts for Persistent Profile

**Files:**
- Modify: `modules/meeting-bot-service/src/lib/chromium.ts`

- [ ] **Step 1: Add persistent context support**

Replace the entire `chromium.ts` with this updated version that supports both ephemeral and persistent contexts:

```typescript
import { chromium, Page, BrowserContext } from 'playwright';
import config from '../config';
import * as fs from 'fs';

/**
 * Browser launch options shared between ephemeral and persistent contexts.
 */
const BROWSER_ARGS = [
  '--no-sandbox',
  '--disable-setuid-sandbox',
  '--disable-dev-shm-usage',
  '--window-size=1920,1080',
  '--use-fake-ui-for-media-stream',
  '--use-fake-device-for-media-stream',
  '--auto-select-desktop-capture-source=Entire screen',
  '--autoplay-policy=no-user-gesture-required',
  '--enable-features=SharedArrayBuffer',
  // Stealth args to avoid detection
  '--disable-blink-features=AutomationControlled',
  '--disable-features=IsolateOrigins,site-per-process',
  '--disable-infobars',
  '--enable-webgl',
  '--use-gl=swiftshader',
];

const CONTEXT_OPTIONS = {
  viewport: { width: 1920, height: 1080 },
  permissions: ['camera', 'microphone'] as ('camera' | 'microphone')[],
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
  ignoreHTTPSErrors: true,
  locale: 'en-US',
  timezoneId: 'America/New_York',
  screen: { width: 1920, height: 1080 },
  colorScheme: 'light' as const,
  deviceScaleFactor: 1,
  isMobile: false,
  hasTouch: false,
};

/**
 * Inject stealth scripts to avoid bot detection.
 */
async function injectStealthScripts(page: Page): Promise<void> {
  await page.addInitScript(() => {
    // Remove webdriver flag
    Object.defineProperty(navigator, 'webdriver', {
      get: () => undefined,
    });

    // Fake plugins array
    Object.defineProperty(navigator, 'plugins', {
      get: () => [1, 2, 3, 4, 5],
    });

    // Fake languages
    Object.defineProperty(navigator, 'languages', {
      get: () => ['en-US', 'en'],
    });

    // Override chrome runtime
    (window as any).chrome = {
      runtime: {},
    };
  });
}

interface CreateBrowserOptions {
  usePersistentProfile?: boolean;
  headless?: boolean;
}

/**
 * Creates a new browser context and page for meeting bot automation.
 * 
 * @param url - Initial URL (for correlation tracking)
 * @param correlationId - Request correlation ID for logging
 * @param options - Browser options (persistent profile, headless mode)
 */
export default async function createBrowserContext(
  url: string,
  correlationId: string,
  options: CreateBrowserOptions = {}
): Promise<Page> {
  const { usePersistentProfile = true, headless = true } = options;
  
  const profileDir = config.chromeProfileDir;
  const profileExists = fs.existsSync(profileDir);

  // Use persistent context if profile exists and requested
  if (usePersistentProfile && profileExists) {
    const context = await chromium.launchPersistentContext(profileDir, {
      headless,
      executablePath: config.chromeExecutablePath,
      args: BROWSER_ARGS,
      ...CONTEXT_OPTIONS,
    });

    const page = context.pages()[0] || await context.newPage();
    await injectStealthScripts(page);
    return page;
  }

  // Fall back to ephemeral context (no profile)
  const browser = await chromium.launch({
    headless,
    executablePath: config.chromeExecutablePath,
    args: BROWSER_ARGS,
  });

  const context: BrowserContext = await browser.newContext(CONTEXT_OPTIONS);
  const page = await context.newPage();
  await injectStealthScripts(page);

  return page;
}
```

- [ ] **Step 2: Commit**

```bash
git add modules/meeting-bot-service/src/lib/chromium.ts
git commit -m "$(cat <<'EOF'
feat(meeting-bot): add persistent browser profile support

- launchPersistentContext() reuses saved Google auth
- Shared stealth scripts between persistent/ephemeral modes
- Configurable headless mode for headed setup flow
EOF
)"
```

---

## Task 4: Add Auth API Endpoints

**Files:**
- Modify: `modules/meeting-bot-service/src/api_server.ts`

- [ ] **Step 1: Import AuthManager at top of file**

Add after existing imports:

```typescript
import { AuthManager } from './lib/auth_manager';
import { loggerFactory } from './util/logger';
```

- [ ] **Step 2: Create auth manager instance**

Add after `const activeBots = new Map<string, GoogleMeetBot>();`:

```typescript
// Auth manager for persistent profile authentication
const authLogger = loggerFactory('auth-manager', 'auth');
const authManager = new AuthManager(authLogger);
```

- [ ] **Step 3: Add /api/auth/status endpoint**

Add before the health check endpoint:

```typescript
/**
 * GET /api/auth/status
 * Check if the bot has a valid authenticated Google profile
 */
app.get('/api/auth/status', async (req: Request, res: Response) => {
  try {
    const status = await authManager.checkStatus();
    return res.status(200).json({
      success: true,
      ...status
    });
  } catch (error: any) {
    return res.status(500).json({
      success: false,
      error: error.message
    });
  }
});
```

- [ ] **Step 4: Add /api/auth/setup endpoint**

Add after the status endpoint:

```typescript
/**
 * POST /api/auth/setup
 * Launch headed browser for manual Google sign-in.
 * 
 * IMPORTANT: This endpoint blocks until the user completes sign-in
 * or closes the browser. It opens a visible browser window.
 * 
 * @tag-user This step requires manual Google sign-in
 */
app.post('/api/auth/setup', async (req: Request, res: Response) => {
  try {
    console.log('🔐 Starting Google auth setup - a browser window will open...');
    console.log('👤 @USER: Please sign in to your Google account in the browser window');
    
    const status = await authManager.launchSetupFlow();
    
    if (status.authenticated) {
      console.log('✅ Google auth setup complete!');
    } else {
      console.log('⚠️ Google auth setup incomplete - browser was closed');
    }
    
    return res.status(200).json({
      success: status.authenticated,
      ...status
    });
  } catch (error: any) {
    return res.status(500).json({
      success: false,
      error: error.message
    });
  }
});
```

- [ ] **Step 5: Commit**

```bash
git add modules/meeting-bot-service/src/api_server.ts
git commit -m "$(cat <<'EOF'
feat(meeting-bot): add auth setup and status endpoints

- GET /api/auth/status - check if authenticated profile exists
- POST /api/auth/setup - launch headed browser for Google sign-in

@tag-user: /api/auth/setup requires manual Google sign-in
EOF
)"
```

---

## Task 5: Update Selectors for 2026 Google Meet UI

**Files:**
- Modify: `modules/meeting-bot-service/src/chat/selectors.ts`

- [ ] **Step 1: Update selectors with multiple fallbacks**

Replace the entire file with updated 2026 selectors:

```typescript
/**
 * Google Meet DOM selectors — single source of truth.
 * 
 * Updated: 2026-04-12
 * Primary strategy: aria-label (survives most restructures).
 * Secondary: data attributes and button text content.
 * 
 * Each selector group includes multiple fallbacks for resilience.
 */

import { Page, Locator } from 'playwright';

// =============================================================================
// PRE-JOIN SELECTORS
// =============================================================================

/** Turn off microphone before joining */
export const MIC_OFF_SELECTORS = [
  'button[aria-label*="Turn off microphone"]',
  'button[data-is-muted="false"][aria-label*="microphone"]',
  '[data-tooltip*="microphone"] button',
];

/** Turn off camera before joining */
export const CAM_OFF_SELECTORS = [
  'button[aria-label*="Turn off camera"]',
  'button[data-is-muted="false"][aria-label*="camera"]',
  '[data-tooltip*="camera"] button',
];

/** Join meeting buttons */
export const JOIN_BUTTON_SELECTORS = [
  'button[aria-label*="Join now"]',
  'button[aria-label*="Ask to join"]',
  'button:has-text("Join now")',
  'button:has-text("Ask to join")',
  '[data-idom-class*="join"] button',
];

/** Name input for guest joining */
export const NAME_INPUT = 'input[type="text"][aria-label="Your name"]';

/** Continue without mic/cam dialogs */
export const CONTINUE_WITHOUT_SELECTORS = [
  'button:has-text("Continue without microphone")',
  'button:has-text("Continue without camera")',
  'button:has-text("Continue without microphone and camera")',
  'button:has-text("Dismiss")',
];

// =============================================================================
// IN-MEETING CONTROLS
// =============================================================================

/** Leave meeting */
export const LEAVE_BUTTON_SELECTORS = [
  'button[aria-label*="Leave call"]',
  'button[aria-label*="Leave meeting"]',
  'button[aria-label*="End call"]',
  '[data-tooltip*="Leave"] button',
];

/** Toggle captions on */
export const CAPTIONS_ON_SELECTORS = [
  'button[aria-label*="Turn on captions"]',
  'button[aria-label*="Enable captions"]',
  '[data-tooltip*="captions"][data-is-toggled="false"] button',
];

/** Toggle captions off */
export const CAPTIONS_OFF_SELECTORS = [
  'button[aria-label*="Turn off captions"]',
  'button[aria-label*="Disable captions"]',
  '[data-tooltip*="captions"][data-is-toggled="true"] button',
];

/** Captions display region */
export const CAPTIONS_REGION_SELECTORS = [
  '[role="region"][aria-label*="Captions"]',
  '[aria-live="polite"][aria-label*="Captions"]',
  '.a4cQT', // Google's caption text class
  '[jscontroller="D1tHje"]', // Caption controller
];

// =============================================================================
// CHAT SELECTORS
// =============================================================================

/** Open chat panel */
export const CHAT_BUTTON_SELECTORS = [
  'button[aria-label*="Chat with everyone"]',
  'button[aria-label*="Open chat"]',
  '[data-panel-id="chat"] button',
  '[data-tooltip*="Chat"] button',
];

/** Chat input textarea */
export const CHAT_INPUT_SELECTORS = [
  'textarea[aria-label*="Send a message"]',
  'textarea[aria-label*="Send a message to everyone"]',
  '[data-panel-id="chat"] textarea',
  'textarea[placeholder*="Send a message"]',
];

/** Chat messages container */
export const CHAT_MESSAGES_SELECTORS = [
  '[aria-label="Chat messages"]',
  '[data-message-id]',
  '[role="log"][aria-label*="Chat"]',
];

/** Individual chat message with ID */
export const CHAT_MESSAGE_ITEM = '[data-message-id]';

// =============================================================================
// DISMISSAL DIALOGS
// =============================================================================

/** Got it / OK buttons for dialogs */
export const GOT_IT_SELECTORS = [
  'button:has-text("Got it")',
  'button:has-text("OK")',
  'button:has-text("Dismiss")',
  'button[jsname="EszDEe"]',
];

/** Close / X buttons for dialogs */
export const CLOSE_BUTTON_SELECTORS = [
  'button[aria-label="Close"]',
  'button[aria-label="Dismiss"]',
  'button[aria-label*="close"]',
  '[data-dismiss] button',
];

// =============================================================================
// EXIT DETECTION
// =============================================================================

/** Indicators that we've left the meeting */
export const LEFT_MEETING_SELECTORS = [
  'div[role="heading"]:has-text("You left the meeting")',
  ':has-text("You\'ve left the meeting")',
  '[data-call-ended="true"]',
  ':has-text("Return to home screen")',
];

// =============================================================================
// PEOPLE PANEL
// =============================================================================

/** Open people panel */
export const PEOPLE_BUTTON_SELECTORS = [
  'button[aria-label^="Show everyone"]',
  'button[aria-label^="People"]',
  '[data-panel-id="people"] button',
];

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

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

/**
 * Click the first visible element from a list of selectors.
 * @returns true if clicked, false if none found
 */
export async function clickFirst(page: Page, selectors: string[], options?: { timeout?: number }): Promise<boolean> {
  const { timeout = 3000 } = options || {};
  
  for (const sel of selectors) {
    try {
      const loc = page.locator(sel).first();
      await loc.click({ timeout });
      return true;
    } catch {
      // Try next selector
    }
  }
  return false;
}

/**
 * Wait for any of the selectors to become visible.
 */
export async function waitForAny(page: Page, selectors: string[], timeout = 30000): Promise<Locator | null> {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    const visible = await findVisible(page, selectors);
    if (visible) return visible;
    await page.waitForTimeout(500);
  }
  
  return null;
}

// =============================================================================
// LEGACY EXPORTS (for backward compatibility)
// =============================================================================

// Keep single-value exports for existing code that imports them directly
export const CHAT_BUTTON = CHAT_BUTTON_SELECTORS[0];
export const CHAT_PANEL = '[aria-label="Chat panel"]';
export const CHAT_INPUT = CHAT_INPUT_SELECTORS[0];
export const CHAT_SEND_BUTTON = '[aria-label="Send a message"]';
export const CHAT_MESSAGES_CONTAINER = CHAT_MESSAGES_SELECTORS[0];
export const LEAVE_CALL_BUTTON = LEAVE_BUTTON_SELECTORS[0];
export const PEOPLE_BUTTON_PREFIX = PEOPLE_BUTTON_SELECTORS[0];
export const CAMERA_BUTTON = '[aria-label="Turn on camera"]';
export const MIC_BUTTON = '[aria-label="Turn on microphone"]';
export const ASK_TO_JOIN_BUTTON = JOIN_BUTTON_SELECTORS[1];
export const JOIN_NOW_BUTTON = JOIN_BUTTON_SELECTORS[0];
export const CONTINUE_WITHOUT_MIC_CAM = CONTINUE_WITHOUT_SELECTORS[2];
export const GOT_IT_BUTTON = GOT_IT_SELECTORS[0];
export const CC_BUTTON = CAPTIONS_ON_SELECTORS[0];
export const CC_CONTAINER = CAPTIONS_REGION_SELECTORS[3];
export const CC_TEXT_SPAN = CAPTIONS_REGION_SELECTORS[2];
```

- [ ] **Step 2: Commit**

```bash
git add modules/meeting-bot-service/src/chat/selectors.ts
git commit -m "$(cat <<'EOF'
feat(meeting-bot): update selectors for 2026 Google Meet UI

- Multiple fallback selectors per action for resilience
- Added clickFirst(), waitForAny() utilities
- Preserved legacy single-value exports for compatibility
- Grouped by: pre-join, in-meeting, chat, dismissals, exit
EOF
)"
```

---

## Task 6: Update GoogleMeetBot to Use New Selectors

**Files:**
- Modify: `modules/meeting-bot-service/src/bots/GoogleMeetBot.ts`

- [ ] **Step 1: Update imports at top of file**

Replace selector imports with:

```typescript
import {
  clickFirst,
  findVisible,
  waitForAny,
  JOIN_BUTTON_SELECTORS,
  MIC_OFF_SELECTORS,
  CAM_OFF_SELECTORS,
  CONTINUE_WITHOUT_SELECTORS,
  GOT_IT_SELECTORS,
  CLOSE_BUTTON_SELECTORS,
  LEAVE_BUTTON_SELECTORS,
  CAPTIONS_ON_SELECTORS,
  CHAT_BUTTON_SELECTORS,
  CHAT_INPUT_SELECTORS,
  LEFT_MEETING_SELECTORS,
  NAME_INPUT,
} from '../chat/selectors';
```

- [ ] **Step 2: Replace hardcoded selector arrays with imported ones**

In `dismissGoogleSignInPopup()`, replace the `gotItSelectors` array with:

```typescript
// Use imported GOT_IT_SELECTORS instead of hardcoded array
await clickFirst(this.page, GOT_IT_SELECTORS);
```

In `dismissDeviceCheck()`, replace hardcoded selectors with:

```typescript
await clickFirst(this.page, CONTINUE_WITHOUT_SELECTORS);
```

- [ ] **Step 3: Update join button detection**

Replace `askToJoinButton` and `joinNowButton` logic with:

```typescript
// Try to find and click any join button
const joinClicked = await clickFirst(this.page, JOIN_BUTTON_SELECTORS, { timeout: 5000 });
if (!joinClicked) {
  throw new Error('Could not find join button');
}
```

- [ ] **Step 4: Update exit detection**

Replace meeting-ended detection with:

```typescript
// Check if we've left the meeting
const leftMeeting = await findVisible(this.page, LEFT_MEETING_SELECTORS);
if (leftMeeting) {
  this._logger.info('Detected meeting ended');
  // ... existing cleanup logic
}
```

- [ ] **Step 5: Commit**

```bash
git add modules/meeting-bot-service/src/bots/GoogleMeetBot.ts
git commit -m "$(cat <<'EOF'
refactor(meeting-bot): use new selector utilities in GoogleMeetBot

- Import selector arrays and utility functions
- Replace hardcoded selectors with resilient fallback arrays
- clickFirst() handles selector fallback automatically
EOF
)"
```

---

## Review Wave 1: Auth + Selectors Complete

At this point, the authentication foundation and updated selectors are in place.

**Manual verification needed:**

1. **Check auth status:**
   ```bash
   curl http://localhost:5005/api/auth/status
   ```
   Expected: `{"authenticated": false, "profileExists": false}`

2. **Run auth setup (requires human):**
   ```bash
   curl -X POST http://localhost:5005/api/auth/setup
   ```
   **@USER: A browser window will open. Please sign in to Google.**

3. **Verify auth persisted:**
   ```bash
   curl http://localhost:5005/api/auth/status
   ```
   Expected: `{"authenticated": true, "profileExists": true, "account": "...", "lastAuthTime": "..."}`

4. **Test bot join with auth:**
   ```bash
   curl -X POST http://localhost:5005/api/bot/join \
     -H "Content-Type: application/json" \
     -d '{"meetingUrl": "https://meet.google.com/xxx-xxxx-xxx", "botName": "TestBot", "botId": "test-1", "userId": "user-1"}'
   ```
   Expected: Bot joins meeting without being redirected to homepage.

---

## Success Criteria (Phase 1)

- [ ] `GET /api/auth/status` returns correct profile status
- [ ] `POST /api/auth/setup` opens headed browser for manual sign-in
- [ ] Profile persists in `/data/chrome-profile/` after sign-in
- [ ] Bot joins Google Meet without bot detection redirect
- [ ] New selectors work with current Google Meet UI
- [ ] Legacy selector exports maintain backward compatibility
