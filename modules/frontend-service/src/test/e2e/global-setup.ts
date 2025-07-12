import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  // Start the browser for setup
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // Wait for the application to be ready
  try {
    await page.goto('http://localhost:3000');
    await page.waitForSelector('[data-testid="app-loaded"]', { timeout: 30000 });
    console.log('✅ Application is ready for testing');
  } catch (error) {
    console.error('❌ Application failed to load:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

export default globalSetup;