// Simple test to join Google Meet with name input - handles popups
import createBrowserContext from './src/lib/chromium';
import { v4 } from 'uuid';

async function testJoinWithName() {
  const correlationId = v4();
  const meetingUrl = 'https://meet.google.com/oss-kqzr-ztg';
  const botName = 'LiveTranslate-Bot';

  console.log('🧪 Testing Google Meet Join with Name Input');
  console.log('Meeting:', meetingUrl);
  console.log('Bot Name:', botName);
  console.log('Correlation ID:', correlationId);
  console.log();

  try {
    console.log('🚀 Creating browser context...');
    const page = await createBrowserContext(meetingUrl, correlationId);

    console.log('📍 Navigating to meeting...');
    await page.goto(meetingUrl, { waitUntil: 'networkidle', timeout: 60000 });

    console.log('⏳ Waiting for page to load...');
    await page.waitForTimeout(5000);

    // Try to dismiss any Google Sign In popups
    console.log('🚫 Checking for Google Sign In popup...');
    try {
      const closeButtons = [
        'button[aria-label="Close"]',
        'button[aria-label="Dismiss"]',
        '[data-dismiss]',
        '.VfPpkd-Bz112c-LgbsSe', // Google Material Design close button
      ];

      for (const selector of closeButtons) {
        try {
          const button = await page.locator(selector).first();
          if (await button.isVisible({ timeout: 2000 })) {
            console.log(`   Found close button with selector: ${selector}`);
            await button.click();
            console.log('   ✅ Dismissed popup');
            await page.waitForTimeout(1000);
            break;
          }
        } catch (e) {
          // Button not found or not visible, continue
        }
      }
    } catch (e) {
      console.log('   No popup found or already dismissed');
    }

    // Try to dismiss "Continue without microphone and camera" button
    console.log('🔇 Checking for device permission dialog...');
    try {
      const deviceButton = await page.getByRole('button', { name: 'Continue without microphone and camera' });
      if (await deviceButton.isVisible({ timeout: 5000 })) {
        console.log('   Found device dialog');
        await deviceButton.click();
        console.log('   ✅ Dismissed device dialog');
        await page.waitForTimeout(2000);
      }
    } catch (e) {
      console.log('   No device dialog found');
    }

    console.log('📝 Looking for name input...');

    // Try multiple selectors for the name input
    const nameSelectors = [
      'input[type="text"][aria-label="Your name"]',
      'input[type="text"][placeholder*="name" i]',
      'input[type="text"]',
    ];

    let nameInput = null;
    for (const selector of nameSelectors) {
      try {
        const input = await page.locator(selector).first();
        if (await input.isVisible({ timeout: 3000 })) {
          nameInput = input;
          console.log(`   ✅ Found name input with selector: ${selector}`);
          break;
        }
      } catch (e) {
        // Try next selector
      }
    }

    if (nameInput) {
      console.log(`   Filling in name: ${botName}`);
      await nameInput.fill(botName);
      console.log('   ✅ Name filled!');
      await page.waitForTimeout(2000);

      console.log('📸 Taking screenshot after name filled...');
      await page.screenshot({ path: '/tmp/screenapp-name-filled.png' });
    } else {
      console.log('   ⚠️  Could not find name input');
      console.log('📸 Taking screenshot of current state...');
      await page.screenshot({ path: '/tmp/screenapp-no-name-input.png' });
    }

    console.log('🚪 Looking for join button...');
    const joinButtonTexts = ['Ask to join', 'Join now', 'Join anyway'];

    let joined = false;
    for (const text of joinButtonTexts) {
      try {
        const button = await page.locator(`button:has-text("${text}")`).first();
        if (await button.isVisible({ timeout: 2000 })) {
          console.log(`   ✅ Found join button: "${text}"`);
          await button.click();
          console.log('   Clicked join button!');
          joined = true;
          break;
        }
      } catch (e) {
        // Try next button text
      }
    }

    if (joined) {
      console.log('⏳ Waiting to see if we join...');
      await page.waitForTimeout(10000);

      console.log('📸 Taking screenshot after join...');
      await page.screenshot({ path: '/tmp/screenapp-after-join.png' });
      console.log('   Screenshots saved to /tmp/screenapp-*.png');

      console.log();
      console.log('✅ Test complete - check browser window and screenshots');
    } else {
      console.log('⚠️  Could not find join button');
      console.log('📸 Taking final screenshot...');
      await page.screenshot({ path: '/tmp/screenapp-no-join-button.png' });
    }

    console.log();
    console.log('🔍 Keeping browser open for 60 seconds to inspect...');
    await page.waitForTimeout(60000);

    console.log('🧹 Cleaning up...');
    await page.context().browser()?.close();

    console.log('✅ Done');
  } catch (error) {
    console.error('❌ Error:', error.message);
    throw error;
  }
}

testJoinWithName().catch(console.error);
