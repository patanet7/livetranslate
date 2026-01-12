// Test ScreenApp's GoogleMeetBot to see if it can join
import createBrowserContext from './src/lib/chromium';
import { v4 } from 'uuid';

async function testJoin() {
  const correlationId = v4();
  const meetingUrl = 'https://meet.google.com/oss-kqzr-ztg';

  console.log('🧪 Testing ScreenApp Google Meet Join');
  console.log('Meeting:', meetingUrl);
  console.log('Correlation ID:', correlationId);
  console.log();

  try {
    console.log('🚀 Creating browser context...');
    const page = await createBrowserContext(meetingUrl, correlationId);

    console.log('📍 Navigating to meeting...');
    await page.goto(meetingUrl, { waitUntil: 'networkidle', timeout: 60000 });

    console.log('⏳ Waiting for page to load...');
    await page.waitForTimeout(3000);

    console.log('📝 Looking for name input...');
    const nameInput = await page.locator('input[type="text"][placeholder*="name" i]').first();
    if (await nameInput.isVisible()) {
      console.log('✅ Found name input!');
      await nameInput.fill('LiveTranslate-ScreenApp-Bot');
      console.log('   Filled in name: LiveTranslate-ScreenApp-Bot');
      await page.waitForTimeout(1000);
    }

    console.log('🚪 Looking for join button...');
    const joinButton = await page.locator('button:has-text("Ask to join"), button:has-text("Join now")').first();
    if (await joinButton.isVisible()) {
      console.log('✅ Found join button!');
      console.log('📸 Taking screenshot before join...');
      await page.screenshot({ path: '/tmp/screenapp-before-join.png' });

      await joinButton.click();
      console.log('   Clicked join button!');

      console.log('⏳ Waiting to see if we join...');
      await page.waitForTimeout(10000);

      console.log('📸 Taking screenshot after join...');
      await page.screenshot({ path: '/tmp/screenapp-after-join.png' });
      console.log('   Screenshots saved to /tmp/screenapp-*.png');
    } else {
      console.log('⚠️  Could not find join button');
    }

    console.log();
    console.log('🧹 Cleaning up...');
    await page.context().browser()?.close();

    console.log('✅ Test complete');
  } catch (error) {
    console.error('❌ Error:', error);
    throw error;
  }
}

testJoin().catch(console.error);
