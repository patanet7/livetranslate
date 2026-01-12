// Test ScreenApp's full GoogleMeetBot
import { GoogleMeetBot } from './src/bots/GoogleMeetBot';
import { loggerFactory } from './src/util/logger';
import { v4 } from 'uuid';
import DiskUploader from './src/middleware/disk-uploader';

async function testFullBot() {
  const correlationId = v4();
  const logger = loggerFactory(correlationId, 'test');
  const meetingUrl = 'https://meet.google.com/oss-kqzr-ztg';
  const botName = 'LiveTranslate-Bot';

  console.log('🧪 Testing ScreenApp GoogleMeetBot (FULL)');
  console.log('Meeting:', meetingUrl);
  console.log('Bot Name:', botName);
  console.log('Correlation ID:', correlationId);
  console.log();

  const bot = new GoogleMeetBot(logger, correlationId);

  // Use DiskUploader.initialize() since constructor is private
  const uploader = await DiskUploader.initialize(
    'test-token',      // token
    'test-team',       // teamId
    'America/Los_Angeles', // timezone
    'test-user',       // userId
    'test-bot-id',     // botId
    'LiveTranslate',   // namePrefix
    correlationId,     // tempFileId
    logger
  );

  try {
    console.log('🚀 Launching bot...');

    // Call the join method - this does EVERYTHING
    await bot.join({
      url: meetingUrl,
      name: botName,
      bearerToken: 'test-token', // dummy for testing
      teamId: 'test-team',
      timezone: 'America/Los_Angeles',
      userId: 'test-user',
      eventId: 'test-event',
      botId: 'test-bot-id',
      uploader
    });

    console.log('✅ Bot joined successfully!');
  } catch (error) {
    console.error('❌ Error:', error.message);

    // Leave browser open to inspect
    console.log('⏳ Keeping browser open for 30 seconds to inspect...');
    await new Promise(resolve => setTimeout(resolve, 30000));
  }
}

testFullBot().catch(console.error);
