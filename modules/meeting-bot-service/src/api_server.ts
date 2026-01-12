/**
 * HTTP API Server for Google Meet Bot
 *
 * This service wraps the battle-tested ScreenApp GoogleMeetBot with a simple HTTP API
 * so the Python orchestration service can call it.
 */

import express, { Request, Response } from 'express';
import { GoogleMeetBot } from './bots/GoogleMeetBot';
import { loggerFactory } from './util/logger';
import { v4 } from 'uuid';
import DiskUploader from './middleware/disk-uploader';

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 5005;

// Store active bot sessions
const activeBots = new Map<string, GoogleMeetBot>();

interface JoinRequest {
  meetingUrl: string;
  botName: string;
  botId: string;
  userId: string;
  teamId?: string;
  timezone?: string;
  eventId?: string;
  bearerToken?: string;
  orchestrationUrl?: string;  // WebSocket URL for audio streaming
}

interface JoinResponse {
  success: boolean;
  botId: string;
  correlationId: string;
  message?: string;
  error?: string;
}

/**
 * POST /api/bot/join
 * Join a Google Meet meeting
 */
app.post('/api/bot/join', async (req: Request, res: Response) => {
  const {
    meetingUrl,
    botName,
    botId,
    userId,
    teamId = 'livetranslate-team',
    timezone = 'UTC',
    eventId = v4(),
    bearerToken = 'internal-token',
    orchestrationUrl = 'ws://localhost:3000/api/audio/stream'
  }: JoinRequest = req.body;

  if (!meetingUrl || !botName || !botId || !userId) {
    return res.status(400).json({
      success: false,
      error: 'Missing required fields: meetingUrl, botName, botId, userId'
    });
  }

  const correlationId = v4();
  const logger = loggerFactory(correlationId, 'api');

  try {
    logger.info('API: Joining Google Meet', { meetingUrl, botName, botId, userId });

    const bot = new GoogleMeetBot(logger, correlationId);
    activeBots.set(botId, bot);

    // Initialize uploader
    const uploader = await DiskUploader.initialize(
      bearerToken,
      teamId,
      timezone,
      userId,
      botId,
      `LiveTranslate-${botName}`,
      correlationId,
      logger
    );

    // Join the meeting (this runs in background)
    bot.join({
      url: meetingUrl,
      name: botName,
      bearerToken,
      teamId,
      timezone,
      userId,
      eventId,
      botId,
      uploader,
      orchestrationUrl  // Pass orchestration URL for audio streaming
    }).catch((error) => {
      logger.error('Bot join failed', { error: error.message, botId });
      activeBots.delete(botId);
    });

    // Return immediately - bot is joining
    const response: JoinResponse = {
      success: true,
      botId,
      correlationId,
      message: 'Bot is joining the meeting'
    };

    return res.status(200).json(response);

  } catch (error: any) {
    logger.error('Failed to start bot', { error: error.message });

    const response: JoinResponse = {
      success: false,
      botId,
      correlationId,
      error: error.message
    };

    return res.status(500).json(response);
  }
});

/**
 * GET /api/bot/status/:botId
 * Get status of a bot
 */
app.get('/api/bot/status/:botId', (req: Request, res: Response) => {
  const { botId } = req.params;

  const bot = activeBots.get(botId);
  if (!bot) {
    return res.status(404).json({
      success: false,
      error: 'Bot not found'
    });
  }

  // Get bot state
  const state = bot.getState ? bot.getState() : 'unknown';

  return res.status(200).json({
    success: true,
    botId,
    state
  });
});

/**
 * POST /api/bot/leave/:botId
 * Leave a meeting and cleanup
 */
app.post('/api/bot/leave/:botId', async (req: Request, res: Response) => {
  const { botId } = req.params;

  const bot = activeBots.get(botId);
  if (!bot) {
    return res.status(404).json({
      success: false,
      error: 'Bot not found'
    });
  }

  try {
    // Leave the meeting
    await bot.leave();
    activeBots.delete(botId);

    return res.status(200).json({
      success: true,
      botId,
      message: 'Bot left the meeting'
    });

  } catch (error: any) {
    return res.status(500).json({
      success: false,
      botId,
      error: error.message
    });
  }
});

/**
 * GET /api/health
 * Health check endpoint
 */
app.get('/api/health', (req: Request, res: Response) => {
  return res.status(200).json({
    status: 'healthy',
    service: 'meeting-bot-service',
    activeBots: activeBots.size,
    timestamp: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🤖 Meeting Bot Service running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`🚀 Join endpoint: POST http://localhost:${PORT}/api/bot/join`);
});
