# Meeting Bot Service - ScreenApp Integration

## 🎉 SUCCESS! Google Meet Bot Detection Bypassed

The ScreenApp GoogleMeetBot has been successfully integrated into LiveTranslate. This Node.js service uses battle-tested browser automation that **successfully bypasses Google's bot detection**.

### What Works

✅ **Bot joins Google Meet meetings** - Successfully tested and verified
✅ **Name input filled** - Bot displays with custom name
✅ **Join button clicked** - Automatic admission flow
✅ **Cross-platform Chrome path** - Works on macOS and Docker/Linux
✅ **HTTP API wrapper** - Python orchestration service can call it

## Architecture

```
┌─────────────────────────────────────┐
│  Orchestration Service (Python)    │
│  - FastAPI                          │
│  - MeetingBotServiceClient         │
└─────────────┬───────────────────────┘
              │ HTTP
              │ (Port 5005)
              ▼
┌─────────────────────────────────────┐
│  Meeting Bot Service (Node.js)      │
│  - Express API                      │
│  - GoogleMeetBot (ScreenApp)       │
│  - Playwright + Stealth Plugins    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Google Meet                        │
│  ✅ Bot joins successfully!         │
└─────────────────────────────────────┘
```

## Quick Start

### 1. Start the Meeting Bot Service

```bash
cd modules/meeting-bot-service
npm install
npm run api
```

The service will start on **port 5005**.

### 2. Use from Python (Orchestration Service)

```python
from src.clients.meeting_bot_service_client import MeetingBotServiceClient

# Initialize client
client = MeetingBotServiceClient(base_url="http://localhost:5005")

# Join a meeting
response = await client.join_meeting(
    meeting_url="https://meet.google.com/abc-defg-hij",
    bot_name="LiveTranslate Bot",
    bot_id="bot-123",
    user_id="user-456"
)

print(f"Success: {response.success}")
print(f"Bot ID: {response.botId}")
print(f"Correlation ID: {response.correlationId}")
```

## API Endpoints

### `POST /api/bot/join`
Join a Google Meet meeting.

**Request:**
```json
{
  "meetingUrl": "https://meet.google.com/abc-defg-hij",
  "botName": "LiveTranslate Bot",
  "botId": "bot-123",
  "userId": "user-456",
  "teamId": "team-789",
  "timezone": "America/Los_Angeles"
}
```

**Response:**
```json
{
  "success": true,
  "botId": "bot-123",
  "correlationId": "8039c59a-15fc-4448-8137-819d8eea4ae0",
  "message": "Bot is joining the meeting"
}
```

### `GET /api/bot/status/:botId`
Get the current status of a bot.

**Response:**
```json
{
  "success": true,
  "botId": "bot-123",
  "state": "in_meeting"
}
```

### `POST /api/bot/leave/:botId`
Leave a meeting and cleanup.

**Response:**
```json
{
  "success": true,
  "botId": "bot-123",
  "message": "Bot left the meeting"
}
```

### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "meeting-bot-service",
  "activeBots": 2,
  "timestamp": "2025-10-29T00:52:00.000Z"
}
```

## Configuration

### Chrome Path (Automatic Detection)

The service automatically detects the Chrome path based on the platform:

- **macOS**: `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`
- **Linux/Docker**: `/usr/bin/google-chrome`

You can override this with the environment variable:
```bash
export CHROME_EXECUTABLE_PATH="/custom/path/to/chrome"
```

### Environment Variables

```bash
# Optional - Service port (default: 5005)
PORT=5005

# Optional - Override Chrome path
CHROME_EXECUTABLE_PATH=/usr/bin/google-chrome

# Optional - Storage configuration
GCP_DEFAULT_REGION=us-west1
GCP_MISC_BUCKET=my-bucket
S3_ENDPOINT=https://s3.amazonaws.com
S3_BUCKET_NAME=my-bucket
UPLOADER_TYPE=s3
```

## Docker Support

The service is Docker-ready with automatic Chrome path detection.

### Dockerfile (example)

```dockerfile
FROM node:18

# Install Chrome for Docker
RUN apt-get update && apt-get install -y \\
    google-chrome-stable \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5005
CMD ["npm", "run", "api"]
```

## Testing

### Manual Test

```bash
# Start the service
npm run api

# In another terminal, test the join endpoint
curl -X POST http://localhost:5005/api/bot/join \\
  -H "Content-Type: application/json" \\
  -d '{
    "meetingUrl": "https://meet.google.com/oss-kqzr-ztg",
    "botName": "Test Bot",
    "botId": "test-123",
    "userId": "user-456"
  }'
```

### Python Integration Test

```python
import asyncio
from src.clients.meeting_bot_service_client import MeetingBotServiceClient

async def test_join():
    client = MeetingBotServiceClient()

    # Check health
    health = await client.health_check()
    print(f"Service status: {health['status']}")

    # Join meeting
    response = await client.join_meeting(
        meeting_url="https://meet.google.com/oss-kqzr-ztg",
        bot_name="LiveTranslate Test Bot",
        bot_id="test-bot-1",
        user_id="test-user-1"
    )

    print(f"Join result: {response}")

asyncio.run(test_join())
```

## Key Features from ScreenApp

### 1. Bot Detection Bypass
- **playwright-extra** with stealth plugins
- **puppeteer-extra-plugin-stealth**
- Custom user agent (Chrome 135)
- Battle-tested browser args

### 2. Automatic Join Flow
- Dismisses device permission dialogs
- Fills in bot name
- Clicks appropriate join button ("Ask to join", "Join now", "Join anyway")
- Handles waiting room scenarios

### 3. Recording & Upload
- Disk-based recording with buffer queue
- S3-compatible storage support
- Retry logic with exponential backoff
- Multi-part upload for large files

## Next Steps

1. **Wire into orchestration service** - Update bot management to use this service
2. **Add audio streaming** - Connect bot audio to whisper service
3. **Docker deployment** - Create production-ready container
4. **Multi-platform support** - Add Teams and Zoom bots (already in ScreenApp code)

## Credits

This integration uses the battle-tested GoogleMeetBot from [ScreenApp](https://screenapp.io), which has successfully handled thousands of meeting bot sessions in production.
