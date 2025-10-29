# Google Meet Bot Testing Guide

This guide explains how to test the Google Meet bot system to ensure it can successfully join meetings and log in.

## 📋 Prerequisites

Before testing the bot, ensure you have:

1. **Orchestration Service Running**
   ```bash
   cd modules/orchestration-service
   python src/main.py
   ```
   Service should be available at: http://localhost:3000

2. **Docker Running** (required for bot containers)
   ```bash
   docker --version  # Verify Docker is installed
   docker ps         # Verify Docker daemon is running
   ```

3. **Redis Running** (for bot command communication)
   ```bash
   # Option 1: Docker
   docker run -d -p 6379:6379 redis:alpine

   # Option 2: Local installation
   redis-server
   ```

4. **Python Dependencies**
   ```bash
   cd modules/orchestration-service
   pip install -r requirements.txt
   pip install httpx  # For test scripts
   ```

## 🚀 Quick Test (Recommended)

The fastest way to test if a bot can log into Google Meet:

```bash
cd modules/orchestration-service
python quick_bot_test.py
```

This will:
- ✅ Check if services are running
- ✅ Start a bot with a test meeting URL
- ✅ Monitor the bot in real-time
- ✅ Show clear success/failure status

### With Custom Meeting URL

```bash
python quick_bot_test.py --url https://meet.google.com/your-meeting-code
```

### Options

```bash
python quick_bot_test.py --help

Options:
  --url URL          Google Meet URL (default: test URL)
  --service URL      Orchestration service URL (default: http://localhost:3000)
  --timeout SECONDS  Maximum wait time (default: 90)
```

## 🔬 Comprehensive Test

For more detailed testing with full monitoring:

```bash
python test_bot_summon.py --meeting-url https://meet.google.com/abc-defg-hij
```

### Options

```bash
python test_bot_summon.py --help

Options:
  --meeting-url URL          Google Meet URL to join (required)
  --orchestration-url URL    Orchestration service URL (default: http://localhost:3000)
  --monitor-time SECONDS     How long to monitor (default: 60)
  --auto-stop                Automatically stop bot after test
  --language LANG            Transcription language (default: en)
  --enable-webcam            Enable virtual webcam output
```

### Example: Full Test with Webcam

```bash
python test_bot_summon.py \
  --meeting-url https://meet.google.com/abc-defg-hij \
  --monitor-time 120 \
  --enable-webcam \
  --auto-stop
```

## 📊 Expected Output

### Successful Bot Join

```
╔═══════════════════════════════════════════════════════════════════╗
║           GOOGLE MEET BOT - QUICK LOGIN TEST                      ║
╚═══════════════════════════════════════════════════════════════════╝

🔍 Checking orchestration service...
✅ Orchestration service is running

🚀 Starting bot for meeting: https://meet.google.com/test

✅ Bot started with ID: bot_abc12345

======================================================================
Monitoring Bot: bot_abc12345
======================================================================

12:34:56 | 🔄 SPAWNING - Bot container starting...
12:34:58 | 🚀 STARTING - Bot initializing...
12:35:03 | 🚪 JOINING - Bot joining Google Meet...
12:35:15 | ✅ ACTIVE - Bot is in the meeting!

SUCCESS! Bot successfully joined Google Meet!

Bot Details:
  • Container: livetranslate-bot-abc12345
  • Uptime: 19.3s
  • Healthy: True

======================================================================
✅ TEST PASSED
The bot successfully logged into Google Meet!
======================================================================
```

## 🎯 Bot Lifecycle States

The bot goes through these states:

1. **spawning** - Container is being created
2. **starting** - Bot process is initializing
3. **joining** - Bot is joining the Google Meet
4. **active** - ✅ Bot is in the meeting (SUCCESS!)
5. **completed** - Bot left meeting gracefully
6. **failed** - ❌ Bot encountered an error

## 🔍 Troubleshooting

### Service Not Available

```
❌ Orchestration service not available at http://localhost:3000
```

**Solution:**
```bash
cd modules/orchestration-service
python src/main.py
```

### Bot Stays in "spawning" State

**Possible causes:**
- Docker is not running
- Docker image not built
- Resource constraints

**Solution:**
```bash
# Check Docker
docker ps

# Build bot image
cd modules/bot-container
docker build -t livetranslate-bot .
```

### Bot Fails to Join Meeting

**Possible causes:**
- Invalid meeting URL
- Meeting requires authentication
- Meeting requires lobby approval
- Network connectivity issues

**Solution:**
- Use a valid Google Meet URL
- Create a test meeting without authentication requirements
- Check bot logs for specific error messages

### Redis Connection Issues

```
RuntimeError: Redis client not available
```

**Solution:**
```bash
# Start Redis
docker run -d -p 6379:6379 redis:alpine

# Or use local Redis
redis-server
```

## 🐛 Manual Testing via API

You can also test the bot system directly using the API:

### 1. Start a Bot

```bash
curl -X POST http://localhost:3000/api/bots/start \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_url": "https://meet.google.com/test",
    "user_token": "test-token",
    "user_id": "test-user",
    "language": "en",
    "task": "transcribe"
  }'
```

Response:
```json
{
  "connection_id": "bot_abc12345",
  "status": "spawning",
  "message": "Bot bot_abc12345 is starting..."
}
```

### 2. Check Bot Status

```bash
curl http://localhost:3000/api/bots/status/bot_abc12345
```

Response:
```json
{
  "connection_id": "bot_abc12345",
  "user_id": "test-user",
  "status": "active",
  "is_healthy": true,
  "uptime_seconds": 45.2,
  "container_id": "abc123...",
  "container_name": "livetranslate-bot-abc12345"
}
```

### 3. List All Bots

```bash
curl http://localhost:3000/api/bots/list
```

### 4. Stop a Bot

```bash
curl -X POST http://localhost:3000/api/bots/stop/bot_abc12345 \
  -H "Content-Type: application/json" \
  -d '{"timeout": 30}'
```

### 5. Get Bot Manager Stats

```bash
curl http://localhost:3000/api/bots/stats
```

Response:
```json
{
  "total_bots": 5,
  "active_bots": 2,
  "total_started": 5,
  "total_completed": 2,
  "total_failed": 1,
  "success_rate": 0.4
}
```

## 📝 Creating a Test Meeting

To test the bot system, you can create a Google Meet:

1. **Via Google Calendar:**
   - Create a new event
   - Click "Add Google Meet video conferencing"
   - Copy the meeting URL

2. **Via meet.google.com:**
   - Go to https://meet.google.com
   - Click "New meeting" → "Create an instant meeting"
   - Copy the meeting URL

3. **Via API (if you have credentials):**
   - The bot can also create meetings programmatically
   - See `google_meet_client.py` for details

## 🎥 Testing Virtual Webcam

To test the virtual webcam feature:

```bash
python test_bot_summon.py \
  --meeting-url https://meet.google.com/your-meeting \
  --enable-webcam
```

The bot will:
- Display transcriptions with speaker attribution
- Show translations in real-time
- Generate professional overlay graphics
- Stream via virtual camera to Google Meet

## 📚 Additional Resources

- **Bot Manager Code:** `src/bot/docker_bot_manager.py`
- **Bot Router API:** `src/routers/bot_management.py`
- **Bot Container:** `modules/bot-container/`
- **Google Meet Automation:** `src/bot/google_meet_automation.py`

## 🎯 Success Criteria

A successful bot test should show:
- ✅ Bot reaches "active" state
- ✅ Container is running (`docker ps` shows it)
- ✅ No error messages in logs
- ✅ Bot appears in Google Meet participant list
- ✅ Audio is being captured (if transcription enabled)

## 🚨 Important Notes

1. **Meeting Access:** Some Google Meet meetings require:
   - Authentication (Google account)
   - Lobby approval from host
   - Organization membership

2. **Rate Limits:** Google Meet may rate-limit bot joins
   - Wait between tests if you get blocked
   - Use different meeting URLs for testing

3. **Resources:** Each bot uses:
   - ~100-200MB RAM
   - 1 Docker container
   - Network bandwidth for audio streaming

4. **Cleanup:** Stop test bots when done:
   ```bash
   # List all bots
   curl http://localhost:3000/api/bots/list

   # Stop specific bot
   curl -X POST http://localhost:3000/api/bots/stop/{connection_id}
   ```

## 📞 Support

If you encounter issues:
1. Check orchestration service logs
2. Check Docker container logs: `docker logs <container-name>`
3. Verify all prerequisites are met
4. Review bot status via API for error messages
5. Check the troubleshooting section above

---

**Happy Testing! 🤖🎉**
