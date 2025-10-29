# 🚀 Quick Start: Test Google Meet Bot Login

Follow these steps to test if your bot can successfully log into Google Meet.

## Step 1: Start Required Services

### Terminal 1: Start Orchestration Service
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/orchestration-service
python src/main.py
```

Wait until you see:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:3000
```

### Terminal 2: Start Redis (if not already running)
```bash
# Check if Redis is running
redis-cli ping

# If not running, start it:
docker run -d -p 6379:6379 redis:alpine

# Or if installed locally:
redis-server
```

### Terminal 3: Verify Docker is Running
```bash
docker ps
```

Should show running containers. If Docker isn't running, start Docker Desktop.

## Step 2: Run the Bot Test

Open a new terminal and run:

```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/orchestration-service
python quick_bot_test.py
```

This will:
1. ✅ Check if services are available
2. ✅ Start a test bot
3. ✅ Monitor the bot in real-time
4. ✅ Show if it successfully logs in

## Step 3: Test with Your Own Meeting

Create a Google Meet meeting:
1. Go to https://meet.google.com
2. Click "New meeting" → "Create an instant meeting"
3. Copy the meeting URL

Then run:
```bash
python quick_bot_test.py --url https://meet.google.com/your-meeting-code
```

## 🎯 What Success Looks Like

You should see output like:
```
╔═══════════════════════════════════════════════════════════════════╗
║           GOOGLE MEET BOT - QUICK LOGIN TEST                      ║
╚═══════════════════════════════════════════════════════════════════╝

🔍 Checking orchestration service...
✅ Orchestration service is running

🚀 Starting bot for meeting: https://meet.google.com/abc-defg-hij

✅ Bot started with ID: bot_xyz123

======================================================================
Monitoring Bot: bot_xyz123
======================================================================

12:34:56 | 🔄 SPAWNING - Bot container starting...
12:34:58 | 🚀 STARTING - Bot initializing...
12:35:03 | 🚪 JOINING - Bot joining Google Meet...
12:35:15 | ✅ ACTIVE - Bot is in the meeting!

SUCCESS! Bot successfully joined Google Meet!

Bot Details:
  • Container: livetranslate-bot-xyz123
  • Uptime: 19.3s
  • Healthy: True

======================================================================
✅ TEST PASSED
The bot successfully logged into Google Meet!
======================================================================
```

## 🔧 Quick Troubleshooting

### "Orchestration service not available"
**Fix:** Start the orchestration service (see Terminal 1 above)

### "Failed to start bot"
**Check:**
- Is Docker running? → `docker ps`
- Is Redis running? → `redis-cli ping`
- Check service logs in Terminal 1

### Bot stuck in "spawning"
**Check:**
- Docker container logs: `docker ps` then `docker logs <container-name>`
- Bot image exists: `docker images | grep livetranslate-bot`

### Bot stuck in "joining"
**This is normal!** The bot might be waiting for:
- Meeting lobby approval
- Authentication (if required)

**Try:**
- Join the meeting yourself and approve the bot
- Use a meeting without authentication requirements

## 📊 View Bot Status Manually

```bash
# List all bots
curl http://localhost:3000/api/bots/list | jq

# Get specific bot status
curl http://localhost:3000/api/bots/status/<bot-id> | jq

# Get manager stats
curl http://localhost:3000/api/bots/stats | jq
```

## 🎥 Advanced: Test with Virtual Webcam

```bash
python test_bot_summon.py \
  --meeting-url https://meet.google.com/your-meeting \
  --enable-webcam \
  --monitor-time 120
```

This will enable the virtual webcam that displays transcriptions and translations.

## 📚 More Information

See `BOT_TESTING_GUIDE.md` for comprehensive testing documentation.

---

**Need Help?**
- Check orchestration service logs (Terminal 1)
- Check Docker container logs: `docker logs <container-name>`
- See BOT_TESTING_GUIDE.md for detailed troubleshooting
