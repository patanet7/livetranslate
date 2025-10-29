# Google Meet Bot Authentication Guide

## Overview

The bot supports **two authentication methods** for joining Google Meet:

1. **Persistent Browser Profile (RECOMMENDED)** - Stays logged in like a real browser
2. **Username/Password (For initial setup only)** - Use once, then rely on saved state

## Method 1: Persistent Browser Profile (Production-Ready) ✅

This is the **recommended approach** because it:
- ✅ **Avoids 2FA/MFA challenges** - No repeated logins
- ✅ **Bypasses captchas** - Google trusts the saved session
- ✅ **Prevents rate limiting** - No suspicious repeated login attempts
- ✅ **Works indefinitely** - Session stays valid for weeks/months
- ✅ **Production-grade** - Used by Recall.ai, Fireflies, etc.

### How It Works

```
First Run:
1. Bot logs in with email/password
2. Saves browser state (cookies, tokens) to persistent storage
3. Joins meeting

Subsequent Runs:
1. Bot loads saved state from storage
2. Already logged in - no authentication flow needed
3. Joins meeting immediately
```

### Setup Instructions

#### Step 1: Configure Persistent Storage

**Docker Compose:**
```yaml
services:
  bot-container:
    volumes:
      - bot-profile:/app/browser-profile  # Persistent volume

volumes:
  bot-profile:
```

**Docker CLI:**
```bash
docker volume create bot-browser-profile

docker run -v bot-browser-profile:/app/browser-profile \
  -e GOOGLE_EMAIL=your@gmail.com \
  -e GOOGLE_PASSWORD=yourpassword \
  -e USER_DATA_DIR=/app/browser-profile \
  livetranslate-bot:latest
```

#### Step 2: First-Time Login

On the **first run only**, provide credentials:

```bash
# Environment variables
GOOGLE_EMAIL=bot-account@gmail.com
GOOGLE_PASSWORD=your-app-specific-password  # Use App Password if 2FA enabled
USER_DATA_DIR=/app/browser-profile

# Or via API
curl -X POST http://localhost:3000/api/bots/start \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_url": "https://meet.google.com/xxx-xxxx-xxx",
    "user_token": "...",
    "google_email": "bot-account@gmail.com",
    "google_password": "your-password",
    "user_data_dir": "/app/browser-profile"
  }'
```

The bot will:
1. Login to Google account
2. Save session state to `/app/browser-profile/state.json`
3. Join the meeting

#### Step 3: Subsequent Runs (No Login Needed)

After the first successful login, **remove the password** and just provide the profile path:

```bash
# Only need this now!
USER_DATA_DIR=/app/browser-profile

# Or via API (no password needed)
curl -X POST http://localhost:3000/api/bots/start \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_url": "https://meet.google.com/xxx-xxxx-xxx",
    "user_token": "...",
    "user_data_dir": "/app/browser-profile"
  }'
```

The bot will:
1. Load saved state from `/app/browser-profile/state.json`
2. Already logged in ✅
3. Join the meeting immediately

## Method 2: Username/Password (Initial Setup Only)

**⚠️ WARNING**: Only use this for **first-time setup** or **testing**. Do not use in production without persistent storage!

### Issues with Password-Only Auth:
- ❌ Google's 2FA will block automated logins
- ❌ Captchas will appear frequently
- ❌ "Suspicious activity" alerts
- ❌ Rate limiting after multiple attempts
- ❌ Account lockout risk

### When to Use:
- First-time setup to save browser state
- Local testing/development
- Manual testing (not automated production)

## Best Practices

### Production Deployment

```yaml
# docker-compose.yml
services:
  bot-container:
    image: livetranslate-bot:latest
    volumes:
      - bot-profile:/app/browser-profile:rw
    environment:
      - USER_DATA_DIR=/app/browser-profile
      # No need for EMAIL/PASSWORD after first run!
    restart: unless-stopped

volumes:
  bot-profile:
    driver: local
```

### Google Account Recommendations

1. **Create a dedicated bot account**
   - Use a separate Gmail account for bots
   - Enable "Less secure app access" if needed
   - Use App-Specific Password if 2FA is enabled

2. **First-time setup**
   - Run bot once with credentials
   - Verify `state.json` is created
   - Remove credentials from config
   - Bot will use saved state from now on

3. **Session Refresh**
   - Google sessions last **weeks to months**
   - If session expires, re-run with credentials
   - State will automatically refresh

### Security Considerations

1. **Protect the persistent volume**
   ```bash
   chmod 700 /var/lib/docker/volumes/bot-profile
   ```

2. **Never commit state.json to git**
   ```gitignore
   browser-profile/
   state.json
   ```

3. **Use environment variables for credentials**
   ```bash
   # Never hardcode in source
   GOOGLE_EMAIL=...
   GOOGLE_PASSWORD=...
   ```

4. **Use App-Specific Passwords**
   - Go to: https://myaccount.google.com/apppasswords
   - Generate password for "Playwright Bot"
   - Use that instead of real password

## Troubleshooting

### "You can't join this video call"

**Cause**: Meeting requires authenticated Google account
**Solution**: Use persistent browser profile with saved login state

### Session Expired

**Cause**: Saved state is too old (usually after 30+ days)
**Solution**: Re-run bot with credentials once to refresh state

### 2FA/Captcha Blocking Login

**Cause**: Google detected automated login
**Solution**:
1. Login manually in non-headless mode first
2. Save the state
3. Use saved state for all future runs

### Manual Login Setup (Safest)

```bash
# Run bot in non-headless mode
docker run -e HEADLESS=false \
  -e USER_DATA_DIR=/app/browser-profile \
  -v bot-profile:/app/browser-profile \
  livetranslate-bot:latest

# Manually login when browser opens
# Complete any 2FA/captcha challenges
# Bot will save the authenticated state
# Future runs will use this saved state
```

## Summary

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Persistent Profile** | Production | No repeated logins, no 2FA issues, indefinite session | Requires volume setup |
| **Username/Password** | Initial setup | Quick start | 2FA blocks, captchas, not reliable |

**Recommended Flow**:
1. Setup: Use credentials once with persistent storage
2. Production: Remove credentials, rely on saved state
3. Maintenance: Refresh state if session expires

---

**Status**: ✅ Production-Ready with Persistent Browser Profile
