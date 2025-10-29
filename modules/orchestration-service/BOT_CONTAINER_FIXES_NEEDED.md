# Bot Container Fixes Needed

## Current Status
- ✅ Orchestration service configured properly
- ✅ Docker image built (`livetranslate-bot:latest`)
- ✅ Docker network created (`livetranslate_default`)
- ✅ Bot can reach orchestration via `host.docker.internal`
- ❌ Bot authentication protocol doesn't match orchestration WebSocket
- ❌ Bot callback URLs missing `/api` prefix

## Issues to Fix

### 1. Authentication Protocol Mismatch

**Location:** `modules/bot-container/src/orchestration_client.py:179`

**Problem:**
```python
if response.get("type") != "authenticated":
    raise RuntimeError(f"Authentication failed: {response}")
```

Bot expects `"authenticated"` response, but orchestration sends `"connection:established"`.

**Fix Options:**

**Option A: Update bot to handle actual protocol (RECOMMENDED)**
```python
# In orchestration_client.py _authenticate() method:

# Wait for connection established
response_raw = await self.websocket.recv()
response = json.loads(response_raw)

if response.get("type") != "connection:established":
    raise RuntimeError(f"Connection failed: {response}")

logger.info("✅ Connection established")
self.authenticated = True  # For now, skip separate auth step
```

**Option B: Update orchestration WebSocket to support bot auth**
- Add authentication handling to `/ws` endpoint in `main_fastapi.py`
- Send `"authenticated"` message after receiving `"authenticate"` message
- More complex, requires changes to orchestration service

### 2. Callback URL Missing `/api` Prefix

**Location:** `modules/bot-container/src/bot_main.py` (callback sending code)

**Problem:**
Bot is calling: `http://host.docker.internal:3000/bots/internal/callback/failed`
Should be calling: `http://host.docker.internal:3000/api/bots/internal/callback/failed`

**Fix:**
Find where `BOT_MANAGER_URL` is used and ensure `/api` is prepended to callback paths:

```python
# Before:
callback_url = f"{self.bot_manager_url}/bots/internal/callback/{status}"

# After:
callback_url = f"{self.bot_manager_url}/api/bots/internal/callback/{status}"
```

### 3. Error Callback Issue

**Location:** `modules/bot-container/src/orchestration_client.py:151`

**Problem:**
```python
await self.error_callback(f"Connection failed: {e}")
# TypeError: object NoneType can't be used in 'await' expression
```

`error_callback` is None but being awaited.

**Fix:**
```python
# Check if callback exists and is async
if self.error_callback:
    if asyncio.iscoroutinefunction(self.error_callback):
        await self.error_callback(f"Connection failed: {e}")
    else:
        self.error_callback(f"Connection failed: {e}")
```

## How to Apply Fixes

### Step 1: Fix bot-container code

```bash
cd modules/bot-container

# Edit src/orchestration_client.py
# Edit src/bot_main.py (callback URLs)

# Verify changes
git diff src/
```

### Step 2: Rebuild Docker image

```bash
docker build -t livetranslate-bot:latest .
```

### Step 3: Test again

```bash
# Start a new bot
curl -X POST http://localhost:3000/api/bots/start \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_url": "https://meet.google.com/koz-misu-odh",
    "user_token": "test-token",
    "user_id": "test-user",
    "language": "en",
    "task": "transcribe"
  }'

# Check logs
docker logs bot-<connection-id>
```

## Quick Test

To verify the bot container can reach orchestration:

```bash
# From inside a test container:
docker run --rm --network livetranslate_default alpine:latest \
  wget -O- http://host.docker.internal:3000/api/health

# Should return health check JSON
```

## Expected Flow After Fixes

1. **Container starts** → Sends callback to `/api/bots/internal/callback/started`
2. **Connects to WS** → Receives `connection:established`, treats as authenticated
3. **Starts session** → Begins audio streaming
4. **Joins meeting** → Sends callback to `/api/bots/internal/callback/joining`
5. **In meeting** → Sends callback to `/api/bots/internal/callback/active`
6. **Streams audio** → Receives transcription segments
7. **Exits** → Sends callback to `/api/bots/internal/callback/completed`

## Alternative: Simplified Testing

If you want to test without fixing the bot container authentication:

1. Create a simple test bot that just verifies Docker connectivity
2. Skip the WebSocket authentication for now
3. Focus on getting the bot to appear in Google Meet first
4. Add proper protocol handling later

---

**Next Steps:**
1. Fix `orchestration_client.py` authentication handling
2. Fix callback URLs in `bot_main.py`
3. Rebuild Docker image
4. Test bot joining Google Meet
