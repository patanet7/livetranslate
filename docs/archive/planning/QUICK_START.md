# 🚀 Quick Start Guide - Fixed WebSocket Connection

## Prerequisites

✅ `.env.development` file created
✅ Environment configuration ready

## Start Services (Correct Order)

### Terminal 1: Orchestration Service (FastAPI)

```bash
cd modules/orchestration-service

# IMPORTANT: Run FastAPI, NOT Flask!
python src/main_fastapi.py
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:3000
```

### Terminal 2: Frontend Service (Vite)

```bash
cd modules/frontend-service

# Check that .env.development exists
ls -la .env.development

# Start Vite dev server
npm run dev
```

**Expected Output:**
```
VITE v5.x.x  ready in XXX ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

## Verification Steps

### 1. Test Orchestration Service

```bash
# Test API endpoint
curl http://localhost:3000/api/health

# Expected: JSON response with "status": "healthy"
```

### 2. Test WebSocket (from browser console)

```javascript
// Open http://localhost:5173 and run in DevTools Console:
const ws = new WebSocket('ws://localhost:5173/ws');
ws.onopen = () => console.log('✅ WebSocket Connected!');
ws.onerror = (e) => console.error('❌ WebSocket Error:', e);
```

### 3. Check Environment Variables (browser console)

```javascript
// Verify Vite loaded environment variables:
console.log('WS URL:', import.meta.env.VITE_WS_BASE_URL);
// Expected: ws://localhost:5173
```

## Troubleshooting

### Issue: "Connection reset by peer"

**Cause:** Running Flask (`src/main.py`) instead of FastAPI (`src/main_fastapi.py`)

**Fix:**
```bash
# Kill old process
pkill -f "python src/main.py"

# Start FastAPI
cd modules/orchestration-service
python src/main_fastapi.py
```

### Issue: WebSocket still fails

**Check 1:** Verify both services are running
```bash
# Orchestration on 3000
lsof -i :3000

# Frontend on 5173
lsof -i :5173
```

**Check 2:** Verify .env.development exists
```bash
cd modules/frontend-service
cat .env.development
```

**Check 3:** Restart Vite dev server
```bash
# Vite needs restart to load new .env files
# Press Ctrl+C in Terminal 2, then:
npm run dev
```

### Issue: API requests fail

**Check Vite proxy configuration:**
```bash
cd modules/frontend-service
grep -A 5 "proxy:" vite.config.ts
```

Should show:
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:3000',
    changeOrigin: true,
  },
  '/ws': {
    target: 'ws://localhost:3000',
    ws: true,
    changeOrigin: true,
  },
}
```

## Current Status Checklist

- [ ] Orchestration service running FastAPI (not Flask)
- [ ] Frontend dev server on port 5173
- [ ] `.env.development` file exists
- [ ] WebSocket connects successfully
- [ ] No "Connection reset" errors
- [ ] Browser console shows "WebSocket Connected"

## What Changed

### Before
- ❌ Hardcoded WebSocket URL: `ws://localhost:3000/ws`
- ❌ Direct connection bypassing proxy
- ❌ CORS issues
- ❌ No environment configuration

### After
- ✅ Environment-based URL: `ws://localhost:5173/ws`
- ✅ Uses Vite proxy in development
- ✅ No CORS issues
- ✅ Proper dev/prod separation

## Quick Test Script

```bash
#!/bin/bash

echo "🔍 Testing LiveTranslate services..."

# Test orchestration
echo "1. Testing orchestration service..."
curl -s http://localhost:3000/api/health > /dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ Orchestration service OK"
else
    echo "   ❌ Orchestration service FAILED"
fi

# Test frontend
echo "2. Testing frontend dev server..."
curl -s http://localhost:5173 > /dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ Frontend dev server OK"
else
    echo "   ❌ Frontend dev server FAILED"
fi

# Test env file
echo "3. Checking .env.development..."
if [ -f "modules/frontend-service/.env.development" ]; then
    echo "   ✅ .env.development exists"
else
    echo "   ❌ .env.development missing"
fi

echo ""
echo "✨ Open http://localhost:5173 in browser"
echo "📊 Check DevTools Console for WebSocket connection"
```

Save as `test-services.sh`, then run: `bash test-services.sh`
