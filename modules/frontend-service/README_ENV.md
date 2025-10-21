# Frontend Service Environment Configuration

## Quick Setup

### Development Mode

```bash
# Copy the development template
cp .env.development.template .env.development

# Start the dev server
npm run dev
```

### Production Build

```bash
# Copy the production template
cp .env.production.template .env.production

# Build the application
npm run build
```

## Environment Files

- **`.env.development.template`** - Development configuration template (COMMITTED)
- **`.env.production.template`** - Production configuration template (COMMITTED)
- **`.env.example`** - General example configuration (COMMITTED)
- **`.env.development`** - Your local development config (NOT COMMITTED)
- **`.env.production`** - Your local production config (NOT COMMITTED)
- **`.env.local`** - Your local overrides (NOT COMMITTED)

## Important Configuration

### WebSocket URL (Critical!)

**Development** (`.env.development`):
```bash
# MUST use localhost:5173 to use Vite proxy and avoid CORS
VITE_WS_BASE_URL=ws://localhost:5173
```

**Production** (`.env.production`):
```bash
# Direct connection to orchestration service
VITE_WS_BASE_URL=ws://localhost:3000
```

### Why This Matters

The frontend runs on different ports in development vs production:
- **Development**: Port 5173 (Vite dev server)
- **Production**: Port 3000 (served by orchestration service)

Using `ws://localhost:5173` in development allows Vite to proxy WebSocket connections to the orchestration service (`ws://localhost:3000`), avoiding CORS issues.

## Available Variables

### Required

| Variable | Description | Development | Production |
|----------|-------------|-------------|------------|
| `VITE_WS_BASE_URL` | WebSocket base URL | `ws://localhost:5173` | `ws://localhost:3000` |
| `VITE_API_BASE_URL` | API base URL | `http://localhost:5173` | `http://localhost:3000` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_WS_AUTO_RECONNECT` | `true` | Enable automatic reconnection |
| `VITE_WS_MAX_RECONNECT_ATTEMPTS` | `3` | Max attempts before API fallback |
| `VITE_WS_RECONNECT_INTERVAL` | `10000` | Reconnection delay (ms) |
| `VITE_WS_HEARTBEAT_INTERVAL` | `45000` | Heartbeat interval (ms) |
| `VITE_ENABLE_DEBUG` | `false` | Enable debug logging |
| `VITE_LOG_LEVEL` | `error` | Log level (debug, info, warn, error) |

## Troubleshooting

### WebSocket Connection Failed

**Symptom**: Console shows "WebSocket failed 3+ times, switching to API mode"

**Solution**: Ensure you're using the correct WebSocket URL:
- Development: `ws://localhost:5173/ws` (via proxy)
- Production: `ws://localhost:3000/ws` (direct)

### CORS Errors

**Symptom**: CORS errors in browser console

**Solution**:
1. Verify you're using `ws://localhost:5173` in development
2. Check that Vite dev server is running on port 5173
3. Check that orchestration service is running on port 3000

### 404 on /ws

**Symptom**: WebSocket connection gets 404 error

**Solution**:
1. Ensure orchestration service is running
2. Check `modules/orchestration-service/src/main_fastapi.py` has WebSocket endpoint at `/ws`
3. Verify no firewall/proxy blocking WebSocket connections

## Architecture

### Development Flow
```
Frontend (localhost:5173)
    ↓ WebSocket: ws://localhost:5173/ws
    ↓
Vite Proxy (vite.config.ts)
    ↓ Forwards to: ws://localhost:3000/ws
    ↓
Orchestration Service (localhost:3000)
```

### Production Flow
```
Frontend (served from localhost:3000)
    ↓ WebSocket: ws://localhost:3000/ws
    ↓
Orchestration Service (localhost:3000)
```

## Related Documentation

- **WebSocket Fix**: `../../WEBSOCKET_FIX.md`
- **API Integration**: `../../api_integration_report.md`
- **Vite Config**: `vite.config.ts`
- **Project CLAUDE.md**: `../CLAUDE.md`
