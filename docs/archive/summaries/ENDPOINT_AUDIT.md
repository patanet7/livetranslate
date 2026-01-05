# 📋 API Endpoint Audit - Complete Inventory

**Generated**: 2025-10-19
**Total Backend Endpoints**: 178
**Total Frontend Endpoints Used**: ~35
**Usage Rate**: ~20%

---

## Summary Statistics

| Category | Total Endpoints | Used by Frontend | Usage % | Status |
|----------|----------------|------------------|---------|---------|
| **Audio** | 30+ | 8 | 27% | 🟡 Partial |
| **Bot** | 24 | 8 | 33% | 🟡 Partial |
| **Translation** | 12 | 2 | 17% | 🔴 Low |
| **System** | 13 | 5 | 38% | 🟡 Partial |
| **Settings** | 69 | 0 | 0% | 🔴 None |
| **Analytics** | 11 | 1 | 9% | 🔴 Low |
| **WebSocket** | 8 | 1 | 13% | 🔴 Low |
| **Pipeline** | 5 | 2 | 40% | 🟢 Good |
| **Seamless** | 5 | 0 | 0% | 🔴 None |

---

## ✅ Endpoints USED by Frontend

### Audio Processing (8 endpoints)

| Endpoint | Method | Frontend Call | Component | Status |
|----------|--------|---------------|-----------|--------|
| `/audio/upload` | POST | `uploadAudioFile` | MeetingTest | ✅ Active |
| `/audio/process` | POST | `processAudio` | AudioProcessing | ✅ Active |
| `/audio/presets` | GET | `getProcessingPresets` | Pipeline | ✅ Active |
| `/audio/presets/save` | POST | `saveProcessingPreset` | Pipeline | ✅ Active |
| `/audio/analyze/fft` | POST | `getFFTAnalysis` | apiSlice | ✅ Active |
| `/audio/analyze/lufs` | POST | `getLUFSAnalysis` | apiSlice | ✅ Active |
| `/audio/process/stage/{stageType}` | POST | `processSingleStage` | Pipeline | ✅ Active |
| `/audio/pipeline/process` | POST | `processPipeline` | Pipeline | ✅ Active |

### Bot Management (8 endpoints)

| Endpoint | Method | Frontend Call | Component | Status |
|----------|--------|---------------|-----------|--------|
| `/bot` | GET | `getBots` | BotManagement | ✅ Active |
| `/bot/spawn` | POST | `spawnBot` | CreateBotModal | ✅ Active |
| `/bot/{botId}` | GET | `getBot` | BotManagement | ✅ Active |
| `/bot/{botId}/status` | GET | `getBotStatus` | BotManagement | ✅ Active |
| `/bot/{botId}/terminate` | POST | `terminateBot` | BotManagement | ✅ Active |
| `/bot/sessions` | GET | `getBotSessions` | SessionDatabase | ✅ Active |
| `/bot/{botId}/webcam/frame` | GET | `getWebcamFrame` | VirtualWebcam | ✅ Active |
| `/bot/{botId}/webcam/config` | PATCH | `updateWebcamConfig` | VirtualWebcam | ✅ Active |

### System & Health (5 endpoints)

| Endpoint | Method | Frontend Call | Component | Status |
|----------|--------|---------------|-----------|--------|
| `/system/health` | GET | `getSystemHealth` | Dashboard | ✅ Active |
| `/system/services` | GET | `getServiceHealth` | Dashboard | ✅ Active |
| `/system/metrics` | GET | `getSystemMetrics` | Dashboard | ✅ Active |
| `/system/config` | GET | `getConfiguration` | Settings | ✅ Active |
| `/system/config` | PATCH | `updateConfiguration` | Settings | ✅ Active |

### Translation (2 endpoints)

| Endpoint | Method | Frontend Call | Component | Status |
|----------|--------|---------------|-----------|--------|
| `/translations` | GET | `getTranslations` | Translation | ✅ Active |
| `/translations/translate` | POST | `translateText` | Translation | ✅ Active |

### Analytics (1 endpoint)

| Endpoint | Method | Frontend Call | Component | Status |
|----------|--------|---------------|-----------|--------|
| `/analytics/overview` | GET | `getAnalyticsOverview` | Dashboard | ✅ Active |

### Pipeline (2 endpoints)

| Endpoint | Method | Frontend Call | Component | Status |
|----------|--------|---------------|-----------|--------|
| `/audio/pipeline/realtime/start` | POST | `startRealtimeSession` | Pipeline | ✅ Active |
| `/audio/pipeline/process` | POST | `processPipeline` | Pipeline | ✅ Active |

### WebSocket (1 endpoint)

| Endpoint | Method | Frontend Call | Component | Status |
|----------|--------|---------------|-----------|--------|
| `/websocket/info` | GET | `getWebSocketInfo` | Settings | ✅ Active |

---

## ❌ Endpoints NOT USED by Frontend

### Audio Analysis (3 unused)

| Endpoint | Method | Purpose | Action |
|----------|--------|---------|--------|
| `/audio/analyze/spectrum/{session_id}` | GET | Get spectrum analysis | 🔗 **Connect to Dashboard** |
| `/audio/analyze/quality` | POST | Analyze audio quality | 🔗 **Connect to Dashboard** |

### Audio Coordination (9 unused)

| Endpoint | Method | Purpose | Action |
|----------|--------|---------|--------|
| `/audio-coordination/sessions` | GET | List audio sessions | ⚠️ Evaluate need |
| `/audio-coordination/sessions/{id}` | GET | Get session details | ⚠️ Evaluate need |
| `/audio-coordination/statistics` | GET | Get statistics | 🔗 **Connect to Dashboard** |
| `/audio-coordination/config/schema` | GET | Get config schema | 📝 Keep for tooling |
| All other coordination endpoints | Various | Session management | ⚠️ Evaluate need |

### Bot Analytics (6 unused)

| Endpoint | Method | Purpose | Action |
|----------|--------|---------|--------|
| `/bot/{bot_id}/analytics` | GET | Bot analytics | 🔗 **Connect to Dashboard** |
| `/bot/{bot_id}/performance` | GET | Performance metrics | 🔗 **Connect to Dashboard** |
| `/bot/{bot_id}/quality-report` | GET | Quality report | 🔗 **Connect to Dashboard** |
| `/bot/analytics/sessions` | GET | Session analytics | 🔗 **Connect to Dashboard** |
| `/bot/analytics/quality` | GET | Quality analytics | 🔗 **Connect to Dashboard** |
| `/bot/analytics/database` | GET | Database analytics | 🔗 **Connect to Dashboard** |

### Settings (69 unused - ALL!)

| Category | Endpoints | Purpose | Action |
|----------|-----------|---------|--------|
| Audio Processing | 5 | Audio settings CRUD | 🔗 **High Priority** |
| Bot Settings | 8 | Bot configuration | 🔗 **High Priority** |
| Translation Settings | 7 | Translation config | 🔗 **High Priority** |
| Prompts | 12 | Prompt management | 📝 Keep backend-only |
| Correlation | 8 | Time correlation settings | 📝 Advanced feature |
| Chunking | 4 | Chunking settings | 📝 Advanced feature |
| System Settings | 12 | System configuration | ⚠️ Some needed |
| Config Sync | 9 | Configuration sync | ✅ **Already Connected** |
| Backups/Import/Export | 4 | Settings management | 📝 Future feature |

**Note**: Settings endpoints exist but frontend uses config sync instead!

### Translation (10 unused)

| Endpoint | Method | Purpose | Action |
|----------|--------|---------|--------|
| `/translation/health` | GET | Service health | ⚠️ Use system health |
| `/translation/languages` | GET | Supported languages | 🔗 **Connect to UI** |
| `/translation/models` | GET | Available models | 🔗 **Connect to UI** |
| `/translation/batch` | POST | Batch translation | 📝 Future feature |
| `/translation/detect` | POST | Language detection | 📝 Future feature |
| `/translation/stream` | POST | Streaming translation | 📝 Future feature |
| `/translation/session/*` | POST | Session management | ⚠️ Evaluate need |
| `/translation/quality` | POST | Quality assessment | 🔗 **Connect to Dashboard** |

### Analytics (10 unused)

| Endpoint | Method | Purpose | Action |
|----------|--------|---------|--------|
| `/analytics/trends` | GET | Trend analysis | 🔗 **Connect to Dashboard** |
| `/analytics/alerts` | GET | Active alerts | 🔗 **Connect to Dashboard** |
| `/analytics/metrics/{type}` | GET | Specific metrics | 🔗 **Connect to Dashboard** |
| `/analytics/audio/processing` | GET | Audio analytics | 🔗 **Connect to Dashboard** |
| `/analytics/bots/sessions` | GET | Bot analytics | 🔗 **Connect to Dashboard** |
| `/analytics/translation/performance` | GET | Translation analytics | 🔗 **Connect to Dashboard** |
| `/analytics/websocket/connections` | GET | WebSocket analytics | 🔗 **Connect to Dashboard** |
| `/analytics/dashboard/*` | Various | Custom dashboards | 📝 Future feature |

### System (8 unused)

| Endpoint | Method | Purpose | Action |
|----------|--------|---------|--------|
| `/system/status` | GET | System status | ⚠️ Duplicate of health |
| `/system/metrics/performance` | GET | Performance metrics | 🔗 **Connect to Dashboard** |
| `/system/services/{name}` | GET | Service status | ✅ **Already have** `/api/health/{name}` |
| `/system/maintenance/*` | POST | Maintenance mode | 📝 Future feature |
| `/system/services/{name}/restart` | POST | Restart service | 📝 Admin feature |

### WebSocket (7 unused)

| Endpoint | Method | Purpose | Action |
|----------|--------|---------|--------|
| `/websocket/stats` | GET | WebSocket stats | 🔗 **Connect to Dashboard** |
| `/websocket/connections` | GET | Active connections | 🔗 **Connect to Dashboard** |
| `/websocket/sessions` | GET | Active sessions | 📝 Future feature |
| `/websocket/broadcast` | POST | Broadcast message | 📝 Backend-only |
| All other WebSocket endpoints | Various | Connection management | 📝 Backend-only |

### Pipeline (3 unused)

| Endpoint | Method | Purpose | Action |
|----------|--------|---------|--------|
| `/pipeline/realtime/sessions` | GET | Active sessions | 📝 Future feature |
| `/pipeline/realtime/{id}` | DELETE | Stop session | ⚠️ Missing in frontend |
| `/pipeline/realtime/{id}` | WEBSOCKET | Realtime WebSocket | ⚠️ May be used via WebSocket |

### Seamless (5 unused - ALL!)

| Endpoint | Method | Purpose | Action |
|----------|--------|---------|--------|
| `/seamless/sessions` | GET | List sessions | ❓ Unknown feature |
| `/seamless/sessions/{id}` | GET | Get session | ❓ Unknown feature |
| `/seamless/sessions/{id}/events` | GET | Get events | ❓ Unknown feature |
| `/seamless/sessions/{id}/transcripts` | GET | Get transcripts | ❓ Unknown feature |
| `/seamless/realtime/{id}` | WEBSOCKET | Realtime WebSocket | ❓ Unknown feature |

**Note**: Seamless router appears to be a duplicate/alternative system!

---

## 🎯 Recommended Actions

### Priority 1: Connect Missing Dashboard Features

**Audio Analysis Dashboard**
- ✅ Already have: FFT, LUFS analysis
- 🔗 Need to connect: Spectrum analysis, Quality analysis
- **Files to update**:
  - `src/pages/AudioAnalysis/` (if exists)
  - `src/store/slices/apiSlice.ts` (add missing endpoints)

**Bot Analytics Dashboard**
- 🔗 Connect: `/bot/{id}/analytics`, `/bot/{id}/performance`, `/bot/{id}/quality-report`
- **Files to update**:
  - `src/pages/BotManagement/components/BotAnalytics.tsx`
  - Add to `apiSlice.ts`

**System Analytics Dashboard**
- 🔗 Connect: All `/analytics/*` endpoints
- **Files to update**:
  - `src/pages/Dashboard/` components
  - Add comprehensive analytics to `apiSlice.ts`

### Priority 2: Remove Dead Code

**Candidates for Removal**:
- ❌ Seamless router (duplicate system?)
- ❌ Some audio coordination endpoints (if unused)
- ❌ Unused WebSocket management endpoints

**Before Removing**:
1. Verify not used by other services
2. Check git history for usage
3. Document removal reason

### Priority 3: Document Intentional Gaps

**Backend-Only Endpoints** (Keep, don't connect):
- Prompt management (admin/backend config)
- Maintenance mode (admin feature)
- Service restart (admin feature)
- WebSocket broadcast (system internal)

**Future Features** (Document as planned):
- Batch translation
- Language detection
- Custom dashboards
- Settings backup/restore

### Priority 4: Improve API Discoverability

**Add to OpenAPI/Swagger**:
- Tag endpoints as: `frontend-ready`, `backend-only`, `admin`, `future`
- Add `x-frontend-component` to show which component uses it
- Add usage examples

---

## 📊 Coverage by Router

```
Router                  Total  Used  Unused  Coverage
======================= ====== ===== ======= ========
audio_core.py           5      4     1       80%  🟢
audio_analysis.py       4      2     2       50%  🟡
audio_presets.py        6      2     4       33%  🟡
audio_stages.py         4      1     3       25%  🔴
audio_coordination.py   10     0     10      0%   🔴
bot_lifecycle.py        7      5     2       71%  🟢
bot_analytics.py        8      0     8       0%   🔴
bot_configuration.py    2      0     2       0%   🔴
bot_webcam.py           5      2     3       40%  🟡
bot_system.py           3      0     3       0%   🔴
translation.py          12     2     10      17%  🔴
system.py               13     5     8       38%  🟡
settings.py             69     0     69      0%   🔴
analytics.py            11     1     10      9%   🔴
websocket.py            8      1     7       13%  🔴
pipeline.py             5      2     3       40%  🟡
seamless.py             5      0     5       0%   🔴
======================= ====== ===== ======= ========
TOTAL                   178    27    151     15%  🔴
```

---

## 🔍 Next Steps

1. **Complete Audio Analysis Dashboard** ← **DOING NOW**
2. Connect Bot Analytics Dashboard
3. Connect System Analytics Dashboard
4. Audit Settings router for removal
5. Audit Seamless router for removal/integration
6. Add endpoint usage tracking (metrics)
7. Generate OpenAPI spec with frontend tags
8. Create integration tests for all connected endpoints

---

*Last Updated: 2025-10-19*
*Generated By: Claude Code Endpoint Audit*
