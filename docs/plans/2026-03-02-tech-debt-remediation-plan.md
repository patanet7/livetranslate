# Tech Debt Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Eliminate runtime bugs, dead code, type mismatches, and inconsistencies across the orchestration and dashboard services identified during the full-system audit.

**Architecture:** Surgical fixes organized by blast radius — runtime bugs first, then dead code removal, then dashboard cleanup, then configuration standardization. No new features, no refactors beyond what's broken.

**Tech Stack:** Python/FastAPI (orchestration), SvelteKit/TypeScript (dashboard), structlog

---

## Context

A comprehensive audit of both services revealed:
- 2 runtime type mismatches that break UI rendering
- 6 orphaned files totaling ~1,567 lines of dead code
- 2 dead API client methods calling non-existent endpoints
- Inconsistent env var naming for service URLs
- 21 TODO stubs (many are unimplemented method bodies)
- Hardcoded WebSocket fallback in dashboard config

This plan covers only the **actionable, high-impact** items. Large-scope items (test coverage expansion, auth system, demo mode redesign) are out of scope and should be separate plans.

---

## Task 1: Fix `nativeName` → `native` Type Mismatch

The backend (`system_constants.py:27`) sends language objects with key `native`. The dashboard TypeScript type (`config.ts:29`) expects `nativeName`. This causes the Languages checkbox grid on `/config/system` to show `undefined` for native names.

**Files:**
- Modify: `modules/dashboard-service/src/lib/types/config.ts:29`

**Step 1: Fix the type**

Change line 29 from:
```typescript
languages: Array<{ code: string; name: string; nativeName: string }>;
```
to:
```typescript
languages: Array<{ code: string; name: string; native: string; rtl?: boolean }>;
```

This also adds the `rtl` field the backend sends (used for RTL language badges on the config page).

**Step 2: Update any references to `nativeName` in Svelte files**

Search for `nativeName` in dashboard-service and replace with `native`:
- `modules/dashboard-service/src/routes/(app)/config/system/+page.svelte` — language grid rendering

**Step 3: Run svelte-check**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service && npx svelte-check --threshold error`
Expected: 0 errors

**Step 4: Commit**

```bash
git add modules/dashboard-service/src/lib/types/config.ts modules/dashboard-service/src/routes/
git commit -m "fix: align UiConfig.languages type with backend (native, rtl fields)"
```

---

## Task 2: Remove Dead API Methods from Dashboard

`translationApi` in `translation.ts` has two methods that call endpoints that don't exist on the orchestration service:
- `batchTranslate()` → `POST /api/translation/batch` (404)
- `detectLanguage()` → `POST /api/translation/detect` (404)

**Files:**
- Modify: `modules/dashboard-service/src/lib/api/translation.ts:10-17`

**Step 1: Verify no callers exist**

Search the dashboard for `batchTranslate` and `detectLanguage` usage. If no callers, proceed to remove.

**Step 2: Remove dead methods**

Delete lines 10-17 (`batchTranslate` and `detectLanguage` methods) from `translation.ts`.

**Step 3: Remove unused types**

If `TranslateRequest[]` or `{ detected_language, confidence }` types are only used by these methods, remove them too.

**Step 4: Run svelte-check**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service && npx svelte-check --threshold error`
Expected: 0 errors

**Step 5: Commit**

```bash
git add modules/dashboard-service/src/lib/api/translation.ts
git commit -m "fix: remove dead batchTranslate and detectLanguage API methods"
```

---

## Task 3: Delete Orphaned Files from Orchestration Service

Six files in the orchestration service are completely dead — not imported, not registered, not referenced by any live code path. They total ~1,567 lines and create confusion during audits.

**Files to delete:**
1. `src/routers/seamless.py` (~140 lines) — SeamLess M4T proxy router, never registered in `main_fastapi.py`
2. `src/gateway/api_gateway.py` (~622 lines) — Standalone API gateway class, never imported
3. `src/dashboard/real_time_dashboard.py` (~421 lines) — Standalone Flask dashboard, never imported
4. `src/utils/dependency_check.py` (~170 lines) — Dependency checker, never imported
5. `src/main.py` (~104 lines) — Legacy Flask entry point, replaced by `main_fastapi.py`
6. `src/routers/audio.py` (~110 lines) — Legacy audio wrapper, replaced by `src/routers/audio/` package

**Step 1: Verify each file is truly dead**

For each file, confirm:
- Not imported by any other file: `grep -r "from <module>" src/ --include="*.py"`
- Not registered in `main_fastapi.py`
- Not referenced in any config or startup script

**Step 2: Delete the files**

```bash
rm src/routers/seamless.py
rm src/gateway/api_gateway.py
rm src/dashboard/real_time_dashboard.py
rm src/utils/dependency_check.py
rm src/main.py
rm src/routers/audio.py
```

**Step 3: Clean up empty directories**

If `src/gateway/` or `src/dashboard/` become empty (or only have `__init__.py`), check if any other files reference them. If the directories are dead, delete them too.

**Step 4: Run unit tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service && uv run pytest tests/fireflies/unit/ -x -q`
Expected: All 509+ unit tests pass

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove 6 orphaned files (~1,567 lines of dead code)"
```

---

## Task 4: Fix Dashboard Health Store and WebSocket Config

Two issues in the dashboard's runtime configuration:

### 4a: Health store response shape

`health.svelte.ts:25` expects `data.services` as `Record<string, boolean>`, but the backend `/api/system/health` returns a richer object from `health_monitor.get_system_health()`. The store should handle whatever shape the backend sends gracefully, but most importantly it should not crash if the shape differs.

**Files:**
- Modify: `modules/dashboard-service/src/lib/stores/health.svelte.ts`

Read the actual backend health response shape (from `src/monitoring/health_monitor.py`) and update the store's type or parsing to match.

### 4b: Hardcoded WS_BASE fallback

`config.ts:4` falls back to `ws://localhost:3000` which won't work in production. The fallback should derive from the current window location instead.

**Files:**
- Modify: `modules/dashboard-service/src/lib/config.ts:4`

Change from:
```typescript
export const WS_BASE = browser ? (PUBLIC_WS_URL || 'ws://localhost:3000') : '';
```
to:
```typescript
export const WS_BASE = browser
  ? (PUBLIC_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`)
  : '';
```

**Step 1: Implement both fixes**

**Step 2: Run svelte-check**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service && npx svelte-check --threshold error`
Expected: 0 errors

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/lib/stores/health.svelte.ts modules/dashboard-service/src/lib/config.ts
git commit -m "fix: derive WS_BASE from window.location, handle health response shape"
```

---

## Task 5: Standardize Service URL Env Vars

The orchestration service uses multiple env var names for the same service:
- Whisper/Audio: `WHISPER_SERVICE_URL` (in `health_monitor.py`, `web_server.py`) vs `AUDIO_SERVICE_URL` (in `dependencies.py`, `config_manager.py`)
- Both default to `http://localhost:5001`

The canonical settings model in `config.py:118` uses `whisper_service_url`. The centralized pattern should be: read from `config.py` Settings, which reads from `WHISPER_SERVICE_URL` env var.

**Files:**
- Modify: `src/dependencies.py` — Replace `os.getenv("AUDIO_SERVICE_URL")` with settings-based lookup
- Modify: `src/managers/config_manager.py` — Replace `AUDIO_SERVICE_URL` references
- Verify: `src/config.py` already has the canonical names

**Step 1: Identify all `AUDIO_SERVICE_URL` references**

These should all become `WHISPER_SERVICE_URL` or read from the Settings object.

**Step 2: Update `dependencies.py`**

Replace direct `os.getenv("AUDIO_SERVICE_URL")` calls with `get_settings().whisper_service_url`.

**Step 3: Update `config_manager.py`**

Replace `os.getenv("AUDIO_SERVICE_URL")` with `WHISPER_SERVICE_URL` or settings-based lookup.

**Step 4: Run unit tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service && uv run pytest tests/fireflies/unit/ -x -q`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/dependencies.py src/managers/config_manager.py
git commit -m "fix: standardize whisper service URL env var (AUDIO_SERVICE_URL → WHISPER_SERVICE_URL)"
```

---

## Task 6: Clean Up TODO Stubs in Bot/Webcam Routers

Several router files have `TODO` comments that are actually **unimplemented method stubs** — the endpoint exists, returns a 500, and the TODO says "implement this method in bot_manager". These are misleading because they appear as working endpoints in the API docs.

The safe fix: make these return `501 Not Implemented` with a clear message, rather than calling non-existent methods and 500-ing.

**Files with unimplemented TODO stubs:**
- `src/routers/bot/bot_webcam.py:50,88` — `get_virtual_webcam_frame`, `get_virtual_webcam_status`
- `src/routers/bot/bot_configuration.py:43,73` — `update_bot_config`, `get_bot_config`
- `src/routers/bot/bot_system.py:92` — `cleanup_system_resources`

**Step 1: For each file, wrap the unimplemented call in a 501 response**

```python
raise HTTPException(
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    detail="Virtual webcam frame generation not yet implemented",
)
```

**Step 2: Remove misleading TODO comments (replace with the 501)**

**Step 3: Run unit tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service && uv run pytest tests/fireflies/unit/ -x -q`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/routers/bot/
git commit -m "fix: return 501 for unimplemented bot endpoints instead of silent 500"
```

---

## Task 7: Address Authentication TODO Stubs

Three endpoints have `# TODO: Implement authentication` comments with no auth check. While full auth is out of scope for this plan, we should add a clearly-documented auth bypass so it's explicit rather than silently open.

**Files:**
- `src/routers/settings/general.py:65,102` — user settings GET/PUT
- `src/websocket_frontend_handler.py:176` — WebSocket auth
- `src/routers/websocket.py:68` — WebSocket token verification
- `src/routers/audio/websocket_audio.py:102` — Audio WebSocket auth

**Step 1: Add explicit auth bypass constant**

In `src/config.py`, add:
```python
# Authentication is not yet implemented. All endpoints are currently open.
# When auth is added, set this to True and implement token verification.
AUTH_ENABLED: bool = False
```

**Step 2: Replace TODO comments with references to the constant**

In each file, replace:
```python
# TODO: Implement user authentication
```
with:
```python
# Auth: Not yet implemented (see config.AUTH_ENABLED)
```

This makes the security posture explicit and grep-able without changing behavior.

**Step 3: Run unit tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service && uv run pytest tests/fireflies/unit/ -x -q`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/config.py src/routers/settings/general.py src/websocket_frontend_handler.py src/routers/websocket.py src/routers/audio/websocket_audio.py
git commit -m "chore: make auth bypass explicit with AUTH_ENABLED flag, replace TODO stubs"
```

---

## Task 8: Run Full Verification Suite

**Step 1: Run orchestration unit tests**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service
uv run pytest tests/fireflies/unit/ -x -q
```
Expected: All 509+ tests pass

**Step 2: Run svelte-check on dashboard**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service
npx svelte-check --threshold error
```
Expected: 0 errors

**Step 3: Verify no orphaned imports**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service
grep -r "from gateway\." src/ --include="*.py"
grep -r "from dashboard\." src/ --include="*.py"
grep -r "import seamless" src/ --include="*.py"
grep -r "from utils.dependency_check" src/ --include="*.py"
```
Expected: No results (all dead references removed)

**Step 4: Update plan.md**

Mark this remediation plan as complete in the project plan.

**Step 5: Commit**

```bash
git add plan.md
git commit -m "docs: mark tech debt remediation plan as complete"
```

---

## Files Summary

| Action | File |
|--------|------|
| Modify | `modules/dashboard-service/src/lib/types/config.ts` |
| Modify | `modules/dashboard-service/src/routes/(app)/config/system/+page.svelte` |
| Modify | `modules/dashboard-service/src/lib/api/translation.ts` |
| Delete | `src/routers/seamless.py` |
| Delete | `src/gateway/api_gateway.py` |
| Delete | `src/dashboard/real_time_dashboard.py` |
| Delete | `src/utils/dependency_check.py` |
| Delete | `src/main.py` |
| Delete | `src/routers/audio.py` |
| Modify | `modules/dashboard-service/src/lib/stores/health.svelte.ts` |
| Modify | `modules/dashboard-service/src/lib/config.ts` |
| Modify | `src/dependencies.py` |
| Modify | `src/managers/config_manager.py` |
| Modify | `src/routers/bot/bot_webcam.py` |
| Modify | `src/routers/bot/bot_configuration.py` |
| Modify | `src/routers/bot/bot_system.py` |
| Modify | `src/config.py` |
| Modify | `src/routers/settings/general.py` |
| Modify | `src/websocket_frontend_handler.py` |
| Modify | `src/routers/websocket.py` |
| Modify | `src/routers/audio/websocket_audio.py` |
