# SvelteKit Dashboard Rebuild Plan -- Full Feature Parity

**Date**: 2026-02-26
**Status**: Ready for implementation
**Goal**: Transform the hollow SvelteKit scaffold into a fully functional dashboard with 100% feature parity to the old Fireflies dashboard (3,874 lines, 9 tabs), plus the session manager and captions overlay.

---

## Table of Contents

1. [Feature Parity Matrix](#1-feature-parity-matrix)
2. [API Endpoint Inventory](#2-api-endpoint-inventory)
3. [Critical Fixes (Dark Mode, Layout)](#3-batch-1-critical-fixes)
4. [Core Feature Build-out](#4-batch-2-core-feature-build-out)
5. [Missing Pages and Features](#5-batch-3-missing-pages-and-features)
6. [Polish, Integration, and Testing](#6-batch-4-polish-integration-and-testing)
7. [Model and LLM Selection Guide](#7-model-and-llm-selection-guide)
8. [Test Specifications](#8-test-specifications)

---

## 1. Feature Parity Matrix

Every feature from the old Fireflies dashboard (`fireflies-dashboard.html`), session manager (`session-manager.html`), and captions overlay (`captions.html`) mapped to a SvelteKit route/component.

### Old Dashboard Tab: Connect

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| Transcript ID input | `#transcriptId` input | `(app)/fireflies/+page.svelte` | EXISTS but incomplete |
| Multi-select target languages | `<select multiple>` populated from `/api/system/ui-config` | `(app)/fireflies/+page.svelte` | BROKEN -- uses comma text input instead of multi-select |
| API key validation before connect | Checks `apiKey` from localStorage, validates via `/fireflies/meetings` | `(app)/fireflies/+page.svelte` | MISSING -- no API key concept at all |
| Translation model selection on connect | `selectedModel` sent in connect payload | `(app)/fireflies/+page.svelte` | MISSING |
| Active meetings list (Refresh Meetings) | `fetchMeetings()` -> `POST /fireflies/meetings` | `(app)/fireflies/+page.svelte` | MISSING |
| Click meeting to populate ID | `selectMeeting()` | `(app)/fireflies/+page.svelte` | MISSING |
| Domain selector (populated from backend) | `#glossaryDomain` populated from `/api/system/ui-config` | `(app)/fireflies/+page.svelte` | EXISTS but hardcoded domain list |

### Old Dashboard Tab: Live Feed

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| Session selector dropdown | `#feedSessionSelect` populated from active sessions | NONE | MISSING -- entire tab absent |
| Target language filter | `#feedTargetLang` with "All Languages" option | NONE | MISSING |
| Connect/Disconnect to WebSocket feed | `connectToFeed()` / `disconnectFeed()` | NONE | MISSING |
| Two-column layout: Original + Translated | Side-by-side `#originalFeed` and `#translatedFeed` panels | NONE | MISSING |
| Feed status badge | `#feedStatus` (connected/disconnected/error) | NONE | MISSING |
| Feed stats (entry count, speaker count) | `#feedStats` text | NONE | MISSING |
| Save Feed to local storage | `saveFeedToLocal()` | NONE | MISSING |
| Export Feed as JSON | `exportFeedJson()` -> blob download | NONE | MISSING |
| Demo mode banner | `#demoFeedBanner` with session/mode/speakers info | NONE | MISSING |

### Old Dashboard Tab: Sessions

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| Stats grid (Total/Connected/Chunks/Translations) | `.stats-grid` with 4 stat cards | `(app)/fireflies/+page.svelte` | MISSING -- only shows session list |
| Session list with status badges | `renderSessions()` | `(app)/fireflies/+page.svelte` | EXISTS but minimal |
| View Captions button per session | `viewCaptions()` -> opens overlay in new tab | `(app)/fireflies/+page.svelte` | MISSING |
| View Data button per session | `viewSessionData()` -> switches to Data tab | `(app)/fireflies/+page.svelte` | MISSING |
| Disconnect button per session | `disconnectSession()` | `(app)/fireflies/connect/+page.svelte` | EXISTS but only on connect page |
| Refresh button | `refreshSessions()` | `(app)/fireflies/+page.svelte` | MISSING -- no explicit refresh |

### Old Dashboard Tab: Glossary

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| Glossary selector dropdown | `#glossarySelect` | `(app)/fireflies/glossary/+page.svelte` | MISSING -- no selector, uses default |
| Create New glossary button | `createNewGlossary()` -> `POST /api/glossaries` | `(app)/fireflies/glossary/+page.svelte` | MISSING |
| Glossary list panel | `#glossaryList` with click-to-select | `(app)/fireflies/glossary/+page.svelte` | MISSING |
| Glossary details panel (name, domain, source lang) | `#glossaryDetails` form with save/delete/set-default | `(app)/fireflies/glossary/+page.svelte` | MISSING |
| Domain picker (populated from backend) | `#glossaryDomain` select | `(app)/fireflies/glossary/+page.svelte` | MISSING |
| Set as Default button | `setDefaultGlossary()` -> `PATCH /api/glossaries/{id}` | `(app)/fireflies/glossary/+page.svelte` | MISSING |
| Delete glossary button | `deleteGlossary()` -> `DELETE /api/glossaries/{id}` | `(app)/fireflies/glossary/+page.svelte` | MISSING |
| Entries table with multi-lang columns | `renderGlossaryEntries()` (es/fr/de columns) | `(app)/fireflies/glossary/+page.svelte` | EXISTS but simplified |
| Add term with multi-language translations | `addGlossaryEntry()` (prompts for es/fr/de) | `(app)/fireflies/glossary/+page.svelte` | EXISTS but single-lang only |
| Import CSV button | `bulkImportGlossary()` | `(app)/fireflies/glossary/+page.svelte` | MISSING |
| Export CSV button | `exportGlossary()` -> blob download | `(app)/fireflies/glossary/+page.svelte` | MISSING |
| Edit entry | `editGlossaryEntry()` | `(app)/fireflies/glossary/+page.svelte` | MISSING |

### Old Dashboard Tab: History

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| Date range filter (from/to date inputs) | `#historyDateFrom`, `#historyDateTo` | `(app)/fireflies/history/+page.svelte` | MISSING |
| Fetch Past Meetings button | `fetchPastMeetings()` -> `POST /fireflies/transcripts` | `(app)/fireflies/history/+page.svelte` | MISSING -- only shows active sessions |
| Past meetings table (date/title/duration/speakers) | `#pastMeetingsTable` | `(app)/fireflies/history/+page.svelte` | MISSING |
| View transcript button -> modal viewer | `viewTranscript()` -> opens modal | `(app)/fireflies/history/+page.svelte` | MISSING |
| Translate All button (batch with progress bar) | `translateFullTranscript()` with batched progress | `(app)/fireflies/history/+page.svelte` | MISSING |
| Save Locally button | `saveTranscriptLocal()` | `(app)/fireflies/history/+page.svelte` | MISSING |
| Import to DB button | `importToDatabase()` -> `POST /fireflies/import/{id}` | `(app)/fireflies/history/+page.svelte` | MISSING |
| Saved Transcripts table (local storage) | `loadSavedTranscripts()` | `(app)/fireflies/history/+page.svelte` | MISSING |
| Transcript Viewer modal | `#transcriptViewerModal` full modal with close/footer | `(app)/fireflies/history/+page.svelte` | MISSING |

### Old Dashboard Tab: Data & Logs

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| Session selector for data viewing | `#dataSessionSelect` | NONE | MISSING -- entire tab absent |
| Transcripts panel (scrollable) | `#transcriptsPanel` | NONE | MISSING |
| Translations panel (scrollable) | `#translationsPanel` | NONE | MISSING |
| Database entries table (time/speaker/original/translation/lang/confidence) | `#dbEntriesTable` | NONE | MISSING |
| API call log panel | `#apiLogPanel` | NONE | MISSING |

### Old Dashboard Tab: Translation

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| Current model info display (name/backend/device) | `#currentModelInfo` | `(app)/config/translation/+page.svelte` | BROKEN -- hardcoded backend list instead of fetched models |
| Available models dropdown (from `/api/translation/models`) | `#modelSelect` populated dynamically | `(app)/config/translation/+page.svelte` | BROKEN -- hardcoded list |
| Switch Model button | `switchModel()` | `(app)/config/translation/+page.svelte` | MISSING -- save button exists but wrong action |
| Prompt template editor with style selector | `#promptTemplate` textarea + `#templateStyleSelect` | `(app)/config/translation/+page.svelte` | MISSING |
| Save/Reset prompt buttons | `savePromptTemplate()` / `resetPromptTemplate()` | `(app)/config/translation/+page.svelte` | MISSING |
| Test Translation section | `testTranslation()` with text/lang/result display | `(app)/translation/test/+page.svelte` | EXISTS -- works |
| Translation result details (model/backend/time/confidence) | Result display with 6 fields | `(app)/translation/test/+page.svelte` | EXISTS -- works |

### Old Dashboard Tab: Settings

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| API Key input with show/hide toggle | `#apiKeyInput` (password) + Show button | NONE | MISSING |
| Save API Key (validates via `/fireflies/meetings`) | `saveApiKey()` | NONE | MISSING |
| Test Connection button | `testApiKey()` | NONE | MISSING |
| Clear API Key button | `clearApiKey()` | NONE | MISSING |
| Masked saved key display | `#savedKeyDisplay` with masked key + "Saved" badge | NONE | MISSING |
| Demo Mode launcher | `launchDemoFromSettings()` with mode/speakers/exchanges/delay controls | NONE | MISSING |
| Demo mode selector (passthrough/pretranslated) | `#settingsDemoMode` select | NONE | MISSING |
| Demo speakers/exchanges/delay inputs | Number inputs for configuration | NONE | MISSING |
| Open Captions Overlay button | `openCaptionsOverlay()` | NONE | MISSING |
| Service Status panel | `checkServices()` -> health checks for 4 services | `(app)/config/system/+page.svelte` | MISSING -- system page has no health checks |
| Activity Log panel | `#logPanel` with timestamped entries | NONE | MISSING |

### Old Dashboard Tab: Intelligence

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| Session selector for intelligence | `#intelSessionSelect` | NONE | MISSING -- entire tab absent |
| Meeting Notes (manual + auto-generated) | `#notesList` + `addManualNote()` | NONE | MISSING |
| Analyze prompt input | `analyzeNote()` -> `POST .../notes/analyze` | NONE | MISSING |
| Insight templates selector | `#insightTemplateSelect` | NONE | MISSING |
| Generate Insight / Generate All buttons | `generateInsight()` / `generateAllInsights()` | NONE | MISSING |
| Insight results display | `#insightResults` with cards | NONE | MISSING |
| Meeting Q&A Agent chat | `#agentChatMessages` with send/receive/SSE streaming | NONE | MISSING |
| Suggested queries buttons | `#suggestedQueries` populated from API | NONE | MISSING |

### Header and Global

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| API status badge in header | `#apiStatus` (Connected/Not Connected) | TopBar.svelte | MISSING -- TopBar is near-empty |
| Translation status badge in header | `#translationStatus` (Online/Offline) | TopBar.svelte | MISSING |
| Demo mode toggle in header | `#demoBtn` with Launch/Stop Demo | TopBar.svelte | MISSING |
| Demo mode selector in header | `#demoModeSelect` (passthrough/pretranslated) | TopBar.svelte | MISSING |
| Dark theme by default | `background: linear-gradient(135deg, #1a1a2e, #16213e)` | root `+layout.svelte` | BROKEN -- `dark` class never applied |

### Session Manager (session-manager.html)

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| Create/Join session with ID | `createSession()` | `(app)/sessions/` or merged into Fireflies | MISSING |
| Generate random session ID | `generateId()` | NONE | MISSING |
| Show overlay URL + API URL | URL display boxes with copy | NONE | MISSING |
| Test Caption Sender | Speaker select + text + send | NONE | MISSING |
| Run Demo button | Scripted conversation send | NONE | MISSING |
| WebSocket connect/status | Connect button + status badge + log | NONE | MISSING |

### Captions Overlay (captions.html)

| Feature | Old Implementation | SvelteKit Route | Status |
|---|---|---|---|
| URL-parameter-driven config | `?session=X&lang=Y&fontSize=32&position=bottom` etc | `(overlay)/captions/+page.svelte` | EXISTS but needs verification |
| WebSocket connection with reconnect | Auto-connect + max 10 attempts + page refresh | `(overlay)/captions/+page.svelte` | EXISTS via wsStore |
| Caption rendering (speaker/original/translated) | `createCaptionElement()` | CaptionBox.svelte | EXISTS |
| Caption expiry with fade animation | `setTimeout` + `.fading` class | CaptionStream.svelte | PARTIAL -- store handles expiry but animation unclear |
| Caption aggregation (same speaker appending) | `updateCaption()` resets timer | captionStore | EXISTS |
| Configurable max captions | `config.maxCaptions` enforcement | UNKNOWN | NEEDS VERIFICATION |
| Setup help screen (no session) | `#setup-help` displayed when no session param | `(overlay)/captions/+page.svelte` | UNKNOWN |
| Connection status indicator (colored dot) | `#status` with green/yellow/red dot | `(overlay)/captions/+page.svelte` | UNKNOWN |

---

## 2. API Endpoint Inventory

Every orchestration service endpoint relevant to the dashboard, organized by router prefix as mounted in `main_fastapi.py`.

### Fireflies (`/fireflies/...`)
- `POST /fireflies/connect` -- Connect to a Fireflies transcript (body: `api_key`, `transcript_id`, `target_languages`, `translation_model`)
- `POST /fireflies/disconnect` -- Disconnect session (body: `session_id`)
- `GET /fireflies/sessions` -- List all active sessions
- `GET /fireflies/sessions/{session_id}` -- Get specific session
- `POST /fireflies/meetings` -- List meetings from Fireflies API (body: `api_key`)
- `POST /fireflies/transcripts` -- List past transcripts (body: `api_key`, `limit`)
- `POST /fireflies/transcript/{transcript_id}` -- Get transcript content (body: `api_key`)
- `POST /fireflies/import/{transcript_id}` -- Import transcript to database (body: `api_key`, `transcript_id`, `target_language`)
- `GET /fireflies/health` -- Fireflies router health
- `GET /fireflies/dashboard/config` -- Dashboard configuration (legacy, prefer `/api/system/ui-config`)
- `POST /fireflies/demo/start?mode=passthrough|pretranslated` -- Start demo mode
- `POST /fireflies/demo/stop` -- Stop demo mode
- `GET /fireflies/demo/status` -- Check if demo is running

### System (`/api/system/...`)
- `GET /api/system/ui-config` -- **CRITICAL**: Single source of truth for languages, domains, models, prompt templates, defaults
- `GET /api/system/health` -- System health
- `GET /api/system/health/detailed` -- Detailed health
- `GET /api/system/services` -- All service statuses
- `GET /api/system/services/{service_name}` -- Specific service status
- `GET /api/system/metrics` -- Full system metrics
- `GET /api/system/status` -- Simple status check

### Translation (`/api/translation/...`)
- `POST /api/translation/translate` -- Translate text (body: `text`, `target_language`, `source_language`, `service`, `context`, `model`)
- `POST /api/translation/batch` -- Batch translate
- `POST /api/translation/detect` -- Detect language
- `GET /api/translation/health` -- Translation service health (returns backend info, available backends, device)
- `GET /api/translation/models` -- **CRITICAL**: List available translation models (returns array with `name`, `model`, `display_name`, `backend`)
- `POST /api/translation/model` -- Switch model (body: `model`)
- `POST /api/translation/prompt` -- Save prompt template

### Glossaries (`/api/glossaries/...`)
- `GET /api/glossaries` -- List glossaries (query: `domain`, `source_language`, `active_only`)
- `POST /api/glossaries` -- Create glossary
- `GET /api/glossaries/{id}` -- Get glossary
- `PATCH /api/glossaries/{id}` -- Update glossary
- `DELETE /api/glossaries/{id}` -- Delete glossary
- `GET /api/glossaries/{id}/entries` -- List entries
- `POST /api/glossaries/{id}/entries` -- Create entry
- `PATCH /api/glossaries/{id}/entries/{entry_id}` -- Update entry
- `DELETE /api/glossaries/{id}/entries/{entry_id}` -- Delete entry
- `POST /api/glossaries/{id}/import` -- Bulk import entries
- `POST /api/glossaries/{id}/lookup` -- Term lookup in text
- `GET /api/glossaries/terms/{target_language}` -- Get terms for translation

### Captions (`/api/captions/...`)
- `WebSocket /api/captions/stream/{session_id}?target_language=XX` -- Real-time caption stream
- `GET /api/captions/{session_id}` -- Get current active captions
- `GET /api/captions/{session_id}/stats` -- Get caption stats
- `POST /api/captions/{session_id}` -- Add caption (test/debug)
- `DELETE /api/captions/{session_id}` -- Clear session captions

### Meeting Intelligence (`/api/intelligence/...`)
- `GET /api/intelligence/sessions/{session_id}/notes` -- List notes
- `POST /api/intelligence/sessions/{session_id}/notes` -- Create manual note
- `POST /api/intelligence/sessions/{session_id}/notes/analyze` -- LLM-analyzed note
- `DELETE /api/intelligence/notes/{note_id}` -- Delete note
- `POST /api/intelligence/sessions/{session_id}/insights/generate` -- Generate insights
- `GET /api/intelligence/sessions/{session_id}/insights` -- List insights
- `GET /api/intelligence/insights/{insight_id}` -- Get specific insight
- `DELETE /api/intelligence/insights/{insight_id}` -- Delete insight
- `GET /api/intelligence/templates` -- List insight templates
- `GET /api/intelligence/templates/{id}` -- Get template
- `POST /api/intelligence/templates` -- Create template
- `PUT /api/intelligence/templates/{id}` -- Update template
- `DELETE /api/intelligence/templates/{id}` -- Delete template
- `POST /api/intelligence/sessions/{session_id}/agent/conversations` -- Start agent conversation
- `GET /api/intelligence/agent/conversations/{id}` -- Get conversation history
- `POST /api/intelligence/agent/conversations/{id}/messages` -- Send message
- `POST /api/intelligence/agent/conversations/{id}/messages/stream` -- Send message with SSE streaming
- `GET /api/intelligence/sessions/{session_id}/agent/suggestions` -- Get suggested queries

### Settings (`/api/settings/...`)
- `GET /api/settings/user` -- Get user settings
- `PUT /api/settings/user` -- Update user settings
- `GET /api/settings/translation` -- Get translation settings
- `POST /api/settings/translation` -- Save translation settings
- `POST /api/settings/translation/test` -- Test translation

### Data Query (`/api/data/...`)
- `GET /api/data/sessions/{session_id}/transcripts` -- Query transcripts
- `GET /api/data/sessions/{session_id}/translations` -- Query translations
- `GET /api/data/sessions/{session_id}/timeline` -- Complete timeline
- `GET /api/data/sessions/{session_id}/speakers` -- Speaker statistics
- `GET /api/data/sessions/{session_id}/search` -- Full-text search

### Analytics (`/api/analytics/...`)
- Various performance and analytics endpoints for dashboards

---

## 3. Batch 1: Critical Fixes

Fix everything that exists but is broken before building new features.

### Task 1.1: Activate Dark Mode

**What this fixes**: The `app.css` has a `.dark { ... }` block with all the correct dark theme variables, but nothing adds the `dark` class to the HTML element. The old dashboard was entirely dark-themed.

**Requirements**:
- The root `+layout.svelte` (`/Users/thomaspatane/GitHub/personal/livetranslate/.claude/worktrees/sveltekit-dashboard/modules/dashboard-service/src/routes/+layout.svelte`) must apply `class="dark"` to the `<html>` element. In SvelteKit this means using `<svelte:head>` or a `app.html` template.
- Read the file `/Users/thomaspatane/GitHub/personal/livetranslate/.claude/worktrees/sveltekit-dashboard/modules/dashboard-service/src/app.html` and add `class="dark"` to the `<html>` tag.
- Verify that all pages render with the dark background and light text.

**Verification**: Load any page, confirm background is dark (`oklch(0.141 0.005 285.823)`) and foreground is light.

### Task 1.2: Fix the TopBar

**What this fixes**: The TopBar currently shows only a "Services" status indicator with a comment "breadcrumb can go here later". The old dashboard header showed: API status badge, Translation status badge, demo mode selector, and demo launch button.

**Requirements**:
- Add the following to TopBar: API connection status badge (green "API Connected" or red "API Not Connected"), Translation service status badge (green "Translation Online" or red "Translation Offline"), demo mode selector dropdown (passthrough/pretranslated), and Launch Demo / Stop Demo button.
- API status should derive from the health store. Translation status should derive from a dedicated check against `/api/translation/health`.
- The demo button must call `POST /fireflies/demo/start?mode=...` and `POST /fireflies/demo/stop`, and check `GET /fireflies/demo/status` on mount.
- When demo starts, navigate to the Live Feed page (see Batch 2).

**Verification**: Start the dashboard, confirm TopBar shows 2 status badges and demo controls. Click Launch Demo, confirm it calls the API and shows "Stop Demo" state.

### Task 1.3: Fix the Connect Page Language Selector

**What this fixes**: The connect page uses a plain text input for "Target Languages (comma-separated)" instead of a multi-select populated from the backend.

**Requirements**:
- The `+page.server.ts` load function already fetches `uiConfig` from `/api/system/ui-config`. This returns `{ languages: [...], defaults: { default_target_languages: [...] } }`.
- Replace the comma-separated text input with a proper multi-select (or a shadcn-svelte checkbox group) populated from `data.uiConfig.languages`.
- Pre-select languages from `data.uiConfig.defaults.default_target_languages`.
- The form action must send the selected languages as an array, not a comma-split string.

**Verification**: Load `/fireflies`, confirm language selector shows all languages from the backend. Default selections match the backend defaults. Submit the form and verify the `target_languages` array is sent correctly to `POST /fireflies/connect`.

### Task 1.4: Fix the Dashboard Home Page

**What this fixes**: The dashboard home page (`(app)/+page.svelte`) shows 3 minimal cards with almost no information.

**Requirements**:
- Add a stats grid similar to the old Sessions tab: Total Sessions, Connected, Total Chunks, Total Translations. Fetch these from `/fireflies/sessions` on load (via `+page.server.ts`).
- The Quick Actions card should include links to all major sections: Connect to Fireflies, Live Feed, Glossary, History, Translation Test, Intelligence, Configuration, Session Manager.
- The Services card should show status for each service: Orchestration, Translation, Database, Fireflies Router. Fetch from `/api/system/health` and `/api/translation/health`.
- Add a recent sessions list (last 5 sessions) with quick-link to the connect page.

**Verification**: Load `/`, confirm stats are populated from real data, all quick action links work, service health reflects actual service states.

### Task 1.5: Fix the Sidebar Navigation

**What this fixes**: The sidebar has only 4 top-level items (Dashboard, Fireflies, Config, Translation) and is missing entries for Live Feed, Sessions, Data & Logs, History, Intelligence, and Session Manager.

**Requirements**:
- Update `Sidebar.svelte` to include all navigation items matching the old dashboard tabs:
  - Dashboard (home)
  - Fireflies -> Connect, Live Feed, Sessions
  - History
  - Glossary
  - Data & Logs
  - Translation -> Test Bench, Config
  - Intelligence
  - Config -> Audio, System
  - Session Manager
- Each item should have an appropriate icon (use simple text/emoji icons to match current pattern).
- Active state highlighting should work for all routes.

**Verification**: Navigate to each route via the sidebar. Confirm active state highlighting is correct for each page and all links work.

---

## 4. Batch 2: Core Feature Build-out

Build the features that form the core workflow: API key management, model selection, Live Feed, and enhanced Sessions.

### Task 2.1: API Key Management

**What this builds**: The old dashboard had a complete API key management section in the Settings tab. The user enters their Fireflies API key, it is validated, saved to localStorage, and used for all Fireflies operations.

**Route**: `(app)/config/settings/+page.svelte` (new page under config, or add to existing system page)

**API endpoints used**:
- `POST /fireflies/meetings` with `{ api_key: "..." }` for validation
- The API key is stored client-side (localStorage) and sent with each Fireflies request

**Page must contain**:
- A password input for the API key with a Show/Hide toggle button
- A "Save API Key" button that first validates the key by calling `POST /fireflies/meetings`. On success, save to localStorage and show a "Saved" badge with masked key. On failure, show the error.
- A "Test Connection" button that validates the stored key and shows results (number of meetings found)
- A "Clear" button that removes the key from localStorage with a confirmation dialog
- A masked key display section (`****...last4`) with a "Saved" badge

**Special consideration**: Since the API key lives in localStorage (client-side), the form cannot use a standard SvelteKit form action (those run server-side). This must be implemented as a client-side form using `$effect` and `fetch()` directly from the browser. The API key should be passed through to the orchestration service via the browser -- the SvelteKit server does not need to store it.

**Verification**: Enter a test API key, click Save, confirm validation request is made. If valid, the masked key appears with "Saved" badge. Clear the key, confirm it is removed. Navigate away and back, confirm the key persists via localStorage.

### Task 2.2: Translation Model Selection (Config)

**What this builds**: The old dashboard Translation tab showed the current model, a dropdown of available models fetched from `/api/translation/models`, and a Switch Model button.

**Route**: Enhance `(app)/config/translation/+page.svelte`

**API endpoints used**:
- `GET /api/translation/models` -- Returns `{ models: [{ name, model, display_name, backend }] }`
- `GET /api/translation/health` -- Returns current backend, device, available_backends
- `POST /api/translation/model` -- Switch model (body: `{ model: "..." }`)

**Page must contain**:
- A "Current Model" info card showing: model display name, backend name, actual model identifier, device (cpu/gpu/npu), and an "Active" status badge. This data comes from combining `/api/translation/health` and `/api/translation/models`.
- An "Available Models" dropdown populated from `/api/translation/models`. Each option should show the display name and model identifier.
- A "Switch Model" button that calls `POST /api/translation/model` and refreshes the current model display.
- A "Refresh" button to re-fetch models.
- **Prompt Template Editor**: A textarea for the translation prompt with a style selector dropdown (simple/full/minimal). The three default templates should be available (matching the old dashboard's `PROMPT_TEMPLATES` object). Include "Save Prompt" and "Reset to Default" buttons that save to `POST /api/translation/prompt` and localStorage.
- Available template variables display: `{target_language}`, `{current_sentence}`, `{glossary_section}`, `{context_window}`.
- The hardcoded backend list (`['ollama', 'vllm', 'openai', 'groq']`) must be removed and replaced with dynamically fetched data.

**The `+page.server.ts` load function must**:
- Fetch `/api/translation/models` for the models list
- Fetch `/api/translation/health` for current model/backend info
- Fetch `/api/system/ui-config` for languages

**Verification**: Load `/config/translation`, confirm model dropdown is populated from the real API (not hardcoded). Select a different model, click Switch, verify the API call succeeds and the display updates. Edit the prompt template, save it, reload the page and verify it persists.

### Task 2.3: Live Feed Page

**What this builds**: The entire Live Feed tab from the old dashboard. This is one of the most important features -- it shows real-time transcripts and translations side-by-side via WebSocket.

**Route**: `(app)/fireflies/live-feed/+page.svelte` (new route)

**API endpoints used**:
- `GET /fireflies/sessions` -- Populate session selector
- `WebSocket /api/captions/stream/{session_id}?target_language=XX` -- Real-time feed
- `GET /api/system/ui-config` -- Populate language filter

**Page must contain**:
- A session selector dropdown populated from active sessions (fetched on load and refreshable)
- A target language filter dropdown (with "All Languages" option), populated from `/api/system/ui-config`
- Connect and Disconnect buttons
- A connection status badge (Connected/Disconnected/Connecting/Error)
- A stats display (entry count, speaker count)
- A two-column grid layout:
  - Left column: "Original Transcript" panel -- shows speaker-attributed original text, scrollable, auto-scrolls to bottom
  - Right column: "Translation" panel -- shows speaker-attributed translated text with confidence %, scrollable, auto-scrolls to bottom
- A "Save Feed" button that saves all entries to localStorage
- An "Export JSON" button that triggers a browser file download of all feed data as JSON
- When demo mode is active, a banner at the top showing: "DEMO MODE" badge, session ID, mode (passthrough/pretranslated), speakers list, "Open Captions Overlay" button, and "Stop Demo" button

**WebSocket message handling** (same protocol as `captions.html`):
- `connected` event: Load `current_captions` array into both panels
- `caption_added` event: Append to both panels
- `caption_updated` event: Update existing entry or append
- `caption_expired` event: Optionally fade out (or keep in scroll history)

**Important**: WebSocket connections from the browser go directly to the orchestration service, not through SvelteKit. Use the `WS_BASE` config from `$lib/config.ts`.

**Verification**: Start a demo from the TopBar. Verify the Live Feed page auto-navigates (or navigate manually). Confirm the session selector has the demo session. Click Connect. Verify both columns populate with real-time data. Verify speaker names appear. Click Export JSON and confirm a valid JSON file downloads. Click Save Feed and verify data persists in localStorage.

### Task 2.4: Enhanced Sessions Page

**What this builds**: A proper sessions management page with stats grid, per-session actions, and data viewing.

**Route**: `(app)/fireflies/sessions/+page.svelte` (new route, separate from the connect form)

**API endpoints used**:
- `GET /fireflies/sessions` -- List sessions
- `POST /fireflies/disconnect` -- Disconnect a session
- `GET /api/captions/{session_id}/stats` -- Per-session stats

**Page must contain**:
- A stats grid at the top: Total Sessions, Connected, Total Chunks Received, Total Translations Completed (aggregated from all sessions)
- A session list where each session card shows: session ID (truncated), transcript ID, connection status badge, chunks received, translations completed, speakers detected count
- Per-session action buttons: "Captions" (opens overlay URL in new tab), "Data" (navigates to Data & Logs page with session pre-selected), "Disconnect" (with confirmation)
- A Refresh button

**Verification**: Load `/fireflies/sessions`, confirm stats grid shows aggregated numbers. Each session has working Captions/Data/Disconnect buttons. Disconnect a session and verify it is removed from the list.

---

## 5. Batch 3: Missing Pages and Features

Build the remaining missing pages: Data & Logs, Intelligence, Session Manager, History enhancements, Glossary enhancements.

### Task 3.1: Data & Logs Page

**What this builds**: The Data & Logs tab from the old dashboard.

**Route**: `(app)/data/+page.svelte` (new route)

**API endpoints used**:
- `GET /fireflies/sessions` -- Populate session selector
- `GET /api/captions/{session_id}` -- Get session caption data
- `GET /api/data/sessions/{session_id}/transcripts` -- Transcripts
- `GET /api/data/sessions/{session_id}/translations` -- Translations
- `GET /api/data/sessions/{session_id}/timeline` -- Timeline

**Page must contain**:
- A session selector dropdown (populated from active sessions)
- A "Load Data" button
- A two-column layout:
  - Left: "Transcripts" panel (scrollable, showing `[timestamp] speaker: text`)
  - Right: "Translations" panel (scrollable, showing `[language] speaker: translated_text`)
- A "Database Entries" table below with columns: Time, Speaker, Original, Translation, Language, Confidence
- An "API Call Log" panel at the bottom that logs all API calls made from this page (client-side, showing method, endpoint, success/failure, timestamp)

**Verification**: Select a session with data, click Load Data, verify transcripts and translations populate from real API data. Verify the database entries table shows combined and sorted data.

### Task 3.2: Intelligence Page

**What this builds**: The entire Intelligence tab from the old dashboard.

**Route**: `(app)/intelligence/+page.svelte` (new route)

**API endpoints used**:
- `GET /fireflies/sessions` -- Session selector
- `GET /api/intelligence/sessions/{id}/notes` -- List notes
- `POST /api/intelligence/sessions/{id}/notes` -- Create manual note
- `POST /api/intelligence/sessions/{id}/notes/analyze` -- LLM analysis
- `GET /api/intelligence/templates` -- List insight templates
- `POST /api/intelligence/sessions/{id}/insights/generate` -- Generate insights
- `GET /api/intelligence/sessions/{id}/insights` -- List insights
- `POST /api/intelligence/sessions/{id}/agent/conversations` -- Start conversation
- `POST /api/intelligence/agent/conversations/{id}/messages` -- Send message
- `POST /api/intelligence/agent/conversations/{id}/messages/stream` -- Send with SSE streaming
- `GET /api/intelligence/sessions/{id}/agent/suggestions` -- Suggested queries

**Page must contain three sections**:

**Section 1 - Meeting Notes**:
- Session selector dropdown
- Notes list (scrollable, showing note type badge, speaker name, content, timestamp, processing time)
- Manual note input with "Add Note" button
- Analysis prompt input with "Analyze" button
- Color-coded note types: auto (orange border), manual (green border), annotation (purple border)

**Section 2 - Post-Meeting Insights**:
- Insight template selector dropdown (populated from `/api/intelligence/templates`)
- Custom instructions input
- "Generate Insight" and "Generate All" buttons
- Results display showing insight cards with: title, insight type, content, processing time, LLM model, transcript length, creation date

**Section 3 - Meeting Q&A Agent**:
- Chat message area (scrollable, showing user messages right-aligned in blue, AI responses left-aligned in gray)
- Suggested query buttons (populated from API, clicking fills the input and sends)
- Text input with "Send" button (Enter key also sends)
- Support for SSE streaming responses (show tokens as they arrive)
- Typing indicator while waiting for response

**Verification**: Select a session, add a manual note, verify it appears in the list. Use the analyze feature with a prompt, verify an LLM-generated note appears. Generate an insight from a template, verify it displays. Send a chat message, verify a response streams in.

### Task 3.3: History Page Enhancements

**What this builds**: The full History tab functionality that is currently missing.

**Route**: Enhance `(app)/fireflies/history/+page.svelte`

**API endpoints used**:
- `POST /fireflies/transcripts` -- Fetch past meetings (needs API key from localStorage)
- `POST /fireflies/transcript/{id}` -- Get transcript content
- `POST /fireflies/import/{id}` -- Import to database
- `POST /api/translation/translate` -- For batch translation
- `GET /api/system/ui-config` -- For language selector

**Page must contain**:
- A date range filter with "from" and "to" date inputs
- A "Fetch Past Meetings" button that calls `POST /fireflies/transcripts` (sends API key from localStorage)
- A past meetings table with columns: Date, Title, Duration, Speakers, Actions (View, Translate)
- A transcript viewer modal (using shadcn-svelte Dialog) containing:
  - Transcript content display (speaker-attributed, timestamped)
  - Target language selector
  - "Translate All" button with a progress bar (batched translation with 5 sentences in parallel, context window of 3)
  - "Save Locally" button
  - "Import to DB" button
  - Progress display showing percentage and count
- A "Saved Transcripts (Local)" table that reads from localStorage with columns: Session/Transcript ID, Language, Saved At, Items, Actions (View, Export, Delete)
- Viewing saved transcripts should open the same modal with side-by-side original/translated text

**Verification**: Ensure the Fetch Past Meetings button calls the real API with the stored API key. View a transcript, verify content loads. Translate All with progress bar, verify translations appear with confidence scores. Save locally, verify entry appears in Saved Transcripts table. Export to JSON, verify file downloads.

### Task 3.4: Glossary Page Enhancements

**What this builds**: The missing glossary features: glossary CRUD, multi-language support, domain picker, import/export.

**Route**: Enhance `(app)/fireflies/glossary/+page.svelte`

**API endpoints used**:
- `GET /api/glossaries` -- List glossaries
- `POST /api/glossaries` -- Create glossary
- `PATCH /api/glossaries/{id}` -- Update glossary
- `DELETE /api/glossaries/{id}` -- Delete glossary
- `GET /api/glossaries/{id}/entries` -- List entries
- `POST /api/glossaries/{id}/entries` -- Create entry
- `PATCH /api/glossaries/{id}/entries/{eid}` -- Update entry
- `DELETE /api/glossaries/{id}/entries/{eid}` -- Delete entry
- `POST /api/glossaries/{id}/import` -- Bulk import
- `GET /api/system/ui-config` -- For domains and languages

**Page must contain (two-column layout)**:

**Left column**:
- A glossary selector dropdown with all glossaries
- A "New" button to create a glossary (prompts for name via Dialog)
- A glossary list panel showing all glossaries with name, entry count, domain, default star indicator, click-to-select
- A "Glossary Details" form below with: Name input, Domain selector (populated from `ui-config.domains`), Source Language selector, "Save" button, "Set as Default" button, "Delete" button (with confirmation)

**Right column**:
- Entries table with columns: Source Term, translations for each target language (es/fr/de as separate columns), Priority, Actions (Edit, Delete)
- "Add Term" button that opens a Dialog with: source term input, translation inputs for each target language, priority selector
- "Import CSV" button that accepts a CSV file upload and calls `POST /api/glossaries/{id}/import`
- "Export" button that generates and downloads a CSV

**Verification**: Create a new glossary, verify it appears in the list. Add entries with multiple language translations. Set a glossary as default, verify the star appears. Import a CSV (create a test CSV with 3 entries), verify entries appear. Export and verify CSV downloads correctly. Delete a glossary and confirm it is gone.

### Task 3.5: Session Manager Page

**What this builds**: The session manager from `session-manager.html`.

**Route**: `(app)/sessions/+page.svelte` (new route)

**API endpoints used**:
- `POST /api/captions/{session_id}` -- Add test caption
- `DELETE /api/captions/{session_id}` -- Clear session captions
- `WebSocket /api/captions/stream/{session_id}` -- Connect to session

**Page must contain**:
- A "Create / Join Session" card with: session ID input, "Create Session" button, "Generate ID" button (creates random `session-XXXXXXXX`)
- When a session is created, show the OBS Overlay URL and API Endpoint URL with copy-to-clipboard
- A "Test Caption Sender" card with: session ID input, WebSocket connect button + status badge, speaker selector (Alice/Bob/Charlie with colored buttons), caption text textarea, "Send Caption" button, "Run Demo" button (sends scripted conversation), "Clear Session" button
- A log output panel showing all actions with timestamps and color-coded entries (sent=green, received=blue, error=red)

**Verification**: Generate a session ID, create the session, open the overlay URL in another tab. Send a test caption, verify it appears in the overlay. Run the demo, verify scripted messages flow through. Clear the session, verify captions disappear from the overlay.

### Task 3.6: Demo Mode Integration

**What this builds**: The demo mode that was available in the old dashboard header and settings tab.

**Implementation spans**: TopBar.svelte, Live Feed page, Settings area

**API endpoints used**:
- `POST /fireflies/demo/start?mode=passthrough|pretranslated` -- Start demo
- `POST /fireflies/demo/stop` -- Stop demo
- `GET /fireflies/demo/status` -- Check demo status

**Requirements**:
- Create a demo mode Svelte store (`$lib/stores/demo.svelte.ts`) that tracks: `active` (boolean), `sessionId` (string), `mode` (string), `speakers` (string array)
- On app mount (in `(app)/+layout.svelte`), check `GET /fireflies/demo/status` and restore demo state if active
- TopBar shows demo controls: mode selector dropdown, Launch/Stop button that toggles state
- When demo starts: auto-navigate to Live Feed page, auto-select the demo session, auto-connect WebSocket
- Live Feed page shows demo banner when demo is active (matching old dashboard's `#demoFeedBanner`)
- Settings area includes demo configuration: mode, speakers count, exchanges count, delay
- An "Open Captions Overlay" button appears when demo is active

**Verification**: Click Launch Demo in the TopBar. Verify navigation to Live Feed, auto-connection, and banner display. Stop the demo, verify state clears. Reload the page while demo is running, verify state is restored from the server.

---

## 6. Batch 4: Polish, Integration, and Testing

### Task 4.1: Captions Overlay Verification

**Route**: `(overlay)/captions/+page.svelte`

**Requirements**:
- Verify all URL parameters work: `session`, `lang`, `showSpeaker`, `showOriginal`, `fontSize`, `position` (top/center/bottom), `maxCaptions`, `fadeTime`, `bg`, `showStatus`
- Verify the setup help screen appears when no session parameter is provided
- Verify connection status indicator (colored dot) works: green=connected, yellow+count=reconnecting, red=disconnected
- Verify caption fade animations work correctly
- Verify max captions enforcement (oldest removed when limit exceeded)
- Verify caption aggregation (same speaker within time window appends text)

**Verification**: Open `/captions?session=test-123&showOriginal=true&showStatus=true&position=bottom&fontSize=28`. Send test captions via the Session Manager. Verify all parameters affect rendering correctly.

### Task 4.2: Toast Notifications

**Requirements**:
- All form actions and client-side operations must show appropriate success/error toasts
- Use the existing `toastStore` from `$lib/stores/toast.svelte.ts` and the sonner component
- Toasts should auto-dismiss after 5 seconds
- Error toasts should be red/destructive, success toasts green, info toasts blue

### Task 4.3: Error Handling and Loading States

**Requirements**:
- Every page that fetches data must have a loading state (skeleton or spinner)
- Every form action must disable the submit button while pending (use `use:enhance` properly)
- API errors must show meaningful messages (extract `detail` from FastAPI error responses)
- Network errors must show a retry-friendly message
- The `ApiError` class in `orchestration.ts` already handles status extraction -- ensure all API clients use it consistently

### Task 4.4: Responsive Layout

**Requirements**:
- The sidebar should collapse on mobile (add a hamburger toggle)
- All grids should stack on small screens (already using Tailwind responsive classes in some places)
- Tables should be horizontally scrollable on mobile
- The captions overlay should be fully responsive (already uses viewport units)

---

## 7. Model and LLM Selection Guide

This section explains the complete flow for how users select and manage translation models.

### How model discovery works

1. The orchestration service proxies to the translation service (port 5003). The translation service has a `/models` endpoint that returns all configured translation backends and their models.

2. The `/api/system/ui-config` endpoint (in `system.py`) fetches models from the translation service client and includes them in the response under `translation_models`. This is the recommended endpoint for the dashboard because it returns models alongside languages, domains, and defaults in a single request.

3. The `/api/translation/models` endpoint provides a direct route to the same data.

### Model data shape

From `/api/translation/models`, each model looks like:
```json
{
  "name": "ollama",
  "model": "qwen3:4b",
  "display_name": "Ollama (Local)",
  "backend": "openai_compatible"
}
```

The `name` field is the service identifier used when making translation requests (passed as the `service` parameter). The `model` field is the actual model within that service. The `display_name` is for UI display.

### How model selection works in the UI

1. **Translation Config page** (`/config/translation`): The `+page.server.ts` load function fetches models from `/api/translation/models` and health from `/api/translation/health`. The page renders:
   - Current model info card (from health endpoint: backend, device, available backends)
   - Model selector dropdown (from models endpoint)
   - Switch Model button -> calls `POST /api/translation/model` with the selected model name

2. **Connect form** (`/fireflies`): When connecting to a meeting, the selected model is sent as `translation_model` in the connect payload. The `+page.server.ts` should fetch available models and include them in the page data so the connect form can show a model selector.

3. **Translation Test Bench** (`/translation/test`): Already has a "Service" dropdown populated from models. This correctly uses the `backend` field as the service identifier.

4. **Persistence**: The old dashboard stored the selected model in `localStorage.fireflies_translation_model`. The SvelteKit dashboard should do the same for client-side persistence, while also notifying the backend via `POST /api/translation/model`.

### What the dashboard must show for models

- On the Translation Config page: full model management (list, switch, current info)
- On the Connect form: model selector so the user can choose which model to use for this session
- On the Translation Test Bench: model/service selector for test translations
- On the TopBar: Translation service status badge (Online/Offline) derived from `/api/translation/health`

---

## 8. Test Specifications

All tests must be behavioral, using real services. No mocks. Tests verify actual API responses and real data flow.

### Test Category A: Page Load and Data Fetching

**A1: Dashboard home shows real service health**
- Start orchestration service
- Navigate to `/`
- Verify the services card shows at least "orchestration" as healthy
- Verify the stats grid shows numbers (even if 0)
- Verify quick action links are present and clickable

**A2: Fireflies connect page loads real config**
- Navigate to `/fireflies`
- Verify the language selector contains languages from `/api/system/ui-config` (at minimum "Spanish", "French", "German")
- Verify domain selector is populated
- Verify active sessions are fetched (may be empty list)

**A3: Translation config shows real models**
- Navigate to `/config/translation`
- Verify model dropdown is populated from `/api/translation/models` (not hardcoded)
- Verify the current model info card shows backend and device information from `/api/translation/health`
- If translation service is running: verify "Active" badge appears
- If translation service is down: verify "Unavailable" state is shown gracefully

**A4: Glossary page loads real glossaries**
- Navigate to `/fireflies/glossary`
- Verify glossaries are listed (may be empty)
- If glossaries exist, verify entries table loads for the default glossary

### Test Category B: Form Actions Against Real Backend

**B1: Translation test bench round-trip**
- Navigate to `/translation/test`
- Enter "Hello, how are you today?" in the text field
- Select "Spanish" as target language
- Select a service/model from the dropdown
- Submit the form
- Verify the result panel shows: `translated_text` (non-empty string), `confidence` (number between 0 and 1), `processing_time` (positive number), `model_used` (non-empty), `backend_used` (non-empty)

**B2: Glossary entry CRUD**
- Navigate to `/fireflies/glossary`
- If no glossary exists, create one (name: "Test Glossary", target_languages: ["es"])
- Add a term: source="heart attack", translation="infarto de miocardio", language="es"
- Verify the term appears in the entries table
- Delete the term
- Verify the term is gone from the table

**B3: Fireflies connect and disconnect**
- Navigate to `/fireflies`
- Enter a test transcript ID
- Submit the connect form
- Verify redirect to the connect page with session data
- Click Disconnect
- Verify navigation back to `/fireflies`
- Verify the session no longer appears in the list

### Test Category C: Real-Time WebSocket

**C1: Caption overlay receives real captions**
- Create a session via Session Manager or API
- Open `/captions?session={sessionId}&showStatus=true`
- POST a test caption to `/api/captions/{sessionId}` with `{ text: "Hola", speaker_name: "Alice", target_language: "es" }`
- Verify the caption appears in the overlay within 2 seconds
- Verify the connection status dot is green

**C2: Live Feed receives real-time data**
- Start a demo via `POST /fireflies/demo/start?mode=pretranslated`
- Navigate to the Live Feed page
- Select the demo session and click Connect
- Verify entries appear in both the Original and Translation columns
- Verify speaker names are color-coded or labeled
- Stop the demo
- Verify the WebSocket disconnects

### Test Category D: Demo Mode End-to-End

**D1: Full demo lifecycle**
- Load the dashboard
- Click "Launch Demo" in the TopBar (or use the demo controls)
- Verify the dashboard navigates to Live Feed
- Verify the demo session appears in the session selector
- Verify captions flow in real-time in both columns
- Open the captions overlay URL and verify captions appear there too
- Click "Stop Demo"
- Verify the demo stops and the Live Feed disconnects
- Check `GET /fireflies/demo/status` returns `{ active: false }`

### Test Category E: Intelligence Features

**E1: Manual note creation**
- Ensure a session with transcript data exists
- Navigate to `/intelligence`
- Select the session
- Type a manual note and click "Add Note"
- Verify the note appears in the notes list with type "manual" and green border

**E2: Insight generation**
- Select a session with transcript data
- Select an insight template
- Click "Generate Insight"
- Verify an insight card appears with title, content, and processing time

**E3: Agent chat round-trip**
- Select a session
- Type "What were the main topics discussed?" in the chat input
- Press Enter or click Send
- Verify a response appears (either streamed or complete)

---

## Implementation Order Summary

| Batch | Tasks | Estimated Effort | Dependencies |
|---|---|---|---|
| **1: Critical Fixes** | 1.1 Dark Mode, 1.2 TopBar, 1.3 Language Selector, 1.4 Dashboard Home, 1.5 Sidebar | Small-Medium | None |
| **2: Core Features** | 2.1 API Key Mgmt, 2.2 Model Selection, 2.3 Live Feed, 2.4 Sessions | Medium-Large | Batch 1 |
| **3: Missing Pages** | 3.1 Data & Logs, 3.2 Intelligence, 3.3 History, 3.4 Glossary, 3.5 Session Mgr, 3.6 Demo Mode | Large | Batches 1-2 |
| **4: Polish & Test** | 4.1 Overlay Verification, 4.2 Toasts, 4.3 Error Handling, 4.4 Responsive | Medium | Batches 1-3 |

Each task within a batch can be worked on independently (no intra-batch ordering required). Batches should be completed in order since later batches depend on infrastructure from earlier ones.

---

## Key Architecture Reminders

- **API Client Pattern**: Always use `createApi(fetch)` from `$lib/api/orchestration.ts` in `+page.server.ts` files. The `fetch` parameter comes from SvelteKit and is SSR-safe. It proxies to `ORCHESTRATION_URL` (env var).
- **Client-Side Fetch**: For operations that need localStorage (API key, demo state), use direct browser `fetch()` from within `<script>` blocks or `$effect`. These cannot go through `+page.server.ts` because the server has no access to localStorage.
- **WebSocket**: Browser connects directly to orchestration service WebSocket. Use `WS_BASE` from `$lib/config.ts`. Do NOT proxy through SvelteKit.
- **Form Actions**: Use SvelteKit form actions with `use:enhance` for progressive enhancement. The action runs server-side in `+page.server.ts`. Return `fail()` for errors or redirect/data for success.
- **Dark Mode**: The `dark` class on `<html>` activates all CSS custom properties in the `.dark { }` block of `app.css`. This must be set by default.
- **Component Library**: shadcn-svelte components are already installed. Use `$lib/components/ui/*` for all UI elements. Dialog for modals, Table for data, Card for sections, Button/Input/Label/Select for forms.
- **No Mocks**: All tests must hit real endpoints. The orchestration service must be running for tests to pass.
