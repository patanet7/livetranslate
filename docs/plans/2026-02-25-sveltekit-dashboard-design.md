# SvelteKit Dashboard — Design Document

**Date**: 2025-02-25
**Status**: Approved
**Replaces**: React frontend-service (18 pages) + static Fireflies dashboard (9 tabs)
**Approach**: Clean-room phased rebuild — Approach A

---

## 1. Overview

Consolidate both existing UIs into a single standalone SvelteKit 5 application:

| Current UI | LOC | Technology | Status |
|---|---|---|---|
| React frontend-service | 10,653 | React 18 + MUI + Redux | Active, "messy" |
| Static Fireflies dashboard | 4,576 | Vanilla JS, 9 tabs | Active |

**Target**: One SvelteKit app at `modules/dashboard-service/` serving all functionality.

### Migration Phases

| Phase | Scope | Pages |
|---|---|---|
| **1 (MVP)** | Fireflies core + Config + Translation + Captions overlay | ~8 routes |
| **2** | Audio processing + Bot management | ~6 routes |
| **3** | Analytics + Chat history + Remaining pages | ~6 routes |

Each phase is a self-contained milestone. Old UIs stay running until a phase is proven.

---

## 2. Project Structure

```
modules/dashboard-service/
├── src/
│   ├── routes/
│   │   ├── +layout.svelte              # Root: global CSS, font loading only
│   │   ├── (app)/                      # Layout group: sidebar + top bar
│   │   │   ├── +layout.svelte          # App shell with navigation
│   │   │   ├── +layout.ts              # Client-side shared data (no blocking loads)
│   │   │   ├── +error.svelte           # App error page (sidebar intact)
│   │   │   ├── +page.svelte            # Dashboard home
│   │   │   ├── fireflies/
│   │   │   │   ├── +page.svelte        # Connect tab
│   │   │   │   ├── +page.server.ts     # Form action: connect to Fireflies
│   │   │   │   ├── connect/
│   │   │   │   │   ├── +page.svelte    # Active session streaming view
│   │   │   │   │   └── +page.server.ts # Load session data
│   │   │   │   ├── history/
│   │   │   │   │   ├── +page.svelte    # Session history browser
│   │   │   │   │   └── +page.server.ts # Streamed session list
│   │   │   │   └── glossary/
│   │   │   │       ├── +page.svelte    # Glossary term editor
│   │   │   │       └── +page.server.ts # Form actions: CRUD terms
│   │   │   ├── config/
│   │   │   │   ├── +page.svelte        # Config hub (links to sub-pages)
│   │   │   │   ├── audio/
│   │   │   │   │   ├── +page.svelte    # Audio processing settings
│   │   │   │   │   └── +page.server.ts # Form action: update audio config
│   │   │   │   ├── translation/
│   │   │   │   │   ├── +page.svelte    # Translation model/language config
│   │   │   │   │   └── +page.server.ts # Form action: update translation config
│   │   │   │   └── system/
│   │   │   │       ├── +page.svelte    # System settings, feature flags
│   │   │   │       └── +page.server.ts # Form action: update system config
│   │   │   └── translation/
│   │   │       └── test/
│   │   │           ├── +page.svelte    # Translation test bench
│   │   │           └── +page.server.ts # Form action: run translation
│   │   ├── (overlay)/                  # Layout group: bare, no chrome
│   │   │   ├── +layout.svelte          # Transparent, full-viewport
│   │   │   └── captions/
│   │   │       └── +page.svelte        # OBS caption overlay
│   │   └── +error.svelte               # Global error fallback
│   ├── lib/
│   │   ├── components/
│   │   │   ├── ui/                     # shadcn-svelte base components
│   │   │   ├── layout/                 # Sidebar, TopBar, PageHeader
│   │   │   ├── captions/              # CaptionBox, CaptionStream, InterimCaption
│   │   │   ├── fireflies/            # ConnectForm, SessionCard, GlossaryEditor
│   │   │   └── config/               # ConfigSection, ModelSelector, LanguagePicker
│   │   ├── stores/
│   │   │   ├── websocket.svelte.ts    # WebSocket connection + reconnect
│   │   │   ├── captions.svelte.ts     # Live captions + interim + expiry
│   │   │   ├── session.svelte.ts      # Active Fireflies sessions
│   │   │   ├── health.svelte.ts       # Client-side health polling
│   │   │   └── toast.svelte.ts        # Notification toasts
│   │   ├── api/
│   │   │   ├── orchestration.ts       # Server-side API client (accepts SvelteKit fetch)
│   │   │   ├── fireflies.ts           # Fireflies-specific endpoints
│   │   │   ├── config.ts              # Configuration endpoints
│   │   │   └── translation.ts         # Translation endpoints
│   │   ├── types/
│   │   │   ├── caption.ts
│   │   │   ├── session.ts
│   │   │   ├── config.ts
│   │   │   └── api.ts
│   │   └── config.ts                  # Environment config (WS_BASE, etc.)
│   ├── hooks.server.ts                # Request logging + timing
│   └── app.html                       # HTML template
├── static/                            # Favicon, OG images
├── tests/
│   ├── unit/                          # Vitest component + store tests
│   ├── e2e/                           # Playwright E2E tests
│   └── browser/                       # Agent-browser visual verification
│       ├── conftest.py                # agent-browser fixtures
│       ├── test_app_routes.py         # Route rendering verification
│       ├── test_fireflies_flow.py     # Connect → stream → captions flow
│       ├── test_config_forms.py       # Config form submissions
│       ├── test_captions_overlay.py   # OBS overlay rendering + modes
│       ├── test_translation_bench.py  # Translation test bench
│       └── screenshots/              # Visual verification screenshots
├── svelte.config.js
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── package.json
├── .env                               # Environment variables
└── .env.example                       # Documented env schema
```

---

## 3. Layout Groups

### `(app)` — Main Application Shell

```svelte
<!-- src/routes/(app)/+layout.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { healthStore } from '$lib/stores/health.svelte';
  import Sidebar from '$lib/components/layout/Sidebar.svelte';
  import TopBar from '$lib/components/layout/TopBar.svelte';

  let { children } = $props();

  onMount(() => {
    healthStore.startPolling();
    return () => healthStore.stopPolling();
  });
</script>

<div class="flex h-screen">
  <Sidebar />
  <div class="flex flex-col flex-1 overflow-hidden">
    <TopBar health={healthStore.status} />
    <main class="flex-1 overflow-y-auto p-6">
      {@render children()}
    </main>
  </div>
</div>
```

### `(overlay)` — OBS Browser Source

```svelte
<!-- src/routes/(overlay)/+layout.svelte -->
<script lang="ts">
  let { children } = $props();
</script>

<div class="overlay-root">
  {@render children()}
</div>

<style>
  .overlay-root {
    background: transparent;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
  }
</style>
```

No sidebar, no navigation, no chrome. Just captions on a transparent background.

---

## 4. State Management

### Principle: Three Layers

| Layer | Mechanism | Scope | SSR-Safe |
|---|---|---|---|
| Server state | `+page.server.ts` load functions | Per-page | Yes (server-only) |
| Shared stores | Svelte 5 runes in `.svelte.ts` | Cross-page | Must guard with `browser` |
| Component state | `$state` in components | Per-component | Yes |

### SSR Guard Pattern

All stores that touch browser APIs must check `browser` before acting:

```typescript
// src/lib/stores/websocket.svelte.ts
import { browser } from '$app/environment';

class WebSocketStore {
  url = $state('');
  status = $state<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  #socket: WebSocket | null = null;
  #reconnectAttempt = 0;
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #maxReconnectDelay = 30_000;
  onMessage: ((event: MessageEvent) => void) | null = null;

  connect(url: string) {
    if (!browser) return;
    this.url = url;
    this.status = 'connecting';

    this.#socket = new WebSocket(url);
    this.#socket.onopen = () => {
      this.status = 'connected';
      this.#reconnectAttempt = 0;
    };
    this.#socket.onmessage = (event) => this.onMessage?.(event);
    this.#socket.onclose = (event) => {
      this.status = 'disconnected';
      if (!event.wasClean) this.#scheduleReconnect();
    };
    this.#socket.onerror = () => { this.status = 'error'; };
  }

  send(data: unknown) {
    if (this.#socket?.readyState === WebSocket.OPEN) {
      this.#socket.send(JSON.stringify(data));
    }
  }

  disconnect() {
    this.#reconnectTimer && clearTimeout(this.#reconnectTimer);
    this.#socket?.close(1000, 'Client disconnect');
    this.#socket = null;
    this.status = 'disconnected';
    this.#reconnectAttempt = 0;
  }

  #scheduleReconnect() {
    const delay = Math.min(1000 * 2 ** this.#reconnectAttempt, this.#maxReconnectDelay);
    this.#reconnectAttempt++;
    this.#reconnectTimer = setTimeout(() => this.connect(this.url), delay);
  }
}

export const wsStore = new WebSocketStore();
```

### Health Polling (client-side, not in layout load)

```typescript
// src/lib/stores/health.svelte.ts
import { browser } from '$app/environment';

class HealthStore {
  status = $state<'healthy' | 'degraded' | 'down' | 'unknown'>('unknown');
  services = $state<Record<string, boolean>>({});
  #interval: ReturnType<typeof setInterval> | null = null;

  startPolling(intervalMs = 30_000) {
    if (!browser) return;
    this.#poll();
    this.#interval = setInterval(() => this.#poll(), intervalMs);
  }

  stopPolling() {
    this.#interval && clearInterval(this.#interval);
  }

  async #poll() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      this.status = data.status;
      this.services = data.services;
    } catch {
      this.status = 'down';
    }
  }
}

export const healthStore = new HealthStore();
```

---

## 5. API Integration

### Server-Side: Load Functions + Form Actions

Load functions are the proxy — they run on the SvelteKit server and fetch from orchestration:

```typescript
// src/lib/api/orchestration.ts
import { ORCHESTRATION_URL } from '$env/static/private';

export async function fetchSessions(fetch: typeof globalThis.fetch) {
  const res = await fetch(`${ORCHESTRATION_URL}/api/sessions`);
  if (!res.ok) throw new Error(`Sessions API returned ${res.status}`);
  return res.json() as Promise<Session[]>;
}
```

**Always accept `fetch` as a parameter** — SvelteKit provides a special `fetch` in load functions that handles cookies and relative URLs.

### Form Actions for Mutations

Every config page uses form actions for progressive enhancement:

```typescript
// src/routes/(app)/config/audio/+page.server.ts
import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { ORCHESTRATION_URL } from '$env/static/private';

export const load: PageServerLoad = async ({ fetch }) => {
  const res = await fetch(`${ORCHESTRATION_URL}/api/config/audio`);
  return { config: await res.json() };
};

export const actions: Actions = {
  update: async ({ request, fetch }) => {
    const data = await request.formData();
    const sampleRate = Number(data.get('sampleRate'));

    if (sampleRate < 8000 || sampleRate > 48000) {
      return fail(400, { sampleRate, errors: { sampleRate: 'Must be 8000-48000' } });
    }

    const res = await fetch(`${ORCHESTRATION_URL}/api/config/audio`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_rate: sampleRate })
    });

    if (!res.ok) return fail(res.status, { error: 'Backend update failed' });
    return { success: true };
  }
};
```

### Explicit `+server.ts` Routes — Only When Needed

Only create proxy server routes for:
1. Client-initiated fetches not tied to navigation (polling, search-as-you-type)
2. Mutations that can't use form actions (`DELETE` methods)
3. Blob/stream responses (audio downloads)
4. Health endpoint for client-side polling

```typescript
// src/routes/api/health/+server.ts
import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch }) => {
  const res = await fetch(`${ORCHESTRATION_URL}/api/system/health`);
  return new Response(res.body, {
    status: res.status,
    headers: { 'Content-Type': 'application/json' }
  });
};
```

### WebSocket: Direct Browser → Orchestration

WebSocket connections go directly from browser to orchestration service — SvelteKit can't proxy persistent WS connections:

```
Browser ──WS──→ ws://localhost:3000/ws/captions/{session_id}
Browser ──REST──→ SvelteKit (:5180) ──REST──→ Orchestration (:3000)
```

```typescript
// src/lib/config.ts
import { browser } from '$app/environment';
import { PUBLIC_WS_URL } from '$env/static/public';

export const WS_BASE = browser ? (PUBLIC_WS_URL || 'ws://localhost:3000') : '';
```

### Streaming for Slow Data

Session history and transcript pages use SvelteKit's streaming pattern:

```typescript
// src/routes/(app)/fireflies/history/+page.server.ts
export const load: PageServerLoad = async ({ fetch }) => {
  return {
    summary: await fetchSessionSummary(fetch),       // awaited — SSR-rendered
    sessions: fetchSessions(fetch)                    // NOT awaited — streamed
  };
};
```

```svelte
{#await data.sessions}
  <p>Loading sessions...</p>
{:then sessions}
  {#each sessions as session (session.id)}
    <SessionCard {session} />
  {/each}
{:catch error}
  <p>Failed to load: {error.message}</p>
{/await}
```

---

## 6. Component Library: shadcn-svelte

**Decision**: Use [shadcn-svelte](https://www.shadcn-svelte.com/) for base UI components.

**Why**:
- Components are copied into `src/lib/components/ui/` — we own the source
- Bits UI underneath provides accessible primitives (dialogs, selects, popovers, comboboxes)
- Tailwind-first styling — consistent with our setup
- Zero runtime dependency — nothing in the bundle we didn't choose
- Active Svelte 5 support

**Initial components to install**:
```bash
npx shadcn-svelte@latest init
npx shadcn-svelte@latest add button card dialog input select tabs toast badge table textarea separator
```

---

## 7. Styling: Tailwind CSS v4

- Tailwind v4 with SvelteKit's first-class integration
- `@tailwindcss/vite` plugin for zero-config setup
- Dark mode support via `class` strategy (toggleable)
- Captions overlay uses scoped `<style>` for pixel-perfect OBS rendering
- Design tokens (colors, spacing) defined in `tailwind.config.ts` for consistency

---

## 8. Environment Configuration

```bash
# .env
# Server-side only (not exposed to browser)
ORCHESTRATION_URL=http://localhost:3000

# Client-side (exposed to browser via PUBLIC_ prefix)
PUBLIC_WS_URL=ws://localhost:3000
PUBLIC_APP_NAME=LiveTranslate

# SvelteKit
ORIGIN=http://localhost:5180
```

Access via SvelteKit's built-in env modules:
- `$env/static/private` → server-only variables
- `$env/static/public` → client-visible variables (must start with `PUBLIC_`)

---

## 9. Error Handling

```
src/routes/
├── +error.svelte                    # Global fallback (no layout)
├── (app)/
│   └── +error.svelte               # App error (sidebar intact)
└── (overlay)/
    └── +error.svelte               # Overlay error (minimal, dark bg)
```

Each error page shows `$page.status` and `$page.error.message` with contextual navigation.

---

## 10. Server Hooks

```typescript
// src/hooks.server.ts
import type { Handle } from '@sveltejs/kit';

export const handle: Handle = async ({ event, resolve }) => {
  const start = performance.now();
  const response = await resolve(event);
  const duration = Math.round(performance.now() - start);
  console.log(`${event.request.method} ${event.url.pathname} ${response.status} ${duration}ms`);
  return response;
};
```

---

## 11. Dependencies (Minimal)

```json
{
  "name": "@livetranslate/dashboard-service",
  "type": "module",
  "scripts": {
    "dev": "vite dev --port 5180",
    "build": "vite build",
    "preview": "vite preview --port 5180",
    "check": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json",
    "test": "vitest run",
    "test:e2e": "playwright test"
  },
  "dependencies": {
    "@sveltejs/adapter-node": "^5",
    "@sveltejs/kit": "^2",
    "svelte": "^5"
  },
  "devDependencies": {
    "@playwright/test": "^1",
    "@sveltejs/vite-plugin-svelte": "^4",
    "@tailwindcss/vite": "^4",
    "tailwindcss": "^4",
    "typescript": "^5",
    "vite": "^6",
    "vitest": "^2",
    "svelte-check": "^4"
  }
}
```

No MUI. No Redux. No Emotion. No RxJS. Just SvelteKit + Tailwind + shadcn-svelte.

---

## 12. Testing Strategy

### 12a. Unit Tests (Vitest)

- Store logic (WebSocket reconnect, caption expiry, health polling)
- API client functions
- Type validations
- Run with `vitest run`

### 12b. E2E Tests (Playwright)

- Full page rendering for every route
- Form submission flows (config pages)
- Navigation between pages
- Error page rendering
- Run with `playwright test`

### 12c. Agent-Browser Visual Verification

**Every route and flow must be verified with agent-browser screenshots.** This is the definitive acceptance test.

#### Test Structure

```
tests/browser/
├── conftest.py                     # Fixtures: SvelteKit dev server, agent-browser
├── test_app_routes.py              # Screenshot every route, verify render
├── test_fireflies_flow.py          # Connect → session → captions end-to-end
├── test_config_forms.py            # Fill + submit every config form
├── test_captions_overlay.py        # OBS overlay: display modes, params, styling
├── test_translation_bench.py       # Translation input → output verification
├── test_navigation.py              # Sidebar nav, breadcrumbs, back/forward
├── test_error_states.py            # Error pages, offline orchestration
└── screenshots/                    # Timestamped visual evidence
```

#### Verification Checklist

For each route, agent-browser must:
1. **Navigate** to the page
2. **Take a screenshot** (saved to `screenshots/`)
3. **Verify DOM structure** — expected elements present (headings, forms, buttons)
4. **Verify no errors** — no console errors, no error boundaries triggered
5. **Verify responsive layout** — sidebar collapses, content reflows

For interactive flows:
1. **Fill forms** and submit via form actions
2. **Verify success states** — toast notifications, updated values
3. **Verify error states** — validation messages, backend failure handling
4. **Screenshot before and after** each action

For the captions overlay:
1. **Verify URL params** — `?session=X&lang=en&fontSize=24` configures correctly
2. **Verify display modes** — `?mode=both`, `?mode=translated`, `?mode=english`
3. **Verify transparent background** — no chrome, no padding leaks
4. **Verify live updates** — WebSocket captions render in real-time
5. **Screenshot each mode** with captions visible

#### Fixtures

```python
# tests/browser/conftest.py
import pytest
import subprocess
import time
import httpx

@pytest.fixture(scope="session")
def sveltekit_server():
    """Start SvelteKit dev server for testing."""
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="modules/dashboard-service",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server ready
    for _ in range(30):
        try:
            resp = httpx.get("http://localhost:5180/", timeout=1)
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(1)
    yield "http://localhost:5180"
    proc.terminate()
    proc.wait(timeout=5)

@pytest.fixture
def browser(sveltekit_server):
    """Agent-browser instance pointed at SvelteKit server."""
    from tests.fireflies.e2e.browser.browser_helpers import AgentBrowser
    b = AgentBrowser()
    b.open(sveltekit_server)
    yield b
    b.close()
```

---

## 13. Development Workflow

### Running Alongside Existing Services

```bash
# Terminal 1: Orchestration service (backend API)
cd modules/orchestration-service && uv run python src/main_fastapi.py  # :3000

# Terminal 2: New SvelteKit dashboard (development)
cd modules/dashboard-service && npm run dev                            # :5180

# Terminal 3: Old React frontend (kept running until phase complete)
cd modules/frontend-service && npm run dev                             # :5173
```

Both UIs can run simultaneously during migration. The SvelteKit app talks to the same orchestration service.

### Build & Deploy

```bash
npm run build        # Produces Node server in build/
node build/index.js  # Production server
```

Uses `@sveltejs/adapter-node` for standalone Node.js deployment.

---

## 14. Phase 1 MVP — Detailed Route Inventory

| Route | Purpose | Data Source | Mutations |
|---|---|---|---|
| `/` | Dashboard home | Health store (client poll) | None |
| `/fireflies` | Connect form | None (form only) | Form action: POST /fireflies/connect |
| `/fireflies/connect` | Live streaming session | Load: session data + WS captions | Form action: disconnect |
| `/fireflies/history` | Session history browser | Load (streamed): session list | None |
| `/fireflies/glossary` | Glossary term management | Load: glossary terms | Form actions: add/edit/delete |
| `/config` | Config hub | Load: current config | None |
| `/config/audio` | Audio settings | Load: audio config | Form action: update |
| `/config/translation` | Translation settings | Load: translation config | Form action: update |
| `/config/system` | System settings | Load: system config | Form action: update |
| `/translation/test` | Translation test bench | None (interactive) | Form action: translate |
| `/captions` | OBS overlay | URL params + WS | None |

---

## 15. API Endpoints Consumed (Phase 1)

### Fireflies
- `POST /fireflies/connect` — Start session
- `POST /fireflies/disconnect` — Stop session
- `GET /fireflies/sessions` — List active sessions
- `WS /api/captions/stream/{session_id}` — Live caption stream

### Configuration
- `GET /api/config` — Full configuration
- `PUT /api/config/audio` — Update audio config
- `PUT /api/config/translation` — Update translation config
- `PUT /api/config/system` — Update system config

### Translation
- `POST /api/translations/translate` — Run translation

### System
- `GET /api/system/health` — Service health
- `GET /api/system/services` — Service status

### Glossary
- `GET /api/glossary` — List terms
- `POST /api/glossary` — Add term
- `PUT /api/glossary/{id}` — Update term
- `DELETE /api/glossary/{id}` — Delete term

---

## 16. Architectural Decisions Record

| Decision | Choice | Rationale |
|---|---|---|
| Framework | SvelteKit 5 with runes | Compiler-driven, minimal JS, full-stack |
| Styling | Tailwind CSS v4 | Zero runtime, utility-first, SvelteKit native |
| Components | shadcn-svelte | Owned source, accessible primitives, Tailwind-first |
| State | Svelte 5 runes + stores | No external library needed |
| API pattern | Load functions + form actions | Idiomatic SvelteKit, progressive enhancement |
| REST proxy | SvelteKit server-side fetch | Single origin, no CORS |
| WebSocket | Direct browser → orchestration | Can't proxy persistent WS through SvelteKit |
| Deployment | Standalone Node service (adapter-node) | Clean separation from orchestration |
| Verification | Agent-browser + Playwright + Vitest | Full visual + E2E + unit coverage |
| Migration | Clean-room rebuild in phases | No legacy baggage carried forward |
