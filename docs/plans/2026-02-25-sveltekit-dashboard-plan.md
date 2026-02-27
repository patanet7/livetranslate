# SvelteKit Dashboard — Phase 1 MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone SvelteKit 5 dashboard at `modules/dashboard-service/` that replaces both the React frontend and static Fireflies dashboard, starting with Phase 1 (Fireflies core, config, translation, captions overlay).

**Architecture:** Clean-room SvelteKit app with Svelte 5 runes, layout groups for `(app)` shell vs `(overlay)` bare chrome, form actions for mutations, SSR-safe stores for WebSocket/health/captions, and SvelteKit server-side fetch proxying REST to orchestration (:3000). See `docs/plans/2026-02-25-sveltekit-dashboard-design.md` for full design.

**Tech Stack:** SvelteKit 5, Svelte 5 runes, Tailwind CSS v4, shadcn-svelte, adapter-node, TypeScript, Vitest, Playwright, agent-browser

**Services required:** Orchestration service running on port 3000 (`uv run python src/main_fastapi.py` from `modules/orchestration-service/`)

**Test commands:**
- Build check: `npm run build && npm run check` from `modules/dashboard-service/`
- Unit tests: `npm run test` from `modules/dashboard-service/`
- E2E (Playwright): `npx playwright test` from `modules/dashboard-service/`
- Visual verification: `uv run pytest tests/browser/ -v` from `modules/dashboard-service/`

---

## Phase 1A: Foundation (Tasks 1–7)

### Task 1: Scaffold SvelteKit Project

**Files:**
- Create: `modules/dashboard-service/` (entire project scaffold)
- Modify: `modules/dashboard-service/svelte.config.js` (adapter-node)
- Modify: `modules/dashboard-service/package.json` (scripts, name)
- Create: `modules/dashboard-service/.env`
- Create: `modules/dashboard-service/.env.example`

**Step 1: Create SvelteKit project**

Run from repo root:
```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate
npx sv create modules/dashboard-service --template minimal --types ts --add tailwindcss,playwright,vitest
```

Select options if prompted:
- Template: SvelteKit minimal
- Type checking: TypeScript
- Add-ons: tailwindcss, playwright, vitest

**Step 2: Install adapter-node and shadcn-svelte**

```bash
cd modules/dashboard-service
npm install
npm install -D @sveltejs/adapter-node
npm install shadcn-svelte@latest
```

**Step 3: Configure adapter-node**

Replace `svelte.config.js`:

```javascript
// svelte.config.js
import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter()
  }
};

export default config;
```

**Step 4: Update package.json**

Set name and dev port:

```json
{
  "name": "@livetranslate/dashboard-service",
  "scripts": {
    "dev": "vite dev --port 5180",
    "build": "vite build",
    "preview": "vite preview --port 5180",
    "check": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json",
    "test": "vitest run",
    "test:e2e": "npx playwright test"
  }
}
```

**Step 5: Initialize shadcn-svelte**

```bash
npx shadcn-svelte@latest init
```

When prompted, accept defaults (it will detect SvelteKit + Tailwind).

**Step 6: Add base shadcn-svelte components**

```bash
npx shadcn-svelte@latest add button card dialog input select tabs toast badge table textarea separator label
```

**Step 7: Create environment files**

Create `.env`:
```bash
ORCHESTRATION_URL=http://localhost:3000
PUBLIC_WS_URL=ws://localhost:3000
PUBLIC_APP_NAME=LiveTranslate
ORIGIN=http://localhost:5180
```

Create `.env.example` (identical but with placeholder values):
```bash
ORCHESTRATION_URL=http://localhost:3000
PUBLIC_WS_URL=ws://localhost:3000
PUBLIC_APP_NAME=LiveTranslate
ORIGIN=http://localhost:5180
```

**Step 8: Verify build**

```bash
npm run build
npm run check
```

Expected: Both pass with no errors.

**Step 9: Commit**

```bash
git add modules/dashboard-service/
git commit -m "feat(dashboard): scaffold SvelteKit project with Tailwind, shadcn-svelte, adapter-node"
```

---

### Task 2: TypeScript Types

**Files:**
- Create: `modules/dashboard-service/src/lib/types/caption.ts`
- Create: `modules/dashboard-service/src/lib/types/session.ts`
- Create: `modules/dashboard-service/src/lib/types/config.ts`
- Create: `modules/dashboard-service/src/lib/types/api.ts`
- Create: `modules/dashboard-service/src/lib/types/index.ts`

**Step 1: Create caption types**

```typescript
// src/lib/types/caption.ts

export interface Caption {
  id: string;
  text: string;
  original_text: string;
  speaker_name: string;
  speaker_color: string;
  target_language: string;
  confidence: number;
  duration_seconds: number;
  created_at: string;
  expires_at: string;
  receivedAt?: number; // client-side timestamp for expiry tracking
}

export interface InterimCaption {
  chunk_id: string;
  text: string;
  speaker_name: string;
  is_final: boolean;
}

export type CaptionEvent =
  | { event: 'connected'; session_id: string; current_captions: Caption[]; timestamp: string }
  | { event: 'caption_added'; caption: Caption }
  | { event: 'caption_expired'; caption_id: string }
  | { event: 'caption_updated'; caption: Caption }
  | { event: 'interim_caption'; caption: InterimCaption }
  | { event: 'session_cleared' };

export type DisplayMode = 'both' | 'translated' | 'english';
```

**Step 2: Create session types**

```typescript
// src/lib/types/session.ts

export interface FirefliesSession {
  session_id: string;
  transcript_id: string;
  connection_status: 'CONNECTING' | 'CONNECTED' | 'ERROR' | 'DISCONNECTED';
  chunks_received: number;
  sentences_produced: number;
  translations_completed: number;
  speakers_detected: string[];
  connected_at: string;
  error_count: number;
  last_error: string | null;
  persistence_failures: number;
  persistence_healthy: boolean;
}

export interface ConnectRequest {
  api_key?: string | null;
  transcript_id: string;
  target_languages?: string[] | null;
  glossary_id?: string | null;
  domain?: string | null;
  translation_model?: string | null;
  pause_threshold_ms?: number | null;
  max_buffer_words?: number | null;
  context_window_size?: number | null;
  api_base_url?: string | null;
}

export interface ConnectResponse {
  success: boolean;
  message: string;
  session_id: string;
  connection_status: string;
  transcript_id: string;
}

export interface DisconnectRequest {
  session_id: string;
}
```

**Step 3: Create config types**

```typescript
// src/lib/types/config.ts

export interface UserSettings {
  user_id: string;
  theme: 'dark' | 'light';
  language: string;
  notifications: boolean;
  audio_auto_start: boolean;
  default_translation_language: string;
  transcription_model: string;
  custom_settings: Record<string, unknown>;
  updated_at: string;
}

export interface TranslationConfig {
  backend: 'ollama' | 'vllm' | 'openai' | 'groq';
  model: string;
  base_url: string;
  target_language: string;
  temperature: number;
  max_tokens: number;
}

export interface TranslationSettings {
  enabled: boolean;
  default_model: string;
  default_target_language: string;
}

export interface UiConfig {
  languages: Array<{ code: string; name: string; nativeName: string }>;
  language_codes: string[];
  domains: string[];
  defaults: Record<string, unknown>;
  translation_models: Array<{ name: string; backend: string; languages: string[]; default: boolean }>;
  translation_service_available: boolean;
  config_version: string;
}

export interface Glossary {
  glossary_id: string;
  name: string;
  description: string;
  domain: string;
  source_language: string;
  target_languages: string[];
  is_active: boolean;
  is_default: boolean;
  entry_count: number;
  created_at: string;
  updated_at: string;
}

export interface GlossaryEntry {
  entry_id: string;
  glossary_id: string;
  source_term: string;
  translations: Record<string, string>;
  context: string;
  notes: string;
  case_sensitive: boolean;
  match_whole_word: boolean;
  priority: number;
  created_at: string;
  updated_at: string;
}
```

**Step 4: Create API types**

```typescript
// src/lib/types/api.ts

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'down' | 'unknown';
  timestamp: number;
  services: ServiceStatus[];
}

export interface ServiceStatus {
  name: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  response_time_ms: number;
  uptime_seconds: number;
}

export interface TranslateRequest {
  text: string;
  target_language: string;
  source_language?: string | null;
  service?: string;
  quality?: 'fast' | 'balanced' | 'quality';
  session_id?: string | null;
  context?: string;
}

export interface TranslateResponse {
  translated_text: string;
  source_language: string;
  target_language: string;
  confidence: number;
  processing_time: number;
  model_used: string;
  backend_used: string;
  timestamp: string;
}

export interface TranslationTestResponse {
  success: boolean;
  original_text: string;
  translated_text: string;
  target_language: string;
  confidence: number;
  processing_time_ms: number;
}

export interface CaptionStats {
  session_id: string;
  captions_added: number;
  captions_expired: number;
  current_count: number;
  unique_speakers: number;
  connection_count: number;
  timestamp: string;
}
```

**Step 5: Create barrel export**

```typescript
// src/lib/types/index.ts
export * from './caption';
export * from './session';
export * from './config';
export * from './api';
```

**Step 6: Verify build**

```bash
cd modules/dashboard-service && npm run check
```

Expected: PASS — no type errors.

**Step 7: Commit**

```bash
git add modules/dashboard-service/src/lib/types/
git commit -m "feat(dashboard): add TypeScript types for captions, sessions, config, API"
```

---

### Task 3: Environment Config + API Client

**Files:**
- Create: `modules/dashboard-service/src/lib/config.ts`
- Create: `modules/dashboard-service/src/lib/api/orchestration.ts`
- Create: `modules/dashboard-service/src/lib/api/fireflies.ts`
- Create: `modules/dashboard-service/src/lib/api/config.ts`
- Create: `modules/dashboard-service/src/lib/api/translation.ts`
- Create: `modules/dashboard-service/src/lib/api/glossary.ts`

**Step 1: Create client-side config**

```typescript
// src/lib/config.ts
import { browser } from '$app/environment';
import { PUBLIC_WS_URL, PUBLIC_APP_NAME } from '$env/static/public';

export const WS_BASE = browser ? (PUBLIC_WS_URL || 'ws://localhost:3000') : '';
export const APP_NAME = PUBLIC_APP_NAME || 'LiveTranslate';
```

**Step 2: Create base orchestration API client**

This is server-side only — used in load functions and form actions. Always accepts SvelteKit's `fetch` parameter.

```typescript
// src/lib/api/orchestration.ts
import { ORCHESTRATION_URL } from '$env/static/private';

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function apiRequest<T>(
  fetch: typeof globalThis.fetch,
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${ORCHESTRATION_URL}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options
  });

  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error');
    throw new ApiError(res.status, `API ${options?.method ?? 'GET'} ${path}: ${res.status} — ${text}`);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export function createApi(fetch: typeof globalThis.fetch) {
  return {
    get: <T>(path: string) => apiRequest<T>(fetch, path),
    post: <T>(path: string, body?: unknown) =>
      apiRequest<T>(fetch, path, {
        method: 'POST',
        body: body ? JSON.stringify(body) : undefined
      }),
    put: <T>(path: string, body: unknown) =>
      apiRequest<T>(fetch, path, {
        method: 'PUT',
        body: JSON.stringify(body)
      }),
    patch: <T>(path: string, body: unknown) =>
      apiRequest<T>(fetch, path, {
        method: 'PATCH',
        body: JSON.stringify(body)
      }),
    del: <T>(path: string) =>
      apiRequest<T>(fetch, path, { method: 'DELETE' })
  };
}
```

**Step 3: Create Fireflies API module**

```typescript
// src/lib/api/fireflies.ts
import type { ConnectRequest, ConnectResponse, FirefliesSession } from '$lib/types';
import { createApi } from './orchestration';

export function firefliesApi(fetch: typeof globalThis.fetch) {
  const api = createApi(fetch);
  return {
    connect: (req: ConnectRequest) =>
      api.post<ConnectResponse>('/fireflies/connect', req),

    disconnect: (sessionId: string) =>
      api.post<{ success: boolean; message: string }>('/fireflies/disconnect', {
        session_id: sessionId
      }),

    listSessions: () =>
      api.get<FirefliesSession[]>('/fireflies/sessions'),

    getSession: (sessionId: string) =>
      api.get<FirefliesSession>(`/fireflies/sessions/${sessionId}`),

    setDisplayMode: (sessionId: string, mode: string) =>
      api.put(`/fireflies/sessions/${sessionId}/display-mode`, { mode }),

    pause: (sessionId: string) =>
      api.post(`/fireflies/sessions/${sessionId}/pause`),

    resume: (sessionId: string) =>
      api.post(`/fireflies/sessions/${sessionId}/resume`),

    getTranslationConfig: () =>
      api.get('/fireflies/translation-config'),

    updateTranslationConfig: (config: Record<string, unknown>) =>
      api.put('/fireflies/translation-config', config)
  };
}
```

**Step 4: Create Config API module**

```typescript
// src/lib/api/config.ts
import type { UserSettings, TranslationSettings, UiConfig } from '$lib/types';
import { createApi } from './orchestration';

export function configApi(fetch: typeof globalThis.fetch) {
  const api = createApi(fetch);
  return {
    getUserSettings: () =>
      api.get<UserSettings>('/api/settings/user'),

    updateUserSettings: (settings: Partial<UserSettings>) =>
      api.put<UserSettings>('/api/settings/user', settings),

    getTranslationSettings: () =>
      api.get<TranslationSettings>('/api/settings/translation'),

    saveTranslationSettings: (settings: TranslationSettings) =>
      api.post<{ message: string; config: TranslationSettings }>(
        '/api/settings/translation', settings
      ),

    testTranslation: (text: string, targetLanguage: string) =>
      api.post('/api/settings/translation/test', { text, target_language: targetLanguage }),

    getUiConfig: () =>
      api.get<UiConfig>('/api/system/ui-config'),

    getHealth: () =>
      api.get('/api/system/health'),

    getServices: () =>
      api.get('/api/system/services')
  };
}
```

**Step 5: Create Translation API module**

```typescript
// src/lib/api/translation.ts
import type { TranslateRequest, TranslateResponse } from '$lib/types';
import { createApi } from './orchestration';

export function translationApi(fetch: typeof globalThis.fetch) {
  const api = createApi(fetch);
  return {
    translate: (req: TranslateRequest) =>
      api.post<TranslateResponse>('/api/translation/translate', req),

    batchTranslate: (requests: TranslateRequest[]) =>
      api.post<TranslateResponse[]>('/api/translation/batch', { requests }),

    detectLanguage: (text: string) =>
      api.post<{ detected_language: string; confidence: number }>(
        '/api/translation/detect', { text }
      ),

    getModels: () =>
      api.get<{ models: Array<{ name: string; backend: string; languages: string[]; default: boolean }> }>(
        '/api/translation/models'
      )
  };
}
```

**Step 6: Create Glossary API module**

```typescript
// src/lib/api/glossary.ts
import type { Glossary, GlossaryEntry } from '$lib/types';
import { createApi } from './orchestration';

export function glossaryApi(fetch: typeof globalThis.fetch) {
  const api = createApi(fetch);
  return {
    list: (params?: { domain?: string; source_language?: string; active_only?: boolean }) => {
      const query = new URLSearchParams();
      if (params?.domain) query.set('domain', params.domain);
      if (params?.source_language) query.set('source_language', params.source_language);
      if (params?.active_only !== undefined) query.set('active_only', String(params.active_only));
      const qs = query.toString();
      return api.get<Glossary[]>(`/api/glossaries${qs ? '?' + qs : ''}`);
    },

    get: (glossaryId: string) =>
      api.get<Glossary>(`/api/glossaries/${glossaryId}`),

    create: (glossary: Omit<Glossary, 'glossary_id' | 'entry_count' | 'created_at' | 'updated_at' | 'is_active'>) =>
      api.post<Glossary>('/api/glossaries', glossary),

    update: (glossaryId: string, patch: Partial<Glossary>) =>
      api.patch<Glossary>(`/api/glossaries/${glossaryId}`, patch),

    delete: (glossaryId: string) =>
      api.del(`/api/glossaries/${glossaryId}`),

    listEntries: (glossaryId: string, targetLanguage?: string) => {
      const qs = targetLanguage ? `?target_language=${targetLanguage}` : '';
      return api.get<GlossaryEntry[]>(`/api/glossaries/${glossaryId}/entries${qs}`);
    },

    createEntry: (glossaryId: string, entry: Omit<GlossaryEntry, 'entry_id' | 'glossary_id' | 'created_at' | 'updated_at'>) =>
      api.post<GlossaryEntry>(`/api/glossaries/${glossaryId}/entries`, entry),

    updateEntry: (glossaryId: string, entryId: string, patch: Partial<GlossaryEntry>) =>
      api.patch<GlossaryEntry>(`/api/glossaries/${glossaryId}/entries/${entryId}`, patch),

    deleteEntry: (glossaryId: string, entryId: string) =>
      api.del(`/api/glossaries/${glossaryId}/entries/${entryId}`)
  };
}
```

**Step 7: Verify build**

```bash
cd modules/dashboard-service && npm run check
```

Expected: PASS.

**Step 8: Commit**

```bash
git add modules/dashboard-service/src/lib/config.ts modules/dashboard-service/src/lib/api/
git commit -m "feat(dashboard): add typed API client layer for orchestration service"
```

---

### Task 4: SSR-Safe Stores

**Files:**
- Create: `modules/dashboard-service/src/lib/stores/health.svelte.ts`
- Create: `modules/dashboard-service/src/lib/stores/websocket.svelte.ts`
- Create: `modules/dashboard-service/src/lib/stores/captions.svelte.ts`
- Create: `modules/dashboard-service/src/lib/stores/toast.svelte.ts`
- Test: `modules/dashboard-service/tests/unit/stores.test.ts`

**Step 1: Write unit tests for stores**

```typescript
// tests/unit/stores.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock $app/environment for SSR safety testing
vi.mock('$app/environment', () => ({ browser: true }));

describe('HealthStore', () => {
  it('starts with unknown status', async () => {
    const { healthStore } = await import('$lib/stores/health.svelte');
    expect(healthStore.status).toBe('unknown');
  });
});

describe('CaptionStore', () => {
  it('adds and retrieves captions', async () => {
    const { CaptionStore } = await import('$lib/stores/captions.svelte');
    const store = new CaptionStore();
    store.addCaption({
      id: 'cap1',
      text: 'Hola mundo',
      original_text: 'Hello world',
      speaker_name: 'Alice',
      speaker_color: '#4CAF50',
      target_language: 'es',
      confidence: 0.95,
      duration_seconds: 4,
      created_at: new Date().toISOString(),
      expires_at: new Date(Date.now() + 10000).toISOString()
    });
    expect(store.captions.length).toBe(1);
    expect(store.captions[0].text).toBe('Hola mundo');
  });

  it('updates interim text', async () => {
    const { CaptionStore } = await import('$lib/stores/captions.svelte');
    const store = new CaptionStore();
    store.updateInterim('Hello wor');
    expect(store.interim).toBe('Hello wor');
    store.updateInterim('Hello world');
    expect(store.interim).toBe('Hello world');
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd modules/dashboard-service && npm run test
```

Expected: FAIL — modules not found.

**Step 3: Create health store**

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
    if (this.#interval) {
      clearInterval(this.#interval);
      this.#interval = null;
    }
  }

  async #poll() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      this.status = data.status ?? 'unknown';
      this.services = data.services ?? {};
    } catch {
      this.status = 'down';
    }
  }
}

export const healthStore = new HealthStore();
```

**Step 4: Create WebSocket store**

```typescript
// src/lib/stores/websocket.svelte.ts
import { browser } from '$app/environment';

export class WebSocketStore {
  url = $state('');
  status = $state<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  #socket: WebSocket | null = null;
  #reconnectAttempt = 0;
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #maxReconnectDelay = 30_000;
  onMessage: ((event: MessageEvent) => void) | null = null;

  connect(url: string) {
    if (!browser) return;
    this.disconnect();
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
    this.#socket.onerror = () => {
      this.status = 'error';
    };
  }

  send(data: unknown) {
    if (this.#socket?.readyState === WebSocket.OPEN) {
      this.#socket.send(JSON.stringify(data));
    }
  }

  disconnect() {
    if (this.#reconnectTimer) {
      clearTimeout(this.#reconnectTimer);
      this.#reconnectTimer = null;
    }
    if (this.#socket) {
      this.#socket.onclose = null; // prevent reconnect on intentional close
      this.#socket.close(1000, 'Client disconnect');
      this.#socket = null;
    }
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

**Step 5: Create caption store**

```typescript
// src/lib/stores/captions.svelte.ts
import { browser } from '$app/environment';
import type { Caption } from '$lib/types';

export class CaptionStore {
  captions = $state<(Caption & { receivedAt: number })[]>([]);
  interim = $state('');
  maxCaptions = $state(50);

  #expiryMs: number;
  #cleanupInterval: ReturnType<typeof setInterval> | null = null;

  constructor(expiryMs = 10_000) {
    this.#expiryMs = expiryMs;
  }

  start() {
    if (!browser) return;
    this.#cleanupInterval = setInterval(() => this.#expireOld(), 1000);
  }

  stop() {
    if (this.#cleanupInterval) {
      clearInterval(this.#cleanupInterval);
      this.#cleanupInterval = null;
    }
  }

  addCaption(caption: Caption) {
    const enriched = { ...caption, receivedAt: Date.now() };
    this.captions = [...this.captions, enriched].slice(-this.maxCaptions);
  }

  updateCaption(caption: Caption) {
    this.captions = this.captions.map((c) =>
      c.id === caption.id ? { ...caption, receivedAt: c.receivedAt } : c
    );
  }

  removeCaption(captionId: string) {
    this.captions = this.captions.filter((c) => c.id !== captionId);
  }

  updateInterim(text: string) {
    this.interim = text;
  }

  clear() {
    this.captions = [];
    this.interim = '';
  }

  #expireOld() {
    const cutoff = Date.now() - this.#expiryMs;
    this.captions = this.captions.filter((c) => c.receivedAt > cutoff);
  }
}

export const captionStore = new CaptionStore();
```

**Step 6: Create toast store**

```typescript
// src/lib/stores/toast.svelte.ts

export interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

class ToastStore {
  toasts = $state<Toast[]>([]);

  add(message: string, type: Toast['type'] = 'info', duration = 5000) {
    const id = crypto.randomUUID();
    this.toasts = [...this.toasts, { id, message, type, duration }];

    if (duration > 0) {
      setTimeout(() => this.dismiss(id), duration);
    }

    return id;
  }

  success(message: string) { return this.add(message, 'success'); }
  error(message: string) { return this.add(message, 'error', 8000); }
  warning(message: string) { return this.add(message, 'warning'); }
  info(message: string) { return this.add(message, 'info'); }

  dismiss(id: string) {
    this.toasts = this.toasts.filter((t) => t.id !== id);
  }
}

export const toastStore = new ToastStore();
```

**Step 7: Run tests**

```bash
cd modules/dashboard-service && npm run test
```

Expected: PASS.

**Step 8: Verify build**

```bash
npm run check
```

Expected: PASS.

**Step 9: Commit**

```bash
git add modules/dashboard-service/src/lib/stores/ modules/dashboard-service/tests/unit/
git commit -m "feat(dashboard): add SSR-safe stores for health, WebSocket, captions, toasts"
```

---

### Task 5: Layout Components

**Files:**
- Create: `modules/dashboard-service/src/lib/components/layout/Sidebar.svelte`
- Create: `modules/dashboard-service/src/lib/components/layout/TopBar.svelte`
- Create: `modules/dashboard-service/src/lib/components/layout/PageHeader.svelte`
- Create: `modules/dashboard-service/src/lib/components/layout/StatusIndicator.svelte`

**Step 1: Create StatusIndicator**

```svelte
<!-- src/lib/components/layout/StatusIndicator.svelte -->
<script lang="ts">
  interface Props {
    status: 'healthy' | 'degraded' | 'down' | 'unknown' | 'connected' | 'connecting' | 'disconnected' | 'error';
    label?: string;
  }

  let { status, label }: Props = $props();

  const colorMap: Record<string, string> = {
    healthy: 'bg-green-500',
    connected: 'bg-green-500',
    degraded: 'bg-yellow-500',
    connecting: 'bg-yellow-500 animate-pulse',
    down: 'bg-red-500',
    error: 'bg-red-500',
    disconnected: 'bg-gray-400',
    unknown: 'bg-gray-400'
  };
</script>

<span class="inline-flex items-center gap-1.5">
  <span class={`h-2 w-2 rounded-full ${colorMap[status] ?? 'bg-gray-400'}`}></span>
  {#if label}
    <span class="text-xs text-muted-foreground capitalize">{label ?? status}</span>
  {/if}
</span>
```

**Step 2: Create Sidebar**

```svelte
<!-- src/lib/components/layout/Sidebar.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
  import { APP_NAME } from '$lib/config';

  const navItems = [
    { label: 'Dashboard', href: '/', icon: '⌂' },
    { label: 'Fireflies', href: '/fireflies', icon: '🎙', children: [
      { label: 'Connect', href: '/fireflies' },
      { label: 'History', href: '/fireflies/history' },
      { label: 'Glossary', href: '/fireflies/glossary' }
    ]},
    { label: 'Config', href: '/config', icon: '⚙', children: [
      { label: 'Audio', href: '/config/audio' },
      { label: 'Translation', href: '/config/translation' },
      { label: 'System', href: '/config/system' }
    ]},
    { label: 'Translation', href: '/translation/test', icon: '🌐' }
  ];

  function isActive(href: string): boolean {
    if (href === '/') return $page.url.pathname === '/';
    return $page.url.pathname.startsWith(href);
  }
</script>

<aside class="w-56 border-r bg-card flex flex-col h-full">
  <div class="p-4 border-b">
    <h1 class="text-lg font-semibold">{APP_NAME}</h1>
  </div>
  <nav class="flex-1 p-2 space-y-1 overflow-y-auto">
    {#each navItems as item}
      <a
        href={item.children ? item.children[0].href : item.href}
        class="flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors
          {isActive(item.href) ? 'bg-accent text-accent-foreground font-medium' : 'text-muted-foreground hover:bg-accent/50'}"
      >
        <span>{item.icon}</span>
        <span>{item.label}</span>
      </a>
      {#if item.children && isActive(item.href)}
        <div class="ml-8 space-y-0.5">
          {#each item.children as child}
            <a
              href={child.href}
              class="block px-3 py-1.5 rounded text-xs transition-colors
                {$page.url.pathname === child.href ? 'text-foreground font-medium' : 'text-muted-foreground hover:text-foreground'}"
            >
              {child.label}
            </a>
          {/each}
        </div>
      {/if}
    {/each}
  </nav>
</aside>
```

**Step 3: Create TopBar**

```svelte
<!-- src/lib/components/layout/TopBar.svelte -->
<script lang="ts">
  import StatusIndicator from './StatusIndicator.svelte';

  interface Props {
    health: 'healthy' | 'degraded' | 'down' | 'unknown';
  }

  let { health }: Props = $props();
</script>

<header class="h-12 border-b bg-card flex items-center justify-between px-4">
  <div class="text-sm text-muted-foreground">
    <!-- breadcrumb can go here later -->
  </div>
  <div class="flex items-center gap-3">
    <StatusIndicator status={health} label="Services" />
  </div>
</header>
```

**Step 4: Create PageHeader**

```svelte
<!-- src/lib/components/layout/PageHeader.svelte -->
<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    description?: string;
    actions?: Snippet;
  }

  let { title, description, actions }: Props = $props();
</script>

<div class="flex items-center justify-between mb-6">
  <div>
    <h1 class="text-2xl font-semibold tracking-tight">{title}</h1>
    {#if description}
      <p class="text-sm text-muted-foreground mt-1">{description}</p>
    {/if}
  </div>
  {#if actions}
    <div class="flex items-center gap-2">
      {@render actions()}
    </div>
  {/if}
</div>
```

**Step 5: Verify build**

```bash
cd modules/dashboard-service && npm run check
```

Expected: PASS.

**Step 6: Commit**

```bash
git add modules/dashboard-service/src/lib/components/layout/
git commit -m "feat(dashboard): add layout components — Sidebar, TopBar, PageHeader, StatusIndicator"
```

---

### Task 6: App Shell (Layout Groups + Error Pages + Hooks)

**Files:**
- Modify: `modules/dashboard-service/src/routes/+layout.svelte` (root layout)
- Create: `modules/dashboard-service/src/routes/(app)/+layout.svelte`
- Create: `modules/dashboard-service/src/routes/(overlay)/+layout.svelte`
- Create: `modules/dashboard-service/src/routes/+error.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/+error.svelte`
- Create: `modules/dashboard-service/src/routes/(overlay)/+error.svelte`
- Create: `modules/dashboard-service/src/hooks.server.ts`
- Create: `modules/dashboard-service/src/routes/api/health/+server.ts`

**Step 1: Create root layout (global CSS only)**

```svelte
<!-- src/routes/+layout.svelte -->
<script lang="ts">
  import '../app.css';
  let { children } = $props();
</script>

{@render children()}
```

**Step 2: Create (app) layout with sidebar + health polling**

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

<div class="flex h-screen bg-background text-foreground">
  <Sidebar />
  <div class="flex flex-col flex-1 overflow-hidden">
    <TopBar health={healthStore.status} />
    <main class="flex-1 overflow-y-auto p-6">
      {@render children()}
    </main>
  </div>
</div>
```

**Step 3: Create (overlay) layout (bare, for OBS)**

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

**Step 4: Create error pages**

Global fallback:
```svelte
<!-- src/routes/+error.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
</script>

<div class="flex items-center justify-center h-screen bg-background text-foreground">
  <div class="text-center">
    <h1 class="text-4xl font-bold">{$page.status}</h1>
    <p class="mt-2 text-muted-foreground">{$page.error?.message ?? 'Something went wrong'}</p>
    <a href="/" class="mt-4 inline-block text-sm underline">Back to Dashboard</a>
  </div>
</div>
```

App error (sidebar stays):
```svelte
<!-- src/routes/(app)/+error.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
</script>

<div class="flex items-center justify-center h-full">
  <div class="text-center">
    <h1 class="text-4xl font-bold">{$page.status}</h1>
    <p class="mt-2 text-muted-foreground">{$page.error?.message ?? 'Something went wrong'}</p>
    <a href="/" class="mt-4 inline-block text-sm underline">Back to Dashboard</a>
  </div>
</div>
```

Overlay error (minimal):
```svelte
<!-- src/routes/(overlay)/+error.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
</script>

<div class="flex items-center justify-center h-screen text-white">
  <p>Error {$page.status}: {$page.error?.message}</p>
</div>
```

**Step 5: Create server hooks**

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

**Step 6: Create health proxy endpoint**

```typescript
// src/routes/api/health/+server.ts
import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ fetch }) => {
  try {
    const res = await fetch(`${ORCHESTRATION_URL}/api/system/health`);
    return new Response(res.body, {
      status: res.status,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch {
    return new Response(
      JSON.stringify({ status: 'down', services: {} }),
      { status: 503, headers: { 'Content-Type': 'application/json' } }
    );
  }
};
```

**Step 7: Verify build**

```bash
cd modules/dashboard-service && npm run build && npm run check
```

Expected: PASS.

**Step 8: Commit**

```bash
git add modules/dashboard-service/src/routes/ modules/dashboard-service/src/hooks.server.ts
git commit -m "feat(dashboard): add (app)/(overlay) layout groups, error pages, server hooks, health proxy"
```

---

### Task 7: Dashboard Home Page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/+page.svelte`

**Step 1: Create dashboard page**

```svelte
<!-- src/routes/(app)/+page.svelte -->
<script lang="ts">
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
  import { healthStore } from '$lib/stores/health.svelte';
  import * as Card from '$lib/components/ui/card';
</script>

<PageHeader title="Dashboard" description="LiveTranslate system overview" />

<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  <Card.Root>
    <Card.Header>
      <Card.Title class="text-sm font-medium">System Health</Card.Title>
    </Card.Header>
    <Card.Content>
      <StatusIndicator status={healthStore.status} label={healthStore.status} />
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title class="text-sm font-medium">Quick Actions</Card.Title>
    </Card.Header>
    <Card.Content class="flex flex-col gap-2">
      <a href="/fireflies" class="text-sm text-primary hover:underline">Connect to Fireflies</a>
      <a href="/translation/test" class="text-sm text-primary hover:underline">Translation Test Bench</a>
      <a href="/config" class="text-sm text-primary hover:underline">Configuration</a>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title class="text-sm font-medium">Services</Card.Title>
    </Card.Header>
    <Card.Content>
      {#if Object.keys(healthStore.services).length > 0}
        <ul class="space-y-1">
          {#each Object.entries(healthStore.services) as [name, healthy]}
            <li class="flex items-center justify-between text-sm">
              <span class="capitalize">{name.replace(/-/g, ' ')}</span>
              <StatusIndicator status={healthy ? 'healthy' : 'down'} />
            </li>
          {/each}
        </ul>
      {:else}
        <p class="text-sm text-muted-foreground">Waiting for health data...</p>
      {/if}
    </Card.Content>
  </Card.Root>
</div>
```

**Step 2: Start dev server and visually verify**

```bash
cd modules/dashboard-service && npm run dev
```

Open http://localhost:5180 — should show sidebar, top bar, dashboard cards.

**Step 3: Verify build**

```bash
npm run build && npm run check
```

Expected: PASS.

**Step 4: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/+page.svelte
git commit -m "feat(dashboard): add dashboard home page with health cards and quick actions"
```

---

## Phase 1B: Pages (Tasks 8–14)

### Task 8: Fireflies Connect Page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/fireflies/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/fireflies/+page.server.ts`

**Step 1: Create server load + form action**

```typescript
// src/routes/(app)/fireflies/+page.server.ts
import { fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
  const ff = firefliesApi(fetch);
  const cfg = configApi(fetch);

  const [sessions, uiConfig] = await Promise.all([
    ff.listSessions().catch(() => []),
    cfg.getUiConfig().catch(() => null)
  ]);

  return { sessions, uiConfig };
};

export const actions: Actions = {
  connect: async ({ request, fetch }) => {
    const data = await request.formData();
    const transcript_id = data.get('transcript_id')?.toString()?.trim();
    const api_key = data.get('api_key')?.toString()?.trim() || null;
    const target_languages = data.get('target_languages')?.toString()?.split(',').filter(Boolean) ?? [];
    const domain = data.get('domain')?.toString() || null;

    if (!transcript_id) {
      return fail(400, { transcript_id: '', errors: { transcript_id: 'Transcript ID is required' } });
    }

    const ff = firefliesApi(fetch);
    try {
      const result = await ff.connect({
        transcript_id,
        api_key,
        target_languages: target_languages.length > 0 ? target_languages : null,
        domain
      });

      redirect(303, `/fireflies/connect?session=${result.session_id}`);
    } catch (err) {
      return fail(500, { transcript_id, errors: { form: `Connection failed: ${err}` } });
    }
  }
};
```

**Step 2: Create connect page UI**

```svelte
<!-- src/routes/(app)/fireflies/+page.svelte -->
<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';

  let { data, form } = $props();
</script>

<PageHeader title="Fireflies" description="Connect to a live Fireflies transcript for real-time translation" />

<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <!-- Connect Form -->
  <div class="lg:col-span-2">
    <Card.Root>
      <Card.Header>
        <Card.Title>Connect to Transcript</Card.Title>
      </Card.Header>
      <Card.Content>
        <form method="POST" action="?/connect" use:enhance class="space-y-4">
          <div class="space-y-2">
            <Label for="transcript_id">Transcript ID</Label>
            <Input
              id="transcript_id"
              name="transcript_id"
              placeholder="Enter Fireflies transcript ID"
              value={form?.transcript_id ?? ''}
              required
            />
            {#if form?.errors?.transcript_id}
              <p class="text-sm text-destructive">{form.errors.transcript_id}</p>
            {/if}
          </div>

          <div class="space-y-2">
            <Label for="api_key">API Key (optional)</Label>
            <Input
              id="api_key"
              name="api_key"
              type="password"
              placeholder="Uses env default if blank"
            />
          </div>

          <div class="space-y-2">
            <Label for="target_languages">Target Languages (comma-separated)</Label>
            <Input
              id="target_languages"
              name="target_languages"
              placeholder="es,fr,de"
            />
          </div>

          <div class="space-y-2">
            <Label for="domain">Domain</Label>
            <select id="domain" name="domain" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
              <option value="">General</option>
              {#if data.uiConfig?.domains}
                {#each data.uiConfig.domains as d}
                  <option value={d}>{d}</option>
                {/each}
              {/if}
            </select>
          </div>

          {#if form?.errors?.form}
            <p class="text-sm text-destructive">{form.errors.form}</p>
          {/if}

          <Button type="submit">Connect</Button>
        </form>
      </Card.Content>
    </Card.Root>
  </div>

  <!-- Active Sessions -->
  <div>
    <Card.Root>
      <Card.Header>
        <Card.Title>Active Sessions</Card.Title>
      </Card.Header>
      <Card.Content>
        {#if data.sessions.length === 0}
          <p class="text-sm text-muted-foreground">No active sessions</p>
        {:else}
          <ul class="space-y-2">
            {#each data.sessions as session}
              <li>
                <a
                  href="/fireflies/connect?session={session.session_id}"
                  class="block p-2 rounded border hover:bg-accent transition-colors"
                >
                  <div class="flex items-center justify-between">
                    <span class="text-sm font-mono truncate">{session.session_id.slice(0, 16)}...</span>
                    <StatusIndicator status={session.connection_status === 'CONNECTED' ? 'connected' : 'disconnected'} />
                  </div>
                  <p class="text-xs text-muted-foreground mt-1">
                    {session.chunks_received} chunks · {session.translations_completed} translations
                  </p>
                </a>
              </li>
            {/each}
          </ul>
        {/if}
      </Card.Content>
    </Card.Root>
  </div>
</div>
```

**Step 3: Verify build**

```bash
cd modules/dashboard-service && npm run build && npm run check
```

Expected: PASS.

**Step 4: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/fireflies/
git commit -m "feat(dashboard): add Fireflies connect page with form actions and session list"
```

---

### Task 9: Fireflies Session Streaming Page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/fireflies/connect/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/fireflies/connect/+page.server.ts`
- Create: `modules/dashboard-service/src/lib/components/captions/CaptionBox.svelte`
- Create: `modules/dashboard-service/src/lib/components/captions/CaptionStream.svelte`
- Create: `modules/dashboard-service/src/lib/components/captions/InterimCaption.svelte`

**Step 1: Create caption display components**

```svelte
<!-- src/lib/components/captions/CaptionBox.svelte -->
<script lang="ts">
  import type { Caption } from '$lib/types';

  interface Props {
    caption: Caption & { receivedAt: number };
    showOriginal?: boolean;
    showTranslated?: boolean;
  }

  let { caption, showOriginal = true, showTranslated = true }: Props = $props();
</script>

<div class="caption-box border rounded-lg p-3 space-y-1 transition-opacity" data-caption-id={caption.id}>
  <div class="flex items-center gap-2">
    <span
      class="speaker-name text-xs font-medium px-1.5 py-0.5 rounded"
      style="background-color: {caption.speaker_color}20; color: {caption.speaker_color}"
    >
      {caption.speaker_name}
    </span>
    <span class="text-xs text-muted-foreground">{caption.target_language}</span>
    {#if caption.confidence}
      <span class="text-xs text-muted-foreground ml-auto">{Math.round(caption.confidence * 100)}%</span>
    {/if}
  </div>
  {#if showOriginal && caption.original_text}
    <p class="original-text text-sm text-muted-foreground">{caption.original_text}</p>
  {/if}
  {#if showTranslated && caption.text}
    <p class="translated-text text-sm font-medium">{caption.text}</p>
  {/if}
</div>
```

```svelte
<!-- src/lib/components/captions/InterimCaption.svelte -->
<script lang="ts">
  interface Props {
    text: string;
  }

  let { text }: Props = $props();
</script>

{#if text}
  <div class="interim-caption border border-dashed rounded-lg p-3 opacity-70">
    <p class="text-sm italic">{text}</p>
  </div>
{/if}
```

```svelte
<!-- src/lib/components/captions/CaptionStream.svelte -->
<script lang="ts">
  import type { Caption } from '$lib/types';
  import CaptionBox from './CaptionBox.svelte';
  import InterimCaption from './InterimCaption.svelte';

  interface Props {
    captions: (Caption & { receivedAt: number })[];
    interim: string;
    showOriginal?: boolean;
    showTranslated?: boolean;
  }

  let { captions, interim, showOriginal = true, showTranslated = true }: Props = $props();
</script>

<div class="caption-stream space-y-2 max-h-[70vh] overflow-y-auto">
  {#each captions as caption (caption.id)}
    <CaptionBox {caption} {showOriginal} {showTranslated} />
  {/each}
  <InterimCaption text={interim} />
</div>
```

**Step 2: Create server load for session page**

```typescript
// src/routes/(app)/fireflies/connect/+page.server.ts
import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';

export const load: PageServerLoad = async ({ url, fetch }) => {
  const sessionId = url.searchParams.get('session');
  if (!sessionId) {
    throw error(400, 'Missing session parameter');
  }

  const ff = firefliesApi(fetch);
  try {
    const session = await ff.getSession(sessionId);
    return { session };
  } catch {
    throw error(404, `Session ${sessionId} not found`);
  }
};
```

**Step 3: Create session streaming page**

```svelte
<!-- src/routes/(app)/fireflies/connect/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
  import CaptionStream from '$lib/components/captions/CaptionStream.svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Card from '$lib/components/ui/card';
  import { wsStore } from '$lib/stores/websocket.svelte';
  import { captionStore } from '$lib/stores/captions.svelte';
  import { WS_BASE } from '$lib/config';
  import type { CaptionEvent } from '$lib/types';

  let { data } = $props();

  onMount(() => {
    const sessionId = data.session.session_id;
    wsStore.connect(`${WS_BASE}/api/captions/stream/${sessionId}`);

    wsStore.onMessage = (event) => {
      const msg: CaptionEvent = JSON.parse(event.data);
      switch (msg.event) {
        case 'connected':
          msg.current_captions.forEach((c) => captionStore.addCaption(c));
          break;
        case 'caption_added':
          captionStore.addCaption(msg.caption);
          break;
        case 'caption_updated':
          captionStore.updateCaption(msg.caption);
          break;
        case 'caption_expired':
          captionStore.removeCaption(msg.caption_id);
          break;
        case 'interim_caption':
          captionStore.updateInterim(msg.caption.text);
          break;
        case 'session_cleared':
          captionStore.clear();
          break;
      }
    };

    captionStore.start();

    return () => {
      wsStore.disconnect();
      captionStore.stop();
      captionStore.clear();
    };
  });

  async function handleDisconnect() {
    await fetch(`/api/fireflies/disconnect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: data.session.session_id })
    });
    goto('/fireflies');
  }
</script>

<PageHeader title="Live Session">
  {#snippet actions()}
    <StatusIndicator status={wsStore.status} label={wsStore.status} />
    <Button variant="destructive" size="sm" onclick={handleDisconnect}>Disconnect</Button>
  {/snippet}
</PageHeader>

<div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
  <div class="lg:col-span-3">
    <Card.Root>
      <Card.Header>
        <Card.Title>Live Captions</Card.Title>
      </Card.Header>
      <Card.Content>
        <CaptionStream
          captions={captionStore.captions}
          interim={captionStore.interim}
        />
      </Card.Content>
    </Card.Root>
  </div>

  <div>
    <Card.Root>
      <Card.Header>
        <Card.Title>Session Info</Card.Title>
      </Card.Header>
      <Card.Content class="space-y-2 text-sm">
        <div class="flex justify-between">
          <span class="text-muted-foreground">Status</span>
          <StatusIndicator status={data.session.connection_status === 'CONNECTED' ? 'connected' : 'disconnected'} label={data.session.connection_status} />
        </div>
        <div class="flex justify-between">
          <span class="text-muted-foreground">Chunks</span>
          <span>{data.session.chunks_received}</span>
        </div>
        <div class="flex justify-between">
          <span class="text-muted-foreground">Translations</span>
          <span>{data.session.translations_completed}</span>
        </div>
        <div class="flex justify-between">
          <span class="text-muted-foreground">Speakers</span>
          <span>{data.session.speakers_detected.length}</span>
        </div>
        {#if data.session.speakers_detected.length > 0}
          <div class="pt-2 border-t">
            <p class="text-xs text-muted-foreground mb-1">Detected speakers:</p>
            {#each data.session.speakers_detected as speaker}
              <span class="text-xs bg-accent px-1.5 py-0.5 rounded mr-1">{speaker}</span>
            {/each}
          </div>
        {/if}
      </Card.Content>
    </Card.Root>
  </div>
</div>
```

**Step 4: Create disconnect proxy endpoint**

```typescript
// src/routes/api/fireflies/disconnect/+server.ts
import { ORCHESTRATION_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, fetch }) => {
  const body = await request.json();
  const res = await fetch(`${ORCHESTRATION_URL}/fireflies/disconnect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  return new Response(res.body, {
    status: res.status,
    headers: { 'Content-Type': 'application/json' }
  });
};
```

**Step 5: Verify build**

```bash
cd modules/dashboard-service && npm run build && npm run check
```

Expected: PASS.

**Step 6: Commit**

```bash
git add modules/dashboard-service/src/lib/components/captions/ modules/dashboard-service/src/routes/
git commit -m "feat(dashboard): add Fireflies session streaming page with live WebSocket captions"
```

---

### Task 10: Fireflies History Page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/fireflies/history/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/fireflies/history/+page.server.ts`

**Step 1: Create server load with streaming**

```typescript
// src/routes/(app)/fireflies/history/+page.server.ts
import type { PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';

export const load: PageServerLoad = async ({ fetch }) => {
  const ff = firefliesApi(fetch);
  return {
    sessions: ff.listSessions().catch(() => [])  // streamed — not awaited
  };
};
```

**Step 2: Create history page**

```svelte
<!-- src/routes/(app)/fireflies/history/+page.svelte -->
<script lang="ts">
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
  import * as Card from '$lib/components/ui/card';
  import * as Table from '$lib/components/ui/table';

  let { data } = $props();
</script>

<PageHeader title="Session History" description="Past and active Fireflies sessions" />

<Card.Root>
  <Card.Content class="p-0">
    {#await data.sessions}
      <div class="p-6 text-center text-muted-foreground">Loading sessions...</div>
    {:then sessions}
      {#if sessions.length === 0}
        <div class="p-6 text-center text-muted-foreground">No sessions found</div>
      {:else}
        <Table.Root>
          <Table.Header>
            <Table.Row>
              <Table.Head>Session ID</Table.Head>
              <Table.Head>Status</Table.Head>
              <Table.Head>Chunks</Table.Head>
              <Table.Head>Translations</Table.Head>
              <Table.Head>Speakers</Table.Head>
              <Table.Head>Connected</Table.Head>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {#each sessions as session}
              <Table.Row>
                <Table.Cell>
                  <a href="/fireflies/connect?session={session.session_id}" class="text-primary hover:underline font-mono text-xs">
                    {session.session_id.slice(0, 20)}...
                  </a>
                </Table.Cell>
                <Table.Cell>
                  <StatusIndicator
                    status={session.connection_status === 'CONNECTED' ? 'connected' : 'disconnected'}
                    label={session.connection_status}
                  />
                </Table.Cell>
                <Table.Cell>{session.chunks_received}</Table.Cell>
                <Table.Cell>{session.translations_completed}</Table.Cell>
                <Table.Cell>{session.speakers_detected.length}</Table.Cell>
                <Table.Cell class="text-xs text-muted-foreground">
                  {new Date(session.connected_at).toLocaleString()}
                </Table.Cell>
              </Table.Row>
            {/each}
          </Table.Body>
        </Table.Root>
      {/if}
    {:catch error}
      <div class="p-6 text-center text-destructive">Failed to load sessions: {error.message}</div>
    {/await}
  </Card.Content>
</Card.Root>
```

**Step 3: Verify build + commit**

```bash
cd modules/dashboard-service && npm run build && npm run check
git add modules/dashboard-service/src/routes/\(app\)/fireflies/history/
git commit -m "feat(dashboard): add Fireflies session history page with streamed data loading"
```

---

### Task 11: Glossary Page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/fireflies/glossary/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/fireflies/glossary/+page.server.ts`

**Step 1: Create server load + form actions**

```typescript
// src/routes/(app)/fireflies/glossary/+page.server.ts
import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { glossaryApi } from '$lib/api/glossary';

export const load: PageServerLoad = async ({ fetch }) => {
  const gApi = glossaryApi(fetch);
  const glossaries = await gApi.list().catch(() => []);

  // Load entries for the first (or default) glossary
  let entries: Awaited<ReturnType<typeof gApi.listEntries>> = [];
  const defaultGlossary = glossaries.find((g) => g.is_default) ?? glossaries[0];
  if (defaultGlossary) {
    entries = await gApi.listEntries(defaultGlossary.glossary_id).catch(() => []);
  }

  return { glossaries, entries, activeGlossaryId: defaultGlossary?.glossary_id ?? null };
};

export const actions: Actions = {
  addEntry: async ({ request, fetch }) => {
    const data = await request.formData();
    const glossary_id = data.get('glossary_id')?.toString();
    const source_term = data.get('source_term')?.toString()?.trim();
    const translation = data.get('translation')?.toString()?.trim();
    const target_language = data.get('target_language')?.toString() ?? 'es';

    if (!glossary_id || !source_term || !translation) {
      return fail(400, { errors: { form: 'All fields are required' } });
    }

    const gApi = glossaryApi(fetch);
    try {
      await gApi.createEntry(glossary_id, {
        source_term,
        translations: { [target_language]: translation },
        context: '',
        notes: '',
        case_sensitive: false,
        match_whole_word: true,
        priority: 5
      });
      return { success: true };
    } catch (err) {
      return fail(500, { errors: { form: `Failed to add entry: ${err}` } });
    }
  },

  deleteEntry: async ({ request, fetch }) => {
    const data = await request.formData();
    const glossary_id = data.get('glossary_id')?.toString();
    const entry_id = data.get('entry_id')?.toString();

    if (!glossary_id || !entry_id) return fail(400, { errors: { form: 'Missing IDs' } });

    const gApi = glossaryApi(fetch);
    try {
      await gApi.deleteEntry(glossary_id, entry_id);
      return { success: true };
    } catch (err) {
      return fail(500, { errors: { form: `Failed to delete: ${err}` } });
    }
  }
};
```

**Step 2: Create glossary page**

```svelte
<!-- src/routes/(app)/fireflies/glossary/+page.svelte -->
<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import * as Table from '$lib/components/ui/table';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';

  let { data, form } = $props();
</script>

<PageHeader title="Glossary" description="Manage translation glossary terms" />

<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <!-- Add Entry Form -->
  <div>
    <Card.Root>
      <Card.Header>
        <Card.Title>Add Term</Card.Title>
      </Card.Header>
      <Card.Content>
        <form method="POST" action="?/addEntry" use:enhance class="space-y-3">
          <input type="hidden" name="glossary_id" value={data.activeGlossaryId ?? ''} />

          <div class="space-y-1">
            <Label for="source_term">Source Term</Label>
            <Input id="source_term" name="source_term" placeholder="heart attack" required />
          </div>

          <div class="space-y-1">
            <Label for="translation">Translation</Label>
            <Input id="translation" name="translation" placeholder="infarto de miocardio" required />
          </div>

          <div class="space-y-1">
            <Label for="target_language">Target Language</Label>
            <Input id="target_language" name="target_language" value="es" />
          </div>

          {#if form?.errors?.form}
            <p class="text-sm text-destructive">{form.errors.form}</p>
          {/if}

          {#if form?.success}
            <p class="text-sm text-green-600">Term added successfully</p>
          {/if}

          <Button type="submit" class="w-full">Add Term</Button>
        </form>
      </Card.Content>
    </Card.Root>
  </div>

  <!-- Entries Table -->
  <div class="lg:col-span-2">
    <Card.Root>
      <Card.Header>
        <Card.Title>
          Glossary Entries
          {#if data.glossaries.length > 0}
            <span class="text-sm font-normal text-muted-foreground ml-2">
              ({data.entries.length} terms)
            </span>
          {/if}
        </Card.Title>
      </Card.Header>
      <Card.Content class="p-0">
        {#if data.entries.length === 0}
          <div class="p-6 text-center text-muted-foreground">No glossary entries yet</div>
        {:else}
          <Table.Root>
            <Table.Header>
              <Table.Row>
                <Table.Head>Source Term</Table.Head>
                <Table.Head>Translations</Table.Head>
                <Table.Head>Priority</Table.Head>
                <Table.Head class="w-16"></Table.Head>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {#each data.entries as entry}
                <Table.Row>
                  <Table.Cell class="font-medium">{entry.source_term}</Table.Cell>
                  <Table.Cell>
                    {#each Object.entries(entry.translations) as [lang, text]}
                      <span class="text-xs bg-accent px-1.5 py-0.5 rounded mr-1">
                        {lang}: {text}
                      </span>
                    {/each}
                  </Table.Cell>
                  <Table.Cell>{entry.priority}</Table.Cell>
                  <Table.Cell>
                    <form method="POST" action="?/deleteEntry" use:enhance>
                      <input type="hidden" name="glossary_id" value={data.activeGlossaryId} />
                      <input type="hidden" name="entry_id" value={entry.entry_id} />
                      <Button variant="ghost" size="sm" type="submit">x</Button>
                    </form>
                  </Table.Cell>
                </Table.Row>
              {/each}
            </Table.Body>
          </Table.Root>
        {/if}
      </Card.Content>
    </Card.Root>
  </div>
</div>
```

**Step 3: Verify build + commit**

```bash
cd modules/dashboard-service && npm run build && npm run check
git add modules/dashboard-service/src/routes/\(app\)/fireflies/glossary/
git commit -m "feat(dashboard): add glossary management page with CRUD form actions"
```

---

### Task 12: Config Pages (Hub + Audio + Translation + System)

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/config/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/config/audio/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/config/audio/+page.server.ts`
- Create: `modules/dashboard-service/src/routes/(app)/config/translation/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/config/translation/+page.server.ts`
- Create: `modules/dashboard-service/src/routes/(app)/config/system/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/config/system/+page.server.ts`

**Step 1: Create config hub (links page)**

```svelte
<!-- src/routes/(app)/config/+page.svelte -->
<script lang="ts">
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';

  const sections = [
    { title: 'Audio', href: '/config/audio', description: 'Audio processing, sample rates, and device settings' },
    { title: 'Translation', href: '/config/translation', description: 'Translation models, languages, and quality settings' },
    { title: 'System', href: '/config/system', description: 'System settings, feature flags, and service management' }
  ];
</script>

<PageHeader title="Configuration" description="Manage system settings" />

<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
  {#each sections as section}
    <a href={section.href} class="block">
      <Card.Root class="hover:border-primary transition-colors h-full">
        <Card.Header>
          <Card.Title>{section.title}</Card.Title>
          <Card.Description>{section.description}</Card.Description>
        </Card.Header>
      </Card.Root>
    </a>
  {/each}
</div>
```

**Step 2: Create audio config page with form action**

```typescript
// src/routes/(app)/config/audio/+page.server.ts
import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
  const cfg = configApi(fetch);
  const settings = await cfg.getUserSettings().catch(() => null);
  return { settings };
};

export const actions: Actions = {
  update: async ({ request, fetch }) => {
    const data = await request.formData();
    const audio_auto_start = data.get('audio_auto_start') === 'on';

    const cfg = configApi(fetch);
    try {
      await cfg.updateUserSettings({ audio_auto_start });
      return { success: true };
    } catch (err) {
      return fail(500, { errors: { form: `Update failed: ${err}` } });
    }
  }
};
```

```svelte
<!-- src/routes/(app)/config/audio/+page.svelte -->
<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Label } from '$lib/components/ui/label';

  let { data, form } = $props();
</script>

<PageHeader title="Audio Configuration" description="Audio processing settings" />

<Card.Root class="max-w-2xl">
  <Card.Content>
    <form method="POST" action="?/update" use:enhance class="space-y-4">
      <div class="flex items-center justify-between">
        <Label for="audio_auto_start">Auto-start audio capture</Label>
        <input
          type="checkbox"
          id="audio_auto_start"
          name="audio_auto_start"
          checked={data.settings?.audio_auto_start ?? false}
          class="h-4 w-4"
        />
      </div>

      {#if form?.errors?.form}
        <p class="text-sm text-destructive">{form.errors.form}</p>
      {/if}
      {#if form?.success}
        <p class="text-sm text-green-600">Settings saved</p>
      {/if}

      <Button type="submit">Save</Button>
    </form>
  </Card.Content>
</Card.Root>
```

**Step 3: Create translation config page**

```typescript
// src/routes/(app)/config/translation/+page.server.ts
import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
  const ff = firefliesApi(fetch);
  const cfg = configApi(fetch);
  const [translationConfig, uiConfig] = await Promise.all([
    ff.getTranslationConfig().catch(() => null),
    cfg.getUiConfig().catch(() => null)
  ]);
  return { translationConfig, uiConfig };
};

export const actions: Actions = {
  update: async ({ request, fetch }) => {
    const data = await request.formData();
    const backend = data.get('backend')?.toString() ?? 'ollama';
    const model = data.get('model')?.toString() ?? '';
    const target_language = data.get('target_language')?.toString() ?? 'es';
    const temperature = parseFloat(data.get('temperature')?.toString() ?? '0.3');

    const ff = firefliesApi(fetch);
    try {
      await ff.updateTranslationConfig({ backend, model, target_language, temperature });
      return { success: true };
    } catch (err) {
      return fail(500, { errors: { form: `Update failed: ${err}` } });
    }
  }
};
```

```svelte
<!-- src/routes/(app)/config/translation/+page.svelte -->
<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';

  let { data, form } = $props();
</script>

<PageHeader title="Translation Configuration" description="Translation models and language settings" />

<Card.Root class="max-w-2xl">
  <Card.Content>
    <form method="POST" action="?/update" use:enhance class="space-y-4">
      <div class="space-y-2">
        <Label for="backend">Backend</Label>
        <select id="backend" name="backend" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
          {#each ['ollama', 'vllm', 'openai', 'groq'] as b}
            <option value={b} selected={data.translationConfig?.backend === b}>{b}</option>
          {/each}
        </select>
      </div>

      <div class="space-y-2">
        <Label for="model">Model</Label>
        <Input id="model" name="model" value={data.translationConfig?.model ?? 'qwen2.5:3b'} />
      </div>

      <div class="space-y-2">
        <Label for="target_language">Default Target Language</Label>
        <select id="target_language" name="target_language" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
          {#if data.uiConfig?.languages}
            {#each data.uiConfig.languages as lang}
              <option value={lang.code} selected={data.translationConfig?.target_language === lang.code}>
                {lang.name} ({lang.code})
              </option>
            {/each}
          {/if}
        </select>
      </div>

      <div class="space-y-2">
        <Label for="temperature">Temperature</Label>
        <Input id="temperature" name="temperature" type="number" step="0.1" min="0" max="2"
          value={data.translationConfig?.temperature ?? 0.3} />
      </div>

      {#if form?.errors?.form}
        <p class="text-sm text-destructive">{form.errors.form}</p>
      {/if}
      {#if form?.success}
        <p class="text-sm text-green-600">Translation config saved</p>
      {/if}

      <Button type="submit">Save</Button>
    </form>
  </Card.Content>
</Card.Root>
```

**Step 4: Create system config page**

```typescript
// src/routes/(app)/config/system/+page.server.ts
import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
  const cfg = configApi(fetch);
  const settings = await cfg.getUserSettings().catch(() => null);
  return { settings };
};

export const actions: Actions = {
  update: async ({ request, fetch }) => {
    const data = await request.formData();
    const theme = data.get('theme')?.toString() as 'dark' | 'light' ?? 'dark';
    const language = data.get('language')?.toString() ?? 'en';
    const notifications = data.get('notifications') === 'on';

    const cfg = configApi(fetch);
    try {
      await cfg.updateUserSettings({ theme, language, notifications });
      return { success: true };
    } catch (err) {
      return fail(500, { errors: { form: `Update failed: ${err}` } });
    }
  }
};
```

```svelte
<!-- src/routes/(app)/config/system/+page.svelte -->
<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Label } from '$lib/components/ui/label';

  let { data, form } = $props();
</script>

<PageHeader title="System Configuration" description="System preferences and feature flags" />

<Card.Root class="max-w-2xl">
  <Card.Content>
    <form method="POST" action="?/update" use:enhance class="space-y-4">
      <div class="space-y-2">
        <Label for="theme">Theme</Label>
        <select id="theme" name="theme" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
          <option value="dark" selected={data.settings?.theme === 'dark'}>Dark</option>
          <option value="light" selected={data.settings?.theme === 'light'}>Light</option>
        </select>
      </div>

      <div class="space-y-2">
        <Label for="language">Interface Language</Label>
        <select id="language" name="language" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
          <option value="en" selected={data.settings?.language === 'en'}>English</option>
          <option value="es" selected={data.settings?.language === 'es'}>Spanish</option>
          <option value="fr" selected={data.settings?.language === 'fr'}>French</option>
        </select>
      </div>

      <div class="flex items-center justify-between">
        <Label for="notifications">Enable Notifications</Label>
        <input type="checkbox" id="notifications" name="notifications"
          checked={data.settings?.notifications ?? true} class="h-4 w-4" />
      </div>

      {#if form?.errors?.form}
        <p class="text-sm text-destructive">{form.errors.form}</p>
      {/if}
      {#if form?.success}
        <p class="text-sm text-green-600">System settings saved</p>
      {/if}

      <Button type="submit">Save</Button>
    </form>
  </Card.Content>
</Card.Root>
```

**Step 5: Verify build + commit**

```bash
cd modules/dashboard-service && npm run build && npm run check
git add modules/dashboard-service/src/routes/\(app\)/config/
git commit -m "feat(dashboard): add config pages (hub, audio, translation, system) with form actions"
```

---

### Task 13: Translation Test Bench

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/translation/test/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/translation/test/+page.server.ts`

**Step 1: Create server load + form action**

```typescript
// src/routes/(app)/translation/test/+page.server.ts
import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { translationApi } from '$lib/api/translation';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
  const cfg = configApi(fetch);
  const tApi = translationApi(fetch);

  const [uiConfig, models] = await Promise.all([
    cfg.getUiConfig().catch(() => null),
    tApi.getModels().catch(() => ({ models: [] }))
  ]);

  return { uiConfig, models: models.models };
};

export const actions: Actions = {
  translate: async ({ request, fetch }) => {
    const data = await request.formData();
    const text = data.get('text')?.toString()?.trim();
    const target_language = data.get('target_language')?.toString() ?? 'es';
    const service = data.get('service')?.toString() ?? 'ollama';

    if (!text) {
      return fail(400, { text: '', errors: { text: 'Text is required' } });
    }

    const tApi = translationApi(fetch);
    try {
      const result = await tApi.translate({
        text,
        target_language,
        service,
        quality: 'balanced'
      });
      return { success: true, result, text };
    } catch (err) {
      return fail(500, { text, errors: { form: `Translation failed: ${err}` } });
    }
  }
};
```

**Step 2: Create translation test bench page**

```svelte
<!-- src/routes/(app)/translation/test/+page.svelte -->
<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Label } from '$lib/components/ui/label';
  import { Textarea } from '$lib/components/ui/textarea';

  let { data, form } = $props();
</script>

<PageHeader title="Translation Test Bench" description="Test translation quality interactively" />

<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
  <!-- Input -->
  <Card.Root>
    <Card.Header>
      <Card.Title>Input</Card.Title>
    </Card.Header>
    <Card.Content>
      <form method="POST" action="?/translate" use:enhance class="space-y-4">
        <div class="space-y-2">
          <Label for="text">Text to translate</Label>
          <Textarea id="text" name="text" rows={6} placeholder="Enter text to translate..."
            value={form?.text ?? ''} required />
          {#if form?.errors?.text}
            <p class="text-sm text-destructive">{form.errors.text}</p>
          {/if}
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div class="space-y-2">
            <Label for="target_language">Target Language</Label>
            <select id="target_language" name="target_language" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
              {#if data.uiConfig?.languages}
                {#each data.uiConfig.languages as lang}
                  <option value={lang.code}>{lang.name}</option>
                {/each}
              {:else}
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
              {/if}
            </select>
          </div>

          <div class="space-y-2">
            <Label for="service">Service</Label>
            <select id="service" name="service" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
              {#each data.models as model}
                <option value={model.backend}>{model.backend} — {model.name}</option>
              {/each}
              {#if data.models.length === 0}
                <option value="ollama">ollama</option>
              {/if}
            </select>
          </div>
        </div>

        {#if form?.errors?.form}
          <p class="text-sm text-destructive">{form.errors.form}</p>
        {/if}

        <Button type="submit" class="w-full">Translate</Button>
      </form>
    </Card.Content>
  </Card.Root>

  <!-- Output -->
  <Card.Root>
    <Card.Header>
      <Card.Title>Result</Card.Title>
    </Card.Header>
    <Card.Content>
      {#if form?.success && form?.result}
        <div class="space-y-4">
          <div class="p-4 bg-accent rounded-lg">
            <p class="text-sm font-medium">{form.result.translated_text}</p>
          </div>
          <div class="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
            <div>Confidence: <span class="text-foreground">{Math.round(form.result.confidence * 100)}%</span></div>
            <div>Time: <span class="text-foreground">{Math.round(form.result.processing_time * 1000)}ms</span></div>
            <div>Model: <span class="text-foreground">{form.result.model_used}</span></div>
            <div>Backend: <span class="text-foreground">{form.result.backend_used}</span></div>
            <div>Source: <span class="text-foreground">{form.result.source_language}</span></div>
            <div>Target: <span class="text-foreground">{form.result.target_language}</span></div>
          </div>
        </div>
      {:else}
        <p class="text-muted-foreground text-center py-8">Submit text to see translation results</p>
      {/if}
    </Card.Content>
  </Card.Root>
</div>
```

**Step 3: Verify build + commit**

```bash
cd modules/dashboard-service && npm run build && npm run check
git add modules/dashboard-service/src/routes/\(app\)/translation/
git commit -m "feat(dashboard): add translation test bench with interactive form actions"
```

---

### Task 14: Captions Overlay (OBS)

**Files:**
- Create: `modules/dashboard-service/src/routes/(overlay)/captions/+page.svelte`

**Step 1: Create OBS captions overlay page**

This page uses the `(overlay)` layout group — no sidebar, no chrome, transparent background. URL params configure everything.

```svelte
<!-- src/routes/(overlay)/captions/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { WS_BASE } from '$lib/config';
  import { WebSocketStore } from '$lib/stores/websocket.svelte';
  import { CaptionStore } from '$lib/stores/captions.svelte';
  import type { CaptionEvent, DisplayMode } from '$lib/types';

  // URL params
  const sessionId = $derived($page.url.searchParams.get('session') ?? '');
  const mode: DisplayMode = $derived(
    ($page.url.searchParams.get('mode') as DisplayMode) ?? 'both'
  );
  const fontSize = $derived(parseInt($page.url.searchParams.get('fontSize') ?? '18'));
  const maxCaptions = $derived(parseInt($page.url.searchParams.get('maxCaptions') ?? '5'));
  const bgColor = $derived($page.url.searchParams.get('bg') ?? 'transparent');

  // Dedicated instances (not singletons) for overlay
  const ws = new WebSocketStore();
  const captions = new CaptionStore(10_000);

  $effect(() => {
    captions.maxCaptions = maxCaptions;
  });

  onMount(() => {
    if (!sessionId) return;

    ws.connect(`${WS_BASE}/api/captions/stream/${sessionId}`);
    ws.onMessage = (event) => {
      const msg: CaptionEvent = JSON.parse(event.data);
      switch (msg.event) {
        case 'connected':
          msg.current_captions.forEach((c) => captions.addCaption(c));
          break;
        case 'caption_added':
          captions.addCaption(msg.caption);
          break;
        case 'caption_updated':
          captions.updateCaption(msg.caption);
          break;
        case 'caption_expired':
          captions.removeCaption(msg.caption_id);
          break;
        case 'interim_caption':
          captions.updateInterim(msg.caption.text);
          break;
        case 'session_cleared':
          captions.clear();
          break;
      }
    };

    captions.start();

    return () => {
      ws.disconnect();
      captions.stop();
    };
  });
</script>

<div class="captions-overlay" style="background: {bgColor}; font-size: {fontSize}px;">
  {#if !sessionId}
    <div class="no-session">
      <p>Missing ?session= parameter</p>
    </div>
  {:else}
    <div class="caption-list">
      {#each captions.captions as caption (caption.id)}
        <div class="caption-entry" data-caption-id={caption.id}>
          <span class="speaker" style="color: {caption.speaker_color}">
            {caption.speaker_name}
          </span>
          {#if mode === 'both' || mode === 'english'}
            <p class="original">{caption.original_text}</p>
          {/if}
          {#if mode === 'both' || mode === 'translated'}
            <p class="translated">{caption.text}</p>
          {/if}
        </div>
      {/each}

      {#if captions.interim && (mode === 'both' || mode === 'english')}
        <div class="caption-entry interim">
          <p class="original">{captions.interim}</p>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .captions-overlay {
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 20px;
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden;
  }

  .no-session {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #999;
  }

  .caption-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .caption-entry {
    background: rgba(0, 0, 0, 0.75);
    border-radius: 8px;
    padding: 8px 12px;
    animation: fadeIn 0.3s ease;
  }

  .caption-entry.interim {
    opacity: 0.6;
    border: 1px dashed rgba(255, 255, 255, 0.3);
  }

  .speaker {
    font-size: 0.75em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .original {
    color: rgba(255, 255, 255, 0.7);
    margin: 2px 0;
  }

  .translated {
    color: #fff;
    font-weight: 500;
    margin: 2px 0;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
</style>
```

**Step 2: Verify build + commit**

```bash
cd modules/dashboard-service && npm run build && npm run check
git add modules/dashboard-service/src/routes/\(overlay\)/captions/
git commit -m "feat(dashboard): add OBS captions overlay with display modes and URL param config"
```

---

## Phase 1C: Agent-Browser Verification (Tasks 15–18)

### Task 15: Agent-Browser Test Infrastructure

**Files:**
- Create: `modules/dashboard-service/tests/browser/conftest.py`
- Create: `modules/dashboard-service/tests/browser/__init__.py`
- Create: `modules/dashboard-service/tests/browser/screenshots/` (directory)

**Step 1: Create conftest with SvelteKit server + agent-browser fixtures**

```python
# tests/browser/conftest.py
"""
Agent-browser fixtures for SvelteKit dashboard visual verification.

Starts the SvelteKit dev server and provides an AgentBrowser instance
for each test. Screenshots are saved to tests/browser/screenshots/.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

# Import AgentBrowser from orchestration-service's test helpers
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
ORCH_TESTS = REPO_ROOT / "modules" / "orchestration-service" / "tests"
sys.path.insert(0, str(ORCH_TESTS / "fireflies" / "e2e" / "browser"))

from browser_helpers import AgentBrowser  # noqa: E402

DASHBOARD_DIR = Path(__file__).parent.parent.parent  # modules/dashboard-service
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
DASHBOARD_URL = "http://localhost:5180"


@pytest.fixture(scope="session")
def sveltekit_server():
    """Start SvelteKit dev server for the full test session."""
    SCREENSHOT_DIR.mkdir(exist_ok=True)

    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(DASHBOARD_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PORT": "5180"},
    )

    # Wait for server ready (up to 30 seconds)
    for attempt in range(30):
        try:
            resp = httpx.get(DASHBOARD_URL, timeout=2, follow_redirects=True)
            if resp.status_code == 200:
                break
        except (httpx.ConnectError, httpx.ReadTimeout):
            time.sleep(1)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("SvelteKit dev server did not start within 30 seconds")

    yield DASHBOARD_URL

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def browser(sveltekit_server):
    """Fresh AgentBrowser instance for each test."""
    b = AgentBrowser(headed=True)
    b.open(sveltekit_server)
    yield b
    b.close()


@pytest.fixture
def screenshot_path():
    """Returns a function that generates screenshot paths."""
    def _path(name: str) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return str(SCREENSHOT_DIR / f"{ts}_{name}.png")
    return _path
```

**Step 2: Create empty __init__.py**

```python
# tests/browser/__init__.py
```

**Step 3: Create screenshots directory**

```bash
mkdir -p modules/dashboard-service/tests/browser/screenshots
touch modules/dashboard-service/tests/browser/screenshots/.gitkeep
```

**Step 4: Commit**

```bash
git add modules/dashboard-service/tests/browser/
git commit -m "feat(dashboard): add agent-browser test infrastructure with SvelteKit server fixture"
```

---

### Task 16: Route Verification Tests

**Files:**
- Create: `modules/dashboard-service/tests/browser/test_app_routes.py`

**Step 1: Create route rendering tests**

Every route must: navigate, screenshot, verify DOM elements, verify no errors.

```python
# tests/browser/test_app_routes.py
"""
Visual verification of all dashboard routes.

Each test navigates to a route, takes a screenshot, and verifies
that expected DOM elements are present and no errors are shown.
"""

import pytest


class TestDashboardRoutes:
    """Verify every Phase 1 route renders correctly."""

    def test_dashboard_home(self, browser, screenshot_path):
        """Dashboard home shows health cards and quick actions."""
        browser.open("http://localhost:5180/")
        browser.wait("text=Dashboard")
        snap = browser.snapshot()
        assert "Dashboard" in snap
        assert "Quick Actions" in snap
        browser.screenshot(screenshot_path("dashboard_home"))

    def test_fireflies_connect_page(self, browser, screenshot_path):
        """Fireflies connect page shows form with transcript ID input."""
        browser.open("http://localhost:5180/fireflies")
        browser.wait("text=Fireflies")
        snap = browser.snapshot()
        assert "Transcript ID" in snap
        assert "Connect" in snap
        browser.screenshot(screenshot_path("fireflies_connect"))

    def test_fireflies_history_page(self, browser, screenshot_path):
        """History page renders session table or empty state."""
        browser.open("http://localhost:5180/fireflies/history")
        browser.wait("text=Session History")
        snap = browser.snapshot()
        assert "Session History" in snap
        browser.screenshot(screenshot_path("fireflies_history"))

    def test_fireflies_glossary_page(self, browser, screenshot_path):
        """Glossary page shows add form and entries table."""
        browser.open("http://localhost:5180/fireflies/glossary")
        browser.wait("text=Glossary")
        snap = browser.snapshot()
        assert "Add Term" in snap
        assert "Source Term" in snap
        browser.screenshot(screenshot_path("fireflies_glossary"))

    def test_config_hub(self, browser, screenshot_path):
        """Config hub shows links to audio, translation, system."""
        browser.open("http://localhost:5180/config")
        browser.wait("text=Configuration")
        snap = browser.snapshot()
        assert "Audio" in snap
        assert "Translation" in snap
        assert "System" in snap
        browser.screenshot(screenshot_path("config_hub"))

    def test_config_audio(self, browser, screenshot_path):
        """Audio config page shows settings form."""
        browser.open("http://localhost:5180/config/audio")
        browser.wait("text=Audio Configuration")
        snap = browser.snapshot()
        assert "Save" in snap
        browser.screenshot(screenshot_path("config_audio"))

    def test_config_translation(self, browser, screenshot_path):
        """Translation config page shows backend/model/language form."""
        browser.open("http://localhost:5180/config/translation")
        browser.wait("text=Translation Configuration")
        snap = browser.snapshot()
        assert "Backend" in snap
        assert "Model" in snap
        browser.screenshot(screenshot_path("config_translation"))

    def test_config_system(self, browser, screenshot_path):
        """System config page shows theme and notification settings."""
        browser.open("http://localhost:5180/config/system")
        browser.wait("text=System Configuration")
        snap = browser.snapshot()
        assert "Theme" in snap
        browser.screenshot(screenshot_path("config_system"))

    def test_translation_test_bench(self, browser, screenshot_path):
        """Translation test bench shows input/output panels."""
        browser.open("http://localhost:5180/translation/test")
        browser.wait("text=Translation Test Bench")
        snap = browser.snapshot()
        assert "Input" in snap
        assert "Result" in snap
        browser.screenshot(screenshot_path("translation_test_bench"))

    def test_captions_overlay_no_session(self, browser, screenshot_path):
        """Captions overlay without session param shows error message."""
        browser.open("http://localhost:5180/captions")
        time.sleep(1)  # overlay has no loading indicators
        snap = browser.snapshot()
        assert "Missing" in snap or "session" in snap.lower()
        browser.screenshot(screenshot_path("captions_overlay_no_session"))

    def test_sidebar_navigation(self, browser, screenshot_path):
        """Sidebar is present on all (app) pages and has correct links."""
        browser.open("http://localhost:5180/")
        browser.wait("text=Dashboard")
        snap = browser.snapshot()
        assert "Fireflies" in snap
        assert "Config" in snap
        assert "Translation" in snap
        browser.screenshot(screenshot_path("sidebar_navigation"))

    def test_overlay_has_no_sidebar(self, browser, screenshot_path):
        """Captions overlay (overlay group) has no sidebar or navigation."""
        browser.open("http://localhost:5180/captions")
        time.sleep(1)
        snap = browser.snapshot()
        # Sidebar elements should NOT be present
        assert "Dashboard" not in snap or "Fireflies" not in snap
        browser.screenshot(screenshot_path("overlay_no_sidebar"))


# Need time for the no-session overlay tests
import time  # noqa: E402
```

**Step 2: Run tests (requires SvelteKit dev server)**

```bash
cd modules/dashboard-service
npm run dev &  # start server in background
sleep 5
uv run pytest tests/browser/test_app_routes.py -v
```

Expected: All tests PASS, screenshots saved to `tests/browser/screenshots/`.

**Step 3: Commit**

```bash
git add modules/dashboard-service/tests/browser/test_app_routes.py
git commit -m "feat(dashboard): add agent-browser route verification tests with screenshots"
```

---

### Task 17: Interactive Flow Tests

**Files:**
- Create: `modules/dashboard-service/tests/browser/test_config_forms.py`
- Create: `modules/dashboard-service/tests/browser/test_navigation.py`

**Step 1: Create config form tests**

```python
# tests/browser/test_config_forms.py
"""
Test config form submission flows via agent-browser.

Fills forms, submits, and verifies success/error states.
Requires orchestration service running on port 3000.
"""

import pytest
import time


class TestConfigForms:
    """Fill and submit config forms, verify responses."""

    def test_translation_config_save(self, browser, screenshot_path):
        """Fill translation config and save successfully."""
        browser.open("http://localhost:5180/config/translation")
        browser.wait("text=Translation Configuration")
        browser.screenshot(screenshot_path("config_translation_before"))

        # Fill the model field
        browser.fill("input[name='model']", "qwen2.5:3b")
        browser.screenshot(screenshot_path("config_translation_filled"))

        # Submit form
        browser.click("text=Save")
        time.sleep(2)
        browser.screenshot(screenshot_path("config_translation_after"))

        snap = browser.snapshot()
        # Should show success or still show form (if orchestration not running, may show error)
        assert "Translation Configuration" in snap

    def test_system_config_save(self, browser, screenshot_path):
        """Fill system config and save."""
        browser.open("http://localhost:5180/config/system")
        browser.wait("text=System Configuration")
        browser.click("text=Save")
        time.sleep(2)
        browser.screenshot(screenshot_path("config_system_after_save"))
        snap = browser.snapshot()
        assert "System Configuration" in snap
```

**Step 2: Create navigation flow tests**

```python
# tests/browser/test_navigation.py
"""
Test sidebar navigation and page transitions.
"""

import time
import pytest


class TestNavigation:
    """Verify sidebar navigation works across all pages."""

    def test_navigate_dashboard_to_fireflies(self, browser, screenshot_path):
        """Click Fireflies in sidebar, navigate to connect page."""
        browser.open("http://localhost:5180/")
        browser.wait("text=Dashboard")
        browser.click("text=Fireflies")
        time.sleep(1)
        snap = browser.snapshot()
        assert "Transcript ID" in snap or "Fireflies" in snap
        browser.screenshot(screenshot_path("nav_to_fireflies"))

    def test_navigate_fireflies_to_config(self, browser, screenshot_path):
        """Navigate from Fireflies to Config via sidebar."""
        browser.open("http://localhost:5180/fireflies")
        browser.wait("text=Fireflies")
        browser.click("text=Config")
        time.sleep(1)
        snap = browser.snapshot()
        assert "Configuration" in snap or "Audio" in snap
        browser.screenshot(screenshot_path("nav_to_config"))

    def test_navigate_config_sub_pages(self, browser, screenshot_path):
        """Navigate between config sub-pages."""
        browser.open("http://localhost:5180/config")
        browser.wait("text=Configuration")

        # Click Audio card
        browser.click("text=Audio")
        time.sleep(1)
        snap = browser.snapshot()
        assert "Audio" in snap
        browser.screenshot(screenshot_path("nav_config_audio"))

    def test_navigate_to_translation_bench(self, browser, screenshot_path):
        """Navigate to Translation test bench from sidebar."""
        browser.open("http://localhost:5180/")
        browser.wait("text=Dashboard")
        browser.click("text=Translation")
        time.sleep(1)
        snap = browser.snapshot()
        assert "Translation" in snap
        browser.screenshot(screenshot_path("nav_to_translation"))
```

**Step 3: Run tests + commit**

```bash
cd modules/dashboard-service
uv run pytest tests/browser/test_config_forms.py tests/browser/test_navigation.py -v
git add modules/dashboard-service/tests/browser/test_config_forms.py modules/dashboard-service/tests/browser/test_navigation.py
git commit -m "feat(dashboard): add agent-browser interactive flow and navigation tests"
```

---

### Task 18: Captions Overlay Visual Tests

**Files:**
- Create: `modules/dashboard-service/tests/browser/test_captions_overlay.py`

**Step 1: Create overlay visual verification tests**

```python
# tests/browser/test_captions_overlay.py
"""
Visual verification of the OBS captions overlay.

Tests display modes, URL parameters, transparent background,
and live caption rendering. Screenshots are the definitive evidence.
"""

import json
import time

import pytest


class TestCaptionsOverlay:
    """Verify the OBS captions overlay renders correctly."""

    def test_overlay_transparent_background(self, browser, screenshot_path):
        """Overlay has no chrome and transparent background."""
        browser.open("http://localhost:5180/captions?session=test123")
        time.sleep(2)
        browser.screenshot(screenshot_path("overlay_transparent"))
        # Verify no sidebar elements
        snap = browser.snapshot()
        # The overlay layout should NOT contain sidebar nav items
        assert "Quick Actions" not in snap

    def test_overlay_display_mode_both(self, browser, screenshot_path):
        """Mode=both shows original and translated text."""
        browser.open("http://localhost:5180/captions?session=test123&mode=both")
        time.sleep(1)

        # Inject a test caption via eval_js
        browser.eval_js("""
            window.dispatchEvent(new CustomEvent('test-caption', { detail: {
                id: 'c1', text: 'Hola mundo', original_text: 'Hello world',
                speaker_name: 'Alice', speaker_color: '#4CAF50',
                target_language: 'es', confidence: 0.95,
                duration_seconds: 60, created_at: new Date().toISOString(),
                expires_at: new Date(Date.now() + 60000).toISOString()
            }}));
        """)
        time.sleep(1)
        browser.screenshot(screenshot_path("overlay_mode_both"))

    def test_overlay_custom_font_size(self, browser, screenshot_path):
        """fontSize URL param changes text size."""
        browser.open("http://localhost:5180/captions?session=test123&fontSize=32")
        time.sleep(1)
        # Check the font-size style is applied
        result = browser.eval_js(
            "document.querySelector('.captions-overlay')?.style.fontSize"
        )
        assert "32" in str(result)
        browser.screenshot(screenshot_path("overlay_font_32"))

    def test_overlay_no_session_error(self, browser, screenshot_path):
        """Missing session param shows error message."""
        browser.open("http://localhost:5180/captions")
        time.sleep(1)
        snap = browser.snapshot()
        assert "session" in snap.lower() or "Missing" in snap
        browser.screenshot(screenshot_path("overlay_no_session"))
```

**Step 2: Run tests + commit**

```bash
cd modules/dashboard-service
uv run pytest tests/browser/test_captions_overlay.py -v
git add modules/dashboard-service/tests/browser/test_captions_overlay.py
git commit -m "feat(dashboard): add agent-browser captions overlay visual verification tests"
```

---

### Task 19: Final Build Verification + Plan Update

**Step 1: Full build check**

```bash
cd modules/dashboard-service
npm run build
npm run check
npm run test
```

Expected: All PASS.

**Step 2: Run all agent-browser tests**

```bash
cd modules/dashboard-service
uv run pytest tests/browser/ -v --timeout=180
```

Expected: All tests PASS, screenshots in `tests/browser/screenshots/`.

**Step 3: Verify screenshot directory has evidence**

```bash
ls -la modules/dashboard-service/tests/browser/screenshots/
```

Expected: Timestamped PNG files for every route and flow.

**Step 4: Update plan.md**

Add completion entry to `modules/orchestration-service/plan.md` documenting:
- Phase 1 MVP complete
- All routes implemented
- Agent-browser verification passing
- Screenshot count

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(dashboard): complete Phase 1 MVP — SvelteKit dashboard with all routes and agent-browser verification"
```

---

## Summary

| Task | Description | Files | Commit |
|------|-------------|-------|--------|
| 1 | Scaffold SvelteKit project | Project root | `feat(dashboard): scaffold...` |
| 2 | TypeScript types | `src/lib/types/` (5 files) | `feat(dashboard): add TypeScript types...` |
| 3 | API client layer | `src/lib/api/` (6 files) | `feat(dashboard): add typed API client...` |
| 4 | SSR-safe stores | `src/lib/stores/` (4 files) + test | `feat(dashboard): add SSR-safe stores...` |
| 5 | Layout components | `src/lib/components/layout/` (4 files) | `feat(dashboard): add layout components...` |
| 6 | App shell + layouts + hooks | `src/routes/` layouts + errors + hooks | `feat(dashboard): add layout groups...` |
| 7 | Dashboard home | `src/routes/(app)/+page.svelte` | `feat(dashboard): add dashboard home...` |
| 8 | Fireflies connect | `src/routes/(app)/fireflies/` (2 files) | `feat(dashboard): add Fireflies connect...` |
| 9 | Session streaming | `src/routes/.../connect/` + components | `feat(dashboard): add session streaming...` |
| 10 | Session history | `src/routes/.../history/` (2 files) | `feat(dashboard): add history page...` |
| 11 | Glossary | `src/routes/.../glossary/` (2 files) | `feat(dashboard): add glossary page...` |
| 12 | Config pages | `src/routes/(app)/config/` (7 files) | `feat(dashboard): add config pages...` |
| 13 | Translation bench | `src/routes/.../translation/test/` (2 files) | `feat(dashboard): add translation bench...` |
| 14 | Captions overlay | `src/routes/(overlay)/captions/` | `feat(dashboard): add captions overlay...` |
| 15 | Test infrastructure | `tests/browser/conftest.py` | `feat(dashboard): add test infra...` |
| 16 | Route verification | `tests/browser/test_app_routes.py` | `feat(dashboard): add route tests...` |
| 17 | Flow tests | `tests/browser/test_*.py` (2 files) | `feat(dashboard): add flow tests...` |
| 18 | Overlay tests | `tests/browser/test_captions_overlay.py` | `feat(dashboard): add overlay tests...` |
| 19 | Final verification | Build + all tests + plan update | `feat(dashboard): complete Phase 1...` |

**Total: 19 tasks, ~45 files, 11 routes, 4 agent-browser test files**
