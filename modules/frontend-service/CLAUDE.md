# Frontend Service (Legacy React UI)

**Status**: Being replaced by **dashboard-service** (SvelteKit + Svelte 5 runes). This React app remains for settings management and some testing pages.

**Tech Stack**: React 18 + TypeScript + Material-UI + Vite

## Commands

```bash
cd modules/frontend-service
npm install
npm run dev       # Dev server on :5173
npm test          # Vitest unit tests
npm run test:e2e  # Playwright E2E tests
npm run build     # Production build
```

## Key Pages

- `src/pages/Settings/` — Translation settings, connection management (still active)
- `src/pages/AudioTesting/` — Audio capture testing
- `src/pages/Dashboard/` — System overview

## Proxy

Vite proxies `/api` → orchestration service at `:3000`.

## Note

The primary UI is now `modules/dashboard-service/` (SvelteKit). This React frontend is kept for specific features not yet migrated.
