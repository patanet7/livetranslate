# Dashboard Redesign Inventory

Captured 2026-04-26 as part of D0.1. Snapshot of what exists *before* the
Editorial Riso redesign. Use this as a checklist when propagating tokens
and a sanity check when nothing is being missed.

## Stack

| Layer | Version | Notes |
|---|---|---|
| Svelte | 5.53.12 | Runes mode |
| SvelteKit | 2.55 | adapter-node |
| Tailwind | v4.2.1 | via `@tailwindcss/vite` + `@theme inline` |
| shadcn-svelte | 1.1.1 | Wraps bits-ui with Tailwind |
| bits-ui | 2.16.3 | Headless primitives |
| @lucide/svelte | 0.577.0 | Icons |
| mode-watcher | 1.1.0 | Theme switching |
| svelte-sonner | 1.0.7 | Toasts |
| tw-animate-css | 1.4.0 | Animation utilities |

## Routes (28 page files)

```
src/routes/
├── +error.svelte
├── +layout.svelte                       (root layout)
├── (app)/                                ← main app shell
│   ├── +page.svelte                     (home)
│   ├── +error.svelte
│   ├── +layout.svelte                   (sidebar + topbar)
│   ├── chat/
│   ├── config/                          (audio · connections · settings · system · translation)
│   ├── data/
│   ├── diarization/
│   ├── fireflies/                       (root · connect · glossary · live-feed)
│   ├── intelligence/
│   ├── loopback/                        ★ HERO SURFACE — D4
│   ├── meetings/                        (root · [id] · [id]/live)
│   ├── sessions/
│   └── translation/test/
└── (overlay)/
    ├── +error.svelte
    ├── +layout.svelte
    └── captions/                        ★ separate aesthetic — overlaid on other apps
```

## Components (60 + .svelte files)

| Group | Files | Purpose |
|---|---|---|
| `captions/` | CaptionBox, CaptionStream, InterimCaption | Live caption rendering primitives |
| `chat/` | ChatInput, ChatMessage, ConversationList, SettingsDrawer, ToolCallIndicator | Chat surface |
| `layout/` | PageHeader, Sidebar, StatusIndicator, TopBar | App chrome — D3 targets |
| `loopback/` | InterpreterView, SplitView, SubtitleView, Toolbar, TranscriptView, TranslationText | Hero surface internals — D4 targets |
| `meetings/` | ConnectionBanner, MeetingSkeleton, SyncBadge | Meetings surface |
| `ui/` | shadcn primitives (badge, button, card, dialog, input, label, select, separator, sonner, table, tabs, textarea) | Design-system primitives |
| top-level | ConnectionCard, ConnectionDialog, ErrorBoundary | Mixed |

## shadcn primitive usage (frequency)

```
28  Button       ← restyle these three and 64% of surface inherits the look
18  Card
18  Badge
13  Label
12  Input
 7  Dialog
 6  Separator
 5  Select
 4  Tabs
 3  Textarea
 2  Table
 1  Sonner
```

## Token leverage

**309 class-attribute callsites** consume the OKLCH neutrals
(`bg-card`, `bg-background`, `text-foreground`, `text-muted`, `border-border`,
`bg-primary`, `bg-accent`, `text-primary`, `text-card`). Rewriting `app.css`
:root + .dark blocks (D1.2) propagates the Editorial Riso palette through
every one of those automatically — no per-file edits needed for the bulk
of the visual change.

## Cross-language synchronisation traps

| TS file | Python mirror | Risk |
|---|---|---|
| `src/lib/theme.ts` (`SPEAKER_COLORS`) | `modules/shared/src/livetranslate_common/theme.py` (`SPEAKER_COLORS`) | Updating only one half silently breaks virtual-cam captions / bot subtitles. CLAUDE.md flags this as canonical. **Both must be updated together in D1.3.** |

## Out of scope

- Marketing site / landing pages — none exist in dashboard
- Email templates — none exist
- Auth/login flows — handled outside dashboard
- Mobile app — n/a

## Hero-feature notes (loopback)

`(app)/loopback/+page.svelte` (504 lines) is the showcase. Composes:
`Toolbar`, `SubtitleView`, `TranscriptView`, `SplitView`, `InterpreterView`,
`TranslationText`, plus the loopback Svelte 5 store
(`src/lib/stores/loopback.svelte.ts`).

CLAUDE.md notes:
- AudioWorklet at `static/audio-worklet-processor.js` already computes RMS
- WebSocket flow is `LoopbackWebSocket` (`src/lib/audio/websocket.ts`)
- Audio capture is `AudioCapture` (`src/lib/audio/capture.ts`)

For D2.3 (audio-RMS plumbing): the RMS value is already computed by the
worklet; we just need to surface it through the loopback store as
`$state` so the Earwyrm mascot can subscribe.
