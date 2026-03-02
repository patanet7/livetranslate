# System Config Editor — Design Document

**Date:** 2026-03-02
**Status:** Approved
**Scope:** Editable system configuration for languages, domains, and defaults

---

## Problem

`system_constants.py` is the hardcoded source of truth for 51 supported languages, 12 glossary domains, and default config values. There is no way to customize these from the dashboard — adding a new domain or disabling an unused language requires a code change and redeploy.

## Solution: Overlay File Pattern

Add a JSON override file (`./config/system.json`) that merges on top of the hardcoded constants at read time. The existing `GET /api/system/ui-config` endpoint becomes the merge point — every page that already reads from it automatically gets customized values.

### Architecture

```
system_constants.py  (factory defaults, read-only)
        +
./config/system.json (user overrides, read-write)
        =
GET /api/system/ui-config  (merged result, served to all pages)
```

## API Changes

### Modified: `GET /api/system/ui-config`

Currently returns raw constants. After this change, it:
1. Loads `./config/system.json` (if exists)
2. Filters `SUPPORTED_LANGUAGES` to only `enabled_languages` (or all if unset)
3. Merges `custom_domains` into `GLOSSARY_DOMAINS`, removes `disabled_domains`
4. Merges `defaults` overrides into `DEFAULT_CONFIG`
5. Returns the merged result (same response shape as today)

### New: `PUT /api/system/ui-config`

Accepts a `SystemConfigUpdate` body and writes to `./config/system.json`.

```python
class SystemConfigUpdate(BaseModel):
    enabled_languages: list[str] | None = None   # ISO codes; None = all enabled
    custom_domains: list[dict] | None = None      # user-created domains
    disabled_domains: list[str] | None = None     # built-in domain values to hide
    defaults: dict | None = None                   # override DEFAULT_CONFIG keys
```

### New: `POST /api/system/ui-config/reset`

Deletes `./config/system.json`, restoring factory defaults.

## JSON File Schema

```json
{
  "enabled_languages": ["en", "es", "fr", "de", "zh", "ja"],
  "custom_domains": [
    {"value": "automotive", "label": "Automotive", "description": "Vehicle and transportation terms"}
  ],
  "disabled_domains": ["pharmaceutical"],
  "defaults": {
    "default_target_languages": ["es", "fr"],
    "confidence_threshold": 0.85
  }
}
```

## Frontend: `/config/system` Page

Three sections:

### 1. Languages (checkbox grid)
- 4-column grid of all 51 languages
- Each: checkbox + code + English name + native name
- RTL languages badged
- Select All / Deselect All buttons
- Saves `enabled_languages` array

### 2. Domains (table with CRUD)
- Table listing built-in + custom domains
- Built-in: toggle to disable (cannot delete)
- Custom: add / edit / delete
- Add form: value (slug), label, description
- Saves `custom_domains` + `disabled_domains`

### 3. Defaults (form)
- Default source language (dropdown)
- Default target languages (multi-select)
- Auto-detect language (toggle)
- Confidence threshold (slider 0-1)
- Context window size (number)
- Max buffer words (number)
- Pause threshold ms (number)
- Saves `defaults` object

### Common
- Save button per section (or global)
- Reset to Defaults button with confirmation
- Toast notifications

## Files

| Action | File |
|--------|------|
| Modify | `src/routers/system.py` |
| Create | `dashboard-service/src/routes/(app)/config/system/+page.svelte` |
| Create | `dashboard-service/src/routes/(app)/config/system/+page.server.ts` |
| Modify | Dashboard sidebar/layout (add nav link) |

## Design Decisions

1. **Built-in domains disabled, not deleted** — prevents breaking glossary logic
2. **`enabled_languages: null` = all** — fresh installs work without config file
3. **Separate `custom_domains` / `disabled_domains`** — upgrades adding new built-in domains appear automatically
4. **Reuse `load_config`/`save_config` from settings._shared** — established pattern
5. **No new router** — extends existing `system.py` to keep read/write at the same URL
