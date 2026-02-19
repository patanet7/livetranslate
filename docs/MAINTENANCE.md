# Documentation Maintenance Standards

This file defines how documentation stays organized and current.

## Scope Buckets

- `README.md` and `docs/guides/*`: current runbooks and onboarding flows.
- `docs/01-context`, `docs/02-containers`, `docs/03-components`: architecture documentation.
- `modules/*/README.md`: service-level operational and implementation details.
- `docs/archive/*`: historical planning, analysis, and one-off reports.

## Update Rules

- Prefer links over duplicated instructions.
- When commands change, update:
  - `README.md`
  - `docs/guides/quick-start.md`
  - affected `modules/*/README.md`
- When a temporary report is no longer active, move it to `docs/archive/` or add it to `docs/archive/root-level-reports.md`.
- Keep filenames stable; do not rename frequently referenced docs without updating links.

## Link Hygiene Checklist

Run this before merging documentation changes:

```bash
rg -n "\]\([^)]*\.md\)" README.md docs modules tests
```

Then confirm referenced files exist.

## Writing Conventions

- Keep procedure steps executable as written.
- Use explicit paths and ports.
- Avoid status language that becomes stale quickly (for example "fully completed"), unless paired with a date.
- Keep architecture docs structural and neutral; keep progress logs in archive.
