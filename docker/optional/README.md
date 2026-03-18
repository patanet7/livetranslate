# Optional Docker Stacks

This directory holds non-default Docker workflows that are still kept for compatibility or troubleshooting.

## Files

- `compose.local.yml`: transitional multi-service compose stack kept for compatibility investigation. It is not the canonical day-to-day development path.

## Canonical Local Development

Use the repository root workflow instead:

```bash
just install
just db-up
just dev
```

Or follow `docs/guides/quick-start.md` for service-by-service startup.
