# Translation Testing Guide

This guide covers current test paths for translation-related runtime checks.

## Test Targets

- Full pipeline (recommended): `tests/e2e/test_loopback_fullstack.py`
- Direct service integration: `tests/integration/test_loopback_translation.py`

## Start Required Services

Use compose profiles:

```bash
just compose-up profiles="core,inference,ui,infra"
```

Or start services manually using the commands in [Quick Start](./quick-start.md).

## Run the Full Pipeline Test

```bash
cd tests/e2e
python test_loopback_fullstack.py
```

Flow under test:

`Audio -> Orchestration (/api/audio/upload) -> Whisper -> Translation`

## Run Direct Integration Test

```bash
cd tests/integration
python test_loopback_translation.py
```

Flow under test:

`Audio -> Whisper -> Translation`

## Health Checks

```bash
curl http://localhost:3000/api/health
curl http://localhost:5001/health
curl http://localhost:5003/api/health
```

## Common Issues

- Missing loopback device: install/configure BlackHole (macOS) or use microphone input.
- 422 upload errors: confirm request fields match orchestration `audio/upload` API expectations.
- Backend unavailable: check service logs with `just compose-logs`.
