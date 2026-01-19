# Pre-commit Issues Classification

**Generated**: 2026-01-18
**Status**: Ready for Systematic Resolution

---

## Summary by Category

| Category | Count | Priority | Effort |
|----------|-------|----------|--------|
| YAML Syntax Errors | 3 | HIGH | Quick Fix |
| Multiple Module Docstrings | 5 | MEDIUM | Quick Fix |
| Shebang Permissions | 80+ | LOW | Batch Fix |
| Branch Protection (main) | 1 | N/A | Config Issue |
| SQL Lint Warnings | Many | LOW | Optional |
| Ruff Formatting | TBD | MEDIUM | Auto-fix |

---

## Category 1: YAML Syntax Errors (HIGH PRIORITY)

**Issue**: Invalid YAML syntax in `.pre-commit-config.yaml` files

**Files**:
1. `modules/frontend-service/.pre-commit-config.yaml` (line 99-102)
2. `modules/translation-service/.pre-commit-config.yaml` (line 100-103)
3. `modules/orchestration-service/.pre-commit-config.yaml` (line 119-122)

**Fix**: These files likely have JSON-style arrays `[]` that should be YAML-style `-` lists.

---

## Category 2: Multiple Module Docstrings (MEDIUM PRIORITY)

**Issue**: Files have docstrings that appear after imports or other code

**Files**:
1. `modules/whisper-service/tests/integration/test_pytorch_real.py:442`
2. `modules/whisper-service/tests/integration/test_real_speech.py:182`
3. `modules/whisper-service/tests/integration/test_openvino_real.py:344`
4. `modules/orchestration-service/src/audio/audio_coordinator_cache_integration.py:24`
5. `modules/orchestration-service/tests/integration/test_audio_orchestration.py:1048`

**Fix**: Remove or relocate the extra docstrings.

---

## Category 3: Shebang Permission Issues (LOW PRIORITY - BATCH FIX)

**Issue**: Python files have `#!/usr/bin/env python3` shebang but aren't marked executable

**Count**: 80+ files

**Fix Options**:
1. **Option A**: Make all files executable with `chmod +x`
2. **Option B**: Remove shebangs from library files (better practice)

**Affected Services**:
- `modules/bot-container/` - 6 files
- `modules/meeting-bot-service/` - 3 files
- `modules/orchestration-service/` - 30+ files
- `modules/translation-service/` - 8 files
- `modules/whisper-service/` - 20+ files
- `reference/` - 2 files
- `scripts/` - 1 file
- `tests/` - 1 file

**Recommendation**: For library modules (src/), remove shebangs. For scripts/tests, add executable bit.

---

## Category 4: Branch Protection

**Issue**: `no-commit-to-branch` hook prevents commits to main

**Resolution**: This is expected - we use `--no-verify` for direct commits to main. For proper workflow, create feature branches.

---

## Category 5: SQL Lint Warnings (LOW PRIORITY)

**Issue**: SQLFluff reports style warnings in SQL files

**Type of warnings**:
- Keyword capitalization (should be uppercase)
- Function name capitalization
- Line length > 80 characters
- Aliasing consistency
- Join qualification

**Affected files**:
- `scripts/bot-sessions-schema.sql`
- Other SQL migration files

**Recommendation**: Fix gradually or configure SQLFluff rules.

---

## Category 6: Ruff Formatting

**Issue**: Some Python files have formatting issues

**Fix**: Run `pdm run ruff format .` to auto-fix

---

## Recommended Fix Order

### Phase 1: Critical Fixes (5 minutes)
```bash
# Fix YAML files (3 files)
# Manually edit the .pre-commit-config.yaml files
```

### Phase 2: Docstring Fixes (10 minutes)
```bash
# Fix the 5 files with multiple docstrings
```

### Phase 3: Batch Permission Fixes (5 minutes)
```bash
# For scripts that should be executable:
find . -name "*.sh" -exec chmod +x {} \;

# For Python files that are entry points:
chmod +x modules/*/src/main.py
chmod +x modules/*/src/api_server.py

# For library files - remove shebang (better approach):
# Or add to .pre-commit-config.yaml to ignore library files
```

### Phase 4: Auto-formatting (1 minute)
```bash
cd /path/to/repo
pdm run ruff format .
```

---

## Files Quick Reference

### YAML Files to Fix:
```
modules/frontend-service/.pre-commit-config.yaml
modules/translation-service/.pre-commit-config.yaml
modules/orchestration-service/.pre-commit-config.yaml
```

### Docstring Files to Fix:
```
modules/whisper-service/tests/integration/test_pytorch_real.py
modules/whisper-service/tests/integration/test_real_speech.py
modules/whisper-service/tests/integration/test_openvino_real.py
modules/orchestration-service/src/audio/audio_coordinator_cache_integration.py
modules/orchestration-service/tests/integration/test_audio_orchestration.py
```
