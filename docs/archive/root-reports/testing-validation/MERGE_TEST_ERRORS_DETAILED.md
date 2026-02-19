# Merge Test Errors - Detailed Analysis

**Date**: 2026-01-05
**Context**: Post-merge testing revealed several issues

## Critical Errors Found

### 1. ❌ Missing Import: `get_event_publisher` in bot_lifecycle.py
**Severity**: 🔴 HIGH (Blocks ALL orchestration tests)
**File**: `modules/orchestration-service/src/routers/bot/bot_lifecycle.py:36`
**Error**:
```python
event_publisher=Depends(get_event_publisher),
                        ^^^^^^^^^^^^^^^^^^^
NameError: name 'get_event_publisher' is not defined
```
**Impact**: Prevents loading of bot routers, blocks all orchestration tests
**Fix**: Add `get_event_publisher` to imports
**Status**: ✅ FIXED in audio_core.py, NEED TO FIX in bot_lifecycle.py

---

### 2. ❌ SQLAlchemy Table Redefinition Error
**Severity**: 🟡 MEDIUM (Blocks orchestration integration tests)
**File**: `modules/orchestration-service/tests/integration/test_audio_orchestration.py`
**Error**:
```python
sqlalchemy.exc.InvalidRequestError: Table 'users' is already defined for this MetaData instance.
Specify 'extend_existing=True' to redefine options and columns on an existing Table object.
```
**Impact**: Database model initialization conflict
**Root Cause**: Test imports models multiple times or conflicting definitions
**Fix**: Add `extend_existing=True` to Table definitions or fix import order

---

### 3. ❌ Missing Module: `timecode` (SMPTE)
**Severity**: 🟡 MEDIUM (Blocks 1 test file)
**File**: `modules/orchestration-service/tests/integration/test_chunking_integration.py:17`
**Error**:
```python
from timecode import Timecode
ModuleNotFoundError: No module named 'timecode'
```
**Impact**: SMPTE timecode chunking tests cannot run
**Fix**: `pip install timecode` or mark test as optional

---

### 4. ❌ Missing Module: `test_utils` in Whisper
**Severity**: 🟡 MEDIUM (Blocks 1 test file)
**File**: `modules/whisper-service/tests/integration/milestone2/test_real_code_switching.py:33`
**Error**:
```python
from test_utils import (
ModuleNotFoundError: No module named 'test_utils'
```
**Impact**: Milestone 2 code-switching test cannot run
**Root Cause**: `test_utils` module missing or wrong import path
**Fix**: Verify test_utils.py exists in `tests/` or `tests/integration/`

---

### 5. ❌ Service Not Running: Whisper WebSocket
**Severity**: ⚠️ LOW (Expected - integration test)
**File**: `modules/whisper-service/tests/integration/tests/integration/test_mixed_direct.py:34`
**Error**:
```python
sio.connect('http://localhost:5001')
socketio.exceptions.ConnectionError: Unexpected status code 404 in server response
```
**Impact**: WebSocket integration test requires running service
**Root Cause**: whisper-service not running on port 5001
**Fix**: Expected behavior - integration tests need services

---

### 6. ❌ Duplicate Test Path Structure
**Severity**: 🟡 MEDIUM (Confusing structure)
**Path**: `modules/whisper-service/tests/integration/tests/integration/`
**Issue**: Nested duplicate `tests/integration/` directories
**Impact**: Confusing structure, some tests in wrong location
**Fix**: Move `tests/integration/tests/integration/*.py` → `tests/integration/`

---

### 7. ❌ Missing pytest marker: 'e2e'
**Severity**: ⚠️ LOW (Config issue)
**File**: `modules/orchestration-service/tests/integration/test_pipeline_e2e.py`
**Error**: `'e2e' not found in \`markers\` configuration option`
**Impact**: pytest warning, test may still run
**Fix**: Add to pytest.ini:
```ini
[pytest]
markers =
    e2e: End-to-end integration tests
```

---

### 8. ❌ Translation Service pytest-cov Config
**Severity**: 🟡 MEDIUM (Blocks translation tests)
**File**: `modules/translation-service/pyproject.toml` or `pytest.ini`
**Error**:
```
ERROR: unrecognized arguments: --cov=src --cov-report=html --cov-report=term-missing
```
**Impact**: Cannot run ANY translation tests
**Fix**: Install `pip install pytest-cov` or remove coverage args

---

## Error Summary by Module

### Orchestration Service
- ❌ Missing import (get_event_publisher) - **BLOCKING**
- ❌ SQLAlchemy table redefinition - **BLOCKING**
- ❌ Missing timecode module - Specific test
- ❌ Missing pytest marker - Warning only

### Whisper Service
- ❌ Missing test_utils module - Specific test
- ❌ Service not running - Expected (integration)
- ❌ Duplicate path structure - Organizational

### Translation Service
- ❌ Missing pytest-cov - **BLOCKING**

### Bot Container
- ❌ Missing bot_main, orchestration_client - Expected (future implementation)

---

## Immediate Fixes Required

### Fix 1: Add get_event_publisher Import
**File**: `modules/orchestration-service/src/routers/bot/bot_lifecycle.py`

Find the imports section and add:
```python
from dependencies import (
    ...existing imports...,
    get_event_publisher,  # ADD THIS LINE
)
```

### Fix 2: Install pytest-cov
```bash
pip install pytest-cov
```

### Fix 3: Install timecode (optional)
```bash
pip install timecode
```

### Fix 4: Fix Whisper test_utils Import
**File**: `modules/whisper-service/tests/integration/milestone2/test_real_code_switching.py`

Change line 33 from:
```python
from test_utils import (
```
To:
```python
from ..test_utils import (  # Relative import
# OR
from tests.test_utils import (  # Absolute import
```

### Fix 5: Add pytest Marker
**File**: `modules/orchestration-service/pytest.ini`

Add:
```ini
markers =
    e2e: End-to-end integration tests
    integration: Integration tests
    slow: Slow running tests
```

### Fix 6: Fix SQLAlchemy Table Redefinition
**File**: Look for duplicate Table('users', ...) definitions

Add to existing tables:
```python
__table_args__ = {'extend_existing': True}
```

---

## Test Statistics After Fixes

### Before Fixes
- ✅ System: 7/7 passed
- ⚠️ Whisper unit: ~85% passed (incomplete)
- ❌ Orchestration unit: BLOCKED
- ❌ Orchestration integration: BLOCKED
- ❌ Translation: BLOCKED
- ❌ Bot: 19/31 failed (expected)

### Expected After Fixes
- ✅ System: 7/7 passed
- ✅ Whisper unit: ~90% passed (full run)
- ✅ Orchestration unit: Should pass
- ⚠️ Orchestration integration: Partial (needs services)
- ✅ Translation: Should pass
- ⚠️ Bot: Still missing implementations (expected)

---

## Priority Action Items

1. **HIGH**: Fix get_event_publisher import → Commit
2. **HIGH**: Install pytest-cov → Re-run translation tests
3. **MEDIUM**: Fix test_utils import in whisper
4. **MEDIUM**: Add pytest markers to config
5. **MEDIUM**: Install timecode package
6. **MEDIUM**: Fix SQLAlchemy table redefinition
7. **LOW**: Reorganize whisper test directory structure

---

## Files Affected by Merge

These errors suggest the merge introduced or exposed:
1. Import completeness issues (get_event_publisher)
2. Dependency issues (pytest-cov, timecode)
3. Test organization issues (test_utils path)
4. Database model conflicts (SQLAlchemy)

**Root Cause**: REMOTE's ruff cleanup may have removed "unused" imports that ARE actually used.

---

## Recommendation

**Immediate**: Fix the blocking import issue and commit
**Short-term**: Install missing dependencies and re-run tests
**Medium-term**: Reorganize test structure and fix model conflicts

**Overall**: These are **FIXABLE** issues, not fundamental merge problems.
