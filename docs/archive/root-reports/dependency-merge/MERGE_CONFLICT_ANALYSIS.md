# Merge Conflict Analysis: HEAD vs origin/main

**Date**: 2026-01-05
**Total Conflicts**: 27 files
**Status**: REQUIRES CAREFUL REVIEW

## Executive Summary

### LOCAL (HEAD) - 142 commits
- **Focus**: Backend feature development
- **Major Work**: Data pipeline, database integration, bot management, refactoring
- **Code Quality**: Refactored, modular, follows best practices
- **Lines Changed**: +4,000+ (substantial additions)

### REMOTE (origin/main) - 33 commits
- **Focus**: Frontend DRY refactoring + Python linting
- **Major Work**: TypeScript cleanup, DRY hooks, ruff linting
- **Code Quality**: DRY patterns, cleaned TypeScript, linting fixes
- **Lines Changed**: -550+ (duplication elimination)

---

## File-by-File Comparison

### Category 1: Import Conflicts (TRIVIAL - Merge Both)

#### 1. `modules/orchestration-service/src/database/models.py`
**Conflict**: Import statement style
```python
# LOCAL (HEAD)
from sqlalchemy.orm import declarative_base, relationship, Session

# REMOTE (origin/main)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
```
**Decision**: ✅ **USE LOCAL** - Modern SQLAlchemy pattern + includes Session import
**Impact**: None - both work, LOCAL is more modern

#### 2. `modules/whisper-service/src/whisper_service.py` (Import section)
**Conflict**: Dataclasses import
```python
# LOCAL
from dataclasses import asdict

# REMOTE
from dataclasses import dataclass
```
**Decision**: ✅ **MERGE BOTH** - `from dataclasses import asdict, dataclass`
**Impact**: Both imports are needed

---

### Category 2: Backend Python - Architecture Conflicts

#### 3. `modules/whisper-service/src/whisper_service.py` (CRITICAL)
**LOCAL**: 756 lines - **Refactored modular architecture**
- Uses `PyTorchModelManager` from `models/pytorch_manager.py` (41KB separate file)
- Uses components from `transcription/` package
- Uses `session/SessionManager`
- Imports: BeamSearchDecoder, AlignAttDecoder, DomainPromptManager, SileroVAD, StabilityTracker
- **Clean Single Responsibility Principle**

**REMOTE**: 1,165 lines - **Monolithic architecture**
- All classes in one file: ModelManager, AudioBufferManager, SessionManager, WhisperService
- 409 MORE lines (no refactoring)

**Recent Commits**:
- LOCAL: Milestone 2 code-switching, VAD-first architecture, performance fixes
- REMOTE: Ruff linting fixes only

**Decision**: ✅ **USE LOCAL** - Properly refactored vs monolithic
**Impact**: Major - maintains clean architecture vs reverting to legacy pattern

#### 4. `modules/orchestration-service/src/audio/audio_coordinator.py`
**LOCAL**: +1,232 lines of features
- Data pipeline integration (TranscriptionDataPipeline)
- Translation caching (TranslationResultCache)
- Client imports (AudioServiceClient, TranslationServiceClient)
- Optimization adapter (TranslationOptimizationAdapter)
- Complete WebM/MP4 audio processing
- Metadata tracking and database integration

**REMOTE**: Simpler version
- Basic ServiceClientPool
- Pass-through mode for compatibility
- No data pipeline integration

**Decision**: ✅ **USE LOCAL** - Actual working features vs stub implementation
**Impact**: Loses data pipeline if REMOTE chosen

#### 5. `modules/orchestration-service/src/audio/config.py`
**LOCAL**: Includes Pydantic imports
```python
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
```

**REMOTE**: No Pydantic imports (removed during linting)

**Decision**: ✅ **USE LOCAL** - Pydantic imports are needed for validators
**Impact**: Breaks validation if REMOTE chosen

#### 6. `modules/orchestration-service/src/audio/config_sync.py`
**LOCAL**: More imports (os, time, defaultdict, timedelta, Path, Union, Tuple)
**REMOTE**: Fewer imports (cleaned by linting)

**Decision**: ⚠️ **REQUIRES REVIEW** - Need to check which imports are actually used
**Recommendation**: Use LOCAL, verify with Python syntax check

#### 7. `modules/orchestration-service/src/audio/models.py`
**LOCAL**: Modern Pydantic validators
```python
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
```

**REMOTE**: Old Pydantic validators
```python
from pydantic import BaseModel, Field, validator, root_validator
```

**Decision**: ✅ **USE LOCAL** - Modern Pydantic v2 syntax
**Impact**: May break with newer Pydantic if REMOTE chosen

#### 8. `modules/orchestration-service/src/clients/translation_service_client.py`
**LOCAL**: More imports (asyncio, json, os, datetime)
**REMOTE**: Minimal imports

**Decision**: ⚠️ **REQUIRES REVIEW** - Check if async features need these imports
**Recommendation**: Use LOCAL if async translation is implemented

#### 9. `modules/orchestration-service/src/database/processing_metrics.py`
**LOCAL**: Modern SQLAlchemy + duplicate logging import
```python
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean, Index, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import logging  # Listed twice in HEAD
```

**REMOTE**: Old SQLAlchemy + numpy import
```python
import numpy as np
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean, Index, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
```

**Decision**: ✅ **USE LOCAL + remove duplicate logging** + add numpy if needed
**Impact**: Minor - need to verify if numpy is actually used

#### 10. `modules/orchestration-service/src/database/unified_bot_session_repository.py`
**LOCAL**: Import structure
```python
from .models import (
    BotSession,
    AudioFile,
    Transcript,
    Translation,
)
from audio.models import SpeakerCorrelation  # Pydantic model
```

**REMOTE**: Different import
```python
from .models import (
    BotSession, AudioFile, Transcript, Translation,
    SpeakerCorrelation
)
```

**Decision**: ✅ **USE LOCAL** - Separates SQLAlchemy models from Pydantic models
**Impact**: Better organization, clearer intent

#### 11. `modules/orchestration-service/src/main_fastapi.py`
**LOCAL**: More imports (time, List, Optional, Response)
**REMOTE**: Minimal imports + different import order for HTMLResponse/JSONResponse

**Decision**: ⚠️ **REQUIRES REVIEW** - Check if time/List/Optional are used
**Recommendation**: Use LOCAL if features need these imports

#### 12. `modules/orchestration-service/src/routers/audio/_shared.py`
**Decision**: ⚠️ **REQUIRES DIFF CHECK**

#### 13. `modules/orchestration-service/src/routers/audio/audio_presets.py`
**Decision**: ⚠️ **REQUIRES DIFF CHECK**

#### 14. `modules/orchestration-service/src/routers/bot/_shared.py`
**Decision**: ⚠️ **REQUIRES DIFF CHECK**

---

### Category 3: Frontend TypeScript - DRY Refactoring

All 10 frontend files are REMOTE (origin/main) DRY refactoring work:
- Eliminates 550+ lines of duplication
- Creates reusable hooks: useNotifications, useAudioStreaming, useAudioDevices
- Applies TypeScript fixes (200+ errors → 0)
- User confirmed: "frontend stuff is likely solid"

#### 15-24. Frontend Files (Accept REMOTE)
✅ **AudioStageNode.tsx** - DRY refactoring
✅ **RealTimeProcessor.tsx** - DRY refactoring
✅ **SettingsPanel.tsx** - DRY refactoring
✅ **useAudioStreaming.ts** - Chunked upload hook (DRY)
✅ **QualityAnalysis.tsx** - useNotifications hook
✅ **AudioProcessingHub/index.tsx** - DRY patterns
✅ **AudioTesting/index.tsx** - DRY patterns
✅ **MeetingTest/index.tsx** - DRY patterns
✅ **PipelineStudio/index.tsx** - DRY patterns
✅ **StreamingProcessor/index.tsx** - DRY patterns

**Decision**: ✅ **USE REMOTE for ALL** - Proven DRY refactoring
**Impact**: Eliminates 550+ lines of duplication

---

### Category 4: Deleted Files (Keep Deleted)

#### 25-27. Legacy Files (REMOTE has, LOCAL deleted)
✅ **lufs_normalization_stage.py** - Intentionally deleted (replaced with enhanced version)
✅ **buffer_manager.py** - Intentionally deleted (refactored to whisper_service.py)
✅ **enhanced_api_server.py** - Intentionally deleted (legacy)
✅ **whisper_service_fixed.py** - Intentionally deleted (legacy)

**Decision**: ✅ **KEEP DELETED** - Removed as part of refactoring
**Impact**: None - functionality moved to better locations

---

## Recommended Resolution Strategy

### Phase 1: Clear Decisions (18 files)
1. **Deleted files** (4): Keep deleted - `git rm`
2. **Frontend** (10): Accept REMOTE - `git checkout --theirs`
3. **Trivial imports** (2): Merge both - manual edit
4. **Modern Pydantic** (2): Use LOCAL - `git checkout --ours`

### Phase 2: Requires Investigation (9 files)
Need to check actual code differences, not just imports:
- audio_coordinator.py ← **CRITICAL - major features**
- config.py
- config_sync.py
- translation_service_client.py
- processing_metrics.py
- main_fastapi.py
- routers/audio/_shared.py
- routers/audio/audio_presets.py
- routers/bot/_shared.py

### Phase 3: Whisper Service Decision
**whisper_service.py** is THE critical file:
- LOCAL: Clean refactored architecture (756 lines)
- REMOTE: Monolithic legacy (1,165 lines)

**Recommendation**: ✅ USE LOCAL (refactored)
**Verification**: Check that models/pytorch_manager.py exists in LOCAL

---

## Risk Assessment

### HIGH RISK if wrong choice:
- ❌ **whisper_service.py**: Choosing REMOTE loses refactoring work
- ❌ **audio_coordinator.py**: Choosing REMOTE loses data pipeline
- ❌ **models.py (audio)**: Choosing REMOTE breaks Pydantic v2

### LOW RISK:
- ✅ **Frontend files**: REMOTE is proven DRY work
- ✅ **Deleted files**: Keep deleted - no functionality lost
- ✅ **Import fixes**: Easily corrected

### MEDIUM RISK:
- ⚠️ **Router files**: Need diff check to verify no logic changes
- ⚠️ **Config files**: Need to verify all imports are used

---

## Next Steps

1. **RUN COMPREHENSIVE DIFF** on the 9 "Requires Investigation" files
2. **VERIFY** models/pytorch_manager.py exists in LOCAL
3. **CREATE RESOLUTION SCRIPT** based on findings
4. **TEST** Python syntax after resolution
5. **RUN TESTS** to verify functionality
6. **COMMIT** with detailed message

---

## Questions to Answer

1. Does `modules/whisper-service/src/models/pytorch_manager.py` exist in LOCAL?
2. Are the extra imports in LOCAL actually used in the code?
3. Do the router files have logic changes or just linting?
4. Is numpy actually used in processing_metrics.py?

**STATUS**: BLOCKED - Need answers before proceeding
