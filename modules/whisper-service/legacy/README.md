# Legacy Whisper Service Files

This directory contains the **WORKING MONOLITHIC** implementation before Phase 2 refactoring broke code-switching.

## Files

### Working Code-Switching Implementation (commit 85d2641)
**Date**: Before "Phase 2 Day 7: Extract core components"
**Status**: ✅ Code-switching WORKING

- `api_server_WORKING.py` (154K, 3642 lines)
  - Flask/SocketIO server with WebSocket streaming
  - Working `join_session` and `transcribe_stream` handlers
  - Proper code-switching configuration flow

- `whisper_service_WORKING.py` (106K, 2392 lines)
  - MONOLITHIC service implementation
  - All transcription logic in one place
  - Clear, traceable code flow

- `vac_online_processor_WORKING.py` (33K, 896 lines)
  - VAC (Voice Activity + Code-switching) processor
  - SimulStreaming incremental processing
  - Working language detection and SOT token handling

- `simul_whisper_WORKING.py` (33K)
  - PaddedAlignAttWhisper stateful wrapper
  - Language detection and task handling
  - Proper `enable_code_switching` support

### Before Phase 2 (commit 20a4c1c)
**Date**: "Phase 1 complete, moving to Phase 2"
**Status**: Basic implementation before SimulStreaming

- `api_server_before_phase2.py` (124K, 3114 lines)
- `whisper_service_before_phase2.py` (54K, 1250 lines)

## What Went Wrong?

### Phase 2 Day 7+ Refactoring Problems:

1. **File Explosion**: Monolithic 2392-line `whisper_service.py` split into 70+ files
2. **Duplication**: Multiple implementations of same functionality (buffer_manager, eow_detection, etc.)
3. **Broken Abstractions**: Clear code flow replaced with circular imports and scattered logic
4. **Lost Configuration**: `enable_code_switching` flag not properly flowing through layers
5. **Language Detection Broken**: All tokens marked as `Lang=zh, SOT=zh` even for English audio

### Current Broken Structure:
```
src/
├── api_server.py              # 3400 lines, still monolithic
├── whisper_service.py         # Gutted, unclear purpose
├── vac_online_processor.py    # Modified, possibly broken
├── buffer_manager.py          # ROOT
├── transcription/
│   └── buffer_manager.py      # DUPLICATE!
├── eow_detection.py           # ROOT
├── simul_whisper/
│   └── eow_detection.py       # DUPLICATE!
├── session/
│   └── session_manager.py
├── stream_session_manager.py  # WHICH ONE IS USED??
├── continuous_stream_processor.py  # vs vac_online_processor.py??
├── audio_processor.py         # Why not in audio/?
├── audio/
│   ├── vad_processor.py
│   └── audio_utils.py
├── vad_detector.py            # 3 VAD implementations!
├── silero_vad_iterator.py
└── [60+ more scattered files]
```

## How to Use This Directory

When debugging current implementation:

1. **Compare implementations**: Look at WORKING files to see what changed
2. **Reference configuration flow**: Check how `enable_code_switching` flowed through
3. **Understand VAC processor**: See how it properly handled language detection
4. **Restore if needed**: Can revert to monolithic structure if refactoring can't be fixed

## Key Lessons

**What Worked (Before)**:
- Monolithic but clear code organization
- Direct configuration passing
- Simple import structure
- Easy to trace execution flow

**What Broke (After)**:
- Premature optimization through file splitting
- Created duplication instead of true modularity
- Lost sight of data flow through layers
- Increased complexity without benefit

**Quote from User**: "ULTRATHINK I think when you extracted from monolithic files you just made a huge mess..."
