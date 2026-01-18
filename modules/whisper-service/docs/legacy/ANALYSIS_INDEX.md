# Whisper Service Codebase Analysis - Document Index

**Date**: 2025-10-29  
**Analyzed**: 74 Python files across /modules/whisper-service/  
**Status**: CRITICAL - Code-switching broken, single-language working

## Quick Start

Start here if you're new to this analysis:

1. **Read this file first** (5 minutes) - Overview and navigation
2. **Read ANALYSIS_SUMMARY.txt** (10 minutes) - Executive summary with key findings
3. **Read whisper_codebase_analysis.md** (30 minutes) - Detailed architecture and issues
4. **Reference whisper_implementation_details.md** (as needed) - Code examples and implementation details

## Documents Overview

### 1. ANALYSIS_SUMMARY.txt (This is your starting point)
**Best for**: Quick understanding of critical issues and recommendations

Contains:
- Critical findings (4-line summary)
- Root cause analysis (3 core issues)
- What's working (10 items) vs. what's broken (8 items)
- 4 broken components with specific line numbers
- Architecture gaps table
- Key files guide (organized by category)
- Recommended action plan (3 options: immediate, short-term, long-term)
- Test results summary
- Quick reference and key takeaway

**Action Items**: Revert broken changes, implement session-restart approach, or switch to standard Whisper

---

### 2. whisper_codebase_analysis.md (Deep technical analysis)
**Best for**: Understanding the full architecture and why it's broken

**14 Sections** (28 KB total):
1. Executive Summary
   - Fundamental architectural incompatibility
   - Code-switching status (0-20% accuracy)
   - Current vs. FEEDBACK.md comparison

2. Current Streaming Implementation (1.1-1.4)
   - SimulStreaming base architecture (799 lines)
   - VAC online processor (inverted logic)
   - Language identification components (passive only)
   - Architecture gaps table

3. Key Files and Implementation (2.0)
   - Core streaming files (4 files)
   - Model and configuration (5 files)
   - Language detection (3 files)
   - Audio processing (5 files)
   - API and server (6 files)
   - Support infrastructure (8+ files)

4. Encoder-Decoder Architecture (3.1-3.2)
   - Current single-decoder design (diagram)
   - What code-switching would require (diagram)
   - Missing components (7 items)

5. KV Cache Management (4.1-4.2)
   - Current single-cache design (code example)
   - Language-conditioning problem
   - Why code-switching breaks cache

6. Token Management and SOT (5.1-5.2)
   - Current token sequence management
   - Code-switching breakage pattern
   - SOT token problem explanation

7. Test Infrastructure (6.1-6.2)
   - Code-switching tests (all failing)
   - Baseline tests (all passing)

8. API Endpoints and Server (7.1-7.2)
   - REST endpoints
   - WebSocket infrastructure
   - Code-switching support status

9. Hardware Acceleration (8.1-8.2)
   - GPU/NPU/CPU support
   - Device selection priority
   - Memory requirements

10. Critical Gaps Summary (9.1-9.3)
    - Architecture gaps (8 items)
    - Implementation gaps (7 tasks, 1100-1650 LOC, 13-21 days)
    - Broken components (4 functions)

11. Test Results (10.1-10.2)
    - Documented failures (4 tests)
    - What's working (6 items)

12. Current vs. FEEDBACK.md Requirements (11.1-11.3)
    - Parallel decoder architecture comparison
    - LID stream comparison
    - Commit policy comparison

13. Recommendations (12.1-12.4)
    - Immediate action (revert changes)
    - Short-term (session restart)
    - Long-term (standard Whisper)
    - Not recommended (parallel decoders)

14. File Reference Guide (13.1-13.6)
    - Streaming core files
    - Language detection files
    - Decoders
    - Audio processing
    - API & Server
    - Models

15. Conclusion (14.1-14.4)
    - Current status
    - What's working
    - What's broken
    - Path forward

---

### 3. whisper_implementation_details.md (Code-level reference)
**Best for**: Implementation planning, code review, debugging

**11 Sections** (20 KB total):
1. Complete File Listing
   - 74 files categorized into 9 groups
   - Lines of code and status for each

2. Critical Code Locations (4 bugs)
   - Bug #1: Dynamic detection (line 482-485)
   - Bug #2: SOT reset (line 251-271)
   - Bug #3: Processing order (line 350-372)
   - Bug #4: Newest segment detection (line 467-474)
   - Each with code examples and impact analysis

3. Current vs. Required Architecture
   - Current single-decoder (code example)
   - Required parallel decoders (code example)
   - Missing components highlighted

4. KV Cache Structure Deep Dive
   - Hook installation (lines 88-102)
   - Cache growth pattern
   - Language-conditioning explanation with example

5. Token Sequence Management
   - Current token initialization (code)
   - Token flow during normal operation
   - Code-switching breakage pattern

6. LID Components Inventory
   - SlidingLIDDetector (passive UI-only)
   - TextLanguageDetector (post-processing)
   - Whisper's detect_language() (called once per session)
   - Each with code examples

7. Alignment-Attention Mechanism
   - AlignAtt concept
   - Cross-attention hook installation
   - Alignment heads extraction (lines 104-111)
   - Frame-level stop condition

8. Hardware Acceleration Paths
   - Device selection (line 47-55)
   - GPU memory requirements
   - Memory analysis with examples

9. Test Coverage Gaps
   - What's tested (passing tests)
   - What's not tested (failing tests)

10. Broken Functions Summary
    - 4-row table with fix for each

11. Effort Estimation for Parallel Decoders
    - 7 components to implement
    - LOC and days for each
    - Total: 1100-1650 LOC, 13-21 days

---

## Critical Information At a Glance

### Broken Components (4)
1. **update_language_tokens()** [simul_whisper.py:251-271]
   - Resets context + clears cache
   - FIX: DELETE

2. **process_iter()** [vac_online_processor.py:350-372]
   - VAD check is second (should be first)
   - FIX: REVERT

3. **Language detection** [simul_whisper.py:482-485]
   - Every-chunk detection (should be once)
   - FIX: REVERT

4. **Newest segment detection** [simul_whisper.py:467-474]
   - Double encoder call
   - FIX: REVERT

### Architecture Gaps (12)
| Component | Gap | Impact |
|-----------|-----|--------|
| Parallel decoders (2+) | Missing | Cannot run multiple languages |
| Per-decoder KV caches | Missing | Cache language-mixes |
| Per-decoder tokenizers | Missing | Cannot switch languages cleanly |
| Cross-attention masking | Missing | Cannot gate attention by language |
| Logit-space fusion | Missing | Cannot select best language output |
| Per-language logit masks | Missing | Cannot apply per-language token gating |
| Hysteresis/dwell logic | Missing | Cannot stabilize language switches |
| LID-gated control | Missing | Language detection not affecting decoder |
| Active LID model | Missing | No frame-level language stream |
| Commit policy | Missing | No stability-based token release |
| Attention masking hooks | Missing | Cannot prevent cross-language hallucinations |
| Token buffer for CS | Missing | Cannot hold tokens during language uncertainty |

### File Priority for Understanding

**Must Read** (understand the code-switching problem):
1. `/src/simul_whisper/simul_whisper.py` (799 lines)
   - Lines 251-271: update_language_tokens() - THE KEY BROKEN FUNCTION
   - Lines 482-485: Dynamic language detection
   - Lines 467-474: Newest segment detection

2. `/src/vac_online_processor.py` (350+ lines)
   - Lines 350-372: process_iter() - INVERTED LOGIC

**Should Understand** (existing good implementation):
3. `/src/api_server.py` (~1000+ lines) - REST API
4. `/src/websocket_stream_server.py` (~500 lines) - WebSocket
5. `/src/models/pytorch_manager.py` (~300 lines) - GPU loading

**Can Skip** (low priority for CS fix):
6. `/src/sliding_lid_detector.py` (210 lines) - UI-only
7. `/src/text_language_detector.py` (214 lines) - Post-processing

---

## Navigation Guide

### If you want to understand...

**...why code-switching is broken**
→ Read: ANALYSIS_SUMMARY.txt (Quick overview)
→ Then: whisper_codebase_analysis.md § 1-5 (Architecture + KV cache)

**...what's implemented currently**
→ Read: whisper_codebase_analysis.md § 2 (Key files)
→ Then: whisper_implementation_details.md § 1 (Complete file listing)

**...how to fix it (revert broken changes)**
→ Read: ANALYSIS_SUMMARY.txt (4 broken components)
→ Then: whisper_implementation_details.md § 2 (Code locations)

**...what would be needed for true code-switching**
→ Read: whisper_codebase_analysis.md § 11-12 (Requirements vs. current)
→ Then: whisper_implementation_details.md § 3-7 (Architecture details)

**...how much effort to implement parallel decoders**
→ Read: whisper_implementation_details.md § 11 (Effort estimation)

**...test results and what's working**
→ Read: whisper_codebase_analysis.md § 6-10 (Tests)
→ Then: ANALYSIS_SUMMARY.txt § TEST RESULTS

---

## Recommended Next Steps

### Option 1: Quick Fix (1-2 hours) - RECOMMENDED FOR TODAY
Revert code-switching changes to restore 75-90% baseline accuracy:
1. Revert vac_online_processor.py:350-372 (VAD check first)
2. Revert simul_whisper.py:482-485 (detect language once)
3. Delete update_language_tokens() (line 251-271)
4. Revert newest-segment detection (line 467-474)
5. Remove enable_code_switching flag

**Where to find specifics**: whisper_implementation_details.md § 2

### Option 2: Short-Term Solution (1-2 weeks) - RECOMMENDED FOR PRODUCTION
Implement session-restart approach for inter-sentence code-switching:
- Create MultiLanguageSessionManager wrapper
- Detect language change → finish session → start new session
- 70-85% accuracy, simple implementation, production-safe

**Where to start**: whisper_codebase_analysis.md § 12.2

### Option 3: Long-Term Solution (1-2 months) - BEST FOR INTRA-SENTENCE CODE-SWITCHING
Replace SimulStreaming with standard Whisper + sliding window:
- 10s window, 5s stride approach
- Native code-switching support from Whisper
- 60-80% accuracy, 5-10s latency (higher than SimulStreaming)

**Where to start**: whisper_codebase_analysis.md § 12.3

---

## Key Statistics

- **Total files analyzed**: 74 Python files
- **Total lines of code**: ~11,000+ lines
- **Core streaming files**: 4 (SimulStreaming)
- **Broken components**: 4 (all documented with line numbers)
- **Architecture gaps**: 12 (detailed analysis provided)
- **Working components**: 10+ (production-ready)
- **Test files**: 30+ (6 passing, 4+ failing on code-switching)
- **Effort to implement parallel decoders**: 13-21 days (NOT RECOMMENDED)
- **Effort to revert changes**: 1-2 hours
- **Effort for session-restart approach**: 5-10 days
- **Effort for standard Whisper approach**: 20-30 days

---

## Document Statistics

- **ANALYSIS_SUMMARY.txt**: Quick reference (8 KB, 250 lines)
- **whisper_codebase_analysis.md**: Full architecture analysis (28 KB, 1200 lines)
- **whisper_implementation_details.md**: Code reference (20 KB, 800 lines)
- **Total documentation**: 56 KB of comprehensive analysis

---

## Questions?

1. **"How do I understand the broken code?"**
   → whisper_implementation_details.md § 2 (Critical code locations)

2. **"Why can't we just add parallel decoders?"**
   → whisper_codebase_analysis.md § 12.4 (Why parallel decoders won't work)
   → whisper_codebase_analysis.md § 3-5 (Architecture constraints)

3. **"What's the simplest way to support code-switching?"**
   → ANALYSIS_SUMMARY.txt § RECOMMENDED ACTION (Session restart)

4. **"Where are the key files?"**
   → whisper_codebase_analysis.md § 13 (File reference guide)

5. **"What's the current architecture?"**
   → whisper_implementation_details.md § 3 (Current vs. required)

---

**Analysis completed**: 2025-10-29  
**Confidence level**: HIGH (Evidence-based, specific code references)  
**Next action**: Choose an option from "Recommended Next Steps" above
