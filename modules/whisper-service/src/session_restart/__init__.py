"""
Session-Restart Code-Switching (Milestone 2)

Per FEEDBACK.md lines 171-184:
- Session-restart approach for inter-sentence language switching
- Restart ASR session with new language SOT when language changes
- Only switch at VAD boundaries (clean speech breaks)
- Expected accuracy: 70-85%

Components:
- SessionRestartTranscriber: Main orchestrator for code-switching
"""

from .session_manager import SessionRestartTranscriber

__all__ = ["SessionRestartTranscriber"]
