"""
Language Identification (LID) Module

Frame-level language identification for code-switching detection.

Per FEEDBACK.md lines 32-38:
- 80-120ms hop for frame-level LID
- Use lightweight MMS-LID model
- Smooth with Viterbi or hysteresis

Components:
- FrameLevelLID: Core LID detector with MMS-LID
- SustainedLanguageDetector: Hysteresis logic for sustained changes
- LIDSmoother: HMM/Viterbi smoothing for stability
"""

from .lid_detector import FrameLevelLID
from .smoother import LIDSmoother
from .sustained_detector import LanguageSwitchEvent, SustainedLanguageDetector

__all__ = ["FrameLevelLID", "LIDSmoother", "LanguageSwitchEvent", "SustainedLanguageDetector"]
