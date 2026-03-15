#!/usr/bin/env python3
"""
Sustained Language Detection with Hysteresis

Per FEEDBACK.md lines 157-167:
- Switch only if P(new) - P(old) > 0.2 for ≥6 consecutive LID frames
- Minimum dwell: 250ms
- Hard stop at VAD boundary

This prevents language flapping and ensures stable language switches.
"""

from collections import deque
from dataclasses import dataclass

from livetranslate_common.logging import get_logger

logger = get_logger()


@dataclass
class LanguageSwitchEvent:
    """Event representing a sustained language switch"""

    from_language: str
    to_language: str
    timestamp: float
    confidence_margin: float  # P(new) - P(old)
    dwell_frames: int  # Number of consecutive frames
    dwell_duration_ms: float  # Duration in milliseconds


class SustainedLanguageDetector:
    """
    Detects sustained language changes with hysteresis logic.

    Per FEEDBACK.md lines 157-167:
    - Requires P(new) - P(old) > margin for ≥ min_frames consecutive frames
    - Requires minimum dwell time (default 250ms)
    - Only switches at VAD boundaries (handled by caller)

    Args:
        confidence_margin: Required confidence margin (default 0.2 per FEEDBACK.md)
        min_dwell_frames: Minimum consecutive frames (default 6 per FEEDBACK.md)
        min_dwell_ms: Minimum dwell time in ms (default 250ms per FEEDBACK.md)
        frame_hop_ms: Frame hop duration in ms (default 100ms = 10Hz)
    """

    def __init__(
        self,
        confidence_margin: float = 0.2,
        min_dwell_frames: int = 6,
        min_dwell_ms: float = 250.0,
        frame_hop_ms: float = 100.0,
    ):
        self.confidence_margin = confidence_margin
        self.min_dwell_frames = min_dwell_frames
        self.min_dwell_ms = min_dwell_ms
        self.frame_hop_ms = frame_hop_ms

        # Current state
        self.current_language: str | None = None
        self.candidate_language: str | None = None
        self.candidate_start_time: float | None = None
        self.candidate_frames: deque = deque(maxlen=min_dwell_frames * 2)

        # Statistics
        self.total_switches = 0
        self.false_positives_prevented = 0  # Times hysteresis prevented premature switch

        logger.info(
            f"SustainedLanguageDetector initialized: margin={confidence_margin}, "
            f"min_frames={min_dwell_frames}, min_dwell={min_dwell_ms}ms"
        )

    def update(self, lid_probs: dict[str, float], timestamp: float) -> LanguageSwitchEvent | None:
        """
        Update with new LID probabilities and check for sustained language change.

        Args:
            lid_probs: Per-language probabilities {'en': 0.7, 'zh': 0.3}
            timestamp: Current timestamp in seconds

        Returns:
            LanguageSwitchEvent if sustained change detected, else None
        """
        # Find top language
        top_language = max(lid_probs, key=lid_probs.get)
        top_prob = lid_probs[top_language]

        # First detection - initialize
        if self.current_language is None:
            self.current_language = top_language
            logger.info(f"✅ Initial language set: {top_language} (p={top_prob:.3f})")
            return None

        # Check if we have a new candidate language
        if top_language != self.current_language:
            current_prob = lid_probs.get(self.current_language, 0.0)
            margin = top_prob - current_prob

            # Check if margin meets threshold
            if margin >= self.confidence_margin:
                # Valid candidate with sufficient margin
                if top_language == self.candidate_language:
                    # Same candidate - increment count
                    self.candidate_frames.append(
                        {
                            "language": top_language,
                            "timestamp": timestamp,
                            "margin": margin,
                            "top_prob": top_prob,
                        }
                    )

                    # Check if candidate is sustained
                    if len(self.candidate_frames) >= self.min_dwell_frames:
                        # Check time duration
                        dwell_duration_ms = (timestamp - self.candidate_start_time) * 1000

                        if dwell_duration_ms >= self.min_dwell_ms:
                            # Sustained change detected!
                            event = self._create_switch_event(
                                from_lang=self.current_language,
                                to_lang=top_language,
                                timestamp=timestamp,
                                margin=margin,
                                frames=len(self.candidate_frames),
                                duration_ms=dwell_duration_ms,
                            )

                            # Update current language
                            self.current_language = top_language
                            self.candidate_language = None
                            self.candidate_start_time = None
                            self.candidate_frames.clear()
                            self.total_switches += 1

                            logger.info(
                                f"🔄 Language switch: {event.from_language} → {event.to_language} "
                                f"(margin={event.confidence_margin:.3f}, "
                                f"frames={event.dwell_frames}, "
                                f"duration={event.dwell_duration_ms:.0f}ms)"
                            )

                            return event
                else:
                    # New candidate - reset tracking
                    self.candidate_language = top_language
                    self.candidate_start_time = timestamp
                    self.candidate_frames.clear()
                    self.candidate_frames.append(
                        {
                            "language": top_language,
                            "timestamp": timestamp,
                            "margin": margin,
                            "top_prob": top_prob,
                        }
                    )

                    logger.debug(
                        f"🔍 New candidate language: {top_language} "
                        f"(margin={margin:.3f} > {self.confidence_margin})"
                    )
            else:
                # Margin too small - noise
                if self.candidate_language is not None:
                    # Hysteresis prevented premature switch
                    self.false_positives_prevented += 1
                    logger.debug(
                        f"⚡ Hysteresis: Prevented premature switch to {top_language} "
                        f"(margin={margin:.3f} < {self.confidence_margin})"
                    )

                # Reset candidate
                self.candidate_language = None
                self.candidate_start_time = None
                self.candidate_frames.clear()
        else:
            # Same as current language - reset candidate
            if self.candidate_language is not None:
                # Hysteresis prevented premature switch
                self.false_positives_prevented += 1
                logger.debug(
                    f"⚡ Hysteresis: Language returned to {self.current_language} "
                    f"before switch completed (prevented {self.candidate_language})"
                )

            self.candidate_language = None
            self.candidate_start_time = None
            self.candidate_frames.clear()

        return None

    def _create_switch_event(
        self,
        from_lang: str,
        to_lang: str,
        timestamp: float,
        margin: float,
        frames: int,
        duration_ms: float,
    ) -> LanguageSwitchEvent:
        """Create a language switch event"""
        return LanguageSwitchEvent(
            from_language=from_lang,
            to_language=to_lang,
            timestamp=timestamp,
            confidence_margin=margin,
            dwell_frames=frames,
            dwell_duration_ms=duration_ms,
        )

    def force_language(self, language: str):
        """
        Force set current language (for initialization or manual override).

        Clears any pending candidates.
        """
        logger.info(f"🔧 Force language: {self.current_language} → {language}")
        self.current_language = language
        self.candidate_language = None
        self.candidate_start_time = None
        self.candidate_frames.clear()

    def get_current_language(self) -> str | None:
        """Get currently stable language"""
        return self.current_language

    def get_candidate_language(self) -> str | None:
        """Get current candidate language (if any)"""
        return self.candidate_language

    def get_statistics(self) -> dict:
        """Get detection statistics"""
        return {
            "current_language": self.current_language,
            "candidate_language": self.candidate_language,
            "total_switches": self.total_switches,
            "false_positives_prevented": self.false_positives_prevented,
            "candidate_frames": len(self.candidate_frames),
            "candidate_progress": (
                f"{len(self.candidate_frames)}/{self.min_dwell_frames} frames"
                if self.candidate_language
                else "none"
            ),
        }

    def reset(self):
        """Reset detector state for new session"""
        logger.info("🔄 Sustained detector reset")
        self.current_language = None
        self.candidate_language = None
        self.candidate_start_time = None
        self.candidate_frames.clear()
