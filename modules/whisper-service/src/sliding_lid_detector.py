#!/usr/bin/env python3
"""
Sliding LID (Language IDentification) Detector

Tracks language detections in a sliding window for UI/formatting/tagging.
Does NOT affect the decoder - this is passive tracking only.

Phase 3: Multi-Language Code-Switching Implementation
"""

import logging
from typing import Optional, List, Tuple
from collections import Counter
import time

logger = logging.getLogger(__name__)


class LanguageDetection:
    """Single language detection with metadata"""
    def __init__(self, language: str, confidence: float, timestamp: float, audio_position: float):
        self.language = language
        self.confidence = confidence
        self.timestamp = timestamp  # When detection was added (monotonic time)
        self.audio_position = audio_position  # Position in audio stream (seconds)

    def __repr__(self):
        return f"LanguageDetection(lang={self.language}, conf={self.confidence:.2f}, pos={self.audio_position:.2f}s)"


class SlidingLIDDetector:
    """
    Tracks language detections in a sliding time window.

    Purpose:
    - Provides detected_language field for UI/formatting
    - Tracks recent language history in configurable window (e.g., 0.9s)
    - Does NOT affect decoder (passive tracking only)
    - Purges old detections automatically

    Usage:
        detector = SlidingLIDDetector(window_size=0.9)

        # After each transcription chunk
        detector.add_detection(
            language='en',
            confidence=0.95,
            audio_position=2.5  # seconds into stream
        )

        # Get current language (majority in window)
        current = detector.get_current_language()

        # Check if language sustained for duration
        sustained = detector.get_sustained_language(min_duration=2.5)
    """

    def __init__(self, window_size: float = 0.9):
        """
        Initialize sliding LID detector.

        Args:
            window_size: Size of sliding window in seconds (default: 0.9s)
        """
        self.window_size = window_size
        self.detections: List[LanguageDetection] = []
        self.current_audio_position = 0.0

        logger.info(f"[SlidingLID] Initialized with window_size={window_size}s")

    def add_detection(self, language: str, confidence: float, audio_position: float):
        """
        Add a language detection to the sliding window.

        Args:
            language: Language code (e.g., 'en', 'zh', 'es')
            confidence: Detection confidence (0.0-1.0)
            audio_position: Position in audio stream (seconds)
        """
        timestamp = time.monotonic()

        detection = LanguageDetection(
            language=language,
            confidence=confidence,
            timestamp=timestamp,
            audio_position=audio_position
        )

        self.detections.append(detection)
        self.current_audio_position = audio_position

        # Purge old detections (outside sliding window)
        self._purge_old_detections()

        logger.debug(f"[SlidingLID] Added: {detection}, window has {len(self.detections)} detections")

    def _purge_old_detections(self):
        """Remove detections older than window_size"""
        if not self.detections:
            return

        current_time = time.monotonic()
        cutoff_time = current_time - self.window_size

        # Keep detections within window
        before_count = len(self.detections)
        self.detections = [d for d in self.detections if d.timestamp >= cutoff_time]
        after_count = len(self.detections)

        if before_count != after_count:
            logger.debug(f"[SlidingLID] Purged {before_count - after_count} old detections, "
                        f"{after_count} remain in window")

    def get_current_language(self) -> Optional[str]:
        """
        Get the current language (majority in sliding window).

        Returns:
            Most frequent language in window, or None if window empty
        """
        self._purge_old_detections()

        if not self.detections:
            return None

        # Count languages in window
        language_counts = Counter(d.language for d in self.detections)

        # Return most common language
        most_common = language_counts.most_common(1)[0]
        language = most_common[0]
        count = most_common[1]

        logger.debug(f"[SlidingLID] Current language: {language} ({count}/{len(self.detections)} detections)")

        return language

    def get_sustained_language(self, min_duration: float) -> Optional[str]:
        """
        Check if a language has been sustained for at least min_duration.

        This checks if ALL recent detections (within window) are the same language
        AND the span covers at least min_duration.

        Args:
            min_duration: Minimum duration in seconds (e.g., 2.5s)

        Returns:
            Language code if sustained, None otherwise
        """
        self._purge_old_detections()

        if not self.detections:
            return None

        # Check if all detections in window are same language
        languages = set(d.language for d in self.detections)
        if len(languages) > 1:
            # Mixed languages in window - not sustained
            return None

        # All same language - check duration
        sustained_language = languages.pop()

        # Calculate span of detections
        min_position = min(d.audio_position for d in self.detections)
        max_position = max(d.audio_position for d in self.detections)
        duration = max_position - min_position

        if duration >= min_duration:
            logger.debug(f"[SlidingLID] Sustained language: {sustained_language} "
                        f"({duration:.2f}s >= {min_duration}s)")
            return sustained_language
        else:
            logger.debug(f"[SlidingLID] Language {sustained_language} not sustained "
                        f"({duration:.2f}s < {min_duration}s)")
            return None

    def get_window_stats(self) -> dict:
        """
        Get statistics about current sliding window.

        Returns:
            Dictionary with window stats (for debugging/monitoring)
        """
        self._purge_old_detections()

        if not self.detections:
            return {
                'detection_count': 0,
                'languages': [],
                'window_size': self.window_size
            }

        language_counts = Counter(d.language for d in self.detections)

        return {
            'detection_count': len(self.detections),
            'languages': dict(language_counts),
            'current_language': self.get_current_language(),
            'window_size': self.window_size,
            'audio_position': self.current_audio_position
        }

    def reset(self):
        """Clear all detections (for new session)"""
        self.detections.clear()
        self.current_audio_position = 0.0
        logger.info("[SlidingLID] Reset - cleared all detections")
