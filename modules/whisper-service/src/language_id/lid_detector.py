#!/usr/bin/env python3
"""
Frame-Level Language ID Detector

Per FEEDBACK.md lines 32-38:
- 80-120ms hop for frame-level LID
- Use lightweight LID model (MMS-LID recommended)
- Fast inference for real-time processing

This implementation provides a base that can use:
1. Whisper's built-in language detection (current)
2. MMS-LID (future - requires ONNX export for speed)
3. XLSR-based LID (alternative)
"""

import numpy as np
import torch
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LIDFrame:
    """Single LID detection result for a frame"""
    timestamp: float  # Frame timestamp in seconds
    language: str  # Detected language code
    probabilities: Dict[str, float]  # Per-language probabilities
    confidence: float  # Confidence of top prediction


class FrameLevelLID:
    """
    Frame-level language identification at 80-120ms resolution.

    Per FEEDBACK.md lines 32-38, 202-212.

    Args:
        hop_ms: Frame hop in milliseconds (default 100ms = 10Hz)
        sample_rate: Audio sample rate (default 16000Hz)
        target_languages: List of target languages to detect
        smoothing: Apply median smoothing to reduce flapping
    """

    def __init__(
        self,
        hop_ms: int = 100,
        sample_rate: int = 16000,
        target_languages: Optional[List[str]] = None,
        smoothing: bool = True
    ):
        self.hop_ms = hop_ms
        self.sample_rate = sample_rate
        self.hop_samples = int((hop_ms / 1000) * sample_rate)  # Samples per hop
        self.target_languages = target_languages or ['en', 'zh']
        self.smoothing = smoothing

        # Detection history for smoothing
        self.detection_history: List[LIDFrame] = []
        self.max_history = 50  # Keep last 5 seconds at 10Hz

        logger.info(
            f"FrameLevelLID initialized: hop={hop_ms}ms ({self.hop_samples} samples), "
            f"languages={self.target_languages}, smoothing={smoothing}"
        )

    def detect(
        self,
        audio_chunk: np.ndarray,
        timestamp: float,
        model=None
    ) -> LIDFrame:
        """
        Detect language for a single audio frame.

        Args:
            audio_chunk: Audio data (numpy array, float32)
            timestamp: Timestamp of this frame in seconds
            model: Optional Whisper model for language detection

        Returns:
            LIDFrame with detected language and probabilities
        """
        # For now, use a simple stub that returns English
        # TODO: Integrate MMS-LID or Whisper language detection

        if model is not None:
            # Use Whisper's built-in language detection
            probs = self._detect_with_whisper(audio_chunk, model)
        else:
            # Fallback: uniform distribution (placeholder)
            probs = {lang: 1.0 / len(self.target_languages)
                    for lang in self.target_languages}

        # Find top language
        top_lang = max(probs, key=probs.get)
        confidence = probs[top_lang]

        # Create LID frame
        frame = LIDFrame(
            timestamp=timestamp,
            language=top_lang,
            probabilities=probs,
            confidence=confidence
        )

        # Add to history
        self.detection_history.append(frame)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        # Apply smoothing if enabled
        if self.smoothing and len(self.detection_history) >= 3:
            frame = self._apply_median_smoothing(frame)

        return frame

    def _detect_with_whisper(
        self,
        audio_chunk: np.ndarray,
        model
    ) -> Dict[str, float]:
        """
        Use Whisper's language detection on audio chunk.

        Note: This is expensive for 10Hz operation. In production,
        replace with lightweight MMS-LID model.
        """
        try:
            # Convert to torch tensor
            if isinstance(audio_chunk, np.ndarray):
                audio_tensor = torch.from_numpy(audio_chunk).float()
            else:
                audio_tensor = audio_chunk.float()

            # Ensure correct shape
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Run language detection (simplified - would need full Whisper API)
            # For now, return uniform distribution
            # TODO: Call model.lang_id() properly

            probs = {lang: 1.0 / len(self.target_languages)
                    for lang in self.target_languages}

            return probs

        except Exception as e:
            logger.warning(f"Whisper LID failed: {e}, using fallback")
            return {lang: 1.0 / len(self.target_languages)
                   for lang in self.target_languages}

    def _apply_median_smoothing(self, current_frame: LIDFrame) -> LIDFrame:
        """
        Apply median filtering to reduce language flapping.

        Per FEEDBACK.md line 37: "Smooth with Viterbi or hysteresis"

        Uses majority vote over last 3 frames for stability.
        """
        # Get last 3 frames
        recent_frames = self.detection_history[-3:]

        # Count language votes
        language_votes: Dict[str, int] = {}
        for frame in recent_frames:
            lang = frame.language
            language_votes[lang] = language_votes.get(lang, 0) + 1

        # Get majority language
        majority_lang = max(language_votes, key=language_votes.get)

        # If majority differs from current, use majority (smoothing effect)
        if majority_lang != current_frame.language:
            logger.debug(
                f"LID smoothing: {current_frame.language} → {majority_lang} "
                f"(votes: {language_votes})"
            )
            # Update frame with smoothed language
            current_frame = LIDFrame(
                timestamp=current_frame.timestamp,
                language=majority_lang,
                probabilities=current_frame.probabilities,
                confidence=current_frame.confidence * 0.9  # Slightly reduce confidence for smoothed
            )

        return current_frame

    def get_recent_detections(self, window_sec: float = 1.0) -> List[LIDFrame]:
        """
        Get recent LID detections within time window.

        Args:
            window_sec: Time window in seconds

        Returns:
            List of LIDFrame within window
        """
        if not self.detection_history:
            return []

        latest_time = self.detection_history[-1].timestamp
        cutoff_time = latest_time - window_sec

        return [
            frame for frame in self.detection_history
            if frame.timestamp >= cutoff_time
        ]

    def get_current_language(self) -> Optional[str]:
        """Get most recent detected language"""
        if not self.detection_history:
            return None
        return self.detection_history[-1].language

    def reset(self):
        """Reset detection history"""
        self.detection_history.clear()
        logger.debug("LID detection history reset")


# TODO: Future enhancement - MMS-LID integration
class MMSLID(FrameLevelLID):
    """
    MMS-LID based language identification (future).

    Per FEEDBACK.md line 37: "MMS-LID works and is fast"

    Requires:
    - Download MMS-LID model from Hugging Face
    - Export to ONNX for fast inference (100Hz target)
    - Integrate with frame-level detection pipeline
    """

    def __init__(self, model_name: str = "facebook/mms-lid-126", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        # TODO: Load MMS-LID model and export to ONNX
        logger.warning(
            "MMS-LID not yet implemented - using fallback LID. "
            "For production, integrate MMS-LID with ONNX export."
        )
