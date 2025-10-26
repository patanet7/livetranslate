#!/usr/bin/env python3
"""
VAD (Voice Activity Detection) Processing

Handles VAD state management and speech detection for audio chunks.
Extracted from whisper_service.py for better modularity and testability.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class VADProcessor:
    """
    Manages VAD state and processes audio chunks for speech detection.

    Handles:
    - Session-specific VAD state tracking
    - VAD preloading with silence frames
    - Speech start/end event detection
    - Chunk filtering based on VAD results
    """

    def __init__(self, vad):
        """
        Initialize VAD processor.

        Args:
            vad: VAD detector instance (SileroVAD or similar)
        """
        self.vad = vad
        self.session_vad_states: Dict[str, Optional[str]] = {}

    def process_chunk(self, audio_chunk: np.ndarray, session_id: str) -> bool:
        """
        Process audio chunk with VAD and determine if it should be kept.

        Args:
            audio_chunk: Audio data as numpy array
            session_id: Session identifier for state tracking

        Returns:
            True if chunk should be kept (contains speech), False if should be discarded
        """
        if self.vad is None:
            # No VAD available - accept all chunks
            return True

        # Check for speech in the chunk
        vad_result = self.vad.check_speech(audio_chunk)

        # Initialize session VAD state if needed
        if session_id not in self.session_vad_states:
            self._initialize_session_vad(session_id)

        # Update VAD state based on result
        if vad_result is not None:
            return self._handle_vad_event(vad_result, session_id)
        else:
            return self._handle_no_vad_event(session_id)

    def _initialize_session_vad(self, session_id: str):
        """
        Initialize VAD state for a new session.

        Preloads VAD with silence frames to build analysis window.
        This "primes" the VAD so it's ready to detect speech immediately.
        SimulStreaming does this implicitly - we do it explicitly.

        Args:
            session_id: Session identifier
        """
        self.session_vad_states[session_id] = None
        logger.info(f"[VAD] Session {session_id}: Initialized VAD state tracking")

        # CRITICAL FIX: Preload VAD with silence to build analysis window
        silence_frames = 3  # Preload 3 frames of silence (3 * 512 samples = 1536 samples = ~96ms)
        silence_chunk = np.zeros(512, dtype=np.float32)
        for _ in range(silence_frames):
            self.vad.check_speech(silence_chunk)

        logger.info(f"[VAD] Session {session_id}: 🔧 Preloaded VAD with {silence_frames} silence frames for immediate readiness")

    def _handle_vad_event(self, vad_result: Dict[str, Any], session_id: str) -> bool:
        """
        Handle explicit VAD start/end events.

        Args:
            vad_result: VAD result dict with 'start' or 'end' keys
            session_id: Session identifier

        Returns:
            True if chunk should be kept, False otherwise
        """
        if 'start' in vad_result:
            self.session_vad_states[session_id] = 'voice'
            logger.info(f"[VAD] Session {session_id}: 🎤 Speech started at {vad_result['start']:.2f}s")
        elif 'end' in vad_result:
            self.session_vad_states[session_id] = 'nonvoice'
            logger.info(f"[VAD] Session {session_id}: 🔇 Speech ended at {vad_result['end']:.2f}s")

        # Only keep chunks when in voice state
        current_state = self.session_vad_states.get(session_id)
        if current_state != 'voice':
            logger.info(f"[VAD] Session {session_id}: ❌ Discarding chunk (state: {current_state}, vad_result: {vad_result})")
            return False

        return True

    def _handle_no_vad_event(self, session_id: str) -> bool:
        """
        Handle case when VAD returns None (no explicit start/end event).

        VAD returned None could mean:
        1. ONGOING speech (between start and end events)
        2. VAD is still accumulating data (first few chunks)
        3. Silence after speech has ended

        Args:
            session_id: Session identifier

        Returns:
            True if chunk should be kept, False otherwise
        """
        current_state = self.session_vad_states.get(session_id)

        if current_state == 'voice':
            # Still in speech - vad_result is None but we're between start and end
            logger.debug(f"[VAD] Session {session_id}: ✅ Ongoing speech (vad_result=None, state=voice)")
            return True

        elif current_state is None:
            # CRITICAL FIX: First chunks - VAD needs data to accumulate
            # ACCEPT these chunks so VAD can build up its analysis window
            logger.debug(f"[VAD] Session {session_id}: ✅ Accepting initial chunk for VAD analysis (state=None)")
            return True

        elif current_state == 'nonvoice':
            # Confirmed non-voice state and still no speech detected - discard
            logger.info(f"[VAD] Session {session_id}: ❌ Discarding chunk (state: nonvoice, vad_result: None)")
            return False

        # Default: accept chunk
        return True

    def clear_session(self, session_id: str):
        """
        Clear VAD state for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.session_vad_states:
            del self.session_vad_states[session_id]
            logger.info(f"[VAD] Session {session_id}: Cleared VAD state")
