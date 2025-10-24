#!/usr/bin/env python3
"""
VAC Online ASR Processor - Computationally Aware Chunking

Following SimulStreaming reference implementation:
- Wraps Whisper transcription with Voice Activity Controller (VAC)
- Small VAD chunks (0.04s) for fast speech detection
- Large Whisper chunks (1.2s) for quality transcription
- Adaptive processing: only when buffer full OR speech ends
- During silence: buffer only (saves compute!)

Reference: SimulStreaming/whisper_streaming/vac_online_processor.py
License: MIT

This is how SimulStreaming achieves computational efficiency!
"""

import numpy as np
import torch
import logging
import time
from typing import Optional, Dict, Any
from silero_vad_iterator import FixedVADIterator
from sentence_segmenter import SentenceSegmenter
from text_language_detector import TextLanguageDetector

logger = logging.getLogger(__name__)

# Helper functions for numpy/torch conversion
def to_torch_tensor(audio):
    """Convert numpy array or torch tensor to torch tensor (on CPU initially)

    Following SimulStreaming pattern - audio MUST be 1D flattened tensor!
    SimulStreaming concatenates segments with torch.cat(self.segments, dim=0)
    which requires all segments to be 1D tensors.
    """
    if isinstance(audio, np.ndarray):
        tensor = torch.from_numpy(audio)
    elif isinstance(audio, torch.Tensor):
        tensor = audio.cpu()  # Ensure on CPU for VAD operations
    else:
        raise TypeError(f"Expected numpy array or torch tensor, got {type(audio)}")

    # CRITICAL: Ensure 1D tensor (SimulStreaming requirement)
    # Reference: simul_whisper.py line 347: torch.cat(self.segments, dim=0)
    if tensor.ndim > 1:
        tensor = tensor.flatten()

    return tensor

def to_numpy(audio):
    """Convert torch tensor or numpy array to numpy (handling device placement)"""
    if isinstance(audio, torch.Tensor):
        return audio.cpu().numpy()  # Move to CPU first if on GPU
    elif isinstance(audio, np.ndarray):
        return audio
    else:
        raise TypeError(f"Expected torch tensor or numpy array, got {type(audio)}")


class VACOnlineASRProcessor:
    """
    Voice Activity Controller (VAC) for Online ASR Processing

    Wraps Whisper transcription with intelligent VAD-based chunking:
    - Detects speech with small VAD chunks (0.04s)
    - Processes with large Whisper chunks (1.2s)
    - Only runs Whisper when necessary (buffer full OR speech ends)
    - Saves compute during silence

    Parameters:
        online_chunk_size (float): Whisper chunk duration in seconds (default 1.2s)
        vad_threshold (float): Speech probability threshold (default 0.5)
        min_buffered_length (float): Minimum buffer length in seconds (default 1.0s)
        sampling_rate (int): Audio sampling rate (default 16000 Hz)

    Usage:
        vac_processor = VACOnlineASRProcessor(
            online_chunk_size=1.2,
            vad_threshold=0.5
        )

        # Initialize VAD and Whisper
        vac_processor.init(model_manager)

        # Process streaming audio
        for audio_chunk in audio_stream:
            vac_processor.insert_audio_chunk(audio_chunk)
            result = vac_processor.process_iter()
            if result and 'text' in result:
                print(f"Transcription: {result['text']}")
    """

    def __init__(
        self,
        online_chunk_size: float = 1.2,
        vad_threshold: float = 0.5,
        vad_min_speech_ms: int = 120,  # Phase 2: Configurable min speech duration
        vad_min_silence_ms: int = 500,  # Phase 2: Min silence (500ms matches SimulStreaming default)
        sliding_lid_window: float = 0.9,  # Phase 3: Sliding LID window size
        min_buffered_length: float = 1.0,
        sampling_rate: int = 16000
    ):
        self.online_chunk_size = online_chunk_size
        self.vad_threshold = vad_threshold
        self.vad_min_speech_ms = vad_min_speech_ms  # Phase 2
        self.vad_min_silence_ms = vad_min_silence_ms  # Phase 2
        self.sliding_lid_window = sliding_lid_window  # Phase 3
        self.min_buffered_length = min_buffered_length
        self.SAMPLING_RATE = sampling_rate

        # VAD and Whisper components
        self.vad = None
        self.model = None
        self.model_manager = None

        # Sentence segmentation for better draft/final detection
        self.sentence_segmenter = SentenceSegmenter()

        # Phase 3: Sliding LID detector for language tracking
        from sliding_lid_detector import SlidingLIDDetector
        self.lid_detector = SlidingLIDDetector(window_size=sliding_lid_window)

        # Phase 5: Token deduplicator for chunk boundary handling
        from token_deduplicator import TokenDeduplicator
        self.token_deduplicator = TokenDeduplicator(lookback_tokens=10)

        # Phase 5: UTF-8 boundary fixer for multi-byte character cleanup
        from utf8_boundary_fixer import UTF8BoundaryFixer
        self.utf8_fixer = UTF8BoundaryFixer()

        # Phase 4: Sustained language detection for SOT reset
        self.current_sustained_language: Optional[str] = None  # Track sustained language
        self.language_start_time: float = 0.0  # When current language was first detected
        self.last_sot_reset_time: float = 0.0  # For cooldown mechanism (5s)
        self.silence_start_time: Optional[float] = None  # Track silence duration
        self.sustained_language_threshold: float = 2.5  # Minimum duration for sustained detection (2.5-3.0s)
        self.sot_reset_cooldown: float = 5.0  # Minimum time between SOT resets
        self.min_silence_for_reset: float = 0.25  # Minimum 250ms silence required for reset

        # Audio buffers (using torch tensors throughout for SimulStreaming compatibility)
        self.audio_buffer = torch.tensor([], dtype=torch.float32)

        # CRITICAL FIX: Match SimulStreaming's chunk accumulation pattern!
        # SimulStreaming accumulates chunks in a LIST, then concatenates before insert_audio()
        # Reference: simulstreaming_whisper.py lines 151-152, 207-217
        self.audio_chunks = []  # List of torch tensors (like SimulStreaming's audio_chunks)
        self.current_online_chunk_buffer_size = 0  # Track total samples in audio_chunks

        # State tracking
        self.status = 'nonvoice'  # 'voice' or 'nonvoice'
        self.is_currently_final = False

        # Hybrid Tracking: Combine SimulStreaming attention + vexa timestamps
        self.audio_start_time = 0.0  # Session start (seconds from session beginning)
        self.audio_end_time = 0.0    # Latest chunk end (seconds)
        self.frames_to_time_offset = 0.0  # Mapping between frames and absolute time
        self.total_chunks_received = 0  # Total chunks received
        self.all_chunks_received = False  # Client signaled end of stream
        self.TOKENS_PER_SECOND = 50  # Whisper mel-spectrogram tokens per second

        # Statistics
        self.vad_checks = 0
        self.whisper_calls = 0
        self.total_audio_processed = 0

        logger.info(
            f"VACOnlineASRProcessor initialized: "
            f"chunk_size={online_chunk_size}s, "
            f"vad_threshold={vad_threshold}, "
            f"sampling_rate={sampling_rate}Hz"
        )

    def init(self, model_manager, model_name: str = "large-v3"):
        """
        Initialize VAD and Whisper models

        Args:
            model_manager: ModelManager instance
            model_name: Whisper model to load (default "large-v3")
        """
        logger.info(f"Initializing VACOnlineASRProcessor with model {model_name}...")

        # Load Silero VAD
        try:
            vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self.vad = FixedVADIterator(
                model=vad_model,
                threshold=self.vad_threshold,
                sampling_rate=self.SAMPLING_RATE,
                min_speech_duration_ms=self.vad_min_speech_ms,  # Phase 2
                min_silence_duration_ms=self.vad_min_silence_ms  # Phase 2
            )
            logger.info(f"✅ Silero VAD initialized (threshold={self.vad_threshold}, "
                       f"min_speech={self.vad_min_speech_ms}ms, min_silence={self.vad_min_silence_ms}ms)")
        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}")
            raise

        # Load Whisper model
        try:
            self.model_manager = model_manager
            self.model = model_manager.load_model(model_name)
            logger.info(f"✅ Whisper model '{model_name}' initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise

        logger.info("✅ VACOnlineASRProcessor ready for streaming")

    def insert_audio_chunk(self, audio_chunk: np.ndarray, chunk_metadata: Optional[Dict[str, Any]] = None):
        """
        Insert audio chunk and run VAD detection

        CRITICAL FIX: Match SimulStreaming's insert_audio_chunk() pattern!
        Reference: simulstreaming_whisper.py lines 151-152

        This method should ONLY:
        1. Buffer audio in self.audio_buffer
        2. Run VAD detection
        3. Update status flags

        NO PROCESSING happens here! All processing occurs in process_iter().

        Args:
            audio_chunk: Audio data (numpy array, float32)
            chunk_metadata: Hybrid tracking metadata (timestamps, indices, etc.)
        """
        # Convert to torch tensor immediately at entry point
        audio_tensor = to_torch_tensor(audio_chunk)

        self.total_audio_processed += len(audio_chunk)

        # Hybrid Tracking: Update timestamp tracking from chunk metadata
        if chunk_metadata:
            chunk_start_time = chunk_metadata.get('audio_start_time', self.audio_end_time)
            chunk_end_time = chunk_metadata.get('audio_end_time', chunk_start_time + len(audio_chunk)/self.SAMPLING_RATE)

            # Update audio timeline
            if self.total_chunks_received == 0:
                self.audio_start_time = chunk_start_time
            self.audio_end_time = max(self.audio_end_time, chunk_end_time)
            self.total_chunks_received += 1

            # Check if this is the last chunk
            if chunk_metadata.get('is_last_chunk', False):
                self.all_chunks_received = True
                logger.info(f"[HYBRID] Received LAST chunk (total: {self.total_chunks_received})")

            logger.debug(
                f"[HYBRID] Chunk {self.total_chunks_received}: "
                f"{chunk_start_time:.2f}s - {chunk_end_time:.2f}s"
            )
        else:
            # Fallback: estimate timestamps from sample count
            chunk_duration = len(audio_chunk) / self.SAMPLING_RATE
            self.audio_end_time += chunk_duration
            self.total_chunks_received += 1

        # Run VAD detection (VAD needs numpy)
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.numpy()
        vad_result = self.vad(audio_chunk, return_seconds=False)
        self.vad_checks += 1

        # Append to audio buffer (torch operations)
        self.audio_buffer = torch.cat([self.audio_buffer, audio_tensor])

        # Handle VAD events - ONLY update status, NO processing!
        if vad_result is not None:
            if 'start' in vad_result:
                # Speech start detected
                self.status = 'voice'
                # Phase 4: Reset silence timer on speech start
                self.silence_start_time = None
                logger.debug(f"VAD: Speech START detected (sample {vad_result['start']})")

            elif 'end' in vad_result:
                # Speech end detected
                self.status = 'nonvoice'
                self.is_currently_final = True
                # Phase 4: Start tracking silence duration
                self.silence_start_time = time.time()
                logger.debug(f"VAD: Speech END detected (sample {vad_result['end']}), silence timer started")

        else:
            # No VAD event - maintain current status
            if self.status != 'voice':
                # During silence: just buffer (NO processing!)
                # Limit buffer to prevent OOM (keep only 1 second)
                max_silence_buffer = self.SAMPLING_RATE * 1.0
                if len(self.audio_buffer) > max_silence_buffer:
                    # Keep only most recent 1 second
                    self.audio_buffer = self.audio_buffer[-int(max_silence_buffer):]

                logger.debug(
                    f"VAD: Silence buffering (buffer size: {len(self.audio_buffer)} samples)"
                )

    def _send_audio_to_online_processor(self, audio: torch.Tensor):
        """
        Accumulate audio chunks for later processing (SimulStreaming pattern)

        CRITICAL FIX: Match SimulStreaming's accumulate→concatenate→insert pattern!
        Reference: simulstreaming_whisper.py line 151-152

        SimulStreaming pattern:
        1. insert_audio_chunk(): Just append to audio_chunks[] list (NO processing)
        2. process_iter(): Concatenate audio_chunks → insert_audio() ONCE → infer()

        We were incorrectly calling insert_audio() immediately for each chunk!
        """
        logger.info(f"[ACCUMULATE] Adding {len(audio)} samples ({len(audio)/16000:.2f}s) to audio_chunks list")

        # CORRECT SimulStreaming pattern: Accumulate chunks in a list
        # Reference: simulstreaming_whisper.py line 152: self.audio_chunks.append(torch.from_numpy(audio))
        self.audio_chunks.append(audio)

        # Track total samples accumulated since last infer() call
        self.current_online_chunk_buffer_size += len(audio)

        logger.info(f"[ACCUMULATED] Total chunks: {len(self.audio_chunks)}, Total samples: {self.current_online_chunk_buffer_size} ({self.current_online_chunk_buffer_size/16000:.2f}s)")

    def _should_reset_sot(self, detected_language: Optional[str]) -> bool:
        """
        Phase 4: Determine if SOT (Start of Transcript) should be reset

        Conditions for SOT reset:
        1. Language sustained for >= 2.5s (sustained_language_threshold)
        2. VAD silence >= 250ms (min_silence_for_reset)
        3. Cooldown allows (>= 5s since last reset)
        4. Language actually changed from previous sustained language

        Args:
            detected_language: Current detected language from Whisper

        Returns:
            True if SOT should be reset, False otherwise
        """
        if not detected_language:
            return False

        current_time = time.time()

        # Check 1: Is this a new language or continuation of current?
        if detected_language != self.current_sustained_language:
            # New language detected - start tracking
            logger.info(f"[SUSTAINED_LID] Language change detected: "
                       f"{self.current_sustained_language or 'None'} → {detected_language}")
            self.current_sustained_language = detected_language
            self.language_start_time = current_time
            return False  # Not sustained yet (just started)

        # Check 2: Has language been sustained long enough?
        language_duration = current_time - self.language_start_time
        if language_duration < self.sustained_language_threshold:
            logger.debug(f"[SUSTAINED_LID] Language '{detected_language}' duration: "
                        f"{language_duration:.2f}s < {self.sustained_language_threshold}s (not sustained yet)")
            return False

        # Check 3: Is there sufficient VAD silence?
        if self.silence_start_time is None:
            logger.debug(f"[SUSTAINED_LID] No VAD silence detected (silence_start_time=None)")
            return False

        silence_duration = current_time - self.silence_start_time
        if silence_duration < self.min_silence_for_reset:
            logger.debug(f"[SUSTAINED_LID] Silence duration: "
                        f"{silence_duration:.2f}s < {self.min_silence_for_reset}s (insufficient)")
            return False

        # Check 4: Cooldown check - prevent too frequent resets
        time_since_last_reset = current_time - self.last_sot_reset_time
        if time_since_last_reset < self.sot_reset_cooldown:
            logger.info(f"[SUSTAINED_LID] SOT reset blocked by cooldown: "
                       f"{time_since_last_reset:.2f}s < {self.sot_reset_cooldown}s")
            return False

        # All conditions met!
        logger.info(f"[SUSTAINED_LID] ✅ SOT reset conditions MET:")
        logger.info(f"  - Language '{detected_language}' sustained for {language_duration:.2f}s (>= {self.sustained_language_threshold}s)")
        logger.info(f"  - Silence duration: {silence_duration:.2f}s (>= {self.min_silence_for_reset}s)")
        logger.info(f"  - Time since last reset: {time_since_last_reset:.2f}s (>= {self.sot_reset_cooldown}s)")

        return True

    def process_iter(self) -> Dict[str, Any]:
        """
        Process iteration - decides when to run Whisper

        CRITICAL FIX: Match SimulStreaming's process_iter() pattern!
        Reference: simulstreaming_whisper.py lines 207-217

        This is the CORE ADAPTIVE LOGIC:
        - If speech ended (is_currently_final): Move buffer to audio_chunks, then process
        - If buffer >= online_chunk_size (1.2s): Move buffer to audio_chunks, then process
        - Otherwise: NO processing (saves compute!)

        Returns:
            - Transcription result if processing occurred
            - Empty dict if still buffering
        """
        # Check if we should process
        if self.is_currently_final:
            # Speech ended - move buffer to audio_chunks, then process final chunk
            logger.info("Processing FINAL chunk (speech ended)")
            if len(self.audio_buffer) > 0:
                self._send_audio_to_online_processor(self.audio_buffer)
                self.audio_buffer = torch.tensor([], dtype=torch.float32)
            return self._finish()

        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            # Accumulated 1.2s+ audio in processor - run inference (matches SimulStreaming line 99)
            # CRITICAL FIX: Check current_online_chunk_buffer_size, not len(audio_buffer)!
            # This triggers processing every 1.2s REGARDLESS of VAD state (critical for code-switching!)
            logger.info(
                f"Processing chunk (accumulated {self.current_online_chunk_buffer_size / self.SAMPLING_RATE:.2f}s > "
                f"{self.online_chunk_size}s in audio_chunks)"
            )
            # Note: audio_buffer should already be empty (sent by VAD in add_chunk)
            # If not empty, send it now before processing
            if len(self.audio_buffer) > 0:
                logger.info(f"Flushing remaining audio_buffer: {len(self.audio_buffer)} samples")
                self._send_audio_to_online_processor(self.audio_buffer)
                self.audio_buffer = torch.tensor([], dtype=torch.float32)
            return self._process_online_chunk()

        else:
            # Still buffering - NO processing yet!
            logger.debug(
                f"Buffering... {len(self.audio_buffer)} samples "
                f"({len(self.audio_buffer) / self.SAMPLING_RATE:.2f}s, status: {self.status})"
            )
            return {}

    def _process_online_chunk(self) -> Dict[str, Any]:
        """
        Process online chunk with SimulStreaming's concatenate→insert→infer pattern

        CRITICAL FIX: Match SimulStreaming's process_iter() pattern!
        Reference: simulstreaming_whisper.py lines 207-217

        SimulStreaming pattern:
        1. Concatenate ALL accumulated chunks: audio = torch.cat(self.audio_chunks, dim=0)
        2. Clear chunk list: self.audio_chunks = []
        3. Insert concatenated audio ONCE: self.model.insert_audio(audio)
        4. Run inference: self.model.infer(is_last=False)
        """
        if self.model is None:
            logger.error("Model not initialized")
            return {}

        try:
            # Step 1: Concatenate ALL accumulated chunks
            # Reference: simulstreaming_whisper.py line 211
            if len(self.audio_chunks) == 0:
                logger.warning("[PROCESS] No audio chunks to process!")
                return {}

            logger.info(f"[CONCATENATE] Concatenating {len(self.audio_chunks)} chunks into single tensor")
            audio = torch.cat(self.audio_chunks, dim=0)  # CONCATENATE FIRST
            logger.info(f"[CONCATENATE] Result: {len(audio)} samples ({len(audio)/16000:.2f}s)")

            # Step 2: Clear chunk list
            # Reference: simulstreaming_whisper.py line 213
            self.audio_chunks = []
            self.current_online_chunk_buffer_size = 0

            # Step 3: Insert concatenated audio ONCE
            # Reference: simulstreaming_whisper.py line 215
            logger.info(f"[INSERT_AUDIO] Sending concatenated audio to model: shape={audio.shape}, dtype={audio.dtype}")
            logger.info(f"[INSERT_AUDIO] Audio range: [{audio.min():.4f}, {audio.max():.4f}], mean={audio.mean():.4f}, std={audio.std():.4f}")
            self.model.insert_audio(audio)

            # Step 4: Run inference
            # Reference: simulstreaming_whisper.py line 217
            logger.info(f"[INFER] Calling model.infer(is_last=False)")
            tokens, generation_progress = self.model.infer(is_last=False)

            # Phase 5: Deduplicate tokens at chunk boundaries
            logger.info(f"[INFER] Generated {len(tokens)} tokens (before dedup): {tokens[:20] if len(tokens) > 20 else tokens}")
            tokens = self.token_deduplicator.deduplicate(tokens)
            logger.info(f"[DEDUP] After deduplication: {len(tokens)} tokens")

            # Decode tokens to text
            logger.info(f"[INFER] Generation progress: {generation_progress}")
            text = self.model.tokenizer.decode(tokens)

            # Phase 5: Fix incomplete UTF-8 characters at chunk boundaries
            text = self.utf8_fixer.fix_boundaries(text)

            self.whisper_calls += 1

            # Check for sentence boundary (complete sentence = is_final)
            # This provides much better draft/final distinction than VAD-only
            is_sentence_end = self.sentence_segmenter.is_sentence_end(text)
            is_final = is_sentence_end

            if is_sentence_end:
                logger.info(f"✅ Complete sentence detected: '{text}' ({len(tokens)} tokens) [is_final=True]")
            else:
                logger.info(f"📝 Incomplete sentence: '{text}' ({len(tokens)} tokens) [is_final=False]")

            # Phase 3: Track language detection in sliding window
            detected_language = None
            if hasattr(self.model, 'detected_language') and self.model.detected_language:
                detected_language = self.model.detected_language
                # Add detection to sliding window
                # Audio position: total samples processed / sample rate
                audio_position = self.total_audio_processed / self.SAMPLING_RATE
                self.lid_detector.add_detection(
                    language=detected_language,
                    confidence=1.0,  # Whisper doesn't provide confidence, use 1.0
                    audio_position=audio_position
                )
                logger.info(f"[LID] Detected language: {detected_language} at {audio_position:.2f}s")

            # Get current language from sliding window (for SOT reset only)
            current_language = self.lid_detector.get_current_language()
            logger.info(f"[LID] Per-chunk: {detected_language}, Window average: {current_language}")

            # Hybrid Tracking: Extract attention tracking from generation_progress
            most_attended_frame = 0
            content_mel_len = 0
            is_caught_up = False

            if generation_progress and isinstance(generation_progress, dict):
                # Extract from top-level generation dict
                content_mel_len = generation_progress.get('frames_len', 0)
                frame_threshold = generation_progress.get('frames_threshold', 4)

                # most_attended_frame is in generation["progress"] list (per-iteration data)
                # Extract from generation["progress"][-1]["most_attended_frames"][0] (last iteration, first beam)
                if 'progress' in generation_progress and len(generation_progress['progress']) > 0:
                    last_iteration = generation_progress['progress'][-1]
                    if 'most_attended_frames' in last_iteration:
                        most_attended_frames_list = last_iteration['most_attended_frames']
                        if isinstance(most_attended_frames_list, list) and len(most_attended_frames_list) > 0:
                            most_attended_frame = most_attended_frames_list[0]  # First beam

                # Check if decoder caught up to available audio
                is_caught_up = (content_mel_len - most_attended_frame) <= frame_threshold

            # Hybrid Tracking: Convert frames to absolute time
            processed_through_time = self.frames_to_time_offset + (most_attended_frame / self.TOKENS_PER_SECOND)

            # Hybrid Tracking: Check if session is complete
            is_session_complete = is_caught_up and self.all_chunks_received

            logger.info(
                f"[HYBRID] Attention: frame {most_attended_frame}/{content_mel_len}, "
                f"caught_up={is_caught_up}, processed={processed_through_time:.2f}s, "
                f"received={self.audio_end_time:.2f}s, session_complete={is_session_complete}"
            )

            return {
                'text': text,
                'tokens': tokens,
                'generation_progress': generation_progress,
                'is_final': is_final,  # ⚠️ SENTENCE complete, NOT session complete!
                'detected_language': detected_language,  # Phase 3: Per-chunk from Whisper (for code-switching)

                # Hybrid Tracking: SimulStreaming attention tracking (internal precision)
                'attention_tracking': {
                    'most_attended_frame': most_attended_frame,
                    'content_mel_len': content_mel_len,
                    'is_caught_up': is_caught_up,
                },

                # Hybrid Tracking: vexa timestamp tracking (external correlation)
                'timestamp_tracking': {
                    'processed_through_time': processed_through_time,
                    'audio_received_through': self.audio_end_time,
                    'is_session_complete': is_session_complete,
                    'lag_seconds': self.audio_end_time - processed_through_time,
                },

                # vexa-style segment metadata (for deduplication)
                'absolute_start_time': processed_through_time - (len(tokens) * 0.02),  # Rough estimate
                'absolute_end_time': processed_through_time,
                'updated_at': time.time(),
            }

        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _finish(self) -> Dict[str, Any]:
        """
        Finish current utterance with SimulStreaming's concatenate→insert→infer pattern

        CRITICAL FIX: Same pattern as _process_online_chunk() but with is_last=True
        Reference: simulstreaming_whisper.py lines 207-217 (same pattern, different is_last flag)
        """
        if self.model is None:
            logger.error("Model not initialized")
            return {}

        try:
            # Step 1: Concatenate ALL accumulated chunks
            if len(self.audio_chunks) == 0:
                logger.info("[FINISH] No audio chunks to process in final chunk")
                self.is_currently_final = False
                return {}

            logger.info(f"[FINISH] Concatenating {len(self.audio_chunks)} final chunks into single tensor")
            audio = torch.cat(self.audio_chunks, dim=0)  # CONCATENATE FIRST
            logger.info(f"[FINISH] Result: {len(audio)} samples ({len(audio)/16000:.2f}s)")

            # Step 2: Clear chunk list
            self.audio_chunks = []
            self.current_online_chunk_buffer_size = 0

            # Step 3: Insert concatenated audio ONCE
            logger.info(f"[INSERT_AUDIO] Sending final concatenated audio to model: shape={audio.shape}, dtype={audio.dtype}")
            self.model.insert_audio(audio)

            # Step 4: Run FINAL inference with is_last=True
            logger.info(f"[INFER] Calling model.infer(is_last=True) for final chunk")
            tokens, generation_progress = self.model.infer(is_last=True)

            # Phase 5: Deduplicate tokens at chunk boundaries
            logger.info(f"[INFER] Generated {len(tokens)} tokens (before dedup)")
            tokens = self.token_deduplicator.deduplicate(tokens)
            logger.info(f"[DEDUP] After deduplication: {len(tokens)} tokens")

            # Decode tokens to text
            text = self.model.tokenizer.decode(tokens)

            # Phase 5: Fix incomplete UTF-8 characters at chunk boundaries
            text = self.utf8_fixer.fix_boundaries(text)

            self.whisper_calls += 1

            # VAD silence = always final (speech ended)
            # Even if no sentence terminal, this is a complete utterance
            logger.info(f"✅ Final stateful infer (VAD silence): '{text}' ({len(tokens)} tokens) [is_final=True]")

            # Check if it also has sentence boundary (for logging)
            is_sentence_end = self.sentence_segmenter.is_sentence_end(text)
            if is_sentence_end:
                logger.info(f"   ✅ Also ends with sentence terminal (clean boundary)")
            else:
                logger.info(f"   ⚠️  No sentence terminal (forced final by VAD)")

            # Phase 3: Track language detection in sliding window
            detected_language = None
            if hasattr(self.model, 'detected_language') and self.model.detected_language:
                detected_language = self.model.detected_language
                # Add detection to sliding window
                audio_position = self.total_audio_processed / self.SAMPLING_RATE
                self.lid_detector.add_detection(
                    language=detected_language,
                    confidence=1.0,
                    audio_position=audio_position
                )
                logger.info(f"[LID] Detected language: {detected_language} at {audio_position:.2f}s")

            # Get current language from sliding window (for SOT reset only)
            current_language = self.lid_detector.get_current_language()
            logger.info(f"[LID] FINAL - Per-chunk: {detected_language}, Window average: {current_language}")

            # Phase 4: Check if SOT should be reset for next utterance
            if self._should_reset_sot(detected_language):
                logger.info(f"[SUSTAINED_LID] 🔄 Resetting SOT for language '{detected_language}'")

                # Reset the model's decoder state (KV cache, accumulated tokens, etc.)
                # CRITICAL: Must call _clean_cache() to clear KV cache and avoid dimension mismatches
                if hasattr(self.model, '_clean_cache'):
                    self.model._clean_cache()
                    logger.info(f"[SUSTAINED_LID] ✅ KV cache cleared")
                elif hasattr(self.model, 'reset'):
                    self.model.reset()
                    logger.info(f"[SUSTAINED_LID] ✅ Model state reset complete")
                else:
                    logger.warning(f"[SUSTAINED_LID] ⚠️  Model has no _clean_cache() or reset() method")

                # Also reset initial tokens to force new SOT
                if hasattr(self.model, 'init_tokens'):
                    self.model.init_tokens()
                    logger.info(f"[SUSTAINED_LID] ✅ Initial tokens reset")

                # Phase 5: Reset token deduplicator on SOT reset (new decoder state = new token context)
                self.token_deduplicator.reset()
                logger.info(f"[SUSTAINED_LID] Token deduplicator reset")

                # Phase 5: Reset UTF-8 fixer on SOT reset (new segment = new boundary context)
                self.utf8_fixer.reset()
                logger.info(f"[SUSTAINED_LID] UTF-8 boundary fixer reset")

                # Update cooldown timer
                self.last_sot_reset_time = time.time()
                logger.info(f"[SUSTAINED_LID] Cooldown timer updated: {self.last_sot_reset_time}")

            # Reset VAC state
            self.is_currently_final = False

            # Hybrid Tracking: Extract attention tracking from generation_progress
            most_attended_frame = 0
            content_mel_len = 0
            is_caught_up = False

            if generation_progress and isinstance(generation_progress, dict):
                # Extract from top-level generation dict
                content_mel_len = generation_progress.get('frames_len', 0)
                frame_threshold = generation_progress.get('frames_threshold', 4)

                # most_attended_frame is in generation["progress"] list (per-iteration data)
                if 'progress' in generation_progress and len(generation_progress['progress']) > 0:
                    last_iteration = generation_progress['progress'][-1]
                    if 'most_attended_frames' in last_iteration:
                        most_attended_frames_list = last_iteration['most_attended_frames']
                        if isinstance(most_attended_frames_list, list) and len(most_attended_frames_list) > 0:
                            most_attended_frame = most_attended_frames_list[0]  # First beam

                is_caught_up = (content_mel_len - most_attended_frame) <= frame_threshold

            # Hybrid Tracking: Convert frames to absolute time
            processed_through_time = self.frames_to_time_offset + (most_attended_frame / self.TOKENS_PER_SECOND)

            # Hybrid Tracking: Check if session is complete (final chunk + caught up)
            is_session_complete = is_caught_up and self.all_chunks_received

            logger.info(
                f"[HYBRID] FINAL chunk - Attention: frame {most_attended_frame}/{content_mel_len}, "
                f"caught_up={is_caught_up}, processed={processed_through_time:.2f}s, "
                f"received={self.audio_end_time:.2f}s, session_complete={is_session_complete}"
            )

            return {
                'text': text,
                'tokens': tokens,
                'generation_progress': generation_progress,
                'is_final': True,  # ⚠️ Always True for _finish() (VAD detected speech end)
                'detected_language': detected_language,  # Phase 3: Per-chunk from Whisper (for code-switching)

                # Hybrid Tracking: SimulStreaming attention tracking (internal precision)
                'attention_tracking': {
                    'most_attended_frame': most_attended_frame,
                    'content_mel_len': content_mel_len,
                    'is_caught_up': is_caught_up,
                },

                # Hybrid Tracking: vexa timestamp tracking (external correlation)
                'timestamp_tracking': {
                    'processed_through_time': processed_through_time,
                    'audio_received_through': self.audio_end_time,
                    'is_session_complete': is_session_complete,
                    'lag_seconds': self.audio_end_time - processed_through_time,
                },

                # vexa-style segment metadata (for deduplication)
                'absolute_start_time': processed_through_time - (len(tokens) * 0.02),  # Rough estimate
                'absolute_end_time': processed_through_time,
                'updated_at': time.time(),
            }

        except Exception as e:
            logger.error(f"Error finishing utterance: {e}")
            import traceback
            traceback.print_exc()
            self.is_currently_final = False
            return {}

    def reset(self):
        """
        Reset processor state for new session

        Clears buffers and resets VAD state
        """
        logger.info("Resetting VACOnlineASRProcessor state")

        self.audio_buffer = torch.tensor([], dtype=torch.float32)
        self.audio_chunks = []  # Clear accumulated chunks list
        self.status = 'nonvoice'
        self.is_currently_final = False
        self.current_online_chunk_buffer_size = 0

        # Phase 4: Reset sustained language detection state
        self.current_sustained_language = None
        self.language_start_time = 0.0
        self.silence_start_time = None
        # Note: We don't reset last_sot_reset_time to maintain cooldown across sessions

        # Phase 5: Reset token deduplicator (new session = no token context to deduplicate)
        self.token_deduplicator.reset()

        # Phase 5: Reset UTF-8 boundary fixer (new session = no boundary context)
        self.utf8_fixer.reset()

        # Hybrid Tracking: Reset timestamp tracking for new session
        self.audio_start_time = 0.0
        self.audio_end_time = 0.0
        self.frames_to_time_offset = 0.0
        self.total_chunks_received = 0
        self.all_chunks_received = False

        if self.vad:
            self.vad.reset_states()

        logger.info("✅ State reset complete")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics

        Returns:
            - vad_checks: Number of VAD checks performed
            - whisper_calls: Number of Whisper transcriptions
            - total_audio_processed: Total audio samples processed
            - compute_efficiency: Ratio of VAD checks to Whisper calls
        """
        compute_efficiency = (
            self.vad_checks / self.whisper_calls
            if self.whisper_calls > 0
            else 0
        )

        return {
            'vad_checks': self.vad_checks,
            'whisper_calls': self.whisper_calls,
            'total_audio_processed': self.total_audio_processed,
            'total_audio_duration': self.total_audio_processed / self.SAMPLING_RATE,
            'compute_efficiency': compute_efficiency,
            'savings_percent': (1 - 1 / compute_efficiency) * 100 if compute_efficiency > 0 else 0
        }


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src directory to path
    SRC_DIR = Path(__file__).parent
    sys.path.insert(0, str(SRC_DIR))

    from whisper_service import ModelManager

    # Initialize processor
    vac = VACOnlineASRProcessor(
        online_chunk_size=1.2,
        vad_threshold=0.5,
        min_buffered_length=1.0
    )

    # Initialize models
    models_dir = Path(__file__).parent.parent / ".models"
    manager = ModelManager(models_dir=str(models_dir))
    vac.init(manager, model_name="large-v3")

    print("\n✅ VACOnlineASRProcessor initialized")
    print(f"   Chunk size: {vac.online_chunk_size}s")
    print(f"   VAD threshold: {vac.vad_threshold}")

    # Simulate streaming (5 seconds of audio in 0.04s chunks)
    vad_chunk_size = int(0.04 * 16000)  # 640 samples
    num_chunks = int(5.0 / 0.04)  # 125 chunks

    print(f"\nSimulating streaming: {num_chunks} chunks (5 seconds)")

    for i in range(num_chunks):
        # Create test chunk
        chunk = np.zeros(vad_chunk_size, dtype=np.float32)

        # Insert and process
        vac.insert_audio_chunk(chunk)
        result = vac.process_iter()

        if result and 'text' in result:
            print(f"  [{i}] Transcription: '{result['text']}'")

    # Print statistics
    stats = vac.get_statistics()
    print(f"\n📊 Processing Statistics:")
    print(f"   VAD checks: {stats['vad_checks']}")
    print(f"   Whisper calls: {stats['whisper_calls']}")
    print(f"   Audio duration: {stats['total_audio_duration']:.2f}s")
    print(f"   Compute efficiency: {stats['compute_efficiency']:.1f}x")
    print(f"   Savings: {stats['savings_percent']:.1f}%")

    print(f"\n✅ VACOnlineASRProcessor test complete")
