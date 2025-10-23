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
from typing import Optional, Dict, Any
from silero_vad_iterator import FixedVADIterator
from sentence_segmenter import SentenceSegmenter

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
        min_buffered_length: float = 1.0,
        sampling_rate: int = 16000
    ):
        self.online_chunk_size = online_chunk_size
        self.vad_threshold = vad_threshold
        self.min_buffered_length = min_buffered_length
        self.SAMPLING_RATE = sampling_rate

        # VAD and Whisper components
        self.vad = None
        self.model = None
        self.model_manager = None

        # Sentence segmentation for better draft/final detection
        self.sentence_segmenter = SentenceSegmenter()

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
                sampling_rate=self.SAMPLING_RATE
            )
            logger.info("✅ Silero VAD initialized")
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

    def insert_audio_chunk(self, audio_chunk: np.ndarray):
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
        """
        # Convert to torch tensor immediately at entry point
        audio_tensor = to_torch_tensor(audio_chunk)

        self.total_audio_processed += len(audio_chunk)

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
                logger.debug(f"VAD: Speech START detected (sample {vad_result['start']})")

            elif 'end' in vad_result:
                # Speech end detected
                self.status = 'nonvoice'
                self.is_currently_final = True
                logger.debug(f"VAD: Speech END detected (sample {vad_result['end']})")

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

        elif len(self.audio_buffer) > self.SAMPLING_RATE * self.online_chunk_size:
            # Buffer full - move buffer to audio_chunks, then process chunk
            logger.info(
                f"Processing chunk (buffer full: "
                f"{len(self.audio_buffer) / self.SAMPLING_RATE:.2f}s >= "
                f"{self.online_chunk_size}s)"
            )
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

            # Decode tokens to text
            logger.info(f"[INFER] Generated {len(tokens)} tokens: {tokens[:20] if len(tokens) > 20 else tokens}")
            logger.info(f"[INFER] Generation progress: {generation_progress}")
            text = self.model.tokenizer.decode(tokens)

            self.whisper_calls += 1

            # Check for sentence boundary (complete sentence = is_final)
            # This provides much better draft/final distinction than VAD-only
            is_sentence_end = self.sentence_segmenter.is_sentence_end(text)
            is_final = is_sentence_end

            if is_sentence_end:
                logger.info(f"✅ Complete sentence detected: '{text}' ({len(tokens)} tokens) [is_final=True]")
            else:
                logger.info(f"📝 Incomplete sentence: '{text}' ({len(tokens)} tokens) [is_final=False]")

            return {
                'text': text,
                'tokens': tokens,
                'generation_progress': generation_progress,
                'is_final': is_final  # True if sentence ends with terminal punctuation
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

            # Decode tokens to text
            text = self.model.tokenizer.decode(tokens)

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

            # Reset VAC state
            self.is_currently_final = False

            return {
                'text': text,
                'tokens': tokens,
                'generation_progress': generation_progress,
                'is_final': True  # Always final when VAD detects speech end
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
