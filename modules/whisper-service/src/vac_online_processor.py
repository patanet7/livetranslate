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

logger = logging.getLogger(__name__)


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

        # Audio buffers
        self.audio_buffer = np.array([], dtype=np.float32)
        self.online_chunk_buffer = np.array([], dtype=np.float32)

        # State tracking
        self.status = 'nonvoice'  # 'voice' or 'nonvoice'
        self.is_currently_final = False
        self.current_online_chunk_buffer_size = 0

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

        This is called for EVERY incoming audio chunk (e.g., 0.04s chunks).
        VAD runs on every chunk to detect speech start/end.
        Whisper only runs when buffer full OR speech ends.

        Args:
            audio_chunk: Audio data (numpy array, float32)
        """
        self.total_audio_processed += len(audio_chunk)

        # Run VAD detection
        vad_result = self.vad(audio_chunk, return_seconds=False)
        self.vad_checks += 1

        # Append to audio buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)

        # Handle VAD events
        if vad_result is not None:
            if 'start' in vad_result:
                # Speech start detected
                self.status = 'voice'
                logger.debug(f"VAD: Speech START detected (sample {vad_result['start']})")

                # Send buffered audio to Whisper
                if len(self.audio_buffer) > 0:
                    self._send_audio_to_online_processor(self.audio_buffer)
                    self.audio_buffer = np.array([], dtype=np.float32)

            elif 'end' in vad_result:
                # Speech end detected
                self.status = 'nonvoice'
                self.is_currently_final = True
                logger.debug(f"VAD: Speech END detected (sample {vad_result['end']})")

                # Send final audio to Whisper
                if len(self.audio_buffer) > 0:
                    self._send_audio_to_online_processor(self.audio_buffer)
                    self.audio_buffer = np.array([], dtype=np.float32)

        else:
            # No VAD event - handle based on current status
            if self.status == 'voice':
                # During speech: send buffered audio to Whisper
                if len(self.audio_buffer) > 0:
                    self._send_audio_to_online_processor(self.audio_buffer)
                    self.audio_buffer = np.array([], dtype=np.float32)

            else:
                # During silence: just buffer (NO processing!)
                # Limit buffer to prevent OOM (keep only 1 second)
                max_silence_buffer = self.SAMPLING_RATE * 1.0
                if len(self.audio_buffer) > max_silence_buffer:
                    # Keep only most recent 1 second
                    self.audio_buffer = self.audio_buffer[-int(max_silence_buffer):]

                logger.debug(
                    f"VAD: Silence buffering (buffer size: {len(self.audio_buffer)} samples)"
                )

    def _send_audio_to_online_processor(self, audio: np.ndarray):
        """
        Send audio to Whisper online processor

        Appends to online chunk buffer for processing
        """
        self.online_chunk_buffer = np.append(self.online_chunk_buffer, audio)
        self.current_online_chunk_buffer_size = len(self.online_chunk_buffer)

        logger.debug(
            f"Online buffer: {self.current_online_chunk_buffer_size} samples "
            f"({self.current_online_chunk_buffer_size / self.SAMPLING_RATE:.2f}s)"
        )

    def process_iter(self) -> Dict[str, Any]:
        """
        Process iteration - decides when to run Whisper

        This is the CORE ADAPTIVE LOGIC:
        - If speech ended (is_currently_final): process immediately
        - If buffer >= online_chunk_size (1.2s): process
        - Otherwise: NO processing (saves compute!)

        Returns:
            - Transcription result if processing occurred
            - Empty dict if still buffering
        """
        # Check if we should process
        if self.is_currently_final:
            # Speech ended - process final chunk
            logger.info("Processing FINAL chunk (speech ended)")
            return self._finish()

        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            # Buffer full - process chunk
            logger.info(
                f"Processing chunk (buffer full: "
                f"{self.current_online_chunk_buffer_size / self.SAMPLING_RATE:.2f}s >= "
                f"{self.online_chunk_size}s)"
            )
            return self._process_online_chunk()

        else:
            # Still buffering - NO processing yet!
            logger.debug(
                f"Buffering... {self.current_online_chunk_buffer_size} samples "
                f"(status: {self.status})"
            )
            return {}

    def _process_online_chunk(self) -> Dict[str, Any]:
        """
        Process online chunk with Whisper

        Extract chunk of online_chunk_size and transcribe
        """
        if self.model is None:
            logger.error("Model not initialized")
            return {}

        try:
            # Extract chunk to process
            chunk_samples = int(self.SAMPLING_RATE * self.online_chunk_size)
            audio_to_process = self.online_chunk_buffer[:chunk_samples]

            # Transcribe with Whisper
            result = self.model.transcribe(
                audio=audio_to_process,
                beam_size=5,
                temperature=0.0
            )

            self.whisper_calls += 1

            # Remove processed audio from buffer
            self.online_chunk_buffer = self.online_chunk_buffer[chunk_samples:]
            self.current_online_chunk_buffer_size = len(self.online_chunk_buffer)

            logger.info(f"✅ Whisper transcribed: '{result['text']}'")

            return {
                'text': result['text'],
                'segments': result.get('segments', []),
                'is_final': False
            }

        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return {}

    def _finish(self) -> Dict[str, Any]:
        """
        Finish current utterance (speech ended)

        Process remaining buffer and reset state
        """
        if self.model is None:
            logger.error("Model not initialized")
            return {}

        try:
            # Process remaining buffer
            if len(self.online_chunk_buffer) > 0:
                result = self.model.transcribe(
                    audio=self.online_chunk_buffer,
                    beam_size=5,
                    temperature=0.0
                )

                self.whisper_calls += 1

                logger.info(f"✅ Final transcription: '{result['text']}'")

                # Reset state
                self.online_chunk_buffer = np.array([], dtype=np.float32)
                self.current_online_chunk_buffer_size = 0
                self.is_currently_final = False

                return {
                    'text': result['text'],
                    'segments': result.get('segments', []),
                    'is_final': True
                }
            else:
                # No audio to process
                self.is_currently_final = False
                return {}

        except Exception as e:
            logger.error(f"Error finishing utterance: {e}")
            self.is_currently_final = False
            return {}

    def reset(self):
        """
        Reset processor state for new session

        Clears buffers and resets VAD state
        """
        logger.info("Resetting VACOnlineASRProcessor state")

        self.audio_buffer = np.array([], dtype=np.float32)
        self.online_chunk_buffer = np.array([], dtype=np.float32)
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
