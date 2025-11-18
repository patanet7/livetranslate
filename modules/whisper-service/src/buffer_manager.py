#!/usr/bin/env python3
"""
Real-time Audio Buffer Manager with Voice Activity Detection

High-performance rolling buffer management for real-time audio processing with
VAD integration, speaker diarization support, and optimized memory usage.

Features:
- Rolling audio buffers with configurable size and overlap
- Voice Activity Detection (WebRTC VAD with Silero fallback)
- Speaker diarization integration
- Speech enhancement and noise reduction
- Thread-safe buffer operations
- Automatic inference timing control
- Memory-efficient circular buffer implementation
"""

import time
import threading
import logging
from collections import deque
from typing import Optional, Dict, Any, List, Callable, Tuple
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

try:
    import silero_vad
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BufferConfig:
    """Configuration for the buffer manager."""
    buffer_duration: float = 4.0  # Reduced from 6.0 to minimize overlap
    inference_interval: float = 3.0  # Keep 3 second intervals
    sample_rate: int = 16000
    overlap_duration: float = 0.2  # Reduced from 1.0 to 0.2 seconds - minimal overlap
    vad_enabled: bool = True
    vad_aggressiveness: int = 2
    vad_frame_duration: int = 30  # ms
    enable_diarization: bool = False
    n_speakers: Optional[int] = None
    enable_speech_enhancement: bool = True
    max_silence_duration: float = 2.0
    min_speech_duration: float = 0.5


@dataclass
class AudioSegment:
    """Represents a segment of audio with metadata."""
    audio_data: np.ndarray
    start_time: float
    end_time: float
    is_speech: bool = True
    speaker_id: Optional[int] = None
    confidence: float = 1.0
    enhanced: bool = False


class VADProcessor:
    """Voice Activity Detection processor with multiple backend support."""
    
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2, 
                 frame_duration: int = 30):
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        # Initialize WebRTC VAD
        self.webrtc_vad = None
        if WEBRTC_VAD_AVAILABLE:
            try:
                self.webrtc_vad = webrtcvad.Vad(aggressiveness)
                logger.info(f"✓ WebRTC VAD initialized (aggressiveness: {aggressiveness})")
            except Exception as e:
                logger.warning(f"WebRTC VAD initialization failed: {e}")
        
        # Initialize Silero VAD as fallback
        self.silero_vad = None
        if SILERO_VAD_AVAILABLE:
            try:
                # Initialize Silero VAD model
                self.silero_vad = silero_vad.load_silero_vad()
                logger.info("✓ Silero VAD initialized as fallback")
            except Exception as e:
                logger.warning(f"Silero VAD initialization failed: {e}")
        
        if not self.webrtc_vad and not self.silero_vad:
            logger.warning("⚠ No VAD backend available - speech detection disabled")
    
    def is_speech(self, audio_data: np.ndarray) -> bool:
        """Detect speech in audio segment."""
        if len(audio_data) == 0:
            return False
        
        try:
            # Try WebRTC VAD first
            if self.webrtc_vad and len(audio_data) >= self.frame_size:
                # Convert to 16-bit PCM for WebRTC VAD
                audio_16bit = (audio_data * 32767).astype(np.int16)
                
                # Process in frames
                speech_frames = 0
                total_frames = 0
                
                for i in range(0, len(audio_16bit) - self.frame_size + 1, self.frame_size):
                    frame = audio_16bit[i:i + self.frame_size]
                    if len(frame) == self.frame_size:
                        try:
                            if self.webrtc_vad.is_speech(frame.tobytes(), self.sample_rate):
                                speech_frames += 1
                            total_frames += 1
                        except Exception:
                            continue
                
                if total_frames > 0:
                    speech_ratio = speech_frames / total_frames
                    return speech_ratio > 0.3  # 30% threshold
            
            # Fallback to Silero VAD
            if self.silero_vad and TORCH_AVAILABLE:
                try:
                    # Ensure correct tensor format for Silero
                    audio_tensor = torch.from_numpy(audio_data).float()
                    if len(audio_tensor.shape) == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)

                    speech_prob = self.silero_vad(audio_tensor, self.sample_rate).item()
                    return speech_prob > 0.5
                except Exception as e:
                    logger.debug(f"Silero VAD error: {e}")
            
            # Fallback to energy-based detection
            return self._energy_based_vad(audio_data)
            
        except Exception as e:
            logger.warning(f"VAD processing failed: {e}")
            return True  # Default to speech if VAD fails
    
    def _energy_based_vad(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """Simple energy-based voice activity detection."""
        if len(audio_data) == 0:
            return False
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > threshold


class RollingBufferManager:
    """
    High-performance rolling buffer manager for real-time audio processing.
    
    Manages circular audio buffers with VAD, speaker diarization support,
    and optimized memory usage for continuous audio streaming.
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        """
        Initialize the buffer manager.
        
        Args:
            config: Buffer configuration object
        """
        self.config = config or BufferConfig()
        
        # Calculate buffer parameters
        self.max_samples = int(self.config.buffer_duration * self.config.sample_rate)
        self.inference_samples = int(self.config.inference_interval * self.config.sample_rate)
        self.overlap_samples = int(self.config.overlap_duration * self.config.sample_rate)
        
        # Rolling buffer for audio samples (memory-efficient circular buffer)
        self.audio_buffer = deque(maxlen=self.max_samples)
        self.buffer_lock = threading.RLock()
        
        # Timing control
        self.last_inference_time = 0
        self.inference_timer = None
        self.is_running = False
        self.start_time = time.time()
        
        # VAD processor
        self.vad_processor = None
        if self.config.vad_enabled:
            self.vad_processor = VADProcessor(
                sample_rate=self.config.sample_rate,
                aggressiveness=self.config.vad_aggressiveness,
                frame_duration=self.config.vad_frame_duration
            )
        
        # Speech enhancement
        self.speech_enhancer = None
        if self.config.enable_speech_enhancement:
            self._initialize_speech_enhancer()
        
        # Speaker diarization (optional)
        self.speaker_diarizer = None
        self.recent_diarization = deque(maxlen=50)
        if self.config.enable_diarization:
            self._initialize_speaker_diarization()
        
        # Transcription management
        self.transcription_segments = deque(maxlen=200)
        self.last_transcription = ""
        self.last_transcription_time = 0
        
        # Track processed audio to prevent duplicates
        self.last_processed_sample_index = 0  # Total samples processed
        self.total_samples_added = 0  # Total samples added to buffer
        self.processed_chunks = deque(maxlen=100)  # Track processed chunk timestamps
        
        # Callbacks
        self.transcription_callback: Optional[Callable] = None
        self.speech_detection_callback: Optional[Callable] = None
        self.buffer_full_callback: Optional[Callable] = None
        
        # Statistics
        self.total_audio_processed = 0
        self.speech_segments_detected = 0
        self.inference_count = 0
        
        logger.info("RollingBufferManager initialized:")
        logger.info(f"  Buffer duration: {self.config.buffer_duration}s ({self.max_samples} samples)")
        logger.info(f"  Inference interval: {self.config.inference_interval}s")
        logger.info(f"  Sample rate: {self.config.sample_rate}Hz")
        logger.info(f"  VAD enabled: {self.config.vad_enabled}")
        logger.info(f"  Speaker diarization: {self.config.enable_diarization}")
        logger.info(f"  Speech enhancement: {self.config.enable_speech_enhancement}")
    
    def _initialize_speech_enhancer(self):
        """Initialize speech enhancement if available."""
        try:
            # Try to import and initialize speech enhancement
            # This would typically be from the speaker_diarization module
            from speaker_diarization import SpeechEnhancer
            self.speech_enhancer = SpeechEnhancer(self.config.sample_rate)
            logger.info("✓ Speech enhancement initialized")
        except ImportError:
            logger.warning("Speech enhancement not available - skipping")
        except Exception as e:
            logger.warning(f"Speech enhancement initialization failed: {e}")
    
    def _initialize_speaker_diarization(self):
        """Initialize speaker diarization if available."""
        try:
            # Try to import and initialize speaker diarization
            from speaker_diarization import AdvancedSpeakerDiarization
            self.speaker_diarizer = AdvancedSpeakerDiarization(
                sample_rate=self.config.sample_rate,
                window_duration=self.config.buffer_duration,
                overlap_duration=self.config.overlap_duration,
                n_speakers=self.config.n_speakers,
                embedding_method='resemblyzer',
                clustering_method='hdbscan' if self.config.n_speakers is None else 'agglomerative',
                enable_enhancement=self.config.enable_speech_enhancement,
                device='cpu'
            )
            logger.info(f"✓ Speaker diarization initialized ({self.config.n_speakers or 'auto'} speakers)")
        except ImportError:
            logger.warning("Speaker diarization not available - skipping")
        except Exception as e:
            logger.warning(f"Speaker diarization initialization failed: {e}")
    
    def add_audio_chunk(self, audio_samples: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Add new audio samples to the rolling buffer.
        
        Args:
            audio_samples: Audio data as numpy array
            timestamp: Optional timestamp for the audio
            
        Returns:
            True if chunk was processed successfully
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            with self.buffer_lock:
                # Validate input
                if not isinstance(audio_samples, np.ndarray):
                    audio_samples = np.array(audio_samples, dtype=np.float32)
                
                if len(audio_samples) == 0:
                    return False
                
                # Normalize audio if needed
                if audio_samples.dtype != np.float32:
                    audio_samples = audio_samples.astype(np.float32)
                
                # Ensure audio is in correct range
                if np.max(np.abs(audio_samples)) > 1.0:
                    audio_samples = audio_samples / np.max(np.abs(audio_samples))
                
                # Add to rolling buffer and track total samples
                self.audio_buffer.extend(audio_samples.tolist())
                self.total_audio_processed += len(audio_samples)
                self.total_samples_added += len(audio_samples)  # Track all samples added
                
                # Process with VAD
                is_speech = True
                if self.vad_processor:
                    try:
                        is_speech = self.vad_processor.is_speech(audio_samples)
                        if is_speech:
                            self.speech_segments_detected += 1
                        
                        # Call speech detection callback
                        if self.speech_detection_callback:
                            self.speech_detection_callback(is_speech, timestamp)
                    except Exception as e:
                        logger.warning(f"VAD processing failed: {e}")
                
                # Process with speaker diarization if enabled
                if self.speaker_diarizer and is_speech:
                    try:
                        diarization_results = self.speaker_diarizer.process_audio_chunk(audio_samples)
                        if diarization_results:
                            self.recent_diarization.extend(diarization_results)
                            logger.debug(f"🎤 Speaker diarization: {len(diarization_results)} segments")
                    except Exception as e:
                        logger.warning(f"Speaker diarization failed: {e}")
                
                # Enhance audio if enabled and speech detected
                if self.speech_enhancer and is_speech:
                    try:
                        # Speech enhancement - could store enhanced version separately or replace
                        self.speech_enhancer.enhance_audio(audio_samples)
                    except Exception as e:
                        logger.debug(f"Speech enhancement failed: {e}")
                
                # Start inference timer if conditions are met
                self._check_inference_trigger(timestamp, is_speech)
                
                # Call buffer full callback if needed
                if len(self.audio_buffer) >= self.max_samples * 0.9:  # 90% full
                    if self.buffer_full_callback:
                        self.buffer_full_callback(len(self.audio_buffer), self.max_samples)
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")
            return False
    
    def _check_inference_trigger(self, timestamp: float, is_speech: bool):
        """Check if inference should be triggered based on timing and speech detection."""
        current_time = timestamp
        time_since_last = current_time - self.last_inference_time
        
        # Trigger conditions
        should_trigger = False
        
        # Time-based trigger
        if time_since_last >= self.config.inference_interval:
            should_trigger = True
        
        # Speech-based trigger (if we have enough new speech)
        if is_speech and time_since_last >= self.config.min_speech_duration:
            buffer_has_speech = len(self.audio_buffer) >= self.inference_samples
            if buffer_has_speech:
                should_trigger = True
        
        if should_trigger and not self.inference_timer:
            self._start_inference_timer()
    
    def _start_inference_timer(self):
        """Start the inference timer."""
        if self.inference_timer:
            return
        
        def trigger_inference():
            try:
                if self.transcription_callback:
                    audio_data = self.get_buffer_for_inference()
                    if len(audio_data) > 0:
                        self.transcription_callback(audio_data, time.time())
                        self.inference_count += 1
                        self.last_inference_time = time.time()
            except Exception as e:
                logger.error(f"Inference callback failed: {e}")
            finally:
                self.inference_timer = None
        
        # Use a small delay to batch multiple rapid calls
        self.inference_timer = threading.Timer(0.1, trigger_inference)
        self.inference_timer.start()
    
    def get_buffer_for_inference(self, include_overlap: bool = False) -> np.ndarray:
        """
        Get audio buffer data for inference with smart duplicate prevention.
        
        Args:
            include_overlap: Whether to include overlap from previous inference
            
        Returns:
            Audio data as numpy array
        """
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return np.array([], dtype=np.float32)
            
            # Convert deque to numpy array
            audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
            
            # Smart overlap: only include minimal overlap to prevent duplicates
            if include_overlap and self.last_transcription_time > 0:
                # Very small overlap (0.2 seconds) to maintain continuity without duplicates
                overlap_samples = min(self.overlap_samples, len(audio_data) // 10)  # Max 10% overlap
                if overlap_samples > 0:
                    logger.debug(f"Including {overlap_samples} overlap samples ({overlap_samples/self.config.sample_rate:.2f}s)")
            
            # After inference, mark this audio as processed to avoid re-transcription
            self.last_transcription_time = time.time()
            
            return audio_data
    
    def get_new_audio_only(self, min_new_samples: int = None) -> Tuple[np.ndarray, int]:
        """
        Get only the new audio that hasn't been processed yet.
        
        Args:
            min_new_samples: Minimum number of new samples needed (default: inference_samples)
            
        Returns:
            Tuple of (new_audio_data, samples_to_mark_processed)
        """
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return np.array([], dtype=np.float32), 0
            
            min_new = min_new_samples or self.inference_samples

            # Calculate how many new samples we have since last processing
            new_samples_available = self.total_samples_added - self.last_processed_sample_index
            
            # If we don't have enough new samples, return empty
            if new_samples_available < min_new:
                logger.debug(f"Not enough new audio: {new_samples_available} < {min_new} samples")
                return np.array([], dtype=np.float32), 0
            
            # Convert buffer to array
            audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
            
            # Calculate which portion is new
            if self.last_processed_sample_index == 0:
                # First time - return inference_samples worth of audio
                new_audio = audio_data[-min(self.inference_samples, len(audio_data)):]
                samples_to_mark = len(new_audio)
            else:
                # Return only the truly new samples + small continuity overlap
                overlap_samples = min(self.overlap_samples, len(audio_data) // 4)  # Max 25% overlap
                
                # Get new samples since last processing
                new_samples_count = min(new_samples_available, self.inference_samples)
                start_idx = max(0, len(audio_data) - new_samples_count - overlap_samples)
                new_audio = audio_data[start_idx:]
                
                # Mark only the truly new samples as processed (not the overlap)
                samples_to_mark = new_samples_count
            
            logger.debug(f"Returning {len(new_audio)} samples ({len(new_audio)/self.config.sample_rate:.2f}s), "
                        f"marking {samples_to_mark} as processed")
            
            return new_audio, samples_to_mark
    
    def get_buffer_audio(self) -> np.ndarray:
        """
        Get current buffer audio (legacy method for compatibility).
        For new code, use get_new_audio_only() instead.
        """
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return np.array([], dtype=np.float32)
            return np.array(list(self.audio_buffer), dtype=np.float32)
    
    def mark_audio_as_processed(self, num_samples: int):
        """Mark the specified number of samples as processed to prevent re-transcription."""
        with self.buffer_lock:
            self.last_processed_sample_index += num_samples
            self.processed_chunks.append({
                'timestamp': time.time(),
                'samples_processed': num_samples,
                'total_processed': self.last_processed_sample_index
            })
            logger.debug(f"Marked {num_samples} samples as processed. Total: {self.last_processed_sample_index}")
    
    def get_recent_segments(self, limit: int = 10) -> List[AudioSegment]:
        """Get recent audio segments with metadata."""
        # This would return processed segments with speaker info, etc.
        # Implementation depends on how segments are stored
        return list(self.transcription_segments)[-limit:]
    
    def clear_buffer(self):
        """Clear the audio buffer."""
        with self.buffer_lock:
            self.audio_buffer.clear()
            logger.debug("Audio buffer cleared")
    
    def set_transcription_callback(self, callback: Callable[[np.ndarray, float], None]):
        """Set callback for transcription triggers."""
        self.transcription_callback = callback
    
    def set_speech_detection_callback(self, callback: Callable[[bool, float], None]):
        """Set callback for speech detection events."""
        self.speech_detection_callback = callback
    
    def set_buffer_full_callback(self, callback: Callable[[int, int], None]):
        """Set callback for buffer full events."""
        self.buffer_full_callback = callback
    
    def update_config(self, new_config: BufferConfig):
        """Update buffer configuration dynamically."""
        with self.buffer_lock:
            old_max_samples = self.max_samples
            
            self.config = new_config
            self.max_samples = int(new_config.buffer_duration * new_config.sample_rate)
            self.inference_samples = int(new_config.inference_interval * new_config.sample_rate)
            self.overlap_samples = int(new_config.overlap_duration * new_config.sample_rate)
            
            # Adjust buffer size if needed
            if self.max_samples != old_max_samples:
                new_buffer = deque(self.audio_buffer, maxlen=self.max_samples)
                self.audio_buffer = new_buffer
            
            # Reinitialize VAD if settings changed
            if new_config.vad_enabled and (not self.vad_processor or 
                new_config.vad_aggressiveness != self.vad_processor.aggressiveness):
                self.vad_processor = VADProcessor(
                    sample_rate=new_config.sample_rate,
                    aggressiveness=new_config.vad_aggressiveness,
                    frame_duration=new_config.vad_frame_duration
                )
            
            logger.info(f"Buffer configuration updated: {new_config}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics."""
        with self.buffer_lock:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            return {
                "buffer_size": len(self.audio_buffer),
                "max_buffer_size": self.max_samples,
                "buffer_usage_percent": (len(self.audio_buffer) / self.max_samples) * 100,
                "total_audio_processed": self.total_audio_processed,
                "speech_segments_detected": self.speech_segments_detected,
                "inference_count": self.inference_count,
                "uptime_seconds": uptime,
                "last_inference_time": self.last_inference_time,
                "vad_enabled": self.config.vad_enabled,
                "diarization_enabled": self.config.enable_diarization,
                "enhancement_enabled": self.config.enable_speech_enhancement,
                "config": {
                    "buffer_duration": self.config.buffer_duration,
                    "inference_interval": self.config.inference_interval,
                    "sample_rate": self.config.sample_rate,
                    "overlap_duration": self.config.overlap_duration
                }
            }
    
    def stop_inference_timer(self):
        """Stop the inference timer."""
        if self.inference_timer:
            self.inference_timer.cancel()
            self.inference_timer = None
    
    def shutdown(self):
        """Graceful shutdown of the buffer manager."""
        logger.info("Shutting down RollingBufferManager")
        try:
            self.stop_inference_timer()
            self.is_running = False
            with self.buffer_lock:
                self.audio_buffer.clear()
            logger.info("✓ RollingBufferManager shutdown complete")
        except Exception as e:
            logger.error(f"Error during buffer manager shutdown: {e}")
    
    @contextmanager
    def buffer_context(self):
        """Context manager for safe buffer operations."""
        with self.buffer_lock:
            try:
                yield self
            except Exception as e:
                logger.error(f"Error in buffer context: {e}")
                raise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Convenience functions
def create_buffer_manager(
    buffer_duration: float = 6.0,
    inference_interval: float = 3.0,
    sample_rate: int = 16000,
    **kwargs
) -> RollingBufferManager:
    """Create a buffer manager with common settings."""
    config = BufferConfig(
        buffer_duration=buffer_duration,
        inference_interval=inference_interval,
        sample_rate=sample_rate,
        **kwargs
    )
    return RollingBufferManager(config)


def create_optimized_buffer_manager(
    device_type: str = "npu",
    **kwargs
) -> RollingBufferManager:
    """Create a buffer manager optimized for specific device types."""
    if device_type.lower() == "npu":
        # NPU-optimized settings
        config = BufferConfig(
            buffer_duration=6.0,
            inference_interval=3.0,
            overlap_duration=1.0,
            vad_enabled=True,
            vad_aggressiveness=2,
            enable_speech_enhancement=True,
            **kwargs
        )
    elif device_type.lower() == "gpu":
        # GPU-optimized settings
        config = BufferConfig(
            buffer_duration=4.0,
            inference_interval=2.0,
            overlap_duration=0.5,
            vad_enabled=True,
            vad_aggressiveness=1,
            enable_speech_enhancement=True,
            **kwargs
        )
    else:
        # CPU-optimized settings
        config = BufferConfig(
            buffer_duration=8.0,
            inference_interval=4.0,
            overlap_duration=1.5,
            vad_enabled=True,
            vad_aggressiveness=3,
            enable_speech_enhancement=False,
            **kwargs
        )
    
    return RollingBufferManager(config)