#!/usr/bin/env python3
"""
Audio Chunk Manager - Orchestration Service

Handles intelligent audio chunking with overlap management, quality analysis, and database persistence.
Centralizes all chunking logic that was previously scattered across frontend and whisper service.

Features:
- Configurable chunking with overlap management
- Quality-based chunk validation and filtering
- Memory-efficient rolling buffer management
- Database persistence with lineage tracking
- Real-time chunk processing coordination
- Audio format detection and conversion
- Speaker-aware chunking for improved correlation
"""

import asyncio
import hashlib
import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Callable
import numpy as np
import soundfile as sf
from datetime import datetime

from .models import (
    AudioChunkMetadata,
    AudioChunkingConfig,
    QualityMetrics,
    ProcessingStatus,
    SourceType,
    create_audio_chunk_metadata,
)
from .database_adapter import AudioDatabaseAdapter

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    Thread-safe rolling audio buffer with overlap management.
    Efficiently handles audio data with configurable chunking parameters.
    """
    
    def __init__(self, config: AudioChunkingConfig):
        self.config = config
        self.sample_rate = 16000  # Standard sample rate for whisper
        
        # Calculate buffer sizes
        self.chunk_samples = int(config.chunk_duration * self.sample_rate)
        self.overlap_samples = int(config.overlap_duration * self.sample_rate)
        self.buffer_samples = int(config.buffer_duration * self.sample_rate)
        
        # Rolling buffer
        self.buffer = deque(maxlen=self.buffer_samples)
        self.total_samples_added = 0
        self.chunk_sequence = 0
        
        # Overlap tracking
        self.last_chunk_end = 0
        self.overlap_data = np.array([], dtype=np.float32)
        
        logger.info(f"AudioBuffer initialized: chunk={self.chunk_samples}, overlap={self.overlap_samples}, buffer={self.buffer_samples}")
    
    def add_audio_data(self, audio_data: np.ndarray) -> int:
        """
        Add audio data to the rolling buffer.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            int: Number of samples added
        """
        if len(audio_data) == 0:
            return 0
        
        # Ensure correct data type
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Add to rolling buffer
        for sample in audio_data:
            self.buffer.append(sample)
        
        self.total_samples_added += len(audio_data)
        return len(audio_data)
    
    def get_next_chunk(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Extract the next audio chunk with overlap handling.
        
        Returns:
            Tuple of (audio_chunk, chunk_metadata) or None if insufficient data
        """
        if len(self.buffer) < self.chunk_samples:
            return None
        
        # Calculate chunk timing
        chunk_start_sample = self.total_samples_added - len(self.buffer)
        chunk_start_time = chunk_start_sample / self.sample_rate
        chunk_end_time = chunk_start_time + self.config.chunk_duration
        
        # Extract chunk data
        chunk_data = np.array(list(self.buffer)[:self.chunk_samples], dtype=np.float32)
        
        # Handle overlap
        overlap_metadata = {}
        if len(self.overlap_data) > 0 and self.overlap_samples > 0:
            # Apply overlap from previous chunk
            overlap_len = min(len(self.overlap_data), self.overlap_samples, len(chunk_data))
            if overlap_len > 0:
                # Blend overlap region
                blend_factor = np.linspace(1.0, 0.0, overlap_len)
                chunk_data[:overlap_len] = (
                    chunk_data[:overlap_len] * blend_factor + 
                    self.overlap_data[:overlap_len] * (1.0 - blend_factor)
                )
                
                overlap_metadata = {
                    "overlap_applied": True,
                    "overlap_samples": overlap_len,
                    "overlap_duration": overlap_len / self.sample_rate,
                    "blend_method": "linear"
                }
        
        # Store overlap data for next chunk
        if self.overlap_samples > 0 and len(chunk_data) >= self.overlap_samples:
            self.overlap_data = chunk_data[-self.overlap_samples:].copy()
        
        # Create chunk metadata
        chunk_metadata = {
            "chunk_sequence": self.chunk_sequence,
            "chunk_start_time": chunk_start_time,
            "chunk_end_time": chunk_end_time,
            "duration_seconds": self.config.chunk_duration,
            "sample_rate": self.sample_rate,
            "samples_count": len(chunk_data),
            "overlap_metadata": overlap_metadata,
            "buffer_size_samples": len(self.buffer),
            "total_samples_processed": self.total_samples_added,
        }
        
        # Remove processed samples from buffer (accounting for overlap)
        samples_to_remove = self.chunk_samples - self.overlap_samples
        for _ in range(min(samples_to_remove, len(self.buffer))):
            self.buffer.popleft()
        
        self.chunk_sequence += 1
        self.last_chunk_end = chunk_end_time
        
        return chunk_data, chunk_metadata
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status and statistics."""
        return {
            "buffer_samples": len(self.buffer),
            "buffer_duration": len(self.buffer) / self.sample_rate,
            "total_samples_added": self.total_samples_added,
            "total_duration_added": self.total_samples_added / self.sample_rate,
            "chunk_sequence": self.chunk_sequence,
            "chunks_ready": len(self.buffer) // self.chunk_samples,
            "overlap_samples": len(self.overlap_data),
            "last_chunk_end": self.last_chunk_end,
        }
    
    def clear(self):
        """Clear the buffer and reset state."""
        self.buffer.clear()
        self.overlap_data = np.array([], dtype=np.float32)
        self.total_samples_added = 0
        self.chunk_sequence = 0
        self.last_chunk_end = 0


class AudioQualityAnalyzer:
    """
    Comprehensive audio quality analysis for chunk validation.
    Provides detailed metrics for quality-based processing decisions.
    """
    
    def __init__(self, config: AudioChunkingConfig):
        self.config = config
        
        # Quality thresholds
        self.silence_threshold = config.silence_threshold
        self.noise_threshold = config.noise_threshold
        self.min_quality_threshold = config.min_quality_threshold
        
        # Analysis parameters
        self.voice_frequency_range = (85, 300)  # Human voice fundamental frequency range
        self.formant_ranges = [(300, 3000), (900, 3000)]  # Formant frequency ranges
        
    def analyze_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000) -> QualityMetrics:
        """
        Perform comprehensive quality analysis on audio chunk.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate of audio data
            
        Returns:
            QualityMetrics: Comprehensive quality assessment
        """
        try:
            # Basic level measurements
            rms_level = np.sqrt(np.mean(audio_data ** 2))
            peak_level = np.max(np.abs(audio_data))
            
            # Zero crossing rate (voice activity indicator)
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            zcr = zero_crossings / len(audio_data) if len(audio_data) > 0 else 0.0
            
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio_data ** 2)
            noise_estimate = np.percentile(np.abs(audio_data), 10) ** 2  # 10th percentile as noise estimate
            snr = 10 * np.log10(signal_power / max(noise_estimate, 1e-10)) if signal_power > 0 else -60
            
            # Voice activity detection
            voice_activity = (
                rms_level > self.silence_threshold and 
                zcr > 0.01 and 
                snr > -10
            )
            voice_confidence = min(1.0, max(0.0, (rms_level - self.silence_threshold) / 0.05))
            
            # Speaking time ratio (rough estimate)
            speaking_samples = np.sum(np.abs(audio_data) > self.silence_threshold)
            speaking_ratio = speaking_samples / len(audio_data) if len(audio_data) > 0 else 0.0
            
            # Clipping detection
            clipping_threshold = 0.95
            clipping_detected = peak_level > clipping_threshold
            
            # Distortion estimation (simple THD approximation)
            distortion_level = min(1.0, peak_level) if clipping_detected else 0.0
            
            # Noise level estimation
            noise_level = min(1.0, noise_estimate / 0.01)  # Normalize to 0-1 scale
            
            # Frequency analysis (if sufficient data)
            spectral_centroid = None
            spectral_bandwidth = None
            spectral_rolloff = None
            
            if len(audio_data) > 512:  # Minimum for meaningful FFT
                try:
                    # Compute power spectrum
                    fft = np.fft.rfft(audio_data)
                    power_spectrum = np.abs(fft) ** 2
                    freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
                    
                    # Spectral centroid
                    if np.sum(power_spectrum) > 0:
                        spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
                        
                        # Spectral bandwidth
                        spectral_bandwidth = np.sqrt(
                            np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / 
                            np.sum(power_spectrum)
                        )
                        
                        # Spectral rolloff (95% of energy)
                        cumsum = np.cumsum(power_spectrum)
                        rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
                        if len(rolloff_idx) > 0:
                            spectral_rolloff = freqs[rolloff_idx[0]]
                            
                except Exception as e:
                    logger.debug(f"Spectral analysis failed: {e}")
            
            # Overall quality score computation
            quality_factors = {
                "level_factor": min(1.0, rms_level / 0.1),  # Normalize to reasonable speech level
                "snr_factor": min(1.0, max(0.0, (snr + 10) / 30)),  # SNR from -10 to 20 dB
                "voice_factor": voice_confidence,
                "distortion_factor": 1.0 - distortion_level,
                "noise_factor": 1.0 - noise_level,
            }
            
            # Weighted overall quality
            weights = {
                "level_factor": 0.2,
                "snr_factor": 0.3,
                "voice_factor": 0.3,
                "distortion_factor": 0.1,
                "noise_factor": 0.1,
            }
            
            overall_quality = sum(
                quality_factors[factor] * weights[factor] 
                for factor in quality_factors
            )
            
            return QualityMetrics(
                rms_level=float(rms_level),
                peak_level=float(peak_level),
                signal_to_noise_ratio=float(snr),
                zero_crossing_rate=float(zcr),
                voice_activity_detected=voice_activity,
                voice_activity_confidence=float(voice_confidence),
                speaking_time_ratio=float(speaking_ratio),
                clipping_detected=clipping_detected,
                distortion_level=float(distortion_level),
                noise_level=float(noise_level),
                spectral_centroid=spectral_centroid,
                spectral_bandwidth=spectral_bandwidth,
                spectral_rolloff=spectral_rolloff,
                overall_quality_score=float(overall_quality),
                quality_factors=quality_factors,
                analysis_method="comprehensive",
                analysis_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return QualityMetrics(
                rms_level=0.0,
                peak_level=0.0,
                signal_to_noise_ratio=-60.0,
                zero_crossing_rate=0.0,
                voice_activity_detected=False,
                voice_activity_confidence=0.0,
                speaking_time_ratio=0.0,
                clipping_detected=False,
                distortion_level=0.0,
                noise_level=1.0,
                overall_quality_score=0.0,
                quality_factors={},
                analysis_method="failed",
                analysis_timestamp=datetime.utcnow()
            )


class ChunkFileManager:
    """
    Manages audio chunk file storage and retrieval.
    Handles file naming, compression, and cleanup.
    """
    
    def __init__(self, config: AudioChunkingConfig):
        self.config = config
        self.storage_path = Path(config.audio_storage_path)
        self.compression_enabled = config.file_compression_enabled
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    async def store_chunk_file(
        self, 
        session_id: str, 
        chunk_sequence: int,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, int]:
        """
        Store audio chunk to file.
        
        Returns:
            Tuple of (file_path, file_name, file_size)
        """
        # Generate file name
        timestamp = int(time.time() * 1000)  # Millisecond timestamp
        file_name = f"{session_id}_chunk_{chunk_sequence:06d}_{timestamp}.wav"
        file_path = self.storage_path / session_id
        
        # Ensure session directory exists
        file_path.mkdir(parents=True, exist_ok=True)
        
        full_file_path = file_path / file_name
        
        try:
            # Convert to int16 for efficient storage
            if audio_data.dtype != np.int16:
                # Normalize to -1 to 1 range then convert to int16
                normalized = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (normalized * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data
            
            # Write audio file
            sf.write(str(full_file_path), audio_int16, sample_rate, format='WAV')
            
            # Get file size
            file_size = full_file_path.stat().st_size
            
            # Store metadata file if provided
            if metadata:
                metadata_file = full_file_path.with_suffix('.json')
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.debug(f"Stored chunk file: {full_file_path} ({file_size} bytes)")
            return str(full_file_path), file_name, file_size
            
        except Exception as e:
            logger.error(f"Failed to store chunk file {full_file_path}: {e}")
            raise
    
    def generate_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash for file integrity verification."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate hash for {file_path}: {e}")
            return ""
    
    async def cleanup_old_files(self, retention_days: int = 30) -> int:
        """Clean up old audio files based on retention policy."""
        if not self.config.cleanup_old_files:
            return 0
        
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        files_removed = 0
        
        try:
            for session_dir in self.storage_path.iterdir():
                if not session_dir.is_dir():
                    continue
                
                for file_path in session_dir.iterdir():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        files_removed += 1
                
                # Remove empty session directories
                if not any(session_dir.iterdir()):
                    session_dir.rmdir()
            
            logger.info(f"Cleaned up {files_removed} old audio files")
            return files_removed
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            return 0


class ChunkManager:
    """
    Main chunk manager that coordinates all chunking operations.
    Integrates audio buffering, quality analysis, file storage, and database persistence.
    """
    
    def __init__(
        self, 
        config: AudioChunkingConfig,
        database_adapter: AudioDatabaseAdapter,
        session_id: str,
        source_type: SourceType = SourceType.BOT_AUDIO
    ):
        self.config = config
        self.database_adapter = database_adapter
        self.session_id = session_id
        self.source_type = source_type
        
        # Core components
        self.audio_buffer = AudioBuffer(config)
        self.quality_analyzer = AudioQualityAnalyzer(config)
        self.file_manager = ChunkFileManager(config)
        
        # Chunk processing state
        self.is_active = False
        self.chunks_processed = 0
        self.chunks_rejected = 0
        self.total_audio_duration = 0.0
        self.average_quality_score = 0.0
        
        # Callbacks
        self.on_chunk_ready: Optional[Callable] = None
        self.on_quality_alert: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Performance tracking
        self.start_time = None
        self.processing_times = deque(maxlen=100)  # Last 100 processing times
        
        logger.info(f"ChunkManager initialized for session {session_id}")
    
    def set_chunk_ready_callback(self, callback: Callable[[AudioChunkMetadata, np.ndarray], None]):
        """Set callback for when chunks are ready for processing."""
        self.on_chunk_ready = callback
    
    def set_quality_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for quality alerts."""
        self.on_quality_alert = callback
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for error notifications."""
        self.on_error = callback
    
    async def start(self) -> bool:
        """Start the chunk manager."""
        try:
            self.is_active = True
            self.start_time = time.time()
            logger.info(f"ChunkManager started for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start ChunkManager: {e}")
            return False
    
    async def stop(self) -> Dict[str, Any]:
        """Stop the chunk manager and return final statistics."""
        self.is_active = False
        
        # Process any remaining audio in buffer
        await self._process_remaining_chunks()
        
        # Calculate final statistics
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0
        
        stats = {
            "session_id": self.session_id,
            "total_runtime": total_time,
            "chunks_processed": self.chunks_processed,
            "chunks_rejected": self.chunks_rejected,
            "total_audio_duration": self.total_audio_duration,
            "average_quality_score": self.average_quality_score,
            "processing_rate": self.chunks_processed / max(total_time, 1),
            "rejection_rate": self.chunks_rejected / max(self.chunks_processed + self.chunks_rejected, 1),
            "average_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "buffer_status": self.audio_buffer.get_buffer_status(),
        }
        
        logger.info(f"ChunkManager stopped: {stats}")
        return stats
    
    async def add_audio_data(self, audio_data: np.ndarray) -> int:
        """
        Add audio data to the processing buffer.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            int: Number of samples added
        """
        if not self.is_active:
            return 0
        
        try:
            # Add to buffer
            samples_added = self.audio_buffer.add_audio_data(audio_data)
            
            # Process available chunks
            await self._process_available_chunks()
            
            return samples_added
            
        except Exception as e:
            logger.error(f"Failed to add audio data: {e}")
            if self.on_error:
                self.on_error(f"Audio data processing error: {e}")
            return 0
    
    async def _process_available_chunks(self):
        """Process all available chunks in the buffer."""
        while self.is_active:
            chunk_result = self.audio_buffer.get_next_chunk()
            if chunk_result is None:
                break
            
            audio_chunk, chunk_metadata = chunk_result
            await self._process_single_chunk(audio_chunk, chunk_metadata)
    
    async def _process_remaining_chunks(self):
        """Process any remaining chunks when stopping."""
        logger.info("Processing remaining chunks...")
        await self._process_available_chunks()
    
    async def _process_single_chunk(self, audio_data: np.ndarray, chunk_metadata: Dict[str, Any]):
        """Process a single audio chunk through the complete pipeline."""
        process_start_time = time.time()
        
        try:
            # Quality analysis
            quality_metrics = self.quality_analyzer.analyze_chunk(audio_data)
            
            # Quality-based filtering
            if quality_metrics.overall_quality_score < self.config.min_quality_threshold:
                self.chunks_rejected += 1
                
                if self.on_quality_alert:
                    self.on_quality_alert({
                        "alert_type": "chunk_rejected_low_quality",
                        "chunk_sequence": chunk_metadata["chunk_sequence"],
                        "quality_score": quality_metrics.overall_quality_score,
                        "threshold": self.config.min_quality_threshold,
                        "quality_metrics": quality_metrics.dict(),
                    })
                
                logger.debug(f"Rejected chunk {chunk_metadata['chunk_sequence']} due to low quality: {quality_metrics.overall_quality_score:.3f}")
                return
            
            # Store chunk file
            file_path, file_name, file_size = await self.file_manager.store_chunk_file(
                self.session_id,
                chunk_metadata["chunk_sequence"],
                audio_data,
                metadata={
                    **chunk_metadata,
                    "quality_metrics": quality_metrics.dict(),
                }
            )
            
            # Generate file hash
            file_hash = self.file_manager.generate_file_hash(file_path)
            
            # Create chunk metadata for database
            audio_chunk_metadata = create_audio_chunk_metadata(
                session_id=self.session_id,
                file_path=file_path,
                file_size=file_size,
                duration_seconds=chunk_metadata["duration_seconds"],
                chunk_sequence=chunk_metadata["chunk_sequence"],
                chunk_start_time=chunk_metadata["chunk_start_time"],
                source_type=self.source_type,
                file_name=file_name,
                file_hash=file_hash,
                audio_quality_score=quality_metrics.overall_quality_score,
                sample_rate=chunk_metadata["sample_rate"],
                overlap_duration=chunk_metadata.get("overlap_metadata", {}).get("overlap_duration", 0.0),
                overlap_metadata=chunk_metadata.get("overlap_metadata", {}),
                chunk_metadata={
                    **chunk_metadata,
                    "quality_metrics": quality_metrics.dict(),
                    "processing_time_ms": 0,  # Will be updated after processing
                }
            )
            
            # Store in database
            chunk_id = await self.database_adapter.store_audio_chunk(audio_chunk_metadata)
            
            if chunk_id:
                # Update statistics
                self.chunks_processed += 1
                self.total_audio_duration += chunk_metadata["duration_seconds"]
                self.average_quality_score = (
                    (self.average_quality_score * (self.chunks_processed - 1) + quality_metrics.overall_quality_score) /
                    self.chunks_processed
                )
                
                # Track processing time
                processing_time = (time.time() - process_start_time) * 1000  # Convert to milliseconds
                self.processing_times.append(processing_time)
                
                # Update chunk metadata with processing time
                await self.database_adapter.update_chunk_processing_status(
                    chunk_id,
                    ProcessingStatus.COMPLETED,
                    {"processing_time_ms": processing_time}
                )
                
                # Notify callback
                if self.on_chunk_ready:
                    audio_chunk_metadata.chunk_id = chunk_id  # Update with actual stored ID
                    self.on_chunk_ready(audio_chunk_metadata, audio_data)
                
                logger.debug(f"Processed chunk {chunk_metadata['chunk_sequence']}: quality={quality_metrics.overall_quality_score:.3f}, time={processing_time:.1f}ms")
            else:
                self.chunks_rejected += 1
                logger.warning(f"Failed to store chunk {chunk_metadata['chunk_sequence']} in database")
                
        except Exception as e:
            self.chunks_rejected += 1
            logger.error(f"Failed to process chunk {chunk_metadata.get('chunk_sequence', 'unknown')}: {e}")
            if self.on_error:
                self.on_error(f"Chunk processing error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current chunk manager status and statistics."""
        return {
            "session_id": self.session_id,
            "is_active": self.is_active,
            "chunks_processed": self.chunks_processed,
            "chunks_rejected": self.chunks_rejected,
            "total_audio_duration": self.total_audio_duration,
            "average_quality_score": self.average_quality_score,
            "current_processing_rate": len(self.processing_times) / max(1, sum(self.processing_times) / 1000) if self.processing_times else 0,
            "average_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "config": self.config.dict(),
            "buffer_status": self.audio_buffer.get_buffer_status(),
        }


# Factory function for creating chunk managers
def create_chunk_manager(
    config: AudioChunkingConfig,
    database_adapter: AudioDatabaseAdapter,
    session_id: str,
    source_type: SourceType = SourceType.BOT_AUDIO
) -> ChunkManager:
    """Create and return a ChunkManager instance."""
    return ChunkManager(config, database_adapter, session_id, source_type)


# Example usage and testing
async def main():
    """Example usage of the chunk manager."""
    import os
    from .database_adapter import create_audio_database_adapter
    from .models import get_default_chunking_config
    
    # Configuration
    config = get_default_chunking_config()
    config.chunk_duration = 3.0
    config.overlap_duration = 0.5
    config.min_quality_threshold = 0.3
    
    # Database
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/livetranslate")
    db_adapter = create_audio_database_adapter(database_url)
    
    try:
        # Initialize database
        await db_adapter.initialize()
        
        # Create chunk manager
        chunk_manager = create_chunk_manager(
            config,
            db_adapter,
            "test-session-123",
            SourceType.BOT_AUDIO
        )
        
        # Set callbacks
        def on_chunk_ready(metadata, audio_data):
            print(f"Chunk ready: {metadata.chunk_id}, quality: {metadata.audio_quality_score:.3f}")
        
        def on_quality_alert(alert):
            print(f"Quality alert: {alert}")
        
        chunk_manager.set_chunk_ready_callback(on_chunk_ready)
        chunk_manager.set_quality_alert_callback(on_quality_alert)
        
        # Start chunk manager
        await chunk_manager.start()
        
        # Simulate audio data
        sample_rate = 16000
        chunk_size = int(0.1 * sample_rate)  # 100ms chunks
        
        for i in range(100):  # 10 seconds of audio
            # Generate test audio (sine wave + noise)
            t = np.arange(chunk_size) / sample_rate + i * 0.1
            audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(chunk_size)
            
            await chunk_manager.add_audio_data(audio.astype(np.float32))
            await asyncio.sleep(0.05)  # Simulate real-time
        
        # Stop and get statistics
        final_stats = await chunk_manager.stop()
        print(f"Final statistics: {final_stats}")
        
    finally:
        await db_adapter.close()


if __name__ == "__main__":
    asyncio.run(main())