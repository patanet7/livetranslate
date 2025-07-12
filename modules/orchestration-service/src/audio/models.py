#!/usr/bin/env python3
"""
Audio Processing Data Models - Orchestration Service

Pydantic models for the centralized audio chunking and processing system.
Provides comprehensive data validation and serialization for all audio pipeline components.

Models:
- AudioChunkMetadata: Complete audio chunk information with database persistence
- SpeakerCorrelation: Speaker mapping between whisper and Google Meet
- ProcessingResult: Unified result format for all processing stages
- AudioChunkingConfig: Configuration parameters for chunking system
- ChunkLineage: Track processing lineage and dependencies
- QualityMetrics: Audio quality assessment metrics
"""

import time
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    WEBM = "webm"
    OGG = "ogg"
    MP4 = "mp4"


class ProcessingStatus(str, Enum):
    """Audio processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class SourceType(str, Enum):
    """Audio source types."""
    BOT_AUDIO = "bot_audio"
    FRONTEND_TEST = "frontend_test"
    GOOGLE_MEET = "google_meet"
    MANUAL_UPLOAD = "manual_upload"


class CorrelationType(str, Enum):
    """Speaker correlation types."""
    EXACT = "exact"
    INTERPOLATED = "interpolated"
    INFERRED = "inferred"
    TEMPORAL = "temporal"
    MANUAL = "manual"
    FALLBACK = "fallback"
    GOOGLE_MEET_API = "google_meet_api"
    ACOUSTIC = "acoustic"


class AudioChunkMetadata(BaseModel):
    """
    Complete metadata for audio chunks in the centralized processing system.
    Corresponds to bot_sessions.audio_files table structure.
    """
    
    # Primary identifiers
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk identifier")
    session_id: str = Field(..., description="Bot session identifier")
    
    # File information
    file_path: str = Field(..., description="Storage path for audio file")
    file_name: str = Field(..., description="Audio file name")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio file format")
    file_hash: Optional[str] = Field(None, description="SHA256 hash for integrity verification")
    
    # Audio properties
    duration_seconds: float = Field(..., ge=0, description="Audio duration in seconds")
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Audio sample rate")
    channels: int = Field(default=1, ge=1, le=2, description="Number of audio channels")
    
    # Chunking information
    chunk_sequence: int = Field(..., ge=0, description="Sequence number in session")
    chunk_start_time: float = Field(..., description="Chunk start timestamp (relative to session)")
    chunk_end_time: float = Field(..., description="Chunk end timestamp (relative to session)")
    overlap_duration: float = Field(default=0.0, ge=0, description="Overlap with previous chunk")
    
    # Processing information
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    source_type: SourceType = Field(..., description="Source of audio data")
    audio_quality_score: float = Field(default=0.0, ge=0, le=1, description="Quality assessment score")
    
    # Metadata and configuration
    processing_pipeline_version: str = Field(default="1.0", description="Pipeline version for compatibility")
    overlap_metadata: Dict[str, Any] = Field(default_factory=dict, description="Overlap processing details")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk information")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator("chunk_end_time")
    def validate_time_order(cls, v, values):
        """Ensure chunk_end_time > chunk_start_time."""
        if "chunk_start_time" in values and v <= values["chunk_start_time"]:
            raise ValueError("chunk_end_time must be greater than chunk_start_time")
        return v
    
    @validator("duration_seconds")
    def validate_duration_consistency(cls, v, values):
        """Ensure duration matches chunk timing."""
        if "chunk_start_time" in values and "chunk_end_time" in values:
            expected_duration = values["chunk_end_time"] - values["chunk_start_time"]
            if abs(v - expected_duration) > 0.1:  # 100ms tolerance
                raise ValueError(f"Duration {v}s doesn't match chunk timing {expected_duration}s")
        return v


class SpeakerCorrelation(BaseModel):
    """
    Speaker correlation data between whisper service and Google Meet speakers.
    Corresponds to bot_sessions.correlations table structure.
    """
    
    # Primary identifiers
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(..., description="Bot session identifier")
    
    # Speaker mapping
    whisper_speaker_id: str = Field(..., description="Whisper service speaker ID")
    google_meet_speaker_id: Optional[str] = Field(None, description="Google Meet speaker ID")
    google_meet_speaker_name: Optional[str] = Field(None, description="Google Meet speaker display name")
    external_speaker_id: Optional[str] = Field(None, description="External system speaker ID")
    external_speaker_name: Optional[str] = Field(None, description="External system speaker name")
    
    # Correlation details
    correlation_confidence: float = Field(..., ge=0, le=1, description="Confidence in correlation")
    correlation_type: CorrelationType = Field(..., description="Type of correlation method used")
    correlation_method: str = Field(..., description="Specific algorithm or method used")
    
    # Temporal information
    start_timestamp: float = Field(..., description="Start time of correlation window")
    end_timestamp: float = Field(..., description="End time of correlation window")
    timing_offset: float = Field(default=0.0, description="Timing offset between sources")
    
    # Evidence and metadata
    text_similarity_score: Optional[float] = Field(None, ge=0, le=1, description="Text similarity evidence")
    temporal_alignment_score: Optional[float] = Field(None, ge=0, le=1, description="Temporal alignment evidence")
    historical_pattern_score: Optional[float] = Field(None, ge=0, le=1, description="Historical pattern evidence")
    correlation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional correlation data")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator("end_timestamp")
    def validate_timestamp_order(cls, v, values):
        """Ensure end_timestamp > start_timestamp."""
        if "start_timestamp" in values and v <= values["start_timestamp"]:
            raise ValueError("end_timestamp must be greater than start_timestamp")
        return v


class ProcessingResult(BaseModel):
    """
    Unified result format for all audio processing stages.
    Used for communication between components and API responses.
    """
    
    # Processing identifiers
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str = Field(..., description="Source chunk identifier")
    session_id: str = Field(..., description="Bot session identifier")
    processing_stage: str = Field(..., description="Processing stage (chunking, transcription, translation)")
    
    # Result data
    status: ProcessingStatus = Field(..., description="Processing status")
    result_data: Dict[str, Any] = Field(default_factory=dict, description="Stage-specific result data")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    # Performance metrics
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time in milliseconds")
    quality_score: float = Field(default=0.0, ge=0, le=1, description="Result quality score")
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="Result confidence score")
    
    # Lineage tracking
    input_chunk_ids: List[str] = Field(default_factory=list, description="Input chunk dependencies")
    output_chunk_ids: List[str] = Field(default_factory=list, description="Generated output chunks")
    correlation_ids: List[str] = Field(default_factory=list, description="Related correlations")
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    
    @validator("completed_at")
    def validate_completion_time(cls, v, values):
        """Ensure completed_at >= started_at."""
        if v and "started_at" in values and v < values["started_at"]:
            raise ValueError("completed_at must be greater than or equal to started_at")
        return v


class AudioChunkingConfig(BaseModel):
    """
    Configuration parameters for the centralized audio chunking system.
    Hot-reloadable configuration with validation.
    """
    
    # Chunking parameters
    chunk_duration: float = Field(default=3.0, ge=0.5, le=10.0, description="Chunk duration in seconds")
    overlap_duration: float = Field(default=0.5, ge=0.0, le=2.0, description="Overlap between chunks")
    processing_interval: float = Field(default=2.5, ge=0.1, le=5.0, description="Processing interval")
    buffer_duration: float = Field(default=10.0, ge=5.0, le=60.0, description="Rolling buffer duration")
    
    # Quality thresholds
    min_quality_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum quality for processing")
    silence_threshold: float = Field(default=0.01, ge=0.0, le=0.1, description="Silence detection threshold")
    noise_threshold: float = Field(default=0.02, ge=0.0, le=0.1, description="Noise level threshold")
    
    # Speaker correlation settings
    speaker_correlation_enabled: bool = Field(default=True, description="Enable speaker correlation")
    correlation_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum correlation confidence")
    correlation_temporal_window: float = Field(default=2.0, ge=0.5, le=10.0, description="Temporal correlation window")
    
    # Database integration settings
    store_audio_files: bool = Field(default=True, description="Store audio files in database")
    store_transcripts: bool = Field(default=True, description="Store transcripts in database")
    store_translations: bool = Field(default=True, description="Store translations in database")
    store_correlations: bool = Field(default=True, description="Store speaker correlations")
    track_chunk_lineage: bool = Field(default=True, description="Track chunk processing lineage")
    
    # Performance settings
    max_concurrent_chunks: int = Field(default=10, ge=1, le=50, description="Maximum concurrent chunk processing")
    chunk_processing_timeout: float = Field(default=30.0, ge=5.0, le=120.0, description="Chunk processing timeout")
    database_batch_size: int = Field(default=100, ge=1, le=1000, description="Database batch operation size")
    
    # File storage settings
    audio_storage_path: str = Field(default="/data/audio", description="Base path for audio file storage")
    file_compression_enabled: bool = Field(default=True, description="Enable audio file compression")
    cleanup_old_files: bool = Field(default=True, description="Automatically cleanup old files")
    file_retention_days: int = Field(default=30, ge=1, le=365, description="File retention period in days")
    
    @root_validator(skip_on_failure=True)
    def validate_timing_consistency(cls, values):
        """Ensure timing parameters are consistent."""
        chunk_duration = values.get("chunk_duration", 3.0)
        overlap_duration = values.get("overlap_duration", 0.5)
        processing_interval = values.get("processing_interval", 2.5)
        
        if overlap_duration >= chunk_duration:
            raise ValueError("overlap_duration must be less than chunk_duration")
            
        if processing_interval >= chunk_duration + overlap_duration:
            raise ValueError("processing_interval should be less than chunk_duration + overlap_duration")
            
        return values


class ChunkLineage(BaseModel):
    """
    Track processing lineage and dependencies between chunks.
    Enables debugging and quality analysis of the processing pipeline.
    """
    
    # Lineage identifiers
    lineage_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(..., description="Bot session identifier")
    
    # Chunk relationships
    source_chunk_id: str = Field(..., description="Source chunk that generated this lineage")
    derived_chunk_ids: List[str] = Field(default_factory=list, description="Chunks derived from source")
    parent_lineage_ids: List[str] = Field(default_factory=list, description="Parent lineage entries")
    
    # Processing information
    processing_stage: str = Field(..., description="Processing stage that created this lineage")
    transformation_type: str = Field(..., description="Type of transformation applied")
    processing_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used")
    
    # Quality and performance
    input_quality_score: float = Field(default=0.0, ge=0, le=1, description="Input quality score")
    output_quality_score: float = Field(default=0.0, ge=0, le=1, description="Output quality score")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time in milliseconds")
    
    # Metadata
    lineage_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional lineage information")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QualityMetrics(BaseModel):
    """
    Comprehensive audio quality assessment metrics.
    Used for monitoring and optimization of the audio processing pipeline.
    """
    
    # Basic quality metrics
    rms_level: float = Field(..., ge=0, description="RMS audio level")
    peak_level: float = Field(..., ge=0, le=1, description="Peak audio level")
    signal_to_noise_ratio: float = Field(..., description="Estimated SNR in dB")
    zero_crossing_rate: float = Field(..., ge=0, description="Zero crossing rate")
    
    # Voice activity metrics
    voice_activity_detected: bool = Field(..., description="Voice activity detection result")
    voice_activity_confidence: float = Field(default=0.0, ge=0, le=1, description="VAD confidence")
    speaking_time_ratio: float = Field(default=0.0, ge=0, le=1, description="Ratio of speaking time")
    
    # Distortion metrics
    clipping_detected: bool = Field(default=False, description="Audio clipping detection")
    distortion_level: float = Field(default=0.0, ge=0, le=1, description="Overall distortion level")
    noise_level: float = Field(default=0.0, ge=0, le=1, description="Background noise level")
    
    # Frequency analysis
    spectral_centroid: Optional[float] = Field(None, description="Spectral centroid frequency")
    spectral_bandwidth: Optional[float] = Field(None, description="Spectral bandwidth")
    spectral_rolloff: Optional[float] = Field(None, description="Spectral rolloff frequency")
    
    # Overall quality
    overall_quality_score: float = Field(..., ge=0, le=1, description="Overall quality assessment")
    quality_factors: Dict[str, float] = Field(default_factory=dict, description="Detailed quality factors")
    
    # Assessment metadata
    analysis_method: str = Field(default="standard", description="Quality analysis method used")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


class AudioStreamingSession(BaseModel):
    """
    Audio streaming session configuration for real-time processing.
    Used for managing active audio streams through the coordination system.
    """
    
    # Session identifiers
    streaming_session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    bot_session_id: str = Field(..., description="Parent bot session identifier")
    
    # Stream configuration
    source_type: SourceType = Field(..., description="Audio stream source")
    chunk_config: AudioChunkingConfig = Field(..., description="Chunking configuration")
    target_languages: List[str] = Field(default_factory=list, description="Target translation languages")
    
    # Processing settings
    real_time_processing: bool = Field(default=True, description="Enable real-time processing")
    speaker_correlation_enabled: bool = Field(default=True, description="Enable speaker correlation")
    quality_monitoring_enabled: bool = Field(default=True, description="Enable quality monitoring")
    
    # Stream state
    stream_status: str = Field(default="initialized", description="Current stream status")
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    total_duration: float = Field(default=0.0, description="Total stream duration")
    
    # Performance metrics
    average_processing_time: float = Field(default=0.0, description="Average chunk processing time")
    average_quality_score: float = Field(default=0.0, description="Average quality score")
    error_count: int = Field(default=0, description="Number of processing errors")
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = Field(None)


# Factory functions for creating commonly used models
def create_audio_chunk_metadata(
    session_id: str,
    file_path: str,
    file_size: int,
    duration_seconds: float,
    chunk_sequence: int,
    chunk_start_time: float,
    source_type: SourceType = SourceType.BOT_AUDIO,
    **kwargs
) -> AudioChunkMetadata:
    """Create AudioChunkMetadata with required fields and sensible defaults."""
    
    file_name = file_path.split("/")[-1] if "/" in file_path else file_path
    chunk_end_time = chunk_start_time + duration_seconds
    
    return AudioChunkMetadata(
        session_id=session_id,
        file_path=file_path,
        file_name=file_name,
        file_size=file_size,
        duration_seconds=duration_seconds,
        chunk_sequence=chunk_sequence,
        chunk_start_time=chunk_start_time,
        chunk_end_time=chunk_end_time,
        source_type=source_type,
        **kwargs
    )


def create_speaker_correlation(
    session_id: str,
    whisper_speaker_id: str,
    correlation_confidence: float,
    correlation_type: CorrelationType,
    start_timestamp: float,
    end_timestamp: float,
    correlation_method: str = "temporal_alignment",
    **kwargs
) -> SpeakerCorrelation:
    """Create SpeakerCorrelation with required fields."""
    
    return SpeakerCorrelation(
        session_id=session_id,
        whisper_speaker_id=whisper_speaker_id,
        correlation_confidence=correlation_confidence,
        correlation_type=correlation_type,
        correlation_method=correlation_method,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        **kwargs
    )


def create_processing_result(
    chunk_id: str,
    session_id: str,
    processing_stage: str,
    status: ProcessingStatus = ProcessingStatus.COMPLETED,
    **kwargs
) -> ProcessingResult:
    """Create ProcessingResult with required fields."""
    
    return ProcessingResult(
        chunk_id=chunk_id,
        session_id=session_id,
        processing_stage=processing_stage,
        status=status,
        **kwargs
    )


# Configuration validation and loading
def load_chunking_config(config_dict: Dict[str, Any]) -> AudioChunkingConfig:
    """Load and validate audio chunking configuration from dictionary."""
    return AudioChunkingConfig(**config_dict)


def get_default_chunking_config() -> AudioChunkingConfig:
    """Get default audio chunking configuration."""
    return AudioChunkingConfig()


if __name__ == "__main__":
    # Example usage and validation
    config = get_default_chunking_config()
    print(f"Default config: {config.json(indent=2)}")
    
    # Example chunk metadata
    chunk = create_audio_chunk_metadata(
        session_id="test-session-123",
        file_path="/data/audio/chunk_001.wav",
        file_size=64000,
        duration_seconds=3.0,
        chunk_sequence=1,
        chunk_start_time=0.0,
        audio_quality_score=0.85
    )
    print(f"Example chunk: {chunk.json(indent=2)}")
    
    # Example speaker correlation
    correlation = create_speaker_correlation(
        session_id="test-session-123",
        whisper_speaker_id="speaker_0",
        correlation_confidence=0.92,
        correlation_type=CorrelationType.TEMPORAL,
        start_timestamp=0.0,
        end_timestamp=3.0,
        google_meet_speaker_name="John Doe"
    )
    print(f"Example correlation: {correlation.json(indent=2)}")