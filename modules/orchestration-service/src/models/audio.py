"""
Audio processing-related Pydantic models
"""

from enum import Enum
from typing import Any, Self

from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .base import BaseModel, ResponseMixin, TimestampMixin


class AudioFormat(str, Enum):
    """Supported audio formats"""

    WAV = "wav"
    MP3 = "mp3"
    WEBM = "webm"
    OGG = "ogg"
    MP4 = "mp4"
    FLAC = "flac"


class ProcessingStage(str, Enum):
    """Audio processing stages"""

    VAD = "vad"
    VOICE_FILTER = "voice_filter"
    NOISE_REDUCTION = "noise_reduction"
    VOICE_ENHANCEMENT = "voice_enhancement"
    NORMALIZATION = "normalization"
    COMPRESSOR = "compressor"
    DE_ESSER = "de_esser"
    EQ = "equalizer"
    LIMITER = "limiter"
    OUTPUT = "output"


class ProcessingQuality(str, Enum):
    """Processing quality levels"""

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class AudioConfiguration(BaseModel):
    """Audio processing configuration"""

    # Basic settings
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz",
        ge=8000,
        le=48000,
        json_schema_extra={"example": 16000},
    )
    channels: int = Field(
        default=1,
        description="Number of audio channels",
        ge=1,
        le=2,
        json_schema_extra={"example": 1},
    )
    bit_depth: int = Field(
        default=16, description="Audio bit depth", json_schema_extra={"example": 16}
    )
    format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio format")

    # Processing stages
    enabled_stages: list[ProcessingStage] = Field(
        default_factory=lambda: [
            ProcessingStage.VAD,
            ProcessingStage.VOICE_FILTER,
            ProcessingStage.NOISE_REDUCTION,
            ProcessingStage.VOICE_ENHANCEMENT,
        ],
        description="Enabled processing stages",
    )

    # Quality settings
    quality: ProcessingQuality = Field(
        default=ProcessingQuality.BALANCED, description="Processing quality level"
    )

    # VAD settings
    vad_aggressiveness: int = Field(
        default=2, description="VAD aggressiveness level (0-3)", ge=0, le=3
    )
    vad_energy_threshold: float = Field(
        default=0.01, description="VAD energy threshold", ge=0.0, le=1.0
    )

    # Voice filter settings
    voice_freq_min: float = Field(
        default=85.0, description="Minimum voice frequency in Hz", ge=50.0, le=150.0
    )
    voice_freq_max: float = Field(
        default=300.0, description="Maximum voice frequency in Hz", ge=200.0, le=500.0
    )

    # Noise reduction settings
    noise_reduction_strength: float = Field(
        default=0.5, description="Noise reduction strength (0-1)", ge=0.0, le=1.0
    )
    voice_protection: float = Field(
        default=0.8, description="Voice protection level (0-1)", ge=0.0, le=1.0
    )

    # Enhancement settings
    compressor_threshold: float = Field(
        default=-20.0, description="Compressor threshold in dB", ge=-60.0, le=0.0
    )
    compressor_ratio: float = Field(default=4.0, description="Compressor ratio", ge=1.0, le=20.0)
    clarity_enhancement: float = Field(
        default=0.3, description="Clarity enhancement level (0-1)", ge=0.0, le=1.0
    )

    @field_validator("enabled_stages")
    @classmethod
    def validate_stages_order(
        cls, v: list[ProcessingStage], info: ValidationInfo | None = None
    ) -> list[ProcessingStage]:
        """Validate processing stages are in correct order"""
        stage_order = [
            ProcessingStage.VAD,
            ProcessingStage.VOICE_FILTER,
            ProcessingStage.NOISE_REDUCTION,
            ProcessingStage.VOICE_ENHANCEMENT,
            ProcessingStage.NORMALIZATION,
            ProcessingStage.COMPRESSOR,
            ProcessingStage.DE_ESSER,
            ProcessingStage.EQ,
            ProcessingStage.LIMITER,
            ProcessingStage.OUTPUT,
        ]

        # Check if stages are in correct order
        last_index = -1
        for stage in v:
            try:
                current_index = stage_order.index(stage)
                if current_index < last_index:
                    raise ValueError(f"Stage {stage} is out of order")
                last_index = current_index
            except ValueError as e:
                if "is not in list" in str(e):
                    raise ValueError(f"Unknown stage: {stage}") from e
                raise

        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sample_rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "format": "wav",
                "enabled_stages": [
                    "vad",
                    "voice_filter",
                    "noise_reduction",
                    "voice_enhancement",
                ],
                "quality": "balanced",
                "vad_aggressiveness": 2,
                "vad_energy_threshold": 0.01,
                "voice_freq_min": 85.0,
                "voice_freq_max": 300.0,
                "noise_reduction_strength": 0.5,
                "voice_protection": 0.8,
                "compressor_threshold": -20.0,
                "compressor_ratio": 4.0,
                "clarity_enhancement": 0.3,
            }
        }
    )


class AudioProcessingRequest(BaseModel):
    """Audio processing request"""

    # Audio data (base64 encoded or file reference)
    audio_data: str | None = Field(default=None, description="Base64 encoded audio data")
    audio_url: str | None = Field(default=None, description="URL to audio file")
    file_upload: str | None = Field(default=None, description="File upload reference")

    # Processing configuration
    config: AudioConfiguration = Field(
        default_factory=AudioConfiguration, description="Audio processing configuration"
    )

    # Processing options
    streaming: bool = Field(default=False, description="Enable streaming processing")
    real_time: bool = Field(default=False, description="Enable real-time processing")
    transcription: bool = Field(default=True, description="Enable transcription")
    speaker_diarization: bool = Field(default=True, description="Enable speaker diarization")

    # Session information
    session_id: str | None = Field(default=None, description="Session identifier")
    user_id: str | None = Field(default=None, description="User identifier")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode="after")
    def validate_audio_source(self) -> Self:
        """Ensure at least one audio source is provided"""
        audio_sources = [
            self.audio_data,
            self.audio_url,
            self.file_upload,
        ]

        provided_sources = [source for source in audio_sources if source is not None]

        if not provided_sources:
            raise ValueError("At least one audio source must be provided")

        if len(provided_sources) > 1:
            raise ValueError("Only one audio source can be provided")

        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAEQr...",
                "config": {
                    "sample_rate": 16000,
                    "format": "wav",
                    "quality": "balanced",
                },
                "streaming": False,
                "real_time": False,
                "transcription": True,
                "speaker_diarization": True,
                "session_id": "session_abc123",
                "user_id": "user_def456",
                "metadata": {"source": "microphone", "device": "web_browser"},
            }
        }
    )


class AudioProcessingResult(BaseModel):
    """Audio processing result for a single stage"""

    stage: ProcessingStage = Field(description="Processing stage")
    success: bool = Field(description="Whether stage completed successfully")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    input_level_db: float | None = Field(default=None, description="Input audio level in dB")
    output_level_db: float | None = Field(default=None, description="Output audio level in dB")
    quality_score: float | None = Field(
        default=None, description="Quality score (0-1)", ge=0.0, le=1.0
    )
    artifacts_detected: bool = Field(default=False, description="Whether artifacts were detected")
    stage_specific_data: dict[str, Any] = Field(
        default_factory=dict, description="Stage-specific processing data"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "stage": "noise_reduction",
                "success": True,
                "processing_time_ms": 45.2,
                "input_level_db": -18.5,
                "output_level_db": -16.2,
                "quality_score": 0.85,
                "artifacts_detected": False,
                "stage_specific_data": {
                    "noise_reduction_db": 12.3,
                    "speech_preservation": 0.92,
                },
            }
        }
    )


class SpeakerInfo(BaseModel):
    """Speaker information from diarization"""

    speaker_id: str = Field(
        description="Speaker identifier", json_schema_extra={"example": "speaker_1"}
    )
    start_time: float = Field(
        description="Start time in seconds", json_schema_extra={"example": 0.5}
    )
    end_time: float = Field(description="End time in seconds", json_schema_extra={"example": 3.2})
    confidence: float = Field(
        description="Speaker confidence score (0-1)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.95},
    )
    text: str | None = Field(
        default=None,
        description="Transcribed text for this speaker segment",
        json_schema_extra={"example": "Hello, this is a test."},
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "speaker_id": "speaker_1",
                "start_time": 0.5,
                "end_time": 3.2,
                "confidence": 0.95,
                "text": "Hello, this is a test.",
            }
        }
    )


class TranscriptionResult(BaseModel):
    """Transcription result"""

    text: str = Field(
        description="Transcribed text",
        json_schema_extra={"example": "Hello, this is a test transcription."},
    )
    confidence: float = Field(
        description="Overall transcription confidence (0-1)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92},
    )
    language: str = Field(description="Detected language code", json_schema_extra={"example": "en"})
    speakers: list[SpeakerInfo] = Field(
        default_factory=list, description="Speaker diarization results"
    )
    processing_time_ms: float = Field(
        description="Transcription processing time in milliseconds",
        json_schema_extra={"example": 234.5},
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Hello, this is a test transcription.",
                "confidence": 0.92,
                "language": "en",
                "speakers": [
                    {
                        "speaker_id": "speaker_1",
                        "start_time": 0.0,
                        "end_time": 5.5,
                        "confidence": 0.95,
                        "text": "Hello, this is a test transcription.",
                    }
                ],
                "processing_time_ms": 234.5,
            }
        }
    )


class AudioProcessingResponse(ResponseMixin, TimestampMixin):
    """Audio processing response"""

    request_id: str = Field(
        description="Request identifier", json_schema_extra={"example": "req_abc123def456"}
    )
    session_id: str | None = Field(default=None, description="Session identifier")

    # Processing results
    stage_results: list[AudioProcessingResult] = Field(
        description="Results from each processing stage"
    )
    overall_quality_score: float = Field(
        description="Overall audio quality score (0-1)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.87},
    )
    total_processing_time_ms: float = Field(
        description="Total processing time in milliseconds", json_schema_extra={"example": 145.7}
    )

    # Audio metrics
    input_duration_s: float = Field(
        description="Input audio duration in seconds", json_schema_extra={"example": 5.5}
    )
    output_duration_s: float = Field(
        description="Output audio duration in seconds", json_schema_extra={"example": 5.3}
    )
    signal_to_noise_ratio: float | None = Field(
        default=None, description="Signal-to-noise ratio in dB", json_schema_extra={"example": 15.2}
    )

    # Transcription
    transcription: TranscriptionResult | None = Field(
        default=None, description="Transcription result if enabled"
    )

    # Output
    output_audio_url: str | None = Field(default=None, description="URL to processed audio file")
    output_audio_data: str | None = Field(
        default=None, description="Base64 encoded processed audio data"
    )

    # Streaming support
    is_streaming: bool = Field(default=False, description="Whether this is a streaming response")
    stream_chunk_id: int | None = Field(default=None, description="Streaming chunk identifier")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Audio processed successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_abc123def456",
                "session_id": "session_abc123",
                "stage_results": [
                    {
                        "stage": "vad",
                        "success": True,
                        "processing_time_ms": 15.2,
                        "quality_score": 0.92,
                    },
                    {
                        "stage": "noise_reduction",
                        "success": True,
                        "processing_time_ms": 45.2,
                        "quality_score": 0.85,
                    },
                ],
                "overall_quality_score": 0.87,
                "total_processing_time_ms": 145.7,
                "input_duration_s": 5.5,
                "output_duration_s": 5.3,
                "signal_to_noise_ratio": 15.2,
                "transcription": {
                    "text": "Hello, this is a test transcription.",
                    "confidence": 0.92,
                    "language": "en",
                    "speakers": [],
                    "processing_time_ms": 234.5,
                },
                "output_audio_url": "/api/audio/processed/req_abc123def456.wav",
                "is_streaming": False,
            }
        }
    )


class AudioStats(BaseModel):
    """Audio processing statistics"""

    total_requests: int = Field(
        description="Total processing requests", json_schema_extra={"example": 1250}
    )
    successful_requests: int = Field(
        description="Successful processing requests", json_schema_extra={"example": 1200}
    )
    failed_requests: int = Field(
        description="Failed processing requests", json_schema_extra={"example": 50}
    )
    average_processing_time_ms: float = Field(
        description="Average processing time in milliseconds", json_schema_extra={"example": 156.7}
    )
    average_quality_score: float = Field(
        description="Average quality score (0-1)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.84},
    )
    total_audio_duration_s: float = Field(
        description="Total audio processed in seconds", json_schema_extra={"example": 12500.5}
    )
    popular_formats: dict[str, int] = Field(
        description="Usage count by audio format",
        json_schema_extra={"example": {"wav": 800, "mp3": 350, "webm": 100}},
    )
    stage_performance: dict[str, float] = Field(
        description="Average processing time by stage (ms)",
        json_schema_extra={
            "example": {"vad": 15.2, "noise_reduction": 45.3, "voice_enhancement": 67.1}
        },
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_requests": 1250,
                "successful_requests": 1200,
                "failed_requests": 50,
                "average_processing_time_ms": 156.7,
                "average_quality_score": 0.84,
                "total_audio_duration_s": 12500.5,
                "popular_formats": {"wav": 800, "mp3": 350, "webm": 100},
                "stage_performance": {
                    "vad": 15.2,
                    "noise_reduction": 45.3,
                    "voice_enhancement": 67.1,
                },
            }
        }
    )
