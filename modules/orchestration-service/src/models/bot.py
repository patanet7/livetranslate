"""
Bot management-related Pydantic models
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from .base import BaseModel, IDMixin, ResponseMixin, TimestampMixin


class BotStatus(str, Enum):
    """Bot status enumeration"""

    SPAWNING = "spawning"
    JOINING = "joining"
    ACTIVE = "active"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class BotPriority(str, Enum):
    """Bot priority levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MeetingPlatform(str, Enum):
    """Supported meeting platforms"""

    GOOGLE_MEET = "google_meet"
    ZOOM = "zoom"
    TEAMS = "teams"
    WEBEX = "webex"


class WebcamDisplayMode(str, Enum):
    """Virtual webcam display modes"""

    OVERLAY = "overlay"
    SIDEBAR = "sidebar"
    FULLSCREEN = "fullscreen"
    PICTURE_IN_PICTURE = "picture_in_picture"


class WebcamTheme(str, Enum):
    """Virtual webcam themes"""

    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"
    CUSTOM = "custom"


class MeetingInfo(BaseModel):
    """Meeting information"""

    meeting_id: str = Field(
        alias="meetingId",
        description="Meeting ID (e.g., Google Meet code)",
        json_schema_extra={"example": "abc-defg-hij"},
    )
    meeting_title: str | None = Field(
        default=None,
        alias="meetingTitle",
        description="Meeting title",
        json_schema_extra={"example": "Weekly Team Standup"},
    )
    meeting_url: str | None = Field(
        default=None,
        alias="meetingUrl",
        description="Full meeting URL",
        json_schema_extra={"example": "https://meet.google.com/abc-defg-hij"},
    )
    platform: MeetingPlatform = Field(
        default=MeetingPlatform.GOOGLE_MEET, description="Meeting platform"
    )
    organizer_email: str | None = Field(
        default=None,
        alias="organizerEmail",
        description="Organizer email address",
        json_schema_extra={"example": "organizer@example.com"},
    )
    scheduled_start: datetime | None = Field(
        default=None, alias="scheduledStart", description="Scheduled meeting start time"
    )
    scheduled_duration_minutes: int | None = Field(
        default=None,
        alias="scheduledDurationMinutes",
        description="Scheduled duration in minutes",
        json_schema_extra={"example": 60},
    )
    participant_count: int = Field(
        default=0,
        alias="participantCount",
        description="Current participant count",
        json_schema_extra={"example": 5},
    )

    @field_validator("meeting_id")
    @classmethod
    def validate_meeting_id(cls, v: str, info: ValidationInfo | None = None) -> str:
        """Validate meeting ID format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Meeting ID cannot be empty")

        # Basic validation for Google Meet format
        if len(v) >= 3 and "-" in v:
            parts = v.split("-")
            if len(parts) >= 3:
                return v

        # Allow other formats but ensure not empty
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "meeting_id": "abc-defg-hij",
                "meeting_title": "Weekly Team Standup",
                "meeting_url": "https://meet.google.com/abc-defg-hij",
                "platform": "google_meet",
                "organizer_email": "organizer@example.com",
                "scheduled_start": "2024-01-15T15:00:00Z",
                "scheduled_duration_minutes": 60,
                "participant_count": 5,
            }
        }
    )


class AudioCaptureConfig(BaseModel):
    """Audio capture configuration"""

    device_id: str | None = Field(
        default=None, alias="deviceId", description="Audio device identifier"
    )
    sample_rate: int = Field(
        default=16000,
        alias="sampleRate",
        description="Audio sample rate in Hz",
        ge=8000,
        le=48000,
    )
    channels: int = Field(default=1, description="Number of audio channels", ge=1, le=2)
    chunk_size: int = Field(
        default=1024, alias="chunkSize", description="Audio chunk size", ge=256, le=4096
    )
    enable_noise_suppression: bool = Field(
        default=True,
        alias="enableNoiseSuppression",
        description="Enable noise suppression",
    )
    enable_echo_cancellation: bool = Field(
        default=True,
        alias="enableEchoCancellation",
        description="Enable echo cancellation",
    )
    enable_auto_gain: bool = Field(
        default=True,
        alias="enableAutoGain",
        description="Enable automatic gain control",
    )


class TranslationConfig(BaseModel):
    """Translation configuration"""

    target_languages: list[str] = Field(
        default_factory=lambda: ["en", "es"],
        alias="targetLanguages",
        description="Target language codes",
        json_schema_extra={"example": ["en", "es", "fr"]},
    )
    source_language: str | None = Field(
        default=None,
        alias="sourceLanguage",
        description="Source language code (auto-detect if None)",
        json_schema_extra={"example": "en"},
    )
    enable_auto_translation: bool = Field(
        default=True,
        alias="enableAutoTranslation",
        description="Enable automatic translation",
    )
    translation_quality: str = Field(
        default="balanced",
        alias="translationQuality",
        description="Translation quality setting",
        json_schema_extra={"example": "balanced"},
    )
    real_time_translation: bool = Field(
        default=True,
        alias="realTimeTranslation",
        description="Enable real-time translation",
    )

    @field_validator("target_languages")
    @classmethod
    def validate_target_languages(
        cls, v: list[str], info: ValidationInfo | None = None
    ) -> list[str]:
        """Validate target languages list"""
        if not v:
            raise ValueError("At least one target language must be specified")

        # Check for valid language codes (basic validation)
        valid_codes = {
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "zh",
            "ja",
            "ko",
            "ar",
            "hi",
            "th",
            "vi",
            "nl",
            "sv",
            "da",
            "no",
            "fi",
            "pl",
        }

        for lang in v:
            if lang not in valid_codes:
                raise ValueError(f"Unsupported language code: {lang}")

        return list(set(v))  # Remove duplicates

    @field_validator("translation_quality")
    @classmethod
    def validate_quality(cls, v: str, info: ValidationInfo | None = None) -> str:
        """Validate translation quality"""
        valid_qualities = ["fast", "balanced", "accurate"]
        if v not in valid_qualities:
            raise ValueError(f"Quality must be one of: {valid_qualities}")
        return v


class WebcamConfig(BaseModel):
    """Virtual webcam configuration"""

    width: int = Field(default=1280, description="Webcam width in pixels", ge=640, le=1920)
    height: int = Field(default=720, description="Webcam height in pixels", ge=480, le=1080)
    fps: int = Field(default=30, description="Frames per second", ge=15, le=60)
    display_mode: WebcamDisplayMode = Field(
        default=WebcamDisplayMode.OVERLAY, description="Display mode for translations"
    )
    theme: WebcamTheme = Field(default=WebcamTheme.DARK, description="Visual theme")
    max_translations_displayed: int = Field(
        default=5, description="Maximum number of translations to display", ge=1, le=10
    )
    font_size: int = Field(default=16, description="Font size for translations", ge=10, le=32)
    background_opacity: float = Field(
        default=0.8, description="Background opacity (0-1)", ge=0.0, le=1.0
    )


class BotConfiguration(BaseModel):
    """Complete bot configuration"""

    meeting_info: MeetingInfo = Field(description="Meeting information")
    audio_capture: AudioCaptureConfig = Field(
        default_factory=AudioCaptureConfig, description="Audio capture configuration"
    )
    translation: TranslationConfig = Field(
        default_factory=TranslationConfig, description="Translation configuration"
    )
    webcam: WebcamConfig = Field(
        default_factory=WebcamConfig, description="Virtual webcam configuration"
    )
    priority: BotPriority = Field(default=BotPriority.MEDIUM, description="Bot priority level")
    auto_terminate_minutes: int | None = Field(
        default=180,
        description="Auto-terminate after minutes (None = no limit)",
        ge=5,
        le=480,
    )
    enable_recording: bool = Field(default=True, description="Enable audio recording")
    enable_transcription: bool = Field(default=True, description="Enable transcription")
    enable_speaker_diarization: bool = Field(default=True, description="Enable speaker diarization")
    enable_virtual_webcam: bool = Field(default=True, description="Enable virtual webcam output")


class BotSpawnRequest(BaseModel):
    """Bot spawn request"""

    config: BotConfiguration = Field(description="Bot configuration")
    user_id: str | None = Field(default=None, alias="userId", description="User identifier")
    session_id: str | None = Field(
        default=None, alias="sessionId", description="Session identifier"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config": {
                    "meeting_info": {
                        "meeting_id": "abc-defg-hij",
                        "meeting_title": "Weekly Team Standup",
                        "platform": "google_meet",
                        "organizer_email": "organizer@example.com",
                    },
                    "translation": {
                        "target_languages": ["en", "es", "fr"],
                        "enable_auto_translation": True,
                    },
                    "priority": "medium",
                    "auto_terminate_minutes": 180,
                },
                "user_id": "user_123",
                "session_id": "session_abc",
                "metadata": {"department": "engineering", "project": "livetranslate"},
            }
        }
    )


class AudioCaptureStats(BaseModel):
    """Audio capture statistics"""

    is_capturing: bool = Field(description="Whether actively capturing audio")
    total_chunks_captured: int = Field(
        description="Total audio chunks captured", json_schema_extra={"example": 1250}
    )
    average_chunk_size_bytes: float = Field(
        description="Average chunk size in bytes", json_schema_extra={"example": 2048.5}
    )
    total_audio_duration_s: float = Field(
        description="Total audio duration in seconds", json_schema_extra={"example": 1875.3}
    )
    average_quality_score: float = Field(
        description="Average audio quality score (0-1)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.87},
    )
    last_capture_timestamp: datetime = Field(description="Last capture timestamp")
    device_info: str = Field(
        description="Audio device information",
        json_schema_extra={"example": "Default Audio Device"},
    )
    sample_rate_actual: int = Field(
        description="Actual sample rate being used", json_schema_extra={"example": 16000}
    )
    channels_actual: int = Field(
        description="Actual number of channels", json_schema_extra={"example": 1}
    )


class CaptionProcessorStats(BaseModel):
    """Caption processor statistics"""

    total_captions_processed: int = Field(
        description="Total captions processed", json_schema_extra={"example": 350}
    )
    total_speakers: int = Field(
        description="Total number of speakers identified", json_schema_extra={"example": 3}
    )
    speaker_timeline: list[dict[str, Any]] = Field(
        description="Speaker timeline data",
        json_schema_extra={
            "example": [
                {"speaker_id": "speaker_1", "start": 0.0, "end": 15.5},
                {"speaker_id": "speaker_2", "start": 15.5, "end": 32.1},
            ]
        },
    )
    average_confidence: float = Field(
        description="Average caption confidence (0-1)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92},
    )
    last_caption_timestamp: datetime = Field(description="Last caption processed timestamp")
    processing_latency_ms: float = Field(
        description="Average processing latency in milliseconds",
        json_schema_extra={"example": 125.3},
    )


class VirtualWebcamStats(BaseModel):
    """Virtual webcam statistics"""

    is_streaming: bool = Field(description="Whether webcam is actively streaming")
    frames_generated: int = Field(
        description="Total frames generated", json_schema_extra={"example": 54000}
    )
    current_translations: list[dict[str, str]] = Field(
        description="Currently displayed translations",
        json_schema_extra={
            "example": [
                {"language": "es", "text": "Hola, ¿cómo estás?"},
                {"language": "fr", "text": "Bonjour, comment allez-vous?"},
            ]
        },
    )
    average_fps: float = Field(
        description="Average frames per second", json_schema_extra={"example": 29.7}
    )
    webcam_config: WebcamConfig = Field(description="Current webcam configuration")
    last_frame_timestamp: datetime = Field(description="Last frame generation timestamp")


class TimeCorrelationStats(BaseModel):
    """Time correlation statistics"""

    total_correlations: int = Field(
        description="Total time correlations performed", json_schema_extra={"example": 890}
    )
    success_rate: float = Field(
        description="Correlation success rate (0-1)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.94},
    )
    average_timing_offset_ms: float = Field(
        description="Average timing offset in milliseconds", json_schema_extra={"example": 45.2}
    )
    last_correlation_timestamp: datetime = Field(description="Last correlation timestamp")
    correlation_accuracy: float = Field(
        description="Correlation accuracy score (0-1)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.88},
    )


class BotPerformanceStats(BaseModel):
    """Bot performance statistics"""

    session_duration_s: float = Field(
        description="Session duration in seconds", json_schema_extra={"example": 3600.5}
    )
    total_processing_time_s: float = Field(
        description="Total processing time in seconds", json_schema_extra={"example": 2847.3}
    )
    cpu_usage_percent: float = Field(
        description="Average CPU usage percentage",
        ge=0.0,
        le=100.0,
        json_schema_extra={"example": 25.8},
    )
    memory_usage_mb: float = Field(
        description="Memory usage in MB", json_schema_extra={"example": 512.7}
    )
    network_bytes_sent: int = Field(
        description="Network bytes sent", json_schema_extra={"example": 1048576}
    )
    network_bytes_received: int = Field(
        description="Network bytes received", json_schema_extra={"example": 2097152}
    )
    average_latency_ms: float = Field(
        description="Average operation latency in milliseconds", json_schema_extra={"example": 67.4}
    )
    error_count: int = Field(
        description="Number of errors encountered", json_schema_extra={"example": 2}
    )
    last_error: str | None = Field(default=None, description="Last error message")


class BotInstance(IDMixin, TimestampMixin):
    """Bot instance information"""

    bot_id: str = Field(
        description="Bot identifier", json_schema_extra={"example": "bot_abc123def456"}
    )
    status: BotStatus = Field(description="Current bot status")
    config: BotConfiguration = Field(description="Bot configuration")

    # Statistics
    audio_capture: AudioCaptureStats = Field(description="Audio capture statistics")
    caption_processor: CaptionProcessorStats = Field(description="Caption processor statistics")
    virtual_webcam: VirtualWebcamStats = Field(description="Virtual webcam statistics")
    time_correlation: TimeCorrelationStats = Field(description="Time correlation statistics")
    performance: BotPerformanceStats = Field(description="Performance statistics")

    # Runtime information
    last_active_at: datetime = Field(description="Last activity timestamp")
    error_messages: list[str] = Field(default_factory=list, description="Recent error messages")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "bot_instance_123",
                "bot_id": "bot_abc123def456",
                "status": "active",
                "config": {
                    "meeting_info": {
                        "meeting_id": "abc-defg-hij",
                        "meeting_title": "Weekly Team Standup",
                    }
                },
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "last_active_at": "2024-01-15T10:30:00Z",
            }
        }
    )


class BotResponse(ResponseMixin, TimestampMixin):
    """Bot operation response"""

    bot_id: str = Field(description="Bot identifier")
    operation: str = Field(
        description="Operation performed", json_schema_extra={"example": "spawn"}
    )
    bot: BotInstance | None = Field(default=None, description="Bot instance data")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Bot spawned successfully",
                "timestamp": "2024-01-15T10:30:00Z",
                "bot_id": "bot_abc123def456",
                "operation": "spawn",
                "bot": {"bot_id": "bot_abc123def456", "status": "spawning"},
            }
        }
    )


class BotStats(BaseModel):
    """System-wide bot statistics"""

    total_bots_spawned: int = Field(
        description="Total bots spawned", json_schema_extra={"example": 125}
    )
    active_bots: int = Field(description="Currently active bots", json_schema_extra={"example": 8})
    completed_sessions: int = Field(
        description="Completed bot sessions", json_schema_extra={"example": 115}
    )
    failed_sessions: int = Field(
        description="Failed bot sessions", json_schema_extra={"example": 2}
    )
    average_session_duration_minutes: float = Field(
        description="Average session duration in minutes", json_schema_extra={"example": 45.2}
    )
    total_audio_processed_hours: float = Field(
        description="Total audio processed in hours", json_schema_extra={"example": 87.5}
    )
    total_translations_generated: int = Field(
        description="Total translations generated", json_schema_extra={"example": 15420}
    )
    error_rate_percent: float = Field(
        description="Error rate percentage", ge=0.0, le=100.0, json_schema_extra={"example": 1.6}
    )
    platform_distribution: dict[str, int] = Field(
        description="Distribution by meeting platform",
        json_schema_extra={"example": {"google_meet": 95, "zoom": 25, "teams": 5}},
    )
    language_distribution: dict[str, int] = Field(
        description="Distribution by target languages",
        json_schema_extra={"example": {"en": 120, "es": 85, "fr": 60}},
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_bots_spawned": 125,
                "active_bots": 8,
                "completed_sessions": 115,
                "failed_sessions": 2,
                "average_session_duration_minutes": 45.2,
                "total_audio_processed_hours": 87.5,
                "total_translations_generated": 15420,
                "error_rate_percent": 1.6,
                "platform_distribution": {"google_meet": 95, "zoom": 25, "teams": 5},
                "language_distribution": {"en": 120, "es": 85, "fr": 60},
            }
        }
    )
