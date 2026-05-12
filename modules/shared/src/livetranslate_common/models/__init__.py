"""Shared Pydantic models for LiveTranslate services."""

from livetranslate_common.models.transcription import ModelInfo, Segment, TranscriptionResult
from livetranslate_common.models.audio import AudioChunk, MeetingAudioStream
from livetranslate_common.models.llm import (
    LLMConnection,
    LLMEngine,
    LLMParameterOverrides,
)
from livetranslate_common.models.translation import (
    TranslationContext,
    TranslationRequest,
    TranslationResponse,
)
from livetranslate_common.models.ws_messages import (
    PROTOCOL_VERSION,
    BackendSwitchedMessage,
    ConfigMessage,
    ConnectedMessage,
    EndMessage,
    EndMeetingMessage,
    EndSessionMessage,
    InterimMessage,
    LanguageDetectedMessage,
    MeetingStartedMessage,
    PromoteToMeetingMessage,
    RecordingStatusMessage,
    SegmentMessage,
    ServiceStatusMessage,
    StartSessionMessage,
    TranslationMessage,
    parse_ws_message,
)
from livetranslate_common.models.registry import BackendConfig

__all__ = [
    # transcription
    "ModelInfo",
    "Segment",
    "TranscriptionResult",
    # audio
    "AudioChunk",
    "MeetingAudioStream",
    # llm
    "LLMConnection",
    "LLMEngine",
    "LLMParameterOverrides",
    # translation
    "TranslationContext",
    "TranslationRequest",
    "TranslationResponse",
    # ws_messages
    "PROTOCOL_VERSION",
    "BackendSwitchedMessage",
    "ConfigMessage",
    "ConnectedMessage",
    "EndMessage",
    "EndMeetingMessage",
    "EndSessionMessage",
    "InterimMessage",
    "LanguageDetectedMessage",
    "MeetingStartedMessage",
    "PromoteToMeetingMessage",
    "RecordingStatusMessage",
    "SegmentMessage",
    "ServiceStatusMessage",
    "StartSessionMessage",
    "TranslationMessage",
    "parse_ws_message",
    # registry
    "BackendConfig",
]
