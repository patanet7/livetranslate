"""WebSocket message schemas for the LiveTranslate protocol.

All messages are Pydantic v2 BaseModel subclasses with a discriminated
``type`` Literal field.  Use ``parse_ws_message`` to deserialise raw
JSON strings received over a WebSocket connection.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1

# ---------------------------------------------------------------------------
# Client → Server (session lifecycle)
# ---------------------------------------------------------------------------


class StartSessionMessage(BaseModel):
    """Client requests a new streaming session.

    Args:
        sample_rate: PCM sample rate in Hz.
        channels: Number of audio channels.
        device_id: Optional source device identifier.
    """

    type: Literal["start_session"] = "start_session"
    sample_rate: int
    channels: int
    device_id: str | None = None


class EndSessionMessage(BaseModel):
    """Client signals end of the current streaming session."""

    type: Literal["end_session"] = "end_session"


class PromoteToMeetingMessage(BaseModel):
    """Client promotes an active session to a full meeting session."""

    type: Literal["promote_to_meeting"] = "promote_to_meeting"


class EndMeetingMessage(BaseModel):
    """Client signals end of the current meeting session."""

    type: Literal["end_meeting"] = "end_meeting"


# ---------------------------------------------------------------------------
# Client → Transcription Service
# ---------------------------------------------------------------------------


class ConfigMessage(BaseModel):
    """Runtime configuration update sent to the transcription service.

    Args:
        model: Model identifier override (None = keep current).
        language: BCP-47 language code override (None = auto-detect).
        initial_prompt: Prompt text to condition the model.
        glossary_terms: Domain-specific term hints for the decoder.
    """

    type: Literal["config"] = "config"
    model: str | None = None
    language: str | None = None
    initial_prompt: str | None = None
    glossary_terms: list[str] | None = None


class EndMessage(BaseModel):
    """Client signals end of audio stream to the transcription service."""

    type: Literal["end"] = "end"


# ---------------------------------------------------------------------------
# Transcription Service → Client
# ---------------------------------------------------------------------------


class LanguageDetectedMessage(BaseModel):
    """Transcription service reports an authoritative language detection.

    Args:
        language: BCP-47 language code.
        confidence: Detection confidence in [0.0, 1.0].
    """

    type: Literal["language_detected"] = "language_detected"
    language: str
    confidence: float = Field(ge=0.0, le=1.0)


class BackendSwitchedMessage(BaseModel):
    """Transcription service reports a backend/model switch.

    Args:
        backend: New backend identifier.
        model: New model identifier.
        language: Language the backend was switched to.
    """

    type: Literal["backend_switched"] = "backend_switched"
    backend: str
    model: str
    language: str


# ---------------------------------------------------------------------------
# Server → Client
# ---------------------------------------------------------------------------


class ConnectedMessage(BaseModel):
    """Server confirms the WebSocket connection and session creation.

    Args:
        protocol_version: Protocol version in use.
        session_id: Unique session identifier assigned by the server.
    """

    type: Literal["connected"] = "connected"
    protocol_version: int = PROTOCOL_VERSION
    session_id: str


class SegmentMessage(BaseModel):
    """Server sends a completed (or draft) transcription segment.

    Args:
        text: Full segment text.
        language: BCP-47 language code.
        confidence: Overall confidence in [0.0, 1.0].
        stable_text: Finalized portion of the text.
        unstable_text: Still-being-refined portion of the text.
        is_final: True when the segment will not be updated further.
        speaker_id: Optional speaker diarization identifier.
    """

    type: Literal["segment"] = "segment"
    text: str
    language: str
    confidence: float = Field(ge=0.0, le=1.0)
    stable_text: str
    unstable_text: str
    is_final: bool
    speaker_id: str | None = None


class InterimMessage(BaseModel):
    """Server sends a low-latency partial transcription result.

    Args:
        text: Partial transcription text.
        confidence: Confidence estimate for this partial result.
    """

    type: Literal["interim"] = "interim"
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class TranslationMessage(BaseModel):
    """Server sends a translation result linked to a transcription.

    Args:
        text: Translated text.
        source_lang: BCP-47 source language code.
        target_lang: BCP-47 target language code.
        transcript_id: Internal ID of the source transcription.
        context_used: Number of context pairs used for this translation.
    """

    type: Literal["translation"] = "translation"
    text: str
    source_lang: str
    target_lang: str
    transcript_id: int
    context_used: int = 0


class MeetingStartedMessage(BaseModel):
    """Server confirms the meeting session has been started.

    Args:
        session_id: Unique meeting session identifier.
        started_at: ISO-8601 timestamp when the meeting started.
    """

    type: Literal["meeting_started"] = "meeting_started"
    session_id: str
    started_at: str


class RecordingStatusMessage(BaseModel):
    """Server reports the current FLAC recording status.

    Args:
        recording: True when recording is active.
        chunks_written: Number of audio chunks written to disk so far.
    """

    type: Literal["recording_status"] = "recording_status"
    recording: bool
    chunks_written: int


class ServiceStatusMessage(BaseModel):
    """Server broadcasts health status of downstream services.

    Args:
        transcription: "up" or "down" for the transcription service.
        translation: "up" or "down" for the translation service.
    """

    type: Literal["service_status"] = "service_status"
    transcription: Literal["up", "down"]
    translation: Literal["up", "down"]


# ---------------------------------------------------------------------------
# Message registries
# ---------------------------------------------------------------------------

_CLIENT_MESSAGES: dict[str, type[BaseModel]] = {
    "start_session": StartSessionMessage,
    "end_session": EndSessionMessage,
    "promote_to_meeting": PromoteToMeetingMessage,
    "end_meeting": EndMeetingMessage,
    "config": ConfigMessage,
    "end": EndMessage,
}

_SERVER_MESSAGES: dict[str, type[BaseModel]] = {
    "connected": ConnectedMessage,
    "segment": SegmentMessage,
    "interim": InterimMessage,
    "translation": TranslationMessage,
    "meeting_started": MeetingStartedMessage,
    "recording_status": RecordingStatusMessage,
    "service_status": ServiceStatusMessage,
    "language_detected": LanguageDetectedMessage,
    "backend_switched": BackendSwitchedMessage,
}

_ALL_MESSAGES: dict[str, type[BaseModel]] = {**_CLIENT_MESSAGES, **_SERVER_MESSAGES}


def parse_ws_message(raw: str) -> BaseModel | None:
    """Parse a raw JSON WebSocket frame into a typed message model.

    Args:
        raw: JSON string received from the WebSocket.

    Returns:
        A validated message model instance, or None if the ``type``
        field is unrecognised or the JSON is malformed.
    """
    try:
        data = json.loads(raw)
        msg_type = data.get("type")
        model_cls = _ALL_MESSAGES.get(msg_type)
        if model_cls is None:
            return None
        return model_cls.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return None
